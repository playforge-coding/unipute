//! Vector types end to end: buffers of them, helpers that take and return
//! them, and the built-in maths that goes with them.
//!
//! A direct n-body simulation. Every body is pulled on by every other one, so
//! each invocation reads the whole position buffer and writes a single
//! element. That is why the new positions go into a second buffer: an
//! invocation moving its own body in place would change what its neighbours
//! are still reading. Velocities are not shared that way, so those are updated
//! where they sit.
//!
//! The host runs a few steps on a GPU through wgpu, checks the first one
//! against the same maths on the CPU, and watches the total momentum, which
//! the physics says should not change.
//!
//! ```text
//! cargo run --example nbody
//! ```
//!
//! The wgpu side is in `examples/host/mod.rs`. It is not part of Unipute.

mod host;

use std::process::ExitCode;

use unipute::{Kernel, kernel};

/// Advances every body by one time step.
///
/// A body is a `Vec4<f32>`: position in `xyz` and mass in `w`. Packing the two
/// together means one buffer read per body rather than two, which matters in a
/// kernel that reads every body once per invocation.
#[kernel(workgroup_size(64))]
fn step_bodies(
    bodies: &[Vec4<f32>],
    velocities: &mut [Vec4<f32>],
    moved: &mut [Vec4<f32>],
    dt: &f32,
    softening: &f32,
) {
    /// The pull `other` puts on `body`.
    ///
    /// Newton's law with a softening term: at very short range the true
    /// force goes to infinity, and a simulation stepping in fixed time slices
    /// turns that into a body flung off the screen. Adding `softening` to the
    /// squared distance rounds the peak off.
    fn pull(body: Vec4<f32>, other: Vec4<f32>, softening: f32) -> Vec3<f32> {
        let offset = vec3(other.x - body.x, other.y - body.y, other.z - body.z);
        let distance_squared = dot(offset, offset) + softening * softening;
        let inverse_distance = inverse_sqrt(distance_squared);
        // Mass over distance squared, times the unit vector towards `other`,
        // which is the offset over distance. That is three divisions by the
        // distance, so one reciprocal cubed does the lot.
        let strength = other.w * inverse_distance * inverse_distance * inverse_distance;
        offset * strength
    }

    let index = global_id().x;
    if index >= bodies.len() {
        return;
    }

    let body = bodies[index];
    let mut acceleration = vec3(0.0f32, 0.0, 0.0);
    for other in 0..bodies.len() {
        // A body does not pull on itself, and the softening term would hide
        // the mistake rather than blow up on it.
        if other == index {
            continue;
        }
        acceleration = acceleration + pull(body, bodies[other], softening);
    }

    let velocity = velocities[index];
    let moved_velocity = vec3(
        velocity.x + acceleration.x * dt,
        velocity.y + acceleration.y * dt,
        velocity.z + acceleration.z * dt,
    );

    velocities[index] = vec4(
        moved_velocity.x,
        moved_velocity.y,
        moved_velocity.z,
        velocity.w,
    );
    moved[index] = vec4(
        body.x + moved_velocity.x * dt,
        body.y + moved_velocity.y * dt,
        body.z + moved_velocity.z * dt,
        // The mass rides along untouched.
        body.w,
    );
}

/// How many bodies to simulate, and for how long.
const BODIES: u32 = 1024;
const STEPS: u32 = 20;
const DT: f32 = 0.005;
const SOFTENING: f32 = 0.1;

/// A body on the host: the same 16 bytes the kernel sees.
type Body = [f32; 4];

fn main() -> ExitCode {
    let Some(gpu) = host::Gpu::open() else {
        eprintln!("no GPU adapter found, so there is nothing to run this on");
        return ExitCode::FAILURE;
    };
    println!("running on {}", gpu.describe());

    let bodies = scatter();
    let velocities = vec![[0.0f32; 4]; bodies.len()];

    let pipeline = gpu.pipeline::<step_bodies>();
    let a = gpu.storage(&bodies);
    let b = gpu.storage(&vec![[0.0f32; 4]; bodies.len()]);
    let velocity_buffer = gpu.storage(&velocities);
    let dt = gpu.uniform(&DT);
    let softening = gpu.uniform(&SOFTENING);

    let groups = host::workgroups([BODIES, 1, 1], step_bodies::WORKGROUP_SIZE);
    println!(
        "{BODIES} bodies, {} workgroups, {} interactions a step",
        groups[0],
        u64::from(BODIES) * u64::from(BODIES - 1)
    );
    println!();

    // One step on the CPU, to compare the first GPU step against.
    let (expected_bodies, expected_velocities) = step_on_the_cpu(&bodies, &velocities);

    // Positions ping pong between two buffers. Velocities stay put.
    let (mut current, mut next) = (&a, &b);
    let mut first_step_difference = 0.0f32;
    let momentum_at_start = momentum(&bodies, &velocities);
    for step in 1..=STEPS {
        gpu.dispatch(
            &pipeline,
            &[current, &velocity_buffer, next, &dt, &softening],
            groups,
        );
        std::mem::swap(&mut current, &mut next);

        if step == 1 {
            let gpu_bodies = gpu.read::<Body>(current);
            let gpu_velocities = gpu.read::<Body>(&velocity_buffer);
            first_step_difference = largest_difference(&gpu_bodies, &expected_bodies)
                .max(largest_difference(&gpu_velocities, &expected_velocities));
        }
        if step % 5 == 0 || step == 1 {
            let gpu_bodies = gpu.read::<Body>(current);
            let gpu_velocities = gpu.read::<Body>(&velocity_buffer);
            let [px, py, pz] = momentum(&gpu_bodies, &gpu_velocities);
            let [x, y, z, _] = gpu_bodies[0];
            println!(
                "  step {step:>2}  body 0 at ({x:>7.3}, {y:>7.3}, {z:>7.3})  momentum ({px:>8.5}, {py:>8.5}, {pz:>8.5})"
            );
        }
    }
    println!();

    println!("largest difference from the CPU on the first step: {first_step_difference:e}");
    let [px, py, pz] = momentum_at_start;
    println!(
        "momentum started at ({px:.5}, {py:.5}, {pz:.5}), and every pull has an equal and opposite \
         one, so what the column above shows is rounding rather than physics"
    );

    if first_step_difference < 1e-4 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Bodies spread through a box, with the same layout every run.
///
/// A small linear congruential generator, so the picture is repeatable and
/// nothing outside the standard library is needed for it.
fn scatter() -> Vec<Body> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    (0..BODIES)
        .map(|_| {
            let (x, y, z) = (next(), next(), next());
            let mass = 0.5 + (next() + 1.0) * 0.5;
            [x * 4.0, y * 4.0, z * 4.0, mass]
        })
        .collect()
}

/// The kernel's step, written as loops on the host.
fn step_on_the_cpu(bodies: &[Body], velocities: &[Body]) -> (Vec<Body>, Vec<Body>) {
    let mut moved = Vec::with_capacity(bodies.len());
    let mut new_velocities = Vec::with_capacity(bodies.len());
    for (index, body) in bodies.iter().enumerate() {
        let mut acceleration = [0.0f32; 3];
        for (other_index, other) in bodies.iter().enumerate() {
            if other_index == index {
                continue;
            }
            let offset = [other[0] - body[0], other[1] - body[1], other[2] - body[2]];
            let distance_squared = offset[0] * offset[0]
                + offset[1] * offset[1]
                + offset[2] * offset[2]
                + SOFTENING * SOFTENING;
            let inverse_distance = 1.0 / distance_squared.sqrt();
            let strength = other[3] * inverse_distance * inverse_distance * inverse_distance;
            for axis in 0..3 {
                acceleration[axis] += offset[axis] * strength;
            }
        }
        let velocity = velocities[index];
        let moved_velocity = [
            velocity[0] + acceleration[0] * DT,
            velocity[1] + acceleration[1] * DT,
            velocity[2] + acceleration[2] * DT,
        ];
        new_velocities.push([
            moved_velocity[0],
            moved_velocity[1],
            moved_velocity[2],
            velocity[3],
        ]);
        moved.push([
            body[0] + moved_velocity[0] * DT,
            body[1] + moved_velocity[1] * DT,
            body[2] + moved_velocity[2] * DT,
            body[3],
        ]);
    }
    (moved, new_velocities)
}

/// Total momentum, mass times velocity summed over every body.
fn momentum(bodies: &[Body], velocities: &[Body]) -> [f32; 3] {
    let mut total = [0.0f32; 3];
    for (body, velocity) in bodies.iter().zip(velocities) {
        for axis in 0..3 {
            total[axis] += body[3] * velocity[axis];
        }
    }
    total
}

fn largest_difference(a: &[Body], b: &[Body]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f32::max)
}
