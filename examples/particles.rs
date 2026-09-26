//! Structs in buffers: one type read by the host and the kernel alike.
//!
//! A particle system. Each particle is a struct with a position, a velocity,
//! a mass and a lifetime, held in one buffer of `Particle` rather than four
//! parallel buffers of numbers. The settings for a step are a struct too,
//! bound as a uniform. Both are ordinary `#[repr(C)]` Rust structs with
//! `#[derive(Layout)]` on them, which is what lets a kernel name them and
//! what checks that the host and the GPU agree about where every field is.
//!
//! The host runs a few steps on a GPU through wgpu, checks the first against
//! the same maths on the CPU, and counts how many particles are still alive
//! as the steps go by.
//!
//! ```text
//! cargo run --example particles
//! ```
//!
//! The wgpu side is in `examples/host/mod.rs`. It is not part of Unipute.

mod host;

use std::process::ExitCode;

use unipute::{Kernel, Layout, kernel};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// One particle, 32 bytes on the host and on the GPU.
///
/// The order of the fields is deliberate. A three component vector takes
/// twelve bytes but has to start at a multiple of sixteen, so putting a
/// scalar right after each one fills the gap and no padding is needed. Swap
/// `position` and `mass` around and the derive will explain what to add.
///
/// The zerocopy derives are how this example uploads the particles as
/// bytes. They are the host's business, not Unipute's, which only needs
/// `Layout`.
#[derive(Layout, Clone, Copy, Debug, IntoBytes, FromBytes, Immutable, KnownLayout)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
    velocity: [f32; 3],
    /// Seconds left before the particle stops moving.
    lifetime: f32,
}

/// What one step needs to know, bound once as a uniform.
///
/// This one does need padding. `gravity` and `dt` fill sixteen bytes between
/// them, `drag` starts the next sixteen, and the GPU rounds the struct up to
/// 32 bytes because its largest field aligns to 16. Rust would stop at 20, so
/// the last field makes up the difference. Leave it out and the derive says
/// to put it back, byte count included.
#[derive(Layout, Clone, Copy, IntoBytes, Immutable)]
#[repr(C)]
struct Settings {
    gravity: [f32; 3],
    dt: f32,
    /// How strongly a particle is slowed by the air, per second.
    drag: f32,
    _pad: [u8; 12],
}

/// Advances every live particle by one step.
#[kernel(workgroup_size(64))]
fn step_particles(particles: &mut [Particle], settings: &Settings) {
    /// A particle after one step, worked out from the one before.
    ///
    /// Takes the struct by value and gives one back, the way a helper takes
    /// a vector. Inside the kernel `position` is a `Vec3<f32>`, whatever it
    /// is spelled as on the host.
    fn moved(particle: Particle, settings: Settings) -> Particle {
        let slowed = particle.velocity * (1.0 - settings.drag * settings.dt);
        let velocity = slowed + settings.gravity * settings.dt;
        Particle {
            position: particle.position + velocity * settings.dt,
            mass: particle.mass,
            velocity,
            lifetime: particle.lifetime - settings.dt,
        }
    }

    let index = global_id().x;
    if index >= particles.len() {
        return;
    }
    // A dead particle stays where it is. Reading a field through the index
    // is a load of that field alone, not of the whole struct.
    if particles[index].lifetime <= 0.0 {
        return;
    }
    particles[index] = moved(particles[index], settings);
}

const PARTICLES: u32 = 4096;
const STEPS: u32 = 50;

fn main() -> ExitCode {
    let Some(gpu) = host::Gpu::open() else {
        eprintln!("no GPU adapter found, so there is nothing to run this on");
        return ExitCode::FAILURE;
    };
    println!("running on {}", gpu.describe());

    let settings = Settings {
        gravity: [0.0, -9.8, 0.0],
        dt: 0.05,
        drag: 0.3,
        _pad: [0; 12],
    };
    let particles = burst();

    let pipeline = gpu.pipeline::<step_particles>();
    let particle_buffer = gpu.storage(&particles);
    let settings_buffer = gpu.uniform(&settings);
    let groups = host::workgroups([PARTICLES, 1, 1], step_particles::WORKGROUP_SIZE);

    println!(
        "{PARTICLES} particles of {} bytes each, {} workgroups a step",
        std::mem::size_of::<Particle>(),
        groups[0]
    );
    println!();

    // One step on the CPU, to compare the first GPU step against.
    let expected = step_on_the_cpu(&particles, &settings);

    let mut first_step_difference = 0.0f32;
    for step in 1..=STEPS {
        gpu.dispatch(&pipeline, &[&particle_buffer, &settings_buffer], groups);

        if step == 1 {
            let after_one = gpu.read::<Particle>(&particle_buffer);
            first_step_difference = largest_difference(&after_one, &expected);
        }
        if step % 10 == 0 || step == 1 {
            let current = gpu.read::<Particle>(&particle_buffer);
            let alive = current.iter().filter(|p| p.lifetime > 0.0).count();
            let highest = current
                .iter()
                .filter(|p| p.lifetime > 0.0)
                .map(|p| p.position[1])
                .fold(f32::MIN, f32::max);
            println!("  step {step:>2}  {alive:>4} alive  highest at y = {highest:>7.3}");
        }
    }
    println!();

    println!("largest difference from the CPU on the first step: {first_step_difference:e}");
    if first_step_difference < 1e-4 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Particles fired upwards from the origin in every direction, with the same
/// spread every run.
fn burst() -> Vec<Particle> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    (0..PARTICLES)
        .map(|_| Particle {
            position: [0.0; 3],
            mass: 1.0 + next().abs(),
            velocity: [next() * 3.0, 8.0 + next() * 4.0, next() * 3.0],
            lifetime: 1.0 + next().abs() * 2.0,
        })
        .collect()
}

/// The kernel's step, written as loops on the host.
fn step_on_the_cpu(particles: &[Particle], settings: &Settings) -> Vec<Particle> {
    particles
        .iter()
        .map(|particle| {
            if particle.lifetime <= 0.0 {
                return *particle;
            }
            let mut velocity = particle.velocity;
            let mut position = particle.position;
            for axis in 0..3 {
                velocity[axis] = velocity[axis] * (1.0 - settings.drag * settings.dt)
                    + settings.gravity[axis] * settings.dt;
                position[axis] += velocity[axis] * settings.dt;
            }
            Particle {
                position,
                mass: particle.mass,
                velocity,
                lifetime: particle.lifetime - settings.dt,
            }
        })
        .collect()
}

fn largest_difference(a: &[Particle], b: &[Particle]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            let xs = [x.position, x.velocity, [x.mass, x.lifetime, 0.0]];
            let ys = [y.position, y.velocity, [y.mass, y.lifetime, 0.0]];
            xs.into_iter()
                .zip(ys)
                .flat_map(|(p, q)| p.into_iter().zip(q).map(|(a, b)| (a - b).abs()))
        })
        .fold(0.0, f32::max)
}
