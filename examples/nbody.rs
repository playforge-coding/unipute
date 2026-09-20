//! Vector types end to end: buffers of them, helpers that take and return
//! them, and the built-in maths that goes with them.
//!
//! One step of a direct n-body simulation. Every body is pulled on by every
//! other one, so each invocation reads the whole position buffer and writes a
//! single element. That is why the new positions go into a second buffer: an
//! invocation moving its own body in place would change what its neighbours
//! are still reading. Velocities are not shared that way, so those are updated
//! where they sit.
//!
//! ```text
//! cargo run --example nbody
//! ```

use unipute::{Kernel, WgslKernel, kernel};

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

/// How many bodies the printed dispatch is for.
const BODIES: u32 = 8192;

fn main() {
    println!("kernel `{}`", step_bodies::NAME);
    println!("workgroup size {:?}", step_bodies::WORKGROUP_SIZE);
    println!();

    // `BINDINGS` says where each buffer goes, and the IR says what is in it.
    // The two are in parameter order, so they line up.
    let ir = step_bodies::ir();
    println!("buffers");
    for (binding, resource) in step_bodies::BINDINGS.iter().zip(&ir.resources) {
        println!(
            "  binding {}: {:<11} {}",
            binding.binding, binding.name, resource.ty
        );
    }
    println!();

    let groups = BODIES.div_ceil(step_bodies::WORKGROUP_SIZE[0]);
    println!("{BODIES} bodies dispatch as {groups} workgroups");
    println!(
        "  every invocation reads all {BODIES} bodies, so one step is {} interactions",
        u64::from(BODIES) * u64::from(BODIES - 1)
    );
    println!();

    println!("--- WGSL ---");
    println!("{}", step_bodies::WGSL);
}
