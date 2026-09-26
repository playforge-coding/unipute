use unipute::{Layout, kernel};

#[derive(Layout, Clone, Copy)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
}

#[kernel(workgroup_size(64))]
fn weigh(particles: &[Particle], output: &mut [f32]) {
    output[0] = particles[0].weight;
}

fn main() {}
