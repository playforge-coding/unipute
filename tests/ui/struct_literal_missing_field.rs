use unipute::{Layout, kernel};

#[derive(Layout, Clone, Copy)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
}

#[kernel(workgroup_size(64))]
fn reset(particles: &mut [Particle]) {
    particles[0] = Particle { mass: 1.0 };
}

fn main() {}
