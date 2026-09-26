use unipute::kernel;

#[repr(C)]
struct Particle {
    mass: f32,
}

#[kernel(workgroup_size(64))]
fn heavier(particles: &mut [Particle]) {
    particles[0].mass = 2.0;
}

fn main() {}
