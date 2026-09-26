use unipute::Layout;

#[derive(Layout)]
#[repr(C)]
struct Particle {
    position: Vec3<f32>,
    mass: f32,
}

fn main() {}
