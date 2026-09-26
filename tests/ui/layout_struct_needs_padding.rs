use unipute::Layout;

#[derive(Layout)]
#[repr(C)]
struct Weighted {
    mass: f32,
    _pad: [u8; 12],
    position: [f32; 3],
}

fn main() {}
