use unipute::Layout;

#[derive(Layout)]
#[repr(C)]
struct Flag {
    raised: bool,
    value: f32,
}

fn main() {}
