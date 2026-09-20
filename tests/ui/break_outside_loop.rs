use unipute::kernel;

#[kernel(workgroup_size(64))]
fn stray_break(output: &mut [f32]) {
    output[0] = 1.0;
    break;
}

fn main() {}
