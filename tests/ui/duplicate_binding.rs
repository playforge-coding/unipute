use unipute::kernel;

#[kernel(workgroup_size(64))]
fn clashing(
    #[binding(index = 0)] input: &[f32],
    #[binding(index = 0)] output: &mut [f32],
) {
    output[0] = input[0];
}

fn main() {}
