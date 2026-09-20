use unipute::kernel;

#[kernel(workgroup_size(64))]
fn whole_buffer(input: &[f32], output: &mut [f32]) {
    output = input;
}

fn main() {}
