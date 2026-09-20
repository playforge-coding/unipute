use unipute::kernel;

#[kernel(workgroup_size(64))]
fn reads_a_buffer(input: &[f32], output: &mut [f32]) {
    fn first() -> f32 {
        input[0]
    }

    output[0] = first();
}

fn main() {}
