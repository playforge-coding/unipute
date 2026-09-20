use unipute::kernel;

#[kernel(workgroup_size(64))]
fn calls_nothing(output: &mut [f32]) {
    output[0] = frobnicate(1.0);
}

fn main() {}
