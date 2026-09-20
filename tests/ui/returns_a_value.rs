use unipute::kernel;

#[kernel(workgroup_size(64))]
fn gives_back(input: &[f32]) -> f32 {
    input[0]
}

fn main() {}
