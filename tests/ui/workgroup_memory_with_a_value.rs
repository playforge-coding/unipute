use unipute::kernel;

#[kernel(workgroup_size(64))]
fn with_a_value(output: &mut [f32]) {
    #[workgroup]
    let total: f32 = 0.0;
    output[0] = total;
}

fn main() {}
