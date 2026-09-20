use unipute::kernel;

#[kernel]
fn no_size(output: &mut [f32]) {
    output[0] = 1.0;
}

fn main() {}
