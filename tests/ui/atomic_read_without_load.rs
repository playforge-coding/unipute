use unipute::kernel;

#[kernel(workgroup_size(64))]
fn peek(counter: &mut [AtomicU32], output: &mut [u32]) {
    output[0] = counter[0];
}

fn main() {}
