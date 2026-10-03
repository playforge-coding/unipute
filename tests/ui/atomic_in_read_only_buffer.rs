use unipute::kernel;

#[kernel(workgroup_size(64))]
fn watch(counter: &[AtomicU32], output: &mut [u32]) {
    output[0] = counter[0].load();
}

fn main() {}
