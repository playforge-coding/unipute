use unipute::kernel;

#[kernel(workgroup_size(64))]
fn bump(counter: &mut [AtomicU32]) {
    counter[0] += 1u32;
}

fn main() {}
