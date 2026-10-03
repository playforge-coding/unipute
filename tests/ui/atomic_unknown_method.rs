use unipute::kernel;

#[kernel(workgroup_size(64))]
fn count(counter: &mut [AtomicU32]) {
    counter[0].increment();
}

fn main() {}
