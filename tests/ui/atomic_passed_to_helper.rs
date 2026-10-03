use unipute::kernel;

#[kernel(workgroup_size(64))]
fn count(counter: &mut [AtomicU32]) {
    fn bump(slot: AtomicU32) {
        slot.fetch_add(1u32);
    }

    bump(counter[0]);
}

fn main() {}
