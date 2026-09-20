use unipute::kernel;

#[kernel(workgroup_size(64))]
fn reads_a_builtin(output: &mut [f32]) {
    fn where_am_i() -> u32 {
        global_id().x
    }

    output[where_am_i()] = 1.0;
}

fn main() {}
