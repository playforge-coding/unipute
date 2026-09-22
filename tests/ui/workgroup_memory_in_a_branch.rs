use unipute::kernel;

#[kernel(workgroup_size(64))]
fn in_a_branch(output: &mut [f32]) {
    if global_id().x == 0u32 {
        #[workgroup]
        let tile: [f32; 64];
        tile[0] = 1.0;
    }
    output[0] = 1.0;
}

fn main() {}
