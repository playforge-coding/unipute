use unipute::kernel;

#[kernel(workgroup_size(64))]
fn counts_down(output: &mut [f32]) {
    fn steps(n: u32) -> u32 {
        if n == 0u32 {
            return 0u32;
        }
        steps(n - 1u32) + 1u32
    }

    output[0] = steps(4u32) as f32;
}

fn main() {}
