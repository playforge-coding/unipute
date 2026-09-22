use unipute::kernel;

#[kernel(workgroup_size(64))]
fn in_nested_function(output: &mut [f32]) {
    fn first() -> f32 {
        #[workgroup]
        let tile: [f32; 64];
        tile[0]
    }

    output[0] = first();
}

fn main() {}
