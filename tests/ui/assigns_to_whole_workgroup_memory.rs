use unipute::kernel;

#[kernel(workgroup_size(64))]
fn assigns_whole_tile(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let tile: [f32; 64];
    tile = input[0];
    output[0] = tile[0];
}

fn main() {}
