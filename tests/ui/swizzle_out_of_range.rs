use unipute::kernel;

#[kernel(workgroup_size(64))]
fn widen(points: &[Vec2<f32>], output: &mut [Vec3<f32>]) {
    let index = global_id().x;
    output[index] = points[index].xyz;
}

fn main() {}
