use unipute::kernel;

#[kernel(workgroup_size(64))]
fn spread(points: &mut [Vec4<f32>]) {
    let index = global_id().x;
    let mut point = points[index];
    point.xx = vec2(1.0f32, 2.0);
    points[index] = point;
}

fn main() {}
