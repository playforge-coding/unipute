use unipute::kernel;

#[kernel(workgroup_size(64))]
fn darken(colors: &mut [Vec4<f32>]) {
    let index = global_id().x;
    let color = colors[index];
    colors[index] = vec4(0.0f32, 0.0, 0.0, 0.0) + color.rgba;
}

fn main() {}
