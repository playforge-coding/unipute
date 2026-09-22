//! Runs kernels on a GPU and checks what they compute.
//!
//! The other tests read the generated shaders. These execute them, which is
//! the only way to catch a kernel that is well formed and wrong. Each test
//! computes the same thing on the CPU and compares.
//!
//! A machine with no adapter skips every test here with a note. CI sets
//! `UNIPUTE_REQUIRE_GPU` so that a broken adapter setup fails instead of
//! quietly passing.

#![cfg(feature = "wgsl")]

// The wgpu glue lives with the examples, since it is what they run through
// too. It is not part of the library.
#[path = "../examples/host/mod.rs"]
mod host;

use host::{Gpu, Pipeline};
use unipute::{Kernel, kernel};

/// A device for one test. Opening one is cheap enough that each test has its
/// own, which keeps them independent.
fn gpu() -> Option<Gpu> {
    let gpu = Gpu::open();
    if gpu.is_none() {
        if std::env::var_os("UNIPUTE_REQUIRE_GPU").is_some() {
            panic!("UNIPUTE_REQUIRE_GPU is set and no GPU adapter was found");
        }
        eprintln!("skipped: no GPU adapter found");
    }
    gpu
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "lengths differ");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "element {index}: got {a}, expected {e}"
        );
    }
}

/// The shape most compute kernels take: a uniform, a bounds check, one
/// element of work.
#[kernel(workgroup_size(64))]
fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index] * factor;
}

/// Runs `scale` through whichever pipeline it is handed, so the same check
/// covers every input wgpu accepts.
fn check_scale(gpu: &Gpu, pipeline: Pipeline) {
    let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let expected: Vec<f32> = input.iter().map(|value| value * 3.0).collect();

    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; input.len()]);
    let factor = gpu.uniform(&3.0f32);
    let groups = host::workgroups([input.len() as u32, 1, 1], scale::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer, &factor], groups);

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

#[test]
fn wgsl_runs() {
    let Some(gpu) = gpu() else { return };
    check_scale(&gpu, gpu.pipeline::<scale>());
}

#[cfg(feature = "spv")]
#[test]
fn spirv_runs() {
    let Some(gpu) = gpu() else { return };
    check_scale(&gpu, gpu.pipeline_from::<scale>(host::spirv::<scale>()));
}

/// The `GLSL` constant is OpenGL ES 3.10, which naga writes and does not
/// read, so this goes through the writer again for a desktop profile that
/// naga reads back. Everything but the version line is the same code path,
/// including the binding numbers wgpu needs to find the buffers.
#[cfg(all(feature = "glsl", feature = "runtime"))]
#[test]
fn glsl_runs() {
    use unipute::naga_backend::{compile_module, write};

    let Some(gpu) = gpu() else { return };
    let validated = compile_module(&scale::ir()).expect("the kernel should compile");
    let desktop = write::glsl_with(&validated, wgpu::naga::back::glsl::Version::Desktop(450))
        .expect("the kernel should write as desktop GLSL");
    // A GLSL compute shader's entry point is `main`, whatever the kernel was
    // called, so the name is not passed along.
    check_scale(
        &gpu,
        gpu.pipeline_with_entry::<scale>(host::glsl(desktop), None),
    );
}

#[cfg(feature = "runtime")]
#[test]
fn naga_ir_runs_without_any_shader_text() {
    let Some(gpu) = gpu() else { return };
    check_scale(&gpu, gpu.pipeline_from::<scale>(host::naga::<scale>()));
}

/// A `for` loop over a range, compound assignment and a math call.
#[kernel(workgroup_size(32, 2))]
fn windowed_sum(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= output.len() {
        return;
    }
    let mut total = 0.0;
    for offset in 0..4u32 {
        let sample = index + offset;
        if sample < input.len() {
            total += sqrt(input[sample]);
        }
    }
    output[index] = total;
}

#[test]
fn a_for_loop_visits_every_offset() {
    let Some(gpu) = gpu() else { return };

    // Squares, so the square roots are exact and the sums are integers.
    let input: Vec<f32> = (0..100).map(|i| (i * i) as f32).collect();
    let expected: Vec<f32> = (0..100)
        .map(|index| {
            (index..index + 4)
                .filter(|&sample| sample < 100)
                .map(|sample| sample as f32)
                .sum()
        })
        .collect();

    let pipeline = gpu.pipeline::<windowed_sum>();
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; 100]);
    let groups = host::workgroups([100, 1, 1], windowed_sum::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], groups);

    // A GPU's square root is allowed a rounding error even on a perfect
    // square, so this is a tolerance rather than an equality.
    assert_close(&gpu.read::<f32>(&output_buffer), &expected, 1e-5);
}

/// `continue` has to reach the loop counter's increment, or the loop never
/// ends and this test never finishes.
#[kernel(workgroup_size(64))]
fn skips_odd(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= output.len() {
        return;
    }
    let mut total = 0.0;
    for offset in 0..8u32 {
        if offset % 2u32 == 1u32 {
            continue;
        }
        total += input[index + offset];
    }
    output[index] = total;
}

#[test]
fn continue_skips_the_body_and_still_advances() {
    let Some(gpu) = gpu() else { return };

    let input: Vec<f32> = (0..72).map(|i| i as f32).collect();
    let expected: Vec<f32> = (0..64)
        .map(|index| {
            [0, 2, 4, 6]
                .iter()
                .map(|offset| (index + offset) as f32)
                .sum()
        })
        .collect();

    let pipeline = gpu.pipeline::<skips_odd>();
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; 64]);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], [1, 1, 1]);

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

/// Nested helpers calling one another, a vector parameter, `dot`, and a
/// helper that returns nothing.
#[kernel(workgroup_size(64))]
fn tonemap(input: &[f32], output: &mut [f32]) {
    fn brightness(color: Vec3<f32>) -> f32 {
        compress(dot(color, vec3(0.2126, 0.7152, 0.0722)))
    }

    fn compress(value: f32) -> f32 {
        if value <= 0.0 {
            return 0.0;
        }
        value / (value + 1.0)
    }

    fn warm_up(rounds: u32) {
        let mut spent = 0u32;
        while spent < rounds {
            spent += 1u32;
        }
    }

    let pixel = global_id().x;
    if pixel >= output.len() {
        return;
    }
    warm_up(2u32);
    let index = pixel * 3u32;
    let color = vec3(input[index], input[index + 1u32], input[index + 2u32]);
    output[pixel] = brightness(color);
}

#[test]
fn nested_functions_compute_what_they_say() {
    let Some(gpu) = gpu() else { return };

    let pixels = 200;
    let input: Vec<f32> = (0..pixels * 3)
        .map(|i| ((i * 37) % 101) as f32 / 25.0 - 0.5)
        .collect();
    let expected: Vec<f32> = input
        .chunks(3)
        .map(|rgb| {
            let luma = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
            if luma <= 0.0 {
                0.0
            } else {
                luma / (luma + 1.0)
            }
        })
        .collect();

    let pipeline = gpu.pipeline::<tonemap>();
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; pixels]);
    let groups = host::workgroups([pixels as u32, 1, 1], tonemap::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], groups);

    assert_close(&gpu.read::<f32>(&output_buffer), &expected, 1e-5);
}

/// A buffer of vectors, a helper that takes one and returns another, and a
/// vector times a scalar.
#[kernel(workgroup_size(32))]
fn drift(points: &[Vec4<f32>], output: &mut [Vec4<f32>], step: &f32) {
    fn advance(point: Vec4<f32>, amount: f32) -> Vec3<f32> {
        vec3(point.x, point.y, point.z) * amount
    }

    let index = global_id().x;
    if index >= points.len() {
        return;
    }
    let point = points[index];
    let moved = advance(point, step);
    output[index] = vec4(moved.x, moved.y, moved.z, point.w);
}

#[test]
fn vector_buffers_keep_their_layout() {
    let Some(gpu) = gpu() else { return };

    let points: Vec<[f32; 4]> = (0..50)
        .map(|i| [i as f32, -(i as f32), i as f32 * 0.25, 1.0 + i as f32])
        .collect();
    let expected: Vec<[f32; 4]> = points
        .iter()
        .map(|[x, y, z, w]| [x * 2.0, y * 2.0, z * 2.0, *w])
        .collect();

    let pipeline = gpu.pipeline::<drift>();
    let points_buffer = gpu.storage(&points);
    let output_buffer = gpu.storage(&vec![[0.0f32; 4]; points.len()]);
    let step = gpu.uniform(&2.0f32);
    let groups = host::workgroups([points.len() as u32, 1, 1], drift::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&points_buffer, &output_buffer, &step], groups);

    assert_eq!(gpu.read::<[f32; 4]>(&output_buffer), expected);
}

/// A two dimensional dispatch, where both axes of the invocation id matter.
///
/// It transposes, but it is not called `transpose`: that is a WGSL built-in,
/// and naga renames an entry point that collides with one.
#[kernel(workgroup_size(8, 8))]
fn flip(input: &[f32], output: &mut [f32], width: &u32, height: &u32) {
    let x = global_id().x;
    let y = global_id().y;
    if x >= width || y >= height {
        return;
    }
    output[x * height + y] = input[y * width + x];
}

#[test]
fn two_dimensional_dispatches_cover_the_image() {
    let Some(gpu) = gpu() else { return };

    // Neither axis is a multiple of the workgroup size, so the edge groups
    // have invocations that must notice they are outside and stop.
    let (width, height) = (21u32, 13u32);
    let input: Vec<f32> = (0..width * height).map(|i| i as f32).collect();
    let mut expected = vec![0.0f32; input.len()];
    for y in 0..height {
        for x in 0..width {
            expected[(x * height + y) as usize] = input[(y * width + x) as usize];
        }
    }

    let pipeline = gpu.pipeline::<flip>();
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; input.len()]);
    let width_buffer = gpu.uniform(&width);
    let height_buffer = gpu.uniform(&height);
    let groups = host::workgroups([width, height, 1], flip::WORKGROUP_SIZE);
    gpu.dispatch(
        &pipeline,
        &[&input_buffer, &output_buffer, &width_buffer, &height_buffer],
        groups,
    );

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

/// Workgroup memory with a tree reduction over it. Every invocation writes
/// one element of the tile, then half of them at a time fold the top half
/// onto the bottom, with a barrier between rounds, until element zero holds
/// the sum of the workgroup's slice.
#[kernel(workgroup_size(64))]
fn block_sum(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let tile: [f32; 64];

    let index = global_id().x;
    let lane = local_index();
    // An `if` rather than an early `return`: every invocation has to reach
    // the barriers below, including the ones past the end of the input.
    if index < input.len() {
        tile[lane] = input[index];
    } else {
        tile[lane] = 0.0;
    }
    workgroup_barrier();

    let mut stride = 32u32;
    while stride > 0u32 {
        if lane < stride {
            tile[lane] += tile[lane + stride];
        }
        workgroup_barrier();
        stride /= 2u32;
    }

    if lane == 0u32 {
        output[workgroup_id().x] = tile[0];
    }
}

#[test]
fn workgroup_memory_sums_each_block() {
    let Some(gpu) = gpu() else { return };

    // Not a multiple of the workgroup size, so the last group has lanes past
    // the end that have to contribute zero and still reach every barrier.
    let elements = 1000u32;
    let input: Vec<f32> = (0..elements).map(|i| (i % 7) as f32).collect();
    let expected: Vec<f32> = input.chunks(64).map(|chunk| chunk.iter().sum()).collect();

    let pipeline = gpu.pipeline::<block_sum>();
    let groups = host::workgroups([elements, 1, 1], block_sum::WORKGROUP_SIZE);
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; groups[0] as usize]);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], groups);

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

/// A single shared value, written by one invocation and read by the rest of
/// its workgroup after the barrier: each group subtracts its first element
/// from every element.
#[kernel(workgroup_size(64))]
fn subtract_first(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let first: f32;

    let index = global_id().x;
    if local_index() == 0u32 {
        first = input[index];
    }
    workgroup_barrier();
    if index < input.len() {
        output[index] = input[index] - first;
    }
}

#[test]
fn a_shared_scalar_broadcasts_within_the_workgroup() {
    let Some(gpu) = gpu() else { return };

    let input: Vec<f32> = (0..200).map(|i| ((i * 13) % 31) as f32).collect();
    let expected: Vec<f32> = input
        .chunks(64)
        .flat_map(|chunk| chunk.iter().map(|value| value - chunk[0]))
        .collect();

    let pipeline = gpu.pipeline::<subtract_first>();
    let groups = host::workgroups([input.len() as u32, 1, 1], subtract_first::WORKGROUP_SIZE);
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0.0f32; input.len()]);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], groups);

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

/// A second bind group. `amount` takes index 2 rather than 0 so that its
/// binding number is unique across groups, which is what the GLSL target
/// needs.
#[kernel(workgroup_size(64))]
fn offset_by(input: &[u32], output: &mut [u32], #[binding(group = 1, index = 2)] amount: &u32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index] + amount;
}

#[test]
fn a_second_bind_group_is_bound_where_it_says() {
    let Some(gpu) = gpu() else { return };

    let input: Vec<u32> = (0..300).collect();
    let expected: Vec<u32> = input.iter().map(|value| value + 1000).collect();

    let pipeline = gpu.pipeline::<offset_by>();
    let input_buffer = gpu.storage(&input);
    let output_buffer = gpu.storage(&vec![0u32; input.len()]);
    let amount = gpu.uniform(&1000u32);
    let groups = host::workgroups([input.len() as u32, 1, 1], offset_by::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer, &amount], groups);

    assert_eq!(gpu.read::<u32>(&output_buffer), expected);
}

/// Explicit placement in a group the kernel otherwise skips over, which
/// leaves groups 0 and 1 empty on the host side.
#[kernel(workgroup_size(1))]
fn placed(
    #[binding(group = 2, index = 5)] input: &[u32],
    #[binding(group = 2, index = 7)] output: &mut [u32],
) {
    output[0] = input[0] * 2u32;
}

#[test]
fn explicit_bindings_land_in_the_right_slots() {
    let Some(gpu) = gpu() else { return };

    let pipeline = gpu.pipeline::<placed>();
    let input_buffer = gpu.storage(&[21u32]);
    let output_buffer = gpu.storage(&[0u32]);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], [1, 1, 1]);

    assert_eq!(gpu.read::<u32>(&output_buffer), [42]);
}
