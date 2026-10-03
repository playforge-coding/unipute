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
use unipute::{Kernel, Layout, kernel};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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

/// Swizzles: reading several components at once, assigning through one that
/// reads what it writes, assigning into a buffer element, a swizzle of a
/// swizzle, and a component of a value that is not in memory.
#[kernel(workgroup_size(64))]
fn shuffle(points: &mut [Vec4<f32>], sums: &mut [f32]) {
    let index = global_id().x;
    if index >= points.len() {
        return;
    }
    let mut point = points[index];
    point.xy = point.yx;
    point.zw += point.xy * 2.0;
    points[index].wzy = point.xyz.zyx;
    sums[index] = (point.xy + point.zw).x + point.wzyx.yx.y;
}

#[test]
fn swizzles_read_and_write_the_components_they_name() {
    let Some(gpu) = gpu() else { return };

    let points: Vec<[f32; 4]> = (0..100)
        .map(|i| {
            let i = i as f32;
            [i, i + 1.0, i * 2.0, -i]
        })
        .collect();
    let mut expected_points = Vec::with_capacity(points.len());
    let mut expected_sums = Vec::with_capacity(points.len());
    for &[x, y, z, w] in &points {
        // After the first two lines of the kernel: `[y, x, z + 2y, w + 2x]`.
        let point = [y, x, z + 2.0 * y, w + 2.0 * x];
        // `point.xyz.zyx` lands in `w`, `z` and `y`, and `x` is left alone.
        expected_points.push([x, point[0], point[1], point[2]]);
        expected_sums.push((point[0] + point[2]) + point[3]);
    }

    let pipeline = gpu.pipeline::<shuffle>();
    let points_buffer = gpu.storage(&points);
    let sums_buffer = gpu.storage(&vec![0.0f32; points.len()]);
    let groups = host::workgroups([points.len() as u32, 1, 1], shuffle::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&points_buffer, &sums_buffer], groups);

    assert_eq!(gpu.read::<[f32; 4]>(&points_buffer), expected_points);
    assert_eq!(gpu.read::<f32>(&sums_buffer), expected_sums);
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

/// A particle as both sides see it. `mass` sits in the four bytes after the
/// twelve of `position`, which is the one place a `vec3` does not take
/// sixteen, and `velocity` then starts at the next multiple of 16. No padding
/// is needed and the struct is 32 bytes on both sides.
#[derive(Layout, Clone, Copy, Debug, PartialEq, IntoBytes, FromBytes, Immutable, KnownLayout)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
    velocity: [f32; 3],
    lifetime: f32,
}

#[derive(Layout, Clone, Copy, IntoBytes, Immutable)]
#[repr(C)]
struct Settings {
    gravity: [f32; 3],
    dt: f32,
}

/// One integration step: velocity picks up gravity, position picks up
/// velocity, and the lifetime counts down in place.
#[kernel(workgroup_size(64))]
fn integrate(particles: &mut [Particle], settings: &Settings) {
    fn moved(particle: Particle, gravity: Vec3<f32>, dt: f32) -> Particle {
        let velocity = particle.velocity + gravity * dt;
        Particle {
            position: particle.position + velocity * dt,
            mass: particle.mass,
            velocity,
            lifetime: particle.lifetime,
        }
    }

    let index = global_id().x;
    if index >= particles.len() {
        return;
    }
    particles[index] = moved(particles[index], settings.gravity, settings.dt);
    particles[index].lifetime -= settings.dt;
}

#[test]
fn a_buffer_of_structs_round_trips_through_the_gpu() {
    let Some(gpu) = gpu() else { return };

    let particles: Vec<Particle> = (0..100)
        .map(|i| Particle {
            position: [i as f32, 0.0, -(i as f32)],
            mass: 1.0 + i as f32 * 0.5,
            velocity: [0.0, i as f32 * 0.1, 0.0],
            lifetime: 10.0,
        })
        .collect();
    let settings = Settings {
        gravity: [0.0, -9.8, 0.0],
        dt: 0.5,
    };
    let expected: Vec<Particle> = particles
        .iter()
        .map(|particle| {
            let mut velocity = particle.velocity;
            for axis in 0..3 {
                velocity[axis] += settings.gravity[axis] * settings.dt;
            }
            let mut position = particle.position;
            for axis in 0..3 {
                position[axis] += velocity[axis] * settings.dt;
            }
            Particle {
                position,
                mass: particle.mass,
                velocity,
                lifetime: particle.lifetime - settings.dt,
            }
        })
        .collect();

    let pipeline = gpu.pipeline::<integrate>();
    let particle_buffer = gpu.storage(&particles);
    let settings_buffer = gpu.uniform(&settings);
    let groups = host::workgroups([particles.len() as u32, 1, 1], integrate::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&particle_buffer, &settings_buffer], groups);

    assert_eq!(gpu.read::<Particle>(&particle_buffer), expected);
}

/// A struct that needs padding on the host to match the GPU: a `vec3` after
/// a scalar starts at byte 16, and the whole thing is padded to 32.
#[derive(Layout, Clone, Copy, IntoBytes, Immutable)]
#[repr(C)]
struct Weighted {
    mass: f32,
    _pad: [u8; 12],
    position: [f32; 3],
    _pad2: [u8; 4],
}

/// Reads through the padding: if the offsets were wrong on either side, the
/// sum would pick up padding bytes or the wrong field.
#[kernel(workgroup_size(64))]
fn weigh(weighted: &[Weighted], output: &mut [f32]) {
    let index = global_id().x;
    if index >= weighted.len() {
        return;
    }
    let item = weighted[index];
    output[index] = item.mass * (item.position.x + item.position.y + item.position.z);
}

#[test]
fn padding_puts_fields_where_the_kernel_reads_them() {
    let Some(gpu) = gpu() else { return };

    let items: Vec<Weighted> = (0..70)
        .map(|i| Weighted {
            mass: i as f32,
            _pad: [0xAB; 12],
            position: [1.0, 2.0, i as f32],
            _pad2: [0xCD; 4],
        })
        .collect();
    let expected: Vec<f32> = items
        .iter()
        .map(|item| item.mass * (item.position[0] + item.position[1] + item.position[2]))
        .collect();

    let pipeline = gpu.pipeline::<weigh>();
    let input_buffer = gpu.storage(&items);
    let output_buffer = gpu.storage(&vec![0.0f32; items.len()]);
    let groups = host::workgroups([items.len() as u32, 1, 1], weigh::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&input_buffer, &output_buffer], groups);

    assert_eq!(gpu.read::<f32>(&output_buffer), expected);
}

/// A uniform struct smaller than sixteen bytes, and one of integers.
#[derive(Layout, Clone, Copy, IntoBytes, Immutable)]
#[repr(C)]
struct Grid {
    width: u32,
    height: u32,
}

#[kernel(workgroup_size(8, 8))]
fn number_cells(output: &mut [u32], grid: &Grid) {
    let cell = global_id();
    if cell.x >= grid.width || cell.y >= grid.height {
        return;
    }
    output[cell.y * grid.width + cell.x] = cell.y * 100u32 + cell.x;
}

#[test]
fn a_small_uniform_struct_is_read_whole() {
    let Some(gpu) = gpu() else { return };

    let grid = Grid {
        width: 13,
        height: 5,
    };
    let expected: Vec<u32> = (0..grid.height)
        .flat_map(|y| (0..grid.width).map(move |x| y * 100 + x))
        .collect();

    let pipeline = gpu.pipeline::<number_cells>();
    let output_buffer = gpu.storage(&vec![0u32; expected.len()]);
    let grid_buffer = gpu.uniform(&grid);
    let groups = host::workgroups([grid.width, grid.height, 1], number_cells::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&output_buffer, &grid_buffer], groups);

    assert_eq!(gpu.read::<u32>(&output_buffer), expected);
}

/// A histogram with sixteen bins. Every invocation adds its value to a
/// workgroup bin, and after the barrier the first sixteen lanes add the
/// workgroup's bins into the global ones. Both steps race, and both are
/// atomic, so nothing is lost.
#[kernel(workgroup_size(64))]
fn histogram(values: &[u32], bins: &mut [AtomicU32]) {
    #[workgroup]
    let local_bins: [AtomicU32; 16];

    let index = global_id().x;
    let lane = local_index();
    if index < values.len() {
        local_bins[values[index] % 16u32].fetch_add(1u32);
    }
    workgroup_barrier();
    if lane < 16u32 {
        bins[lane].fetch_add(local_bins[lane].load());
    }
}

#[test]
fn atomic_adds_lose_nothing() {
    let Some(gpu) = gpu() else { return };

    // Not a multiple of the workgroup size, and heavily skewed, so many
    // invocations hit the same bin at once.
    let values: Vec<u32> = (0..10_007).map(|i| (i * i + i / 3) % 16).collect();
    let mut expected = vec![0u32; 16];
    for value in &values {
        expected[*value as usize] += 1;
    }

    let pipeline = gpu.pipeline::<histogram>();
    let values_buffer = gpu.storage(&values);
    let bins_buffer = gpu.storage(&[0u32; 16]);
    let groups = host::workgroups([values.len() as u32, 1, 1], histogram::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&values_buffer, &bins_buffer], groups);

    assert_eq!(gpu.read::<u32>(&bins_buffer), expected);
}

/// The largest and smallest value, found by every invocation racing to update
/// one slot each, and a signed version of the same to cover `AtomicI32`.
#[kernel(workgroup_size(64))]
fn extremes(values: &[i32], result: &mut [AtomicI32]) {
    let index = global_id().x;
    if index >= values.len() {
        return;
    }
    result[0].fetch_max(values[index]);
    result[1].fetch_min(values[index]);
}

#[test]
fn atomic_min_and_max_find_the_extremes() {
    let Some(gpu) = gpu() else { return };

    let values: Vec<i32> = (0..3000).map(|i| ((i * 7919) % 2003) - 1000).collect();
    let expected = [*values.iter().max().unwrap(), *values.iter().min().unwrap()];

    let pipeline = gpu.pipeline::<extremes>();
    let values_buffer = gpu.storage(&values);
    let result_buffer = gpu.storage(&[i32::MIN, i32::MAX]);
    let groups = host::workgroups([values.len() as u32, 1, 1], extremes::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&values_buffer, &result_buffer], groups);

    assert_eq!(gpu.read::<i32>(&result_buffer), expected);
}

/// Every invocation claims one slot in `slots` by swapping its own number in
/// where a zero was, retrying on the next slot when someone got there first.
/// Since there are exactly as many slots as invocations, everyone ends up
/// with one, and the set of claimed numbers is every invocation once.
#[kernel(workgroup_size(64))]
fn claim(slots: &mut [AtomicU32], count: &u32) {
    let me = global_id().x + 1u32;
    if me > count {
        return;
    }
    let mut slot = me % count;
    let mut claimed = false;
    while !claimed {
        claimed = slots[slot].compare_exchange(0u32, me);
        if !claimed {
            slot = (slot + 1u32) % count;
        }
    }
}

#[test]
fn compare_exchange_hands_out_each_slot_once() {
    let Some(gpu) = gpu() else { return };

    let count = 500u32;
    let pipeline = gpu.pipeline::<claim>();
    let slots_buffer = gpu.storage(&vec![0u32; count as usize]);
    let count_buffer = gpu.uniform(&count);
    let groups = host::workgroups([count, 1, 1], claim::WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &[&slots_buffer, &count_buffer], groups);

    let mut claimed = gpu.read::<u32>(&slots_buffer);
    claimed.sort_unstable();
    let expected: Vec<u32> = (1..=count).collect();
    assert_eq!(claimed, expected);
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
