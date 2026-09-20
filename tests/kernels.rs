//! End to end checks on `#[kernel]`, from Rust syntax to generated shaders.
//!
//! The shared assertions read WGSL, so the file needs that writer. The
//! other targets have their own tests further down, each behind its feature.

#![cfg(feature = "wgsl")]

use unipute::{Access, Kernel, WgslKernel, kernel};

/// The shape most compute kernels take: guard on the buffer length, then do
/// one element of work.
#[kernel(workgroup_size(64))]
fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index] * factor;
}

/// Exercises a `for` loop over a range, compound assignment and a math call.
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

/// Exercises vectors, casts, a `while` loop and a barrier.
#[kernel(workgroup_size(8, 8, 1), name = "shade")]
fn shade_tile(output: &mut [f32]) {
    let position = global_id();
    let color = vec3(position.x as f32, position.y as f32, 0.5);
    let index = position.y * 8u32 + position.x;

    workgroup_barrier();

    let mut channel = 0u32;
    while channel < 3u32 {
        channel = channel + 1u32;
    }
    output[index] = length(color);
}

/// `continue` has to reach the loop counter's increment, or the loop never
/// ends. The increment lives in the generated loop's continuing block for
/// exactly this reason.
#[kernel(workgroup_size(64))]
fn skips_odd(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    let mut total = 0.0;
    for offset in 0..8u32 {
        if offset % 2u32 == 1u32 {
            continue;
        }
        total += input[index + offset];
    }
    output[index] = total;
}

/// Exercises explicit binding placement.
#[kernel(workgroup_size(1))]
fn placed(
    #[binding(group = 2, index = 5)] input: &[u32],
    #[binding(group = 2, index = 7)] output: &mut [u32],
) {
    output[0] = input[0];
}

#[test]
fn a_kernel_reports_its_own_layout() {
    assert_eq!(scale::NAME, "scale");
    assert_eq!(scale::WORKGROUP_SIZE, [64, 1, 1]);

    let bindings = scale::BINDINGS;
    assert_eq!(bindings.len(), 3);
    assert_eq!(bindings[0].name, "input");
    assert_eq!(bindings[0].access, Access::Read);
    assert_eq!((bindings[0].group, bindings[0].binding), (0, 0));
    assert_eq!(bindings[1].access, Access::ReadWrite);
    assert_eq!((bindings[1].group, bindings[1].binding), (0, 1));
    assert_eq!(bindings[2].access, Access::Uniform);
}

#[test]
fn the_name_option_renames_the_entry_point() {
    assert_eq!(shade_tile::NAME, "shade");
    assert!(
        shade_tile::WGSL.contains("fn shade("),
        "{}",
        shade_tile::WGSL
    );
}

#[test]
fn explicit_bindings_are_honoured() {
    let bindings = placed::BINDINGS;
    assert_eq!((bindings[0].group, bindings[0].binding), (2, 5));
    assert_eq!((bindings[1].group, bindings[1].binding), (2, 7));
    assert!(
        placed::WGSL.contains("@group(2) @binding(5)"),
        "{}",
        placed::WGSL
    );
}

#[test]
fn scale_generates_the_expected_wgsl() {
    let wgsl = scale::WGSL;

    assert!(
        wgsl.contains("@compute @workgroup_size(64, 1, 1)"),
        "{wgsl}"
    );
    assert!(wgsl.contains("var<storage, read_write> output"), "{wgsl}");
    assert!(wgsl.contains("var<uniform> factor"), "{wgsl}");
    assert!(wgsl.contains("arrayLength"), "{wgsl}");
    assert!(wgsl.contains("global_invocation_id"), "{wgsl}");
}

#[test]
fn a_for_loop_becomes_a_counted_loop() {
    let wgsl = windowed_sum::WGSL;

    assert!(wgsl.contains("@workgroup_size(32, 2, 1)"), "{wgsl}");
    assert!(wgsl.contains("loop {"), "{wgsl}");
    assert!(wgsl.contains("sqrt("), "{wgsl}");
}

#[test]
fn continue_still_advances_the_loop_counter() {
    let wgsl = skips_odd::WGSL;

    // The increment has to be in the continuing block rather than the body,
    // otherwise `continue` skips it and the loop spins forever.
    let continuing = wgsl
        .split_once("continuing {")
        .expect("the loop should have a continuing block")
        .1;
    let body = continuing
        .split_once('}')
        .expect("the continuing block should be closed")
        .0;
    assert!(body.contains("offset ="), "{wgsl}");
}

#[test]
fn vectors_casts_and_barriers_survive_the_round_trip() {
    let wgsl = shade_tile::WGSL;

    assert!(wgsl.contains("vec3<f32>"), "{wgsl}");
    assert!(wgsl.contains("workgroupBarrier()"), "{wgsl}");
    assert!(wgsl.contains("length("), "{wgsl}");
}

#[test]
fn the_ir_matches_what_the_shader_was_built_from() {
    let ir = scale::ir();

    assert_eq!(ir.name, "scale");
    assert_eq!(ir.workgroup_size, [64, 1, 1]);
    assert_eq!(ir.resources.len(), 3);
    assert_eq!(ir.resources[0].name, "input");
    assert!(!ir.body.is_empty());
    assert_eq!(ir.stage, unipute::Stage::Compute);
}

/// The IR handed back at run time has to produce the same shader the macro
/// baked in, or the two paths have drifted apart.
#[cfg(feature = "runtime")]
#[test]
fn the_runtime_path_agrees_with_the_compile_time_one() {
    let regenerated = unipute::compile_text(&scale::ir(), unipute::Target::Wgsl).unwrap();
    assert_eq!(regenerated, scale::WGSL);
}

#[cfg(feature = "spv")]
#[test]
fn spirv_is_generated_with_the_right_magic_number() {
    use unipute::SpirvKernel;

    assert_eq!(scale::SPIRV.first(), Some(&0x0723_0203));
    assert!(scale::SPIRV.len() > 16);
}

#[cfg(feature = "msl")]
#[test]
fn msl_is_generated() {
    use unipute::MslKernel;

    assert!(scale::MSL.contains("kernel void"), "{}", scale::MSL);
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_is_generated() {
    use unipute::HlslKernel;

    assert!(scale::HLSL.contains("numthreads(64"), "{}", scale::HLSL);
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_is_generated() {
    use unipute::GlslKernel;

    assert!(scale::GLSL.contains("local_size_x = 64"), "{}", scale::GLSL);
}
