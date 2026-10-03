//! End to end checks on `#[kernel]`, from Rust syntax to generated shaders.
//!
//! The shared assertions read WGSL, so the file needs that writer. The
//! other targets have their own tests further down, each behind its feature.

#![cfg(feature = "wgsl")]

use unipute::{Access, Kernel, Layout, WgslKernel, kernel};

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

/// Exercises nested `fn` helpers: one calling another, a forward reference,
/// a vector parameter, a trailing expression as the return value and a helper
/// that returns nothing.
#[kernel(workgroup_size(64))]
fn tonemap(input: &[f32], output: &mut [f32]) {
    // Declared before the function it calls, which the front end has to sort
    // out before any of this reaches a back end.
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

    let index = global_id().x * 3u32;
    if index + 2u32 >= input.len() {
        return;
    }
    warm_up(2u32);
    let color = vec3(input[index], input[index + 1u32], input[index + 2u32]);
    output[global_id().x] = brightness(color);
}

/// Exercises a buffer whose elements are vectors, a helper that takes one and
/// returns another, and multiplying a vector by a scalar.
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

/// Exercises workgroup memory: a tile every invocation writes one element of,
/// a barrier, `.len()` on the tile, and a read of the whole tile by one
/// invocation.
#[kernel(workgroup_size(64))]
fn block_sum(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let tile: [f32; 64];

    let lane = local_index();
    tile[lane] = input[global_id().x];
    workgroup_barrier();

    if lane == 0u32 {
        let mut total = 0.0;
        for i in 0..tile.len() {
            total += tile[i];
        }
        output[workgroup_id().x] = total;
    }
}

/// A struct the kernel below reads from a buffer. `mass` fits in the four
/// bytes after a `vec3`, so no padding is needed and the struct is 16 bytes on
/// both sides.
#[derive(Layout, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
}

/// A uniform struct, declared after the kernel that uses it, since the macro
/// that carries its fields is found by name rather than by position.
#[derive(Layout, Clone, Copy)]
#[repr(C)]
struct Settings {
    gravity: [f32; 3],
    dt: f32,
}

/// Exercises structs: a buffer of them, a uniform one, a helper taking and
/// returning one, member reads through an index, a struct literal, and a
/// store to one member in place.
#[kernel(workgroup_size(64))]
fn fall(particles: &mut [Particle], settings: &Settings, heaviest: &mut [f32]) {
    fn pulled(particle: Particle, gravity: Vec3<f32>, dt: f32) -> Particle {
        Particle {
            position: particle.position + gravity * dt,
            mass: particle.mass,
        }
    }

    let index = global_id().x;
    if index >= particles.len() {
        return;
    }
    particles[index] = pulled(particles[index], settings.gravity, settings.dt);
    if particles[index].mass > heaviest[0] {
        heaviest[0] = particles[index].mass;
    }
    particles[index].mass = particles[index].mass * 2.0;
}

/// Exercises atomics: a buffer of them, workgroup memory of them, every
/// operation as a value and as a statement, a load, a store, and a compare
/// and exchange loop.
#[kernel(workgroup_size(64))]
fn tally(values: &[u32], bins: &mut [AtomicU32], stats: &mut [AtomicI32]) {
    #[workgroup]
    let local_bins: [AtomicU32; 4];
    #[workgroup]
    let seen: AtomicU32;

    let index = global_id().x;
    let lane = local_index();
    if index < values.len() {
        local_bins[values[index] % 4u32].fetch_add(1u32);
        seen.fetch_add(1);
        stats[0].fetch_max(values[index] as i32);
        stats[1].fetch_min(values[index] as i32);
        stats[2].fetch_and(1);
        stats[3].fetch_or(2);
        stats[4].fetch_xor(4);
    }
    workgroup_barrier();

    if lane < 4u32 {
        let count = local_bins[lane].load();
        let before = bins[lane].fetch_add(count);
        let old = bins[lane + 4u32].swap(before);
        bins[lane + 8u32].store(old + seen.load());
        bins[lane + 12u32].fetch_sub(1);
    }

    // Claim a slot: keep trying until the exchange goes through.
    let mut done = false;
    while !done {
        let current = bins[16].load();
        done = bins[16].compare_exchange(current, current + 1u32);
    }
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
fn nested_functions_become_shader_functions() {
    let wgsl = tonemap::WGSL;

    // A callee has to be written before its caller, whichever order the two
    // were declared in, since a shader language has no forward declarations.
    let compress = wgsl.find("fn compress(").expect("compress is written");
    let brightness = wgsl.find("fn brightness(").expect("brightness is written");
    let entry = wgsl
        .find("fn tonemap(")
        .expect("the entry point is written");
    assert!(compress < brightness, "{wgsl}");
    assert!(brightness < entry, "{wgsl}");

    assert!(wgsl.contains("fn warm_up(rounds: u32)"), "{wgsl}");
    assert!(wgsl.contains("warm_up(2u)"), "{wgsl}");
    // A vector parameter is passed by value, not through a binding.
    assert!(wgsl.contains("color: vec3<f32>"), "{wgsl}");
}

#[test]
fn a_buffer_of_vectors_keeps_its_element_type() {
    let wgsl = drift::WGSL;

    assert!(wgsl.contains("array<vec4<f32>>"), "{wgsl}");
    // A vector times a scalar keeps the shape of the vector.
    assert!(wgsl.contains("-> vec3<f32>"), "{wgsl}");

    let ir = drift::ir();
    assert_eq!(
        ir.resources[0].ty,
        unipute::ir::Type::slice(unipute::ir::Type::vector(
            unipute::ir::VectorSize::Four,
            unipute::ir::Scalar::F32
        ))
    );
}

#[test]
fn nested_functions_reach_the_ir() {
    let ir = tonemap::ir();

    let names: Vec<&str> = ir
        .functions
        .iter()
        .map(|function| function.name.as_str())
        .collect();
    assert_eq!(names, ["compress", "brightness", "warm_up"]);

    let compress = &ir.functions[0];
    assert_eq!(compress.params.len(), 1);
    assert_eq!(
        compress.result,
        Some(unipute::ir::Type::scalar(unipute::ir::Scalar::F32))
    );
    assert_eq!(ir.functions[2].result, None);
}

#[test]
fn workgroup_memory_is_a_workgroup_variable() {
    let wgsl = block_sum::WGSL;

    assert!(
        wgsl.contains("var<workgroup> tile: array<f32, 64>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("workgroupBarrier()"), "{wgsl}");
    // `tile.len()` is known when the kernel is written, so it comes out as
    // a number rather than a question to the driver.
    assert!(!wgsl.contains("arrayLength((&tile"), "{wgsl}");

    // The host never sees it, so it is not in the layout.
    assert_eq!(block_sum::BINDINGS.len(), 2);
    let ir = block_sum::ir();
    assert_eq!(ir.shared.len(), 1);
    assert_eq!(ir.shared[0].name, "tile");
    assert_eq!(
        ir.shared[0].ty,
        unipute::ir::Type::Array {
            element: Box::new(unipute::ir::Type::scalar(unipute::ir::Scalar::F32)),
            len: Some(64),
        }
    );
}

#[test]
fn structs_are_written_with_their_fields() {
    let wgsl = fall::WGSL;

    assert!(wgsl.contains("struct Particle {"), "{wgsl}");
    assert!(wgsl.contains("position: vec3<f32>,"), "{wgsl}");
    assert!(wgsl.contains("array<Particle>"), "{wgsl}");
    assert!(wgsl.contains("var<uniform> settings: Settings"), "{wgsl}");
    assert!(
        wgsl.contains("fn pulled(particle: Particle, gravity: vec3<f32>, dt: f32) -> Particle"),
        "{wgsl}"
    );
    assert!(wgsl.contains("].mass = "), "{wgsl}");
}

#[test]
fn the_derive_and_the_kernel_agree_about_a_struct() {
    let ir = fall::ir();

    // What the kernel macro learned through the derive's macro is exactly
    // what the derive itself reports, offsets included.
    assert_eq!(ir.resources[0].ty, unipute::ir::Type::slice(Particle::ty()));
    assert_eq!(ir.resources[1].ty, Settings::ty());

    let unipute::ir::Type::Struct(def) = Particle::ty() else {
        panic!("a derived layout is a struct");
    };
    assert_eq!(def.name, "Particle");
    assert_eq!(def.members[0].offset, 0);
    assert_eq!(def.members[1].offset, 12);
    assert_eq!(def.size, 16);
    assert_eq!(def.size as usize, std::mem::size_of::<Particle>());
}

#[test]
fn atomics_are_atomic_in_the_shader() {
    let wgsl = tally::WGSL;

    assert!(wgsl.contains("array<atomic<u32>>"), "{wgsl}");
    assert!(wgsl.contains("array<atomic<i32>>"), "{wgsl}");
    assert!(
        wgsl.contains("var<workgroup> local_bins: array<atomic<u32>, 4>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("var<workgroup> seen: atomic<u32>"), "{wgsl}");
    for call in [
        "atomicAdd(",
        "atomicSub(",
        "atomicMax(",
        "atomicMin(",
        "atomicAnd(",
        "atomicOr(",
        "atomicXor(",
        "atomicExchange(",
        "atomicLoad(",
        "atomicStore(",
        "atomicCompareExchangeWeak(",
    ] {
        assert!(wgsl.contains(call), "{call} missing from {wgsl}");
    }

    let ir = tally::ir();
    assert_eq!(
        ir.resources[1].ty,
        unipute::ir::Type::slice(unipute::ir::Type::Atomic(unipute::ir::Scalar::U32))
    );
    assert_eq!(
        ir.shared[1].ty,
        unipute::ir::Type::Atomic(unipute::ir::Scalar::U32)
    );
    // An atomic is a plain integer to the host, so the layout is untouched.
    assert_eq!(tally::BINDINGS.len(), 3);
}

#[cfg(feature = "runtime")]
#[test]
fn a_kernel_with_atomics_survives_the_runtime_path() {
    let regenerated = unipute::compile_text(&tally::ir(), unipute::Target::Wgsl).unwrap();
    assert_eq!(regenerated, tally::WGSL);
}

#[cfg(feature = "runtime")]
#[test]
fn a_kernel_with_structs_survives_the_runtime_path() {
    let regenerated = unipute::compile_text(&fall::ir(), unipute::Target::Wgsl).unwrap();
    assert_eq!(regenerated, fall::WGSL);
}

#[cfg(feature = "runtime")]
#[test]
fn workgroup_memory_survives_the_runtime_path() {
    let regenerated = unipute::compile_text(&block_sum::ir(), unipute::Target::Wgsl).unwrap();
    assert_eq!(regenerated, block_sum::WGSL);
}

#[cfg(feature = "msl")]
#[test]
fn msl_spells_workgroup_memory_threadgroup() {
    use unipute::MslKernel;

    assert!(block_sum::MSL.contains("threadgroup"), "{}", block_sum::MSL);
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_spells_workgroup_memory_groupshared() {
    use unipute::HlslKernel;

    assert!(
        block_sum::HLSL.contains("groupshared float tile[64]"),
        "{}",
        block_sum::HLSL
    );
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_spells_workgroup_memory_shared() {
    use unipute::GlslKernel;

    assert!(
        block_sum::GLSL.contains("shared float tile[64]"),
        "{}",
        block_sum::GLSL
    );
}

#[cfg(feature = "runtime")]
#[test]
fn a_kernel_with_helpers_survives_the_runtime_path() {
    let regenerated = unipute::compile_text(&tonemap::ir(), unipute::Target::Wgsl).unwrap();
    assert_eq!(regenerated, tonemap::WGSL);
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
