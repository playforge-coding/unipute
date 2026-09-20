//! Prints the shaders Unipute generated for a kernel while this example was
//! being compiled.
//!
//! Run it with whichever targets you want to see:
//!
//! ```text
//! cargo run --example emit_shaders --features "spv,msl,hlsl,glsl"
//! ```

use unipute::{Kernel, WgslKernel, kernel};

/// Multiplies every element of `input` by `factor` into `output`.
#[kernel(workgroup_size(64))]
fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index] * factor;
}

fn main() {
    println!("kernel `{}`", scale::NAME);
    println!("workgroup size {:?}", scale::WORKGROUP_SIZE);
    println!();

    println!("bindings");
    for binding in scale::BINDINGS {
        println!(
            "  group {} binding {}: {} ({:?})",
            binding.group, binding.binding, binding.name, binding.access
        );
    }

    section("WGSL");
    println!("{}", scale::WGSL);

    #[cfg(feature = "spv")]
    {
        use unipute::SpirvKernel;
        section("SPIR-V");
        println!("{} words", scale::SPIRV.len());
    }

    #[cfg(feature = "msl")]
    {
        use unipute::MslKernel;
        section("MSL");
        println!("{}", scale::MSL);
    }

    #[cfg(feature = "hlsl")]
    {
        use unipute::HlslKernel;
        section("HLSL");
        println!("{}", scale::HLSL);
    }

    #[cfg(feature = "glsl")]
    {
        use unipute::GlslKernel;
        section("GLSL");
        println!("{}", scale::GLSL);
    }
}

fn section(title: &str) {
    println!();
    println!("--- {title} ---");
}
