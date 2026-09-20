//! Picking a target while the program runs instead of while it compiles.
//!
//! The generated constants cover the case where you know the target up front.
//! A tool that takes one on the command line does not, so it goes through the
//! IR and the `runtime` feature instead.
//!
//! ```text
//! cargo run --example retarget --features "runtime,msl,spv" -- msl spirv
//! ```
//!
//! With no arguments it prints what this build can produce, which depends on
//! the features it was compiled with.

use std::process::ExitCode;

use unipute::{Kernel, Target, ir, kernel};

/// Clamps every element into a range the host picks at dispatch time.
#[kernel(workgroup_size(64))]
fn saturate(input: &[f32], output: &mut [f32], floor: &f32, ceiling: &f32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = clamp(input[index], floor, ceiling);
}

fn main() -> ExitCode {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    if wanted.is_empty() {
        usage();
        return ExitCode::SUCCESS;
    }

    // One call, reused for every target. Nothing about the IR knows which
    // languages are coming.
    let ir = saturate::ir();

    let mut failed = false;
    for name in &wanted {
        match generate(&ir, name) {
            Ok(output) => {
                println!("--- {name} ---");
                println!("{output}");
            }
            Err(message) => {
                eprintln!("error: {message}");
                failed = true;
            }
        }
    }

    if failed {
        return ExitCode::FAILURE;
    }

    // The two paths are the same code, so what comes out at run time is what
    // the macro baked in. This is that check, made in front of you.
    #[cfg(feature = "wgsl")]
    {
        use unipute::WgslKernel;

        let matches = unipute::compile_text(&ir, Target::Wgsl)
            .is_ok_and(|generated| generated == saturate::WGSL);
        println!();
        println!("run time wgsl matches the compile time constant: {matches}");
    }

    ExitCode::SUCCESS
}

/// Prints what this build can do, as opposed to what Unipute can do.
fn usage() {
    println!("usage: retarget <target>...");
    println!();
    println!("this build can generate:");
    let enabled = unipute::enabled_targets();
    if enabled.is_empty() {
        println!("  nothing, it was built with no target features on");
    }
    for target in &enabled {
        println!("  {target}");
    }
    println!();
    println!("unipute knows about:");
    for name in ["wgsl", "spirv", "msl", "hlsl", "glsl", "ptx"] {
        let target = Target::from_name(name).expect("these are the names Target parses");
        let note = if !target.is_implemented() {
            "reserved, no back end yet"
        } else if enabled.contains(&target) {
            "available here"
        } else {
            "needs its feature turned on"
        };
        println!("  {name:<6} {note}");
    }
}

/// Generates one target, named the way the command line spells it.
fn generate(kernel: &ir::Kernel, name: &str) -> Result<String, String> {
    let Some(target) = Target::from_name(name) else {
        return Err(format!(
            "`{name}` is not a target, run with no arguments to see the list"
        ));
    };
    if !target.is_implemented() {
        return Err(format!("{target} is reserved, no back end writes it yet"));
    }
    if target.is_text() {
        return unipute::compile_text(kernel, target).map_err(|error| error.to_string());
    }
    spirv(kernel)
}

/// SPIR-V is words rather than text, so it has its own path and its own
/// feature.
fn spirv(kernel: &ir::Kernel) -> Result<String, String> {
    #[cfg(feature = "spv")]
    {
        let words = unipute::compile_spirv(kernel).map_err(|error| error.to_string())?;
        let magic = words.first().copied().unwrap_or_default();
        Ok(format!(
            "{} words, starting with the magic number {magic:#010x}",
            words.len()
        ))
    }
    #[cfg(not(feature = "spv"))]
    {
        let _ = kernel;
        Err("spirv needs the `spv` feature, rebuild with it on".to_owned())
    }
}
