//! Unipute: compile time GPU kernels for Rust, with no graphics library
//! attached.
//!
//! Write a kernel as an ordinary looking Rust function, put `#[kernel]` on it,
//! and the shader is generated while your crate compiles. What comes out is
//! source text or SPIR-V words. Unipute never talks to a driver, creates a
//! device or allocates a buffer, so it composes with whatever graphics or
//! compute API you already use.
//!
//! ```
//! use unipute::{Kernel, WgslKernel, kernel};
//!
//! #[kernel(workgroup_size(64))]
//! fn double(input: &[f32], output: &mut [f32]) {
//!     let index = global_id().x;
//!     if index < input.len() {
//!         output[index] = input[index] * 2.0;
//!     }
//! }
//!
//! assert_eq!(double::NAME, "double");
//! assert_eq!(double::WORKGROUP_SIZE, [64, 1, 1]);
//! assert!(double::WGSL.contains("@compute"));
//! ```
//!
//! # Targets
//!
//! Each target is a feature, and each enabled feature adds a constant to every
//! kernel in your crate. `wgsl` is on by default.
//!
//! | Feature | Trait | Constant |
//! | ------- | ----- | -------- |
//! | `wgsl`  | [`WgslKernel`]  | `WGSL: &str` |
//! | `spv`   | [`SpirvKernel`] | `SPIRV: &[u32]` |
//! | `msl`   | [`MslKernel`]   | `MSL: &str` |
//! | `hlsl`  | [`HlslKernel`]  | `HLSL: &str` |
//! | `glsl`  | [`GlslKernel`]  | `GLSL: &str` |
//!
//! # Generating at run time instead
//!
//! Turn on the `runtime` feature to translate a kernel while your program is
//! running, which is what you want if the target is not known until then.
//! [`Kernel::ir`] gives you the kernel back as data, and [`compile_text`]
//! takes it from there.

#![forbid(unsafe_code)]

// The macro writes `::unipute` paths, which have to resolve inside this crate
// too, for the doc tests and the test suite.
extern crate self as unipute;

pub use unipute_ir as ir;
pub use unipute_ir::{Access, Stage, Target};
pub use unipute_macros::kernel;

#[cfg(feature = "runtime")]
pub use unipute_naga as naga_backend;

/// Runs the code in the README as doctests, so its examples cannot drift away
/// from the crate. It does not appear in the documentation.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct Readme;

/// Runs every Rust example in the guide as a doctest.
///
/// The guide is built with mdBook, but its examples are checked here rather
/// than with `mdbook test`. Going through `cargo test` means they are compiled
/// against this exact build of the crate, with the same features, instead of
/// whatever `mdbook test` happens to find on a library path.
///
/// Add a chapter here when you add one to `docs/src/SUMMARY.md`.
#[cfg(doctest)]
mod guide {
    macro_rules! chapters {
        ($($name:ident => $path:literal,)*) => {
            $(
                #[doc = include_str!($path)]
                pub struct $name;
            )*
        };
    }

    chapters! {
        Introduction => "../docs/src/introduction.md",
        Installing => "../docs/src/start/installing.md",
        FirstKernel => "../docs/src/start/first-kernel.md",
        Parameters => "../docs/src/kernels/parameters.md",
        Types => "../docs/src/kernels/types.md",
        ControlFlow => "../docs/src/kernels/control-flow.md",
        Builtins => "../docs/src/kernels/builtins.md",
        Targets => "../docs/src/output/targets.md",
        Running => "../docs/src/output/running.md",
        Runtime => "../docs/src/output/runtime.md",
        HowItWorks => "../docs/src/how-it-works.md",
        Troubleshooting => "../docs/src/troubleshooting.md",
        Roadmap => "../docs/src/roadmap.md",
    }
}

/// Where one of a kernel's resources is bound.
///
/// This is the layout information a host needs in order to build a bind group
/// or a descriptor set, in a form that does not name any graphics API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingInfo {
    /// The parameter name this resource came from.
    pub name: &'static str,
    /// The bind group index.
    pub group: u32,
    /// The binding number inside the group.
    pub binding: u32,
    /// Whether the kernel reads, writes or only reads uniformly.
    pub access: Access,
}

/// What every `#[kernel]` function turns into.
pub trait Kernel {
    /// The entry point name in the generated shader.
    const NAME: &'static str;

    /// The size of one workgroup, with unused dimensions set to 1.
    const WORKGROUP_SIZE: [u32; 3];

    /// Every resource the kernel binds, in parameter order.
    const BINDINGS: &'static [BindingInfo];

    /// Rebuilds the kernel as IR.
    ///
    /// The shader constants are generated at compile time, so you only need
    /// this to retarget a kernel at run time or to inspect one.
    fn ir() -> ir::Kernel;
}

/// A kernel whose WGSL was generated at compile time.
pub trait WgslKernel: Kernel {
    const WGSL: &'static str;
}

/// A kernel whose SPIR-V was generated at compile time.
pub trait SpirvKernel: Kernel {
    const SPIRV: &'static [u32];
}

/// A kernel whose Metal Shading Language was generated at compile time.
pub trait MslKernel: Kernel {
    const MSL: &'static str;
}

/// A kernel whose HLSL was generated at compile time.
pub trait HlslKernel: Kernel {
    const HLSL: &'static str;
}

/// A kernel whose GLSL was generated at compile time.
pub trait GlslKernel: Kernel {
    const GLSL: &'static str;
}

/// Translates a kernel to a text target while the program is running.
///
/// Prefer the generated constants when the target is known at compile time.
/// This is for the case where it is not, such as a tool that picks a language
/// from a command line flag.
#[cfg(feature = "runtime")]
pub fn compile_text(kernel: &ir::Kernel, target: Target) -> Result<String, unipute_naga::Error> {
    unipute_naga::compile_text(kernel, target)
}

/// Translates a kernel to SPIR-V while the program is running.
#[cfg(all(feature = "runtime", feature = "spv"))]
pub fn compile_spirv(kernel: &ir::Kernel) -> Result<Vec<u32>, unipute_naga::Error> {
    unipute_naga::compile_spirv(kernel)
}

/// The targets this build can produce.
///
/// The list depends on which features are on, so it is the honest answer to
/// "what can this binary do", as opposed to [`Target::is_implemented`], which
/// answers it for the library as a whole.
#[cfg(feature = "runtime")]
pub fn enabled_targets() -> Vec<Target> {
    unipute_naga::enabled_targets()
}
