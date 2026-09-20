//! The naga back end for Unipute.
//!
//! This crate turns a [`Kernel`] into naga IR, validates it, and hands it to
//! one of naga's writers. That gets you WGSL, SPIR-V, MSL, HLSL and GLSL from
//! a single kernel definition.
//!
//! It brings in no graphics library. The output is source text or a SPIR-V
//! word buffer, and what you do with it is up to you.
//!
//! ```
//! use unipute_ir::{Access, Expr, Kernel, Literal, Resource, ResourceId, Scalar, Stmt, Type};
//!
//! let mut kernel = Kernel::new("fill", [64, 1, 1]);
//! kernel.resources.push(Resource {
//!     name: "out".to_owned(),
//!     group: 0,
//!     binding: 0,
//!     ty: Type::slice(Type::scalar(Scalar::F32)),
//!     access: Access::ReadWrite,
//! });
//! kernel.body.push(Stmt::Store {
//!     place: Expr::Index {
//!         base: Box::new(Expr::Resource(ResourceId(0))),
//!         index: Box::new(Expr::Component {
//!             base: Box::new(Expr::BuiltIn(unipute_ir::BuiltIn::GlobalInvocationId)),
//!             index: 0,
//!         }),
//!     },
//!     value: Expr::Literal(Literal::F32(1.0)),
//! });
//!
//! let wgsl = unipute_naga::compile_wgsl(&kernel).unwrap();
//! assert!(wgsl.contains("@compute"));
//! ```

#![forbid(unsafe_code)]

pub mod error;
pub mod lower;
pub mod write;

pub use error::{Error, Result};
pub use lower::lower;
pub use write::{Validated, enabled_targets, validate};

use unipute_ir::{Backend, Kernel, Target};

/// Lowers a kernel and validates the result in one step.
///
/// Most callers want one of the `compile_*` functions instead. This is the
/// entry point for anyone who wants the naga module itself, for example to
/// merge several kernels into one module or to run their own passes.
pub fn compile_module(kernel: &Kernel) -> Result<Validated> {
    validate(lower(kernel)?)
}

/// Compiles a kernel to WGSL.
#[cfg(feature = "wgsl")]
pub fn compile_wgsl(kernel: &Kernel) -> Result<String> {
    write::wgsl(&compile_module(kernel)?)
}

/// Compiles a kernel to SPIR-V words.
#[cfg(feature = "spv")]
pub fn compile_spirv(kernel: &Kernel) -> Result<Vec<u32>> {
    write::spirv(&compile_module(kernel)?)
}

/// Compiles a kernel to Metal Shading Language.
#[cfg(feature = "msl")]
pub fn compile_msl(kernel: &Kernel) -> Result<String> {
    write::msl(&compile_module(kernel)?)
}

/// Compiles a kernel to HLSL.
#[cfg(feature = "hlsl")]
pub fn compile_hlsl(kernel: &Kernel) -> Result<String> {
    write::hlsl(&compile_module(kernel)?)
}

/// Compiles a kernel to GLSL.
#[cfg(feature = "glsl")]
pub fn compile_glsl(kernel: &Kernel) -> Result<String> {
    write::glsl(&compile_module(kernel)?)
}

/// Compiles a kernel to any text target this build supports.
///
/// SPIR-V is not a text target, so it is rejected here. Use
/// [`compile_spirv`] for that.
pub fn compile_text(kernel: &Kernel, target: Target) -> Result<String> {
    // Every arm that reads the kernel is behind a feature, so in a build with
    // no text target on there is nothing left that touches it.
    let _ = kernel;
    match target {
        #[cfg(feature = "wgsl")]
        Target::Wgsl => compile_wgsl(kernel),
        #[cfg(feature = "msl")]
        Target::Msl => compile_msl(kernel),
        #[cfg(feature = "hlsl")]
        Target::Hlsl => compile_hlsl(kernel),
        #[cfg(feature = "glsl")]
        Target::Glsl => compile_glsl(kernel),
        Target::SpirV => Err(Error::Invalid(
            "SPIR-V is not text, use compile_spirv".to_owned(),
        )),
        other => Err(Error::TargetNotEnabled(other)),
    }
}

/// The naga back end, exposed through the shared [`Backend`] trait.
///
/// This is what lets a caller be generic over back ends. When a PTX back end
/// lands it will implement the same trait, and code written against the trait
/// will not have to change.
#[derive(Debug, Clone, Copy)]
pub struct NagaBackend;

impl Backend for NagaBackend {
    type Output = Validated;
    type Error = Error;

    const NAME: &'static str = "naga";

    fn compile(kernel: &Kernel) -> Result<Validated> {
        compile_module(kernel)
    }
}
