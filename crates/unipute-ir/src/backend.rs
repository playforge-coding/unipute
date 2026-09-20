//! The seam every code generator plugs into.
//!
//! Unipute does not care what a back end produces. The naga back end in
//! `unipute-naga` produces shader source or SPIR-V words, but nothing in this
//! module assumes that. A future PTX back end would produce a `String` of PTX
//! assembly and a future CPU fallback would produce machine code, and both fit
//! the same trait.

use core::fmt;

use crate::kernel::Kernel;

/// A code generator that turns a [`Kernel`] into something a driver can load.
pub trait Backend {
    /// What this back end produces, such as a source string or a word buffer.
    type Output;

    /// Why this back end refused a kernel.
    type Error: fmt::Display;

    /// A stable name for this back end, used in error messages.
    const NAME: &'static str;

    /// Generates output for one kernel.
    fn compile(kernel: &Kernel) -> Result<Self::Output, Self::Error>;
}

/// The output languages Unipute knows about.
///
/// Variants that no back end implements yet are still listed. Keeping them
/// here means the host side, the C bindings and the CLI can all talk about a
/// target before the code generator for it exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Target {
    /// WebGPU Shading Language, text.
    Wgsl,
    /// SPIR-V, a word buffer.
    SpirV,
    /// Metal Shading Language, text.
    Msl,
    /// High Level Shading Language, text.
    Hlsl,
    /// OpenGL Shading Language, text.
    Glsl,
    /// Reserved for the CUDA back end. Not implemented.
    Ptx,
}

impl Target {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wgsl => "wgsl",
            Self::SpirV => "spirv",
            Self::Msl => "msl",
            Self::Hlsl => "hlsl",
            Self::Glsl => "glsl",
            Self::Ptx => "ptx",
        }
    }

    /// Parses a target name such as `"wgsl"`.
    pub fn from_name(name: &str) -> Option<Self> {
        const ALL: &[Target] = &[
            Target::Wgsl,
            Target::SpirV,
            Target::Msl,
            Target::Hlsl,
            Target::Glsl,
            Target::Ptx,
        ];
        ALL.iter().copied().find(|target| target.name() == name)
    }

    /// Whether the target is text rather than binary.
    pub const fn is_text(self) -> bool {
        !matches!(self, Self::SpirV)
    }

    /// Whether any back end shipped with Unipute can produce this target.
    ///
    /// This answers the question for the library as a whole. Whether a given
    /// build can produce it also depends on the cargo features that are on.
    pub const fn is_implemented(self) -> bool {
        !matches!(self, Self::Ptx)
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
