//! Kernels: the unit of work a back end turns into a shader or a module.

use crate::expr::{LocalId, ResourceId, Stmt};
use crate::types::Type;

/// The pipeline stage a kernel runs in.
///
/// Only [`Stage::Compute`] is implemented. The graphics stages are listed so
/// that the IR, the back end trait and the generated code all have a place to
/// put them once graphics support lands, and so that the macro can report a
/// clear error instead of a parse failure today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    Compute,
    /// Reserved. Not accepted by any back end yet.
    Vertex,
    /// Reserved. Not accepted by any back end yet.
    Fragment,
}

impl Stage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }

    /// Whether a back end can be expected to handle this stage today.
    pub const fn is_implemented(self) -> bool {
        matches!(self, Self::Compute)
    }
}

/// How a kernel is allowed to touch a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// A read only storage buffer, from a `&[T]` parameter.
    Read,
    /// A read and write storage buffer, from a `&mut [T]` parameter.
    ReadWrite,
    /// A uniform buffer, from a `&T` parameter where `T` is not a slice.
    Uniform,
}

/// A buffer the host binds before dispatching the kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    pub name: String,
    /// The bind group this resource belongs to.
    pub group: u32,
    /// The binding number inside the group.
    pub binding: u32,
    pub ty: Type,
    pub access: Access,
}

/// A variable declared inside the kernel body.
#[derive(Clone, Debug, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: Type,
}

/// A complete kernel, ready for a back end to consume.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub stage: Stage,
    /// Size of one workgroup. Unused dimensions are `1`.
    pub workgroup_size: [u32; 3],
    pub resources: Vec<Resource>,
    pub locals: Vec<Local>,
    pub body: Vec<Stmt>,
}

impl Kernel {
    /// Starts a compute kernel with an empty body.
    pub fn new(name: impl Into<String>, workgroup_size: [u32; 3]) -> Self {
        Self {
            name: name.into(),
            stage: Stage::Compute,
            workgroup_size,
            resources: Vec::new(),
            locals: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn resource(&self, id: ResourceId) -> &Resource {
        &self.resources[id.0 as usize]
    }

    pub fn local(&self, id: LocalId) -> &Local {
        &self.locals[id.0 as usize]
    }

    /// The total number of invocations in one workgroup.
    pub const fn invocations_per_workgroup(&self) -> u32 {
        self.workgroup_size[0] * self.workgroup_size[1] * self.workgroup_size[2]
    }
}
