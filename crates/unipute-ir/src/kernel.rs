//! Kernels: the unit of work a back end turns into a shader or a module.

use crate::expr::{FunctionId, LocalId, ParamId, ResourceId, SharedId, Stmt};
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

/// Memory shared by every invocation in one workgroup.
///
/// It exists for as long as the workgroup runs and is visible to nothing
/// outside it. The host never sees it, so it has no binding and no place in
/// the layout. A write by one invocation reaches the others only after a
/// [`Stmt::Barrier`] with [`BarrierScope::Workgroup`](crate::BarrierScope),
/// which is what that barrier is for.
#[derive(Clone, Debug, PartialEq)]
pub struct Shared {
    pub name: String,
    /// A scalar, a vector, or a fixed length array of either, nested as deep
    /// as needed. Never a runtime sized array, since the memory is laid out
    /// before the workgroup starts.
    pub ty: Type,
}

/// A variable declared inside a function body.
#[derive(Clone, Debug, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: Type,
}

/// A parameter of a helper function.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

/// A helper function the kernel can call.
///
/// A helper is an ordinary function: it works on its parameters and locals and
/// nothing else. It cannot read resources or built-ins, because a shader
/// language only offers those to the entry point, so anything it needs is
/// passed in as an argument.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    /// The type the function returns, or `None` when it returns nothing.
    pub result: Option<Type>,
    pub locals: Vec<Local>,
    pub body: Vec<Stmt>,
}

impl Function {
    /// Starts a function with no parameters, no result and an empty body.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            result: None,
            locals: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn local(&self, id: LocalId) -> &Local {
        &self.locals[id.0 as usize]
    }

    pub fn param(&self, id: ParamId) -> &Param {
        &self.params[id.0 as usize]
    }
}

/// A complete kernel, ready for a back end to consume.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub stage: Stage,
    /// Size of one workgroup. Unused dimensions are `1`.
    pub workgroup_size: [u32; 3],
    pub resources: Vec<Resource>,
    /// Workgroup shared memory. Only the entry point body reaches it, for the
    /// same reason only the entry point reaches resources: a shader language
    /// hands neither to a function called from it.
    pub shared: Vec<Shared>,
    /// Helper functions, ordered so that a callee always comes before the
    /// functions that call it.
    ///
    /// Back ends rely on that order, since most of them have to emit a
    /// function before a call to it can name it, and a shader language has no
    /// forward declarations. It also means recursion cannot be represented,
    /// which matches every target Unipute generates for. A front end is what
    /// puts the list in order and reports a cycle to the user.
    pub functions: Vec<Function>,
    /// The entry point's own locals.
    pub locals: Vec<Local>,
    /// The entry point's own body.
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
            shared: Vec::new(),
            functions: Vec::new(),
            locals: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn resource(&self, id: ResourceId) -> &Resource {
        &self.resources[id.0 as usize]
    }

    pub fn shared(&self, id: SharedId) -> &Shared {
        &self.shared[id.0 as usize]
    }

    pub fn local(&self, id: LocalId) -> &Local {
        &self.locals[id.0 as usize]
    }

    pub fn function(&self, id: FunctionId) -> &Function {
        &self.functions[id.0 as usize]
    }

    /// The total number of invocations in one workgroup.
    pub const fn invocations_per_workgroup(&self) -> u32 {
        self.workgroup_size[0] * self.workgroup_size[1] * self.workgroup_size[2]
    }
}
