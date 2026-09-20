//! Expressions and statements.

use crate::types::{Scalar, VectorSize};

/// Index of a local variable in [`Kernel::locals`](crate::Kernel::locals).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(pub u32);

/// Index of a resource in [`Kernel::resources`](crate::Kernel::resources).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(pub u32);

/// A compile time constant value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
}

impl Literal {
    pub const fn scalar(self) -> Scalar {
        match self {
            Self::Bool(_) => Scalar::Bool,
            Self::I32(_) => Scalar::I32,
            Self::U32(_) => Scalar::U32,
            Self::F32(_) => Scalar::F32,
        }
    }
}

/// A value the hardware provides to every invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// Position of this invocation in the whole dispatch.
    GlobalInvocationId,
    /// Position of this invocation inside its workgroup.
    LocalInvocationId,
    /// Flattened form of [`BuiltIn::LocalInvocationId`].
    LocalInvocationIndex,
    /// Position of this workgroup in the dispatch.
    WorkgroupId,
    /// Number of workgroups in the dispatch.
    NumWorkgroups,
}

impl BuiltIn {
    /// The name of the function that the `kernel` macro maps to this built-in.
    pub const fn intrinsic_name(self) -> &'static str {
        match self {
            Self::GlobalInvocationId => "global_id",
            Self::LocalInvocationId => "local_id",
            Self::LocalInvocationIndex => "local_index",
            Self::WorkgroupId => "workgroup_id",
            Self::NumWorkgroups => "num_workgroups",
        }
    }

    /// Looks up a built-in by the name the macro accepts.
    pub fn from_intrinsic_name(name: &str) -> Option<Self> {
        [
            Self::GlobalInvocationId,
            Self::LocalInvocationId,
            Self::LocalInvocationIndex,
            Self::WorkgroupId,
            Self::NumWorkgroups,
        ]
        .into_iter()
        .find(|candidate| candidate.intrinsic_name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Xor,
    LogicalAnd,
    LogicalOr,
    ShiftLeft,
    ShiftRight,
}

/// A built-in numeric function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathFn {
    Abs,
    Min,
    Max,
    Clamp,
    Floor,
    Ceil,
    Round,
    Sqrt,
    InverseSqrt,
    Exp,
    Log,
    Pow,
    Sin,
    Cos,
    Tan,
    Sign,
    Fma,
    Mix,
    Step,
    Dot,
    Cross,
    Length,
    Normalize,
}

impl MathFn {
    /// How many arguments this function takes.
    pub const fn arity(self) -> usize {
        match self {
            Self::Abs
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sqrt
            | Self::InverseSqrt
            | Self::Exp
            | Self::Log
            | Self::Sign
            | Self::Length
            | Self::Normalize
            | Self::Sin
            | Self::Cos
            | Self::Tan => 1,
            Self::Min | Self::Max | Self::Pow | Self::Step | Self::Dot | Self::Cross => 2,
            Self::Clamp | Self::Fma | Self::Mix => 3,
        }
    }

    /// The name the `kernel` macro accepts for this function.
    pub const fn intrinsic_name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Min => "min",
            Self::Max => "max",
            Self::Clamp => "clamp",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Sqrt => "sqrt",
            Self::InverseSqrt => "inverse_sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Pow => "pow",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Sign => "sign",
            Self::Fma => "fma",
            Self::Mix => "mix",
            Self::Step => "step",
            Self::Dot => "dot",
            Self::Cross => "cross",
            Self::Length => "length",
            Self::Normalize => "normalize",
        }
    }

    /// Looks up a function by the name the macro accepts.
    pub fn from_intrinsic_name(name: &str) -> Option<Self> {
        const ALL: &[MathFn] = &[
            MathFn::Abs,
            MathFn::Min,
            MathFn::Max,
            MathFn::Clamp,
            MathFn::Floor,
            MathFn::Ceil,
            MathFn::Round,
            MathFn::Sqrt,
            MathFn::InverseSqrt,
            MathFn::Exp,
            MathFn::Log,
            MathFn::Pow,
            MathFn::Sin,
            MathFn::Cos,
            MathFn::Tan,
            MathFn::Sign,
            MathFn::Fma,
            MathFn::Mix,
            MathFn::Step,
            MathFn::Dot,
            MathFn::Cross,
            MathFn::Length,
            MathFn::Normalize,
        ];
        ALL.iter()
            .copied()
            .find(|candidate| candidate.intrinsic_name() == name)
    }
}

/// An expression that produces a value, or a place that can be assigned to.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(Literal),
    /// Reads a local variable.
    Local(LocalId),
    /// Names a resource so it can be indexed.
    Resource(ResourceId),
    /// Reads a hardware provided value.
    BuiltIn(BuiltIn),
    /// `base[index]`.
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// Selects one component of a vector, such as `.x`.
    Component {
        base: Box<Expr>,
        index: u8,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// A bit preserving or value converting cast, such as `x as f32`.
    Cast {
        value: Box<Expr>,
        to: Scalar,
    },
    Math {
        function: MathFn,
        args: Vec<Expr>,
    },
    /// Builds a vector out of scalar components.
    Compose {
        size: VectorSize,
        scalar: Scalar,
        components: Vec<Expr>,
    },
    /// `slice.len()` on a runtime sized resource.
    ArrayLength(ResourceId),
}

/// Which memory a barrier synchronises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierScope {
    /// Workgroup shared memory.
    Workgroup,
    /// Storage buffers.
    Storage,
}

/// A statement in a kernel body.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// Introduces a local variable, optionally with an initial value.
    Declare {
        local: LocalId,
        value: Option<Expr>,
    },
    /// Writes `value` into the place named by `place`.
    Store {
        place: Expr,
        value: Expr,
    },
    If {
        condition: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Vec<Stmt>,
    },
    /// A `while` loop. `loop { .. }` is represented as a `While` over a `true`
    /// literal so that the back ends only have one loop form to handle.
    While {
        condition: Expr,
        body: Vec<Stmt>,
        /// Statements that run at the end of every iteration, including the
        /// ones cut short by [`Stmt::Continue`].
        ///
        /// This is what a `for` loop's counter increment goes in. Putting it
        /// at the end of `body` instead would let `continue` skip it and spin
        /// forever. Control flow is not allowed here: no `Return`, `Break`,
        /// `Continue` or nested loop.
        continuing: Vec<Stmt>,
    },
    Break,
    Continue,
    Return,
    Barrier(BarrierScope),
}
