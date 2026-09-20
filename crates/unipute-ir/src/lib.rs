//! The Unipute intermediate representation.
//!
//! This crate is the middle of the pipeline. A front end, such as the
//! `#[kernel]` macro in `unipute-macros`, builds a [`Kernel`]. A back end, such
//! as `unipute-naga`, reads one and emits code. Neither side depends on the
//! other, which is what makes Unipute language and graphics API agnostic.
//!
//! The IR has no dependencies on purpose. It is plain data, so a front end for
//! another language or a back end for another platform can be added without
//! pulling in a shader compiler or a graphics library.
//!
//! ```
//! use unipute_ir::{Access, Expr, Kernel, Literal, Resource, Scalar, Stmt, Type};
//!
//! let mut kernel = Kernel::new("clear", [64, 1, 1]);
//! kernel.resources.push(Resource {
//!     name: "data".to_owned(),
//!     group: 0,
//!     binding: 0,
//!     ty: Type::slice(Type::scalar(Scalar::F32)),
//!     access: Access::ReadWrite,
//! });
//! kernel.body.push(Stmt::Store {
//!     place: Expr::Index {
//!         base: Box::new(Expr::Resource(unipute_ir::ResourceId(0))),
//!         index: Box::new(Expr::Literal(Literal::U32(0))),
//!     },
//!     value: Expr::Literal(Literal::F32(0.0)),
//! });
//! assert_eq!(kernel.invocations_per_workgroup(), 64);
//! ```

#![forbid(unsafe_code)]

pub mod backend;
pub mod expr;
pub mod kernel;
pub mod types;

pub use backend::{Backend, Target};
pub use expr::{
    BarrierScope, BinaryOp, BuiltIn, Expr, Literal, LocalId, MathFn, ResourceId, Stmt, UnaryOp,
};
pub use kernel::{Access, Kernel, Local, Resource, Stage};
pub use types::{Scalar, Type, VectorSize};
