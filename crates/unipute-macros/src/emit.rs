//! Writing a [`ir::Kernel`] back out as Rust code.
//!
//! The macro already has the kernel in hand at compile time, so most output is
//! finished shader text. This module covers the other half: rebuilding the IR
//! at run time, which is what a caller needs in order to retarget a kernel
//! that was not compiled ahead of time, or to inspect it.

use proc_macro2::TokenStream;
use quote::quote;
use unipute_ir as ir;

/// Builds an expression that reconstructs `kernel`.
pub fn kernel(kernel: &ir::Kernel) -> TokenStream {
    let name = &kernel.name;
    let [x, y, z] = kernel.workgroup_size;
    let resources = kernel.resources.iter().map(resource);
    let locals = kernel.locals.iter().map(local);
    let body = kernel.body.iter().map(stmt);

    quote! {
        {
            let mut kernel = ::unipute::ir::Kernel::new(#name, [#x, #y, #z]);
            kernel.resources = ::std::vec![#(#resources),*];
            kernel.locals = ::std::vec![#(#locals),*];
            kernel.body = ::std::vec![#(#body),*];
            kernel
        }
    }
}

fn resource(resource: &ir::Resource) -> TokenStream {
    let name = &resource.name;
    let group = resource.group;
    let binding = resource.binding;
    let ty = ty(&resource.ty);
    let access = access(resource.access);
    quote! {
        ::unipute::ir::Resource {
            name: ::std::string::ToString::to_string(#name),
            group: #group,
            binding: #binding,
            ty: #ty,
            access: #access,
        }
    }
}

fn local(local: &ir::Local) -> TokenStream {
    let name = &local.name;
    let ty = ty(&local.ty);
    quote! {
        ::unipute::ir::Local {
            name: ::std::string::ToString::to_string(#name),
            ty: #ty,
        }
    }
}

pub fn access(access: ir::Access) -> TokenStream {
    match access {
        ir::Access::Read => quote!(::unipute::ir::Access::Read),
        ir::Access::ReadWrite => quote!(::unipute::ir::Access::ReadWrite),
        ir::Access::Uniform => quote!(::unipute::ir::Access::Uniform),
    }
}

fn scalar(scalar: ir::Scalar) -> TokenStream {
    match scalar {
        ir::Scalar::Bool => quote!(::unipute::ir::Scalar::Bool),
        ir::Scalar::I32 => quote!(::unipute::ir::Scalar::I32),
        ir::Scalar::U32 => quote!(::unipute::ir::Scalar::U32),
        ir::Scalar::F32 => quote!(::unipute::ir::Scalar::F32),
    }
}

fn vector_size(size: ir::VectorSize) -> TokenStream {
    match size {
        ir::VectorSize::Two => quote!(::unipute::ir::VectorSize::Two),
        ir::VectorSize::Three => quote!(::unipute::ir::VectorSize::Three),
        ir::VectorSize::Four => quote!(::unipute::ir::VectorSize::Four),
    }
}

fn ty(node: &ir::Type) -> TokenStream {
    match node {
        ir::Type::Scalar(value) => {
            let value = scalar(*value);
            quote!(::unipute::ir::Type::Scalar(#value))
        }
        ir::Type::Vector { size, scalar: kind } => {
            let size = vector_size(*size);
            let kind = scalar(*kind);
            quote!(::unipute::ir::Type::Vector { size: #size, scalar: #kind })
        }
        ir::Type::Array { element, len } => {
            let element = ty(element);
            let len = match len {
                Some(len) => quote!(::std::option::Option::Some(#len)),
                None => quote!(::std::option::Option::None),
            };
            quote! {
                ::unipute::ir::Type::Array {
                    element: ::std::boxed::Box::new(#element),
                    len: #len,
                }
            }
        }
    }
}

fn literal(literal: ir::Literal) -> TokenStream {
    match literal {
        ir::Literal::Bool(value) => quote!(::unipute::ir::Literal::Bool(#value)),
        ir::Literal::I32(value) => quote!(::unipute::ir::Literal::I32(#value)),
        ir::Literal::U32(value) => quote!(::unipute::ir::Literal::U32(#value)),
        ir::Literal::F32(value) => quote!(::unipute::ir::Literal::F32(#value)),
    }
}

fn built_in(built_in: ir::BuiltIn) -> TokenStream {
    match built_in {
        ir::BuiltIn::GlobalInvocationId => quote!(::unipute::ir::BuiltIn::GlobalInvocationId),
        ir::BuiltIn::LocalInvocationId => quote!(::unipute::ir::BuiltIn::LocalInvocationId),
        ir::BuiltIn::LocalInvocationIndex => quote!(::unipute::ir::BuiltIn::LocalInvocationIndex),
        ir::BuiltIn::WorkgroupId => quote!(::unipute::ir::BuiltIn::WorkgroupId),
        ir::BuiltIn::NumWorkgroups => quote!(::unipute::ir::BuiltIn::NumWorkgroups),
    }
}

fn unary_op(op: ir::UnaryOp) -> TokenStream {
    match op {
        ir::UnaryOp::Negate => quote!(::unipute::ir::UnaryOp::Negate),
        ir::UnaryOp::Not => quote!(::unipute::ir::UnaryOp::Not),
    }
}

fn binary_op(op: ir::BinaryOp) -> TokenStream {
    let name = match op {
        ir::BinaryOp::Add => "Add",
        ir::BinaryOp::Subtract => "Subtract",
        ir::BinaryOp::Multiply => "Multiply",
        ir::BinaryOp::Divide => "Divide",
        ir::BinaryOp::Modulo => "Modulo",
        ir::BinaryOp::Equal => "Equal",
        ir::BinaryOp::NotEqual => "NotEqual",
        ir::BinaryOp::Less => "Less",
        ir::BinaryOp::LessEqual => "LessEqual",
        ir::BinaryOp::Greater => "Greater",
        ir::BinaryOp::GreaterEqual => "GreaterEqual",
        ir::BinaryOp::And => "And",
        ir::BinaryOp::Or => "Or",
        ir::BinaryOp::Xor => "Xor",
        ir::BinaryOp::LogicalAnd => "LogicalAnd",
        ir::BinaryOp::LogicalOr => "LogicalOr",
        ir::BinaryOp::ShiftLeft => "ShiftLeft",
        ir::BinaryOp::ShiftRight => "ShiftRight",
    };
    let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
    quote!(::unipute::ir::BinaryOp::#ident)
}

fn math_fn(function: ir::MathFn) -> TokenStream {
    let name = match function {
        ir::MathFn::Abs => "Abs",
        ir::MathFn::Min => "Min",
        ir::MathFn::Max => "Max",
        ir::MathFn::Clamp => "Clamp",
        ir::MathFn::Floor => "Floor",
        ir::MathFn::Ceil => "Ceil",
        ir::MathFn::Round => "Round",
        ir::MathFn::Sqrt => "Sqrt",
        ir::MathFn::InverseSqrt => "InverseSqrt",
        ir::MathFn::Exp => "Exp",
        ir::MathFn::Log => "Log",
        ir::MathFn::Pow => "Pow",
        ir::MathFn::Sin => "Sin",
        ir::MathFn::Cos => "Cos",
        ir::MathFn::Tan => "Tan",
        ir::MathFn::Sign => "Sign",
        ir::MathFn::Fma => "Fma",
        ir::MathFn::Mix => "Mix",
        ir::MathFn::Step => "Step",
        ir::MathFn::Dot => "Dot",
        ir::MathFn::Cross => "Cross",
        ir::MathFn::Length => "Length",
        ir::MathFn::Normalize => "Normalize",
    };
    let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
    quote!(::unipute::ir::MathFn::#ident)
}

fn expr(node: &ir::Expr) -> TokenStream {
    match node {
        ir::Expr::Literal(value) => {
            let value = literal(*value);
            quote!(::unipute::ir::Expr::Literal(#value))
        }
        ir::Expr::Local(id) => {
            let id = id.0;
            quote!(::unipute::ir::Expr::Local(::unipute::ir::LocalId(#id)))
        }
        ir::Expr::Resource(id) => {
            let id = id.0;
            quote!(::unipute::ir::Expr::Resource(::unipute::ir::ResourceId(#id)))
        }
        ir::Expr::BuiltIn(value) => {
            let value = built_in(*value);
            quote!(::unipute::ir::Expr::BuiltIn(#value))
        }
        ir::Expr::Index { base, index } => {
            let base = expr(base);
            let index = expr(index);
            quote! {
                ::unipute::ir::Expr::Index {
                    base: ::std::boxed::Box::new(#base),
                    index: ::std::boxed::Box::new(#index),
                }
            }
        }
        ir::Expr::Component { base, index } => {
            let base = expr(base);
            quote! {
                ::unipute::ir::Expr::Component {
                    base: ::std::boxed::Box::new(#base),
                    index: #index,
                }
            }
        }
        ir::Expr::Unary { op, value } => {
            let op = unary_op(*op);
            let value = expr(value);
            quote! {
                ::unipute::ir::Expr::Unary {
                    op: #op,
                    value: ::std::boxed::Box::new(#value),
                }
            }
        }
        ir::Expr::Binary { op, lhs, rhs } => {
            let op = binary_op(*op);
            let lhs = expr(lhs);
            let rhs = expr(rhs);
            quote! {
                ::unipute::ir::Expr::Binary {
                    op: #op,
                    lhs: ::std::boxed::Box::new(#lhs),
                    rhs: ::std::boxed::Box::new(#rhs),
                }
            }
        }
        ir::Expr::Cast { value, to } => {
            let value = expr(value);
            let to = scalar(*to);
            quote! {
                ::unipute::ir::Expr::Cast {
                    value: ::std::boxed::Box::new(#value),
                    to: #to,
                }
            }
        }
        ir::Expr::Math { function, args } => {
            let function = math_fn(*function);
            let args = args.iter().map(expr);
            quote! {
                ::unipute::ir::Expr::Math {
                    function: #function,
                    args: ::std::vec![#(#args),*],
                }
            }
        }
        ir::Expr::Compose {
            size,
            scalar: kind,
            components,
        } => {
            let size = vector_size(*size);
            let kind = scalar(*kind);
            let components = components.iter().map(expr);
            quote! {
                ::unipute::ir::Expr::Compose {
                    size: #size,
                    scalar: #kind,
                    components: ::std::vec![#(#components),*],
                }
            }
        }
        ir::Expr::ArrayLength(id) => {
            let id = id.0;
            quote!(::unipute::ir::Expr::ArrayLength(::unipute::ir::ResourceId(#id)))
        }
    }
}

fn stmt(node: &ir::Stmt) -> TokenStream {
    match node {
        ir::Stmt::Declare { local, value } => {
            let id = local.0;
            let value = match value {
                Some(value) => {
                    let value = expr(value);
                    quote!(::std::option::Option::Some(#value))
                }
                None => quote!(::std::option::Option::None),
            };
            quote! {
                ::unipute::ir::Stmt::Declare {
                    local: ::unipute::ir::LocalId(#id),
                    value: #value,
                }
            }
        }
        ir::Stmt::Store { place, value } => {
            let place = expr(place);
            let value = expr(value);
            quote!(::unipute::ir::Stmt::Store { place: #place, value: #value })
        }
        ir::Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = expr(condition);
            let then_branch = then_branch.iter().map(stmt);
            let else_branch = else_branch.iter().map(stmt);
            quote! {
                ::unipute::ir::Stmt::If {
                    condition: #condition,
                    then_branch: ::std::vec![#(#then_branch),*],
                    else_branch: ::std::vec![#(#else_branch),*],
                }
            }
        }
        ir::Stmt::While {
            condition,
            body,
            continuing,
        } => {
            let condition = expr(condition);
            let body = body.iter().map(stmt);
            let continuing = continuing.iter().map(stmt);
            quote! {
                ::unipute::ir::Stmt::While {
                    condition: #condition,
                    body: ::std::vec![#(#body),*],
                    continuing: ::std::vec![#(#continuing),*],
                }
            }
        }
        ir::Stmt::Break => quote!(::unipute::ir::Stmt::Break),
        ir::Stmt::Continue => quote!(::unipute::ir::Stmt::Continue),
        ir::Stmt::Return => quote!(::unipute::ir::Stmt::Return),
        ir::Stmt::Barrier(scope) => {
            let scope = match scope {
                ir::BarrierScope::Workgroup => quote!(::unipute::ir::BarrierScope::Workgroup),
                ir::BarrierScope::Storage => quote!(::unipute::ir::BarrierScope::Storage),
            };
            quote!(::unipute::ir::Stmt::Barrier(#scope))
        }
    }
}
