//! End to end checks from Unipute IR through naga to shader output.
//!
//! Every check here reads generated WGSL, so the whole file needs that
//! writer. Without it there is nothing to assert against.

#![cfg(feature = "wgsl")]

use unipute_ir::{
    Access, BinaryOp, BuiltIn, Expr, Kernel, Literal, Local, LocalId, MathFn, Resource, ResourceId,
    Scalar, Stmt, Type,
};

/// A kernel that scales one buffer into another, guarded by a bounds check.
///
/// It exercises resources with both access modes, a built-in, a local, an
/// `if`, indexing and arithmetic, which between them cover most of what the
/// lowering pass has to do.
fn scale_kernel() -> Kernel {
    let mut kernel = Kernel::new("scale", [64, 1, 1]);
    kernel.resources.push(Resource {
        name: "input".to_owned(),
        group: 0,
        binding: 0,
        ty: Type::slice(Type::scalar(Scalar::F32)),
        access: Access::Read,
    });
    kernel.resources.push(Resource {
        name: "output".to_owned(),
        group: 0,
        binding: 1,
        ty: Type::slice(Type::scalar(Scalar::F32)),
        access: Access::ReadWrite,
    });
    kernel.locals.push(Local {
        name: "index".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });

    let index = LocalId(0);
    kernel.body.push(Stmt::Declare {
        local: index,
        value: Some(Expr::Component {
            base: Box::new(Expr::BuiltIn(BuiltIn::GlobalInvocationId)),
            index: 0,
        }),
    });
    kernel.body.push(Stmt::If {
        condition: Expr::Binary {
            op: BinaryOp::GreaterEqual,
            lhs: Box::new(Expr::Local(index)),
            rhs: Box::new(Expr::ArrayLength(ResourceId(0))),
        },
        then_branch: vec![Stmt::Return],
        else_branch: Vec::new(),
    });
    kernel.body.push(Stmt::Store {
        place: Expr::Index {
            base: Box::new(Expr::Resource(ResourceId(1))),
            index: Box::new(Expr::Local(index)),
        },
        value: Expr::Binary {
            op: BinaryOp::Multiply,
            lhs: Box::new(Expr::Index {
                base: Box::new(Expr::Resource(ResourceId(0))),
                index: Box::new(Expr::Local(index)),
            }),
            rhs: Box::new(Expr::Literal(Literal::F32(2.0))),
        },
    });
    kernel
}

/// A kernel with a loop, a math call and a cast.
fn accumulate_kernel() -> Kernel {
    let mut kernel = Kernel::new("accumulate", [32, 2, 1]);
    kernel.resources.push(Resource {
        name: "data".to_owned(),
        group: 1,
        binding: 3,
        ty: Type::slice(Type::scalar(Scalar::F32)),
        access: Access::ReadWrite,
    });
    kernel.locals.push(Local {
        name: "i".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });
    kernel.locals.push(Local {
        name: "total".to_owned(),
        ty: Type::scalar(Scalar::F32),
    });

    let i = LocalId(0);
    let total = LocalId(1);
    kernel.body.push(Stmt::Declare {
        local: i,
        value: Some(Expr::Literal(Literal::U32(0))),
    });
    kernel.body.push(Stmt::Declare {
        local: total,
        value: Some(Expr::Literal(Literal::F32(0.0))),
    });
    kernel.body.push(Stmt::While {
        condition: Expr::Binary {
            op: BinaryOp::Less,
            lhs: Box::new(Expr::Local(i)),
            rhs: Box::new(Expr::Literal(Literal::U32(4))),
        },
        body: vec![Stmt::Store {
            place: Expr::Local(total),
            value: Expr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(Expr::Local(total)),
                rhs: Box::new(Expr::Math {
                    function: MathFn::Sqrt,
                    args: vec![Expr::Cast {
                        value: Box::new(Expr::Local(i)),
                        to: Scalar::F32,
                    }],
                }),
            },
        }],
        continuing: vec![Stmt::Store {
            place: Expr::Local(i),
            value: Expr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(Expr::Local(i)),
                rhs: Box::new(Expr::Literal(Literal::U32(1))),
            },
        }],
    });
    kernel.body.push(Stmt::Store {
        place: Expr::Index {
            base: Box::new(Expr::Resource(ResourceId(0))),
            index: Box::new(Expr::Literal(Literal::U32(0))),
        },
        value: Expr::Local(total),
    });
    kernel
}

#[test]
fn scale_kernel_produces_wgsl() {
    let wgsl = unipute_naga::compile_wgsl(&scale_kernel()).unwrap();

    assert!(
        wgsl.contains("@compute @workgroup_size(64, 1, 1)"),
        "{wgsl}"
    );
    assert!(wgsl.contains("fn scale("), "{wgsl}");
    assert!(
        wgsl.contains("@group(0) @binding(0)") && wgsl.contains("@group(0) @binding(1)"),
        "{wgsl}"
    );
    assert!(wgsl.contains("read_write"), "{wgsl}");
    assert!(wgsl.contains("global_invocation_id"), "{wgsl}");
    assert!(wgsl.contains("arrayLength"), "{wgsl}");
}

#[test]
fn accumulate_kernel_produces_a_loop() {
    let wgsl = unipute_naga::compile_wgsl(&accumulate_kernel()).unwrap();

    assert!(wgsl.contains("@workgroup_size(32, 2, 1)"), "{wgsl}");
    assert!(wgsl.contains("loop {"), "{wgsl}");
    assert!(wgsl.contains("break"), "{wgsl}");
    assert!(wgsl.contains("sqrt("), "{wgsl}");
    assert!(wgsl.contains("@group(1) @binding(3)"), "{wgsl}");
}

#[test]
fn lowering_is_deterministic() {
    let first = unipute_naga::compile_wgsl(&scale_kernel()).unwrap();
    let second = unipute_naga::compile_wgsl(&scale_kernel()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn graphics_stages_are_rejected_with_a_clear_message() {
    let mut kernel = scale_kernel();
    kernel.stage = unipute_ir::Stage::Vertex;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("vertex"), "{error}");
}

#[test]
fn a_zero_workgroup_dimension_is_rejected() {
    let mut kernel = scale_kernel();
    kernel.workgroup_size = [64, 0, 1];

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("at least 1"), "{error}");
}

#[test]
fn a_uniform_slice_is_rejected() {
    let mut kernel = scale_kernel();
    kernel.resources[0].access = Access::Uniform;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("cannot be a slice"), "{error}");
}

#[test]
#[ignore = "prints generated shaders for eyeballing"]
fn dump() {
    println!("{}", unipute_naga::compile_wgsl(&scale_kernel()).unwrap());
    println!(
        "{}",
        unipute_naga::compile_wgsl(&accumulate_kernel()).unwrap()
    );
}
