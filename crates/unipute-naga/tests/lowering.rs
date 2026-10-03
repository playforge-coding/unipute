//! End to end checks from Unipute IR through naga to shader output.
//!
//! Every check here reads generated WGSL, so the whole file needs that
//! writer. Without it there is nothing to assert against.

#![cfg(feature = "wgsl")]

use unipute_ir::{
    Access, AtomicOp, BarrierScope, BinaryOp, BuiltIn, Expr, Function, FunctionId, Kernel, Literal,
    Local, LocalId, MathFn, Param, ParamId, Resource, ResourceId, Scalar, Shared, SharedId, Stmt,
    StructType, Type, VectorSize,
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
        then_branch: vec![Stmt::Return { value: None }],
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

/// A kernel that calls two helpers, one of which calls the other.
///
/// This is where a call has to be hoisted out of the expression it sits in,
/// since naga has no call expression, and where the order of
/// `Kernel::functions` has to reach the generated shader intact.
fn helper_kernel() -> Kernel {
    let mut kernel = Kernel::new("mix_pair", [64, 1, 1]);
    kernel.resources.push(Resource {
        name: "output".to_owned(),
        group: 0,
        binding: 0,
        ty: Type::slice(Type::scalar(Scalar::F32)),
        access: Access::ReadWrite,
    });

    // `double(x) = x * 2.0`
    let mut double = Function::new("double");
    double.params.push(Param {
        name: "x".to_owned(),
        ty: Type::scalar(Scalar::F32),
    });
    double.result = Some(Type::scalar(Scalar::F32));
    double.body.push(Stmt::Return {
        value: Some(Expr::Binary {
            op: BinaryOp::Multiply,
            lhs: Box::new(Expr::Param(ParamId(0))),
            rhs: Box::new(Expr::Literal(Literal::F32(2.0))),
        }),
    });

    // `blend(a, b) = double(a) + b`, which calls the helper above.
    let mut blend = Function::new("blend");
    blend.params.push(Param {
        name: "a".to_owned(),
        ty: Type::scalar(Scalar::F32),
    });
    blend.params.push(Param {
        name: "b".to_owned(),
        ty: Type::scalar(Scalar::F32),
    });
    blend.result = Some(Type::scalar(Scalar::F32));
    blend.body.push(Stmt::Return {
        value: Some(Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(Expr::Call {
                function: FunctionId(0),
                args: vec![Expr::Param(ParamId(0))],
            }),
            rhs: Box::new(Expr::Param(ParamId(1))),
        }),
    });

    kernel.functions.push(double);
    kernel.functions.push(blend);

    kernel.body.push(Stmt::Store {
        place: Expr::Index {
            base: Box::new(Expr::Resource(ResourceId(0))),
            index: Box::new(Expr::Component {
                base: Box::new(Expr::BuiltIn(BuiltIn::GlobalInvocationId)),
                index: 0,
            }),
        },
        value: Expr::Call {
            function: FunctionId(1),
            args: vec![
                Expr::Literal(Literal::F32(1.5)),
                Expr::Literal(Literal::F32(0.5)),
            ],
        },
    });
    kernel
}

/// A kernel that sums each workgroup's slice of the input through workgroup
/// memory: every invocation drops its element into a tile, a barrier makes
/// the writes visible, and one invocation per group adds the tile up.
fn block_sum_kernel() -> Kernel {
    let mut kernel = Kernel::new("block_sum", [64, 1, 1]);
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
    kernel.shared.push(Shared {
        name: "tile".to_owned(),
        ty: Type::Array {
            element: Box::new(Type::scalar(Scalar::F32)),
            len: Some(64),
        },
    });
    kernel.locals.push(Local {
        name: "lane".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });
    kernel.locals.push(Local {
        name: "total".to_owned(),
        ty: Type::scalar(Scalar::F32),
    });
    kernel.locals.push(Local {
        name: "i".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });

    let lane = LocalId(0);
    let total = LocalId(1);
    let i = LocalId(2);
    let tile = SharedId(0);

    kernel.body.push(Stmt::Declare {
        local: lane,
        value: Some(Expr::BuiltIn(BuiltIn::LocalInvocationIndex)),
    });
    kernel.body.push(Stmt::Store {
        place: Expr::Index {
            base: Box::new(Expr::Shared(tile)),
            index: Box::new(Expr::Local(lane)),
        },
        value: Expr::Index {
            base: Box::new(Expr::Resource(ResourceId(0))),
            index: Box::new(Expr::Component {
                base: Box::new(Expr::BuiltIn(BuiltIn::GlobalInvocationId)),
                index: 0,
            }),
        },
    });
    kernel.body.push(Stmt::Barrier(BarrierScope::Workgroup));
    kernel.body.push(Stmt::If {
        condition: Expr::Binary {
            op: BinaryOp::Equal,
            lhs: Box::new(Expr::Local(lane)),
            rhs: Box::new(Expr::Literal(Literal::U32(0))),
        },
        then_branch: vec![
            Stmt::Declare {
                local: total,
                value: Some(Expr::Literal(Literal::F32(0.0))),
            },
            Stmt::Declare {
                local: i,
                value: Some(Expr::Literal(Literal::U32(0))),
            },
            Stmt::While {
                condition: Expr::Binary {
                    op: BinaryOp::Less,
                    lhs: Box::new(Expr::Local(i)),
                    rhs: Box::new(Expr::Literal(Literal::U32(64))),
                },
                body: vec![Stmt::Store {
                    place: Expr::Local(total),
                    value: Expr::Binary {
                        op: BinaryOp::Add,
                        lhs: Box::new(Expr::Local(total)),
                        rhs: Box::new(Expr::Index {
                            base: Box::new(Expr::Shared(tile)),
                            index: Box::new(Expr::Local(i)),
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
            },
            Stmt::Store {
                place: Expr::Index {
                    base: Box::new(Expr::Resource(ResourceId(1))),
                    index: Box::new(Expr::Component {
                        base: Box::new(Expr::BuiltIn(BuiltIn::WorkgroupId)),
                        index: 0,
                    }),
                },
                value: Expr::Local(total),
            },
        ],
        else_branch: Vec::new(),
    });
    kernel
}

fn particle_type() -> StructType {
    StructType::new(
        "Particle",
        [
            (
                "position".to_owned(),
                Type::vector(VectorSize::Three, Scalar::F32),
            ),
            ("mass".to_owned(), Type::scalar(Scalar::F32)),
        ],
    )
    .unwrap()
}

/// A kernel over a buffer of structs and a uniform struct: it reads a member
/// through an index, reads a member of the uniform, builds a new struct and
/// stores it back whole, and assigns to one member in place.
fn particle_kernel() -> Kernel {
    let particle = particle_type();
    let settings = StructType::new(
        "Settings",
        [
            (
                "gravity".to_owned(),
                Type::vector(VectorSize::Three, Scalar::F32),
            ),
            ("dt".to_owned(), Type::scalar(Scalar::F32)),
        ],
    )
    .unwrap();

    let mut kernel = Kernel::new("integrate", [64, 1, 1]);
    kernel.resources.push(Resource {
        name: "particles".to_owned(),
        group: 0,
        binding: 0,
        ty: Type::slice(Type::Struct(particle.clone())),
        access: Access::ReadWrite,
    });
    kernel.resources.push(Resource {
        name: "settings".to_owned(),
        group: 0,
        binding: 1,
        ty: Type::Struct(settings),
        access: Access::Uniform,
    });
    kernel.locals.push(Local {
        name: "index".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });

    let index = LocalId(0);
    let element = || Expr::Index {
        base: Box::new(Expr::Resource(ResourceId(0))),
        index: Box::new(Expr::Local(index)),
    };
    kernel.body.push(Stmt::Declare {
        local: index,
        value: Some(Expr::Component {
            base: Box::new(Expr::BuiltIn(BuiltIn::GlobalInvocationId)),
            index: 0,
        }),
    });
    // particles[index] = Particle { position: particles[index].position + settings.gravity, mass: particles[index].mass }
    kernel.body.push(Stmt::Store {
        place: element(),
        value: Expr::Construct {
            ty: particle,
            members: vec![
                Expr::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(Expr::Member {
                        base: Box::new(element()),
                        index: 0,
                    }),
                    rhs: Box::new(Expr::Member {
                        base: Box::new(Expr::Resource(ResourceId(1))),
                        index: 0,
                    }),
                },
                Expr::Member {
                    base: Box::new(element()),
                    index: 1,
                },
            ],
        },
    });
    // particles[index].mass = settings.dt
    kernel.body.push(Stmt::Store {
        place: Expr::Member {
            base: Box::new(element()),
            index: 1,
        },
        value: Expr::Member {
            base: Box::new(Expr::Resource(ResourceId(1))),
            index: 1,
        },
    });
    kernel
}

/// A histogram: every invocation counts its value into a workgroup bin, then
/// one invocation per bin adds the bin into the global one. Between them the
/// statements use an atomic for its effect, an atomic for its old value, a
/// load and a store of an atomic, and a compare and exchange.
fn histogram_kernel() -> Kernel {
    let mut kernel = Kernel::new("histogram", [64, 1, 1]);
    kernel.resources.push(Resource {
        name: "values".to_owned(),
        group: 0,
        binding: 0,
        ty: Type::slice(Type::scalar(Scalar::U32)),
        access: Access::Read,
    });
    kernel.resources.push(Resource {
        name: "bins".to_owned(),
        group: 0,
        binding: 1,
        ty: Type::slice(Type::Atomic(Scalar::U32)),
        access: Access::ReadWrite,
    });
    kernel.shared.push(Shared {
        name: "local_bins".to_owned(),
        ty: Type::Array {
            element: Box::new(Type::Atomic(Scalar::U32)),
            len: Some(4),
        },
    });
    kernel.locals.push(Local {
        name: "lane".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });
    kernel.locals.push(Local {
        name: "previous".to_owned(),
        ty: Type::scalar(Scalar::U32),
    });

    let lane = LocalId(0);
    let previous = LocalId(1);
    let values = ResourceId(0);
    let bins = ResourceId(1);
    let local_bins = SharedId(0);
    let global_bin = || Expr::Index {
        base: Box::new(Expr::Resource(bins)),
        index: Box::new(Expr::Local(lane)),
    };
    let local_bin = || Expr::Index {
        base: Box::new(Expr::Shared(local_bins)),
        index: Box::new(Expr::Local(lane)),
    };

    kernel.body.push(Stmt::Declare {
        local: lane,
        value: Some(Expr::BuiltIn(BuiltIn::LocalInvocationIndex)),
    });
    // local_bins[values[global_id().x] % 4].fetch_add(1);
    kernel.body.push(Stmt::Atomic {
        op: AtomicOp::Add,
        place: Expr::Index {
            base: Box::new(Expr::Shared(local_bins)),
            index: Box::new(Expr::Binary {
                op: BinaryOp::Modulo,
                lhs: Box::new(Expr::Index {
                    base: Box::new(Expr::Resource(values)),
                    index: Box::new(Expr::Component {
                        base: Box::new(Expr::BuiltIn(BuiltIn::GlobalInvocationId)),
                        index: 0,
                    }),
                }),
                rhs: Box::new(Expr::Literal(Literal::U32(4))),
            }),
        },
        value: Expr::Literal(Literal::U32(1)),
    });
    kernel.body.push(Stmt::Barrier(BarrierScope::Workgroup));
    kernel.body.push(Stmt::If {
        condition: Expr::Binary {
            op: BinaryOp::Less,
            lhs: Box::new(Expr::Local(lane)),
            rhs: Box::new(Expr::Literal(Literal::U32(4))),
        },
        then_branch: vec![
            // let previous = bins[lane].fetch_add(local_bins[lane].load());
            Stmt::Declare {
                local: previous,
                value: Some(Expr::Atomic {
                    op: AtomicOp::Add,
                    place: Box::new(global_bin()),
                    value: Box::new(local_bin()),
                }),
            },
            // if bins[lane].compare_exchange(previous, 0) { local_bins[lane].store(0); }
            Stmt::If {
                condition: Expr::AtomicCompareExchange {
                    place: Box::new(global_bin()),
                    compare: Box::new(Expr::Local(previous)),
                    value: Box::new(Expr::Literal(Literal::U32(0))),
                },
                then_branch: vec![Stmt::Store {
                    place: local_bin(),
                    value: Expr::Literal(Literal::U32(0)),
                }],
                else_branch: Vec::new(),
            },
        ],
        else_branch: Vec::new(),
    });
    kernel
}

#[test]
fn atomics_come_out_as_atomic_calls() {
    let wgsl = unipute_naga::compile_wgsl(&histogram_kernel()).unwrap();

    assert!(wgsl.contains("array<atomic<u32>>"), "{wgsl}");
    assert!(
        wgsl.contains("var<workgroup> local_bins: array<atomic<u32>, 4>"),
        "{wgsl}"
    );
    // The first add is a statement with no result, the second binds its old
    // value.
    assert!(wgsl.contains("atomicAdd("), "{wgsl}");
    assert!(wgsl.contains("= atomicAdd("), "{wgsl}");
    assert!(wgsl.contains("atomicLoad("), "{wgsl}");
    assert!(wgsl.contains("atomicStore("), "{wgsl}");
    assert!(wgsl.contains("atomicCompareExchangeWeak("), "{wgsl}");
    assert!(wgsl.contains(".exchanged"), "{wgsl}");
}

#[test]
fn a_swap_for_its_effect_still_binds_a_result() {
    let mut kernel = histogram_kernel();
    let Stmt::Atomic { op, .. } = &mut kernel.body[1] else {
        unreachable!("the second statement is the atomic add");
    };
    *op = AtomicOp::Swap;

    // Naga requires an exchange to bind its old value, so the lowering has
    // to make one up. The interesting part is that this compiles at all.
    let wgsl = unipute_naga::compile_wgsl(&kernel).unwrap();
    assert!(wgsl.contains("atomicExchange("), "{wgsl}");
}

#[test]
fn an_atomic_in_a_read_only_buffer_is_rejected() {
    let mut kernel = histogram_kernel();
    kernel.resources[1].access = Access::Read;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("read and write"), "{error}");
}

#[test]
fn an_atomic_operation_on_a_plain_integer_is_rejected() {
    let mut kernel = histogram_kernel();
    kernel.resources[1].ty = Type::slice(Type::scalar(Scalar::U32));

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("needs an atomic"), "{error}");
}

#[test]
fn a_float_atomic_is_rejected() {
    let mut kernel = histogram_kernel();
    kernel.resources[1].ty = Type::slice(Type::Atomic(Scalar::F32));

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("not a `f32`"), "{error}");
}

#[cfg(feature = "msl")]
#[test]
fn msl_writes_atomics() {
    let msl = unipute_naga::compile_msl(&histogram_kernel()).unwrap();
    assert!(msl.contains("atomic_fetch_add_explicit"), "{msl}");
    assert!(
        msl.contains("atomic_compare_exchange_weak_explicit"),
        "{msl}"
    );
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_writes_atomics() {
    let hlsl = unipute_naga::compile_hlsl(&histogram_kernel()).unwrap();
    assert!(hlsl.contains("InterlockedAdd"), "{hlsl}");
    assert!(hlsl.contains("InterlockedCompareExchange"), "{hlsl}");
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_writes_atomics() {
    let glsl = unipute_naga::compile_glsl(&histogram_kernel()).unwrap();
    assert!(glsl.contains("atomicAdd("), "{glsl}");
    assert!(glsl.contains("atomicCompSwap("), "{glsl}");
}

#[cfg(feature = "spv")]
#[test]
fn spirv_takes_atomics() {
    let words = unipute_naga::compile_spirv(&histogram_kernel()).unwrap();
    assert_eq!(words.first(), Some(&0x0723_0203));
}

#[test]
fn structs_are_declared_with_their_members() {
    let wgsl = unipute_naga::compile_wgsl(&particle_kernel()).unwrap();

    assert!(wgsl.contains("struct Particle {"), "{wgsl}");
    assert!(wgsl.contains("position: vec3<f32>,"), "{wgsl}");
    assert!(wgsl.contains("mass: f32,"), "{wgsl}");
    assert!(wgsl.contains("array<Particle>"), "{wgsl}");
    assert!(wgsl.contains("var<uniform> settings: Settings"), "{wgsl}");
    // Building a struct comes out as a constructor call, and a member store
    // as an assignment through the index.
    assert!(wgsl.contains("Particle("), "{wgsl}");
    assert!(wgsl.contains("].mass = "), "{wgsl}");
}

#[test]
fn a_struct_with_the_wrong_offsets_is_rejected() {
    let mut kernel = particle_kernel();
    let Type::Array { element, .. } = &mut kernel.resources[0].ty else {
        unreachable!("the first resource is a buffer");
    };
    let Type::Struct(def) = &mut **element else {
        unreachable!("the buffer holds structs");
    };
    // `mass` fits in the gap after a `vec3`, at byte 12. Pushing it out to 16
    // is a layout naga's writers would each handle their own way.
    def.members[1].offset = 16;
    def.size = 32;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("at byte 16"), "{error}");
    assert!(error.to_string().contains("byte 12"), "{error}");
}

#[test]
fn a_struct_with_the_wrong_size_is_rejected() {
    let mut kernel = particle_kernel();
    let Type::Struct(def) = &mut kernel.resources[1].ty else {
        unreachable!("the second resource is a uniform struct");
    };
    def.size = 20;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("20 bytes"), "{error}");
}

#[test]
fn constructing_a_struct_needs_every_member() {
    let mut kernel = particle_kernel();
    let Stmt::Store { value, .. } = &mut kernel.body[1] else {
        unreachable!("the second statement stores a new struct");
    };
    let Expr::Construct { members, .. } = value else {
        unreachable!("the value is a struct literal");
    };
    members.pop();

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("2 members"), "{error}");
}

#[cfg(feature = "msl")]
#[test]
fn msl_packs_a_vec3_that_has_a_member_after_it() {
    // Metal's `float3` is 16 bytes, so `mass` at byte 12 needs the packed
    // form of the vector. Naga writes it, and this is here to notice if that
    // ever changes.
    let msl = unipute_naga::compile_msl(&particle_kernel()).unwrap();
    assert!(msl.contains("packed_float3 position"), "{msl}");
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_writes_the_struct() {
    let hlsl = unipute_naga::compile_hlsl(&particle_kernel()).unwrap();
    assert!(hlsl.contains("struct Particle {"), "{hlsl}");
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_writes_the_struct() {
    let glsl = unipute_naga::compile_glsl(&particle_kernel()).unwrap();
    assert!(glsl.contains("struct Particle {"), "{glsl}");
}

#[cfg(feature = "spv")]
#[test]
fn spirv_takes_the_struct() {
    let words = unipute_naga::compile_spirv(&particle_kernel()).unwrap();
    assert_eq!(words.first(), Some(&0x0723_0203));
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
fn helpers_are_written_before_the_entry_point() {
    let wgsl = unipute_naga::compile_wgsl(&helper_kernel()).unwrap();

    let double = wgsl.find("fn double(").expect("double is written");
    let blend = wgsl.find("fn blend(").expect("blend is written");
    let entry = wgsl
        .find("fn mix_pair(")
        .expect("the entry point is written");

    // A shader language has no forward declarations, so the order a call can
    // be resolved in is the order the functions have to appear in.
    assert!(double < blend, "{wgsl}");
    assert!(blend < entry, "{wgsl}");
}

#[test]
fn a_call_inside_an_expression_becomes_its_own_statement() {
    let wgsl = unipute_naga::compile_wgsl(&helper_kernel()).unwrap();

    // Naga has no call expression, so `double(a) + b` has to come out as a
    // call bound to a name and then the addition.
    assert!(wgsl.contains("double(a)"), "{wgsl}");
    assert!(wgsl.contains("fn blend(a: f32, b: f32) -> f32"), "{wgsl}");
    assert!(wgsl.contains("blend(1.5f, 0.5f)"), "{wgsl}");
}

#[test]
fn calling_a_function_that_is_not_there_yet_is_rejected() {
    let mut kernel = helper_kernel();
    // Turn the call around so `double`, which is lowered first, is the one
    // calling `blend`. That breaks the order `Kernel::functions` documents,
    // and the back end has no way to work around it.
    kernel.functions[1].body = vec![Stmt::Return {
        value: Some(Expr::Param(ParamId(1))),
    }];
    kernel.functions[0].body = vec![Stmt::Return {
        value: Some(Expr::Call {
            function: FunctionId(1),
            args: vec![Expr::Param(ParamId(0)), Expr::Param(ParamId(0))],
        }),
    }];

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(
        error.to_string().contains("before it is defined"),
        "{error}"
    );
}

#[test]
fn calling_a_function_that_returns_nothing_for_a_value_is_rejected() {
    let mut kernel = helper_kernel();
    kernel.functions[0].result = None;

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("returns nothing"), "{error}");
}

#[test]
fn workgroup_memory_is_declared_without_a_binding() {
    let wgsl = unipute_naga::compile_wgsl(&block_sum_kernel()).unwrap();

    assert!(
        wgsl.contains("var<workgroup> tile: array<f32, 64>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("workgroupBarrier()"), "{wgsl}");
    // The two buffers are bound. The tile is not, since the host never sees
    // it, so it must not take a binding slot.
    assert_eq!(wgsl.matches("@binding(").count(), 2, "{wgsl}");
}

#[test]
fn using_a_workgroup_array_as_a_value_is_rejected() {
    let mut kernel = block_sum_kernel();
    kernel.body = vec![Stmt::Store {
        place: Expr::Index {
            base: Box::new(Expr::Resource(ResourceId(1))),
            index: Box::new(Expr::Literal(Literal::U32(0))),
        },
        value: Expr::Shared(SharedId(0)),
    }];

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("index it"), "{error}");
}

#[test]
fn runtime_sized_workgroup_memory_is_rejected() {
    let mut kernel = block_sum_kernel();
    kernel.shared[0].ty = Type::slice(Type::scalar(Scalar::F32));

    let error = unipute_naga::compile_wgsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("fixed length"), "{error}");
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

/// GLSL has no groups, so a resource's binding number has to be written out
/// explicitly or a driver assigns whatever it likes.
#[cfg(feature = "glsl")]
#[test]
fn glsl_writes_each_resource_with_its_binding_number() {
    let glsl = unipute_naga::compile_glsl(&scale_kernel()).unwrap();
    assert!(glsl.contains("binding = 0)"), "{glsl}");
    assert!(glsl.contains("binding = 1)"), "{glsl}");

    // Group 1, binding 3: the group goes, the number stays.
    let glsl = unipute_naga::compile_glsl(&accumulate_kernel()).unwrap();
    assert!(glsl.contains("binding = 3)"), "{glsl}");
}

#[cfg(feature = "glsl")]
#[test]
fn glsl_rejects_a_binding_number_shared_across_groups() {
    let mut kernel = scale_kernel();
    kernel.resources[1].group = 1;
    kernel.resources[1].binding = 0;

    let error = unipute_naga::compile_glsl(&kernel).unwrap_err();
    assert!(error.to_string().contains("both use binding 0"), "{error}");
    assert!(error.to_string().contains("`input`"), "{error}");
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
