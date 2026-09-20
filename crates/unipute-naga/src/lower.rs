//! Lowering from Unipute IR to naga IR.
//!
//! Naga has two rules that shape most of the code here.
//!
//! The first is that expressions live in a flat arena and every expression
//! must appear before the expressions that use it. That falls out naturally
//! from lowering a tree bottom up.
//!
//! The second is that most expressions have to be covered by a
//! [`Statement::Emit`] range, but a handful must not be. The ones that must
//! not be are literals, constants, overrides, zero values, function arguments,
//! globals and locals. We create every one of those up front, before any emit
//! range is open, and look them up from a cache while lowering.
//!
//! A call's result belongs to that second group as well, and is the awkward
//! case: naga has no call expression, so [`Statement::Call`] binds its result
//! to an [`Expression::CallResult`], which cannot be made before the call is
//! known. Lowering one in the middle of an expression therefore closes the
//! open emit range, pushes the call, and opens a new range. That is why the
//! expression lowering carries a block and an emitter around with it.

use std::collections::HashMap;

use naga::{Arena, Block, Expression, Handle, Span, Statement, UniqueArena};
use unipute_ir as ir;

use crate::error::{Error, Result};

const SPAN: Span = Span::UNDEFINED;

/// Lowers a kernel into a naga module holding its helper functions and a
/// single entry point.
pub fn lower(kernel: &ir::Kernel) -> Result<naga::Module> {
    if !kernel.stage.is_implemented() {
        return Err(Error::UnsupportedStage(kernel.stage));
    }
    if kernel.workgroup_size.contains(&0) {
        return Err(Error::EmptyWorkgroup);
    }

    let mut module = naga::Module::default();
    let mut types = Types::default();
    let globals = lower_resources(kernel, &mut module, &mut types)?;

    let mut shared = Shared {
        kernel,
        types,
        globals,
        functions: Vec::new(),
    };

    // Helpers come first. `Kernel::functions` is ordered with callees before
    // callers, so every call already has a handle to name by the time it is
    // lowered, which is what a shader language without forward declarations
    // needs.
    for helper in &kernel.functions {
        let lowered = shared.lower_function(
            &mut module.types,
            Signature {
                name: helper.name.clone(),
                arguments: Arguments::Params(&helper.params),
                result: helper.result.as_ref(),
            },
            &helper.locals,
            &helper.body,
        )?;
        let handle = module.functions.append(lowered, SPAN);
        shared.functions.push(handle);
    }

    // Built-ins become entry point arguments. Only the ones the body actually
    // reads are declared, so the generated shader stays close to what someone
    // would have written by hand.
    let used = used_built_ins(&kernel.body);
    let function = shared.lower_function(
        &mut module.types,
        Signature {
            name: kernel.name.clone(),
            arguments: Arguments::BuiltIns(&used),
            result: None,
        },
        &kernel.locals,
        &kernel.body,
    )?;

    module.entry_points.push(naga::EntryPoint {
        name: kernel.name.clone(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: kernel.workgroup_size,
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    Ok(module)
}

/// What the function being lowered takes as arguments. This is the one place
/// the entry point and a helper really differ: the entry point is handed
/// built-ins by the hardware, a helper is handed its declared parameters.
enum Arguments<'a> {
    BuiltIns(&'a [ir::BuiltIn]),
    Params(&'a [ir::Param]),
}

/// Everything about a function that is decided before its body is read.
struct Signature<'a> {
    name: String,
    arguments: Arguments<'a>,
    result: Option<&'a ir::Type>,
}

/// The parts of the module that every function in it shares.
struct Shared<'a> {
    kernel: &'a ir::Kernel,
    types: Types,
    globals: Vec<Handle<naga::GlobalVariable>>,
    /// One handle per entry in `Kernel::functions`, filled in as they are
    /// lowered. A call looks its target up here.
    functions: Vec<Handle<naga::Function>>,
}

impl Shared<'_> {
    fn lower_function(
        &mut self,
        module_types: &mut UniqueArena<naga::Type>,
        signature: Signature<'_>,
        locals: &[ir::Local],
        body: &[ir::Stmt],
    ) -> Result<naga::Function> {
        let mut function = naga::Function {
            name: Some(signature.name),
            ..Default::default()
        };

        match &signature.arguments {
            Arguments::BuiltIns(built_ins) => {
                for built_in in *built_ins {
                    let ty = self.types.built_in(module_types, *built_in);
                    function.arguments.push(naga::FunctionArgument {
                        name: Some(built_in.intrinsic_name().to_owned()),
                        ty,
                        binding: Some(naga::Binding::BuiltIn(naga_built_in(*built_in))),
                    });
                }
            }
            Arguments::Params(params) => {
                for param in *params {
                    let ty = self.types.lower(module_types, &param.ty)?;
                    function.arguments.push(naga::FunctionArgument {
                        name: Some(param.name.clone()),
                        ty,
                        binding: None,
                    });
                }
            }
        }

        if let Some(result) = signature.result {
            let ty = self.types.lower(module_types, result)?;
            function.result = Some(naga::FunctionResult { ty, binding: None });
        }

        let mut ctx = Context {
            kernel: self.kernel,
            types: &mut self.types,
            module_types,
            globals: &self.globals,
            functions: &self.functions,
            source_locals: locals,
            locals: Vec::new(),
            params: Vec::new(),
            built_in_exprs: HashMap::new(),
            global_exprs: HashMap::new(),
            literals: HashMap::new(),
            expressions: core::mem::take(&mut function.expressions),
            local_variables: core::mem::take(&mut function.local_variables),
        };
        ctx.prepare_pre_emit(&signature.arguments, body)?;
        let lowered = ctx.lower_block(body)?;

        function.expressions = ctx.expressions;
        function.local_variables = ctx.local_variables;
        function.body = lowered;
        Ok(function)
    }
}

fn lower_resources(
    kernel: &ir::Kernel,
    module: &mut naga::Module,
    types: &mut Types,
) -> Result<Vec<Handle<naga::GlobalVariable>>> {
    let mut globals = Vec::with_capacity(kernel.resources.len());
    for resource in &kernel.resources {
        if matches!(resource.access, ir::Access::Uniform)
            && matches!(resource.ty, ir::Type::Array { .. })
        {
            return Err(Error::UnsupportedType(format!(
                "uniform resource `{}` cannot be a slice, take it by `&mut` or pass a length",
                resource.name
            )));
        }
        let ty = types.lower(&mut module.types, &resource.ty)?;
        let space = match resource.access {
            ir::Access::Read => naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD,
            },
            ir::Access::ReadWrite => naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
            },
            ir::Access::Uniform => naga::AddressSpace::Uniform,
        };
        let handle = module.global_variables.append(
            naga::GlobalVariable {
                name: Some(resource.name.clone()),
                space,
                binding: Some(naga::ResourceBinding {
                    group: resource.group,
                    binding: resource.binding,
                }),
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            SPAN,
        );
        globals.push(handle);
    }
    Ok(globals)
}

/// Caches the naga type handles we hand out, so the same Unipute type always
/// maps to the same handle.
#[derive(Default)]
struct Types {
    cache: HashMap<ir::Type, Handle<naga::Type>>,
}

impl Types {
    fn lower(
        &mut self,
        arena: &mut UniqueArena<naga::Type>,
        ty: &ir::Type,
    ) -> Result<Handle<naga::Type>> {
        if let Some(handle) = self.cache.get(ty) {
            return Ok(*handle);
        }
        let inner = match ty {
            ir::Type::Scalar(scalar) => naga::TypeInner::Scalar(naga_scalar(*scalar)),
            ir::Type::Vector { size, scalar } => naga::TypeInner::Vector {
                size: naga_vector_size(*size),
                scalar: naga_scalar(*scalar),
            },
            ir::Type::Array { element, len } => {
                let base = self.lower(arena, element)?;
                let stride = array_stride(element)?;
                let size = match len {
                    Some(len) => {
                        naga::ArraySize::Constant(core::num::NonZeroU32::new(*len).ok_or_else(
                            || Error::UnsupportedType("zero length array".to_owned()),
                        )?)
                    }
                    None => naga::ArraySize::Dynamic,
                };
                naga::TypeInner::Array { base, size, stride }
            }
        };
        let handle = arena.insert(naga::Type { name: None, inner }, SPAN);
        self.cache.insert(ty.clone(), handle);
        Ok(handle)
    }

    fn built_in(
        &mut self,
        arena: &mut UniqueArena<naga::Type>,
        built_in: ir::BuiltIn,
    ) -> Handle<naga::Type> {
        let ty = built_in_type(built_in);
        // Built-in types are scalars and vectors, which `lower` never rejects.
        self.lower(arena, &ty).expect("built-in types always lower")
    }
}

fn built_in_type(built_in: ir::BuiltIn) -> ir::Type {
    match built_in {
        ir::BuiltIn::LocalInvocationIndex => ir::Type::scalar(ir::Scalar::U32),
        _ => ir::Type::vector(ir::VectorSize::Three, ir::Scalar::U32),
    }
}

/// Size in bytes of a value of this type, following the std430 style rules the
/// shader languages agree on for scalars and vectors.
fn type_size(ty: &ir::Type) -> Result<u32> {
    Ok(match ty {
        ir::Type::Scalar(scalar) => u32::from(scalar.width()),
        ir::Type::Vector { size, scalar } => {
            let width = u32::from(scalar.width());
            match size {
                // A three component vector is padded out to four.
                ir::VectorSize::Two => 2 * width,
                ir::VectorSize::Three | ir::VectorSize::Four => 4 * width,
            }
        }
        ir::Type::Array {
            element,
            len: Some(len),
        } => array_stride(element)? * len,
        ir::Type::Array { len: None, .. } => {
            return Err(Error::UnsupportedType(
                "a runtime sized array cannot be nested inside another type".to_owned(),
            ));
        }
    })
}

fn array_stride(element: &ir::Type) -> Result<u32> {
    type_size(element)
}

fn naga_scalar(scalar: ir::Scalar) -> naga::Scalar {
    let kind = match scalar {
        ir::Scalar::Bool => naga::ScalarKind::Bool,
        ir::Scalar::I32 => naga::ScalarKind::Sint,
        ir::Scalar::U32 => naga::ScalarKind::Uint,
        ir::Scalar::F32 => naga::ScalarKind::Float,
    };
    naga::Scalar {
        kind,
        width: scalar.width(),
    }
}

fn naga_vector_size(size: ir::VectorSize) -> naga::VectorSize {
    match size {
        ir::VectorSize::Two => naga::VectorSize::Bi,
        ir::VectorSize::Three => naga::VectorSize::Tri,
        ir::VectorSize::Four => naga::VectorSize::Quad,
    }
}

fn naga_built_in(built_in: ir::BuiltIn) -> naga::BuiltIn {
    match built_in {
        ir::BuiltIn::GlobalInvocationId => naga::BuiltIn::GlobalInvocationId,
        ir::BuiltIn::LocalInvocationId => naga::BuiltIn::LocalInvocationId,
        ir::BuiltIn::LocalInvocationIndex => naga::BuiltIn::LocalInvocationIndex,
        ir::BuiltIn::WorkgroupId => naga::BuiltIn::WorkGroupId,
        ir::BuiltIn::NumWorkgroups => naga::BuiltIn::NumWorkGroups,
    }
}

/// Whether a statement may appear in a loop's continuing block.
///
/// Naga allows straight line code and nested `if`s there, but nothing that
/// leaves the block.
fn allowed_in_continuing(stmt: &ir::Stmt) -> bool {
    match stmt {
        ir::Stmt::Declare { .. } | ir::Stmt::Store { .. } | ir::Stmt::Barrier(_) => true,
        ir::Stmt::If {
            then_branch,
            else_branch,
            ..
        } => then_branch
            .iter()
            .chain(else_branch)
            .all(allowed_in_continuing),
        // A call is not allowed either. Naga's continuing block takes straight
        // line code only, and nothing a front end puts in there needs one.
        ir::Stmt::While { .. }
        | ir::Stmt::Break
        | ir::Stmt::Continue
        | ir::Stmt::Return { .. }
        | ir::Stmt::Call { .. } => false,
    }
}

/// Whether an expression produces a value directly rather than naming memory.
/// Reading a component of one of these is an access on the value, since there
/// is no pointer to load through.
fn names_no_memory(expr: &ir::Expr) -> bool {
    matches!(
        expr,
        ir::Expr::BuiltIn(_)
            | ir::Expr::Param(_)
            | ir::Expr::Call { .. }
            | ir::Expr::Compose { .. }
            | ir::Expr::Math { .. }
    )
}

/// Collects the built-ins a body reads, in the order they are first seen.
fn used_built_ins(body: &[ir::Stmt]) -> Vec<ir::BuiltIn> {
    let mut found = Vec::new();
    for stmt in body {
        walk_stmt(stmt, &mut |expr| {
            if let ir::Expr::BuiltIn(built_in) = expr
                && !found.contains(built_in)
            {
                found.push(*built_in);
            }
        });
    }
    found
}

fn walk_stmt(stmt: &ir::Stmt, visit: &mut impl FnMut(&ir::Expr)) {
    match stmt {
        ir::Stmt::Declare { value, .. } => {
            if let Some(value) = value {
                walk_expr(value, visit);
            }
        }
        ir::Stmt::Store { place, value } => {
            walk_expr(place, visit);
            walk_expr(value, visit);
        }
        ir::Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            walk_expr(condition, visit);
            for stmt in then_branch.iter().chain(else_branch) {
                walk_stmt(stmt, visit);
            }
        }
        ir::Stmt::While {
            condition,
            body,
            continuing,
        } => {
            walk_expr(condition, visit);
            for stmt in body.iter().chain(continuing) {
                walk_stmt(stmt, visit);
            }
        }
        ir::Stmt::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        ir::Stmt::Return { value } => {
            if let Some(value) = value {
                walk_expr(value, visit);
            }
        }
        ir::Stmt::Break | ir::Stmt::Continue | ir::Stmt::Barrier(_) => {}
    }
}

fn walk_expr(expr: &ir::Expr, visit: &mut impl FnMut(&ir::Expr)) {
    visit(expr);
    match expr {
        ir::Expr::Index { base, index } => {
            walk_expr(base, visit);
            walk_expr(index, visit);
        }
        ir::Expr::Component { base, .. } => walk_expr(base, visit),
        ir::Expr::Unary { value, .. } => walk_expr(value, visit),
        ir::Expr::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, visit);
            walk_expr(rhs, visit);
        }
        ir::Expr::Cast { value, .. } => walk_expr(value, visit),
        ir::Expr::Math { args, .. } => {
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        ir::Expr::Compose { components, .. } => {
            for component in components {
                walk_expr(component, visit);
            }
        }
        ir::Expr::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        ir::Expr::Literal(_)
        | ir::Expr::Local(_)
        | ir::Expr::Param(_)
        | ir::Expr::Resource(_)
        | ir::Expr::BuiltIn(_)
        | ir::Expr::ArrayLength(_) => {}
    }
}

/// A literal, keyed so it can be looked up in a hash map. Floats are keyed by
/// their bits, which is exactly the identity we want for deduplication.
#[derive(PartialEq, Eq, Hash)]
enum LiteralKey {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(u32),
}

impl From<ir::Literal> for LiteralKey {
    fn from(literal: ir::Literal) -> Self {
        match literal {
            ir::Literal::Bool(value) => Self::Bool(value),
            ir::Literal::I32(value) => Self::I32(value),
            ir::Literal::U32(value) => Self::U32(value),
            ir::Literal::F32(value) => Self::F32(value.to_bits()),
        }
    }
}

struct Context<'a> {
    kernel: &'a ir::Kernel,
    types: &'a mut Types,
    module_types: &'a mut UniqueArena<naga::Type>,
    globals: &'a [Handle<naga::GlobalVariable>],
    functions: &'a [Handle<naga::Function>],
    /// The locals of the function being lowered, in declaration order.
    source_locals: &'a [ir::Local],
    /// One pointer expression per local, in the same order.
    locals: Vec<Handle<Expression>>,
    /// One expression per parameter, for a helper. Empty for the entry point,
    /// which is handed built-ins instead.
    params: Vec<Handle<Expression>>,
    built_in_exprs: HashMap<ir::BuiltIn, Handle<Expression>>,
    global_exprs: HashMap<usize, Handle<Expression>>,
    literals: HashMap<LiteralKey, Handle<Expression>>,
    expressions: Arena<Expression>,
    local_variables: Arena<naga::LocalVariable>,
}

impl Context<'_> {
    /// Creates every expression that naga requires to sit outside an emit
    /// range, before the first range opens.
    fn prepare_pre_emit(&mut self, arguments: &Arguments<'_>, body: &[ir::Stmt]) -> Result<()> {
        match arguments {
            Arguments::BuiltIns(built_ins) => {
                for (index, built_in) in built_ins.iter().enumerate() {
                    let handle = self
                        .expressions
                        .append(Expression::FunctionArgument(index as u32), SPAN);
                    self.built_in_exprs.insert(*built_in, handle);
                }
            }
            Arguments::Params(params) => {
                for index in 0..params.len() {
                    let handle = self
                        .expressions
                        .append(Expression::FunctionArgument(index as u32), SPAN);
                    self.params.push(handle);
                }
            }
        }

        for index in 0..self.globals.len() {
            let global = self.globals[index];
            let handle = self
                .expressions
                .append(Expression::GlobalVariable(global), SPAN);
            self.global_exprs.insert(index, handle);
        }

        for local in self.source_locals {
            let ty = self.types.lower(self.module_types, &local.ty)?;
            let variable = self.local_variables.append(
                naga::LocalVariable {
                    name: Some(local.name.clone()),
                    ty,
                    init: None,
                },
                SPAN,
            );
            let handle = self
                .expressions
                .append(Expression::LocalVariable(variable), SPAN);
            self.locals.push(handle);
        }

        let mut literals = Vec::new();
        for stmt in body {
            walk_stmt(stmt, &mut |expr| {
                if let ir::Expr::Literal(literal) = expr {
                    literals.push(*literal);
                }
            });
        }
        // `while` conditions are negated when lowered, which needs a `true`
        // literal in the rare case the condition is itself a literal. Adding it
        // unconditionally is cheaper than reasoning about when it is needed,
        // and naga drops unused expressions when writing.
        for literal in literals {
            self.intern_literal(literal);
        }
        Ok(())
    }

    fn intern_literal(&mut self, literal: ir::Literal) -> Handle<Expression> {
        let key = LiteralKey::from(literal);
        if let Some(handle) = self.literals.get(&key) {
            return *handle;
        }
        let value = match literal {
            ir::Literal::Bool(value) => naga::Literal::Bool(value),
            ir::Literal::I32(value) => naga::Literal::I32(value),
            ir::Literal::U32(value) => naga::Literal::U32(value),
            ir::Literal::F32(value) => naga::Literal::F32(value),
        };
        let handle = self.expressions.append(Expression::Literal(value), SPAN);
        self.literals.insert(key, handle);
        handle
    }

    fn lower_block(&mut self, stmts: &[ir::Stmt]) -> Result<Block> {
        let mut block = Block::new();
        for stmt in stmts {
            self.lower_stmt(stmt, &mut block)?;
        }
        Ok(block)
    }

    fn lower_stmt(&mut self, stmt: &ir::Stmt, block: &mut Block) -> Result<()> {
        match stmt {
            ir::Stmt::Declare { local, value } => {
                let Some(value) = value else {
                    // A declaration with no initialiser needs no code. Naga
                    // zero initialises function locals.
                    return Ok(());
                };
                let value = self.lower_value(block, value)?;
                let pointer = self.local_pointer(*local)?;
                block.push(Statement::Store { pointer, value }, SPAN);
            }
            ir::Stmt::Store { place, value } => {
                let mut emitter = naga::proc::Emitter::default();
                emitter.start(&self.expressions);
                let pointer = self.lower_place(block, &mut emitter, place)?;
                let value = self.lower_expr(block, &mut emitter, value)?;
                block.extend(emitter.finish(&self.expressions));
                block.push(Statement::Store { pointer, value }, SPAN);
            }
            ir::Stmt::Call { function, args } => {
                let mut emitter = naga::proc::Emitter::default();
                emitter.start(&self.expressions);
                let mut arguments = Vec::with_capacity(args.len());
                for arg in args {
                    arguments.push(self.lower_expr(block, &mut emitter, arg)?);
                }
                block.extend(emitter.finish(&self.expressions));
                // A callee that returns a value is still allowed here. Naga
                // wants the result bound either way, and dropping it on the
                // floor is what the caller asked for.
                self.push_call(block, *function, arguments)?;
            }
            ir::Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = self.lower_value(block, condition)?;
                let accept = self.lower_block(then_branch)?;
                let reject = self.lower_block(else_branch)?;
                block.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    SPAN,
                );
            }
            ir::Stmt::While {
                condition,
                body,
                continuing,
            } => {
                // Naga only has `loop`, so the condition becomes an early break
                // at the top of the body.
                let mut loop_body = Block::new();
                let mut emitter = naga::proc::Emitter::default();
                emitter.start(&self.expressions);
                let condition = self.lower_expr(&mut loop_body, &mut emitter, condition)?;
                let keep_going = self.expressions.append(
                    Expression::Unary {
                        op: naga::UnaryOperator::LogicalNot,
                        expr: condition,
                    },
                    SPAN,
                );
                loop_body.extend(emitter.finish(&self.expressions));
                loop_body.push(
                    Statement::If {
                        condition: keep_going,
                        accept: Block::from_vec(vec![Statement::Break]),
                        reject: Block::new(),
                    },
                    SPAN,
                );
                let rest = self.lower_block(body)?;
                loop_body.extend_block(rest);

                // The continuing block also runs when the body hits
                // `continue`, which is how a `for` loop's increment stays
                // reachable. Naga forbids control flow in there, so reject it
                // here with a message that says which statement is at fault
                // rather than letting the validator produce one.
                for stmt in continuing {
                    if !allowed_in_continuing(stmt) {
                        return Err(Error::Invalid(format!(
                            "{stmt:?} is not allowed in a loop's continuing block"
                        )));
                    }
                }
                let continuing = self.lower_block(continuing)?;

                block.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing,
                        break_if: None,
                    },
                    SPAN,
                );
            }
            ir::Stmt::Break => block.push(Statement::Break, SPAN),
            ir::Stmt::Continue => block.push(Statement::Continue, SPAN),
            ir::Stmt::Return { value } => {
                let value = match value {
                    Some(value) => Some(self.lower_value(block, value)?),
                    None => None,
                };
                block.push(Statement::Return { value }, SPAN);
            }
            ir::Stmt::Barrier(scope) => {
                let barrier = match scope {
                    ir::BarrierScope::Workgroup => naga::Barrier::WORK_GROUP,
                    ir::BarrierScope::Storage => naga::Barrier::STORAGE,
                };
                block.push(Statement::ControlBarrier(barrier), SPAN);
            }
        }
        Ok(())
    }

    /// Lowers an expression that stands on its own, in an emit range of its
    /// own. Use this where there is a single expression to read; where several
    /// belong in one range, drive the emitter directly.
    fn lower_value(&mut self, block: &mut Block, expr: &ir::Expr) -> Result<Handle<Expression>> {
        let mut emitter = naga::proc::Emitter::default();
        emitter.start(&self.expressions);
        let handle = self.lower_expr(block, &mut emitter, expr)?;
        block.extend(emitter.finish(&self.expressions));
        Ok(handle)
    }

    /// Pushes a call statement, handing back the expression holding its result
    /// when the callee produces one.
    ///
    /// The caller must have closed the current emit range first, because
    /// [`Expression::CallResult`] is one of the expressions naga requires to
    /// sit outside one.
    fn push_call(
        &mut self,
        block: &mut Block,
        function: ir::FunctionId,
        arguments: Vec<Handle<Expression>>,
    ) -> Result<Option<Handle<Expression>>> {
        let kernel = self.kernel;
        let callee = kernel
            .functions
            .get(function.0 as usize)
            .ok_or_else(|| Error::Invalid(format!("function {} is out of range", function.0)))?;
        if arguments.len() != callee.params.len() {
            return Err(Error::Invalid(format!(
                "`{}` takes {} arguments but got {}",
                callee.name,
                callee.params.len(),
                arguments.len()
            )));
        }
        let handle = *self.functions.get(function.0 as usize).ok_or_else(|| {
            Error::Invalid(format!(
                "`{}` is called before it is defined, order `Kernel::functions` with callees first",
                callee.name
            ))
        })?;
        let result = callee.result.is_some().then(|| {
            self.expressions
                .append(Expression::CallResult(handle), SPAN)
        });
        block.push(
            Statement::Call {
                function: handle,
                arguments,
                result,
            },
            SPAN,
        );
        Ok(result)
    }

    /// Lowers an expression used as the destination of a store, producing a
    /// pointer rather than a value.
    fn lower_place(
        &mut self,
        block: &mut Block,
        emitter: &mut naga::proc::Emitter,
        expr: &ir::Expr,
    ) -> Result<Handle<Expression>> {
        match expr {
            ir::Expr::Local(local) => self.local_pointer(*local),
            ir::Expr::Index { base, index } => {
                let base = self.lower_place(block, emitter, base)?;
                let index = self.lower_expr(block, emitter, index)?;
                Ok(self
                    .expressions
                    .append(Expression::Access { base, index }, SPAN))
            }
            ir::Expr::Component { base, index } => {
                let base = self.lower_place(block, emitter, base)?;
                Ok(self.expressions.append(
                    Expression::AccessIndex {
                        base,
                        index: u32::from(*index),
                    },
                    SPAN,
                ))
            }
            ir::Expr::Resource(resource) => self.global_pointer(*resource),
            ir::Expr::Param(param) => Err(Error::Invalid(format!(
                "parameter {} cannot be assigned to",
                param.0
            ))),
            other => Err(Error::Invalid(format!("{other:?} cannot be assigned to"))),
        }
    }

    fn local_pointer(&mut self, local: ir::LocalId) -> Result<Handle<Expression>> {
        self.locals
            .get(local.0 as usize)
            .copied()
            .ok_or_else(|| Error::Invalid(format!("local {} is out of range", local.0)))
    }

    fn global_pointer(&mut self, resource: ir::ResourceId) -> Result<Handle<Expression>> {
        self.global_exprs
            .get(&(resource.0 as usize))
            .copied()
            .ok_or_else(|| Error::Invalid(format!("resource {} is out of range", resource.0)))
    }

    fn lower_expr(
        &mut self,
        block: &mut Block,
        emitter: &mut naga::proc::Emitter,
        expr: &ir::Expr,
    ) -> Result<Handle<Expression>> {
        match expr {
            ir::Expr::Literal(literal) => Ok(self
                .literals
                .get(&LiteralKey::from(*literal))
                .copied()
                .expect("every literal in the body was interned up front")),
            ir::Expr::Local(local) => {
                let pointer = self.local_pointer(*local)?;
                Ok(self.expressions.append(Expression::Load { pointer }, SPAN))
            }
            ir::Expr::Param(param) => self
                .params
                .get(param.0 as usize)
                .copied()
                .ok_or_else(|| Error::Invalid(format!("parameter {} is out of range", param.0))),
            ir::Expr::Call { function, args } => {
                let mut arguments = Vec::with_capacity(args.len());
                for arg in args {
                    arguments.push(self.lower_expr(block, emitter, arg)?);
                }
                // Naga has no call expression: a call is a statement that
                // binds its result. Close the range so the result lands
                // outside it, push the call, then open a new range for
                // whatever reads the result.
                block.extend(emitter.finish(&self.expressions));
                let result = self.push_call(block, *function, arguments)?;
                emitter.start(&self.expressions);
                result.ok_or_else(|| {
                    Error::Invalid(format!(
                        "`{}` returns nothing, so a call to it has no value",
                        self.kernel.function(*function).name
                    ))
                })
            }
            ir::Expr::BuiltIn(built_in) => self
                .built_in_exprs
                .get(built_in)
                .copied()
                .ok_or_else(|| Error::Invalid(format!("unknown built-in {built_in:?}"))),
            ir::Expr::Resource(resource) => {
                // A whole array has no value, but a uniform scalar or vector
                // reads straight through its pointer.
                if matches!(self.kernel.resource(*resource).ty, ir::Type::Array { .. }) {
                    return Err(Error::Invalid(format!(
                        "resource `{}` is a buffer, index it before using its value",
                        self.kernel.resource(*resource).name
                    )));
                }
                let pointer = self.global_pointer(*resource)?;
                Ok(self.expressions.append(Expression::Load { pointer }, SPAN))
            }
            // Reading through an index is normally a pointer followed by a
            // load. Some bases are plain values rather than memory, though,
            // and those are read with an access on the value itself.
            ir::Expr::Component { base, index } if names_no_memory(base) => {
                let base = self.lower_expr(block, emitter, base)?;
                Ok(self.expressions.append(
                    Expression::AccessIndex {
                        base,
                        index: u32::from(*index),
                    },
                    SPAN,
                ))
            }
            ir::Expr::Index { base, index } if names_no_memory(base) => {
                let base = self.lower_expr(block, emitter, base)?;
                let index = self.lower_expr(block, emitter, index)?;
                Ok(self
                    .expressions
                    .append(Expression::Access { base, index }, SPAN))
            }
            ir::Expr::Index { .. } | ir::Expr::Component { .. } => {
                let pointer = self.lower_place(block, emitter, expr)?;
                Ok(self.expressions.append(Expression::Load { pointer }, SPAN))
            }
            ir::Expr::Unary { op, value } => {
                let expr = self.lower_expr(block, emitter, value)?;
                let op = match op {
                    ir::UnaryOp::Negate => naga::UnaryOperator::Negate,
                    ir::UnaryOp::Not => naga::UnaryOperator::LogicalNot,
                };
                Ok(self
                    .expressions
                    .append(Expression::Unary { op, expr }, SPAN))
            }
            ir::Expr::Binary { op, lhs, rhs } => {
                let left = self.lower_expr(block, emitter, lhs)?;
                let right = self.lower_expr(block, emitter, rhs)?;
                Ok(self.expressions.append(
                    Expression::Binary {
                        op: naga_binary_op(*op),
                        left,
                        right,
                    },
                    SPAN,
                ))
            }
            ir::Expr::Cast { value, to } => {
                let expr = self.lower_expr(block, emitter, value)?;
                let scalar = naga_scalar(*to);
                Ok(self.expressions.append(
                    Expression::As {
                        expr,
                        kind: scalar.kind,
                        convert: Some(scalar.width),
                    },
                    SPAN,
                ))
            }
            ir::Expr::Math { function, args } => self.lower_math(block, emitter, *function, args),
            ir::Expr::Compose {
                size,
                scalar,
                components,
            } => {
                if components.len() != usize::from(size.count()) {
                    return Err(Error::Invalid(format!(
                        "vec{} needs {} components but got {}",
                        size.count(),
                        size.count(),
                        components.len()
                    )));
                }
                let ty = self
                    .types
                    .lower(self.module_types, &ir::Type::vector(*size, *scalar))?;
                let mut lowered = Vec::with_capacity(components.len());
                for component in components {
                    lowered.push(self.lower_expr(block, emitter, component)?);
                }
                Ok(self.expressions.append(
                    Expression::Compose {
                        ty,
                        components: lowered,
                    },
                    SPAN,
                ))
            }
            ir::Expr::ArrayLength(resource) => {
                let global = self.global_pointer(*resource)?;
                Ok(self
                    .expressions
                    .append(Expression::ArrayLength(global), SPAN))
            }
        }
    }

    fn lower_math(
        &mut self,
        block: &mut Block,
        emitter: &mut naga::proc::Emitter,
        function: ir::MathFn,
        args: &[ir::Expr],
    ) -> Result<Handle<Expression>> {
        if args.len() != function.arity() {
            return Err(Error::Invalid(format!(
                "{} takes {} arguments but got {}",
                function.intrinsic_name(),
                function.arity(),
                args.len()
            )));
        }
        let mut lowered = Vec::with_capacity(args.len());
        for arg in args {
            lowered.push(self.lower_expr(block, emitter, arg)?);
        }
        let mut lowered = lowered.into_iter();
        let arg = lowered.next().expect("arity is at least one");
        Ok(self.expressions.append(
            Expression::Math {
                fun: naga_math_fn(function),
                arg,
                arg1: lowered.next(),
                arg2: lowered.next(),
                arg3: lowered.next(),
            },
            SPAN,
        ))
    }
}

fn naga_binary_op(op: ir::BinaryOp) -> naga::BinaryOperator {
    use naga::BinaryOperator as B;
    match op {
        ir::BinaryOp::Add => B::Add,
        ir::BinaryOp::Subtract => B::Subtract,
        ir::BinaryOp::Multiply => B::Multiply,
        ir::BinaryOp::Divide => B::Divide,
        ir::BinaryOp::Modulo => B::Modulo,
        ir::BinaryOp::Equal => B::Equal,
        ir::BinaryOp::NotEqual => B::NotEqual,
        ir::BinaryOp::Less => B::Less,
        ir::BinaryOp::LessEqual => B::LessEqual,
        ir::BinaryOp::Greater => B::Greater,
        ir::BinaryOp::GreaterEqual => B::GreaterEqual,
        ir::BinaryOp::And => B::And,
        ir::BinaryOp::Or => B::InclusiveOr,
        ir::BinaryOp::Xor => B::ExclusiveOr,
        ir::BinaryOp::LogicalAnd => B::LogicalAnd,
        ir::BinaryOp::LogicalOr => B::LogicalOr,
        ir::BinaryOp::ShiftLeft => B::ShiftLeft,
        ir::BinaryOp::ShiftRight => B::ShiftRight,
    }
}

fn naga_math_fn(function: ir::MathFn) -> naga::MathFunction {
    use naga::MathFunction as M;
    match function {
        ir::MathFn::Abs => M::Abs,
        ir::MathFn::Min => M::Min,
        ir::MathFn::Max => M::Max,
        ir::MathFn::Clamp => M::Clamp,
        ir::MathFn::Floor => M::Floor,
        ir::MathFn::Ceil => M::Ceil,
        ir::MathFn::Round => M::Round,
        ir::MathFn::Sqrt => M::Sqrt,
        ir::MathFn::InverseSqrt => M::InverseSqrt,
        ir::MathFn::Exp => M::Exp,
        ir::MathFn::Log => M::Log,
        ir::MathFn::Pow => M::Pow,
        ir::MathFn::Sin => M::Sin,
        ir::MathFn::Cos => M::Cos,
        ir::MathFn::Tan => M::Tan,
        ir::MathFn::Sign => M::Sign,
        ir::MathFn::Fma => M::Fma,
        ir::MathFn::Mix => M::Mix,
        ir::MathFn::Step => M::Step,
        ir::MathFn::Dot => M::Dot,
        ir::MathFn::Cross => M::Cross,
        ir::MathFn::Length => M::Length,
        ir::MathFn::Normalize => M::Normalize,
    }
}
