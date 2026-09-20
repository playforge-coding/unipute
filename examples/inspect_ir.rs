//! Reading a kernel back as data.
//!
//! `#[kernel]` writes shaders, but it also hands the kernel back as IR, and the
//! IR is plain Rust data with no dependencies on anything. This example walks
//! it: printing the body back out and counting what is in it. The same walk is
//! how a new back end, a linter or an editor tooltip would start.
//!
//! None of this needs a target feature, since nothing here generates a shader.
//!
//! ```text
//! cargo run --example inspect_ir
//! ```

use unipute::ir::{self, Expr, Stmt};
use unipute::{Kernel, kernel};

/// Colours one pixel by how long the Mandelbrot iteration takes to escape.
///
/// Picked for this example because it has something of everything: a two
/// dimensional guard, casts between integers and floats, a vector, a `while`
/// loop with a `break` in it, and a helper function.
#[kernel(workgroup_size(8, 8))]
fn mandelbrot(output: &mut [f32], width: &u32, height: &u32, iteration_limit: &u32) {
    /// Maps a pixel coordinate onto the part of the plane worth looking at.
    fn plane(coordinate: u32, size: u32, span: f32, offset: f32) -> f32 {
        coordinate as f32 / size as f32 * span + offset
    }

    let x = global_id().x;
    let y = global_id().y;
    if x >= width || y >= height {
        return;
    }

    let point = vec2(plane(x, width, 3.5, -2.5), plane(y, height, 2.0, -1.0));

    let mut zx = 0.0f32;
    let mut zy = 0.0f32;
    let mut iteration = 0u32;
    while iteration < iteration_limit {
        // Once the orbit leaves a circle of radius two it never comes back,
        // so there is nothing left to learn from iterating further. `dot` of a
        // vector with itself is its length squared, which saves the square
        // root that comparing lengths would need.
        let z = vec2(zx, zy);
        if dot(z, z) > 4.0 {
            break;
        }
        let next_zx = zx * zx - zy * zy + point.x;
        zy = 2.0 * zx * zy + point.y;
        zx = next_zx;
        iteration += 1u32;
    }

    output[y * width + x] = iteration as f32 / iteration_limit as f32;
}

fn main() {
    let ir = mandelbrot::ir();

    println!("kernel `{}`, {} stage", ir.name, ir.stage.name());
    println!(
        "  workgroup {:?}, {} invocations a group",
        ir.workgroup_size,
        ir.invocations_per_workgroup()
    );
    for resource in &ir.resources {
        println!(
            "  @{}:{} {}: {} ({:?})",
            resource.group, resource.binding, resource.name, resource.ty, resource.access
        );
    }
    println!();

    for function in &ir.functions {
        let params: Vec<String> = function
            .params
            .iter()
            .map(|param| format!("{}: {}", param.name, param.ty))
            .collect();
        let result = match &function.result {
            Some(ty) => format!(" -> {ty}"),
            None => String::new(),
        };
        println!("fn {}({}){result}", function.name, params.join(", "));
        let printer = Printer {
            kernel: &ir,
            locals: &function.locals,
            params: &function.params,
        };
        printer.block(&function.body, 1);
        println!();
    }

    println!("fn {}()", ir.name);
    let printer = Printer {
        kernel: &ir,
        locals: &ir.locals,
        params: &[],
    };
    printer.block(&ir.body, 1);
    println!();

    let mut stats = Stats::default();
    stats.block(&ir.body, 0);
    stats.report();
}

/// Prints a body back out in something close to the source it came from.
struct Printer<'a> {
    kernel: &'a ir::Kernel,
    /// The locals of the function being printed, which is what a [`LocalId`]
    /// indexes into.
    ///
    /// [`LocalId`]: ir::LocalId
    locals: &'a [ir::Local],
    /// Its parameters, empty for the entry point.
    params: &'a [ir::Param],
}

impl Printer<'_> {
    fn block(&self, stmts: &[Stmt], depth: usize) {
        for stmt in stmts {
            self.stmt(stmt, depth);
        }
    }

    fn stmt(&self, stmt: &Stmt, depth: usize) {
        let pad = "    ".repeat(depth);
        match stmt {
            Stmt::Declare { local, value } => {
                let local = &self.locals[local.0 as usize];
                match value {
                    Some(value) => println!(
                        "{pad}let mut {}: {} = {}",
                        local.name,
                        local.ty,
                        self.expr(value)
                    ),
                    None => println!("{pad}let mut {}: {}", local.name, local.ty),
                }
            }
            Stmt::Store { place, value } => {
                println!("{pad}{} = {}", self.expr(place), self.expr(value));
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                println!("{pad}if {} {{", self.expr(condition));
                self.block(then_branch, depth + 1);
                if !else_branch.is_empty() {
                    println!("{pad}}} else {{");
                    self.block(else_branch, depth + 1);
                }
                println!("{pad}}}");
            }
            Stmt::While {
                condition,
                body,
                continuing,
            } => {
                println!("{pad}while {} {{", self.expr(condition));
                self.block(body, depth + 1);
                if !continuing.is_empty() {
                    // Where a `for` loop's counter increment ends up, so that
                    // `continue` cannot skip past it.
                    println!("{pad}}} continuing {{");
                    self.block(continuing, depth + 1);
                }
                println!("{pad}}}");
            }
            Stmt::Call { function, args } => {
                println!("{pad}{}", self.call(*function, args));
            }
            Stmt::Break => println!("{pad}break"),
            Stmt::Continue => println!("{pad}continue"),
            Stmt::Return { value: Some(value) } => {
                println!("{pad}return {}", self.expr(value));
            }
            Stmt::Return { value: None } => println!("{pad}return"),
            Stmt::Barrier(scope) => match scope {
                ir::BarrierScope::Workgroup => println!("{pad}workgroup_barrier()"),
                ir::BarrierScope::Storage => println!("{pad}storage_barrier()"),
            },
        }
    }

    fn expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Literal(literal) => literal_text(*literal),
            Expr::Local(local) => self.locals[local.0 as usize].name.clone(),
            Expr::Param(param) => self.params[param.0 as usize].name.clone(),
            Expr::Resource(resource) => self.kernel.resource(*resource).name.clone(),
            Expr::BuiltIn(built_in) => format!("{}()", built_in.intrinsic_name()),
            Expr::Index { base, index } => {
                format!("{}[{}]", self.expr(base), self.expr(index))
            }
            Expr::Component { base, index } => {
                let name = ["x", "y", "z", "w"][*index as usize];
                format!("{}.{name}", self.expr(base))
            }
            Expr::Unary { op, value } => {
                let symbol = match op {
                    ir::UnaryOp::Negate => "-",
                    ir::UnaryOp::Not => "!",
                };
                format!("{symbol}{}", self.expr(value))
            }
            Expr::Binary { op, lhs, rhs } => {
                format!("({} {} {})", self.expr(lhs), operator(*op), self.expr(rhs))
            }
            Expr::Cast { value, to } => format!("{} as {to}", self.expr(value)),
            Expr::Math { function, args } => {
                format!("{}({})", function.intrinsic_name(), self.args(args))
            }
            Expr::Compose {
                size, components, ..
            } => {
                format!("vec{}({})", size.count(), self.args(components))
            }
            Expr::Call { function, args } => self.call(*function, args),
            Expr::ArrayLength(resource) => {
                format!("{}.len()", self.kernel.resource(*resource).name)
            }
        }
    }

    fn call(&self, function: ir::FunctionId, args: &[Expr]) -> String {
        format!(
            "{}({})",
            self.kernel.function(function).name,
            self.args(args)
        )
    }

    fn args(&self, args: &[Expr]) -> String {
        args.iter()
            .map(|arg| self.expr(arg))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

fn literal_text(literal: ir::Literal) -> String {
    match literal {
        ir::Literal::Bool(value) => value.to_string(),
        ir::Literal::I32(value) => value.to_string(),
        ir::Literal::U32(value) => format!("{value}u32"),
        // `{}` would print a whole number as `2`, which reads as an integer.
        ir::Literal::F32(value) => format!("{value:?}"),
    }
}

fn operator(op: ir::BinaryOp) -> &'static str {
    match op {
        ir::BinaryOp::Add => "+",
        ir::BinaryOp::Subtract => "-",
        ir::BinaryOp::Multiply => "*",
        ir::BinaryOp::Divide => "/",
        ir::BinaryOp::Modulo => "%",
        ir::BinaryOp::Equal => "==",
        ir::BinaryOp::NotEqual => "!=",
        ir::BinaryOp::Less => "<",
        ir::BinaryOp::LessEqual => "<=",
        ir::BinaryOp::Greater => ">",
        ir::BinaryOp::GreaterEqual => ">=",
        ir::BinaryOp::And => "&",
        ir::BinaryOp::Or => "|",
        ir::BinaryOp::Xor => "^",
        ir::BinaryOp::LogicalAnd => "&&",
        ir::BinaryOp::LogicalOr => "||",
        ir::BinaryOp::ShiftLeft => "<<",
        ir::BinaryOp::ShiftRight => ">>",
    }
}

/// What a second walk of the same body can tell you about it.
#[derive(Default)]
struct Stats {
    statements: usize,
    branches: usize,
    loops: usize,
    deepest_loop: usize,
    built_ins: Vec<ir::BuiltIn>,
    math: Vec<ir::MathFn>,
}

impl Stats {
    fn block(&mut self, stmts: &[Stmt], loop_depth: usize) {
        for stmt in stmts {
            self.statements += 1;
            match stmt {
                Stmt::Declare { value, .. } => {
                    if let Some(value) = value {
                        self.expr(value);
                    }
                }
                Stmt::Store { place, value } => {
                    self.expr(place);
                    self.expr(value);
                }
                Stmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    self.branches += 1;
                    self.expr(condition);
                    self.block(then_branch, loop_depth);
                    self.block(else_branch, loop_depth);
                }
                Stmt::While {
                    condition,
                    body,
                    continuing,
                } => {
                    self.loops += 1;
                    self.deepest_loop = self.deepest_loop.max(loop_depth + 1);
                    self.expr(condition);
                    self.block(body, loop_depth + 1);
                    self.block(continuing, loop_depth + 1);
                }
                Stmt::Call { args, .. } => self.exprs(args),
                Stmt::Return { value: Some(value) } => self.expr(value),
                Stmt::Return { value: None } | Stmt::Break | Stmt::Continue | Stmt::Barrier(_) => {}
            }
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::BuiltIn(built_in) => {
                if !self.built_ins.contains(built_in) {
                    self.built_ins.push(*built_in);
                }
            }
            Expr::Math { function, args } => {
                if !self.math.contains(function) {
                    self.math.push(*function);
                }
                self.exprs(args);
            }
            Expr::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
            Expr::Component { base, .. } => self.expr(base),
            Expr::Unary { value, .. } => self.expr(value),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Cast { value, .. } => self.expr(value),
            Expr::Compose { components, .. } => self.exprs(components),
            Expr::Call { args, .. } => self.exprs(args),
            Expr::Literal(_)
            | Expr::Local(_)
            | Expr::Param(_)
            | Expr::Resource(_)
            | Expr::ArrayLength(_) => {}
        }
    }

    fn exprs(&mut self, exprs: &[Expr]) {
        for expr in exprs {
            self.expr(expr);
        }
    }

    fn report(&self) {
        println!("the entry point body holds");
        println!("  statements: {}", self.statements);
        println!("  branches: {}", self.branches);
        println!("  loops: {}, nested {} deep", self.loops, self.deepest_loop);
        println!(
            "  built-ins: {}",
            names(&self.built_ins, ir::BuiltIn::intrinsic_name)
        );
        println!("  maths: {}", names(&self.math, ir::MathFn::intrinsic_name));
    }
}

/// Joins a list of things that have an intrinsic name, in the order they were
/// first seen.
fn names<T: Copy>(items: &[T], name: impl Fn(T) -> &'static str) -> String {
    if items.is_empty() {
        return "none".to_owned();
    }
    items
        .iter()
        .map(|item| name(*item))
        .collect::<Vec<&str>>()
        .join(", ")
}
