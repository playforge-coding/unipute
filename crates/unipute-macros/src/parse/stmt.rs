//! Turning Rust statement syntax into Unipute IR statements.

use unipute_ir as ir;

use super::Scope;
use super::expr::{Typed, compound_op};
use super::types::{shared_type, value_type};

/// What the attributes on a `let` ask for.
#[derive(PartialEq)]
enum LetAttr {
    /// An ordinary local.
    None,
    /// `#[workgroup]`: memory shared by the whole workgroup rather than a
    /// local of one invocation.
    Workgroup,
}

/// Reads the attributes on a `let`, of which `#[workgroup]` is the only one
/// that means anything here.
fn let_attr(local: &syn::Local) -> syn::Result<LetAttr> {
    let mut found = LetAttr::None;
    for attr in &local.attrs {
        if !attr.path().is_ident("workgroup") {
            return Err(syn::Error::new_spanned(
                attr,
                "the only attribute allowed on a `let` inside a kernel is `#[workgroup]`",
            ));
        }
        if !matches!(attr.meta, syn::Meta::Path(_)) {
            return Err(syn::Error::new_spanned(
                attr,
                "`#[workgroup]` takes no arguments",
            ));
        }
        if found == LetAttr::Workgroup {
            return Err(syn::Error::new_spanned(
                attr,
                "`#[workgroup]` is written twice on this `let`",
            ));
        }
        found = LetAttr::Workgroup;
    }
    Ok(found)
}

impl Scope<'_> {
    /// Reads a braced block, in its own scope.
    pub fn block(&mut self, block: &syn::Block) -> syn::Result<Vec<ir::Stmt>> {
        self.push_frame();
        let result = self.statements(&block.stmts);
        self.pop_frame();
        result
    }

    /// Reads the kernel's own body.
    ///
    /// The nested `fn` items have already been read into helpers, so they are
    /// passed over here rather than reported as stray items. This is also the
    /// only place a `#[workgroup]` declaration is accepted: it is memory of
    /// the whole kernel rather than a local of one block, so it sits at the
    /// top level the way a `fn` item does.
    pub fn kernel_body(&mut self, block: &syn::Block) -> syn::Result<Vec<ir::Stmt>> {
        self.push_frame();
        let mut out = Vec::new();
        let result = (|| {
            for stmt in &block.stmts {
                if matches!(stmt, syn::Stmt::Item(syn::Item::Fn(_))) {
                    continue;
                }
                if let syn::Stmt::Local(local) = stmt
                    && let_attr(local)? == LetAttr::Workgroup
                {
                    self.workgroup_declaration(local)?;
                    continue;
                }
                self.stmt(stmt, &mut out)?;
            }
            Ok(())
        })();
        self.pop_frame();
        result.map(|()| out)
    }

    /// Reads `#[workgroup] let name: [T; N];` into workgroup memory.
    ///
    /// No statement comes out of it. The memory is declared for the whole
    /// kernel, not created when the line runs, so all that happens here is
    /// that the name goes into scope.
    fn workgroup_declaration(&mut self, local: &syn::Local) -> syn::Result<()> {
        let (name, annotation) = binding_name(&local.pat)?;
        let Some(annotation) = annotation else {
            return Err(syn::Error::new_spanned(
                &local.pat,
                format!("workgroup memory needs a type, write `let {name}: [f32; 64]`"),
            ));
        };
        if let Some(init) = &local.init {
            return Err(syn::Error::new_spanned(
                &init.expr,
                "workgroup memory cannot be given a value, since every invocation writes its \
                 own part of it",
            ));
        }
        let ty = shared_type(&annotation)?;
        if self.shared.iter().any(|shared| shared.name == name) {
            return Err(syn::Error::new_spanned(
                &local.pat,
                format!("`{name}` is already declared as workgroup memory"),
            ));
        }
        self.declare_shared(&name, ty);
        Ok(())
    }

    /// Reads a helper's body, where a trailing expression is the return value
    /// just as it is in Rust.
    pub fn function_body(
        &mut self,
        block: &syn::Block,
        result: Option<&ir::Type>,
    ) -> syn::Result<Vec<ir::Stmt>> {
        self.push_frame();
        let out = self.function_statements(&block.stmts, result);
        self.pop_frame();
        out
    }

    fn function_statements(
        &mut self,
        stmts: &[syn::Stmt],
        result: Option<&ir::Type>,
    ) -> syn::Result<Vec<ir::Stmt>> {
        let mut out = Vec::new();
        for (index, stmt) in stmts.iter().enumerate() {
            let is_last = index + 1 == stmts.len();
            if is_last && let syn::Stmt::Expr(expr, None) = stmt {
                self.tail_expr(expr, result, &mut out)?;
                return Ok(out);
            }
            self.stmt(stmt, &mut out)?;
        }
        Ok(out)
    }

    /// Handles the last expression of a helper's body, the one with no
    /// semicolon after it.
    fn tail_expr(
        &mut self,
        expr: &syn::Expr,
        result: Option<&ir::Type>,
        out: &mut Vec<ir::Stmt>,
    ) -> syn::Result<()> {
        let Some(ty) = result else {
            // No result, so this is an ordinary statement that happens to be
            // written without a semicolon.
            return self.expr_stmt(expr, out);
        };
        if is_block_shaped(expr) {
            return Err(syn::Error::new_spanned(
                expr,
                "a block cannot be used as a value inside a kernel, use `return` in each branch",
            ));
        }
        let value = self.expr(expr, ty.component_scalar())?;
        out.push(ir::Stmt::Return {
            value: Some(value.expr),
        });
        Ok(())
    }

    fn statements(&mut self, stmts: &[syn::Stmt]) -> syn::Result<Vec<ir::Stmt>> {
        let mut out = Vec::new();
        for stmt in stmts {
            self.stmt(stmt, &mut out)?;
        }
        Ok(out)
    }

    fn stmt(&mut self, stmt: &syn::Stmt, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        match stmt {
            syn::Stmt::Local(local) => self.let_binding(local, out),
            syn::Stmt::Expr(expr, _) => self.expr_stmt(expr, out),
            syn::Stmt::Item(syn::Item::Fn(item)) => Err(syn::Error::new_spanned(
                &item.sig.ident,
                "a nested `fn` has to be declared at the top level of the kernel body",
            )),
            syn::Stmt::Item(item) => Err(syn::Error::new_spanned(
                item,
                "only `fn` items can be declared inside a kernel",
            )),
            syn::Stmt::Macro(mac) => Err(syn::Error::new_spanned(
                mac,
                "macros are not expanded inside a kernel",
            )),
        }
    }

    fn let_binding(&mut self, local: &syn::Local, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        if let_attr(local)? == LetAttr::Workgroup {
            // The kernel body handles these itself before they get here, so
            // reaching this point means the declaration is somewhere it
            // cannot go.
            return Err(syn::Error::new_spanned(
                &local.pat,
                match self.helper {
                    Some(helper) => format!(
                        "workgroup memory belongs to the kernel body, so `{}` cannot declare it",
                        helper.name
                    ),
                    None => "workgroup memory is declared at the top level of the kernel body, \
                             not inside a block"
                        .to_owned(),
                },
            ));
        }
        let (name, annotation) = binding_name(&local.pat)?;
        let Some(init) = &local.init else {
            return Err(syn::Error::new_spanned(
                local,
                "a `let` inside a kernel needs a value",
            ));
        };
        if let Some((diverge, _)) = &init.diverge {
            return Err(syn::Error::new_spanned(
                diverge,
                "`let ... else` is not supported inside a kernel",
            ));
        }

        let annotated = annotation.as_ref().map(value_type).transpose()?;
        let expected = annotated.as_ref().and_then(ir::Type::component_scalar);
        let value = self.expr(&init.expr, expected)?;

        let ty = match annotated {
            Some(ty) => ty,
            None => value.ty.clone().ok_or_else(|| {
                syn::Error::new_spanned(
                    &init.expr,
                    "the type of this value is unclear, annotate it as in `let x: f32 = ...`",
                )
            })?,
        };

        let id = self.declare_local(&name, ty);
        out.push(ir::Stmt::Declare {
            local: id,
            value: Some(value.expr),
        });
        Ok(())
    }

    fn expr_stmt(&mut self, expr: &syn::Expr, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        match expr {
            syn::Expr::Assign(assign) => self.assign(assign, out),
            syn::Expr::Binary(binary) if compound_op(binary.op).is_some() => {
                self.compound_assign(binary, out)
            }
            syn::Expr::If(branch) => self.if_branch(branch, out),
            syn::Expr::While(loop_) => self.while_loop(loop_, out),
            syn::Expr::Loop(loop_) => self.infinite_loop(loop_, out),
            syn::Expr::ForLoop(loop_) => self.for_loop(loop_, out),
            syn::Expr::Block(block) => {
                let inner = self.block(&block.block)?;
                out.extend(inner);
                Ok(())
            }
            syn::Expr::Return(ret) => self.return_stmt(ret, out),
            syn::Expr::Break(brk) => {
                if brk.label.is_some() || brk.expr.is_some() {
                    return Err(syn::Error::new_spanned(
                        brk,
                        "`break` inside a kernel cannot take a label or a value",
                    ));
                }
                self.check_in_loop(brk, "break")?;
                out.push(ir::Stmt::Break);
                Ok(())
            }
            syn::Expr::Continue(cont) => {
                if cont.label.is_some() {
                    return Err(syn::Error::new_spanned(
                        cont,
                        "`continue` inside a kernel cannot take a label",
                    ));
                }
                self.check_in_loop(cont, "continue")?;
                out.push(ir::Stmt::Continue);
                Ok(())
            }
            syn::Expr::Call(call) => self.call_stmt(call, out),
            other => Err(syn::Error::new_spanned(
                other,
                "this statement has no effect, a kernel statement must assign, branch, loop or call a built-in",
            )),
        }
    }

    fn return_stmt(&mut self, ret: &syn::ExprReturn, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        let expected = self.helper.and_then(|helper| helper.result.clone());
        match (&ret.expr, expected) {
            (Some(value), Some(ty)) => {
                let value = self.expr(value, ty.component_scalar())?;
                out.push(ir::Stmt::Return {
                    value: Some(value.expr),
                });
            }
            (Some(value), None) => {
                return Err(syn::Error::new_spanned(
                    value,
                    match self.helper {
                        Some(helper) => format!(
                            "`{}` returns nothing, so `return` cannot take a value",
                            helper.name
                        ),
                        None => "a kernel returns nothing, write results into a `&mut` parameter"
                            .to_owned(),
                    },
                ));
            }
            (None, Some(_)) => {
                let name = &self
                    .helper
                    .expect("only a helper can have a result type")
                    .name;
                return Err(syn::Error::new_spanned(
                    ret,
                    format!("`{name}` returns a value, so `return` needs one"),
                ));
            }
            (None, None) => out.push(ir::Stmt::Return { value: None }),
        }
        Ok(())
    }

    fn check_in_loop(&self, node: impl quote::ToTokens, keyword: &str) -> syn::Result<()> {
        if self.loop_depth == 0 {
            return Err(syn::Error::new_spanned(
                node,
                format!("`{keyword}` is only allowed inside a loop"),
            ));
        }
        Ok(())
    }

    /// Handles a call written as a statement: a helper called for its effects,
    /// or one of the intrinsics that are statements rather than values.
    fn call_stmt(&mut self, call: &syn::ExprCall, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        let name = match &*call.func {
            syn::Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
            _ => None,
        };

        if let Some(name) = &name
            && let Some(signature) = self.function_named(name)
        {
            let args = self.call_args(call, signature)?;
            out.push(ir::Stmt::Call {
                function: signature.id,
                args,
            });
            return Ok(());
        }

        let scope = match name.as_deref() {
            Some("workgroup_barrier") => ir::BarrierScope::Workgroup,
            Some("storage_barrier") => ir::BarrierScope::Storage,
            _ => {
                return Err(syn::Error::new_spanned(
                    call,
                    "this call produces a value that is discarded, assign it to something",
                ));
            }
        };
        if !call.args.is_empty() {
            return Err(syn::Error::new_spanned(
                &call.args,
                "a barrier takes no arguments",
            ));
        }
        out.push(ir::Stmt::Barrier(scope));
        Ok(())
    }

    fn assign(&mut self, assign: &syn::ExprAssign, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        let place = self.place(&assign.left)?;
        let expected = place.ty.as_ref().and_then(ir::Type::component_scalar);
        let value = self.expr(&assign.right, expected)?;
        out.push(ir::Stmt::Store {
            place: place.expr,
            value: value.expr,
        });
        Ok(())
    }

    fn compound_assign(
        &mut self,
        binary: &syn::ExprBinary,
        out: &mut Vec<ir::Stmt>,
    ) -> syn::Result<()> {
        let op = compound_op(binary.op).expect("checked by the caller");
        let place = self.place(&binary.left)?;
        // The destination is also the first operand, so read it as a value too.
        let current = self.expr(&binary.left, None)?;
        let expected = place.ty.as_ref().and_then(ir::Type::component_scalar);
        let operand = self.expr(&binary.right, expected)?;
        out.push(ir::Stmt::Store {
            place: place.expr,
            value: ir::Expr::Binary {
                op,
                lhs: Box::new(current.expr),
                rhs: Box::new(operand.expr),
            },
        });
        Ok(())
    }

    /// Reads an expression that is being assigned to, rejecting the ones that
    /// name no storage.
    fn place(&mut self, expr: &syn::Expr) -> syn::Result<Typed> {
        match expr {
            syn::Expr::Path(_) | syn::Expr::Index(_) | syn::Expr::Field(_) => {
                let typed = self.expr(expr, None)?;
                match &typed.expr {
                    ir::Expr::Local(_) | ir::Expr::Index { .. } | ir::Expr::Component { .. } => {
                        Ok(typed)
                    }
                    // A single shared value is assigned by name. A shared
                    // array is written one element at a time, like a buffer.
                    ir::Expr::Shared(id) => {
                        let shared = &self.shared[id.0 as usize];
                        if matches!(shared.ty, ir::Type::Array { .. }) {
                            Err(syn::Error::new_spanned(
                                expr,
                                format!(
                                    "`{}` is workgroup memory, assign to one element as in `{}[i] = ...`",
                                    shared.name, shared.name
                                ),
                            ))
                        } else {
                            Ok(typed)
                        }
                    }
                    ir::Expr::Resource(id) => {
                        let resource = &self.resources[id.0 as usize];
                        if matches!(resource.access, ir::Access::Uniform) {
                            Err(syn::Error::new_spanned(
                                expr,
                                format!(
                                    "`{}` is a uniform, it cannot be written to",
                                    resource.name
                                ),
                            ))
                        } else {
                            Err(syn::Error::new_spanned(
                                expr,
                                format!(
                                    "`{}` is a buffer, assign to one element as in `{}[i] = ...`",
                                    resource.name, resource.name
                                ),
                            ))
                        }
                    }
                    _ => Err(syn::Error::new_spanned(expr, "this cannot be assigned to")),
                }
            }
            other => Err(syn::Error::new_spanned(other, "this cannot be assigned to")),
        }
    }

    fn if_branch(&mut self, branch: &syn::ExprIf, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        if let syn::Expr::Let(let_expr) = &*branch.cond {
            return Err(syn::Error::new_spanned(
                let_expr,
                "`if let` is not supported inside a kernel",
            ));
        }
        let condition = self.expr(&branch.cond, Some(ir::Scalar::Bool))?;
        let then_branch = self.block(&branch.then_branch)?;
        let else_branch = match &branch.else_branch {
            Some((_, otherwise)) => {
                let mut stmts = Vec::new();
                self.expr_stmt(otherwise, &mut stmts)?;
                stmts
            }
            None => Vec::new(),
        };
        out.push(ir::Stmt::If {
            condition: condition.expr,
            then_branch,
            else_branch,
        });
        Ok(())
    }

    fn while_loop(&mut self, loop_: &syn::ExprWhile, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        if loop_.label.is_some() {
            return Err(syn::Error::new_spanned(
                loop_,
                "loop labels are not supported inside a kernel",
            ));
        }
        let condition = self.expr(&loop_.cond, Some(ir::Scalar::Bool))?;
        self.loop_depth += 1;
        let body = self.block(&loop_.body);
        self.loop_depth -= 1;
        out.push(ir::Stmt::While {
            condition: condition.expr,
            body: body?,
            continuing: Vec::new(),
        });
        Ok(())
    }

    fn infinite_loop(&mut self, loop_: &syn::ExprLoop, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        if loop_.label.is_some() {
            return Err(syn::Error::new_spanned(
                loop_,
                "loop labels are not supported inside a kernel",
            ));
        }
        self.loop_depth += 1;
        let body = self.block(&loop_.body);
        self.loop_depth -= 1;
        out.push(ir::Stmt::While {
            condition: ir::Expr::Literal(ir::Literal::Bool(true)),
            body: body?,
            continuing: Vec::new(),
        });
        Ok(())
    }

    /// Rewrites `for i in a..b { .. }` into a counted `while` loop.
    fn for_loop(&mut self, loop_: &syn::ExprForLoop, out: &mut Vec<ir::Stmt>) -> syn::Result<()> {
        if loop_.label.is_some() {
            return Err(syn::Error::new_spanned(
                loop_,
                "loop labels are not supported inside a kernel",
            ));
        }
        let syn::Expr::Range(range) = &*loop_.expr else {
            return Err(syn::Error::new_spanned(
                &loop_.expr,
                "a kernel `for` loop must iterate over a range such as `0..n`",
            ));
        };
        let (Some(start), Some(end)) = (&range.start, &range.end) else {
            return Err(syn::Error::new_spanned(
                range,
                "a kernel `for` loop needs both ends of the range",
            ));
        };
        let inclusive = matches!(range.limits, syn::RangeLimits::Closed(_));

        let (name, annotation) = binding_name(&loop_.pat)?;
        let annotated = annotation.as_ref().map(value_type).transpose()?;
        let expected = annotated
            .as_ref()
            .and_then(ir::Type::component_scalar)
            .or(Some(ir::Scalar::U32));

        let (start_value, end_value) = self.balanced_operands(start, end)?;
        let ty = annotated
            .or_else(|| start_value.ty.clone())
            .or_else(|| end_value.ty.clone())
            .unwrap_or(ir::Type::scalar(expected.unwrap_or(ir::Scalar::U32)));
        let scalar = ty
            .component_scalar()
            .ok_or_else(|| syn::Error::new_spanned(range, "a loop counter must be a scalar"))?;

        // The counter is scoped to the loop, like it is in Rust.
        self.push_frame();
        let counter = self.declare_local(&name, ty);
        out.push(ir::Stmt::Declare {
            local: counter,
            value: Some(start_value.expr),
        });

        self.loop_depth += 1;
        let body = self.block(&loop_.body);
        self.loop_depth -= 1;
        let body = match body {
            Ok(body) => body,
            Err(error) => {
                self.pop_frame();
                return Err(error);
            }
        };
        self.pop_frame();

        let one = match scalar {
            ir::Scalar::U32 => ir::Literal::U32(1),
            ir::Scalar::I32 => ir::Literal::I32(1),
            ir::Scalar::F32 => ir::Literal::F32(1.0),
            ir::Scalar::Bool => {
                return Err(syn::Error::new_spanned(
                    range,
                    "a loop counter cannot be a `bool`",
                ));
            }
        };
        // The increment goes in the continuing block rather than at the end
        // of the body, so that `continue` still reaches it.
        let increment = vec![ir::Stmt::Store {
            place: ir::Expr::Local(counter),
            value: ir::Expr::Binary {
                op: ir::BinaryOp::Add,
                lhs: Box::new(ir::Expr::Local(counter)),
                rhs: Box::new(ir::Expr::Literal(one)),
            },
        }];

        out.push(ir::Stmt::While {
            condition: ir::Expr::Binary {
                op: if inclusive {
                    ir::BinaryOp::LessEqual
                } else {
                    ir::BinaryOp::Less
                },
                lhs: Box::new(ir::Expr::Local(counter)),
                rhs: Box::new(end_value.expr),
            },
            body,
            continuing: increment,
        });
        Ok(())
    }
}

/// Whether an expression is one of the block shaped ones, which Rust allows
/// to stand as a statement without a semicolon.
fn is_block_shaped(expr: &syn::Expr) -> bool {
    matches!(
        expr,
        syn::Expr::If(_)
            | syn::Expr::While(_)
            | syn::Expr::Loop(_)
            | syn::Expr::ForLoop(_)
            | syn::Expr::Block(_)
    )
}

/// Pulls the name and optional type annotation out of a binding pattern.
fn binding_name(pat: &syn::Pat) -> syn::Result<(String, Option<syn::Type>)> {
    match pat {
        syn::Pat::Ident(ident) => {
            if ident.by_ref.is_some() || ident.subpat.is_some() {
                return Err(syn::Error::new_spanned(
                    pat,
                    "a kernel binding must be a plain name",
                ));
            }
            Ok((ident.ident.to_string(), None))
        }
        syn::Pat::Type(typed) => {
            let (name, _) = binding_name(&typed.pat)?;
            Ok((name, Some((*typed.ty).clone())))
        }
        other => Err(syn::Error::new_spanned(
            other,
            "a kernel binding must be a plain name, patterns are not supported",
        )),
    }
}
