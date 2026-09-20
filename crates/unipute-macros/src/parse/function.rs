//! Nested `fn` items: the helper functions a kernel can call.
//!
//! A helper is written at the top level of the kernel body and behaves like a
//! nested `fn` does in Rust. It captures nothing, which lines up with what a
//! shader language allows: bindings and hardware built-ins reach the entry
//! point only, so a helper takes what it needs as a parameter.
//!
//! Order matters to a back end. Shader languages have no forward
//! declarations, so a callee has to be written before its callers, and none of
//! them can express recursion. Both fall out of a topological sort of the call
//! graph, which is why the graph is read off the syntax here before any body
//! is parsed.

use std::collections::HashMap;

use syn::visit::Visit;
use unipute_ir as ir;

use super::Scope;
use super::types::value_type;

/// A helper's name and signature.
///
/// These are worked out for every helper before any body is read, so that a
/// call can be resolved no matter which of the two functions came first.
pub struct Signature {
    pub id: ir::FunctionId,
    pub name: String,
    pub params: Vec<ir::Param>,
    pub result: Option<ir::Type>,
}

/// The helpers of one kernel, ready for both the IR and the parser.
pub struct Helpers {
    /// Signatures in the same order as `functions`, for resolving calls.
    pub signatures: Vec<Signature>,
    pub functions: Vec<ir::Function>,
}

/// Reads every nested `fn` at the top of a kernel body.
pub fn helpers(block: &syn::Block) -> syn::Result<Helpers> {
    let items = collect(block)?;
    let order = order(&items)?;

    let mut signatures = Vec::with_capacity(order.len());
    for (id, index) in order.iter().enumerate() {
        signatures.push(signature(items[*index], ir::FunctionId(id as u32))?);
    }

    let mut functions = Vec::with_capacity(order.len());
    for (position, index) in order.iter().enumerate() {
        functions.push(body(items[*index], &signatures[position], &signatures)?);
    }

    Ok(Helpers {
        signatures,
        functions,
    })
}

/// Picks the `fn` items out of the kernel body and checks what they are
/// allowed to be.
fn collect(block: &syn::Block) -> syn::Result<Vec<&syn::ItemFn>> {
    let mut items: Vec<&syn::ItemFn> = Vec::new();
    for stmt in &block.stmts {
        let syn::Stmt::Item(item) = stmt else {
            continue;
        };
        let syn::Item::Fn(item) = item else {
            return Err(syn::Error::new_spanned(
                item,
                "only `fn` items can be declared inside a kernel",
            ));
        };
        check_signature(item)?;

        let name = item.sig.ident.to_string();
        if is_reserved(&name) {
            return Err(syn::Error::new_spanned(
                &item.sig.ident,
                format!("`{name}` is the name of a built-in Unipute provides, pick another"),
            ));
        }
        if items.iter().any(|seen| seen.sig.ident == item.sig.ident) {
            return Err(syn::Error::new_spanned(
                &item.sig.ident,
                format!("`{name}` is declared twice in this kernel"),
            ));
        }
        items.push(item);
    }
    Ok(items)
}

/// Whether a name already means something to the front end.
fn is_reserved(name: &str) -> bool {
    ir::BuiltIn::from_intrinsic_name(name).is_some()
        || ir::MathFn::from_intrinsic_name(name).is_some()
        || matches!(
            name,
            "vec2" | "vec3" | "vec4" | "workgroup_barrier" | "storage_barrier"
        )
}

fn check_signature(item: &syn::ItemFn) -> syn::Result<()> {
    let name = item.sig.ident.to_string();
    if let Some(asyncness) = item.sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            format!("`{name}` cannot be `async`"),
        ));
    }
    if let syn::Safety::Unsafe(unsafety) = item.sig.safety {
        return Err(syn::Error::new_spanned(
            unsafety,
            format!("`{name}` cannot be `unsafe`"),
        ));
    }
    if !item.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.sig.generics,
            format!("`{name}` cannot be generic"),
        ));
    }
    Ok(())
}

/// Reads a helper's parameters and result type.
fn signature(item: &syn::ItemFn, id: ir::FunctionId) -> syn::Result<Signature> {
    let name = item.sig.ident.to_string();

    let mut params = Vec::new();
    for argument in &item.sig.inputs {
        let syn::FnArg::Typed(typed) = argument else {
            return Err(syn::Error::new_spanned(
                argument,
                format!("`{name}` is a free function, it cannot take `self`"),
            ));
        };
        let syn::Pat::Ident(pattern) = &*typed.pat else {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "a parameter must be a plain name",
            ));
        };
        if pattern.mutability.is_some() {
            return Err(syn::Error::new_spanned(
                pattern,
                "a `mut` parameter is not supported, copy it into a `let` instead",
            ));
        }
        // `value_type` only ever yields a scalar or a vector, which is exactly
        // what a shader function can take by value.
        params.push(ir::Param {
            name: pattern.ident.to_string(),
            ty: value_type(&typed.ty)?,
        });
    }

    let result = match &item.sig.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, ty) => Some(value_type(ty)?),
    };

    Ok(Signature {
        id,
        name,
        params,
        result,
    })
}

/// Reads a helper's body, once every signature is known.
fn body(
    item: &syn::ItemFn,
    signature: &Signature,
    signatures: &[Signature],
) -> syn::Result<ir::Function> {
    let mut scope = Scope::helper(signature, signatures);
    let body = scope.function_body(&item.block, signature.result.as_ref())?;

    if signature.result.is_some() && !always_returns(&body) {
        return Err(syn::Error::new_spanned(
            &item.sig.ident,
            format!(
                "`{}` returns a value, so its body has to end with one or with a `return`",
                signature.name
            ),
        ));
    }

    Ok(ir::Function {
        name: signature.name.clone(),
        params: signature.params.clone(),
        result: signature.result.clone(),
        locals: scope.locals,
        body,
    })
}

/// Whether a body is certain to leave through a `return`.
///
/// This is deliberately simple: a trailing `return`, or a trailing `if` whose
/// branches both return. A loop never counts, even one that cannot be left any
/// other way, so that the error asks for a `return` the reader can see rather
/// than depending on an analysis they cannot.
fn always_returns(body: &[ir::Stmt]) -> bool {
    match body.last() {
        Some(ir::Stmt::Return { .. }) => true,
        Some(ir::Stmt::If {
            then_branch,
            else_branch,
            ..
        }) => !else_branch.is_empty() && always_returns(then_branch) && always_returns(else_branch),
        _ => false,
    }
}

/// Orders the helpers so that a callee always comes before its callers,
/// rejecting recursion on the way.
fn order(items: &[&syn::ItemFn]) -> syn::Result<Vec<usize>> {
    let index_of: HashMap<String, usize> = items
        .iter()
        .enumerate()
        .map(|(index, item)| (item.sig.ident.to_string(), index))
        .collect();

    // Reading the calls off the syntax rather than the IR is what lets this
    // run before any body is parsed. It can only over-approximate, by picking
    // up a call in code that never runs, and an extra edge costs nothing.
    let calls: Vec<Vec<usize>> = items
        .iter()
        .map(|item| {
            let mut visitor = Calls {
                names: &index_of,
                found: Vec::new(),
            };
            visitor.visit_block(&item.block);
            visitor.found
        })
        .collect();

    let mut marks = vec![Mark::New; items.len()];
    let mut order = Vec::with_capacity(items.len());
    for index in 0..items.len() {
        walk(index, items, &calls, &mut marks, &mut order)?;
    }
    Ok(order)
}

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    New,
    /// On the current path, so meeting it again is a cycle.
    Active,
    Done,
}

fn walk(
    index: usize,
    items: &[&syn::ItemFn],
    calls: &[Vec<usize>],
    marks: &mut [Mark],
    order: &mut Vec<usize>,
) -> syn::Result<()> {
    match marks[index] {
        Mark::Done => return Ok(()),
        Mark::Active => {
            return Err(syn::Error::new_spanned(
                &items[index].sig.ident,
                format!(
                    "`{}` takes part in a cycle of calls, and a shader function cannot be recursive",
                    items[index].sig.ident
                ),
            ));
        }
        Mark::New => {}
    }

    marks[index] = Mark::Active;
    for callee in &calls[index] {
        if *callee == index {
            return Err(syn::Error::new_spanned(
                &items[index].sig.ident,
                format!(
                    "`{}` calls itself, and a shader function cannot be recursive",
                    items[index].sig.ident
                ),
            ));
        }
        walk(*callee, items, calls, marks, order)?;
    }
    marks[index] = Mark::Done;
    order.push(index);
    Ok(())
}

/// Collects the helpers one body calls, by name.
struct Calls<'a> {
    names: &'a HashMap<String, usize>,
    found: Vec<usize>,
}

impl<'ast> Visit<'ast> for Calls<'_> {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func
            && let Some(ident) = path.path.get_ident()
            && let Some(index) = self.names.get(&ident.to_string())
            && !self.found.contains(index)
        {
            self.found.push(*index);
        }
        syn::visit::visit_expr_call(self, call);
    }
}
