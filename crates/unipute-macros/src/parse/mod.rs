//! The Rust front end: `syn` syntax in, Unipute IR out.

mod expr;
mod stmt;
mod types;

use std::collections::HashMap;

use unipute_ir as ir;

use crate::attr::KernelAttr;
use types::param_type;

/// What a name in scope refers to.
#[derive(Clone)]
pub enum Binding {
    Local { id: ir::LocalId, ty: ir::Type },
    Resource { id: ir::ResourceId, ty: ir::Type },
}

/// Names visible while reading the body, plus the kernel being built.
pub struct Scope<'a> {
    kernel: &'a mut ir::Kernel,
    /// One map per nested block, innermost last.
    frames: Vec<HashMap<String, Binding>>,
    /// How deep we are inside loops, so `break` outside one is an error.
    loop_depth: u32,
}

impl<'a> Scope<'a> {
    fn new(kernel: &'a mut ir::Kernel) -> Self {
        Self {
            kernel,
            frames: vec![HashMap::new()],
            loop_depth: 0,
        }
    }

    fn lookup(&self, name: &str) -> Option<Binding> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.get(name))
            .cloned()
    }

    fn define(&mut self, name: String, binding: Binding) {
        self.frames
            .last_mut()
            .expect("there is always at least one scope")
            .insert(name, binding);
    }

    /// Adds a local variable to the kernel and puts its name in scope.
    fn declare_local(&mut self, name: &str, ty: ir::Type) -> ir::LocalId {
        let id = ir::LocalId(self.kernel.locals.len() as u32);
        self.kernel.locals.push(ir::Local {
            name: name.to_owned(),
            ty: ty.clone(),
        });
        self.define(name.to_owned(), Binding::Local { id, ty });
        id
    }

    fn push_frame(&mut self) {
        self.frames.push(HashMap::new());
    }

    fn pop_frame(&mut self) {
        self.frames.pop();
    }
}

/// Reads a whole `#[kernel]` function into a [`ir::Kernel`].
pub fn kernel(attr: &KernelAttr, function: &syn::ItemFn) -> syn::Result<ir::Kernel> {
    check_signature(function)?;

    let name = attr
        .name
        .clone()
        .unwrap_or_else(|| function.sig.ident.to_string());
    let mut kernel = ir::Kernel::new(name, attr.workgroup_size);

    let mut resources = Vec::new();
    for (index, argument) in function.sig.inputs.iter().enumerate() {
        resources.push(resource(argument, index)?);
    }
    check_unique_bindings(&resources, &function.sig)?;
    kernel.resources = resources
        .iter()
        .map(|(resource, _)| resource.clone())
        .collect();

    let mut scope = Scope::new(&mut kernel);
    for (index, (resource, name)) in resources.iter().enumerate() {
        scope.define(
            name.clone(),
            Binding::Resource {
                id: ir::ResourceId(index as u32),
                ty: resource.ty.clone(),
            },
        );
    }
    let body = scope.block(&function.block)?;
    kernel.body = body;

    Ok(kernel)
}

fn check_signature(function: &syn::ItemFn) -> syn::Result<()> {
    if let Some(asyncness) = function.sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "a kernel cannot be `async`",
        ));
    }
    if let Some(constness) = function.sig.constness {
        return Err(syn::Error::new_spanned(
            constness,
            "a kernel cannot be `const`, it is already evaluated at compile time",
        ));
    }
    if !function.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &function.sig.generics,
            "a kernel cannot be generic",
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &function.sig.output {
        return Err(syn::Error::new_spanned(
            ty,
            "a kernel returns nothing, write results into a `&mut` parameter",
        ));
    }
    Ok(())
}

/// Reads one parameter into a resource, returning it with the name it binds.
fn resource(argument: &syn::FnArg, index: usize) -> syn::Result<(ir::Resource, String)> {
    let syn::FnArg::Typed(typed) = argument else {
        return Err(syn::Error::new_spanned(
            argument,
            "a kernel is a free function, it cannot take `self`",
        ));
    };
    let syn::Pat::Ident(pattern) = &*typed.pat else {
        return Err(syn::Error::new_spanned(
            &typed.pat,
            "a kernel parameter must be a plain name",
        ));
    };
    let name = pattern.ident.to_string();
    let parsed = param_type(&typed.ty)?;
    let placement = crate::attr::binding_placement(&typed.attrs, index)?;

    Ok((
        ir::Resource {
            name: name.clone(),
            group: placement.group,
            binding: placement.binding,
            ty: parsed.ty,
            access: parsed.access,
        },
        name,
    ))
}

fn check_unique_bindings(
    resources: &[(ir::Resource, String)],
    signature: &syn::Signature,
) -> syn::Result<()> {
    let mut seen = HashMap::new();
    for (resource, _) in resources {
        let slot = (resource.group, resource.binding);
        if let Some(previous) = seen.insert(slot, resource.name.clone()) {
            return Err(syn::Error::new_spanned(
                signature,
                format!(
                    "`{}` and `{}` both use group {} binding {}, set one with #[binding(...)]",
                    previous, resource.name, resource.group, resource.binding
                ),
            ));
        }
    }
    Ok(())
}
