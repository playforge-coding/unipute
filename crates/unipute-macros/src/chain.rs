//! How `#[kernel]` learns about the structs a kernel uses.
//!
//! A proc macro sees the one item it is attached to. A struct defined
//! somewhere else in the crate is just a name to it, and a kernel cannot be
//! compiled without knowing that struct's fields. So `#[derive(Layout)]` on
//! the struct leaves a `macro_rules!` behind, named after the struct, that
//! carries the fields, and the kernel macro asks it.
//!
//! The asking is an expansion chain. When `#[kernel]` finds names it does not
//! know, it expands to a call of the first one's macro, passing along the
//! names still to look up, the definitions gathered so far, and the kernel
//! itself. That macro adds its struct to the list and calls back into
//! [`__kernel_with_layouts`](crate::__kernel_with_layouts), which asks the
//! next name or, once the list is empty, expands the kernel for real. Every
//! token of the original function is forwarded as is, so spans survive and
//! errors still point at the kernel.
//!
//! One consequence is that a kernel and its structs have to be in the same
//! crate. A `macro_rules!` without `#[macro_export]` cannot be re-exported,
//! and one with it cannot be reached by path from the crate that expanded it,
//! so the macro the derive leaves behind is only visible within the crate.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::visit::Visit;
use unipute_ir as ir;

use crate::layout;

/// What one step of the chain carries: the structs gathered so far, the
/// names still to ask, the kernel attribute and the kernel itself.
pub struct Step {
    pub known: Vec<syn::ItemStruct>,
    pub pending: Vec<syn::Path>,
    pub attr: TokenStream,
    pub function: syn::ItemFn,
}

impl Parse for Step {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let content;
        syn::bracketed!(content in input);
        let mut known = Vec::new();
        while !content.is_empty() {
            known.push(content.parse()?);
        }

        let content;
        syn::bracketed!(content in input);
        let pending = content
            .parse_terminated(syn::Path::parse_mod_style, syn::Token![,])?
            .into_iter()
            .collect();

        let content;
        syn::parenthesized!(content in input);
        let attr = content.parse()?;

        let function = input.parse()?;
        Ok(Self {
            known,
            pending,
            attr,
            function,
        })
    }
}

impl Step {
    /// The first step, taken by `#[kernel]` itself: every name the function
    /// mentions that is not a type Unipute already knows.
    pub fn first(attr: TokenStream, function: syn::ItemFn) -> Self {
        let mut names = Names::default();
        names.visit_item_fn(&function);
        Self {
            known: Vec::new(),
            pending: names.found,
            attr,
            function,
        }
    }

    /// Expands to a call of the next struct's macro, or `None` when every
    /// name has been looked up.
    pub fn ask_next(&self) -> Option<TokenStream> {
        let (next, rest) = self.pending.split_first()?;
        let known = &self.known;
        let attr = &self.attr;
        let function = &self.function;
        // If `next` is not a struct with `#[derive(Layout)]`, the macro call
        // fails to resolve. The trait check gives a second, clearer error next
        // to it, since the trait says what to do.
        let checks = self.pending.iter().map(|path| {
            quote! { let _ = __unipute_needs_layout::<#path>; }
        });
        Some(quote! {
            #next! { ::unipute::__kernel_with_layouts [ #(#known)* ] [ #(#rest),* ] ( #attr ) #function }
            const _: () = {
                fn __unipute_needs_layout<T: ::unipute::Layout>() {}
                #(#checks)*
            };
        })
    }

    /// The structs gathered along the chain, as the front end wants them.
    ///
    /// The same struct can arrive twice when a kernel names it two ways, such
    /// as `Body` and `crate::Body`, which is fine as long as both are the same
    /// definition.
    pub fn structs(&self) -> syn::Result<Vec<ir::StructType>> {
        let mut structs: Vec<ir::StructType> = Vec::new();
        for item in &self.known {
            let analysed = layout::analyse(item)?;
            match structs.iter().find(|seen| seen.name == analysed.ty.name) {
                Some(seen) if *seen == analysed.ty => {}
                Some(_) => {
                    return Err(syn::Error::new_spanned(
                        &item.ident,
                        format!(
                            "two different structs named `{}` reach this kernel",
                            item.ident
                        ),
                    ));
                }
                None => structs.push(analysed.ty),
            }
        }
        Ok(structs)
    }
}

/// Collects every type name in a function that Unipute does not know on its
/// own, in the order they are first seen.
#[derive(Default)]
struct Names {
    found: Vec<syn::Path>,
    seen: HashSet<String>,
}

impl Names {
    fn note(&mut self, path: &syn::Path) {
        let key = path.to_token_stream().to_string();
        if self.seen.insert(key) {
            self.found.push(path.clone());
        }
    }
}

impl<'ast> Visit<'ast> for Names {
    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        if ty.qself.is_none()
            && let Some(last) = ty.path.segments.last()
            && !is_built_in_type(&last.ident.to_string())
        {
            self.note(&ty.path);
        }
        syn::visit::visit_type_path(self, ty);
    }

    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        if expr.qself.is_none() {
            self.note(&expr.path);
        }
        syn::visit::visit_expr_struct(self, expr);
    }
}

fn is_built_in_type(name: &str) -> bool {
    ir::Scalar::from_rust_name(name).is_some() || matches!(name, "Vec2" | "Vec3" | "Vec4")
}
