//! Producing the Rust code that replaces a `#[kernel]` function.
//!
//! This is where the compile time part of "compile time generating" actually
//! happens. Each enabled target is compiled here, inside the proc macro, and
//! the result is pasted into the caller's crate as a constant.

use proc_macro2::TokenStream;
use quote::quote;
use unipute_ir as ir;

use crate::emit;

/// Builds the type that stands in for the kernel function.
pub fn kernel_type(function: &syn::ItemFn, kernel: &ir::Kernel) -> syn::Result<TokenStream> {
    let ident = &function.sig.ident;
    let visibility = &function.vis;
    let docs = doc_attrs(&function.attrs);
    reject_other_attrs(&function.attrs)?;

    let name = &kernel.name;
    let [x, y, z] = kernel.workgroup_size;
    let bindings = kernel.resources.iter().map(binding_info);
    let ir_expr = emit::kernel(kernel);
    let targets = compile_targets(function, kernel)?;

    let summary =
        format!("The `{name}` compute kernel, generated from a Rust function by `#[kernel]`.",);

    Ok(quote! {
        #(#docs)*
        #[doc = ""]
        #[doc = #summary]
        #[allow(non_camel_case_types)]
        #[derive(::std::clone::Clone, ::std::marker::Copy, ::std::fmt::Debug)]
        #visibility struct #ident;

        impl ::unipute::Kernel for #ident {
            const NAME: &'static str = #name;
            const WORKGROUP_SIZE: [u32; 3] = [#x, #y, #z];
            const BINDINGS: &'static [::unipute::BindingInfo] = &[#(#bindings),*];

            fn ir() -> ::unipute::ir::Kernel {
                #ir_expr
            }
        }

        #targets
    })
}

fn binding_info(resource: &ir::Resource) -> TokenStream {
    let name = &resource.name;
    let group = resource.group;
    let binding = resource.binding;
    let access = emit::access(resource.access);
    quote! {
        ::unipute::BindingInfo {
            name: #name,
            group: #group,
            binding: #binding,
            access: #access,
        }
    }
}

/// Runs the naga back end once per enabled target.
///
/// A failure here is reported against the function, so the error shows up
/// where the kernel is written rather than somewhere inside the macro.
fn compile_targets(function: &syn::ItemFn, kernel: &ir::Kernel) -> syn::Result<TokenStream> {
    let ident = &function.sig.ident;
    let mut impls = TokenStream::new();

    // Every block below is behind a feature. A build with no target on is
    // unusual but valid: it still gets the kernel type and its IR, just no
    // shader constants. Naming the inputs once keeps that case warning free.
    let _ = (ident, function, kernel, &mut impls);

    #[cfg(feature = "wgsl")]
    {
        let source = run(function, unipute_naga::compile_wgsl(kernel))?;
        impls.extend(quote! {
            impl ::unipute::WgslKernel for #ident {
                const WGSL: &'static str = #source;
            }
        });
    }

    #[cfg(feature = "spv")]
    {
        let words = run(function, unipute_naga::compile_spirv(kernel))?;
        impls.extend(quote! {
            impl ::unipute::SpirvKernel for #ident {
                const SPIRV: &'static [u32] = &[#(#words),*];
            }
        });
    }

    #[cfg(feature = "msl")]
    {
        let source = run(function, unipute_naga::compile_msl(kernel))?;
        impls.extend(quote! {
            impl ::unipute::MslKernel for #ident {
                const MSL: &'static str = #source;
            }
        });
    }

    #[cfg(feature = "hlsl")]
    {
        let source = run(function, unipute_naga::compile_hlsl(kernel))?;
        impls.extend(quote! {
            impl ::unipute::HlslKernel for #ident {
                const HLSL: &'static str = #source;
            }
        });
    }

    #[cfg(feature = "glsl")]
    {
        let source = run(function, unipute_naga::compile_glsl(kernel))?;
        impls.extend(quote! {
            impl ::unipute::GlslKernel for #ident {
                const GLSL: &'static str = #source;
            }
        });
    }

    Ok(impls)
}

/// Turns a back end error into one that points at the kernel.
#[cfg(any(
    feature = "wgsl",
    feature = "spv",
    feature = "msl",
    feature = "hlsl",
    feature = "glsl"
))]
fn run<T>(function: &syn::ItemFn, result: unipute_naga::Result<T>) -> syn::Result<T> {
    result.map_err(|error| syn::Error::new_spanned(&function.sig, error.to_string()))
}

fn doc_attrs(attrs: &[syn::Attribute]) -> Vec<&syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .collect()
}

fn reject_other_attrs(attrs: &[syn::Attribute]) -> syn::Result<()> {
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            return Err(syn::Error::new_spanned(
                attr,
                "attributes other than doc comments are not carried over to a kernel",
            ));
        }
    }
    Ok(())
}
