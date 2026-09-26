//! `#[derive(Layout)]`: a struct the host and a kernel both read.
//!
//! A struct in a buffer has two layouts, the one rustc gives it and the one
//! the GPU expects, and they have to agree byte for byte or the kernel reads
//! garbage without anyone noticing. This module works both out from the
//! struct's syntax and refuses to compile the two apart, with a message
//! saying which field to move or pad.
//!
//! The derive leaves three things behind: an `impl Layout` handing the struct
//! back as IR, a `const` block re-checking the offsets against what rustc
//! actually did, and a `macro_rules!` with the struct's own name that
//! `#[kernel]` calls to learn the fields. That last one is how a macro on one
//! item finds out about another, and [`crate::chain`] is the other half of it.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use unipute_ir as ir;

use crate::emit;

/// A struct that has been checked and laid out.
pub struct Analysed {
    /// What the kernel sees.
    pub ty: ir::StructType,
    /// Every field in source order, including padding, as rustc lays them.
    host: Vec<HostField>,
    /// The struct's alignment on the host, after any `align(N)` in its repr.
    host_alignment: u32,
    host_size: u32,
}

struct HostField {
    name: syn::Ident,
    /// The type as written, so it can be forwarded to the kernel macro.
    ty: syn::Type,
    offset: u32,
    /// `None` for padding, which the kernel does not see.
    member: Option<usize>,
}

/// Reads a struct that has `#[derive(Layout)]` on it, or one that a derive
/// forwarded to `#[kernel]`, into its two layouts.
pub fn analyse(item: &syn::ItemStruct) -> syn::Result<Analysed> {
    let name = item.ident.to_string();
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            format!("`{name}` cannot be generic, a kernel needs to know every field's type"),
        ));
    }
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(syn::Error::new_spanned(
            &item.fields,
            format!("`{name}` needs named fields, a kernel reads a struct's fields by name"),
        ));
    };
    if fields.named.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.ident,
            format!("`{name}` has no fields, so there is nothing to put in a buffer"),
        ));
    }
    let repr = repr(item)?;

    let mut members = Vec::new();
    let mut host = Vec::new();
    let mut cursor = 0;
    let mut host_alignment = repr.align.unwrap_or(1);
    for field in &fields.named {
        let ident = field.ident.clone().expect("named fields have names");
        let field_name = ident.to_string();
        let (member, size, align) = if field_name.starts_with('_') {
            (None, padding_size(field)?, 1)
        } else {
            let ty = field_type(field)?;
            let size = ty.size().expect("scalars and vectors have a size");
            members.push((field_name, ty));
            (Some(members.len() - 1), size, 4)
        };
        let offset = ir::round_up(cursor, align);
        host.push(HostField {
            name: ident,
            ty: field.ty.clone(),
            offset,
            member,
        });
        cursor = offset + size;
        host_alignment = host_alignment.max(align);
    }
    if members.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.ident,
            format!("`{name}` is nothing but padding, a kernel needs at least one field to read"),
        ));
    }

    let ty = ir::StructType::new(name, members).expect("every member has a fixed size");
    Ok(Analysed {
        ty,
        host,
        host_alignment,
        host_size: ir::round_up(cursor, host_alignment),
    })
}

/// What `#[repr(...)]` on the struct asks for.
struct Repr {
    align: Option<u32>,
}

/// Reads the repr, which has to be `C`, optionally with `align(N)`.
///
/// Without `repr(C)` rustc is free to reorder fields, and then the offsets
/// worked out here mean nothing.
fn repr(item: &syn::ItemStruct) -> syn::Result<Repr> {
    let mut is_c = false;
    let mut align = None;
    for attr in &item.attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("C") {
                is_c = true;
                Ok(())
            } else if meta.path.is_ident("align") {
                let content;
                syn::parenthesized!(content in meta.input);
                let value: syn::LitInt = content.parse()?;
                align = Some(value.base10_parse()?);
                Ok(())
            } else {
                let other = meta
                    .path
                    .get_ident()
                    .map_or_else(|| "that".to_owned(), |ident| format!("`{ident}`"));
                Err(meta.error(format!(
                    "a struct in a buffer is `#[repr(C)]`, {other} is not a layout the GPU has"
                )))
            }
        })?;
    }
    if !is_c {
        return Err(syn::Error::new_spanned(
            &item.ident,
            format!(
                "`{}` needs `#[repr(C)]`, so that its fields sit where the kernel expects them",
                item.ident
            ),
        ));
    }
    Ok(Repr { align })
}

/// A padding field is `[u8; N]` with a name starting with an underscore. The
/// kernel never sees it, and the host puts N bytes there.
fn padding_size(field: &syn::Field) -> syn::Result<u32> {
    let wrong = || {
        syn::Error::new_spanned(
            &field.ty,
            format!(
                "`{}` starts with an underscore, which makes it padding the kernel does not see, \
                 so its type has to be `[u8; N]`",
                field.ident.as_ref().expect("named fields have names")
            ),
        )
    };
    let syn::Type::Array(array) = &field.ty else {
        return Err(wrong());
    };
    let syn::Type::Path(element) = &*array.elem else {
        return Err(wrong());
    };
    if !element.path.is_ident("u8") {
        return Err(wrong());
    }
    let len = array_len(&array.len)?;
    if len == 0 {
        return Err(syn::Error::new_spanned(
            &array.len,
            "padding needs at least one byte, or take the field out",
        ));
    }
    Ok(len)
}

fn array_len(len: &syn::Expr) -> syn::Result<u32> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(value),
        ..
    }) = len
    else {
        return Err(syn::Error::new_spanned(
            len,
            "an array length here has to be written as a number, since the layout is worked \
             out before any `const` is evaluated",
        ));
    };
    value.base10_parse()
}

/// Reads a field the kernel will see: a scalar, or an array of two to four
/// scalars, which is how a vector is spelled on the host.
fn field_type(field: &syn::Field) -> syn::Result<ir::Type> {
    let name = field.ident.as_ref().expect("named fields have names");
    match &field.ty {
        syn::Type::Path(path) if path.qself.is_none() => {
            let segment = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(&field.ty, "empty type path"))?;
            let type_name = segment.ident.to_string();
            match ir::Scalar::from_rust_name(&type_name) {
                Some(ir::Scalar::Bool) => Err(syn::Error::new_spanned(
                    &field.ty,
                    format!(
                        "`{name}` is a `bool`, which has no size the host and the GPU agree on, \
                         store a `u32` and compare it with zero"
                    ),
                )),
                Some(scalar) => {
                    if !segment.arguments.is_empty() {
                        return Err(syn::Error::new_spanned(
                            &segment.arguments,
                            format!("`{type_name}` does not take type arguments"),
                        ));
                    }
                    Ok(ir::Type::scalar(scalar))
                }
                None if matches!(type_name.as_str(), "Vec2" | "Vec3" | "Vec4") => {
                    let count = &type_name[3..];
                    Err(syn::Error::new_spanned(
                        &field.ty,
                        format!(
                            "on the host a vector is an array, write `[f32; {count}]` and the \
                             kernel will see `{type_name}<f32>`"
                        ),
                    ))
                }
                None => Err(syn::Error::new_spanned(
                    &field.ty,
                    format!(
                        "`{type_name}` is not a type a kernel can read from a buffer, a field is \
                         `f32`, `u32` or `i32`, or an array of two to four of them; a struct \
                         inside a struct is not supported yet"
                    ),
                )),
            }
        }
        syn::Type::Array(array) => {
            let element = match &*array.elem {
                syn::Type::Path(path) if path.qself.is_none() => path
                    .path
                    .get_ident()
                    .and_then(|ident| ir::Scalar::from_rust_name(&ident.to_string())),
                _ => None,
            };
            let len = array_len(&array.len)?;
            match (element, ir::VectorSize::from_count(len.min(255) as u8)) {
                (Some(ir::Scalar::Bool), _) => Err(syn::Error::new_spanned(
                    &field.ty,
                    format!(
                        "`{name}` holds `bool`s, which have no size the host and the GPU agree on"
                    ),
                )),
                (Some(scalar), Some(size)) => Ok(ir::Type::vector(size, scalar)),
                _ => Err(syn::Error::new_spanned(
                    &field.ty,
                    format!(
                        "`{name}` is an array, and the only arrays a kernel reads from a struct \
                         are `[f32; 2]`, `[f32; 3]` and `[f32; 4]` and their `u32` and `i32` \
                         forms, which it sees as vectors"
                    ),
                )),
            }
        }
        other => Err(syn::Error::new_spanned(
            other,
            format!(
                "`{name}` has a type a kernel cannot read from a buffer, a field is `f32`, `u32` \
                 or `i32`, or an array of two to four of them"
            ),
        )),
    }
}

/// Expands `#[derive(Layout)]`.
pub fn derive(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "`Layout` is for structs, since that is what a buffer holds",
        ));
    };
    let item = syn::ItemStruct {
        attrs: input.attrs.clone(),
        vis: input.vis.clone(),
        struct_token: data.struct_token,
        ident: input.ident.clone(),
        generics: input.generics.clone(),
        fields: data.fields.clone(),
        semi_token: data.semi_token,
    };
    let analysed = analyse(&item)?;
    check_host_layout(&item, &analysed)?;

    let ident = &item.ident;
    let ty = emit::ty(&ir::Type::Struct(analysed.ty.clone()));

    // Rustc's own answer, checked against the one worked out from the syntax.
    // The two only disagree if there is a bug here, so the message says so.
    let offset_checks = analysed.host.iter().map(|field| {
        let name = &field.name;
        let offset = field.offset as usize;
        let message = format!(
            "`{ident}::{name}` is not at byte {offset}, which is where #[derive(Layout)] worked \
             out it would be; this is a bug in Unipute, please report it"
        );
        quote! {
            assert!(::core::mem::offset_of!(#ident, #name) == #offset, #message);
        }
    });
    let size = analysed.host_size as usize;
    let size_message = format!(
        "`{ident}` is not {size} bytes, which is what #[derive(Layout)] worked out; this is a \
         bug in Unipute, please report it"
    );

    // The struct's fields, as `#[kernel]` needs to see them. Attributes and
    // visibility stay behind, apart from the repr, which the layout depends on.
    let reprs = item
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("repr"));
    let forwarded_fields = analysed.host.iter().map(|field| {
        let name = &field.name;
        let ty = &field.ty;
        quote!(#name: #ty)
    });
    let macro_name = format_ident!("__unipute_layout_{ident}");

    Ok(quote! {
        impl ::unipute::Layout for #ident {
            fn ty() -> ::unipute::ir::Type {
                #ty
            }
        }

        const _: () = {
            #(#offset_checks)*
            assert!(::core::mem::size_of::<#ident>() == #size, #size_message);
        };

        // Called by `#[kernel]` when a kernel names this struct. The kernel
        // macro cannot see this item, so it asks, and this answers with the
        // fields and hands control on to the callback.
        #[doc(hidden)]
        macro_rules! #macro_name {
            ($callback:path [ $($known:tt)* ] $($rest:tt)*) => {
                $callback! {
                    [ $($known)* #(#reprs)* struct #ident { #(#forwarded_fields),* } ]
                    $($rest)*
                }
            };
        }
        // Under the struct's own name, so that a `use` of the struct brings
        // the macro along with it. A macro and a type do not collide.
        #[doc(hidden)]
        #[allow(unused_imports)]
        pub(crate) use #macro_name as #ident;
    })
}

/// Refuses a struct whose host layout differs from the GPU one, saying what
/// to add where.
fn check_host_layout(item: &syn::ItemStruct, analysed: &Analysed) -> syn::Result<()> {
    let name = &item.ident;
    for field in &analysed.host {
        let Some(member) = field.member else {
            continue;
        };
        let gpu_offset = analysed.ty.members[member].offset;
        if field.offset < gpu_offset {
            let gap = gpu_offset - field.offset;
            let pad = padding_name(analysed);
            return Err(syn::Error::new(
                field.name.span(),
                format!(
                    "`{}` sits at byte {} on the CPU and byte {} on the GPU, add `{pad}: [u8; \
                     {gap}]` before it",
                    field.name, field.offset, gpu_offset
                ),
            ));
        }
        if field.offset > gpu_offset {
            return Err(syn::Error::new(
                field.name.span(),
                format!(
                    "`{}` sits at byte {} on the CPU and byte {} on the GPU, there are {} bytes \
                     of padding too many before it",
                    field.name,
                    field.offset,
                    gpu_offset,
                    field.offset - gpu_offset
                ),
            ));
        }
    }
    if analysed.host_size < analysed.ty.size {
        let gap = analysed.ty.size - analysed.host_size;
        let pad = padding_name(analysed);
        return Err(syn::Error::new(
            name.span(),
            format!(
                "`{name}` is {} bytes on the CPU and {} on the GPU, add `{pad}: [u8; {gap}]` \
                 as its last field",
                analysed.host_size, analysed.ty.size
            ),
        ));
    }
    if analysed.host_size > analysed.ty.size {
        return Err(syn::Error::new(
            name.span(),
            format!(
                "`{name}` is {} bytes on the CPU and {} on the GPU, its `align({})` asks for \
                 more than the GPU uses",
                analysed.host_size, analysed.ty.size, analysed.host_alignment
            ),
        ));
    }
    Ok(())
}

/// A name for a padding field that the struct does not already use.
fn padding_name(analysed: &Analysed) -> String {
    let taken: Vec<String> = analysed
        .host
        .iter()
        .map(|field| field.name.to_string())
        .collect();
    if !taken.iter().any(|name| name == "_pad") {
        return "_pad".to_owned();
    }
    (2..)
        .map(|n| format!("_pad{n}"))
        .find(|candidate| !taken.contains(candidate))
        .expect("the numbers do not run out")
}
