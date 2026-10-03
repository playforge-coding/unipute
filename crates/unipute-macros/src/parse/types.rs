//! Mapping Rust type syntax onto Unipute IR types.

use syn::spanned::Spanned;
use unipute_ir as ir;

/// What a parameter's type says about how the kernel uses it.
pub struct ParamType {
    pub ty: ir::Type,
    pub access: ir::Access,
}

/// Reads a kernel parameter's type.
///
/// `&[T]` is a read only storage buffer, `&mut [T]` is a read and write one,
/// and anything else is a uniform.
pub fn param_type(ty: &syn::Type, structs: &[ir::StructType]) -> syn::Result<ParamType> {
    match ty {
        syn::Type::Reference(reference) => {
            let access = if reference.mutability.is_some() {
                ir::Access::ReadWrite
            } else {
                ir::Access::Read
            };
            match &*reference.elem {
                syn::Type::Slice(slice) => {
                    let element = value_type(&slice.elem, structs)?;
                    if access == ir::Access::Read
                        && let ir::Type::Atomic(_) = element
                    {
                        return Err(syn::Error::new_spanned(
                            ty,
                            "a buffer of atomics has to be `&mut`, since an atomic is for \
                             updating; a buffer nobody writes can be a plain `&[u32]`",
                        ));
                    }
                    Ok(ParamType {
                        ty: ir::Type::slice(element),
                        access,
                    })
                }
                other => {
                    if reference.mutability.is_some() {
                        return Err(syn::Error::new_spanned(
                            ty,
                            "a `&mut` parameter must be a slice, write `&mut [f32]`",
                        ));
                    }
                    Ok(ParamType {
                        ty: uniform_type(other, structs)?,
                        access: ir::Access::Uniform,
                    })
                }
            }
        }
        other => Ok(ParamType {
            ty: uniform_type(other, structs)?,
            access: ir::Access::Uniform,
        }),
    }
}

/// A uniform's type, which is anything but an atomic: nobody can write a
/// uniform, so there is nothing for an atomic to guard.
fn uniform_type(ty: &syn::Type, structs: &[ir::StructType]) -> syn::Result<ir::Type> {
    let parsed = value_type(ty, structs)?;
    if let ir::Type::Atomic(_) = parsed {
        return Err(syn::Error::new_spanned(
            ty,
            "a uniform cannot be an atomic, since nothing can write to it; put atomics in a \
             `&mut [AtomicU32]` buffer",
        ));
    }
    Ok(parsed)
}

/// Reads the type of workgroup memory: a value type, or a fixed length array
/// of one, nested as deep as the kernel likes.
///
/// The length has to be a number written in the kernel. A `const` from the
/// surrounding crate is not visible in here, the same as everywhere else in a
/// kernel body.
pub fn shared_type(ty: &syn::Type, structs: &[ir::StructType]) -> syn::Result<ir::Type> {
    match ty {
        syn::Type::Array(array) => {
            let element = shared_type(&array.elem, structs)?;
            let len = array_len(&array.len)?;
            Ok(ir::Type::Array {
                element: Box::new(element),
                len: Some(len),
            })
        }
        syn::Type::Slice(_) => Err(syn::Error::new_spanned(
            ty,
            "workgroup memory needs a fixed length, write `[f32; 64]`",
        )),
        other => value_type(other, structs),
    }
}

fn array_len(len: &syn::Expr) -> syn::Result<u32> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(value),
        ..
    }) = len
    else {
        return Err(syn::Error::new_spanned(
            len,
            "an array length has to be written as a number, a `const` is not visible inside a \
             kernel",
        ));
    };
    if !matches!(value.suffix(), "" | "usize") {
        return Err(syn::Error::new_spanned(
            value,
            "an array length is a `usize`, write it without a suffix",
        ));
    }
    let len: u32 = value.base10_parse()?;
    if len == 0 {
        return Err(syn::Error::new_spanned(
            value,
            "workgroup memory needs at least one element",
        ));
    }
    Ok(len)
}

/// Reads a type that describes a value, such as `f32`, `Vec3<f32>` or the
/// name of one of `structs`.
pub fn value_type(ty: &syn::Type, structs: &[ir::StructType]) -> syn::Result<ir::Type> {
    let syn::Type::Path(path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "expected a scalar such as `f32`, a vector such as `Vec3<f32>`, or a struct",
        ));
    };
    if path.qself.is_some() {
        return Err(syn::Error::new_spanned(
            ty,
            "qualified paths are not supported",
        ));
    }
    let segment = path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(ty, "empty type path"))?;
    let name = segment.ident.to_string();

    if let Some(scalar) = ir::Scalar::from_rust_name(&name) {
        if !segment.arguments.is_empty() {
            return Err(syn::Error::new_spanned(
                &segment.arguments,
                format!("`{name}` does not take type arguments"),
            ));
        }
        return Ok(ir::Type::scalar(scalar));
    }

    // Vec2, Vec3 and Vec4, with the component type in angle brackets.
    if let Some(size) = vector_size(&name) {
        let scalar = vector_argument(segment, ty, structs)?;
        return Ok(ir::Type::vector(size, scalar));
    }

    // AtomicU32 and AtomicI32, named the way Rust's own are.
    if let Some(scalar) = atomic_scalar(&name) {
        if !segment.arguments.is_empty() {
            return Err(syn::Error::new_spanned(
                &segment.arguments,
                format!("`{name}` does not take type arguments"),
            ));
        }
        return Ok(ir::Type::Atomic(scalar));
    }

    if let Some(def) = structs.iter().find(|def| def.name == name) {
        if !segment.arguments.is_empty() {
            return Err(syn::Error::new_spanned(
                &segment.arguments,
                format!("`{name}` does not take type arguments"),
            ));
        }
        return Ok(ir::Type::Struct(def.clone()));
    }

    Err(syn::Error::new_spanned(
        ty,
        format!(
            "`{name}` is not a type Unipute knows, use a scalar, `Vec2`, `Vec3` or `Vec4`, \
             `AtomicU32` or `AtomicI32`, or a struct with `#[derive(unipute::Layout)]`"
        ),
    ))
}

/// Reads a type that a value can have: anything [`value_type`] accepts except
/// an atomic, which is a place to operate on rather than a value to hold.
pub fn plain_value_type(
    ty: &syn::Type,
    structs: &[ir::StructType],
    what: &str,
) -> syn::Result<ir::Type> {
    let parsed = value_type(ty, structs)?;
    if let ir::Type::Atomic(_) = parsed {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "{what} cannot be an atomic, an atomic lives in a `&mut` buffer or in \
                 workgroup memory and is reached with `.load()` and the `fetch_` methods"
            ),
        ));
    }
    Ok(parsed)
}

fn vector_size(name: &str) -> Option<ir::VectorSize> {
    match name {
        "Vec2" => Some(ir::VectorSize::Two),
        "Vec3" => Some(ir::VectorSize::Three),
        "Vec4" => Some(ir::VectorSize::Four),
        _ => None,
    }
}

fn atomic_scalar(name: &str) -> Option<ir::Scalar> {
    match name {
        "AtomicU32" => Some(ir::Scalar::U32),
        "AtomicI32" => Some(ir::Scalar::I32),
        _ => None,
    }
}

fn vector_argument(
    segment: &syn::PathSegment,
    ty: &syn::Type,
    structs: &[ir::StructType],
) -> syn::Result<ir::Scalar> {
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            "a vector needs a component type, write `Vec3<f32>`",
        ));
    };
    if arguments.args.len() != 1 {
        return Err(syn::Error::new_spanned(
            arguments,
            "a vector takes exactly one component type",
        ));
    }
    let syn::GenericArgument::Type(argument) = &arguments.args[0] else {
        return Err(syn::Error::new_spanned(
            &arguments.args[0],
            "expected a component type",
        ));
    };
    match value_type(argument, structs)? {
        ir::Type::Scalar(scalar) => Ok(scalar),
        other => Err(syn::Error::new(
            argument.span(),
            format!("a vector component must be a scalar, found `{other}`"),
        )),
    }
}
