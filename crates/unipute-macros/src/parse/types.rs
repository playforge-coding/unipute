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
pub fn param_type(ty: &syn::Type) -> syn::Result<ParamType> {
    match ty {
        syn::Type::Reference(reference) => {
            let access = if reference.mutability.is_some() {
                ir::Access::ReadWrite
            } else {
                ir::Access::Read
            };
            match &*reference.elem {
                syn::Type::Slice(slice) => {
                    let element = value_type(&slice.elem)?;
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
                        ty: value_type(other)?,
                        access: ir::Access::Uniform,
                    })
                }
            }
        }
        other => Ok(ParamType {
            ty: value_type(other)?,
            access: ir::Access::Uniform,
        }),
    }
}

/// Reads a type that describes a value, such as `f32` or `vec3<f32>`.
pub fn value_type(ty: &syn::Type) -> syn::Result<ir::Type> {
    let syn::Type::Path(path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "expected a scalar such as `f32` or a vector such as `Vec3<f32>`",
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
        let scalar = vector_argument(segment, ty)?;
        return Ok(ir::Type::vector(size, scalar));
    }

    Err(syn::Error::new_spanned(
        ty,
        format!("`{name}` is not a type Unipute knows, use a scalar or `Vec2`, `Vec3` or `Vec4`"),
    ))
}

fn vector_size(name: &str) -> Option<ir::VectorSize> {
    match name {
        "Vec2" => Some(ir::VectorSize::Two),
        "Vec3" => Some(ir::VectorSize::Three),
        "Vec4" => Some(ir::VectorSize::Four),
        _ => None,
    }
}

fn vector_argument(segment: &syn::PathSegment, ty: &syn::Type) -> syn::Result<ir::Scalar> {
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
    match value_type(argument)? {
        ir::Type::Scalar(scalar) => Ok(scalar),
        other => Err(syn::Error::new(
            argument.span(),
            format!("a vector component must be a scalar, found `{other}`"),
        )),
    }
}
