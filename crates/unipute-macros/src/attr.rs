//! Parsing the `#[kernel(...)]` attribute and the `#[binding(...)]` markers
//! on parameters.

use syn::parse::{Parse, ParseStream};

/// The arguments to `#[kernel(...)]`.
pub struct KernelAttr {
    pub workgroup_size: [u32; 3],
    /// An entry point name that differs from the function name.
    pub name: Option<String>,
}

impl Default for KernelAttr {
    fn default() -> Self {
        Self {
            workgroup_size: [1, 1, 1],
            name: None,
        }
    }
}

impl Parse for KernelAttr {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attr = Self::default();
        let mut saw_workgroup_size = false;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "workgroup_size" => {
                    if saw_workgroup_size {
                        return Err(syn::Error::new_spanned(
                            &key,
                            "`workgroup_size` is set more than once",
                        ));
                    }
                    saw_workgroup_size = true;
                    attr.workgroup_size = parse_workgroup_size(input)?;
                }
                "name" => {
                    input.parse::<syn::Token![=]>()?;
                    let value: syn::LitStr = input.parse()?;
                    attr.name = Some(value.value());
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "`{other}` is not a kernel option, expected `workgroup_size` or `name`"
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }

        if !saw_workgroup_size {
            // Built by hand rather than through `input.error`, which would
            // prefix the message with "unexpected end of input".
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "a kernel needs a workgroup size, write #[kernel(workgroup_size(64))]",
            ));
        }
        Ok(attr)
    }
}

fn parse_workgroup_size(input: ParseStream<'_>) -> syn::Result<[u32; 3]> {
    let content;
    syn::parenthesized!(content in input);

    let mut size = [1u32; 3];
    let mut index = 0;
    while !content.is_empty() {
        if index == 3 {
            return Err(content.error("a workgroup has at most three dimensions"));
        }
        let value: syn::LitInt = content.parse()?;
        let dimension: u32 = value.base10_parse()?;
        if dimension == 0 {
            return Err(syn::Error::new_spanned(
                value,
                "a workgroup dimension must be at least 1",
            ));
        }
        size[index] = dimension;
        index += 1;
        if content.is_empty() {
            break;
        }
        content.parse::<syn::Token![,]>()?;
    }
    if index == 0 {
        return Err(content.error("`workgroup_size` needs at least one dimension"));
    }
    Ok(size)
}

/// Where a resource sits in the binding layout.
pub struct Placement {
    pub group: u32,
    pub binding: u32,
}

/// Reads the optional `#[binding(...)]` on a parameter.
///
/// Without one, a parameter lands in group 0 at its own position, so the
/// common case needs no annotation at all.
pub fn binding_placement(attrs: &[syn::Attribute], index: usize) -> syn::Result<Placement> {
    let mut placement = Placement {
        group: 0,
        binding: index as u32,
    };

    let mut seen = false;
    for attr in attrs {
        if !attr.path().is_ident("binding") {
            return Err(syn::Error::new_spanned(
                attr,
                "the only attribute allowed on a kernel parameter is `#[binding(...)]`",
            ));
        }
        if seen {
            return Err(syn::Error::new_spanned(
                attr,
                "a parameter can only have one `#[binding(...)]`",
            ));
        }
        seen = true;

        let mut group = None;
        let mut slot = None;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("group") {
                group = Some(parse_u32(&meta)?);
                Ok(())
            } else if meta.path.is_ident("index") {
                slot = Some(parse_u32(&meta)?);
                Ok(())
            } else {
                Err(meta.error("expected `group` or `index`"))
            }
        })?;

        if group.is_none() && slot.is_none() {
            return Err(syn::Error::new_spanned(
                attr,
                "write #[binding(group = 0, index = 1)], at least one of the two is needed",
            ));
        }
        if let Some(group) = group {
            placement.group = group;
        }
        if let Some(slot) = slot {
            placement.binding = slot;
        }
    }

    Ok(placement)
}

fn parse_u32(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<u32> {
    let value: syn::LitInt = meta.value()?.parse()?;
    value.base10_parse()
}
