//! The `#[kernel]` attribute macro.
//!
//! This crate is a front end, not a library you use directly. Everything it
//! produces is re-exported from `unipute`, and the generated code refers to
//! `::unipute`, so that is the crate you depend on.
//!
//! The work happens while your crate is being compiled. The macro reads the
//! function body, builds Unipute IR, hands it to the naga back end, and pastes
//! the finished shader back into your crate as a `&'static str`. Nothing is
//! translated at run time unless you ask for it.

#![forbid(unsafe_code)]

mod attr;
mod emit;
mod generate;
mod parse;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Compiles a Rust function into a GPU kernel.
///
/// ```ignore
/// use unipute::kernel;
///
/// #[kernel(workgroup_size(64))]
/// fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
///     let index = global_id().x;
///     if index >= input.len() {
///         return;
///     }
///     output[index] = input[index] * factor;
/// }
/// ```
///
/// The function is replaced by a type with the same name, carrying the
/// generated shader for every target this build enables.
///
/// # Parameters
///
/// A `&[T]` parameter is a read only storage buffer, `&mut [T]` is a read and
/// write one, and anything else is a uniform. Each parameter takes the next
/// binding in group 0 unless `#[binding(group = 1, index = 3)]` says otherwise.
///
/// # What a body may contain
///
/// `let`, assignment, `if`, `while`, `loop`, `for` over a range, `break`,
/// `continue` and `return`. Values are scalars and vectors, the usual
/// operators, `as` casts, indexing, `.x` through `.w`, and `.len()` on a
/// slice. The built-ins are `global_id`, `local_id`, `local_index`,
/// `workgroup_id`, `num_workgroups`, `workgroup_barrier`, `storage_barrier`,
/// the `vec2` through `vec4` constructors, and the usual numeric functions.
///
/// # Workgroup memory
///
/// A `let` marked `#[workgroup]` at the top level of the body declares memory
/// shared by every invocation in a workgroup rather than a local of one
/// invocation. It takes a type and no value, and the type may be a fixed
/// length array:
///
/// ```ignore
/// #[kernel(workgroup_size(64))]
/// fn block_sum(input: &[f32], output: &mut [f32]) {
///     #[workgroup]
///     let tile: [f32; 64];
///
///     let lane = local_index();
///     tile[lane] = input[global_id().x];
///     workgroup_barrier();
///     if lane == 0u32 {
///         let mut total = 0.0;
///         for i in 0..tile.len() {
///             total += tile[i];
///         }
///         output[workgroup_id().x] = total;
///     }
/// }
/// ```
///
/// The host never sees it, so it has no binding and is not in `BINDINGS`.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as attr::KernelAttr);
    let function = parse_macro_input!(item as syn::ItemFn);

    match expand(&attr, &function) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(
    attr: &attr::KernelAttr,
    function: &syn::ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    let kernel = parse::kernel(attr, function)?;
    generate::kernel_type(function, &kernel)
}
