//! The `#[kernel]` attribute macro and the `Layout` derive.
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
mod chain;
mod emit;
mod generate;
mod layout;
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
/// `T` is a scalar, a vector, or a struct with [`Layout`](macro@Layout)
/// derived on it. A `&mut [AtomicU32]` or `&mut [AtomicI32]` is a buffer of
/// atomics, reached with `.load()`, `.store()`, `.compare_exchange()` and the
/// `fetch_` methods.
///
/// # What a body may contain
///
/// `let`, assignment, `if`, `while`, `loop`, `for` over a range, `break`,
/// `continue` and `return`. Values are scalars, vectors and structs, the
/// usual operators, `as` casts, indexing, `.x` through `.w` on a vector,
/// `.name` on a struct, a struct literal, and `.len()` on a slice. The
/// built-ins are `global_id`, `local_id`, `local_index`, `workgroup_id`,
/// `num_workgroups`, `workgroup_barrier`, `storage_barrier`, the `vec2`
/// through `vec4` constructors, and the usual numeric functions.
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
    let attr_tokens = proc_macro2::TokenStream::from(attr.clone());
    let attr = parse_macro_input!(attr as attr::KernelAttr);
    let function = parse_macro_input!(item as syn::ItemFn);

    // A kernel that names a struct cannot be expanded yet: the struct's
    // fields live on another item, and the chain in `chain.rs` goes and
    // fetches them before coming back here through `__kernel_with_layouts`.
    let step = chain::Step::first(attr_tokens, function);
    if let Some(ask) = step.ask_next() {
        return ask.into();
    }

    match expand(&attr, &step.function, &[]) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// The second half of `#[kernel]`, reached through the macro that
/// `#[derive(Layout)]` leaves on a struct. Not for calling by hand.
#[doc(hidden)]
#[proc_macro]
pub fn __kernel_with_layouts(input: TokenStream) -> TokenStream {
    let step = parse_macro_input!(input as chain::Step);
    if let Some(ask) = step.ask_next() {
        return ask.into();
    }

    let result = syn::parse2::<attr::KernelAttr>(step.attr.clone())
        .and_then(|attr| Ok((attr, step.structs()?)))
        .and_then(|(attr, structs)| expand(&attr, &step.function, &structs));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Lets a kernel put a struct in a buffer.
///
/// ```ignore
/// use unipute::{Layout, kernel};
///
/// #[derive(Layout, Clone, Copy)]
/// #[repr(C)]
/// struct Particle {
///     position: [f32; 3],
///     mass: f32,
/// }
///
/// #[kernel(workgroup_size(64))]
/// fn heavier(particles: &mut [Particle]) {
///     let index = global_id().x;
///     if index < particles.len() {
///         particles[index].mass = particles[index].mass * 2.0;
///     }
/// }
/// ```
///
/// The struct has to be `#[repr(C)]` with named fields, each an `f32`, `u32`
/// or `i32` or an array of two to four of them, which the kernel sees as a
/// vector. A field whose name starts with an underscore is padding of type
/// `[u8; N]`, there to make the Rust layout match the GPU one. When the two
/// differ, the derive says which field to pad and by how much.
///
/// The struct and the kernels using it have to be in the same crate.
#[proc_macro_derive(Layout)]
pub fn derive_layout(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    match layout::derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(
    attr: &attr::KernelAttr,
    function: &syn::ItemFn,
    structs: &[unipute_ir::StructType],
) -> syn::Result<proc_macro2::TokenStream> {
    let kernel = parse::kernel(attr, function, structs)?;
    generate::kernel_type(function, &kernel)
}
