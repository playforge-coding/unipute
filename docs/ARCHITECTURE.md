# Architecture

This describes how Unipute is put together and where the pieces that are not
built yet will attach. It is written for anyone working on Unipute itself.

## The shape of it

```
front end            IR                 back end           output
---------            --                 --------           ------
#[kernel] macro  ->  unipute-ir     ->  unipute-naga   ->  WGSL, SPIR-V,
(unipute-macros)     Kernel             lower + write      MSL, HLSL, GLSL
```

Three properties follow from that shape, and every change should preserve
them.

**The IR has no dependencies.** `unipute-ir` is plain data: structs, enums,
`String` and `Vec`. It pulls in nothing, not a shader compiler and not a
graphics library. That is what lets a new front end or back end be added
without dragging the rest of the project along.

**Neither end knows about the other.** The macro never mentions naga. The naga
back end never mentions Rust syntax. They meet at `Kernel` and nowhere else.

**Nothing touches a device.** No crate here creates an instance, a queue or a
buffer. Output is text or a word buffer, and the caller decides what to do
with it. This is what "graphics API agnostic" means in practice, and it is why
the layout information in `BindingInfo` names groups and bindings rather than
any particular API's descriptor types.

## Crates

| Crate | Depends on | Purpose |
| ----- | ---------- | ------- |
| `unipute-ir` | nothing | `Kernel`, `Type`, `Expr`, `Stmt`, the `Backend` trait, `Target` |
| `unipute-macros` | `syn`, `quote`, `unipute-ir`, `unipute-naga` | the `#[kernel]` attribute |
| `unipute-naga` | `naga`, `unipute-ir` | lowering to naga IR and writing shaders |
| `unipute` | the three above | the crate users depend on |

`unipute-macros` depends on `unipute-naga` because the whole point is that the
back end runs during compilation. Cargo builds a proc macro for the host, so
the two copies of `unipute-naga` in a build (host and target) have their
features resolved separately. That is why the facade's target features name
both, as in `spv = ["unipute-macros/spv", "unipute-naga?/spv"]`.

## How a kernel becomes a shader

1. `unipute_macros::kernel` parses the attribute and the function with `syn`.
2. `parse::kernel` walks the signature into `Resource`s and the body into
   `Stmt`s, resolving names against a scope stack. Untyped integer literals
   take their type from the other side of the operator they appear in.
3. `unipute_naga::lower` builds a `naga::Module`. Two naga rules shape it:
   expressions must appear before their users, and most expressions have to
   sit inside a `Statement::Emit` range while a specific few must not. The
   ones that must not, which are literals, globals, locals and function
   arguments, are all created before the first emit range opens and looked up
   from a cache afterwards.
4. `unipute_naga::validate` runs naga's validator. A failure here means a bug
   in Unipute, not in the user's kernel, so the message is passed through
   as it is.
5. Each enabled target's writer runs, and `generate::kernel_type` pastes the
   result into the caller's crate as a constant.

The function is replaced by a type of the same name implementing `Kernel`,
plus one trait per enabled target. Keeping the shader on a trait rather than an
inherent constant means a caller can be generic over kernels.

## Extension points

Four things are designed for and not implemented. Each one already has a place
to attach.

### Other back ends, such as CUDA and PTX

**Attaches to:** `unipute_ir::Backend` and `unipute_ir::Target`.

`Backend` is deliberately vague about output. It has an associated `Output`
type, so a back end can produce a `String` of PTX, a word buffer, or machine
code, and callers written against the trait do not care which.

`Target::Ptx` is already in the enum and `Target::is_implemented` already
returns false for it, so the rest of the project can talk about a target before
the generator for it exists.

What a PTX back end needs:

- A new crate, `unipute-ptx`, depending on `unipute-ir` and nothing from naga.
- `impl Backend for PtxBackend`, mirroring `NagaBackend`.
- A feature on the facade and on `unipute-macros`, plus a `#[cfg]` block in
  `generate::compile_targets` and a `PtxKernel` trait.

What it does not need: any change to `unipute-ir` or `unipute-macros` beyond
that feature. If a PTX back end turns out to need an IR change, that is a
signal the IR has grown a naga assumption, and the fix belongs in the IR rather
than in the new back end.

The one real gap is that CUDA's memory model does not line up with bind groups.
`Resource` names a `group` and a `binding`, which a PTX back end would have to
flatten into kernel parameters. That mapping belongs in the back end, not in
the IR.

### Graphics stages

**Attaches to:** `unipute_ir::Stage`.

`Stage` already has `Vertex` and `Fragment` alongside `Compute`, and it is
`#[non_exhaustive]` so more can be added. `Stage::is_implemented` returns false
for both, and `lower` checks it first, so a graphics kernel fails with a clear
message today rather than a confusing one.

What graphics support needs:

- `Binding::Location` in the IR, since a vertex shader passes values to a
  fragment shader by location rather than by built-in. This is the one change
  that does touch the IR.
- A return type on `Kernel`, since a vertex shader produces a position and a
  fragment shader produces a colour. Today `check_signature` rejects any return
  type outright.
- The graphics built-ins: `Position`, `VertexIndex`, `InstanceIndex`,
  `FragDepth` and so on. `ir::BuiltIn` is a flat enum, so these are additions
  rather than restructuring.
- Matrix types in `ir::Type`, which is the other likely IR change.
- `#[kernel(vertex)]` and `#[kernel(fragment)]` in the attribute parser, in
  place of the `workgroup_size` that compute requires.

Naga handles all of this already, so `unipute-naga` is mostly a matter of
mapping the new variants across.

### Bindings for C, C++ and other languages

**Attaches to:** `unipute-ir`, from the other side.

There are two different things people mean by bindings, and they need different
work.

*Calling Unipute from C.* A host program written in C or Zig wants to hand
Unipute a kernel and get a shader back. That needs an `unipute-ffi` crate with
`extern "C"` functions over `Kernel`, built as a `cdylib` and `staticlib`, plus
a generated header. The natural surface is narrow: build a kernel, add
resources and statements, compile to a target, free the result. Because the IR
is plain data with no borrowed lifetimes inside it, this is an opaque pointer
API and not much more.

*Writing kernels in another language.* A Zig or C++ programmer wants to write
the kernel itself in their own language. That is a new front end, not an FFI
layer. It produces a `Kernel` the same way `unipute-macros` does, and from
there the pipeline is identical. Nothing in `unipute-ir` or `unipute-naga`
needs to change, which is the whole reason the IR has no dependencies.

The likely shape is a serialised form of `Kernel`, so a front end written in
any language can emit it as data and hand it to a small Rust driver. That
means adding optional `serde` support to `unipute-ir`, which is the one piece
of preparation worth doing before anyone starts.

### More of the Rust language in kernels

The front end covers a useful subset and rejects the rest with a message. The
pieces most likely to be wanted next, in rough order of value:

- User defined structs as buffer element types. Naga supports them and the IR
  needs a `Type::Struct` variant with explicit layout.
- Atomics, which naga has as `TypeInner::Atomic` and `Statement::Atomic`.
- Workgroup shared memory, which needs an address space on locals.
- Calling other functions, which means the IR growing a notion of a function
  besides the entry point.
- Textures and samplers, which mostly matter once graphics stages land.

## Testing

- `crates/unipute-naga/tests/lowering.rs` builds IR by hand and checks the
  generated shaders. This is where lowering bugs surface.
- `tests/kernels.rs` goes from Rust source through the macro to shader text for
  every target, and checks that the compile time and run time paths agree.
- `tests/ui/` pins the error messages, since those are most of what a macro's
  users experience. Run with `TRYBUILD=overwrite` to refresh them after
  changing a message on purpose.

When adding a back end or a stage, the first test to write is a lowering test,
because it fails in a readable way. The macro tests are for the front end.
