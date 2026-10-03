# What is not built yet

This is version 0.1. Rather than let you find the edges by walking into them,
here is the list.

## What does work

Compute kernels, end to end, across WGSL, SPIR-V, MSL, HLSL and GLSL. Scalars,
vectors and their [swizzles](./kernels/types.md#swizzles), [your own
structs](./kernels/structs.md), storage buffers and
uniforms, the control flow in [Control flow](./kernels/control-flow.md), the
functions in [Built-ins and functions](./kernels/builtins.md), the nested
functions in [Your own functions](./kernels/functions.md), [workgroup
memory](./kernels/workgroup-memory.md) and barriers,
[atomics](./kernels/atomics.md), and binding layout you can read from the
host.

That part is tested and is what the rest of this book describes.

## Sharing a function between kernels

A helper declared inside one kernel belongs to that kernel. Two kernels wanting
the same helper each declare their own copy.

Fixing this needs a surface `#[kernel]` does not have, because a macro attached
to one function cannot read another one. The likely shape is a module level
macro wrapping several kernels and their shared helpers at once.

## The rest of structs

A struct's fields are scalars and vectors. A struct holding another struct, or
an array, is refused by the derive. The IR and the naga back end can already
represent both, so the missing piece is the derive working out the host
layout for them, and the uniform rules, which put stricter alignment on a
nested struct than a storage buffer does.

A struct has to be in the same crate as the kernels that name it. The kernel
macro learns a struct's fields through a `macro_rules!` the derive leaves next
to it, and that is the one kind of item Rust will not let a crate re-export by
path. A struct from a dependency would need another way of handing the fields
over, which is the same question the serialised IR under [Other
languages](#other-languages) answers.

## The rest of atomics

An atomic is a `u32` or an `i32` in a buffer or in workgroup memory. Three
things are missing around that.

An atomic as a struct field. The IR and the back end allow it. What is not
worked out is the host side, where the field would be a `u32` to one reader
and an `AtomicU32` to another, and how the derive should spell that.

Float atomics. Naga has `atomicAdd` on an `f32` behind a capability that only
some hardware offers. Turning it on would mean a kernel that compiles for one
device and not another, which needs a story about capabilities first.

An old value from `compare_exchange`. It answers whether the exchange
happened. The old value is there in the IR, but a target may fail the
exchange with the values matching, so handing the old value back on its own
would invite a loop that never ends. The two together are Rust's
`Result<T, T>`, which the kernel language has no way to spell yet.

## Matrices

No matrix types. Multiply vectors by hand, or pass the elements separately.

## Textures and samplers

Not available. These mostly matter once graphics stages exist, so the two are
likely to arrive together.

## Graphics stages

Compute only. `Stage::Vertex` and `Stage::Fragment` exist in the IR and are
rejected with a clear message rather than a confusing one, which is the extent
of the support today.

This is the biggest single piece of work, because it needs location based
bindings, return types on kernels, the graphics built-ins and matrices. The
[architecture notes](./architecture.md) break it down.

## Backends other than naga

Naga gives five languages. Anything naga does not target, meaning CUDA and
PTX in particular, needs its own backend.

`Target::Ptx` is reserved and the `Backend` trait is deliberately vague about
what a backend produces, so one can be added without disturbing the naga path.
CUDA's memory model does not have bind groups, so that mapping would live in
the new backend rather than in the IR.

## Other languages

Two different things people mean by this.

*Calling Unipute from C or Zig*, to generate shaders from a host program
written in something other than Rust. That needs a C ABI layer over the IR.
Because the IR is plain data with no borrowed lifetimes, this is an opaque
pointer API and not much more.

*Writing kernels in C or Zig.* That is a new front end rather than a binding
layer. It produces IR the same way the Rust macro does, and everything
downstream is unchanged. The likely first step is a serialised form of the IR
so a front end in any language can emit it as data.

Neither exists. The IR being dependency free is the groundwork for both.

## Running kernels on the CPU

Your `#[kernel]` function stops being callable from Rust. Being able to run a
kernel on the CPU, for testing without a GPU, would be genuinely useful and is
not there.

## How to read this list

Everything here is a gap rather than a decision against. The one with the
clearest path is matrices, since naga already has them and a matrix that
stays inside a kernel does not touch the host. One in a buffer does, because
the host has to lay out its columns the way the shader reads them.

If one of these is blocking you, saying so is useful. It is easier to
prioritise against a real kernel someone is trying to write than against a
guess.
