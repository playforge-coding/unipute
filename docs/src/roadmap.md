# What is not built yet

This is version 0.1. Rather than let you find the edges by walking into them,
here is the list.

## What does work

Compute kernels, end to end, across WGSL, SPIR-V, MSL, HLSL and GLSL. Scalars
and vectors, storage buffers and uniforms, the control flow in [Control
flow](./kernels/control-flow.md), the functions in [Built-ins and
functions](./kernels/builtins.md), barriers, and binding layout you can read
from the host.

That part is tested and is what the rest of this book describes.

## Calling your own functions

**The gap most likely to affect you.** A kernel body cannot call another
function, so everything has to be inline. A kernel that would naturally be
three helpers has to be written as one block.

Naga supports functions, and the IR would need a notion of a function
alongside the entry point. Nothing about this is hard, it just is not done.

## Structs in buffers

Buffers hold scalars and vectors. A buffer of your own struct type is not
available, so a struct of three fields means three parallel buffers.

Naga supports structs. The IR needs a `Type::Struct` variant carrying explicit
layout, since the host and the shader have to agree byte for byte, and getting
that wrong silently is worse than not offering it.

## Atomics

No atomic operations, so no counters shared across invocations and no lock
free algorithms. Naga has these as a type and a statement, so it is mostly a
matter of surfacing them.

## Workgroup shared memory

No way to declare memory shared within a workgroup. `workgroup_barrier()`
exists, which is a little odd given there is not yet much to synchronise, but
it is there because storage buffer synchronisation does work.

This needs an address space on locals. It matters for anything with a
reduction in it, so it is high on the list.

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

## Swizzles

`color.xy` does not work. Write `vec2(color.x, color.y)`.

## How to read this list

Everything here is a gap rather than a decision against. The ones with the
clearest path are calling your own functions, structs, atomics and workgroup
memory, since naga already supports all four.

If one of these is blocking you, saying so is useful. It is easier to
prioritise against a real kernel someone is trying to write than against a
guess.
