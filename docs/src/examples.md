# Examples

The repository has a set of runnable examples in
[`examples/`](https://github.com/playforge-coding/unipute/tree/main/examples).
Each one is a single file, and each one is about a different part of the job,
so between them they cover most of what this book describes.

None of them need a GPU. Unipute stops at the shader, so an example prints what
it generated rather than running it.

## emit_shaders

One small kernel, every target it was built with.

```bash
cargo run --example emit_shaders --features "spv,msl,hlsl,glsl"
```

The fastest way to see what a kernel actually becomes, and a good first check
when a shader is not behaving.

## image_blur

A two dimensional kernel, and the arithmetic a host does around one.

```bash
cargo run --example image_blur
```

A 3x3 blur over a flat image buffer, with taps clamped to the edge. Around it,
the two things a host has to get right: the layout, read out of `BINDINGS`
rather than written a second time by hand, and the workgroup count for a
1920x1080 picture, including how many invocations land outside it and return
early. See [Running a kernel](./output/running.md) for the same arithmetic in
prose.

## prefix_sum

An algorithm that does not fit in one dispatch.

```bash
cargo run --example prefix_sum
```

A prefix sum needs every element to know the sum of the ones before it, which
no single pass can do. This is three kernels and fourteen dispatches, ping
ponging between two buffers. It also shows what a second bind group is for: the
step size is the only thing that changes between passes, so it is bound on its
own and group 0 is bound once.

## nbody

Vector types from one end to the other.

```bash
cargo run --example nbody
```

One step of a direct n-body simulation. Buffers of `Vec4<f32>`, a helper that
takes vectors and returns one, and `dot` and `inverse_sqrt` doing the work.
Packing position and mass into one `Vec4` is the sort of thing that matters in
a kernel that reads every body once per invocation.

## inspect_ir

Reading a kernel back as data.

```bash
cargo run --example inspect_ir
```

A Mandelbrot kernel, printed back out of its own IR and then counted:
statements, branches, loop nesting, which built-ins and which maths functions
it reaches for. It is a walk over `Kernel::ir()`, which is the same door a new
back end, a linter or an editor tooltip would come in through. Nothing here
needs a target feature, since nothing here generates a shader.

## retarget

Choosing a target while the program runs.

```bash
cargo run --example retarget --features "runtime,msl,spv" -- msl spirv
```

A small command line tool over the `runtime` feature. Run it with no arguments
and it prints what this build can produce, which is not the same question as
what Unipute can produce. At the end it regenerates WGSL at run time and
compares it to the constant the macro baked in, which is the claim
[Generating at run time](./output/runtime.md) makes, checked in front of you.
