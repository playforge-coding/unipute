# Examples

The repository has a set of runnable examples in
[`examples/`](https://github.com/playforge-coding/unipute/tree/main/examples).
Each one is about a different part of the job, so between them they cover most
of what this book describes.

Three of them run their kernels on a GPU and check the answer. Unipute stops at
the shader, so the wgpu code that takes it from there lives in
[`examples/host/mod.rs`](https://github.com/playforge-coding/unipute/blob/main/examples/host/mod.rs),
shared by those examples and by the GPU tests. It is the [Running a
kernel](./output/running.md) chapter written out in full, and it is not part
of the library: wgpu is a dev dependency of the repository, not of Unipute. An
example that needs a device says so and exits if the machine has none.

The other three never touch a device. They print what a kernel becomes.

## emit_shaders

One small kernel, every target it was built with.

```bash
cargo run --example emit_shaders --features "spv,msl,hlsl,glsl"
```

The fastest way to see what a kernel actually becomes, and a good first check
when a shader is not behaving.

## image_blur

A two dimensional kernel, run over a picture you can see.

```bash
cargo run --example image_blur
```

A 3x3 blur over a flat image buffer, with taps clamped to the edge. The host
works out the workgroup count for a 40x12 picture, including how many
invocations land outside it and return early, runs the blur, prints the
picture before and after as characters, and checks the result against the
same filter written as two plain loops. See [Running a
kernel](./output/running.md) for the dispatch arithmetic in prose.

## prefix_sum

An algorithm that does not fit in one dispatch.

```bash
cargo run --example prefix_sum
```

A prefix sum needs every element to know the sum of the ones before it, which
no single pass can do. This is three kernels and fourteen dispatches, ping
ponging between two buffers, with the three results compared against a
running total on the CPU at the end. It also shows what a second bind group
is for: the step size is the only thing that changes between passes, so it is
bound on its own and group 0 is bound once. The step takes binding index 2
rather than 0, which is the [GLSL rule](./output/targets.md#glsl) at work.

## nbody

Vector types from one end to the other.

```bash
cargo run --example nbody
```

A direct n-body simulation, run for twenty steps. Buffers of `Vec4<f32>`, a
helper that takes vectors and returns one, and `dot` and `inverse_sqrt` doing
the work. The first step is checked against the same maths on the CPU, and the
total momentum is printed as the simulation goes, since every pull has an
equal and opposite one and the physics says it should not change. Packing
position and mass into one `Vec4` is the sort of thing that matters in a
kernel that reads every body once per invocation.

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
