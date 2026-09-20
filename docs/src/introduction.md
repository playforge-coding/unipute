# Introduction

Unipute lets you write a GPU compute kernel as an ordinary looking Rust
function. When your crate compiles, the shader is generated and pasted back
into your code as a constant.

```rust
use unipute::{Kernel, WgslKernel, kernel};

#[kernel(workgroup_size(64))]
fn double(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index] * 2.0;
    }
}

fn main() {
    // This string already exists by the time the program starts.
    println!("{}", double::WGSL);
}
```

That is the whole idea. You write Rust, you get WGSL, SPIR-V, MSL, HLSL or
GLSL, and none of it happens while your program is running.

## What Unipute is not

It is not a graphics library. Unipute never creates a device, a queue, a
buffer or a pipeline. It hands you shader source or a buffer of SPIR-V words
and stops there. You keep using wgpu, ash, Metal, OpenGL or whatever else you
already picked, and Unipute sits upstream of that choice.

This sounds like a limitation and is mostly the opposite. Because Unipute
never touches an API, it never locks you into one, and adding a target later
does not mean rewriting your kernels.

## Why generate at compile time

Most ways of getting a shader onto a GPU hand a string to a driver at run
time. If the shader is wrong, you find out when the program runs, and the
error points into a string rather than into your code.

Unipute runs the whole pipeline during compilation instead, so a broken kernel
is a compile error in your own crate:

```text
error: `frobnicate` is not a built-in Unipute provides
 --> src/main.rs:5:17
  |
5 |     output[0] = frobnicate(1.0);
  |                 ^^^^^^^^^^
```

You also get the layout information as data, so the host side of your code can
read the binding numbers from the kernel rather than you keeping two lists in
sync by hand.

## How the pieces fit

There are three stages, and they do not know about each other:

```text
your Rust function  ->  Unipute IR  ->  shader text or SPIR-V
```

The middle stage is a small, backend agnostic description of what the kernel
does. The front end turns Rust into it, and a back end turns it into a
language. Neither end depends on the other, which is what makes it possible to
add a language, or eventually a whole different kind of target like CUDA,
without touching kernels that already work.

[How it works](./how-it-works.md) goes through this in more detail, once the
rest of the book has given it some context.

## Where to go next

If you want to try it, start with [Installing](./start/installing.md) and then
[Your first kernel](./start/first-kernel.md).

If you want to know what a kernel is allowed to contain, the four chapters
under "Writing kernels" are the reference, and
[Control flow](./kernels/control-flow.md) is probably the one with the most
surprises in it.

If you are wondering whether Unipute can do the thing you need,
[What is not built yet](./roadmap.md) is an honest list.

## A note on maturity

This is version 0.1. Compute kernels work end to end across five shader
languages, and that part is tested. Graphics stages, backends other than naga,
and bindings for languages other than Rust are designed for and not built. The
roadmap chapter says what each one would take.
