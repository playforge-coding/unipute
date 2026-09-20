# Unipute

[![CI](https://github.com/playforge-coding/unipute/actions/workflows/ci.yml/badge.svg)](https://github.com/playforge-coding/unipute/actions/workflows/ci.yml)
[![Documentation](https://github.com/playforge-coding/unipute/actions/workflows/docs.yml/badge.svg)](https://github.com/playforge-coding/unipute/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

A compile time generating, language and graphics API agnostic GPU library for
Rust.

**[Read the guide](https://playforge-coding.github.io/unipute/)**

Write a compute kernel as an ordinary looking Rust function, put `#[kernel]` on
it, and the shader is generated while your crate compiles. Unipute brings no
graphics library with it. It produces shader source and SPIR-V words, and
leaves loading them to whatever API you already use.

```rust
use unipute::{Kernel, WgslKernel, kernel};

#[kernel(workgroup_size(64))]
fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index] * factor;
}

fn main() {
    // Generated during compilation, not at run time.
    println!("{}", scale::WGSL);
    println!("{:?}", scale::WORKGROUP_SIZE);

    for binding in scale::BINDINGS {
        println!("group {} binding {}", binding.group, binding.binding);
    }
}
```

That kernel comes out as:

```wgsl
@group(0) @binding(0)
var<storage> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;
@group(0) @binding(2)
var<uniform> factor: f32;

@compute @workgroup_size(64, 1, 1)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index: u32;

    index = global_id.x;
    let _e6 = index;
    if (_e6 >= arrayLength((&input))) {
        return;
    }
    let _e9 = index;
    let _e11 = index;
    let _e13 = input[_e11];
    let _e14 = factor;
    output[_e9] = (_e13 * _e14);
}
```

## Why

Most ways of getting a shader onto a GPU make you pick a language up front, and
often a graphics library with it. Unipute splits those apart. A kernel is
written once as Rust, becomes a small backend agnostic IR, and each target is
generated from that IR. Adding a target does not change your kernel, and
nothing in the pipeline depends on a device, a queue or a buffer.

Generating at compile time also means a broken kernel is a compile error in
your own crate, pointing at the line that caused it, rather than a runtime
failure from a shader compiler.

## Install

```toml
[dependencies]
unipute = "0.1"
```

## Targets

Each target is a cargo feature. Turning one on adds a constant to every kernel
in your crate. `wgsl` is on by default.

| Feature | Trait         | Constant           |
| ------- | ------------- | ------------------ |
| `wgsl`  | `WgslKernel`  | `WGSL: &str`       |
| `spv`   | `SpirvKernel` | `SPIRV: &[u32]`    |
| `msl`   | `MslKernel`   | `MSL: &str`        |
| `hlsl`  | `HlslKernel`  | `HLSL: &str`       |
| `glsl`  | `GlslKernel`  | `GLSL: &str`       |

```toml
unipute = { version = "0.1", features = ["spv", "msl"] }
```

To see them all for one kernel:

```bash
cargo run --example emit_shaders --features "spv,msl,hlsl,glsl"
```

## Writing a kernel

### Parameters

A parameter's type decides how it is bound.

| Rust type   | Becomes                        |
| ----------- | ------------------------------ |
| `&[T]`      | read only storage buffer       |
| `&mut [T]`  | read and write storage buffer  |
| `&T`, `T`   | uniform                        |

Parameters take consecutive bindings in group 0. Override that per parameter:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn placed(
    #[binding(group = 1, index = 4)] input: &[u32],
    #[binding(group = 1, index = 5)] output: &mut [u32],
) {
    output[0] = input[0];
}
```

### Types

Scalars are `f32`, `u32`, `i32` and `bool`. Vectors are `Vec2<T>`, `Vec3<T>`
and `Vec4<T>`, built with `vec2`, `vec3` and `vec4`, and read with `.x` through
`.w`.

### Control flow and operators

`let`, assignment and compound assignment, `if` and `else`, `while`, `loop`,
`for` over a range, `break`, `continue` and `return`. The usual arithmetic,
comparison, bitwise and shift operators, `as` casts, and indexing.

### Built-ins

`global_id()`, `local_id()`, `workgroup_id()` and `num_workgroups()` give a
`Vec3<u32>`, and `local_index()` gives a `u32`. `workgroup_barrier()` and
`storage_barrier()` synchronise. `.len()` on a slice parameter gives its length
at dispatch time.

The numeric functions are `abs`, `min`, `max`, `clamp`, `floor`, `ceil`,
`round`, `sqrt`, `inverse_sqrt`, `exp`, `log`, `pow`, `sin`, `cos`, `tan`,
`sign`, `fma`, `mix`, `step`, `dot`, `cross`, `length` and `normalize`.

## Generating at run time

Sometimes the target is not known until the program runs, such as in a tool
that picks a language from a command line flag. Turn on the `runtime` feature
and go through the IR:

```rust,ignore
use unipute::{Kernel, Target};

let wgsl = unipute::compile_text(&scale::ir(), Target::Wgsl)?;
let msl = unipute::compile_text(&scale::ir(), Target::Msl)?;
```

`Kernel::ir()` is available without the `runtime` feature too, so you can
inspect a kernel even in a build that only generates ahead of time.

## How it fits together

| Crate            | Does                                                      |
| ---------------- | --------------------------------------------------------- |
| `unipute`        | what you depend on: the macro, the traits and the layout types |
| `unipute-ir`     | the IR, plain data with no dependencies                   |
| `unipute-macros` | the Rust front end, which runs the back end at compile time |
| `unipute-naga`   | the naga back end, which writes the five shader languages |

The IR in the middle is what keeps the ends independent. A front end for
another language and a back end for another platform both plug into it without
either knowing about the other. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the extension points and what is planned next.

## Status

This is an MVP. Compute kernels work end to end across all five targets.
Graphics stages, other back ends such as CUDA and PTX, and bindings for C and
other languages are designed for but not implemented. `docs/ARCHITECTURE.md`
says what each one needs.

## Building

```bash
cargo test --workspace --features "spv,msl,hlsl,glsl,runtime"
cargo fmt --all && cargo clippy --workspace --all-targets
```

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs three jobs on every
push and pull request:

- **Test** on Linux, macOS and Windows with every target enabled
- **Format, clippy and docs**, with clippy and rustdoc both at `-D warnings`
- **Feature combinations**, checking that all nine of them build warning free

That last one is worth knowing about. Every target is optional and several
items sit behind a `cfg`, so a combination nobody builds locally is exactly
where an unused import hides. If you add a feature, add it to the list in that
job.

The UI tests in [tests/ui/](tests/ui/) compare against rustc's exact error
output, so they run on Linux only. Set `UNIPUTE_SKIP_UI_TESTS=1` to skip them
locally, and run `TRYBUILD=overwrite cargo test --test ui` to regenerate the
expected output after deliberately changing a message.

The repository turns on [sccache](https://github.com/mozilla/sccache) in
`.cargo/config.toml`, since a clean build compiles naga and a proc macro crate.
If you do not have it, either `cargo install sccache` or prefix a command with
`RUSTC_WRAPPER=` to turn it off.

### The guide

The guide in [docs/](docs/) is an mdBook. To work on it:

```bash
cargo install mdbook
mdbook serve docs --open
```

Every Rust example in it is a doctest of the `unipute` crate, wired up in
[src/lib.rs](src/lib.rs), so `cargo test --doc` is what checks them rather
than `mdbook test`. Going through cargo means the examples compile against
this exact build with the right features on.

When you add a chapter, list it in `docs/src/SUMMARY.md` and, if it has Rust
examples, add it to the `guide` module in `src/lib.rs`.

Pushing to `main` builds the book and publishes it to GitHub Pages through
[.github/workflows/docs.yml](.github/workflows/docs.yml). Pull requests build
it without publishing. The repository's Pages source has to be set to "GitHub
Actions" once, under Settings then Pages, for the deployment to land.

## License

Dual licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
