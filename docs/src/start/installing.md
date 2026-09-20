# Installing

Add Unipute to your crate:

```toml
[dependencies]
unipute = "0.1"
```

That gives you WGSL. Every other language is a feature, and you can turn on as
many as you like:

```toml
[dependencies]
unipute = { version = "0.1", features = ["spv", "msl"] }
```

| Feature | Language |
| ------- | -------- |
| `wgsl`  | WebGPU Shading Language, on by default |
| `spv`   | SPIR-V |
| `msl`   | Metal Shading Language |
| `hlsl`  | High Level Shading Language |
| `glsl`  | OpenGL Shading Language |

[Choosing targets](../output/targets.md) explains what each one gives you and
how to pick.

## What gets pulled in

Unipute depends on [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga), the
shader translator from the wgpu project, and on `syn` and `quote` for reading
Rust syntax. It does not depend on wgpu itself, or on any other graphics
library.

Both naga and the macro crate are compiled during a clean build, which takes a
little while the first time. If you build Unipute often, a compilation cache
is worth having:

```bash
cargo install sccache
```

Then add this to `~/.cargo/config.toml`:

```toml
[build]
rustc-wrapper = "sccache"
```

## Build time

Because the shaders are generated while your crate compiles, kernels cost
compile time rather than run time. A handful of kernels is not noticeable. If
you have hundreds, turning off the targets you are not using will help, since
each enabled target runs its own writer over every kernel.

## Rust version

Unipute uses the 2024 edition. Any reasonably current stable toolchain works,
and there is nothing nightly only in it.

You may notice that Unipute's own repository pins nightly. That is a build speed
choice for people working on Unipute, not something that reaches you. It turns
on the cranelift codegen backend for dev builds, which is still unstable, and it
lives in that repository's cargo config rather than in the published manifest so
that it stays there. Every part of Unipute you touch, the `#[kernel]` macro, the
traits, the layout types and the runtime API, is stable Rust. CI builds a crate
against Unipute on stable on every change to keep that true.

## Checking it worked

Put this in `src/main.rs`:

```rust
use unipute::{Kernel, kernel};

#[kernel(workgroup_size(64))]
fn nothing_much(output: &mut [f32]) {
    output[0] = 1.0;
}

fn main() {
    println!("{} with workgroup {:?}", nothing_much::NAME, nothing_much::WORKGROUP_SIZE);
}
```

```bash
cargo run
```

```text
nothing_much with workgroup [64, 1, 1]
```

If that runs, you are set up. On to [your first real
kernel](./first-kernel.md).
