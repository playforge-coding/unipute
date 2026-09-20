# Choosing targets

Each shader language is a cargo feature. Turning one on adds a constant to
every kernel in your crate.

| Feature | Trait | Constant | For |
| ------- | ----- | -------- | --- |
| `wgsl` | `WgslKernel` | `WGSL: &str` | wgpu, WebGPU, Dawn |
| `spv` | `SpirvKernel` | `SPIRV: &[u32]` | Vulkan, OpenCL, and as an input to other tools |
| `msl` | `MslKernel` | `MSL: &str` | Metal, so Apple platforms |
| `hlsl` | `HlslKernel` | `HLSL: &str` | Direct3D 12 |
| `glsl` | `GlslKernel` | `GLSL: &str` | OpenGL and OpenGL ES |

`wgsl` is on by default. Turn it off with `default-features = false` if you
do not want it.

```toml
[dependencies]
unipute = { version = "0.1", features = ["spv", "msl"] }
```

## Getting at the constant

Import the trait for the target you want, then read the constant off the
kernel type:

```rust
use unipute::{WgslKernel, kernel};

#[kernel(workgroup_size(64))]
fn double(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index] * 2.0;
    }
}

fn main() {
    let source: &str = double::WGSL;
    println!("{} bytes of WGSL", source.len());
}
```

Forgetting the import gives you a "cannot find" error, which is the usual Rust
trait import problem rather than anything specific to Unipute.

## Picking more than one

There is no cost to your program in enabling several. The strings all exist as
constants and the ones you never read get dropped by the linker. What you pay
is compile time, since each target runs its own writer over every kernel.

Supporting several backends usually looks like this:

```rust,ignore
#[cfg(target_vendor = "apple")]
fn shader() -> &'static str {
    use unipute::MslKernel;
    double::MSL
}

#[cfg(not(target_vendor = "apple"))]
fn shader() -> &'static str {
    use unipute::WgslKernel;
    double::WGSL
}
```

If you do not know the target until the program runs, see [Generating at run
time](./runtime.md).

## Notes per target

### WGSL

The best supported path, and the one the tests exercise most. If you are using
wgpu, this is what you want, and you can hand the string straight to
`create_shader_module`.

### SPIR-V

A `&[u32]`, not bytes. Vulkan wants words, so this is the right shape already.
If you need bytes, for writing to a file say, convert with
`to_le_bytes`.

The first word is the SPIR-V magic number `0x07230203`, which is a quick way
to check you have a valid module.

### MSL

Metal has no bind groups, so naga assigns buffer slots itself rather than
using the group and binding numbers. Read the slots from Metal's reflection
rather than assuming they match `BINDINGS`. This is the one target where the
layout information does not carry across directly.

### HLSL

Targets Shader Model 6 by default. Groups and bindings become `register`
assignments.

### GLSL

Targets OpenGL ES 3.10, which is the version with compute shaders. Bindings
become `layout(binding = ...)`.

## Seeing all of them

The repository has an example that prints every enabled target for one kernel:

```bash
cargo run --example emit_shaders --features "spv,msl,hlsl,glsl"
```

That is the fastest way to see what a kernel actually becomes, and a good
sanity check when something is not behaving. [Examples](../examples.md) lists
the others that come with the repository.
