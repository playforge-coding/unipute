# Running a kernel

Unipute stops at the shader. Getting it onto a GPU is your graphics API's job.
This chapter shows how the two meet.

> The code here is sketched against wgpu to show the shape of it. Graphics
> APIs change their signatures often, so treat it as a guide to which pieces
> line up rather than something to paste. The Unipute side, which is the part
> this book can promise, is the constants.

## What a kernel gives the host

Three things:

```rust
# use unipute::{Kernel, WgslKernel, kernel};
# #[kernel(workgroup_size(64))]
# fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
#     let index = global_id().x;
#     if index >= input.len() { return; }
#     output[index] = input[index] * factor;
# }
# fn main() {
scale::WGSL;            // the shader
scale::NAME;            // the entry point name inside it
scale::WORKGROUP_SIZE;  // how many invocations one group runs
scale::BINDINGS;        // where each buffer goes
# }
```

Those four cover everything the host has to know.

## Creating the pipeline

```rust,ignore
let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some(scale::NAME),
    source: wgpu::ShaderSource::Wgsl(scale::WGSL.into()),
});

let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some(scale::NAME),
    layout: None,
    module: &module,
    entry_point: Some(scale::NAME),
    compilation_options: Default::default(),
    cache: None,
});
```

Using `scale::NAME` rather than the string `"scale"` means that renaming the
kernel, or setting `name = "..."` on the attribute, does not leave the host
side pointing at an entry point that no longer exists.

## Building the bind group layout

This is where `BINDINGS` earns its place. Instead of writing the layout by
hand and keeping it in step with the kernel, build it from the kernel:

```rust,ignore
use unipute::Access;

let entries: Vec<_> = scale::BINDINGS
    .iter()
    .map(|binding| wgpu::BindGroupLayoutEntry {
        binding: binding.binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: match binding.access {
                Access::Uniform => wgpu::BufferBindingType::Uniform,
                Access::Read => wgpu::BufferBindingType::Storage { read_only: true },
                Access::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    })
    .collect();
```

Add a parameter to the kernel and this follows along. That is the difference
between a layout that is derived and one that is duplicated.

`binding.group` matters once you use more than one group. Group the entries by
it and create one layout per group.

## Working out the dispatch size

`WORKGROUP_SIZE` tells you how many invocations one workgroup runs, so the
number of workgroups is your element count divided by that, rounded up:

```rust
# use unipute::{Kernel, kernel};
# #[kernel(workgroup_size(64))]
# fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
#     let index = global_id().x;
#     if index >= input.len() { return; }
#     output[index] = input[index] * factor;
# }
/// Number of workgroups needed to cover `elements` items.
fn groups_for(elements: u32, workgroup: u32) -> u32 {
    elements.div_ceil(workgroup)
}

fn main() {
    let elements = 100u32;
    let groups = groups_for(elements, scale::WORKGROUP_SIZE[0]);
    assert_eq!(groups, 2); // 2 groups of 64 covers 100
}
```

Rounding up is why your kernel needs the bounds check. Two groups of 64 is 128
invocations for 100 elements, and the last 28 have to notice and stop. See
[Your first kernel](../start/first-kernel.md) if that check is not yet a
reflex.

Then dispatch:

```rust,ignore
pass.set_pipeline(&pipeline);
pass.set_bind_group(0, &bind_group, &[]);
pass.dispatch_workgroups(groups, 1, 1);
```

For two dimensional work, use `WORKGROUP_SIZE[1]` for the second axis the same
way.

## Other APIs

The pieces are the same everywhere, only the spelling changes.

**Vulkan.** Enable `spv` and hand `SPIRV` to `vkCreateShaderModule` as words.
`BINDINGS` maps onto descriptor set layout bindings, with `group` as the
descriptor set number.

**Metal.** Enable `msl` and compile `MSL` with
`newLibraryWithSource`. Metal assigns its own buffer slots rather than using
group and binding numbers, so read them from Metal's reflection instead of
`BINDINGS`.

**Direct3D 12.** Enable `hlsl` and compile `HLSL` with DXC. Bindings appear as
`register` assignments.

**OpenGL.** Enable `glsl` and compile `GLSL` as a compute shader. Bindings
appear as `layout(binding = ...)`.

## Compiling shaders ahead of time

Nothing stops you writing the generated shaders out to files as part of a
build step, since they are just constants:

```rust,ignore
std::fs::write("shaders/scale.wgsl", scale::WGSL)?;
```

This is useful for shipping precompiled pipelines, or for feeding SPIR-V
through an optimiser before it reaches the driver.
