# Parameters and bindings

A kernel's parameters are the buffers the host binds before dispatching it.
The type you write decides what kind of binding you get.

| You write | You get | Kernel can |
| --------- | ------- | ---------- |
| `&[T]` | read only storage buffer | read |
| `&mut [T]` | read and write storage buffer | read and write |
| `&T` | uniform | read |
| `T` | uniform | read |

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn blend(
    source: &[f32],      // read only buffer
    target: &mut [f32],  // buffer we write into
    amount: &f32,        // a single number, the same for every invocation
) {
    let index = global_id().x;
    if index < source.len() {
        target[index] = mix(target[index], source[index], amount);
    }
}
```

## Storage buffers

A slice parameter is a storage buffer. Its length is not fixed at compile
time, which is why `.len()` works and why the host decides how much data to
bind.

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn sum_lengths(a: &[f32], b: &[f32], output: &mut [u32]) {
    output[0] = a.len() + b.len();
}
```

`.len()` gives a `u32`, not a `usize`, because there is no `usize` on a GPU.
It counts elements, not bytes.

Take a buffer by `&mut` only when you write to it. A read only buffer tells
the driver more about what you are doing, and some APIs place the two in
different descriptor types.

## Uniforms

A parameter that is not a slice is a uniform: one value, the same for every
invocation in the dispatch. Use one for anything the host decides per
dispatch, such as a scale factor, a time value or a grid size.

```rust
# use unipute::kernel;
#[kernel(workgroup_size(8, 8))]
fn fill_grid(output: &mut [f32], width: &u32) {
    let cell = global_id();
    output[cell.y * width + cell.x] = 1.0;
}
```

Using a uniform by name reads its value. There is no dereference step, and
`*factor` is not something a kernel can say.

You cannot write to a uniform. Trying gives you:

```text
error: `width` is a uniform, it cannot be written to
```

A uniform also cannot be a slice. If you need a variable number of values, use
a `&[T]` storage buffer instead.

## Binding numbers

Parameters get consecutive bindings in group 0, in the order you wrote them.
The kernel in the first example ends up as:

```text
group 0 binding 0: source (Read)
group 0 binding 1: target (ReadWrite)
group 0 binding 2: amount (Uniform)
```

Most of the time that is what you want, and you never think about it.

When it is not, say so per parameter:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn split_across_groups(
    #[binding(group = 0, index = 0)] camera: &f32,
    #[binding(group = 1, index = 0)] input: &[f32],
    #[binding(group = 1, index = 1)] output: &mut [f32],
) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index] * camera;
    }
}
```

This is worth doing when some bindings change per frame and others do not,
since most APIs let you rebind one group without touching the others. Putting
the per frame data in its own group means you swap only that group.

Either part can be left out. `#[binding(group = 1)]` keeps the automatic index
and only moves the group.

Two parameters landing on the same slot is an error, caught at compile time:

```text
error: `input` and `output` both use group 0 binding 0, set one with #[binding(...)]
```

## Reading the layout from the host

Whatever the layout ends up being, the kernel will tell you:

```rust
# use unipute::{Kernel, kernel};
# #[kernel(workgroup_size(64))]
# fn blend(source: &[f32], target: &mut [f32], amount: &f32) {
#     let index = global_id().x;
#     if index < source.len() { target[index] = mix(target[index], source[index], amount); }
# }
# fn main() {
for binding in blend::BINDINGS {
    println!(
        "group {} binding {}: {} ({:?})",
        binding.group, binding.binding, binding.name, binding.access
    );
}
# }
```

Build your bind group layout from this rather than from a list you maintain by
hand. If you change the kernel's parameters, the host side follows
automatically instead of going quietly out of sync.

[Running a kernel](../output/running.md) shows this against a real API.

## What a kernel cannot take

- No `self`. A kernel is a free function.
- No generics.
- No return type. Results go into a `&mut` parameter. Trying to return gives
  you a message saying exactly that.
- No `&mut T` for a single value. Write `&mut [T]` and use index 0, or think
  about whether you wanted a storage buffer all along.
