# Your first kernel

We are going to write a kernel that scales every number in a buffer, look at
what it turns into, and then take it apart a piece at a time.

## The kernel

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
    println!("{}", scale::WGSL);
}
```

Run it and you get this:

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

Nothing about that shader was written by hand, and nothing about it was
produced while the program was running. It was already a string constant
before `main` started.

## Reading it line by line

### `#[kernel(workgroup_size(64))]`

This is the only required option. A workgroup is the batch of invocations the
GPU schedules together, and 64 is a reasonable default for one dimensional
work. You can give up to three dimensions:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(8, 8))]
fn over_a_grid(output: &mut [f32]) {
    let cell = global_id();
    output[cell.y * 8u32 + cell.x] = 1.0;
}
```

Dimensions you leave out become 1, so `workgroup_size(64)` means
`[64, 1, 1]`.

### The parameters

```rust,ignore
fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
```

The type of each parameter decides how it is bound:

- `&[f32]` became a read only storage buffer
- `&mut [f32]` became a read and write storage buffer
- `&f32` became a uniform

You did not write any binding numbers. Parameters get consecutive bindings in
group 0, in the order you declared them, which is usually what you want.
[Parameters and bindings](../kernels/parameters.md) covers how to say
otherwise.

### `global_id().x`

`global_id()` is where this invocation sits in the whole dispatch. It is a
three component vector, and `.x` picks the first component. A kernel runs once
per invocation, so this is how each copy of the kernel knows which element of
the buffer belongs to it.

### The bounds check

```rust,ignore
if index >= input.len() {
    return;
}
```

You dispatch whole workgroups, so if the buffer has 100 elements and the
workgroup is 64 wide you end up dispatching 128 invocations. The 28 extra ones
have to be told to do nothing. Forgetting this check is the single most common
GPU bug, and it is worth getting into the habit.

`input.len()` works because Unipute knows `input` is a runtime sized buffer.
It becomes `arrayLength` in the shader, so the length comes from however much
data the host actually bound.

### The work

```rust,ignore
output[index] = input[index] * factor;
```

`factor` is a uniform holding a single `f32`, so using it by name reads the
value. There is no dereference, because there are no pointers to speak of
inside a kernel.

## What else the macro gave you

The function is gone, replaced by a type with the same name. Along with the
shader, that type carries the information the host side needs:

```rust
# use unipute::{Kernel, kernel};
# #[kernel(workgroup_size(64))]
# fn scale(input: &[f32], output: &mut [f32], factor: &f32) {
#     let index = global_id().x;
#     if index >= input.len() { return; }
#     output[index] = input[index] * factor;
# }
fn main() {
    assert_eq!(scale::NAME, "scale");
    assert_eq!(scale::WORKGROUP_SIZE, [64, 1, 1]);

    for binding in scale::BINDINGS {
        println!(
            "group {} binding {}: {} ({:?})",
            binding.group, binding.binding, binding.name, binding.access
        );
    }
}
```

```text
group 0 binding 0: input (Read)
group 0 binding 1: output (ReadWrite)
group 0 binding 2: factor (Uniform)
```

This matters more than it looks. Building a bind group by hand means keeping
the numbers in your host code in step with the numbers in your shader, and
nothing checks that for you. Here there is one source of truth, and it is the
kernel.

## Breaking it on purpose

Try changing the multiply to something Unipute does not have:

```rust,ignore
output[index] = wibble(input[index]);
```

```text
error: `wibble` is not a built-in Unipute provides
 --> src/main.rs:8:21
  |
8 |     output[index] = wibble(input[index]);
  |                     ^^^^^^
```

The error points at your source, at the right span, during compilation. That
is the payoff for doing the work at compile time.

## Where to go next

- [Parameters and bindings](../kernels/parameters.md) for control over the
  binding layout
- [Running a kernel](../output/running.md) for actually getting this onto a
  GPU
- [Control flow](../kernels/control-flow.md) for loops and branches, including
  a couple of places where kernels differ from ordinary Rust
