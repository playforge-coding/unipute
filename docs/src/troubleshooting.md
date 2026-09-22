# When something goes wrong

Unipute tries to tell you what to do rather than only what went wrong. This
chapter covers the messages that need more explanation than fits in one line,
grouped by what you were probably trying to do.

## Writing the kernel signature

### `a kernel needs a workgroup size, write #[kernel(workgroup_size(64))]`

The attribute has one required option. Even a kernel that does one item of
work per invocation needs to say how many invocations a group runs.

### `a kernel returns nothing, write results into a &mut parameter`

Kernels produce nothing. There is nowhere for a return value to go, because
there is no caller waiting for one. Take a `&mut [T]` and write into it.

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn total(input: &[f32], output: &mut [f32]) {
    // not `-> f32`
    output[0] = input[0] + input[1];
}
```

### `a &mut parameter must be a slice, write &mut [f32]`

`&mut f32` is not a thing a kernel can have. A single mutable value has no
meaning when thousands of invocations run at once. Use `&mut [f32]` and index
it, even if you only ever use index 0.

### `a kernel cannot be generic`

Shaders are concrete. If you want the same kernel over `f32` and `u32`, write
two kernels, or generate them with a macro of your own.

## Writing the body

### `X is not a built-in Unipute provides`

You called something that is not on the list. A kernel cannot call your other
Rust functions, only the built-ins in [Built-ins and
functions](./kernels/builtins.md).

If the function you want is a normal numeric one that is missing, that is
worth reporting. If you wanted to call your own function, see [What is not
built yet](./roadmap.md).

### `X is not a kernel parameter or a local variable`

The name is not in scope. Kernels cannot see anything outside themselves, so
constants, statics and other items from your crate are not available. Pass the
value in as a uniform.

```rust,ignore
const SCALE: f32 = 2.0;

#[kernel(workgroup_size(64))]
fn nope(output: &mut [f32]) {
    output[0] = SCALE; // SCALE is not visible in here
}
```

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn yes(output: &mut [f32], scale: &f32) {
    output[0] = scale; // pass it in instead
}
```

### `this statement has no effect, a kernel statement must assign, branch, loop or call a built-in`

Usually an expression written where a statement was expected, often because
`if` was being used as an expression:

```rust,ignore
let sign = if value < 0.0 { -1.0 } else { 1.0 };
```

Assign in both branches instead, or use `step` and `mix`.

### `the type of this value is unclear, annotate it as in let x: f32 = ...`

Type inference here is deliberately simple, and it ran out of information.
Add a suffix or an annotation:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn annotated(output: &mut [u32]) {
    let counter = 0u32;
    let other: u32 = 1;
    output[0] = counter + other;
}
```

### `X is a buffer, assign to one element as in X[i] = ...`

You assigned to a whole buffer. Buffers are not values that can be copied
wholesale, so assign per element inside a loop or per invocation.

### `X is a uniform, it cannot be written to`

Uniforms are read only. If you need to write, take the parameter as
`&mut [T]`.

### `workgroup memory is declared at the top level of the kernel body, not inside a block`

`#[workgroup] let` goes at the top of the kernel body, next to any nested
`fn` items, not inside an `if` or a loop. It exists for the whole kernel
either way, so the top level is the only place that says so. A nested `fn`
cannot declare it either, for the reason [Your own
functions](./kernels/functions.md) gives. See [Workgroup
memory](./kernels/workgroup-memory.md).

## Indices and types

### `X cannot be indexed`

Indexing works on slice parameters. Indexing a scalar or a vector is not
allowed, and to read a vector component use `.x` through `.w` rather than
`[0]`.

### `usize` errors, or a length not matching

Lengths and indices are `u32`, not `usize`. There is no `usize` on a GPU. If
you are comparing a `.len()` against something, make sure both sides are
`u32`.

### `X is not a type Unipute knows, use a scalar or Vec2, Vec3 or Vec4`

The type list is short on purpose. See [Types](./kernels/types.md) for what is
there. `f64`, `u64`, `u8` and `usize` are all deliberately absent.

## Bindings

### `X and Y both use group 0 binding 0, set one with #[binding(...)]`

Two parameters landed on the same slot, which can only happen if you set at
least one by hand. Give them different `index` values, or different `group`
values.

### Bindings not matching what the host expects

Read them from the kernel rather than writing them out twice:

```rust
# use unipute::{Kernel, kernel};
# #[kernel(workgroup_size(64))]
# fn scale(input: &[f32], output: &mut [f32]) {
#     let index = global_id().x;
#     if index < input.len() { output[index] = input[index]; }
# }
# fn main() {
for binding in scale::BINDINGS {
    println!("{} -> group {} binding {}", binding.name, binding.group, binding.binding);
}
# }
```

On Metal this does not apply, because Metal assigns its own buffer slots. See
[Choosing targets](./output/targets.md#msl).

## Targets

### `cannot find WGSL in this scope`, or similar

Import the trait:

```rust,ignore
use unipute::WgslKernel; // for WGSL
use unipute::SpirvKernel; // for SPIRV
```

### `target msl is not available, enable the "msl" feature`

The target exists but this build was not compiled with it. Add the feature.

### `the naga back end does not support the vertex stage yet`

Graphics stages are reserved in the IR and not implemented. Compute only for
now.

## Things that are not your fault

### `naga rejected the generated module: ...`

This one is different from all the others. It means Unipute built a module
that naga considered invalid, which is a bug in Unipute rather than in your
kernel. The naga message is passed through unchanged so it can go in a report.

Worth reporting with the kernel that caused it. There is no workaround to look
for, and you have not done anything wrong.

## Runtime problems Unipute cannot catch

These compile fine and go wrong on the GPU.

**Reading past the end of a buffer.** There is no bounds checking in a kernel.
Guard with `.len()`:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn guarded(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = input[index];
}
```

**Forgetting that dispatch rounds up.** 100 elements with a workgroup of 64 is
128 invocations. The check above is what stops the extra 28.

**A barrier inside a branch.** Every invocation in a workgroup has to reach the
same barrier. Unipute does not check this yet. Keep barriers at the top level
of the body.

**An early `return` before a barrier.** The same mistake in another shape. The
invocations past the end of the input leave, and the rest of the group waits
at the barrier for invocations that are never coming. Guard the work with an
`if` instead and let every invocation reach the barrier. [Workgroup
memory](./kernels/workgroup-memory.md) shows the pattern.

**A `Vec3` taking four slots.** Three component vectors are padded to the size
of four in a buffer. Your host side allocation has to agree, or everything
after the first element reads from the wrong place.

**Races between invocations.** Two invocations writing the same element is
undefined. If invocations need to see each other's writes, a barrier is
required, and even then only within one workgroup.

## Still stuck

Look at the shader. It is the fastest way to find out what your kernel
actually became:

```bash
cargo run --example emit_shaders
```

Or in your own code, print `YourKernel::WGSL`. Read past naga's `_e12`
temporaries and look at the stores and the control flow. See [How it
works](./how-it-works.md#why-the-generated-shaders-look-like-that) for why the
output reads that way.
