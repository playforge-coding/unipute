# Types

Kernels use a small set of types. This chapter is the whole list.

## Scalars

| Type | Notes |
| ---- | ----- |
| `f32` | the usual floating point type |
| `u32` | unsigned, and what indices and lengths are |
| `i32` | signed |
| `bool` | conditions only, not something you put in a buffer |

There is no `f64`, no `u64`, no `usize` and no `u8`. GPUs either do not have
them or make you ask for an extension, and a version 0.1 that quietly picks
for you would be worse than one that says no.

`usize` is the one people reach for out of habit. Lengths and indices are
`u32` here:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn count(input: &[f32], output: &mut [u32]) {
    let total = input.len();   // u32, not usize
    let index = global_id().x; // u32 as well
    if index == 0u32 {
        output[0] = total;
    }
}
```

## Vectors

`Vec2<T>`, `Vec3<T>` and `Vec4<T>`, built with `vec2`, `vec3` and `vec4`:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(8, 8))]
fn gradient(output: &mut [f32], width: &u32) {
    let cell = global_id();
    let color = vec3(cell.x as f32, cell.y as f32, 0.5);
    output[cell.y * width + cell.x] = length(color);
}
```

Note the capital letter in the type and the lower case in the constructor,
which follows how Rust names types against functions.

Components are `.x`, `.y`, `.z` and `.w`, in that order. Reading past the end
of the vector is a compile error, so `.w` on a `Vec3` will tell you off rather
than producing something surprising.

There are no swizzles yet. `color.xy` does not work, and you write
`vec2(color.x, color.y)`.

Vectors can be stored in buffers:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn normalise_all(points: &mut [Vec3<f32>]) {
    let index = global_id().x;
    if index < points.len() {
        points[index] = normalize(points[index]);
    }
}
```

Keep in mind that a `Vec3` takes the space of a `Vec4` in a buffer. That is
not Unipute being wasteful, it is how every shader language lays out three
component vectors, and your host side allocation has to agree.

## Casts

`as` converts between scalar types:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn to_float(input: &[u32], output: &mut [f32]) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index] as f32;
    }
}
```

This is a value conversion, the same as in Rust. You can only cast to a
scalar, so there is no `as Vec3<f32>`.

## How literals get their type

An unsuffixed integer takes its type from whatever it is used with:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn literals(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;

    // `4` becomes a u32, because `index` is one.
    if index < 4 {
        // `2.0` is an f32, since that is the only float type there is.
        output[index] = input[index] * 2.0;
    }
}
```

This is the same idea as Rust's inference, done in a much simpler way: an
untyped literal on one side of an operator picks up the type of the other
side. It handles the cases that come up in practice.

When there is no other side to learn from, an integer literal defaults to
`i32`, exactly as in Rust. In a kernel that is usually not what you meant, so
say which one you want:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn explicit(output: &mut [u32]) {
    let counter = 0u32;        // suffix
    let other: u32 = 0;        // or an annotation
    output[0] = counter + other;
}
```

If Unipute genuinely cannot work out a type it says so rather than guessing:

```text
error: the type of this value is unclear, annotate it as in `let x: f32 = ...`
```

## Arrays with a fixed length

The IR can describe them, but the `#[kernel]` macro has no syntax for them
yet, so buffers are slices. Use a slice and index it.

## What is not here yet

Matrices, atomics, textures, samplers and your own structs. All of them are on
the list, and [What is not built yet](../roadmap.md) says roughly what each
one needs. If you hit one of these, that chapter is the honest answer rather
than a workaround.
