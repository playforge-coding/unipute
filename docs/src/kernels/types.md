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

### Swizzles

Two to four components at once make a new vector, in whatever order you name
them. `color.xy` is a `Vec2`, `color.zyx` is the first three reversed, and
`color.xxx` repeats one. You can assign through a swizzle too, which writes
just the components it names:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn step(bodies: &mut [Vec4<f32>], dt: &f32) {
    let index = global_id().x;
    if index < bodies.len() {
        // Position in `xyz`, mass in `w`. Only the position moves.
        let mut body = bodies[index];
        body.xyz += vec3(0.0f32, -9.8, 0.0) * dt;
        bodies[index] = body;
    }
}
```

The value is worked out in full before anything is written, so `v.xy = v.yx`
swaps the two rather than copying one over the other. A swizzle you assign to
cannot name a component twice, since `v.xx = ...` would have two values for
one place.

The letters are `x`, `y`, `z` and `w` only. Some shader languages also take
`rgba` for colours, but Unipute keeps one set of names.

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

## Structs

A struct you define, with `#[derive(Layout)]` on it, can be a buffer element,
a uniform, a local, or a parameter of a nested function. Its fields are read
with `.name`. [Your own structs](./structs.md) is the chapter on them, and on
the one thing to know about their layout.

## Atomics

`AtomicU32` and `AtomicI32` are integers that many invocations can update at
once without losing an update. They live in `&mut` buffers and in workgroup
memory, and are reached through `.load()`, `.store()` and the `fetch_`
methods rather than read and assigned like a number. [Atomics](./atomics.md)
is the chapter on them.

## Arrays with a fixed length

`[T; N]` is the type of [workgroup memory](./workgroup-memory.md), and that is
the only place the macro accepts it. A local cannot be an array yet, and a
buffer is always a slice, since its length is the host's decision.

## What is not here yet

Matrices, textures and samplers. All of them are on the list, and [What is
not built yet](../roadmap.md) says roughly what each one needs. If you hit
one of these, that chapter is the honest answer rather than a workaround.
