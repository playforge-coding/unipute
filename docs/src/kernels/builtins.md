# Built-ins and functions

This is everything Unipute provides. A kernel can also call functions you
declare inside it, which [Your own functions](./functions.md) covers.

## Knowing where you are

A kernel runs once per invocation. These tell an invocation which one it is.

| Call | Type | Meaning |
| ---- | ---- | ------- |
| `global_id()` | `Vec3<u32>` | position in the whole dispatch |
| `local_id()` | `Vec3<u32>` | position inside this workgroup |
| `local_index()` | `u32` | `local_id` flattened to one number |
| `workgroup_id()` | `Vec3<u32>` | which workgroup this is |
| `num_workgroups()` | `Vec3<u32>` | how many workgroups were dispatched |

`global_id()` is the one you want most of the time. For one dimensional work
it is the index into your buffer:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn one_dimensional(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index];
    }
}
```

For two dimensional work, use two components and a width uniform to flatten
them:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(8, 8))]
fn two_dimensional(output: &mut [f32], width: &u32, height: &u32) {
    let cell = global_id();
    if cell.x >= width {
        return;
    }
    if cell.y >= height {
        return;
    }
    output[cell.y * width + cell.x] = 1.0;
}
```

Notice the bounds check on both axes. A workgroup of 8 by 8 over an image that
is not a multiple of 8 in each direction will run invocations past the edge.

The relationship between them is worth keeping in mind:

```text
global_id = workgroup_id * workgroup_size + local_id
```

`local_id` is what you use to index into workgroup shared memory, which is on
the roadmap rather than in the current release. Until then, `global_id` covers
almost everything.

## Barriers

`workgroup_barrier()` and `storage_barrier()`, both taking no arguments. See
[Control flow](./control-flow.md#barriers) for when you need them and the rule
about keeping them out of branches.

## Buffer length

`.len()` on a slice parameter gives its length in elements, as a `u32`:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn lengths(input: &[f32], output: &mut [u32]) {
    output[0] = input.len();
}
```

It only works on a parameter, not on an arbitrary expression, and it is the
only method a kernel can call.

## Building vectors

`vec2(x, y)`, `vec3(x, y, z)` and `vec4(x, y, z, w)`. All the components have
to agree on a type:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn make_vectors(output: &mut [f32]) {
    let a = vec3(1.0, 0.0, 0.0);
    let b = vec3(0.0, 1.0, 0.0);
    output[0] = dot(a, b);
}
```

## Numeric functions

All of these are available. Most take and return whatever type you give them,
so `abs` works on a scalar or a vector.

| Arguments | Functions |
| --------- | --------- |
| one | `abs`, `sign`, `sqrt`, `inverse_sqrt`, `exp`, `log`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`, `length`, `normalize` |
| two | `min`, `max`, `pow`, `step`, `dot`, `cross` |
| three | `clamp`, `mix`, `fma` |

### The ones worth explaining

`mix(a, b, t)` blends between `a` and `b`. At `t = 0` you get `a` and at
`t = 1` you get `b`.

`step(edge, x)` gives 0 when `x < edge` and 1 otherwise. Together with `mix`
it replaces a lot of small branches:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn branchless(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    let value = input[index];

    // The same as: if value < 0.5 { 0.0 } else { 1.0 }
    output[index] = step(0.5, value);
}
```

`clamp(x, low, high)` keeps `x` between the two bounds.

`fma(a, b, c)` is `a * b + c` done in one step, which is both faster and more
accurate than writing it out.

`dot` and `length` collapse a vector down to a single number. `cross` takes
two three component vectors and gives another one.

## What you cannot call

A free function declared outside the kernel, including another `#[kernel]`. The
macro only sees the function it is attached to, so a helper has to be declared
inside the kernel body. See [Your own functions](./functions.md).
