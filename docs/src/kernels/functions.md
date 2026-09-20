# Your own functions

A kernel can declare `fn` items inside its body and call them, the same way a
nested function works in ordinary Rust:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn tonemap(input: &[f32], output: &mut [f32]) {
    fn compress(value: f32) -> f32 {
        value / (value + 1.0)
    }

    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = compress(input[index]);
}
```

That comes out as two functions in the generated shader, with `compress`
written above the entry point that calls it.

## They capture nothing

This is the rule worth reading twice. A nested `fn` in Rust cannot see the
variables around it, and the same is true here, only more so: a nested function
cannot reach the kernel's buffers or the hardware built-ins either.

```rust,ignore
#[kernel(workgroup_size(64))]
fn wrong(input: &[f32], output: &mut [f32]) {
    fn first() -> f32 {
        input[0] // error: `input` is not a parameter or a local variable
    }

    output[0] = first();
}
```

The reason is not a limitation of Unipute. Shader languages hand bindings and
built-ins to the entry point, and a function called from it has no way to ask
for them. So a nested function takes what it needs as a parameter:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn right(input: &[f32], output: &mut [f32]) {
    fn scale(value: f32, by: f32) -> f32 {
        value * by
    }

    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    output[index] = scale(input[index], 2.0);
}
```

Calling `global_id()` inside a nested function is rejected for the same reason.
Read it in the kernel body and pass it down.

## Parameters and results

A parameter is a scalar or a vector, taken by value. Slices cannot be passed,
which follows from the rule above: a buffer belongs to the entry point.

A result is a scalar or a vector too, or nothing at all. The body ends with a
value or a `return`, as in Rust:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn shades(output: &mut [f32]) {
    fn luminance(color: Vec3<f32>) -> f32 {
        // A trailing expression is the return value.
        dot(color, vec3(0.2126, 0.7152, 0.0722))
    }

    fn clamp_up(value: f32) -> f32 {
        if value < 0.0 {
            return 0.0;
        }
        value
    }

    output[global_id().x] = clamp_up(luminance(vec3(0.5, 0.25, 0.75)));
}
```

One thing Rust allows that a kernel does not is a block used as a value, so
`if a { b } else { c }` cannot be the result of a function. Write a `return` in
each branch instead.

A function that returns nothing is called as a statement:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn with_a_void_helper(output: &mut [f32]) {
    fn fill(_value: f32) {
        // Nothing useful, but it shows the shape.
    }

    fill(1.0);
    output[0] = 1.0;
}
```

## Order does not matter

Declare them in whichever order reads best. Unipute works out which function
calls which and writes them out so that every callee comes before its callers,
which is what a shader language needs since none of them have forward
declarations.

The one thing that follows from this is that **a function cannot call itself**,
directly or through another function. No shader language supports recursion, so
a cycle is a compile error naming the function that closes it.

## Where they go

At the top level of the kernel body. A `fn` inside an `if` or a loop is
rejected, since it would be the only thing in a kernel whose meaning depended
on where it appeared.

Two nested functions in the same kernel cannot share a name, and neither can
take the name of a built-in such as `abs` or `vec3`.

Functions are not shared between kernels. Two kernels that want the same helper
each declare it. Sharing one across kernels needs a different surface than
`#[kernel]` offers, and is not built yet.
