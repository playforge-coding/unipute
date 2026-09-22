# Control flow

Kernels use ordinary Rust control flow. This chapter covers what is supported
and the few places where a kernel behaves differently from a function on the
CPU.

## Branches

`if` and `else` work as you would expect, including `else if` chains:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn classify(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }

    let value = input[index];
    if value < 0.0 {
        output[index] = -1.0;
    } else if value > 0.0 {
        output[index] = 1.0;
    } else {
        output[index] = 0.0;
    }
}
```

`if` is a statement here, not an expression. This does not work:

```rust,ignore
let sign = if value < 0.0 { -1.0 } else { 1.0 };
```

Assign in both branches instead, or reach for `mix` and `step` from
[Built-ins and functions](./builtins.md), which is what shader code usually
does anyway.

There is no `match` and no `if let`.

## Loops

`while`, `loop` and `for` over a range all work:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn loops(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    let mut total = 0.0;

    // A counted loop over a range.
    for offset in 0..4u32 {
        let sample = index + offset;
        if sample < input.len() {
            total += input[sample];
        }
    }

    // The same thing spelled out.
    let mut i = 0u32;
    while i < 4u32 {
        i += 1u32;
    }

    output[index] = total;
}
```

`for` only iterates over a range. There are no iterators on a GPU, so
`for value in buffer` is not something that can be translated. Loop over the
indices and read the buffer inside.

Both `a..b` and `a..=b` work. `break` and `continue` do what they do in Rust,
and using either outside a loop is a compile error rather than something you
find out about later.

### A note on divergence

This is the one place where GPU thinking differs from CPU thinking, and it is
worth knowing even though Unipute does not make you do anything about it.

Invocations in a workgroup run in lockstep. When they take different branches,
the hardware runs both sides and masks off the invocations that should not be
affected. A branch that different invocations resolve differently therefore
costs roughly the sum of both sides rather than the cheaper one.

This does not change what your kernel means, only how fast it goes. It is why
shader code often prefers `mix` and `step` over a short `if`, and why a bounds
check at the top of a kernel is cheap while a long branch in the middle of a
hot loop might not be.

## Returning early

`return` with no value leaves the kernel:

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

`return` never takes a value, because a kernel produces nothing. Results go
into a `&mut` parameter.

## Variables

`let` introduces a local, and `mut` is accepted but not required, since a
kernel local is a shader `var` and writable either way:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn locals(output: &mut [f32]) {
    let base = 1.0;
    let mut total = base;
    total += 2.0;
    output[0] = total;
}
```

Locals are scoped to their block, like in Rust, and shadowing works.

All the compound assignments are available: `+=`, `-=`, `*=`, `/=`, `%=`,
`&=`, `|=`, `^=`, `<<=` and `>>=`.

## Operators

Arithmetic (`+ - * / %`), comparison (`== != < <= > >=`), logical (`&& ||
!`), bitwise (`& | ^`) and shifts (`<< >>`).

Whether `&&` and `||` short circuit depends on which target you generate.
WGSL says they do. The SPIR-V that naga produces evaluates both sides. Since
Unipute's whole point is that one kernel feeds every target, do not write a
kernel that depends on the answer:

```rust,ignore
// Do not do this. Whether the index is evaluated depends on the target.
if index < input.len() && input[index] > 0.0 { }
```

Nest the checks instead:

```rust
# use unipute::kernel;
# #[kernel(workgroup_size(64))]
# fn safe(input: &[f32], output: &mut [f32]) {
# let index = global_id().x;
if index < input.len() {
    if input[index] > 0.0 {
        output[index] = 1.0;
    }
}
# }
```

## Barriers

When invocations in a workgroup need to see each other's writes, put a barrier
between the writing and the reading:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn with_barrier(data: &mut [f32]) {
    let index = global_id().x;
    data[index] = data[index] * 2.0;

    storage_barrier();

    if index > 0u32 {
        data[index] = data[index] + data[index - 1u32];
    }
}
```

`workgroup_barrier()` covers [workgroup memory](./workgroup-memory.md), which
is what most barriers are for, and `storage_barrier()` covers storage buffers.

Every invocation in the workgroup has to reach the same barrier. Putting one
inside an `if` that only some invocations take is undefined behaviour on most
hardware, and an early `return` before a barrier is the same mistake in a
different shape. Unipute does not check this for you yet, so keep barriers at
the top level of the kernel body, or in a loop that every invocation runs the
same number of times.

## Quick reference

| Works | Does not |
| ----- | -------- |
| `if` / `else if` / `else` | `match`, `if let` |
| `while`, `loop` | labelled loops |
| `for i in a..b`, `a..=b` | `for x in slice` |
| `break`, `continue` | `break` with a value |
| `return;` | `return value;` |
| `let`, shadowing, `mut` | `let else`, patterns |
| all the usual operators | relying on `&&` to short circuit |
