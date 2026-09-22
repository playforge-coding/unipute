# Workgroup memory

Every invocation has its own locals, and every invocation in the dispatch
shares the buffers. Workgroup memory sits between the two: one copy per
workgroup, shared by the invocations in that workgroup and by nobody else. It
lives on the chip rather than out in the card's memory, so it is fast, and it
is where the invocations of a group leave things for each other.

## Declaring it

A `let` marked `#[workgroup]`, at the top level of the kernel body, with a
type and no value:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn block_sum(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let tile: [f32; 64];

    let lane = local_index();
    tile[lane] = input[global_id().x];
    workgroup_barrier();

    if lane == 0u32 {
        let mut total = 0.0;
        for i in 0..tile.len() {
            total += tile[i];
        }
        output[workgroup_id().x] = total;
    }
}
```

The type is a scalar, a vector, or a fixed length array of either. `[f32; 64]`
is the usual shape: one slot per invocation, indexed by `local_index()`. An
array of arrays such as `[[f32; 8]; 8]` works too, for a tile with two
dimensions, and is indexed as `tile[y][x]`.

There is no value to give it, because there is no single invocation to give
it. Every invocation writes its own part, and a barrier makes the parts
visible to the rest. Every target Unipute generates zeroes it before the
workgroup starts, so a slot nobody has written reads as zero, but a kernel
that depends on that is usually a kernel with a missing write.

Index a shared array like a buffer, and use `.len()` on it the same way. The
length is written in the kernel, so `.len()` here is a plain number rather
than a question to the driver. A shared scalar or vector is read and assigned
by name:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn subtract_first(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let first: f32;

    let index = global_id().x;
    if local_index() == 0u32 {
        first = input[index];
    }
    workgroup_barrier();
    if index < input.len() {
        output[index] = input[index] - first;
    }
}
```

That is a broadcast: one invocation reads a value, and after the barrier the
whole group has it.

## The barrier

A write to workgroup memory is not seen by the other invocations until
`workgroup_barrier()` has run. The shape is always the same: write, barrier,
read. Between one round of writing and reading and the next, there is another
barrier.

Every invocation in the workgroup has to reach every barrier. That is easy to
break by accident with the bounds check most kernels start with:

```rust,ignore
let index = global_id().x;
if index >= input.len() {
    return;             // this invocation never reaches the barrier below
}
tile[lane] = input[index];
workgroup_barrier();    // the rest of the group waits for it
```

The last workgroup of a dispatch is usually partly past the end of the input,
and the invocations that leave early leave the others waiting at a barrier
that never completes. Guard the reads and writes instead, and let every
invocation fall through to the barrier:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn block_sum(input: &[f32], output: &mut [f32]) {
    #[workgroup]
    let tile: [f32; 64];

    let index = global_id().x;
    let lane = local_index();
    if index < input.len() {
        tile[lane] = input[index];
    } else {
        tile[lane] = 0.0;
    }
    workgroup_barrier();

    // Fold the top half of the tile onto the bottom half, then the top half
    // of what is left, until element 0 holds the whole sum.
    let mut stride = 32u32;
    while stride > 0u32 {
        if lane < stride {
            tile[lane] += tile[lane + stride];
        }
        workgroup_barrier();
        stride /= 2u32;
    }

    if lane == 0u32 {
        output[workgroup_id().x] = tile[0];
    }
}
```

This is a reduction, and it is most of what workgroup memory is for. Six
rounds of adding replace sixty three, and every round reads memory on the chip
rather than the buffer. The barrier inside the loop is fine, because every
invocation runs the loop the same number of times. The `if` around the
addition is fine too, since there is no barrier inside it.

## The rules

- **Top level of the kernel body.** A declaration inside an `if` or a loop is
  rejected. The memory exists for the whole kernel whether or not the line
  runs, so putting it inside a block would suggest something that is not so.
- **Kernel body only.** A nested `fn` cannot declare workgroup memory or reach
  the kernel's, for the same reason it cannot reach a buffer: the shader
  language hands both to the entry point alone. See [Your own
  functions](./functions.md).
- **Not in the layout.** The host never binds it, so it is not in `BINDINGS`.
  Adding workgroup memory to a kernel changes nothing on the host side.
- **One name once.** Two declarations with the same name is an error, where a
  `let` would shadow. Two tiles both called `tile` are two allocations, and
  that is more likely a mistake than a choice.
- **A fixed size.** The length is a number written in the kernel, not a
  `const` from outside, since nothing from outside the kernel is visible in
  it.

How much of it a workgroup can have depends on the hardware. Sixteen kilobytes
is safe everywhere, and most desktop GPUs allow thirty two or more. Unipute
does not check the total, so a kernel asking for more than the device has
fails when the host creates the pipeline rather than at compile time.
