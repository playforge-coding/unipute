# Atomics

Thousands of invocations run at once, and sooner or later two of them want to
update the same place. A counter, a histogram bin, a "who got here first"
flag. A plain `bins[b] += 1` is a read, an add and a write, and another
invocation can do its own read between yours and your write. Both add one,
the bin goes up by one, and nobody is told.

An atomic does the read, the change and the write as one step that nothing
can get between:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn count_values(values: &[u32], bins: &mut [AtomicU32]) {
    let index = global_id().x;
    if index < values.len() {
        bins[values[index] % 16u32].fetch_add(1u32);
    }
}
```

The types are `AtomicU32` and `AtomicI32`, and the methods are named after
the ones on Rust's own atomics, without the ordering argument, since a GPU
gives you one ordering and no choice about it.

## Where they live

An atomic is something to update, so it goes where updates happen: in a
`&mut` buffer, or in [workgroup memory](./workgroup-memory.md).

```rust
# use unipute::kernel;
#[kernel(workgroup_size(256))]
fn count_locally(values: &[u32], bins: &mut [AtomicU32]) {
    #[workgroup]
    let local_bins: [AtomicU32; 16];

    let index = global_id().x;
    if index < values.len() {
        local_bins[values[index] % 16u32].fetch_add(1u32);
    }
    workgroup_barrier();

    if local_index() < 16u32 {
        bins[local_index()].fetch_add(local_bins[local_index()].load());
    }
}
```

This is the usual shape. An atomic in workgroup memory is cheap, an atomic in
a buffer is not, so each workgroup counts into its own bins and adds the whole
set into the buffer once. Two hundred and fifty six adds on the chip, sixteen
in memory.

A uniform cannot be an atomic, since nothing writes to a uniform, and neither
can a `&[AtomicU32]` buffer: a buffer of atomics nobody updates is a buffer of
`u32` with extra steps, and the macro says so. A local cannot be one either,
and nor can a nested function's parameter, because an atomic is a place that
several invocations share rather than a value one of them holds. Pass a
helper what `.load()` gave you.

On the host an atomic is a plain integer. A `&mut [AtomicU32]` is uploaded as
a `Vec<u32>` and read back the same way, and nothing in `BINDINGS` changes.

## The operations

Each takes a value of the atomic's type and gives back what the atomic held
before the change.

| Method | Does |
| ------ | ---- |
| `fetch_add(v)` | adds `v` |
| `fetch_sub(v)` | subtracts `v` |
| `fetch_min(v)` | keeps the smaller of the two |
| `fetch_max(v)` | keeps the larger |
| `fetch_and(v)`, `fetch_or(v)`, `fetch_xor(v)` | bitwise |
| `swap(v)` | replaces the value with `v` |

Use the old value or not, as you like. As a statement on its own, the value
is dropped and the shader does a little less work:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn append(items: &[u32], output: &mut [u32], count: &mut [AtomicU32]) {
    let index = global_id().x;
    if index < items.len() && items[index] > 100u32 {
        // The old value is where this item goes, and nobody else gets it.
        let slot = count[0].fetch_add(1u32);
        output[slot] = items[index];
    }
}
```

That is stream compaction: every invocation that has something to say claims
the next free slot, and the slots come out dense.

`load()` reads the atomic, and `store(v)` writes it. Both are single atomic
accesses, so a value read with `load` is one another invocation wrote whole,
never half of one. Reading an atomic without `load`, or assigning to it, is an
error that tells you which method to use instead. So is `+=`.

## compare_exchange

`compare_exchange(current, new)` stores `new` only if the atomic still holds
`current`, and says whether it did:

```rust
# use unipute::kernel;
#[kernel(workgroup_size(64))]
fn claim_slots(slots: &mut [AtomicU32]) {
    let me = global_id().x + 1u32;
    let mut slot = me % slots.len();
    let mut claimed = false;
    while !claimed {
        claimed = slots[slot].compare_exchange(0u32, me);
        if !claimed {
            slot = (slot + 1u32) % slots.len();
        }
    }
}
```

It answers `true` or `false` rather than handing back the old value as Rust's
does, because on some hardware the exchange is allowed to fail even when the
values matched. The old value alone would not tell you that. Write the loop
and let it try again.

It is the one atomic method that is not allowed as a statement on its own. An
exchange that may not have happened is worth checking, so use the result.

## What atomics do not do

They do not order anything else. `fetch_add` on a counter and a plain store
to another buffer are two separate things, and another invocation may see
them in either order. When one workgroup's atomics have to be visible to the
rest of it before the next step, that is what [the barrier](./workgroup-memory.md#the-barrier)
is for, and `storage_barrier()` is the same for a buffer.

They are integers only. There is no `AtomicF32`, since most hardware has no
such thing, and no atomic on a `Vec` or on a struct's field yet. [What is not
built yet](../roadmap.md) keeps the list.
