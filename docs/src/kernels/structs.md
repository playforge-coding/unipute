# Your own structs

A buffer does not have to hold plain numbers. It can hold a struct you
define, so that a particle is one record with a position, a mass and a
velocity rather than three buffers that happen to line up.

```rust
# use unipute::{Layout, kernel};
#[derive(Layout, Clone, Copy)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
}

#[kernel(workgroup_size(64))]
fn heavier(particles: &mut [Particle]) {
    let index = global_id().x;
    if index < particles.len() {
        particles[index].mass = particles[index].mass * 2.0;
    }
}
```

The same struct is used on the host to fill the buffer and in the kernel to
read it. That is the point of it, and it is also the danger: the two sides
have to agree about where every byte is. `#[derive(Layout)]` is what makes
sure they do.

## Declaring one

Three things are required, and the derive checks all of them.

**`#[repr(C)]`.** Without it, Rust is free to reorder fields, and then nothing
about the struct's memory is knowable. The derive refuses a struct without it.

**Named fields.** A kernel reads a field by name, so a tuple struct has nothing
to read by.

**Field types the GPU has.** A field is an `f32`, a `u32` or an `i32`, or an
array of two, three or four of them. Inside the kernel the array is a vector:
`[f32; 3]` on the host is `Vec3<f32>` in the kernel, and `particle.position.x`
works as you would expect. There is no `Vec3` type on the host, and writing
one in a struct gets you a message saying to use the array.

`bool` is not allowed, since the host and the GPU do not agree on how big one
is. Store a `u32` and compare it with zero.

A struct inside a struct, or an array of more than four elements, is not
supported yet. [What is not built yet](../roadmap.md) has the details.

## Using one

Anywhere a scalar or a vector can go, a struct can go too.

A `&[Particle]` parameter is a read only buffer of them, `&mut [Particle]` a
buffer the kernel writes, and `&Settings` a uniform holding one:

```rust
# use unipute::{Layout, kernel};
# #[derive(Layout, Clone, Copy)]
# #[repr(C)]
# struct Particle { position: [f32; 3], mass: f32 }
#[derive(Layout, Clone, Copy)]
#[repr(C)]
struct Settings {
    gravity: [f32; 3],
    dt: f32,
}

#[kernel(workgroup_size(64))]
fn fall(particles: &mut [Particle], settings: &Settings) {
    let index = global_id().x;
    if index >= particles.len() {
        return;
    }
    let particle = particles[index];
    particles[index] = Particle {
        position: particle.position + settings.gravity * settings.dt,
        mass: particle.mass,
    };
}
```

Reading `particles[index]` copies the whole struct into a local. Reading
`particles[index].mass` loads that one field and nothing else, which is what
you want in a kernel that only looks at one.

Assign to a field in place with `particles[index].mass = ...`, or replace the
whole element with a struct literal. A literal names every field, in any
order, and there is no `..` to fill in the rest, since a shader has no idea
what a default would be. Field shorthand works, so `Particle { position, mass }`
is fine when the locals have those names.

A [nested function](./functions.md) can take a struct and return one:

```rust
# use unipute::{Layout, kernel};
# #[derive(Layout, Clone, Copy)]
# #[repr(C)]
# struct Particle { position: [f32; 3], mass: f32 }
#[kernel(workgroup_size(64))]
fn merge(particles: &mut [Particle]) {
    fn combined(a: Particle, b: Particle) -> Particle {
        Particle {
            position: (a.position + b.position) * 0.5,
            mass: a.mass + b.mass,
        }
    }

    let index = global_id().x;
    if index + 1u32 < particles.len() {
        particles[index] = combined(particles[index], particles[index + 1u32]);
    }
}
```

The struct can be declared before or after the kernels that use it, and in
another module, as long as it is in the same crate. `use` it the way you
would use any type.

## Where the bytes go

Shader languages lay a struct out by rules that are almost, but not quite,
what `#[repr(C)]` does. Every field starts at a multiple of its alignment:

| Field | Size | Aligns to |
| ----- | ---- | --------- |
| `f32`, `u32`, `i32` | 4 | 4 |
| `[T; 2]` | 8 | 8 |
| `[T; 3]` | 12 | 16 |
| `[T; 4]` | 16 | 16 |

Then the whole struct is rounded up to a multiple of its largest alignment.

The one to watch is `[T; 3]`. It takes twelve bytes but has to start at a
multiple of sixteen, and Rust knows nothing about that: to Rust it is three
floats, aligned to four. So `Particle` above works, with `position` at byte 0
and `mass` fitting in the four bytes after it, and this does not:

```rust,compile_fail
# use unipute::Layout;
#[derive(Layout)]
#[repr(C)]
struct Weighted {
    mass: f32,
    position: [f32; 3],
}
```

Rust puts `position` at byte 4. The GPU puts it at byte 16. Rather than let
the kernel read the wrong twelve bytes, the derive stops:

```text
error: `position` sits at byte 4 on the CPU and byte 16 on the GPU, add `_pad: [u8; 12]` before it
```

A field whose name starts with an underscore is padding. Its type is `[u8; N]`,
the kernel never sees it, and it exists to push the next field to where the
GPU expects it. Do what the message says and the derive is satisfied on that
field, and then possibly tells you about the end of the struct:

```text
error: `Weighted` is 28 bytes on the CPU and 32 on the GPU, add `_pad2: [u8; 4]` as its last field
```

The GPU rounds the struct up to 32 because its largest field aligns to 16.
One more padding field and the two agree:

```rust
# use unipute::Layout;
#[derive(Layout)]
#[repr(C)]
struct Weighted {
    mass: f32,
    _pad: [u8; 12],
    position: [f32; 3],
    _pad2: [u8; 4],
}
```

Or reorder the fields so that the vector comes first and the scalar fills the
gap after it, which needs no padding at all. When you have a choice, that is
the better layout, and it is why the examples in this book put the vector
first.

The derive always says exactly what to add, so there is no need to work any
of this out by hand. It also leaves a check behind that compares its answer
against what rustc actually did, so if the two ever disagreed the build would
fail rather than the kernel silently reading garbage.

## Uploading one

Unipute stops at the shader, so getting a `Vec<Particle>` into a buffer is the
host's job, the same as for a `Vec<f32>`. Because the struct is `#[repr(C)]`
with no hidden padding, a crate such as
[zerocopy](https://docs.rs/zerocopy) can turn a slice of them into bytes with
a derive and no `unsafe`:

```rust,ignore
#[derive(Layout, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    mass: f32,
}

let buffer = gpu.storage(&particles); // takes anything IntoBytes
```

That is what the [particles example](../examples.md#particles) does, and the
wgpu code behind it is in `examples/host/mod.rs`. A uniform buffer holding a
struct needs the usual sixteen byte size that uniforms have on every API, and
the example's helper pads to that.

`Particle::ty()` hands the struct back as IR, offsets included, if a host
wants to check a layout or build a descriptor from it.

## What is not here yet

- A struct from another crate. The kernel macro learns a struct's fields
  through a macro the derive leaves next to the struct, and that macro is
  only visible inside the crate that defined it. Define the struct in the
  crate with the kernels.
- Nested structs and arrays as fields.
- Structs as [workgroup memory](./workgroup-memory.md) are not tested yet.
