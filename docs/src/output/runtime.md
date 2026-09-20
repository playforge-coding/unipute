# Generating at run time

Everything so far happened during compilation. Sometimes that is not what you
want.

## When you need this

- A tool that takes a language on the command line and prints the shader
- An editor or asset pipeline that shows several targets side by side
- A program that picks a backend after asking the machine what it supports

In all three the target is not known until the program runs, so a constant is
the wrong shape.

## Turning it on

```toml
[dependencies]
unipute = { version = "0.1", features = ["runtime", "spv", "msl"] }
```

The `runtime` feature pulls in the naga backend as a normal dependency rather
than only as a compile time one. Which languages it can produce still depends
on the target features, so enable the ones you want.

## Using it

Every kernel can hand back its IR, and the IR is what the backend takes:

```rust
# #[cfg(feature = "runtime")]
# fn demo() {
use unipute::{Kernel, Target, kernel};

#[kernel(workgroup_size(64))]
fn double(input: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index < input.len() {
        output[index] = input[index] * 2.0;
    }
}

let ir = double::ir();
let wgsl = unipute::compile_text(&ir, Target::Wgsl).unwrap();
assert!(wgsl.contains("@compute"));
# }
```

`compile_text` covers the text languages. SPIR-V is not text, so it has its
own function:

```rust,ignore
let words: Vec<u32> = unipute::compile_spirv(&double::ir())?;
```

## Picking a target by name

`Target::from_name` turns a string into a target, which is what a command line
flag needs:

```rust
# #[cfg(feature = "runtime")]
# fn demo() {
use unipute::Target;

let wanted = "msl";
match Target::from_name(wanted) {
    Some(target) => println!("generating {target}"),
    None => println!("no target called {wanted}"),
}
# }
```

## Knowing what this build can do

There are two different questions, and they have two different answers.

`Target::is_implemented` asks whether Unipute has a backend for it at all. It
is false for `Ptx`, which is reserved and not written yet.

`enabled_targets` asks what this particular binary can actually produce, which
depends on the features it was built with:

```rust
# #[cfg(feature = "runtime")]
# fn demo() {
for target in unipute::enabled_targets() {
    println!("can generate {target}");
}
# }
```

Use `enabled_targets` when reporting to a user, since it is the honest answer
about the program in front of them. Asking for a target that exists but was
not compiled in gives a clear error rather than a panic:

```text
target msl is not available, enable the "msl" feature
```

## It is the same pipeline

The compile time path and the run time path are the same code. The only
difference is when it runs. The test suite checks this by generating a kernel
both ways and comparing the strings, so if the two ever drift apart, that test
fails.

You can rely on that. Prototyping through the run time path and then switching
to constants will not change your shaders. The `retarget` example takes a
target name on the command line and ends by making that comparison in front of
you. See [Examples](../examples.md).

## Inspecting a kernel without the runtime feature

`Kernel::ir()` does not need the `runtime` feature. Only translating does. So
you can always look at what a kernel contains:

```rust
# use unipute::{Kernel, kernel};
# #[kernel(workgroup_size(64))]
# fn double(input: &[f32], output: &mut [f32]) {
#     let index = global_id().x;
#     if index < input.len() { output[index] = input[index] * 2.0; }
# }
# fn main() {
let ir = double::ir();
println!("{} resources, {} statements", ir.resources.len(), ir.body.len());
# }
```

This is useful for writing your own tooling, and it is the same door a future
backend would come in through. The `inspect_ir` example walks a whole kernel
this way, printing its body back out and counting what is in it.
