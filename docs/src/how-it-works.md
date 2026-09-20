# How it works

You do not need this chapter to use Unipute. It is here because knowing the
shape of the thing makes its error messages and its limits make more sense.

## Three stages

```text
your Rust function
        |
        |  the #[kernel] macro reads it
        v
   Unipute IR          a small description of what the kernel does
        |
        |  a backend reads that
        v
WGSL, SPIR-V, MSL, HLSL, GLSL
```

The middle stage is the whole trick. It is a plain data description of a
kernel: which buffers there are, which variables, and what the body does. It
knows nothing about Rust and nothing about any shader language.

That has a practical consequence. The front end and the backend never talk to
each other, so either can be replaced. A backend for a completely different
kind of target, or a front end that reads a different language, both plug into
the same middle without disturbing the other side.

## What the macro does

When the compiler reaches your `#[kernel]` function, the macro runs. In order:

1. **Reads the attribute.** Workgroup size, and a name if you set one.
2. **Reads the signature.** Each parameter becomes a buffer, with its access
   worked out from whether it is `&[T]`, `&mut [T]` or neither, and its
   binding from its position or its `#[binding(...)]`.
3. **Reads the body.** Statements and expressions become IR, with names
   resolved against a scope stack as it goes. This is where an unknown
   function or an unsupported statement is caught, which is why those errors
   point at the exact line.
4. **Runs the backend, during your compilation.** For each target feature you
   enabled, the IR is lowered to naga IR, validated, and written out.
5. **Emits the replacement.** Your function is gone, replaced by a type of the
   same name carrying the shaders and the layout, plus a method that rebuilds
   the IR.

Step 4 is the unusual one. The shader compiler runs inside the proc macro, so
the finished shader is a string literal in your crate by the time your code is
type checked.

## Why your function disappears

A kernel is not a function your CPU can call. It describes work for a
different processor with a different memory model. Leaving a callable Rust
function behind would be offering something that cannot work.

So the macro replaces it with a type of the same name. `scale` stops being a
function and becomes a type carrying `scale::WGSL`, `scale::BINDINGS` and the
rest. Running kernels on the CPU, which would be genuinely useful for testing,
is on the roadmap and not here yet.

## Where validation happens

Three different things can reject a kernel, at three different points, and the
message tells you which.

**The macro, reading your body.** Unsupported syntax, an unknown function, a
name that is not in scope, two parameters on the same binding. These point at
your source with a span, and are the ones you will see most.

**Lowering, building naga IR.** Things that are structurally wrong, like a
zero workgroup dimension or a uniform declared as a slice. Still a compile
error against your kernel.

**Naga's validator.** If a module Unipute built gets this far and is rejected,
that means Unipute built something wrong. The message says the module was
rejected and passes naga's own text through. If you see one of these it is a
bug worth reporting, not something to work around.

## Why naga

Naga is the shader translator from the wgpu project. It already knows how to
write five languages and, importantly, how to validate a module before
writing. Reimplementing that would have been a lot of work to arrive somewhere
worse.

Unipute does not expose naga in its public API, though, and `unipute-ir` does
not depend on it. That is deliberate. A target that naga has no concept of,
such as PTX, would be a sibling of the naga backend rather than something
bolted onto it.

## Why the generated shaders look like that

If you have read the output, you will have noticed a lot of this:

```wgsl
let _e9 = index;
let _e11 = index;
let _e13 = input[_e11];
```

Those temporaries are naga's doing, not Unipute's. Naga keeps expressions in a
flat list where each one is named, and its writers print that list literally
rather than trying to reconstruct nested expressions.

It is noisier than handwritten code and completely equivalent. Every shader
compiler downstream folds it away immediately. If you are reading the output
to check your kernel, read past the temporaries and look at the stores and the
control flow.

## What the crates are

| Crate | Job |
| ----- | --- |
| `unipute` | what you depend on: the macro, the traits, the layout types |
| `unipute-ir` | the IR. No dependencies at all |
| `unipute-macros` | the Rust front end |
| `unipute-naga` | the naga backend |

`unipute-ir` having no dependencies is load bearing rather than tidiness. It
is what lets a new front end or backend be added without inheriting a shader
compiler it has no use for.

For the level of detail a contributor needs, including exactly where a new
backend or a graphics stage would attach, see the [architecture
notes](./architecture.md).
