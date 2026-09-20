# Unipute

## Overview

A compile time generating, language and graphic API agnostic, GPU library.

## Tests

Always write tests when needed (only when needed though; don't write tests for something as trivial as, say, a hello world).

## Code Quality

Always lint and format:

```bash
cargo fmt && cargo clippy
```

If a clippy error already existed, you don't need to fix it, but do note it and make sure you draft (but don't post) an issue for it.

## Docs (IMPORTANT!!!)

Always update the docs if necessary (again, not if unnecessary) when changing or adding code. If the docs already contain outdated information, you don't need to fix it, but do note it and make sure you draft (but don't post) an issue for it.

## Toolchain

`rust-toolchain.toml` pins nightly and `.cargo/config.toml` builds the dev profile with the cranelift codegen backend. Cranelift is the only reason for nightly. It makes dev builds faster, and the cargo keys that select it are unstable.

So: everything user facing and everything contributor facing must be stable Rust. No nightly language or library features in the API, the macro, the tests or the examples. If something looks like it needs one, that is a design problem to solve rather than a feature to use.

Keep the cranelift keys in `.cargo/config.toml` and out of `Cargo.toml`. A manifest with `cargo-features` in it cannot be read by stable cargo, which breaks every project depending on Unipute. A cargo config only applies inside this repository.

The CI job named "Stable toolchain" builds a crate against Unipute on stable and is what catches a regression here.

## Outdated packages

PLEASE, PLEASE do not use a package that is old or deprecated.

## Keep things human

Do not use em dashes or other special symbols not normally found in writing. Do not word things in a weird way. Keep it looking human.