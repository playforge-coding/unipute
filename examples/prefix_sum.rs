//! An algorithm that does not fit in one dispatch.
//!
//! A prefix sum turns `[a, b, c, d]` into `[a, a+b, a+b+c, a+b+c+d]`. Each
//! element depends on the ones before it, so no single pass over the data can
//! produce it. This example builds one out of three kernels and prints the
//! dispatch plan a host would follow.
//!
//! ```text
//! cargo run --example prefix_sum
//! ```

use unipute::{Kernel, WgslKernel, kernel};

/// One doubling pass of a Hillis and Steele scan.
///
/// Every invocation adds the element `step` places behind it to its own. Run
/// with `step` at 1, 2, 4 and so on until it passes the element count, and the
/// buffer holds an inclusive prefix sum. Reading and writing different buffers
/// is what keeps one invocation from seeing another's half finished work.
///
/// `step` is the only thing that changes between passes, so it is bound in its
/// own group. The host can then bind group 0 once and swap group 1 per pass.
#[kernel(workgroup_size(256))]
fn scan_step(input: &[f32], output: &mut [f32], #[binding(group = 1, index = 0)] step: &u32) {
    let index = global_id().x;
    if index >= input.len() {
        return;
    }
    let mut value = input[index];
    if index >= step {
        value += input[index - step];
    }
    output[index] = value;
}

/// Turns an inclusive scan into an exclusive one by shifting it along.
///
/// An exclusive scan answers "how much came before me", which is what you want
/// when the sums are offsets into somewhere else. Run last, since shifting the
/// buffer moves the total out of the end of it.
#[kernel(workgroup_size(256))]
fn to_exclusive(inclusive: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    if index >= inclusive.len() {
        return;
    }
    if index == 0u32 {
        output[index] = 0.0;
    } else {
        output[index] = inclusive[index - 1u32];
    }
}

/// Divides a scan through by its total, giving a cumulative distribution.
///
/// The total is the last element of the inclusive scan, so this reads one
/// element every invocation shares. A distribution that sums to zero has no
/// shape to sample, and the guard is what keeps that from dividing by zero.
#[kernel(workgroup_size(256))]
fn normalize(inclusive: &[f32], output: &mut [f32]) {
    let index = global_id().x;
    let count = inclusive.len();
    if index >= count {
        return;
    }
    let total = inclusive[count - 1u32];
    if total <= 0.0 {
        output[index] = 0.0;
        return;
    }
    output[index] = inclusive[index] / total;
}

/// How many elements the printed plan is for.
const ELEMENTS: u32 = 4096;

fn main() {
    println!("three kernels, one prefix sum");
    for (name, size, bindings) in [
        (
            scan_step::NAME,
            scan_step::WORKGROUP_SIZE,
            scan_step::BINDINGS,
        ),
        (
            to_exclusive::NAME,
            to_exclusive::WORKGROUP_SIZE,
            to_exclusive::BINDINGS,
        ),
        (
            normalize::NAME,
            normalize::WORKGROUP_SIZE,
            normalize::BINDINGS,
        ),
    ] {
        let groups: Vec<String> = bindings
            .iter()
            .map(|binding| format!("{}:{}={}", binding.group, binding.binding, binding.name))
            .collect();
        println!(
            "  {name:<12} workgroup {:<4} {}",
            size[0],
            groups.join("  ")
        );
    }
    println!();

    print_plan(ELEMENTS);
    println!();

    println!("--- WGSL for {} ---", scan_step::NAME);
    println!("{}", scan_step::WGSL);
}

/// Prints every dispatch needed to scan `elements` items.
fn print_plan(elements: u32) {
    let groups = elements.div_ceil(scan_step::WORKGROUP_SIZE[0]);
    println!("plan for {elements} elements, {groups} workgroups a pass");

    // The scan passes ping pong between two buffers, so the one written last
    // is the one the next pass reads.
    let mut pass = 1;
    let mut step = 1;
    let mut source = 'a';
    while step < elements {
        let destination = other(source);
        println!(
            "  pass {pass:<2} {:<12} step {step:<5} {source} -> {destination}",
            scan_step::NAME
        );
        source = destination;
        step *= 2;
        pass += 1;
    }

    // The last two passes each read what the one before wrote, so they keep
    // alternating buffers in the same way. `normalize` goes first: the total
    // it divides by is the last element of the inclusive scan, and shifting
    // the buffer along would push it out of reach.
    for name in [normalize::NAME, to_exclusive::NAME] {
        let destination = other(source);
        println!(
            "  pass {pass:<2} {name:<12} {:<10} {source} -> {destination}",
            ""
        );
        source = destination;
        pass += 1;
    }

    println!("  result in buffer {source}, after {} dispatches", pass - 1);
}

/// The other half of the ping pong pair.
fn other(buffer: char) -> char {
    if buffer == 'a' { 'b' } else { 'a' }
}
