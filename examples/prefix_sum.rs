//! An algorithm that does not fit in one dispatch.
//!
//! A prefix sum turns `[a, b, c, d]` into `[a, a+b, a+b+c, a+b+c+d]`. Each
//! element depends on the ones before it, so no single pass over the data can
//! produce it. This example builds one out of three kernels, runs them on a
//! GPU through wgpu, and checks the answer against a plain loop.
//!
//! ```text
//! cargo run --example prefix_sum
//! ```
//!
//! The wgpu side is in `examples/host/mod.rs`. It is not part of Unipute.

mod host;

use std::process::ExitCode;

use unipute::{Kernel, kernel};

/// One doubling pass of a Hillis and Steele scan.
///
/// Every invocation adds the element `step` places behind it to its own. Run
/// with `step` at 1, 2, 4 and so on until it passes the element count, and the
/// buffer holds an inclusive prefix sum. Reading and writing different buffers
/// is what keeps one invocation from seeing another's half finished work.
///
/// `step` is the only thing that changes between passes, so it is bound in its
/// own group. The host can then bind group 0 once and swap group 1 per pass.
/// It takes index 2 rather than 0 so that its binding number is unique across
/// both groups, which the GLSL target needs since GLSL has no groups.
#[kernel(workgroup_size(256))]
fn scan_step(input: &[f32], output: &mut [f32], #[binding(group = 1, index = 2)] step: &u32) {
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
///
/// It is not called `normalize` because that is a WGSL built-in, and naga
/// renames an entry point that collides with one.
#[kernel(workgroup_size(256))]
fn to_distribution(inclusive: &[f32], output: &mut [f32]) {
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

/// How many elements to scan.
const ELEMENTS: u32 = 4096;

fn main() -> ExitCode {
    let Some(gpu) = host::Gpu::open() else {
        eprintln!("no GPU adapter found, so there is nothing to run this on");
        return ExitCode::FAILURE;
    };
    println!("running on {}", gpu.describe());
    println!();

    // Small whole numbers, so every partial sum is exact in an f32 and the
    // comparison at the end can be an equality rather than a tolerance.
    let input: Vec<f32> = (0..ELEMENTS).map(|i| ((i * 7) % 13) as f32).collect();

    let scan = gpu.pipeline::<scan_step>();
    let exclusive = gpu.pipeline::<to_exclusive>();
    let distribution = gpu.pipeline::<to_distribution>();

    let a = gpu.storage(&input);
    let b = gpu.storage(&vec![0.0f32; input.len()]);
    let c = gpu.storage(&vec![0.0f32; input.len()]);
    let step = gpu.uniform(&1u32);

    let groups = host::workgroups([ELEMENTS, 1, 1], scan_step::WORKGROUP_SIZE);
    println!("{ELEMENTS} elements, {} workgroups a pass", groups[0]);

    // The scan passes ping pong between two buffers, so the one written last
    // is the one the next pass reads.
    let (mut source, mut destination) = (&a, &b);
    let mut stride = 1;
    let mut passes = 0;
    while stride < ELEMENTS {
        gpu.write(&step, &stride);
        gpu.dispatch(&scan, &[source, destination, &step], groups);
        std::mem::swap(&mut source, &mut destination);
        stride *= 2;
        passes += 1;
    }
    println!("{passes} scan passes, then one to shift and one to normalise");
    println!();

    // `source` now holds the inclusive scan. The other two kernels each read
    // it and write somewhere else, so it is never overwritten.
    gpu.dispatch(&exclusive, &[source, destination], groups);
    gpu.dispatch(&distribution, &[source, &c], groups);

    let inclusive_result = gpu.read::<f32>(source);
    let exclusive_result = gpu.read::<f32>(destination);
    let normalized_result = gpu.read::<f32>(&c);

    print_row("input", &input);
    print_row("inclusive", &inclusive_result);
    print_row("exclusive", &exclusive_result);
    print_row("normalized", &normalized_result);
    println!();

    let (expected_inclusive, expected_exclusive, expected_normalized) = on_the_cpu(&input);
    let inclusive_ok = inclusive_result == expected_inclusive;
    let exclusive_ok = exclusive_result == expected_exclusive;
    let normalized_ok = max_difference(&normalized_result, &expected_normalized) < 1e-6;

    println!(
        "matches the CPU: inclusive {inclusive_ok}, exclusive {exclusive_ok}, normalized {normalized_ok}"
    );
    if inclusive_ok && exclusive_ok && normalized_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// The same three results, computed the boring way.
fn on_the_cpu(input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut inclusive = Vec::with_capacity(input.len());
    let mut running = 0.0f32;
    for value in input {
        running += value;
        inclusive.push(running);
    }
    let mut exclusive = vec![0.0f32];
    exclusive.extend_from_slice(&inclusive[..inclusive.len() - 1]);
    let total = running;
    let normalized = inclusive.iter().map(|value| value / total).collect();
    (inclusive, exclusive, normalized)
}

fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The first few and the last element, which is where a scan shows its work.
fn print_row(name: &str, values: &[f32]) {
    let head: Vec<String> = values.iter().take(8).map(|v| format!("{v:>7.3}")).collect();
    println!(
        "  {name:<10} {} ... {:>10.3}",
        head.join(" "),
        values[values.len() - 1]
    );
}
