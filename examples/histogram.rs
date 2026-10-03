//! Atomics: many invocations updating the same place without losing a count.
//!
//! A brightness histogram of an image. Every pixel lands in one of sixteen
//! bins, and thousands of invocations want to add one to the same few bins at
//! the same moment. A plain `bins[b] += 1` would read, add and write with
//! other invocations doing the same in between, and the total would come up
//! short. `fetch_add` on an atomic does the three as one step.
//!
//! The kernel counts into a workgroup copy of the bins first and adds the
//! whole copy into the buffer once per workgroup, since a workgroup atomic is
//! cheaper than one in a buffer and there are 256 pixels per group but only
//! 16 bins. That is the shape most atomic kernels take: contend locally, then
//! merge.
//!
//! The host runs it on a GPU through wgpu, prints the histogram as bars, and
//! checks every bin against a count made on the CPU.
//!
//! ```text
//! cargo run --example histogram
//! ```
//!
//! The wgpu side is in `examples/host/mod.rs`. It is not part of Unipute.

mod host;

use std::process::ExitCode;

use unipute::{Kernel, kernel};

/// The number of bins. Written out in the kernel as well, since a `const` is
/// not visible in there.
const BINS: usize = 16;

/// Counts every pixel of `image` into `bins` by brightness, with the same
/// number of dispatched invocations as there are pixels or more.
#[kernel(workgroup_size(256))]
fn brightness_histogram(image: &[f32], bins: &mut [AtomicU32]) {
    #[workgroup]
    let local_bins: [AtomicU32; 16];

    let index = global_id().x;
    let lane = local_index();

    // Count into the workgroup's bins. Any invocation past the end of the
    // image skips this but still reaches the barrier below.
    if index < image.len() {
        let bin = min(image[index] * 16.0, 15.0) as u32;
        local_bins[bin].fetch_add(1u32);
    }
    workgroup_barrier();

    // Sixteen lanes carry the sixteen bins out. `load` reads the workgroup
    // atomic and `fetch_add` adds it into the buffer one, which other
    // workgroups are adding into at the same time.
    if lane < 16u32 {
        let count = local_bins[lane].load();
        if count > 0u32 {
            bins[lane].fetch_add(count);
        }
    }
}

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

fn main() -> ExitCode {
    let Some(gpu) = host::Gpu::open() else {
        eprintln!("no GPU adapter found, so there is nothing to run this on");
        return ExitCode::FAILURE;
    };
    println!("running on {}", gpu.describe());

    let image = picture();
    let expected = count_on_the_cpu(&image);

    let pipeline = gpu.pipeline::<brightness_histogram>();
    let image_buffer = gpu.storage(&image);
    // An atomic is a plain `u32` to the host. Nothing here knows the kernel
    // treats the buffer any differently.
    let bins_buffer = gpu.storage(&[0u32; BINS]);
    let groups = host::workgroups(
        [image.len() as u32, 1, 1],
        brightness_histogram::WORKGROUP_SIZE,
    );
    println!(
        "{} pixels in {} workgroups, {} atomic adds into the buffer",
        image.len(),
        groups[0],
        groups[0] as usize * BINS
    );
    println!();
    gpu.dispatch(&pipeline, &[&image_buffer, &bins_buffer], groups);
    let bins = gpu.read::<u32>(&bins_buffer);

    let tallest = *bins.iter().max().unwrap_or(&1) as f32;
    for (bin, count) in bins.iter().enumerate() {
        let bar = "#".repeat((count.min(&u32::MAX) * 50) as usize / tallest.max(1.0) as usize);
        println!(
            "  {:>4.2} to {:>4.2}  {count:>7}  {bar}",
            bin as f32 / BINS as f32,
            (bin + 1) as f32 / BINS as f32
        );
    }
    println!();

    let total: u32 = bins.iter().sum();
    println!("{total} pixels counted, {} in the image", image.len());
    if bins == expected {
        println!("every bin matches the CPU");
        ExitCode::SUCCESS
    } else {
        println!("the bins differ from the CPU: {expected:?}");
        ExitCode::FAILURE
    }
}

/// A soft gradient with a bright disc in it, so the histogram has a shape.
fn picture() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let gradient = x as f32 / WIDTH as f32 * 0.6;
            let dx = x as f32 - WIDTH as f32 * 0.6;
            let dy = y as f32 - HEIGHT as f32 * 0.5;
            let disc = if dx * dx + dy * dy < 120.0 * 120.0 {
                0.35
            } else {
                0.0
            };
            pixels.push((gradient + disc).min(1.0));
        }
    }
    pixels
}

/// The kernel's count, done the slow way.
fn count_on_the_cpu(image: &[f32]) -> Vec<u32> {
    let mut bins = vec![0u32; BINS];
    for value in image {
        let bin = (value * BINS as f32).min(BINS as f32 - 1.0) as usize;
        bins[bin] += 1;
    }
    bins
}
