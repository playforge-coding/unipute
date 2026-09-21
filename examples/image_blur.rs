//! A two dimensional kernel, run over a picture you can see.
//!
//! The kernel blurs a single channel image. The host works out how many
//! workgroups cover it, runs the blur on a GPU through wgpu, prints the
//! picture before and after, and checks the result against the same filter
//! written as two plain loops.
//!
//! ```text
//! cargo run --example image_blur
//! ```
//!
//! The wgpu side is in `examples/host/mod.rs`. It is not part of Unipute.

mod host;

use std::process::ExitCode;

use unipute::{Kernel, kernel};

/// Blurs an image with a 3x3 tent filter, then mixes the result back over the
/// original by `strength`.
///
/// The image is a flat buffer, so the kernel turns its two dimensional
/// invocation id into an index itself. Taps that fall outside the picture are
/// clamped to the edge, which is what stops one side of it from bleeding into
/// the other.
#[kernel(workgroup_size(16, 16))]
fn blur(input: &[f32], output: &mut [f32], width: &u32, height: &u32, strength: &f32) {
    /// One axis of a tap, clamped to the image.
    ///
    /// `offset` runs 0, 1, 2 across a tap centred on `center`, so the
    /// coordinate wanted is `center + offset - 1`. Coordinates are unsigned,
    /// and the `max` before the subtraction is what keeps the left edge from
    /// wrapping around to the far side of the row.
    fn tap(center: u32, offset: u32, limit: u32) -> u32 {
        min(max(center + offset, 1u32) - 1u32, limit - 1u32)
    }

    /// The tent weight along one axis: 2 in the middle, 1 either side.
    fn axis_weight(offset: u32) -> f32 {
        if offset == 1u32 {
            return 2.0;
        }
        1.0
    }

    fn tap_weight(x_offset: u32, y_offset: u32) -> f32 {
        axis_weight(x_offset) * axis_weight(y_offset)
    }

    let x = global_id().x;
    let y = global_id().y;
    if x >= width || y >= height {
        return;
    }

    let mut total = 0.0f32;
    let mut total_weight = 0.0f32;
    for y_offset in 0..3u32 {
        let row = tap(y, y_offset, height) * width;
        for x_offset in 0..3u32 {
            let column = tap(x, x_offset, width);
            let weight = tap_weight(x_offset, y_offset);
            total += input[row + column] * weight;
            total_weight += weight;
        }
    }

    let index = y * width + x;
    output[index] = mix(input[index], total / total_weight, strength);
}

/// The picture: small enough to print, and with a hard edge and a single
/// bright pixel so the blur has something to show.
const WIDTH: u32 = 40;
const HEIGHT: u32 = 12;

fn main() -> ExitCode {
    let Some(gpu) = host::Gpu::open() else {
        eprintln!("no GPU adapter found, so there is nothing to run this on");
        return ExitCode::FAILURE;
    };
    println!("running on {}", gpu.describe());

    let image = picture();
    let strength = 1.0f32;

    let pipeline = gpu.pipeline::<blur>();
    let input = gpu.storage(&image);
    let output = gpu.storage(&vec![0.0f32; image.len()]);
    let width = gpu.uniform(&WIDTH);
    let height = gpu.uniform(&HEIGHT);
    let strength_buffer = gpu.uniform(&strength);

    // Neither axis is a multiple of 16, so the last group along each one has
    // invocations that land outside the image and return early.
    let groups = host::workgroups([WIDTH, HEIGHT, 1], blur::WORKGROUP_SIZE);
    let invocations = groups[0] * groups[1] * blur::WORKGROUP_SIZE[0] * blur::WORKGROUP_SIZE[1];
    println!(
        "{WIDTH}x{HEIGHT} image, {groups:?} workgroups, {} of {invocations} invocations do nothing",
        invocations - WIDTH * HEIGHT
    );
    println!();

    gpu.dispatch(
        &pipeline,
        &[&input, &output, &width, &height, &strength_buffer],
        groups,
    );
    let blurred = gpu.read::<f32>(&output);

    println!("before");
    print_picture(&image);
    println!();
    println!("after");
    print_picture(&blurred);
    println!();

    let expected = blur_on_the_cpu(&image, strength);
    let difference = blurred
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("largest difference from the CPU blur: {difference:e}");

    if difference < 1e-5 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// A dark field with a bright block in it and one lone bright pixel.
fn picture() -> Vec<f32> {
    let mut image = vec![0.0f32; (WIDTH * HEIGHT) as usize];
    for y in 3..9 {
        for x in 6..18 {
            image[(y * WIDTH + x) as usize] = 1.0;
        }
    }
    image[(6 * WIDTH + 30) as usize] = 1.0;
    image
}

/// The filter again, as the loops the kernel is a rearrangement of.
fn blur_on_the_cpu(image: &[f32], strength: f32) -> Vec<f32> {
    let (width, height) = (WIDTH as i64, HEIGHT as i64);
    let at = |x: i64, y: i64| {
        let x = x.clamp(0, width - 1);
        let y = y.clamp(0, height - 1);
        image[(y * width + x) as usize]
    };
    let mut output = Vec::with_capacity(image.len());
    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            let mut total_weight = 0.0;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let weight = (2 - dx.abs()) as f32 * (2 - dy.abs()) as f32;
                    total += at(x + dx, y + dy) * weight;
                    total_weight += weight;
                }
            }
            let original = at(x, y);
            output.push(original + (total / total_weight - original) * strength);
        }
    }
    output
}

/// Brightness as characters, so the blur is something to look at.
fn print_picture(image: &[f32]) {
    const RAMP: &[u8] = b" .:-=+*#%@";
    for row in image.chunks(WIDTH as usize) {
        let line: String = row
            .iter()
            .map(|value| {
                let level = (value.clamp(0.0, 1.0) * (RAMP.len() - 1) as f32).round() as usize;
                RAMP[level] as char
            })
            .collect();
        println!("  {line}");
    }
}
