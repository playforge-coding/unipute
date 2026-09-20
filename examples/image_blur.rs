//! A two dimensional kernel, and the host side arithmetic that goes with one.
//!
//! The kernel blurs a single channel image. Around it is everything a host has
//! to work out before it can dispatch: where each buffer is bound, and how many
//! workgroups cover the picture.
//!
//! ```text
//! cargo run --example image_blur
//! ```

use unipute::{Access, Kernel, WgslKernel, kernel};

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

/// A picture to plan a dispatch for.
const IMAGE: [u32; 2] = [1920, 1080];

fn main() {
    println!("kernel `{}`", blur::NAME);
    println!("workgroup size {:?}", blur::WORKGROUP_SIZE);
    println!();

    print_layout();
    println!();
    print_dispatch();
    println!();

    println!("--- WGSL ---");
    println!("{}", blur::WGSL);
}

/// Prints what the host has to bind, read off the kernel rather than written
/// out by hand a second time.
fn print_layout() {
    println!("layout");
    for binding in blur::BINDINGS {
        let kind = match binding.access {
            Access::Read => "storage, read only",
            Access::ReadWrite => "storage, read and write",
            Access::Uniform => "uniform",
        };
        println!(
            "  group {} binding {}: {:<8} {kind}",
            binding.group, binding.binding, binding.name
        );
    }
}

/// Prints the dispatch this kernel needs for [`IMAGE`].
fn print_dispatch() {
    let [width, height] = IMAGE;
    let size = blur::WORKGROUP_SIZE;
    let groups = [width.div_ceil(size[0]), height.div_ceil(size[1]), 1];
    let invocations = groups[0] * groups[1] * size[0] * size[1];
    let pixels = width * height;

    println!("dispatch for a {width}x{height} image");
    println!("  workgroups {groups:?}");
    println!("  invocations {invocations}, pixels {pixels}");
    println!(
        "  {} invocations land outside the image and return early",
        invocations - pixels
    );
}
