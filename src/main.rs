#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

use std::time::Instant;

use colors_transform::{Color, Hsl};
use image::{Rgb, RgbImage};
use num::complex::ComplexFloat;
use num::Complex;
use rayon::prelude::*;

const MAX_ITERATIONS: usize = 128;

fn get_color(count: f64) -> Rgb<u8> {
    let hue = (count / (MAX_ITERATIONS as f64) * 360.0).round() as f32;
    let saturation = 100.0;
    let lightness = if count < (MAX_ITERATIONS as f64) {
        50.0
    } else {
        0.0
    };
    let (r, g, b) = Hsl::from(hue, saturation, lightness).to_rgb().as_tuple();

    // if count < MAX_ITERATIONS {
    //     println!("HSV {:.2} {:.2} {:.2} RGB {:.2} {:.2} {:.2}", hue, saturation, value, r, g, b);
    //     // assert!(1 == 0);
    // }

    // return Rgb([hue as u8, hue as u8, hue as u8]);
    return Rgb([b as u8, g as u8, r as u8]);
}

/// Try to determine if `c` is in the Mandelbrot set, using at most `limit`
/// iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of
/// iterations it took for `c` to leave the circle of radius two centered on the
/// origin. If `c` seems to be a member (more precisely, if we reached the
/// iteration limit without being able to prove that `c` is not a member),
/// return `None`.
fn escape_time(c: Complex<f64>, limit: usize) -> Option<(usize, f64)> {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some((i, (i as f64) + 1.0 - z.abs().ln().log2()));
        }
        z = z * z + c;
    }

    None
}

/// Given the row and column of a pixel in the output image, return the
/// corresponding point on the complex plane.
///
/// `bounds` is a pair giving the width and height of the image in pixels.
/// `pixel` is a (column, row) pair indicating a particular pixel in that image.
/// The `upper_left` and `lower_right` parameters are points on the complex
/// plane designating the area our image covers.
fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64, // Why subtraction here? pixel.1 increases as we go down,
                                                                       // but the imaginary component increases as we go up.
    }
}

/// Render a rectangle of the Mandelbrot set into a buffer of pixels.
///
/// The `bounds` argument gives the width and height of the buffer `pixels`,
/// which holds one grayscale pixel per byte. The `upper_left` and `lower_right`
/// arguments specify points on the complex plane corresponding to the upper-
/// left and lower-right corners of the pixel buffer.
fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match escape_time(point, MAX_ITERATIONS) {
                None => Rgb([0, 0, 0]),
                Some((_, count)) => get_color(count),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

fn main() {
    let bounds = (10000 as usize, 8000 as usize);
    let upper_left = Complex { re: -2.0, im: 1.0 };
    let lower_right = Complex { re: 0.5, im: -1.0 };

    let mut pixels = RgbImage::new(bounds.0 as u32, bounds.1 as u32);

    // Scope of slicing up `pixels` into horizontal bands.
    println!("Executing...");
    let now = Instant::now();
    {
        let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(3 * bounds.0).enumerate().collect();

        bands.into_par_iter().for_each(|(i, band)| {
            let top = i;
            let band_bounds = (bounds.0, 1);
            let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
            let band_lower_right =
                pixel_to_point(bounds, (bounds.0, top + 1), upper_left, lower_right);
            render(band, band_bounds, band_upper_left, band_lower_right);
        });
    }
    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    pixels.save("mandelbrot.png").unwrap();
}
