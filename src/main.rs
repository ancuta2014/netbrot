// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

use std::time::Instant;

use clap::{Parser, ValueEnum, ValueHint};
use colors_transform::{Color, Hsl};
use image::{Rgb, RgbImage};
use nalgebra::{matrix, SMatrix, SVector};
use num::Complex;
use rayon::prelude::*;

type ComplexMatrix = SMatrix<Complex<f64>, 2, 2>;
type ComplexVector = SVector<Complex<f64>, 2>;

const MAX_ESCAPE_RADIUS: f64 = 100.0;
const MAX_ESCAPE_RADIUS_SQUARED: f64 = MAX_ESCAPE_RADIUS * MAX_ESCAPE_RADIUS;
const MAX_PERIODS: usize = 20;
const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;
const MAX_ITERATIONS: usize = 128;

// https://graphicdesign.stackexchange.com/a/158793
const _COLOR_PALLETTE_V1: [Rgb<u8>; 32] = [
    Rgb([173, 216, 230]),
    Rgb([0, 191, 255]),
    Rgb([30, 144, 255]),
    Rgb([0, 0, 255]),
    Rgb([0, 0, 139]),
    Rgb([72, 61, 139]),
    Rgb([123, 104, 238]),
    Rgb([138, 43, 226]),
    Rgb([128, 0, 128]),
    Rgb([218, 112, 214]),
    Rgb([255, 0, 255]),
    Rgb([255, 20, 147]),
    Rgb([176, 48, 96]),
    Rgb([220, 20, 60]),
    Rgb([240, 128, 128]),
    Rgb([255, 69, 0]),
    Rgb([255, 165, 0]),
    Rgb([244, 164, 96]),
    Rgb([240, 230, 140]),
    Rgb([128, 128, 0]),
    Rgb([139, 69, 19]),
    Rgb([255, 255, 0]),
    Rgb([154, 205, 50]),
    Rgb([124, 252, 0]),
    Rgb([144, 238, 144]),
    Rgb([143, 188, 143]),
    Rgb([34, 139, 34]),
    Rgb([0, 255, 127]),
    Rgb([0, 255, 255]),
    Rgb([0, 139, 139]),
    Rgb([128, 128, 128]),
    Rgb([255, 255, 255]),
];

// https://lospec.com/palette-list/endesga-32
const _COLOR_PALLETTE_V2: [Rgb<u8>; 32] = [
    Rgb([190, 74, 47]),
    Rgb([215, 118, 67]),
    Rgb([234, 212, 170]),
    Rgb([228, 166, 114]),
    Rgb([184, 111, 80]),
    Rgb([115, 62, 57]),
    Rgb([62, 39, 49]),
    Rgb([162, 38, 51]),
    Rgb([228, 59, 68]),
    Rgb([247, 118, 34]),
    Rgb([254, 174, 52]),
    Rgb([254, 231, 97]),
    Rgb([99, 199, 77]),
    Rgb([62, 137, 72]),
    Rgb([38, 92, 66]),
    Rgb([25, 60, 62]),
    Rgb([18, 78, 137]),
    Rgb([0, 153, 219]),
    Rgb([44, 232, 245]),
    Rgb([192, 203, 220]),
    Rgb([139, 155, 180]),
    Rgb([90, 105, 136]),
    Rgb([58, 68, 102]),
    Rgb([38, 43, 68]),
    Rgb([24, 20, 37]),
    Rgb([255, 0, 68]),
    Rgb([104, 56, 108]),
    Rgb([181, 80, 136]),
    Rgb([246, 117, 122]),
    Rgb([232, 183, 150]),
    Rgb([194, 133, 105]),
    Rgb([255, 255, 255]),
];

const _COLOR_PALLETTE_V3: [Rgb<u8>; 32] = [
    Rgb([75, 0, 85]),
    Rgb([123, 0, 140]),
    Rgb([134, 0, 151]),
    Rgb([56, 0, 163]),
    Rgb([0, 0, 181]),
    Rgb([0, 0, 213]),
    Rgb([0, 56, 221]),
    Rgb([0, 125, 221]),
    Rgb([0, 146, 221]),
    Rgb([0, 160, 199]),
    Rgb([0, 170, 168]),
    Rgb([0, 170, 144]),
    Rgb([0, 163, 83]),
    Rgb([0, 154, 0]),
    Rgb([0, 175, 0]),
    Rgb([0, 199, 0]),
    Rgb([0, 220, 0]),
    Rgb([0, 242, 0]),
    Rgb([44, 255, 0]),
    Rgb([176, 255, 0]),
    Rgb([216, 245, 0]),
    Rgb([241, 231, 0]),
    Rgb([252, 210, 0]),
    Rgb([255, 177, 0]),
    Rgb([255, 129, 0]),
    Rgb([255, 33, 0]),
    Rgb([241, 0, 0]),
    Rgb([219, 0, 0]),
    Rgb([208, 0, 0]),
    Rgb([204, 76, 76]),
    Rgb([204, 204, 204]),
    Rgb([0, 0, 0]),
];

macro_rules! c64 {
    ($re: literal) => {
        Complex { re: $re, im: 0.0 }
    };
}

fn get_orbit_color(count: f64) -> Rgb<u8> {
    let hue = (count / (MAX_ITERATIONS as f64) * 360.0).round() as f32;
    let saturation = 100.0;
    let lightness = if count < (MAX_ITERATIONS as f64) {
        50.0
    } else {
        0.0
    };
    let (r, g, b) = Hsl::from(hue, saturation, lightness).to_rgb().as_tuple();

    return Rgb([b as u8, g as u8, r as u8]);
}

fn get_period_color(period: usize) -> Rgb<u8> {
    if 1 <= period && period < MAX_PERIODS - 1 {
        _COLOR_PALLETTE_V3[period - 1]
    } else {
        Rgb([0, 0, 0])
    }
}

/// Try to determine if `c` is in the Mandelbrot set, using at most `limit`
/// iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of
/// iterations it took for `c` to leave the circle of radius two centered on the
/// origin. If `c` seems to be a member (more precisely, if we reached the
/// iteration limit without being able to prove that `c` is not a member),
/// return `None`.

fn escape_time(
    c: Complex<f64>,
    mat: ComplexMatrix,
    limit: usize,
) -> (Option<usize>, ComplexVector) {
    let mut z = ComplexVector::repeat(Complex { re: 0.0, im: 0.0 });
    let mut matz = ComplexVector::repeat(Complex { re: 0.0, im: 0.0 });

    for i in 0..limit {
        if z.norm_squared() > MAX_ESCAPE_RADIUS_SQUARED {
            return (Some(i), z);
        }
        z = matz.component_mul(&matz).add_scalar(c);
        matz = mat * z;
    }

    (None, z)
}

fn escape_period(c: Complex<f64>, mat: ComplexMatrix, limit: usize) -> Option<usize> {
    match escape_time(c, mat, limit) {
        (None, z) => {
            // When the limit was reached but the point did not escape, we look
            // for a period in a very naive way.
            let mut matz = mat * z;
            let mut z_period = vec![ComplexVector::zeros(); PERIOD_WINDOW];

            // Evaluate some more points
            z_period[0] = z;
            for i in 1..PERIOD_WINDOW {
                z_period[i] = matz.component_mul(&matz).add_scalar(c);
                matz = mat * z_period[i];
            }

            // Check newly evaluated points for periodicity
            for i in 2..MAX_PERIODS {
                let mut z_period_norm = 0.0;
                for j in 0..i - 1 {
                    z_period_norm += (z_period[j] - z_period[i + j - 1]).norm_squared();
                }

                if z_period_norm.sqrt() < 1.0e-5 {
                    return Some(i - 1);
                }
            }

            Some(MAX_PERIODS - 1)
        }
        (Some(_), _) => None,
    }
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
        // Why subtraction here? pixel.1 increases as we go down,
        re: upper_left.re + (pixel.0 as f64) * width / (bounds.0 as f64),
        // but the imaginary component increases as we go up.
        im: upper_left.im - (pixel.1 as f64) * height / (bounds.1 as f64),
    }
}

/// Render a rectangle of the Mandelbrot set into a buffer of pixels.
///
/// The `bounds` argument gives the width and height of the buffer `pixels`,
/// which holds one grayscale pixel per byte. The `upper_left` and `lower_right`
/// arguments specify points on the complex plane corresponding to the upper-
/// left and lower-right corners of the pixel buffer.
fn render_orbit(
    pixels: &mut [u8],
    mat: ComplexMatrix,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match escape_time(point, mat, MAX_ITERATIONS) {
                (None, _) => Rgb([0, 0, 0]),
                // https://linas.org/art-gallery/escape/escape.html
                (Some(n), z) => get_orbit_color((n as f64) + 1.0 - z.norm().ln().log2()),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

fn render_period(
    pixels: &mut [u8],
    mat: ComplexMatrix,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match escape_period(point, mat, MAX_ITERATIONS) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(period),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version, about)]
struct Cli {
    /// If given, plot periods instead of orbits
    #[arg(short, long, value_enum, default_value = "orbit")]
    color: ColorType,

    /// Resolution of the resulting image
    #[arg(short, long, default_value_t = 8000)]
    resolution: u32,

    /// Output file name
    #[arg(last = true, value_hint = ValueHint::FilePath)]
    filename: String,
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum ColorType {
    /// Plot orbits.
    Orbit,
    /// Plot periodicity for orbits that do not escape.
    Period,
}

fn main() {
    let args = Cli::parse();

    let color_type = args.color;
    let filename = args.filename;
    println!("Coloring: {:?}", color_type);

    // Full brot interval
    let upper_left = Complex { re: -0.9, im: 0.6 };
    let lower_right = Complex { re: 0.5, im: -0.6 };
    // Baby brot interval
    // let upper_left = Complex {
    //     re: -1.025,
    //     im: 0.025,
    // };
    // let lower_right = Complex {
    //     re: -0.975,
    //     im: -0.025,
    // };
    println!(
        "Bounding box: Top left {} Bottom right {}",
        upper_left, lower_right
    );

    let ratio = (lower_right.re - upper_left.re) / (upper_left.im - lower_right.im);
    let resolution = args.resolution as f64;
    let bounds = ((ratio * resolution).round() as usize, resolution as usize);
    println!("Resolution: {}x{}", bounds.0, bounds.1);

    let mut pixels = RgbImage::new(bounds.0 as u32, bounds.1 as u32);

    // let mat = matrix![
    //     c64!(1.0), c64!(0.0), c64!(0.0);
    //     c64!(-1.0), c64!(1.0), c64!(0.0);
    //     c64!(1.0), c64!(1.0), c64!(-1.0);
    // ];
    let mat = matrix![
        c64!(1.0), c64!(0.8);
        c64!(1.0), c64!(-0.5);
    ];
    // let mat = matrix![
    //     c64!(1.0), c64!(1.0);
    //     c64!(0.0), c64!(1.0);
    // ];

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

            match color_type {
                ColorType::Orbit => {
                    render_orbit(band, mat, band_bounds, band_upper_left, band_lower_right)
                }
                ColorType::Period => {
                    render_period(band, mat, band_bounds, band_upper_left, band_lower_right)
                }
            }
        });
    }
    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    pixels.save(filename).unwrap();
}
