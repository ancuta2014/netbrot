// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::Complex;

use image::Rgb;
use nalgebra::allocator::Allocator;
use nalgebra::storage::Owned;
use nalgebra::{DefaultAllocator, DimMin, DimName};
use nalgebra::{OMatrix, OVector};

use crate::colorschemes::{get_period_color, get_smooth_orbit_color};

pub const MAX_PERIODS: usize = 20;
const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;

// {{{ structs

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;

type Matrix<D> = OMatrix<c64, D, D>;
type Vector<D> = OVector<c64, D>;

#[derive(Clone, Copy, Debug)]
pub struct Netbrot<D: DimName>
where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D, D> + Allocator<c64, D>,
{
    /// Matrix used in the iteration
    pub mat: Matrix<D>,
    /// Starting point for the iteration.
    pub z0: Vector<D>,
    /// Constant offset for the iteration.
    pub c: c64,

    /// Maximum number of iterations before the point is considered in the set.
    pub maxit: usize,
    /// Estimated escape radius (squared).
    pub escape_radius_squared: f64,
}

impl<D: DimName> Netbrot<D>
where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D, D> + Allocator<c64, D>,
{
    pub fn new(mat: Matrix<D>, z0: Vector<D>, maxit: usize) -> Self {
        Netbrot {
            mat: mat.clone(),
            z0: z0,
            c: Complex { re: 0.0, im: 0.0 },
            maxit: maxit,
            escape_radius_squared: escape_radius_squared(mat),
        }
    }

    pub fn at(self, c: c64) -> Self {
        Netbrot {
            mat: self.mat,
            z0: self.z0,
            c: c,
            maxit: self.maxit,
            escape_radius_squared: self.escape_radius_squared,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EscapeResult<D: DimName>
where
    DefaultAllocator: Allocator<c64, D>,
{
    /// Iteration at which the point escaped or None otherwise.
    pub iteration: Option<usize>,
    /// Last point of the iterate (will be very large if the point escaped).
    pub z: Vector<D>,
}

/// Period of a point, if it does not escape.
type PeriodResult = Option<usize>;

// }}}

// {{{ helpers

/// Estimate the escape radius for a given matrix $A$.
///
/// $$
///     R = \frac{2 \sqrt{d}}{\sigma_{\text{min}}(A)^2}.
/// $$
pub fn escape_radius_squared<D: DimName>(mat: Matrix<D>) -> f64
where
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<c64, D> + Allocator<c64, D, D>,
{
    // NOTE: singular values are sorted descendingly, so we can just take the last
    // one here without worrying about it too much :D
    let n = mat.nrows();
    let sigma_min_sqr = mat.view((0, 0), (n, n)).singular_values()[n - 1].powi(4);
    let fac_sqr = 4.0 * (n as f64);

    fac_sqr / sigma_min_sqr
}

pub fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: c64,
    lower_right: c64,
) -> c64 {
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

// }}}

// {{{ escape

/// Compute the escape time for the quadratic Netbrot map
///
/// $$
///     f(z) = (A z) * (A z) + c,
/// $$
///
/// where $A$ is $d \times d$ matrix, $z$ is also a $d$ dimensional vector and
/// $c$ is a complex constant.
pub fn netbrot_orbit_escape<D: DimName>(brot: Netbrot<D>) -> EscapeResult<D>
where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D> + Allocator<c64, D, D>,
{
    let mut z = brot.z0.clone();
    let mat = brot.mat.clone();
    let c = brot.c;
    let maxit = brot.maxit;
    let escape_radius_squared = brot.escape_radius_squared;

    let mut matz = mat * z;

    for i in 0..maxit {
        if z.norm_squared() > escape_radius_squared {
            return EscapeResult {
                iteration: Some(i),
                z: z,
            };
        }

        z = matz.component_mul(&matz).add_scalar(c);
        matz = mat * z;
    }

    EscapeResult {
        iteration: None,
        z: z,
    }
}

// }}}

// {{{ period

/// Compute the period of a point from the set.
///
/// The period is computed by looking at a long time iteration that does not
/// escape and checking the tolerance.
pub fn netbrot_orbit_period<D: DimName>(brot: Netbrot<D>) -> PeriodResult
where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D> + Allocator<c64, D, D>,
{
    match netbrot_orbit_escape(brot) {
        EscapeResult { iteration: None, z } => {
            // When the limit was reached but the point did not escape, we look
            // for a period in a very naive way.
            let mat = brot.mat;
            let c = brot.c;
            let mut matz = mat * z;
            let mut z_period = vec![z.scale(0.0); PERIOD_WINDOW];

            // Evaluate some more points
            z_period[0] = z;
            for i in 1..PERIOD_WINDOW {
                z_period[i] = matz.component_mul(&matz).add_scalar(c);
                matz = mat * z_period[i];
            }

            // Check newly evaluated points for periodicity
            for i in 2..MAX_PERIODS {
                let mut z_period_norm: f64 = 0.0;
                for j in 0..i - 1 {
                    z_period_norm += (z_period[j] - z_period[i + j - 1]).norm_squared();
                }

                if z_period_norm.sqrt() < 1.0e-5 {
                    return Some(i - 1);
                }
            }

            Some(MAX_PERIODS - 1)
        }
        EscapeResult {
            iteration: Some(_),
            z: _,
        } => None,
    }
}

// }}}

// {{{ rendering

pub fn render_orbit<D: DimName>(
    pixels: &mut [u8],
    brot: Netbrot<D>,
    bounds: (usize, usize),
    upper_left: c64,
    lower_right: c64,
) where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D> + Allocator<c64, D, D>,
{
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let maxit = brot.maxit;

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_escape(brot.at(point)) {
                EscapeResult {
                    iteration: None,
                    z: _,
                } => Rgb([0, 0, 0]),
                EscapeResult {
                    iteration: Some(n),
                    z,
                } => get_smooth_orbit_color(n, z.norm(), maxit),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

pub fn render_period<D: DimName>(
    pixels: &mut [u8],
    brot: Netbrot<D>,
    bounds: (usize, usize),
    upper_left: c64,
    lower_right: c64,
) where
    D: DimMin<D, Output = D>,
    Owned<c64, D>: Copy,
    Owned<c64, D, D>: Copy,
    DefaultAllocator: Allocator<c64, D> + Allocator<c64, D, D>,
{
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_period(brot.at(point)) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(period, MAX_PERIODS, 3),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}
