// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

use num::Complex;

use nalgebra::{SMatrix, SVector};

const MAX_ESCAPE_RADIUS: f64 = 100.0;
const MAX_ESCAPE_RADIUS_SQUARED: f64 = MAX_ESCAPE_RADIUS * MAX_ESCAPE_RADIUS;

pub const MAX_PERIODS: usize = 20;
const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;

/// Compute the escape time for the quadratic Mandelbrot map
///
/// $$
///     f(z) = z^2 + c
/// $$
#[allow(dead_code)]
pub fn mandelbrot_orbit_escape(c: Complex<f64>, maxit: usize) -> (Option<usize>, Complex<f64>) {
    let mut z = Complex { re: 0.0, im: 0.0 };

    for i in 0..maxit {
        if z.norm_sqr() > 4.0 {
            return (Some(i), z);
        }

        z = z * z + c;
    }

    (None, z)
}

pub fn netbrot_orbit_escape<const D: usize>(
    c: Complex<f64>,
    mat: SMatrix<Complex<f64>, D, D>,
    z0: SVector<Complex<f64>, D>,
    maxit: usize,
) -> (Option<usize>, SVector<Complex<f64>, D>) {
    let mut z = z0.clone();
    let mut matz = mat * z;

    for i in 0..maxit {
        if z.norm_squared() > MAX_ESCAPE_RADIUS_SQUARED {
            return (Some(i), z);
        }

        z = matz.component_mul(&matz).add_scalar(c);
        matz = mat * z;
    }

    (None, z)
}

pub fn netbrot_orbit_period<const D: usize>(
    c: Complex<f64>,
    mat: SMatrix<Complex<f64>, D, D>,
    z0: SVector<Complex<f64>, D>,
    maxit: usize,
) -> Option<usize> {
    match netbrot_orbit_escape(c, mat, z0, maxit) {
        (None, z) => {
            // When the limit was reached but the point did not escape, we look
            // for a period in a very naive way.
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
