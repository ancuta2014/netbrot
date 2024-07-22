// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

use num::Complex;

use nalgebra::{SMatrix, SVector};

#[derive(Clone, Copy, Debug)]
pub struct Mandelbrot {
    /// Starting offset for the iteration.
    c: Complex<f64>,
    /// Maximum number of iterations before the point is considered in the set.
    maxit: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EscapeResult {
    /// Iteration at which the point escaped or None otherwise.
    iteration: Option<usize>,
    /// Last point of the iterate (will be very large if the point escaped).
    z: Complex<f64>,
}

/// Compute the escape time for the quadratic Mandelbrot map
///
/// $$
///     f(z) = z^2 + c
/// $$
#[allow(dead_code)]
pub fn mandelbrot_orbit_escape(brot: Mandelbrot) -> EscapeResult {
    let mut z = Complex { re: 0.0, im: 0.0 };
    let c = brot.c;
    let maxit = brot.maxit;

    for i in 0..maxit {
        if z.norm_sqr() > 4.0 {
            return EscapeResult {
                iteration: Some(i),
                z: z,
            };
        }

        z = z * z + c;
    }

    EscapeResult {
        iteration: None,
        z: z,
    }
}
