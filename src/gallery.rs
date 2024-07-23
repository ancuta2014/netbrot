// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use nalgebra::{matrix, SMatrix};
use num::complex::Complex64;

macro_rules! c64 {
    ($re: literal) => {
        Complex64 { re: $re, im: 0.0 }
    };
}

pub struct Exhibit<const D: usize> {
    /// Matrix used in the iteration.
    pub mat: SMatrix<Complex64, D, D>,
    /// Bounding box for the points.
    pub upper_left: Complex64,
    pub lower_right: Complex64,
}

#[allow(dead_code)]
pub const EXHIBIT_1_2X2_FULL: Exhibit<2> = Exhibit::<2> {
    mat: matrix![
        c64!(1.0), c64!(0.8);
        c64!(1.0), c64!(-0.5);
    ],
    upper_left: c64!(-0.9),
    lower_right: c64!(0.4),
};

#[allow(dead_code)]
pub const EXHIBIT_2_2X2_FULL: Exhibit<2> = Exhibit::<2> {
    mat: matrix![
        c64!(1.0), c64!(1.0);
        c64!(0.0), c64!(1.0);
    ],
    upper_left: c64!(-0.9),
    lower_right: c64!(0.4),
};

#[allow(dead_code)]
pub const EXHIBIT_3_3X3_FULL: Exhibit<3> = Exhibit::<3> {
    mat: matrix![
        c64!(1.0), c64!(0.0), c64!(0.0);
        c64!(-1.0), c64!(1.0), c64!(0.0);
        c64!(1.0), c64!(1.0), c64!(-1.0);
    ],
    upper_left: c64!(-1.25),
    lower_right: c64!(0.5),
};

#[allow(dead_code)]
pub const EXHIBIT_3_3X3_BABY: Exhibit<3> = Exhibit::<3> {
    mat: matrix![
        c64!(1.0), c64!(0.0), c64!(0.0);
        c64!(-1.0), c64!(1.0), c64!(0.0);
        c64!(1.0), c64!(1.0), c64!(-1.0);
    ],
    upper_left: c64!(-1.025),
    lower_right: c64!(-0.975),
};
