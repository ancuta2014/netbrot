// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use colors_transform::{Color, Hsl};
use image::Rgb;

// https://graphicdesign.stackexchange.com/a/158793
const COLOR_PALLETTE_V1: [Rgb<u8>; 32] = [
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
const COLOR_PALLETTE_V2: [Rgb<u8>; 32] = [
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

const COLOR_PALLETTE_V3: [Rgb<u8>; 32] = [
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

/// Determine the color for a normalized iteration count *c*.
///
/// This function takes a value *c* in [0, 1].
pub fn get_orbit_color(c: f64) -> Rgb<u8> {
    let n = c.clamp(0.0, 1.0);

    // NOTE: in HSL, we have that H in [0, 360], S in [0, 100] and L in [0, 100]
    let hue = (n * 360.0).round() as f32;
    let saturation = 100.0;
    let lightness = if n < 1.0 { 50.0 } else { 0.0 };

    let (r, g, b) = Hsl::from(hue, saturation, lightness).to_rgb().as_tuple();
    Rgb([b as u8, g as u8, r as u8])
}

/// Determine the color for a non-normalized iteration count *c* at *z*.
///
/// This function tries to be a bit smarter with the coloring and uses the
/// renormalization mentioned in [here](https://linas.org/art-gallery/escape/escape.html).
pub fn get_smooth_orbit_color(c: usize, z: f64, limit: usize) -> Rgb<u8> {
    get_orbit_color(((c as f64) + 1.0 - z.ln().log2()) / (limit as f64))
}

/// Determine the color for a given period.
///
/// The period color is determined from a fixed colormap. Currently there are
/// three colormaps implemented with *version* taking values in [1, 2, 3].
pub fn get_period_color(p: usize, limit: usize, version: u8) -> Rgb<u8> {
    if 1 <= p && p < limit - 1 {
        match version {
            1 => COLOR_PALLETTE_V1[p - 1],
            2 => COLOR_PALLETTE_V2[p - 1],
            3 => COLOR_PALLETTE_V3[p - 1],
            _ => panic!("Invalid color palette version"),
        }
    } else {
        Rgb([0, 0, 0])
    }
}
