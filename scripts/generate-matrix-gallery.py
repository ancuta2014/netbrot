# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import jinja2
import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

TEMPLATE = """\
// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// NOTE: This file has been generated by `scripts/generate-matrix-gallery.py`.
// DO NOT MODIFY it manually.

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
    /// Escape radius for this matrix.
    pub escape_radius: f64,
    /// Bounding box for the points.
    pub upper_left: Complex64,
    pub lower_right: Complex64,
}

((( for ex in exhibits )))
#[allow(dead_code)]
pub const ((* ex.identifier *)): Exhibit<((* ex.size *))> = Exhibit::<((* ex.size *))> {
    mat: matrix![
        ((* ex.stringified_mat | indent(width=8) *))
    ],
    escape_radius: ((* ex.escape_radius *)),
    upper_left: Complex64 {
        re: ((* ex.upper_left.real *)),
        im: ((* ex.upper_left.imag *)),
    },
    lower_right: Complex64 {
        re: ((* ex.lower_right.real *)),
        im: ((* ex.lower_right.imag *)),
    },
};
((( endfor )))
"""


@dataclass(frozen=True)
class Exhibit:
    name: str
    """Name of the exihibit (should be a valid Rust identifier)."""
    mat: np.ndarray[Any, Any]
    """Matrix for the Netbrot set."""

    upper_left: complex
    """Upper left corner of the rendering bounding box."""
    lower_right: complex
    """Lower right corner of the rendering bounding box."""

    max_escape_radius: float
    """Maximum desired escape radius. This is meant as a hack around matrices
    where a good estimate is not available.
    """

    def __post_init__(self) -> None:
        assert self.mat.ndim == 2
        assert self.mat.shape[0] == self.mat.shape[1]
        assert self.max_escape_radius > 0.0

    @property
    def identifier(self) -> str:
        return self.name.upper()

    @property
    def size(self) -> int:
        return self.mat.shape[0]

    @property
    def escape_radius_estimate(self) -> float:
        n = self.size
        sigma = np.linalg.svdvals(self.mat)

        return 2.0 * np.sqrt(n) / np.min(sigma) ** 2

    @property
    def escape_radius(self) -> float:
        return min(self.max_escape_radius, self.escape_radius_estimate)

    @property
    def stringified_mat(self) -> str:
        n = self.size
        return "\n".join(
            "{};".format(
                ", ".join(f"c64!({float(self.mat[i, j])!r})" for j in range(n))
            )
            for i in range(n)
        )


DEFAULT_EXHIBITS = [
    Exhibit(
        name="EXHIBIT_1_2X2_FULL",
        mat=np.array([[1.0, 0.8], [1.0, -0.5]]),
        upper_left=complex(-0.9, 0.6),
        lower_right=complex(0.4, -0.6),
        max_escape_radius=np.inf,
    ),
    Exhibit(
        name="EXHIBIT_2_2X2_FULL",
        mat=np.array([[1.0, 1.0], [0.0, 1.0]]),
        upper_left=complex(-0.9, 0.6),
        lower_right=complex(0.4, -0.6),
        max_escape_radius=np.inf,
    ),
    Exhibit(
        name="EXHIBIT_3_3X3_FULL",
        mat=np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]]),
        upper_left=complex(-1.25, 0.75),
        lower_right=complex(0.5, -0.75),
        max_escape_radius=np.inf,
    ),
    Exhibit(
        name="EXHIBIT_3_3X3_BABY",
        mat=np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]]),
        upper_left=complex(-1.025, 0.025),
        lower_right=complex(-0.975, -0.025),
        max_escape_radius=np.inf,
    ),
]


def parse_ranges(ranges: str | None) -> list[slice]:
    if ranges is None:
        return []

    slices: set[int] = set()
    for entry in ranges.split(","):
        parts = [part.strip() for part in entry.split(":")]
        nparts = len(parts)

        if nparts == 0:
            continue
        elif nparts == 1:
            try:
                start = int(parts[0])
            except ValueError:
                log.error("Failed to parse range into integer: '%s'", entry)
                continue

            end = start + 1
        elif nparts == 2:
            try:
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
            except ValueError:
                log.error("Failed to parse range into integer: '%s'", entry)
                continue
        else:
            raise ValueError(f"Invalid range format: '{entry.strip()}'")

        slices.add(slice(start, end))

    return list(slices)


def make_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(
        block_start_string="(((",
        block_end_string=")))",
        variable_start_string="((*",
        variable_end_string="*))",
        comment_start_string="((=",
        comment_end_string="=))",
        autoescape=True,
    )

    return env


def main(
    infile: pathlib.Path | None = None,
    outfile: pathlib.Path | None = None,
    *,
    suffix: str = "STRUCTURE",
    slices: list[slice] | None = None,
    max_escape_radius: float = np.inf,
    overwrite: bool = False,
) -> int:
    if infile is not None and not infile.exists():
        log.error("File does not exist: '%s'", infile)
        return 1

    if not overwrite and outfile is not None and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    exhibits = DEFAULT_EXHIBITS.copy()

    if infile:
        data = np.load(infile)
        structural_connection_matrices = data["structural_connection_matrices"]
        nmatrices = structural_connection_matrices.shape[0]

        if not slices:
            slices = [slice(nmatrices)]

        indices = set()
        for s in slices:
            indices.update(range(*s.indices(nmatrices)))

        width = len(str(nmatrices))
        for i in sorted(indices):
            mat = structural_connection_matrices[i]
            ex = Exhibit(
                name=f"EXHIBIT_{i:0{width}}_{suffix}".upper(),
                mat=mat,
                upper_left=complex(-3.75, 2.5),
                lower_right=complex(1.25, -2.5),
                max_escape_radius=max_escape_radius,
            )

            exhibits.append(ex)
            log.info(
                "Loaded exhibit %d '%s': mat %s (cond %.5e) "
                "escape radius %g (estimate %.5e)",
                i,
                ex.name,
                ex.mat.shape,
                np.linalg.cond(mat),
                ex.escape_radius,
                ex.escape_radius_estimate,
            )

    env = make_jinja_env()
    result = env.from_string(TEMPLATE).render(exhibits=exhibits)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as outf:
            outf.write(result)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=pathlib.Path, default=None)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--max-escape-radius",
        type=float,
        default=np.inf,
        help="Desired maximum escape radius for the infile data",
    )
    parser.add_argument(
        "-r",
        "--ranges",
        help="A range of elements from the INFILE to load (format '1,2,2:6,:6')",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default="STRUCTURE",
        help="A suffix to add to the exhibit identifiers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.infile,
            args.outfile,
            slices=parse_ranges(args.ranges),
            suffix=args.suffix,
            max_escape_radius=args.max_escape_radius,
            overwrite=args.overwrite,
        )
    )
