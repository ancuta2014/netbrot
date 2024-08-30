# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

DEFAULT_UPPER_LEFT = complex(-3.75, 2.5)
DEFAULT_LOWER_RIGHT = complex(1.25, -2.5)


def main(
    filename: pathlib.Path,
    outfile: pathlib.Path | None = None,
    *,
    mat_variable_names: list[str] | None = None,
    upper_left: complex | None = None,
    lower_right: complex | None = None,
    overwrite: bool = False,
) -> int:
    # {{{ sanitize inputs

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_suffix(".npz")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    if mat_variable_names is None:
        mat_variable_names = []

    if upper_left is None:
        upper_left = DEFAULT_UPPER_LEFT

    if lower_right is None:
        lower_right = DEFAULT_LOWER_RIGHT

    if upper_left.real > lower_right.real:
        log.error("Invalid bounds: upperleft %s lowerright %s", upper_left, lower_right)
        return 1

    if upper_left.imag < lower_right.imag:
        log.error("Invalid bounds: upperleft %s lowerright %s", upper_left, lower_right)
        return 1

    # }}}

    # {{{ read matrices

    from scipy.io import loadmat

    result = loadmat(filename)

    ret = 0
    matrices = []
    for name in mat_variable_names:
        mat = result[name]
        if not isinstance(mat, np.ndarray):
            ret = 1
            log.error("Object '%s' is not an ndarray: '%s'", name, type(mat).__name__)
            continue

        if mat.ndim == 2:
            matrices.append(mat)
        elif mat.ndim == 3:
            matrices.extend(mat)
        else:
            ret = 1
            log.error("Object '%s' has unsupported shape: %s", name, mat.shape)
            continue

        log.info("Read a matrix of size '%s' from '%s'.", mat.shape, name)

    if not matrices:
        log.error("Failed to read any matrices from '%s'.", filename)
        return 1

    obj_matrices = np.empty(len(matrices), dtype=object)
    for i, mat in enumerate(matrices):
        obj_matrices[i] = mat

    upper_lefts = np.full(len(matrices), upper_left)
    lower_rights = np.full(len(matrices), lower_right)
    log.info("Converted '%d' matrices to netbrot format.", obj_matrices.size)

    # }}}

    np.savez(
        outfile,
        matrices=obj_matrices,
        upper_lefts=upper_lefts,
        lower_rights=lower_rights,
    )
    log.info("Saved results in '%s'.", outfile)

    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "-n",
        "--variable-name",
        action="append",
        help="Name of the variable containing matrices",
    )
    parser.add_argument(
        "--upper-left",
        type=complex,
        default=DEFAULT_UPPER_LEFT,
        help="Upper left corner coordinates (used for rendering) as a complex number",
    )
    parser.add_argument(
        "--lower-right",
        type=complex,
        default=DEFAULT_LOWER_RIGHT,
        help="Lower right corner coordinates (used for rendering) as a complex number",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.filename,
            outfile=args.outfile,
            mat_variable_names=args.variable_name,
            upper_left=args.upper_left,
            lower_right=args.lower_right,
            overwrite=args.overwrite,
        )
    )
