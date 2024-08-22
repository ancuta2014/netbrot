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


def main(
    filename: pathlib.Path,
    outfile: pathlib.Path | None = None,
    *,
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_suffix(".npz")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    from scipy.io import loadmat

    result = loadmat(filename)
    matrices = result["Structural_Conn"]
    log.info("Read a matrix of size '%s'.", matrices.shape)

    np.savez(outfile, structural_connection_matrices=matrices)
    log.info("Saved results in '%s'.", outfile)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only show error messages",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.filename,
            outfile=args.outfile,
            overwrite=args.overwrite,
        )
    )
