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


def main(outfile: pathlib.Path, *, overwrite: bool = False) -> int:
    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    matrices = np.empty(4, dtype=object)
    matrices[0] = np.array([[1.0, 0.8], [1.0, -0.5]])
    matrices[1] = np.array([[1.0, 1.0], [0.0, 1.0]])
    matrices[2] = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]])
    matrices[3] = matrices[2]

    upper_lefts = np.array([
        complex(-0.9, 0.6),
        complex(-0.9, 0.6),
        complex(-1.25, 0.75),
        complex(-1.025, 0.025),
    ])
    lower_rights = np.array([
        complex(0.4, -0.6),
        complex(0.4, -0.6),
        complex(0.5, -0.75),
        complex(-0.975, -0.025),
    ])

    np.savez(
        outfile,
        matrices=matrices,
        upper_lefts=upper_lefts,
        lower_rights=lower_rights,
    )
    log.info("Saved results in '%s'.", outfile)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
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

    outfile = args.outfile
    if outfile is None:
        outfile = "defaults.npz"

    raise SystemExit(main(outfile, overwrite=args.overwrite))
