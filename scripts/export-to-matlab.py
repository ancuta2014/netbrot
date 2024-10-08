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

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Converts a given set of image files (PNG, JPG, etc) to a matrices and saves them
in a `.mat` MATLAB file. The file will contain two variables
* `{{prefix}}_rgb`: a matrix of size `(n, px, py, 3)`, where `n` is the number of
  input images, `(px, py)` is the resolution of each image and `3` corresponds to
  the RGB color channels.
* `{{prefix}}_binary`: a matrix of size `(n, px, py)` of only 0/1 entries.

This is meant to be used on the images generated by the main `netbrot` executable.
All the images are expected to be the same size. If multiple sized images are
available, just call the script multiple times!

Example:

    > {SCRIPT_PATH.name} --prefix result --outfile result.mat images/*.png
"""


# {{{ main


def main(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path,
    *,
    prefix: str = "images",
    overwrite: bool = False,
) -> int:
    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    import cv2
    from scipy.io import savemat

    ret = 0
    shape = None

    imgs = []
    grays = []
    for filename in filenames:
        if not filename.exists():
            ret = 1
            log.error("File does not exist: '%s'.", filename)
            continue

        # load images
        img = cv2.imread(filename)
        if shape is None:
            shape = img.shape[:2]
        elif shape != img.shape[:2]:
            log.error(
                "Expected size %dx%d but image '%s' has size %dx%d.",
                *shape,
                filename,
                *img.shape[:2],
            )
            return 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        log.info("Loaded image '%s' of size %s.", filename, img.shape)

        # really clip the values
        gray[gray <= 10] = 0
        gray[gray > 10] = 1

        imgs.append(img)
        grays.append(gray)

    prefix = prefix.rstrip("_")
    savemat(
        outfile,
        {
            f"{prefix}_rgb": np.array(imgs),
            f"{prefix}_binary": np.array(grays),
        },
    )
    log.info("Saved image data in '%s'", outfile)

    return ret


# }}}


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default="result.mat",
        help="Basename for output files",
    )
    parser.add_argument(
        "--prefix",
        default="images",
        help="A prefix to use for the variable names",
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
            args.filenames,
            outfile=args.outfile,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
    )
