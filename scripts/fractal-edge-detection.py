# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (16, 8),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "text": {"usetex": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 5},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "image": {
            "cmap": "binary",
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    from contextlib import suppress

    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    for group, params in defaults.items():
        mp.rc(group, **params)


def main(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None,
    *,
    extent: list[int] | None = None,
    overwrite: bool = False,
) -> int:
    try:
        import matplotlib.colors as mc
        import matplotlib.pyplot as mp

        set_recommended_matplotlib()
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    if not overwrite and outfile is not None and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)

    if extent is None:
        extent = [-3.75, 1.25, -2.5, 2.5]

    from itertools import cycle

    fig = mp.Figure()
    ax = fig.gca()
    colors = cycle(mp.rcParams["axes.prop_cycle"].by_key()["color"])

    ret = 0
    for filename in filenames:
        if not filename.exists():
            ret = 1
            log.error("File does not exist: '%s'.", filename)
            continue

        # turn to grayscale
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        log.info("Loaded image '%s' of size %s.", filename, img.shape)

        # really clip the values
        gray[gray > 10] = 255
        gray[gray <= 10] = 0

        # edge detect
        edges = gray
        edges = cv2.Canny(gray, 100, 200)
        edges = 255 - edges

        output = filename.with_stem(f"{filename.stem}-edge")
        cv2.imwrite(output, edges)
        log.info("Saving edge detected image: '%s'", output)

        edges = 255 - edges
        r, g, b = mc.to_rgb(next(colors))
        R = np.minimum(255, edges * r).astype(edges.dtype)
        G = np.minimum(255, edges * g).astype(edges.dtype)
        B = np.minimum(255, edges * b).astype(edges.dtype)
        colored = cv2.merge((B, G, R, edges))
        ax.imshow(colored, origin="lower", extent=extent)

    if outfile is None:
        outfile = "result"

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.savefig(outfile)
    log.info("Saving group image: '%s'.", outfile)

    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=[-3.75, 1.25, -2.5, 2.5],
        help="Extent limits (left, right, bottom, top) passed to imshow",
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
            args.outfile,
            extent=args.extent,
            overwrite=args.overwrite,
        )
    )
