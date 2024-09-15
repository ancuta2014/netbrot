# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.linalg as la
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

Array = np.ndarray[Any, np.dtype[Any]]
Matrix = Array

# {{{ matplotlib


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


# }}}


# {{{ netbrot maps


def netbrot(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    for _ in range(n):
        z = (mat @ z) ** 2 + c

    return z


def netbrot_prime(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    z = z0.reshape(mat.shape[0], -1)
    jac = np.einsum("in,ij->ijn", mat @ z, 2.0 * mat)

    for _ in range(1, n):
        z = (mat @ z) ** 2 + c
        jac = np.einsum(
            "ijn,jkn->ikn",
            np.einsum("in,ij->ijn", mat @ z, 2.0 * mat),
            jac,
        )

    return jac.squeeze()


def netbrot_fp(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    return netbrot(z, mat, c, n) - z


def netbrot_prime_fp(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    return netbrot_prime(z0, mat, c, n) - np.eye(z0.size)


def netbrot_lsq(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    fz = netbrot_fp(z, mat, c, n)
    return 0.5 * np.sum(fz * fz.conj(), axis=0).real


def netbrot_prime_lsq(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    fz = netbrot_fp(z0, mat, c, n).reshape(z0.size, -1)
    jacz = netbrot_prime_fp(z0, mat, c, n).reshape(z0.size, z0.size, -1)

    return np.einsum("in,ijn->in", fz, jacz).squeeze()


# }}}


# {{{ main


def find_unique_fixed_points(fixedpoints: Array, *, eps: float = 1.0e-15) -> Array:
    _, npoints = fixedpoints.shape
    indices = []

    for i in range(npoints):
        isin = any(
            la.norm(fixedpoints[:, i] - fixedpoints[:, j]) < eps for j in indices
        )

        if not isin:
            indices.append(i)

    return np.array(indices)


def main(
    filename: pathlib.Path,
    c: complex,
    *,
    nperiod: int = 1,
    npoints: int = 512,
    check_gradients: bool = False,
) -> int:
    # {{{ read in matrix

    import json

    with open(filename, encoding="utf-8") as inf:
        data = json.load(inf)

    escape_radius = data["escape_radius"]
    elements, *shape = data["mat"]
    assert shape[0] == shape[1]

    mat = np.array([complex(*e) for e in elements]).reshape(*shape).T
    nrows = shape[1]

    # }}}

    # {{{ generate a cloud of points in the escape sphere (?)

    size = (nrows, npoints)

    # fmt: off
    rng = np.random.default_rng(seed=42)
    zs = (
        rng.uniform(-escape_radius, escape_radius, size)
        + 1.0j * rng.uniform(-escape_radius, escape_radius, size)
    )
    # fmt: on
    # zs = rng.uniform(-1.0, 1.0, size) + 1.0j * rng.uniform(-1.0, 1.0, size)
    # zs = escape_radius * rng.random(npoints) * zs / la.norm(zs, axis=0)
    # assert np.all(la.norm(zs, axis=0) <= escape_radius)

    # }}}

    # {{{ check gradients

    if check_gradients:
        jacz = netbrot_prime(zs, mat, c, nperiod)

        eps = 1.0e-9
        jacz_fd = np.empty_like(jacz)
        for m in range(npoints):
            df = netbrot(zs[:, m], mat, c, nperiod)

            for i in range(nrows):
                for j in range(nrows):
                    # FIXME: is this right for complex functions?
                    e_j = np.zeros_like(df)
                    e_j[j] = eps
                    dfeps = netbrot(zs[:, m] + e_j, mat, c, nperiod)

                    # compute J_{ij} = d f_i / d z_j
                    jacz_fd[i, j, m] = (dfeps[i] - df[i]) / eps

        print(np.max(la.norm(jacz - jacz_fd, axis=(0, 1))))

    # }}}

    # {{{ find fixed points

    import scipy.optimize as so

    eps = 1.0e-5
    fixedpoints = np.empty_like(zs)
    for m in range(npoints):
        result = so.root(
            netbrot,
            zs[:, m],
            args=(mat, c, nperiod),
            method="broyden1",
            jac=netbrot_prime,
            tol=eps,
            options={"maxfev": 10000, "maxiter": 10000},
        )

        # result = so.least_squares(
        #     netbrot_fp,
        #     zs[:, m],
        #     args=(mat, c, nperiod),
        #     jac=netbrot_prime_fp,
        #     bounds=(-escape_radius, escape_radius),
        #     method="lm",
        #     ftol=eps,
        # )

        # result = so.minimize(
        #     netbrot_lsq,
        #     zs[:, m],
        #     args=(mat, c, nperiod),
        #     jac=netbrot_prime_lsq,
        #     method="",
        #     tol=eps,
        # )

        # assert result.success, result
        fixedpoints[:, m] = result.x

        log.info("[%04d] Result: %s", m, result.x)
        log.info("               %s", result.message)
        log.info(
            "[%04d]         norm %.8e error %.8e jac %.8e",
            m,
            la.norm(result.x),
            la.norm(result.x - netbrot(result.x, mat, c, nperiod)),
            0.0,  # la.norm(result.jac),
        )

    # }}}

    import matplotlib.pyplot as mp

    set_recommended_matplotlib()
    indices = find_unique_fixed_points(fixedpoints, eps=eps)
    fixednorms = la.norm(fixedpoints, axis=0)
    fixederrors = la.norm(netbrot(fixedpoints, mat, c, nperiod), axis=0)

    # {{{ plot magnitudes

    fig = mp.figure()
    ax = fig.gca()

    ax.plot(fixednorms)
    ax.plot(indices, fixednorms[indices], "o")
    ax.set_ylabel(r"$\|\mathbf{z}^*\|$")

    outfile = filename.parent / f"{filename.stem}-fixedpoints"
    fig.savefig(outfile)
    mp.close(fig)

    log.info("Saved output file for 'c=%s' to '%s'.", c, outfile)

    # }}}

    # {{{ plot errors

    fig = mp.figure()
    ax = fig.gca()

    ax.semilogy(fixederrors)
    ax.semilogy(indices, fixederrors[indices], "o")
    ax.set_ylabel(r"$\|\mathbf{f}^n(\mathbf{z}^*)\|$")

    outfile = filename.parent / f"{filename.stem}-fixederrors"
    fig.savefig(outfile)
    mp.close(fig)

    log.info("Saved output file for 'c=%s' to '%s'.", c, outfile)

    # }}}


# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "-c",
        nargs=2,
        type=float,
        default=(0, 0),
        help="Real and imaginary parts of c",
    )
    parser.add_argument("-n", default=1, type=int, help="Order of the composition")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(main(args.filename, complex(args.c[0], args.c[1]), nperiod=args.n))
