Netbrot
=======

This repository contains some experiments for vector Mandelbrot sets. We look
at the map
```math
f(z) = (A z)^2 + c,$$
```
where $A \in \mathbb{C}^{n \times n}$ and $z \in \mathbb{C}^n$ with just
$c \in \mathbb{C}$. This gives some interesting results that are not directly
analogous to the standard scalar case:

* The escape radius is no longer just $2$.
* The periodicity of the various points is weirder.
* There are (possibly) multiple attractive or repelling fixed points, not just
  $z = 0 + 0\imath$.

Additional math needed! Most of these ideas have no proofs at the moment, but
seem fun to investigate!

Install
-------

This is a Rust app and uses all the standard build infrastructure. To build it,
just run
```bash
cargo build --release
```

Usage
-----

This is currently **very experimental** and just meant for playing around. Even
so, it's nicely parallelized with `rayon` and colored.

It can do the usual orbit iteration
```
netbrot --color orbit -- out.png
```
and it can also look at periodicity of various points
```
netbrot --color period -- out.png
```

Selecting the matrix to use in the rendering is not very user friendly at the moment.
The setup (matrix and rendering window) is hardcoded in ``main.rs`` using the
examples from ``gallery.rs``.

To generate additional hardcoded examples, use the `generate-matrix-gallery.py`
script with a `npz` file. For example

.. code:: sh

    python scripts/generate-matrix-gallery.py \
        --max-escape-radius 100 \
        --ranges 2:10 \
        --overwrite \
        --outfile src/gallery.rs \
        --infile data/matrices.npz

Example
-------

As a simple example, we take the matrix
```math
\begin{bmatrix}
1 & 0.8 \\
1 & -0.5
\end{bmatrix}
```

<p align="center">
    <img src="https://github.com/alexfikl/netbrot/blob/main/docs/netbrot-2x2.png?raw=true" alt="Netbrot 2x2"/>
</p>

License
-------

The code is MIT licensed (see `LICENSES/MIT.txt`). It was originally copied
from the Rust Programming example [here](https://github.com/ProgrammingRust/mandelbrot)
and has since evolved a bit.
