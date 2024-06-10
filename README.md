Netbrot
=======

This repository contains some experiments for vector Mandelbrot sets. We look
at the map
```math
f(z) = (A z)^2 + c,$$
```
where $A \in \mathbb{C}^{n \times n}$ and $z \in \mathbb{C}^n$ with just
$c \in \mathbb{C}$. This gives some interesting results that are not directly
analogous to the standard scalar case.

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
