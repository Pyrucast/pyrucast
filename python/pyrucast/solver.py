"""Résolution de systèmes linéaires — miroir de ``ops::solver`` (Rust).

Solves ``A · x = b`` from the matrix alone: its relation rows, their
coefficients and their sense come with it. One entry, ``solve``: the
constraint relations are enforced by their multipliers (``"lagrange"``, the
active-set loop as soon as one is an inequality) or by elimination
(``"eliminate"``).
"""

from ._pyrucast import solve

__all__ = ["solve"]
