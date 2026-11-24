"""Solving linear systems and eigenproblems — mirror of ``ops::solver`` (Rust).

``solve`` solves ``A · x = b`` from the matrix alone: its relation rows, their
coefficients and their sense come with it. The constraint relations are
enforced by their multipliers (``"lagrange"``, the active-set loop as soon as
one is an inequality) or by elimination (``"eliminate"``).

``eigen`` finds the modes of ``A x = λ B x`` nearest a shift, each mode a
``dict`` holding its node, the support shared by the modes, its eigenvalue, its
shape, the pairs of the shape and its generalized mass.
"""

from ._pyrucast import eigen, solve

__all__ = ["eigen", "solve"]
