"""Résolution de systèmes linéaires — miroir de ``ops::solver`` (Rust).

Solves ``A · x = b``: direct path, path by elimination/condensation of the
constraints, and unilateral (active-set) solver for contact.
"""

from ._pyrucast import (
    solve as solve,
    solve_eliminate as solve_eliminate,
    solve_unilateral as solve_unilateral,
)

__all__ = ["solve", "solve_eliminate", "solve_unilateral"]
