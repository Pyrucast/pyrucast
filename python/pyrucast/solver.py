"""Résolution de systèmes linéaires — miroir de ``ops::solver`` (Rust).

Résout ``A · x = b`` : voie directe, voie par élimination/condensation des
contraintes, et solveur unilatéral (active-set) pour le contact.
"""

from ._pyrucast import (
    solve as solve,
    solve_eliminate as solve_eliminate,
    solve_unilateral as solve_unilateral,
)

__all__ = ["solve", "solve_eliminate", "solve_unilateral"]
