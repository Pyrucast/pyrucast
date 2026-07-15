"""Assemblage de matrices/seconds membres — miroir de ``ops::assemble`` (Rust).

Construit une matrice (rigidité, masse), un flux nodal, ou les forces internes
nodales (`BSIG`, ``∫ Bᵀ σ``) à partir d'un ``Model`` en câblant les intégrandes
par physique.
"""

from ._pyrucast import (
    flux as flux,
    geometric as geometric,
    internal_forces as internal_forces,
    internal_forces_continuum as internal_forces_continuum,
    lump as lump,
    mass as mass,
    stiffness as stiffness,
)

__all__ = [
    "flux",
    "geometric",
    "internal_forces",
    "internal_forces_continuum",
    "lump",
    "mass",
    "stiffness",
]
