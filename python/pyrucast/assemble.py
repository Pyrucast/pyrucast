"""Assemblage de matrices/seconds membres — miroir de ``ops::assemble`` (Rust).

Construit une matrice (rigidité, masse) ou un flux nodal à partir d'un
``Model`` en câblant les intégrandes par physique.
"""

from ._pyrucast import (
    flux as flux,
    mass as mass,
    stiffness as stiffness,
)

__all__ = ["flux", "mass", "stiffness"]
