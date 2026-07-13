"""Forces internes — miroir de ``ops::internal_forces`` (Rust).

Assemble le vecteur des forces internes nodales à partir des contraintes,
en formulation structurale ou milieu continu.
"""

from ._pyrucast import (
    internal_forces as internal_forces,
    internal_forces_continuum as internal_forces_continuum,
)

__all__ = ["internal_forces", "internal_forces_continuum"]
