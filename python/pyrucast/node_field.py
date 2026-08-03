"""Opérateurs produisant un champ aux nœuds — miroir de ``ops::node_field``.

Dérivations (coordonnées d'un maillage, divergence d'un champ par éléments,
restriction, fusion) et assemblage nodal (flux imposé, forces internes). La
résolution produit elle aussi un champ nodal, mais garde son module :
``pyrucast.solver``.
"""

from ._pyrucast import (
    mask_node as mask,
    consolidate_node as consolidate,
    coordinates as coordinates,
    divergence as divergence,
    flux as flux,
    internal_forces as internal_forces,
    internal_forces_continuum as internal_forces_continuum,
    merge as merge,
    restrict as restrict,
    restrict_like as restrict_like,
)

__all__ = [
    "mask",
    "consolidate",
    "coordinates",
    "divergence",
    "flux",
    "internal_forces",
    "internal_forces_continuum",
    "merge",
    "restrict",
    "restrict_like",
]
