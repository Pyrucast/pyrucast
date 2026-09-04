"""Opérateurs produisant un champ aux nœuds — miroir de ``ops::node_field``.

Dérivations (coordonnées d'un maillage, divergence d'un champ par éléments,
restriction, fusion) et assemblage nodal (flux imposé, et les deux côtés du
bilan `Σ f_int = Σ f_ext` dont l'écart est le résidu). La
résolution produit elle aussi un champ nodal, mais garde son module :
``pyrucast.solver``.
"""

from ._pyrucast import (
    mask_node as mask,
    consolidate_node as consolidate,
    positions as positions,
    divergence as divergence,
    external_forces as external_forces,
    internal_forces as internal_forces,
    merge as merge,
    restrict as restrict,
    restrict_like as restrict_like,
)

__all__ = [
    "mask",
    "consolidate",
    "positions",
    "divergence",
    "external_forces",
    "internal_forces",
    "merge",
    "restrict",
    "restrict_like",
]
