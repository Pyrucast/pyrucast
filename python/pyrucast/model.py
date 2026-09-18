"""Opérateurs produisant un modèle — miroir de ``ops::model`` (Rust).

The **physics declarations**: conduction, diffusion, radiation,
transferts, élasticité, plasticité, endommagement, éléments structuraux
(barre, poutre, coque) et contraintes (Dirichlet, MPC, baignage, contact).

Every operator consumes the **parent support** — a ``FiniteElementSpace``, or
the meshes a constraint relates — and returns a ``Model`` covering it in full:
one sub-model per subspace. A one-zone support gives the unit case, an N-zone
support gives N zones. Heterogeneous physics are composed with ``|``:


    modele = model.heat_conduction(fes) | model.dirichlet(...)

None is exposed as a method: their first argument is the support the model
covers, not a subject being transformed.
"""

from ._pyrucast import (
    bernoulli as bernoulli,
    boundary_transfer as boundary_transfer,
    contact as contact,
    creep_blackburn as creep_blackburn,
    creep_lemaitre as creep_lemaitre,
    creep_norton as creep_norton,
    damage_sic_sic as damage_sic_sic,
    damage_tc as damage_tc,
    dirichlet as dirichlet,
    drucker_prager as drucker_prager,
    elasticity as elasticity,
    embedded as embedded,
    fick as fick,
    gurson as gurson,
    heat_conduction as heat_conduction,
    interface_transfer as interface_transfer,
    mazars as mazars,
    mpc as mpc,
    ottosen as ottosen,
    plasticity_isotropic as plasticity_isotropic,
    plasticity_perfect as plasticity_perfect,
    flux as flux,
    radiation as radiation,
    shell as shell,
    timoshenko as timoshenko,
    truss as truss,
    viscoplasticity_chaboche as viscoplasticity_chaboche,
    viscoplasticity_lemaitre_chaboche as viscoplasticity_lemaitre_chaboche,
)

__all__ = [
    "bernoulli",
    "boundary_transfer",
    "contact",
    "creep_blackburn",
    "creep_lemaitre",
    "creep_norton",
    "damage_sic_sic",
    "damage_tc",
    "dirichlet",
    "drucker_prager",
    "elasticity",
    "embedded",
    "fick",
    "gurson",
    "heat_conduction",
    "interface_transfer",
    "mazars",
    "mpc",
    "ottosen",
    "plasticity_isotropic",
    "plasticity_perfect",
    "flux",
    "radiation",
    "shell",
    "timoshenko",
    "truss",
    "viscoplasticity_chaboche",
    "viscoplasticity_lemaitre_chaboche",
]
