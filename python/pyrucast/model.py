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
    bernoulli,
    boundary_transfer,
    contact,
    creep_blackburn,
    creep_lemaitre,
    creep_norton,
    damage_sic_sic,
    damage_tc,
    dirichlet,
    drucker_prager,
    elasticity,
    embedded,
    fick,
    flux,
    gurson,
    heat_conduction,
    interface_transfer,
    mazars,
    mpc,
    ottosen,
    plasticity_isotropic,
    plasticity_perfect,
    radiation,
    shell,
    timoshenko,
    truss,
    viscoplasticity_chaboche,
    viscoplasticity_lemaitre_chaboche,
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
    "flux",
    "gurson",
    "heat_conduction",
    "interface_transfer",
    "mazars",
    "mpc",
    "ottosen",
    "plasticity_isotropic",
    "plasticity_perfect",
    "radiation",
    "shell",
    "timoshenko",
    "truss",
    "viscoplasticity_chaboche",
    "viscoplasticity_lemaitre_chaboche",
]
