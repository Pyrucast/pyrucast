"""Opérateurs produisant un modèle — miroir de ``ops::model`` (Rust).

Les **déclarations de physique** : conduction, diffusion, rayonnement,
transferts, élasticité, plasticité, endommagement, éléments structuraux
(barre, poutre, coque) et contraintes (Dirichlet, MPC, baignage, contact).

Chaque opérateur consomme le **support parent** — un ``FiniteElementSpace``,
ou les maillages que relie une contrainte — et rend un ``Model`` qui le couvre
en entier : un sous-modèle par sous-espace. Un support à une zone donne le cas
unitaire, un support à N zones donne N zones. On compose des physiques
hétérogènes avec ``|`` :

    modele = model.heat_conduction(fes) | model.dirichlet(...)

Aucun n'est exposé en méthode : leur premier argument est le support que le
modèle recouvre, pas un sujet qu'on transforme.
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
    "radiation",
    "shell",
    "timoshenko",
    "truss",
    "viscoplasticity_chaboche",
    "viscoplasticity_lemaitre_chaboche",
]
