"""Opérateurs polymorphes entre sortes de champ — miroir de ``ops::field``.

Les opérateurs **génériques** : ceux dont le produit est un conteneur, mais pas
un conteneur déterminé — il dépend de l'argument. Les fonctions scalaires
élément par élément (abs, sqrt, exp, trigonométrie…) et le produit scalaire
point-à-point ``psca``.

Le masque de bande n'est plus ici : ``mask`` a un produit déterminé, c'est deux
fonctions, et elles vivent dans ``node_field`` et ``element_field``. Le
filtrage et le renommage de composantes non plus : ce sont des méthodes du
champ (``f.filter_components([...])``, ``f.rename_component(a, b)``).
"""

from ._pyrucast import (
    abs as abs,
    cos as cos,
    cosh as cosh,
    exp as exp,
    log as log,
    log10 as log10,
    psca as psca,
    sin as sin,
    sinh as sinh,
    sqrt as sqrt,
    tan as tan,
    tanh as tanh,
)

__all__ = [
    "abs",
    "cos",
    "cosh",
    "exp",
    "log",
    "log10",
    "psca",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]
