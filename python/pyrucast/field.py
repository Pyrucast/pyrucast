"""Opérateurs polymorphes entre sortes de champ — miroir de ``ops::field``.

Ce qui rend un champ de la même sorte que celui reçu : masque de bande,
filtrage et renommage de composantes, produit scalaire point-à-point, et les
fonctions scalaires élément par élément (abs, sqrt, exp, trigonométrie…).
"""

from ._pyrucast import (
    abs as abs,
    cos as cos,
    cosh as cosh,
    exp as exp,
    filter_components as filter_components,
    log as log,
    log10 as log10,
    mask as mask,
    psca as psca,
    rename_component as rename_component,
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
    "filter_components",
    "log",
    "log10",
    "mask",
    "psca",
    "rename_component",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]
