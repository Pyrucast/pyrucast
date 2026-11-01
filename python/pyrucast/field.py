"""Operators polymorphic over field kinds — mirror of ``ops::field``.

The **generic** operators: those whose product is a container, but not a
determined one — it depends on the argument. The element-wise scalar functions
(abs, sqrt, exp, trigonometry…) and the point-by-point scalar product
``psca``.

``mask`` has a determined product, so it lives in ``node_field`` and
``element_field``; component filtering and renaming are methods of the field
(``f.filter_components([...])``, ``f.rename_component(a, b)``).
"""

from ._pyrucast import (
    abs,
    cos,
    cosh,
    exp,
    log,
    log10,
    psca,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
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
