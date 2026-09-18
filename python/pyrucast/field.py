"""Operators polymorphic over field kinds — mirror of ``ops::field``.

The **generic** operators: those whose product is a container, but not a
determined one — it depends on the argument. The element-wise scalar functions
(abs, sqrt, exp, trigonometry…) and the point-by-point scalar product
``psca``.

The band mask is no longer here: ``mask`` has a determined product, it is two
functions, and they live in ``node_field`` and ``element_field``. Component
filtering and renaming are gone too: they are methods of the
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
