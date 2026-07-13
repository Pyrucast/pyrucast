"""pyrucast — librairie éléments finis en Rust, exposée à Python.

Package *mixed Rust/Python* : l'extension compilée est le sous-module privé
`_pyrucast` (tous les `#[pyfunction]`/`#[pyclass]`), ré-exportée ici pour que
`import pyrucast` donne accès à l'ensemble de l'API Rust. S'y ajoute une couche
Python pure de plus haut niveau (orchestration thermo-mécanique pas-à-pas).
"""

from ._pyrucast import *  # noqa: F401,F403  (ré-export de l'API Rust)
from ._pyrucast import __doc__, __version__  # noqa: F401

from .thermomechanics import mechanical_step, step_by_step, thermal_step

# `from ._pyrucast import *` lie aussi le nom `_pyrucast` dans les globals du
# package (import du sous-module) : on peut donc lire son `__all__`.
__all__ = [
    *getattr(_pyrucast, "__all__", []),  # noqa: F405
    "step_by_step",
    "thermal_step",
    "mechanical_step",
]
