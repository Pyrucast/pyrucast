"""pyrucast — librairie éléments finis en Rust, exposée à Python.

Package *mixed Rust/Python*. L'extension compilée est le sous-module privé
`_pyrucast` (tous les `#[pyclass]`/`#[pyfunction]`). L'API publique est
**rangée par thème**, en miroir de l'organisation Rust :

- les **conteneurs** (`containers::…`) restent des classes au top-level :
  `pyrucast.Coords`, `pyrucast.Mesh`, `pyrucast.Model`, … ;
- les **verbes** (`ops::<thème>::f`) vivent dans le sous-module du thème :
  `pyrucast.mesher.pave_surface`, `pyrucast.field.gradient`,
  `pyrucast.assemble.stiffness`, `pyrucast.solver.solve`, … ;
- `pyrucast.consolidate` (dispatch mesh/champ) reste au top-level, comme au
  niveau racine de `ops` ;
- la couche Python pure de plus haut niveau vit dans ses propres sous-modules
  (`pyrucast.thermomechanics`).
"""

# ── Conteneurs (nouns) : classes au top-level, même nom que la struct Rust ──
from ._pyrucast import (
    Cell as Cell,
    Coords as Coords,
    Element as Element,
    ElementField as ElementField,
    Evolution as Evolution,
    FiniteElementSpace as FiniteElementSpace,
    Matrix as Matrix,
    Mesh as Mesh,
    Model as Model,
    Node as Node,
    NodeField as NodeField,
    SubElementField as SubElementField,
    SubEvolution as SubEvolution,
    SubFiniteElementSpace as SubFiniteElementSpace,
    SubMatrix as SubMatrix,
    SubMesh as SubMesh,
    SubModel as SubModel,
    SubNodeField as SubNodeField,
)

# ── Dispatcher à la racine de `ops` : reste au top-level ────────────────────
from ._pyrucast import __doc__, __version__  # noqa: F401
from ._pyrucast import consolidate as consolidate

# ── Verbes rangés par thème (miroir de `src/ops/*`) ─────────────────────────
from . import (
    assemble as assemble,
    behavior as behavior,
    build as build,
    export as export,
    field as field,
    mesher as mesher,
    solver as solver,
    store as store,
)

# ── Couche Python pure de plus haut niveau ──────────────────────────────────
from . import thermomechanics as thermomechanics

__all__ = [
    # conteneurs
    "Cell",
    "Coords",
    "Element",
    "ElementField",
    "Evolution",
    "FiniteElementSpace",
    "Matrix",
    "Mesh",
    "Model",
    "Node",
    "NodeField",
    "SubElementField",
    "SubEvolution",
    "SubFiniteElementSpace",
    "SubMatrix",
    "SubMesh",
    "SubModel",
    "SubNodeField",
    # dispatcher racine
    "consolidate",
    # sous-modules de verbes
    "assemble",
    "behavior",
    "build",
    "export",
    "field",
    "mesher",
    "solver",
    "store",
    # couche haut niveau
    "thermomechanics",
]
