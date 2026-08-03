"""pyrucast — librairie éléments finis en Rust, exposée à Python.

Package *mixed Rust/Python*. L'extension compilée est le sous-module privé
`_pyrucast` (tous les `#[pyclass]`/`#[pyfunction]`). L'API publique est
**rangée par thème**, en miroir de l'organisation Rust :

- les **conteneurs** (`containers::…`) restent des classes au top-level :
  `pyrucast.Coords`, `pyrucast.Mesh`, `pyrucast.Model`, … ;
- les **verbes** (`ops::<module>::f`) vivent dans le sous-module portant le
  nom du conteneur qu'ils **produisent** : `pyrucast.mesh.triangulate_surface`,
  `pyrucast.element_field.gradient`, `pyrucast.matrix.stiffness`,
  `pyrucast.node_field.divergence`. Ceux qui ne produisent aucun conteneur
  sont rangés par activité : `pyrucast.measure.integral`,
  `pyrucast.export.export_vtk`. `pyrucast.solver.solve` est l'exception
  unique et assumée — il produit un champ nodal mais se cherche par son nom ;
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

from ._pyrucast import __doc__, __version__  # noqa: F401

# ── Verbes rangés par thème (miroir de `src/ops/*`) ─────────────────────────
from . import (
    coords as coords,
    element_field as element_field,
    export as export,
    field as field,
    matrix as matrix,
    measure as measure,
    mesh as mesh,
    node_field as node_field,
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
    # sous-modules de verbes
    "coords",
    "element_field",
    "export",
    "field",
    "matrix",
    "measure",
    "mesh",
    "node_field",
    "solver",
    "store",
    # couche haut niveau
    "thermomechanics",
]
