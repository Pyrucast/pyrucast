"""pyrucast — a finite element library in Rust, exposed to Python.

A *mixed Rust/Python* package. The compiled extension is the private submodule
`_pyrucast` (every `#[pyclass]`/`#[pyfunction]`). The public API is **laid out
by theme**, mirroring the Rust organization:

- the **containers** (`containers::…`) stay top-level classes:
  `pyrucast.Coords`, `pyrucast.Mesh`, `pyrucast.Model`, … ;
- the **verbs** (`ops::<module>::f`) live in the submodule bearing the name of
  the container they **produce**: `pyrucast.mesh.triangulate_surface`,
  `pyrucast.element_field.gradient`, `pyrucast.matrix.stiffness`,
  `pyrucast.node_field.divergence`, `pyrucast.model.heat_conduction`.
  Those producing no container are laid out by activity:
  `pyrucast.measure.integral`,
  `pyrucast.export.export_vtk`. `pyrucast.solver.solve` is the single, assumed
  exception — it produces a nodal field but is looked up by its name;
- `pyrucast.save` / `pyrucast.load` stay top-level: they produce no determined
  container, but a dictionary of what they were given;
- the higher-level pure Python layer lives in its own submodules
  (`pyrucast.thermomechanics`).
"""

# ── Containers (nouns): top-level classes, same name as the Rust struct ─────
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

# ── Sauvegarde et relecture d'un graphe d'objets ────────────────────────────
from ._pyrucast import (
    load as load,
    save as save,
)

from ._pyrucast import __doc__, __features__, __version__  # noqa: F401

# ── Verbs laid out by theme (mirror of `src/ops/*`) ─────────────────────────
from . import (
    coords as coords,
    element_field as element_field,
    export as export,
    field as field,
    matrix as matrix,
    measure as measure,
    mesh as mesh,
    model as model,
    node_field as node_field,
    solver as solver,
)

# ── The higher-level pure Python layer ──────────────────────────────────────
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
    # sauvegarde / relecture
    "load",
    "save",
    # sous-modules de verbes
    "coords",
    "element_field",
    "export",
    "field",
    "matrix",
    "measure",
    "mesh",
    "model",
    "node_field",
    "solver",
    # couche haut niveau
    "thermomechanics",
]
