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
  (`pyrucast.stepping`).
"""

# ── Containers (nouns): top-level classes, same name as the Rust struct ─────
# ── Verbs laid out by theme (mirror of `src/ops/*`) ─────────────────────────
from . import (
    coords as coords,
)
from . import (
    element_field as element_field,
)
from . import (
    export as export,
)
from . import (
    field as field,
)
from . import (
    matrix as matrix,
)
from . import (
    measure as measure,
)
from . import (
    mesh as mesh,
)
from . import (
    model as model,
)
from . import (
    node_field as node_field,
)
from . import (
    solver as solver,
)

# ── The higher-level pure Python layer ──────────────────────────────────────
from . import stepping as stepping
from ._pyrucast import (
    Array as Array,
)
from ._pyrucast import (
    Cell as Cell,
)
from ._pyrucast import (
    Coords as Coords,
)
from ._pyrucast import (
    Element as Element,
)
from ._pyrucast import (
    ElementField as ElementField,
)
from ._pyrucast import (
    Evolution as Evolution,
)
from ._pyrucast import (
    FiniteElementSpace as FiniteElementSpace,
)
from ._pyrucast import (
    Matrix as Matrix,
)
from ._pyrucast import (
    Mesh as Mesh,
)
from ._pyrucast import (
    Model as Model,
)
from ._pyrucast import (
    Node as Node,
)
from ._pyrucast import (
    NodeField as NodeField,
)
from ._pyrucast import (
    SubElementField as SubElementField,
)
from ._pyrucast import (
    SubEvolution as SubEvolution,
)
from ._pyrucast import (
    SubFiniteElementSpace as SubFiniteElementSpace,
)
from ._pyrucast import (
    SubMatrix as SubMatrix,
)
from ._pyrucast import (
    SubMesh as SubMesh,
)
from ._pyrucast import (
    SubModel as SubModel,
)
from ._pyrucast import (
    SubNodeField as SubNodeField,
)
from ._pyrucast import __doc__, __features__, __version__  # noqa: F401

# ── Sauvegarde et relecture d'un graphe d'objets ────────────────────────────
from ._pyrucast import (
    load as load,
)
from ._pyrucast import (
    save as save,
)

# ── Diagnostics : ce que l'allocateur a fait déborder sur disque ────────────
from ._pyrucast import (
    spill_stats as spill_stats,
)

__all__ = [
    # conteneurs
    "Array",
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
    # sauvegarde / relecture
    "load",
    "matrix",
    "measure",
    "mesh",
    "model",
    "node_field",
    "save",
    "solver",
    # diagnostics
    "spill_stats",
    # couche haut niveau
    "stepping",
]
