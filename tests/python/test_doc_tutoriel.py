"""Source of the Python examples of `tutoriel/langage-python.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast

# ── The module and its submodules ───────────────────────────────────────────
# ANCHOR: import
import pyrucast as pc

# ANCHOR_END: import

assert hasattr(pc, "mesh")


# ── The three starting objects ──────────────────────────────────────────────

# ANCHOR: objets
c = pc.Coords(2)  # Cast3M : OPTI 'DIME' 2
n = c.add_node([0.0, 0.0])  # Cast3M : POIN 0. 0. ;
mesh = pc.Mesh(c, "TRI3")  # Cast3M : MAILLAGE (implicite via un opérateur)
# ANCHOR_END: objets

assert mesh.element_types() == ["TRI3"]


# ── Composer un modèle ──────────────────────────────────────────────────────

_c = pyrucast.Coords(2)
_n = [_c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
_volume = pyrucast.Mesh(_c, "TRI3")
_volume.unit().add_cell(_n)
fes = pyrucast.FiniteElementSpace(_volume)
_bord = pyrucast.Mesh(_c, "SEG2")
_bord.unit().add_cell([_n[0], _n[1]])
bord_fes = pyrucast.FiniteElementSpace(_bord)
impose = pyrucast.mesh.poi1_from_nodes([_n[2]])

# ANCHOR: composer
conduction = pc.model.heat_conduction(fes)
modele = conduction | pc.model.boundary_transfer(bord_fes, conduction, ["T"])
modele = modele | pc.model.dirichlet(impose, modele, "T")
# ANCHOR_END: composer

assert len(modele) == 3


# ── The boundary conditions, two physics ────────────────────────────────────

# ANCHOR: bloquer
pc.model.dirichlet(impose, modele, "T")  # Cast3M : BLOQ 'T' ...
mecanique = pc.model.elasticity(fes, "plane_stress")
pc.model.dirichlet(impose, mecanique, "u_x")  # Cast3M : BLOQ 'UX' ...
# ANCHOR_END: bloquer
