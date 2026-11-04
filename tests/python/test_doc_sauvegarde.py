"""Source of the Python examples of `book/src/sauvegarde.md`.

Every block of those pages comes from here through `{{#include …:anchor}}`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

The save excerpts write files under short names (`etude.pyr`): the module
switches once and for all into a temporary directory, which leaves the
displayed excerpt as a user would write it.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

import pyrucast

# A throwaway working directory — the excerpts' file names stay short. The
# short. The module gives the current directory back at the end: at module
# level there is no fixture, and leaving it moved would trap the other files.
_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)


def _support_et_champs():
    """A POI1 cloud and two fields laid on it, on the **same** support."""
    c = pyrucast.Coords(2)
    noeuds = [c.add_node([float(i), 0.0]) for i in range(3)]
    support = pyrucast.mesh.poi1_from_nodes(noeuds)
    return c, support, noeuds


def _maillage_et_materiaux():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)
    mat = pyrucast.ElementField(fes, ["k"])
    mat[0].set_uniform("k", 1.0)
    return c, mesh, fes, mat, n


# ── Saving: a dictionary on the way out, a dictionary on the way back ───────

_c, mesh, fes, mat, _n = _maillage_et_materiaux()
temperature = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(_n), ["T"])

# ANCHOR: save_load
pyrucast.save(
    "etude.pyr",
    {
        "maillage fin": mesh,
        "T (°C)": temperature,
        "materiaux": mat,
        "time step": 0.05,
        "instants": [0.0, 0.1, 0.2],
    },
)

objets = pyrucast.load("etude.pyr")
mesh2 = objets["maillage fin"]
t2 = objets["T (°C)"]
# ANCHOR_END: save_load

assert mesh2.cell_count == mesh.cell_count
assert t2.components() == ["T"]


# ── Sharing survives the round trip ─────────────────────────────────────────

_c2, support, _ = _support_et_champs()

# ANCHOR: partage
t = pyrucast.NodeField(support, ["T"])
f = pyrucast.NodeField(support, ["f"])
pyrucast.save("etude.pyr", {"T": t, "f": f})

o = pyrucast.load("etude.pyr")
assert len(o["T"] | o["f"]) == 1  # a single zone: the support is one object
# ANCHOR_END: partage


# ── A mesh drags its dependencies along ─────────────────────────────────────

# ANCHOR: dependances
pyrucast.save("m.pyr", {"maillage": mesh})  # also writes the Coords and the submeshes
# ANCHOR_END: dependances

assert pyrucast.load("m.pyr")["maillage"].cell_count == mesh.cell_count


# ── What is not saved: the derived states ───────────────────────────────────

_imposed = pyrucast.mesh.poi1_from_nodes([_n[0]])
cible = pyrucast.model.heat_conduction(fes)
_appui = pyrucast.model.dirichlet(_imposed, cible, "T")
_mult = _appui.multiplier_mesh()

modele = cible | _appui
materiaux = pyrucast.element_field.material_field(modele, [("k", 1.0)])
chargement = pyrucast.NodeField(_mult, ["imposed"])
chargement[0].set_value(_mult.node(0, 0, 0), "imposed", 1.0)
pyrucast.save(
    "etude.pyr",
    {"modele": modele, "materiaux": materiaux, "chargement": chargement},
)

# ANCHOR: reassemblage
o = pyrucast.load("etude.pyr")
k = pyrucast.matrix.stiffness(o["modele"], o["materiaux"])  # réassemble
u = pyrucast.solver.solve(k, o["chargement"])  # refactorise
# ANCHOR_END: reassemblage

assert u.node_count > 0


# ── The node counters do not cross over ─────────────────────────────────────

_c3 = pyrucast.Coords(2)
_c3.add_node([0.0, 0.0])
pyrucast.save("coords_seules.pyr", {"c": _c3})

# ANCHOR: refcount
c2 = pyrucast.load("coords_seules.pyr")["c"]
c2.gc()  # collects everything: nothing in the file held those nodes
# ANCHOR_END: refcount


# End of the excerpts: the current directory is given back.
os.chdir(_CWD)
