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
    """A POI1 zone over three nodes: the one support the excerpts share."""
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

# ANCHOR: matrice_relue
pyrucast.save("k.pyr", {"k": k, "chargement": o["chargement"]})
o = pyrucast.load("k.pyr")
o["k"].assemble()  # ses valeurs n'étaient pas dans le fichier
u = pyrucast.solver.solve(o["k"], o["chargement"])
# ANCHOR_END: matrice_relue

assert u.node_count > 0
_sans_valeurs = pyrucast.load("k.pyr")
try:
    pyrucast.solver.solve(_sans_valeurs["k"], _sans_valeurs["chargement"])
    raise AssertionError("a reloaded matrix comes back without its values")
except RuntimeError as erreur:
    assert "no assembled values" in str(erreur)


# ── A reloaded field is read with the reloaded nodes ────────────────────────

# ANCHOR: noeud_relu
c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
ligne = pyrucast.Mesh(c, "SEG2")
ligne.unit().add_cell([a, b])
t = pyrucast.NodeField(pyrucast.mesh.to_poi1(ligne), ["T"])
t[0].set_value(b, "T", 20.0)
pyrucast.save("ligne.pyr", {"ligne": ligne, "T": t})

relu = pyrucast.load("ligne.pyr")
b_relu = relu["ligne"].coords().acquire(b.id)  # le même nœud, dans le Coords relu
print(relu["T"].value(b_relu, "T"))  # 20.0
# ANCHOR_END: noeud_relu

assert relu["T"].value(b_relu, "T") == 20.0
try:
    relu["T"].value(b, "T")
    raise AssertionError("a node of the writing script is refused")
except RuntimeError as erreur:
    assert "belongs to another Coords" in str(erreur)


# ── The POI1 companion of a mesh is not saved ───────────────────────────────

# ANCHOR: compagnon
f = pyrucast.NodeField(pyrucast.mesh.to_poi1(relu["ligne"]), ["f"])
print(len(relu["T"] | f))  # 2 : deux supports, aux mêmes nœuds
t_ramene = pyrucast.node_field.restrict(relu["T"], relu["ligne"])
print(len(t_ramene | f))  # 1 : un seul support
# ANCHOR_END: compagnon

assert len(relu["T"] | f) == 2
assert len(t_ramene | f) == 1
assert t_ramene.value(b_relu, "T") == 20.0


# ── The node counters do not cross over ─────────────────────────────────────

_c3 = pyrucast.Coords(2)
_c3.add_node([0.0, 0.0])
pyrucast.save("coords_seules.pyr", {"c": _c3})

# ANCHOR: refcount
c2 = pyrucast.load("coords_seules.pyr")["c"]
print(c2.refcount(0))  # 0: nothing in the file uses this node
# ANCHOR_END: refcount

assert c2.refcount(0) == 0


# End of the excerpts: the current directory is given back.
os.chdir(_CWD)
