"""Source of the Python examples on the book's container pages.

Couvre `field.md`, `mesh.md`, `node.md`, `node-field.md`, `element-field.md`,
`aggregate.md`, `introduction.md`, `model.md` and `matrix.md`. Every block of
those pages comes from here through `{{#include …:anchor}}`, and pytest runs it.

Voir `book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

import pyrucast


def _triangle_et_fes():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell(n)
    return c, mesh, pyrucast.FiniteElementSpace(mesh), n


def _champ_materiau(composantes, valeur=1.0):
    _, _, fes, _ = _triangle_et_fes()
    f = pyrucast.ElementField(fes, composantes)
    for comp in composantes:
        f[0].set_uniform(comp, valeur)
    return f


def _champ_nodal(noeuds, composantes, valeur=1.0, support=None):
    support = support or pyrucast.mesh.poi1_from_nodes(noeuds)
    f = pyrucast.NodeField(support, composantes)
    for comp in composantes:
        for noeud in noeuds:
            f[0].set_value(noeud, comp, valeur)
    return f


# ── Champ : arithmétique ────────────────────────────────────────────────────


# ── arithmetique ───────────────────────────────────────────

_, _, _, n = _triangle_et_fes()
mat = _champ_materiau(["E", "nu"], 2.0)
u = _champ_nodal(n, ["u_x"], 7.0)
# ANCHOR: arithmetique
scaled = mat * 1.1  # nouveau champ, toutes composantes × 1.1
shifted = u - 5.0  # nouveau champ
energy = u**2.0  # element-wise power (fractional exponent is fine)
# ANCHOR_END: arithmetique
assert energy[0][n[0], "u_x"] == 49.0

# ── mul to component ───────────────────────────────────────

mat = _champ_materiau(["E", "nu"], 100.0)
# ANCHOR: mul_to_component
mat.mul_to_component("E", 0.95)  # scales "E" only
# ANCHOR_END: mul_to_component
assert mat.unit().value(0, 0, "E") == 95.0
assert mat.unit().value(0, 0, "nu") == 100.0

# ── maths de champ ─────────────────────────────────────────

_, _, _, n = _triangle_et_fes()
support = pyrucast.mesh.poi1_from_nodes(n)
champ1 = _champ_nodal(n, ["v"], 0.5, support)
u = _champ_nodal(n, ["v"], -1.0, support)
sx = _champ_nodal(n, ["v"], 3.0, support)
sy = _champ_nodal(n, ["v"], 4.0, support)
# ANCHOR: maths_champ
import pyrucast as pc

champ2 = pc.field.cos(champ1)  # cosine of every value
e = pc.field.exp(pc.field.abs(u) * -1.0)  # they compose freely
norme = pc.field.sqrt(sx**2.0 + sy**2.0)
# ANCHOR_END: maths_champ
assert abs(norme[0][n[0], "v"] - 5.0) < 1e-12

# ── Maillage ────────────────────────────────────────────────────────────────


# ── mesh api ───────────────────────────────────────────────

# ANCHOR: mesh_api
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
n3 = c.add_node([0.5, 1.0])

# Mesh(coords, type) creates a mesh with a single submesh; unit() gives the
# view of it, add_cell adds a cell.
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
print(mesh.element_types())  # ['TRI3']
print(mesh.cell_counts())  # [1]

# Composer plusieurs zones : l'union | (jamais +).
quad = pyrucast.Mesh(c, "QUA4")
# … add_cell … ;  combined = mesh | quad
# ANCHOR_END: mesh_api
assert mesh.cell_counts() == [1]

# ── scellement ─────────────────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
n3, n4 = c.add_node([0.5, 1.0]), c.add_node([1.5, 1.0])
# ANCHOR: scellement
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])

pyrucast.FiniteElementSpace(mesh)  # scelle mesh[0]
assert mesh[0].is_sealed
# mesh[0].add_cell([...])           # → RuntimeError (MeshSealed)

copie = mesh.duplicate()  # neuf, modifiable
copie.unit().add_cell([b, n3, n4])  # OK
# ANCHOR_END: scellement
assert copie.cell_count() == 2

# ── aggregat ───────────────────────────────────────────────

# ANCHOR: aggregat
import pyrucast

c = pyrucast.Coords(dim=2)
ns = [c.add_node(p) for p in [(0, 0), (1, 0), (1, 1), (0, 1)]]

tri = pyrucast.Mesh(c, "TRI3")
tri.unit().add_cell([ns[0], ns[1], ns[2]])

qua = pyrucast.Mesh(c, "QUA4")
qua.unit().add_cell([ns[0], ns[1], ns[2], ns[3]])

# Union of two meshes (zones shared by handle).
mesh = tri | qua
print(len(mesh))  # 2 sous-maillages
print(mesh)  # Mesh: 2 submesh(es), 2 cell(s) total

# In-place addition: `add_sub` for one zone, `add_subs` for all those of
# another aggregate (concatenation, without deduplication).
tri.add_subs(qua)
print(len(tri))  # 2 sous-maillages
# ANCHOR_END: aggregat
assert len(mesh) == 2
assert len(tri) == 2

# ── first steps ────────────────────────────────────────────

# ANCHOR: premiers_pas
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
# ANCHOR_END: premiers_pas
assert mesh.cell_count() == 1

# ── Nœud ────────────────────────────────────────────────────────────────────


# ── node api ───────────────────────────────────────────────

# ANCHOR: node_api
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

print(a.id)  # 0
print(a.position())  # [0.0, 0.0]
a.set_position([0.5, 0.5])

# Union de nœuds → maillage POI1 (deux points).
poi = a | b
print(poi)  # Mesh: 1 submesh(es), 2 cell(s) total
# ANCHOR_END: node_api
assert poi.cell_count() == 2

# ── Field at the nodes ──────────────────────────────────────────────────────


def _maillage_a_deux_zones():
    """Two POI1 clouds united: the support of a field with per-zone components.

    Set up here so the test function does not name `pyrucast` before its anchor —
    an `import pyrucast` inside an anchor makes it a local variable.
    """
    c = pyrucast.Coords(2)
    x, y = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
    return pyrucast.mesh.poi1_from_nodes([x]) | pyrucast.mesh.poi1_from_nodes([y])


# ── node field api ─────────────────────────────────────────

two_zone_mesh = _maillage_a_deux_zones()
# ANCHOR: node_field_api
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "POI1")
mesh.unit().add_cell([a])
mesh.unit().add_cell([b])

# One SubNodeField per submesh of the support (Mesh or SubMesh).
u = pyrucast.NodeField(mesh, ["UX", "UY"])
print(u)  # NodeField: 1 subfield(s)
print(u.unit())  # SubNodeField: 2 node(s), 2 component(s) [UX, UY]

# Écriture via la zone, lecture via l'agrégat.
u[0][a, "UX"] = 1.5
print(u.value(a, "UX"))  # 1.5

# Batch read: a list of nodes (or a POI1 Mesh/SubMesh) → an ordered list.
print(u.values([a, b], "UX"))  # [1.5, 0.0]
print(u.values(mesh, "UX"))  # [1.5, 0.0]  — points of the POI1 mesh
print(u.min("UX"), u.max("UX"))  # 0.0 1.5
print(u.sum("UX"))  # 1.5  — Σ over the nodes (resultant of a force field)

# Components per zone (multiphysics):
f = pyrucast.NodeField.with_components_per_submesh(two_zone_mesh, [["T"], ["UX", "UY"]])
print(f.components())  # ['T', 'UX', 'UY']
f.check()  # interface coherence (raises otherwise)
g = pyrucast.node_field.consolidate(f)  # merge as tightly as possible
# ANCHOR_END: node_field_api
assert u.value(a, "UX") == 1.5
assert f.components() == ["T", "UX", "UY"]

# ── Field at the Gauss points ───────────────────────────────────────────────


# ── element field api ──────────────────────────────────────

# ANCHOR: element_field_api
import pyrucast

# Maillage + FE space — préparation.
c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
c2 = c.add_node([0.0, 1.0])
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, c2])
fes = pyrucast.FiniteElementSpace(mesh)

# Material field: one zone per subspace of `fes`.
mat = pyrucast.ElementField(fes, ["E", "nu"])
print(mat)  # ElementField: 1 subfield(s)
print(mat.unit())  # SubElementField: 1 cell(s) × 3 gauss × 2 component(s) [E, nu]

# Writing through the zone; reading through the zone (or the aggregate stats).
z = mat.unit()  # the only zone (an error if there were several)
z.set_uniform("E", 210e9)
z.set_uniform("nu", 0.3)
assert z.value(0, 0, "E") == 210e9

# Dictionary-like access on the zone — `sub[cell, gauss, "name"]`.
z[0, 2, "nu"] = 0.28
assert z[0, 2, "nu"] == 0.28

# Stats and arithmetic at the aggregate level.
print(mat.min("E"), mat.max("E"))  # 210000000000.0 210000000000.0
print(mat.sum("E"))  # Σ over the Gauss points
mat.mul_to_component("E", 0.95)  # en place, seulement "E"
scaled = mat * 1.1  # nouveau champ

# Components per subspace (multiphysics / multi-material).
ef = pyrucast.ElementField.with_components_per_subspace(fes, [["E", "nu"]])
print(ef.components())  # ['E', 'nu']
# ANCHOR_END: element_field_api
assert ef.components() == ["E", "nu"]

# ── Modèle ──────────────────────────────────────────────────────────────────


# ── model api ──────────────────────────────────────────────

# ANCHOR: model_api
import pyrucast

c = pyrucast.Coords(dim=1)
a = c.add_node([0.0])
b = c.add_node([1.0])
mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
fes = pyrucast.FiniteElementSpace(mesh)

# Model: conduction (material supplied at assembly) + Dirichlet on the left.
# Constructors at the parent level, composed with `|` — no SubModel by hand.
# The multipliers' mesh is built from the imposed nodes.
imposed = pyrucast.mesh.poi1_from_nodes([a])
multiplier = pyrucast.mesh.barycenter(imposed)
cible = pyrucast.model.heat_conduction(fes)

model = cible | pyrucast.model.dirichlet(cible, "T", imposed, multiplier)

# Material k = 1 (the Dirichlet sub-models are skipped automatically).
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

K = pyrucast.matrix.stiffness(model, materials)
print("primal_vars =", model.primal_vars())  # ['T', 'lambda_T']
print("dual_vars =", model.dual_vars())  # ['q', 'imposed_T']
print(K)  # Matrix: 3 row(s) × 3 col(s), …
# ANCHOR_END: model_api
assert K.n_rows() == 3

# ── Matrice ─────────────────────────────────────────────────────────────────


def _support_de_deux_noeuds():
    c = pyrucast.Coords(1)
    a, b = c.add_node([0.0]), c.add_node([1.0])
    return pyrucast.mesh.poi1_from_nodes([a, b]), a, b


# ── matrix api ─────────────────────────────────────────────

support, a, b = _support_de_deux_noeuds()
# ANCHOR: matrix_api
import pyrucast

# The entries live in a **block**, never in the aggregate: a block knows its
# POI1 supports (rows and columns) and its variables.
k = pyrucast.Matrix.block(support, support, ["q"], ["T"], symmetry="full")
bloc = k[0]
bloc.add_entry(a, "q", a, "T", 2.0)
bloc.add_entry(a, "q", b, "T", -1.0)
bloc.add_entry(b, "q", a, "T", -1.0)
bloc.add_entry(b, "q", b, "T", 2.0)
k.finalize()  # required before any solver use

assert k.n_rows() == 2
assert k.n_cols() == 2
assert k.symmetric is True
# ANCHOR_END: matrix_api
