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

c = pyrucast.Coords(2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
n3 = c.add_node([1.0, 1.0])
n4 = c.add_node([0.0, 1.0])
n5 = c.add_node([2.0, 0.5])

# Mesh(coords, type) : un maillage d'une zone vide ; unit() l'atteint,
# add_cell y pose une maille (nœuds dans l'ordre local de l'élément) et rend
# son indice.
quad = pyrucast.Mesh(c, "QUA4")
quad.unit().add_cell([a, b, n3, n4])  # 0
tri = pyrucast.Mesh(c, "TRI3")
tri.unit().add_cell([b, n5, n3])

# Deux types d'éléments, deux zones, composées par l'union.
mesh = quad | tri
print(mesh.element_types())  # ['QUA4', 'TRI3']
print(mesh.cell_counts())  # [1, 1]
print(mesh.cell_count())  # 2

# Lire : zone i, maille j, nœud k.
cell = mesh[1][0]  # la même vue que mesh.cell(1, 0)
print(len(cell), cell.element_type)  # 3 TRI3
print(cell[1].position())  # [2.0, 0.5]
print(mesh.node(0, 0, 2).id)  # 2
print(mesh.nearest_node([1.9, 0.4]).id)  # 4
# ANCHOR_END: mesh_api
assert mesh.cell_counts() == [1, 1]
assert mesh.cell(1, 0).index == 0
assert [n.id for n in cell.nodes()] == [1, 4, 2]
assert mesh.nearest_node([1.9, 0.4]).id == 4

# ── scellement ─────────────────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
n3, n4 = c.add_node([0.5, 1.0]), c.add_node([1.5, 1.0])
# ANCHOR: scellement
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])

pyrucast.FiniteElementSpace(mesh)  # indexe les mailles : scelle mesh[0]
print(mesh[0].is_sealed)  # True
# mesh[0].add_cell([b, n3, n4])  # RuntimeError: submesh is sealed: …

copie = mesh.duplicate()  # neuve, non scellée, sur les mêmes nœuds
copie.unit().add_cell([b, n3, n4])
print(copie.cell_count(), mesh.cell_count())  # 2 1
# ANCHOR_END: scellement
assert mesh[0].is_sealed
assert copie.cell_count() == 2
try:
    mesh[0].add_cell([b, n3, n4])
except RuntimeError as e:
    assert "sealed" in str(e)
else:
    raise AssertionError("a sealed submesh must refuse add_cell")

# ── caches ─────────────────────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
n3, n4 = c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0])
# ANCHOR: caches
tri = pyrucast.Mesh(c, "TRI3")
tri.unit().add_cell([a, b, n3])

# Le nuage de nœuds d'une zone se calcule une fois : chaque appel rend
# ensuite la même zone POI1.
nuage = pyrucast.mesh.to_poi1(tri)
print(len(nuage | pyrucast.mesh.to_poi1(tri)))  # 1 — une seule zone

# Deux champs posés sur tri partagent donc leur support : ils se combinent
# nœud à nœud.
f = pyrucast.NodeField(tri, ["T"])
g = pyrucast.NodeField(tri, ["T"])
f[0].set_value(a, "T", 1.0)
g[0].set_value(a, "T", 0.25)
print((f - g).value(a, "T"))  # 0.75

# Ajouter une maille oublie le cache : le nuage suivant est une autre zone.
tri.unit().add_cell([b, n4, n3])
print(len(nuage | pyrucast.mesh.to_poi1(tri)))  # 2
print(f.node_count())  # 3 — f reste sur le nuage d'avant
r = pyrucast.node_field.restrict(f, tri)  # le porte sur le nouveau
print(r.node_count())  # 4
# ANCHOR_END: caches
assert len(nuage | pyrucast.mesh.to_poi1(tri)) == 2
assert (f - g).value(a, "T") == 0.75
assert r.node_count() == 4 and r.value(a, "T") == 1.0

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
print(mesh)  # Mesh: 2 submesh(es) [#…, #…], 2 cell(s) total

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

mesh = pyrucast.mesh.line(a, b, 10)
print(mesh)  # Mesh: 1 submesh(es) [#…], 10 cell(s) total
# ANCHOR_END: premiers_pas
assert mesh.cell_count() == 10

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

# Union de nœuds → maillage POI1 (deux points), puis un point de plus.
poi = a | b
print(poi)  # Mesh: 1 submesh(es) [#…], 2 cell(s) total
poi = poi | c.add_node([2.0, 0.0])
print(poi.cell_count())  # 3

print(a)  # <Node #0>
a.dump()  # Node #0 @ (0.500, 0.500)
# ANCHOR_END: node_api
assert poi.cell_count() == 3

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

# Deux triangles et leur espace éléments finis : 3 points de Gauss par TRI3.
c = pyrucast.Coords(2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
d = c.add_node([1.0, 1.0])
e = c.add_node([0.0, 1.0])
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, d])
mesh.unit().add_cell([a, d, e])
fes = pyrucast.FiniteElementSpace(mesh)

# Une zone par sous-espace de fes, toutes les valeurs à zéro.
mat = pyrucast.ElementField(fes, ["E", "nu"])
z = mat.unit()  # la zone, une vue : écrire dans z, c'est écrire dans mat
print(z)  # SubElementField: 2 cell(s) × 3 gauss × 2 component(s) [E, nu]

# Écrire : toute la zone, une maille entière, un point.
z.set_uniform("E", 210e9)
z.set_uniform("nu", 0.3)
z.set_cell_uniform(1, "E", 70e9)  # la maille 1, à tous ses points de Gauss
z[1, 2, "nu"] = 0.33  # la maille 1, son point de Gauss 2

# Lire : par nom, par indice de composante, ou un point entier.
print(z.value(0, 0, "E"))  # 210000000000.0
print(z[1, 2, "nu"])  # 0.33
print(z.get(1, 0, 0))  # 70000000000.0
print(z.point_values(1, 2))  # [70000000000.0, 0.33]
print(mat.max("E"), mat.min("E"))  # 210000000000.0 70000000000.0

# Une liste de composantes par sous-espace (fes n'en a qu'un).
cond = pyrucast.ElementField.with_components_per_subspace(fes, [["k"]])
print(cond.components())  # ['k']
# ANCHOR_END: element_field_api
assert mat[0].value(1, 1, "E") == 70e9
assert mat.components() == ["E", "nu"]

# ANCHOR: element_field_combiner
# Deux champs posés sur le même espace se combinent point à point.
k1 = pyrucast.ElementField(fes, ["k"])
k2 = pyrucast.ElementField(fes, ["k"])
k1.unit().set_uniform("k", 2.0)
k2.unit().set_uniform("k", 0.5)
somme = k1 + k2
print(len(somme), somme.unit().value(0, 0, "k"))  # 1 2.5

# Un second espace, même sur le même maillage, est un autre support : les
# zones restent côte à côte, rien n'est additionné.
autre = pyrucast.ElementField(pyrucast.FiniteElementSpace(mesh), ["k"])
print(len(k1 + autre))  # 2

# L'espace d'un champ, rendu tel quel : un champ posé dessus s'y combine.
print(len(pyrucast.ElementField(k1.support(), ["k"]) + k1))  # 1

# L'union garde côte à côte deux zones aux composantes disjointes ;
# consolidate les fond en une.
tout = k1 | mat
print(len(tout), tout.components())  # 2 ['k', 'E', 'nu']
print(len(tout.consolidate()))  # 1
# ANCHOR_END: element_field_combiner
assert (k1 + autre)[1].value(0, 0, "k") == 0.0
try:
    k1.unit() + autre.unit()
except RuntimeError as err:
    assert "same support" in str(err)
else:
    raise AssertionError("two zones on distinct supports must not combine")

# ANCHOR: element_field_operateurs
# Le matériau d'un modèle : une zone par sous-modèle qui en consomme.
elasticite = pyrucast.model.elasticity(fes, "plane_stress")
modele = elasticite | pyrucast.model.heat_conduction(fes)
materiau = pyrucast.element_field.material_field(
    modele, [("E", 210e9), ("nu", 0.3), ("k", 45.0)]
)
print([zone.components() for zone in materiau])  # [['E', 'nu'], ['k']]

# Une température T = x aux nœuds, portée aux points de Gauss.
T = pyrucast.NodeField(mesh, ["T"])
for n in (a, b, d, e):
    T[0].set_value(n, "T", n.position()[0])
T_g = pyrucast.element_field.interp_to_gauss(T, fes)
print([T_g.unit().value(0, g, "T") for g in range(3)])  # [0.5, 1.0, 0.5]
grad = pyrucast.element_field.gradient(T, fes)
print(grad.components())  # ['grad_T_x', 'grad_T_y']
print(grad.unit().point_values(0, 0))  # [1.0, 0.0]

# Une somme sur les points de Gauss n'est pas une intégrale : l'intégrale
# par maille se pose sur un espace à un point par maille.
un = pyrucast.ElementField(fes, ["un"]) + 1.0
centres = pyrucast.FiniteElementSpace(mesh, quadrature="REDUCED")
aires = pyrucast.element_field.cell_integral(un, centres)
print(un.sum("un"), aires.sum("un"))  # 6.0 1.0
# ANCHOR_END: element_field_operateurs
assert all(
    grad.unit().value(cell, g, "grad_T_x") == 1.0 for cell in range(2) for g in range(3)
)
assert abs(aires.unit().value(1, 0, "un") - 0.5) < 1e-15

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
# The constraint mints its multipliers beside the imposed nodes.
imposed = pyrucast.mesh.poi1_from_nodes([a])
cible = pyrucast.model.heat_conduction(fes)

model = cible | pyrucast.model.dirichlet(imposed, cible, "T")

# Material k = 1 (the Dirichlet sub-models are skipped automatically).
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

K = pyrucast.matrix.stiffness(model, materials)
print("primal_vars =", model.primal_vars())  # ['T', 'lambda']
print("dual_vars =", model.dual_vars())  # ['q', 'imposed']
print(K)  # Matrix: 3 row(s) × 3 col(s), …

# The Dirichlet pins the temperature: it is a constraint, and thermal too.
thermique = model.filter("thermal")  # conduction + Dirichlet
conduction = thermique.exclude("constraint")  # conduction alone
print(len(thermique), len(conduction))  # 2 1

# The natures say what a sub-model pins, its label what it is.
print([sub.label() for sub in model])  # ['HeatConduction', 'Dirichlet']
# ANCHOR_END: model_api
assert K.n_rows() == 3
assert (len(thermique), len(conduction)) == (2, 1)
assert [sub.label() for sub in model] == ["HeatConduction", "Dirichlet"]

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

# What it weighs, estimated: the four entries of the block, plus the assembled
# CSR. The factorization, larger than both, cannot be counted (faer keeps the
# size of its factors private).
assert bloc.memory_bytes() == 4 * 24
assert k.memory_bytes() > bloc.memory_bytes()
# ANCHOR_END: matrix_api
