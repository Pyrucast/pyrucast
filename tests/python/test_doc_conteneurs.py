"""Source des exemples Python des pages de conteneurs du book.

Couvre `field.md`, `mesh.md`, `node.md`, `node-field.md`, `element-field.md`,
`aggregate.md`, `introduction.md`, `model.md` et `matrix.md`. Chaque bloc de ces
pages vient d'ici par `{{#include …:ancre}}`, et pytest l'exécute.

Voir `book/src/developper/documentation-et-tests.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
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
energy = u**2.0  # puissance élément par élément (exposant fractionnaire OK)
# ANCHOR_END: arithmetique
assert energy[0][n[0], "u_x"] == 49.0

# ── mul to component ───────────────────────────────────────

mat = _champ_materiau(["E", "nu"], 100.0)
# ANCHOR: mul_to_component
mat.mul_to_component("E", 0.95)  # ne met à l'échelle que "E"
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

champ2 = pc.field.cos(champ1)  # cosinus de chaque valeur
e = pc.field.exp(pc.field.abs(u) * -1.0)  # elles se composent librement
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

# Mesh(coords, type) crée un maillage à un seul sous-maillage ; unit() en
# donne la vue, add_cell ajoute une cellule.
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

# Union de deux maillages (zones partagées par handle).
mesh = tri | qua
print(len(mesh))  # 2 sous-maillages
print(mesh)  # Mesh: 2 submesh(es), 2 cell(s) total

# Ajout en place : `add_sub` pour une zone, `add_subs` pour toutes celles
# d'un autre agrégat (concaténation, sans déduplication).
tri.add_subs(qua)
print(len(tri))  # 2 sous-maillages
# ANCHOR_END: aggregat
assert len(mesh) == 2
assert len(tri) == 2

# ── premiers pas ───────────────────────────────────────────

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

# ── Champ aux nœuds ─────────────────────────────────────────────────────────


def _maillage_a_deux_zones():
    """Deux nuages POI1 unis : le support d'un champ à composantes par zone.

    Monté ici pour que la fonction de test ne nomme pas `pyrucast` avant son
    ancre — un `import pyrucast` dans une ancre en fait une variable locale.
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

# Un SubNodeField par submesh du support (Mesh ou SubMesh).
u = pyrucast.NodeField(mesh, ["UX", "UY"])
print(u)  # NodeField: 1 subfield(s)
print(u.unit())  # SubNodeField: 2 node(s), 2 component(s) [UX, UY]

# Écriture via la zone, lecture via l'agrégat.
u[0][a, "UX"] = 1.5
print(u.value(a, "UX"))  # 1.5

# Lecture par lot : liste de nœuds (ou Mesh/SubMesh POI1) → liste ordonnée.
print(u.values([a, b], "UX"))  # [1.5, 0.0]
print(u.values(mesh, "UX"))  # [1.5, 0.0]  — points du maillage POI1
print(u.min("UX"), u.max("UX"))  # 0.0 1.5
print(u.sum("UX"))  # 1.5  — Σ sur les nœuds (résultante d'un champ de forces)

# Composantes par zone (multiphysique) :
f = pyrucast.NodeField.with_components_per_submesh(two_zone_mesh, [["T"], ["UX", "UY"]])
print(f.components())  # ['T', 'UX', 'UY']
f.check()  # cohérence des interfaces (lève sinon)
g = pyrucast.node_field.consolidate(f)  # fusion au plus juste
# ANCHOR_END: node_field_api
assert u.value(a, "UX") == 1.5
assert f.components() == ["T", "UX", "UY"]

# ── Champ aux points de Gauss ───────────────────────────────────────────────


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

# Champ matériau : une zone par sous-espace de `fes`.
mat = pyrucast.ElementField(fes, ["E", "nu"])
print(mat)  # ElementField: 1 subfield(s)
print(mat.unit())  # SubElementField: 1 cell(s) × 3 gauss × 2 component(s) [E, nu]

# Écriture via la zone ; lecture via la zone (ou les stats agrégat).
z = mat.unit()  # la seule zone (erreur s'il y en avait plusieurs)
z.set_uniform("E", 210e9)
z.set_uniform("nu", 0.3)
assert z.value(0, 0, "E") == 210e9

# Accès dictionnaire-like sur la zone — `sub[cell, gauss, "name"]`.
z[0, 2, "nu"] = 0.28
assert z[0, 2, "nu"] == 0.28

# Stats et arithmétique au niveau agrégat.
print(mat.min("E"), mat.max("E"))  # 210000000000.0 210000000000.0
print(mat.sum("E"))  # Σ sur les points de Gauss
mat.mul_to_component("E", 0.95)  # en place, seulement "E"
scaled = mat * 1.1  # nouveau champ

# Composantes par sous-espace (multiphysique / multi-matériau).
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

# Modèle : conduction (matériau fourni à l'assemblage) + Dirichlet à gauche.
# Constructeurs au niveau parent, composés par `|` — pas de SubModel à la main.
# Le maillage des multiplicateurs est fabriqué depuis les nœuds imposés.
imposed = pyrucast.mesh.poi1_from_nodes([a])
multiplier = pyrucast.mesh.barycenter(imposed)
cible = pyrucast.model.heat_conduction(fes)

model = cible | pyrucast.model.dirichlet(cible, "T", imposed, multiplier)

# Matériau k = 1 (les sous-modèles Dirichlet sont ignorés automatiquement).
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

# Les entrées vivent dans un **bloc**, jamais dans l'agrégat : un bloc
# connaît ses supports POI1 (lignes et colonnes) et ses variables.
k = pyrucast.Matrix.block(support, support, ["q"], ["T"], symmetric=True)
bloc = k[0]
bloc.add_entry(a, "q", a, "T", 2.0)
bloc.add_entry(a, "q", b, "T", -1.0)
bloc.add_entry(b, "q", a, "T", -1.0)
bloc.add_entry(b, "q", b, "T", 2.0)
k.finalize()  # requis avant tout usage solveur

assert k.n_rows() == 2
assert k.n_cols() == 2
assert k.symmetric is True
# ANCHOR_END: matrix_api
