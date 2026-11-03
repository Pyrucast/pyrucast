"""Source of the Python examples on the book's container pages.

Couvre `field.md`, `mesh.md`, `node.md`, `node-field.md`, `fe-space.md`,
`element-field.md`, `aggregate.md`, `introduction.md`, `model.md` and
`matrix.md`. Every block of those pages comes from here through
`{{#include …:anchor}}`, and pytest runs it.

Voir `book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

import pyrucast

# ── Champ : le contrat commun ───────────────────────────────────────────────

# Un carré coupé en deux zones de maillage, `gauche` et `droite`, qui partagent
# la diagonale (nœuds 1 et 2).
_c = pyrucast.Coords(2)
n = [_c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])]
gauche = pyrucast.Mesh(_c, "TRI3")
gauche.unit().add_cell([n[0], n[1], n[2]])
droite = pyrucast.Mesh(_c, "TRI3")
droite.unit().add_cell([n[1], n[3], n[2]])

# ── lire ───────────────────────────────────────────────────

# ANCHOR: champ_lire
# T sur les deux zones, q sur celle de droite seulement.
t = pyrucast.NodeField(gauche, ["T"])
for noeud, valeur in zip(n[:3], (0.0, 10.0, 20.0)):
    t[0].set_value(noeud, "T", valeur)
tq = pyrucast.NodeField(droite, ["T", "q"])
for noeud, valeur in zip((n[1], n[3], n[2]), (10.0, 30.0, 20.0)):
    tq[0].set_value(noeud, "T", valeur)
tq[0].add_to_component("q", 5.0)
champ = t | tq

print(champ.components())  # ['T', 'q'] : les composantes de toutes les zones
print(champ.min("T"), champ.max("T"))  # 0.0 30.0 : cherchés dans les deux zones
print(champ.min())  # 0.0 : toutes les valeurs, toutes composantes confondues
print(champ.sum("q"))  # 15.0 : les trois nœuds de droite, à 5
print(len(champ["q"]), champ["q"].components())  # 1 ['q'] : la zone sans q disparaît
# ANCHOR_END: champ_lire
assert champ.components() == ["T", "q"]
assert (champ.min("T"), champ.max("T"), champ.min()) == (0.0, 30.0, 0.0)
assert champ.sum("q") == 15.0
assert len(champ["q"]) == 1

# ── par un scalaire ────────────────────────────────────────

# ANCHOR: arithmetique
plus_chaud = champ + 5.0  # un nouveau champ : chaque valeur + 5
ecart = 100.0 - champ  # le scalaire se place d'un côté ou de l'autre
carre = champ**2.0  # puissance, valeur par valeur
print(plus_chaud.max("T"), ecart.min("T"), carre.max("q"))  # 35.0 70.0 25.0
print((-champ).min("T"), (2.0**champ).max("q"))  # -30.0 32.0

garde = champ
champ += 1.0  # un nouveau champ, rangé sous le même nom
print(champ.max("T"), garde.max("T"))  # 31.0 30.0 : `garde` a l'ancien
# ANCHOR_END: arithmetique
assert (plus_chaud.max("T"), ecart.min("T"), carre.max("q")) == (35.0, 70.0, 25.0)
assert ((-garde).min("T"), (2.0**garde).max("q")) == (-30.0, 32.0)
assert (champ.max("T"), garde.max("T")) == (31.0, 30.0)

# ── par composante ─────────────────────────────────────────

# ANCHOR: par_composante
champ.add_to_component("T", -1.0)  # en place : T seulement, sur ses deux zones
champ.mul_to_component("q", 2.0)
print(champ.max("T"), champ.max("q"))  # 30.0 12.0
# ANCHOR_END: par_composante
assert (champ.max("T"), champ.max("q")) == (30.0, 12.0)

# ── entre deux champs ──────────────────────────────────────

# ANCHOR: champ_binaire
# a porte T et q, b porte T et p, sur la même zone.
a = pyrucast.NodeField(gauche, ["T", "q"])
a[0].add_to_component("T", 10.0)
a[0].add_to_component("q", 1.0)
b = pyrucast.NodeField(gauche, ["T", "p"])
b[0].add_to_component("T", 3.0)
b[0].add_to_component("p", 4.0)

d = a - b  # T se combine ; q et p, d'un seul côté, passent tels quels
print(d.components())  # ['T', 'q', 'p']
print(d.value(n[0], "T"), d.value(n[0], "q"), d.value(n[0], "p"))  # 7.0 1.0 4.0
# ANCHOR_END: champ_binaire
assert (d.value(n[0], "T"), d.value(n[0], "q"), d.value(n[0], "p")) == (7.0, 1.0, 4.0)

# ── même support ───────────────────────────────────────────

# ANCHOR: meme_support
# Deux régions qui se touchent par la diagonale, chacune sur sa zone.
g = pyrucast.NodeField(gauche, ["T"])
g[0].add_to_component("T", 1.0)
dr = pyrucast.NodeField(droite, ["T"])
dr[0].add_to_component("T", 2.0)

try:
    g + dr  # deux supports distincts : n1 vaudrait 1 dans une zone, 2 dans l'autre
except RuntimeError as erreur:
    message = str(erreur)
    print(message)  # incoherent NodeField: node 1, component T: 1 ≠ 2: …

carre = gauche | droite
somme = g.restrict(carre) + dr.restrict(carre)  # les mêmes supports des deux côtés
print(somme.value(n[0], "T"), somme.value(n[1], "T"), somme.value(n[3], "T"))
# 1.0 3.0 2.0 : la diagonale reçoit les deux
# ANCHOR_END: meme_support
assert "restrict_like" in message
assert [somme.value(n[i], "T") for i in (0, 1, 3)] == [1.0, 3.0, 2.0]
assert (dr.restrict_like(g) + g).value(n[1], "T") == 3.0

# ── fonctions et masques ───────────────────────────────────

# ANCHOR: maths_champ
import pyrucast as pc

amorti = pc.field.exp(champ * -0.1)  # e^(-0.1 v), valeur par valeur
print(round(amorti.min("T"), 4))  # 0.0498 : e^-3, au nœud le plus chaud
print(pc.field.cos(champ).max("T"))  # 1.0 : cos 0, au nœud à 0
print(champ.sqrt().max("T"))  # 5.477… : chaque fonction est aussi une méthode

chaud = champ >= 15.0  # 1 où la valeur atteint 15, 0 ailleurs
print(chaud.value(n[0], "T"), chaud.value(n[3], "T"))  # 0.0 1.0
# ANCHOR_END: maths_champ
assert round(amorti.min("T"), 4) == 0.0498
assert pc.field.cos(champ).max("T") == 1.0
assert (chaud.value(n[0], "T"), chaud.value(n[3], "T")) == (0.0, 1.0)

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
print(mesh.cell_count)  # 2

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
print(copie.cell_count, mesh.cell_count)  # 2 1
# ANCHOR_END: scellement
assert mesh[0].is_sealed
assert copie.cell_count == 2
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
print(f.node_count)  # 3 — f reste sur le nuage d'avant
r = pyrucast.node_field.restrict(f, tri)  # le porte sur le nouveau
print(r.node_count)  # 4
# ANCHOR_END: caches
assert len(nuage | pyrucast.mesh.to_poi1(tri)) == 2
assert (f - g).value(a, "T") == 0.75
assert r.node_count == 4 and r.value(a, "T") == 1.0

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
assert mesh.cell_count == 10

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
print(poi.cell_count)  # 3

print(a)  # <Node #0>
a.dump()  # Node #0 @ (0.500, 0.500)
# ANCHOR_END: node_api
assert poi.cell_count == 3

# ── Field at the nodes ──────────────────────────────────────────────────────


# ── node field api ─────────────────────────────────────────

# ANCHOR: node_field_api
import pyrucast

# Un carré coupé en deux triangles, deux zones de maillage qui partagent la
# diagonale a–d.
c = pyrucast.Coords(2)
a, b, d, e = (c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]))
bas = pyrucast.Mesh(c, "TRI3")
bas.unit().add_cell([a, b, d])
haut = pyrucast.Mesh(c, "TRI3")
haut.unit().add_cell([a, d, e])
carre = bas | haut

# Une zone par zone de maillage, toutes les valeurs à zéro.
T = pyrucast.NodeField(carre, ["T"])
print(len(T), T.node_count)  # 2 4 — a et d sont rangés dans les deux zones

# En place, dans toutes les zones qui portent T ; puis dans une zone seulement.
T.add_to_component("T", 20.0)
T[1].set_value(e, "T", 25.0)  # e n'appartient qu'à la zone du haut

# Lire par le champ : la première zone qui porte le nœud répond.
print(T.value(e, "T"))  # 25.0
print(T.values([a, b, d, e], "T"))  # [20.0, 20.0, 20.0, 25.0]
print(T.sum("T"))  # 85.0 — a et d ne comptent qu'une fois

# Une zone est une vue : écrire dans zone, c'est écrire dans T.
zone = T[0]
print(zone)  # SubNodeField: 3 node(s), 1 component(s) [T]
print(zone[d, "T"], zone.get(2, 0))  # 20.0 20.0 — d est la ligne 2 de la zone

# Écrire un nœud partagé dans une seule zone rompt la deuxième règle.
zone[d, "T"] = 30.0
try:
    T.check()
except RuntimeError as err:
    print(err)  # incoherent NodeField: node 2, component T: 30 ≠ 20
T[1][d, "T"] = 30.0
T.check()  # les deux zones s'accordent de nouveau

# Une liste de composantes par zone.
mixte = pyrucast.NodeField.with_components_per_submesh(carre, [["T", "ux"], ["T"]])
print(mixte.components())  # ['T', 'ux']
# ANCHOR_END: node_field_api
assert T.values([a, b, d, e], "T") == [20.0, 20.0, 30.0, 25.0]
assert T.sum("T") == 95.0
assert [z.components() for z in mixte] == [["T", "ux"], ["T"]]

# ── node field support ─────────────────────────────────────

# ANCHOR: node_field_support
# Posé sur une zone TRI3, le champ prend son nuage de nœuds : une zone POI1,
# scellée. La zone TRI3 elle-même reste modifiable.
s = T.support()
print(s[0].element_type, s[0].cell_count)  # POI1 3
print(s[0].is_sealed, carre[0].is_sealed)  # True False

# Ce nuage est celui que rend to_poi1, le même objet : l'union ne garde
# que deux zones, pas quatre.
print(len(s | pyrucast.mesh.to_poi1(carre)))  # 2

# Une zone POI1 est elle-même le support.
points = pyrucast.mesh.poi1_from_nodes([b, d])
F = pyrucast.NodeField(points, ["F"])
print(len(F.support() | points), points[0].is_sealed)  # 1 True
# ANCHOR_END: node_field_support
assert len(pyrucast.NodeField(carre, ["T"]) + T) == 2

# ── node field combiner ────────────────────────────────────

# ANCHOR: node_field_combiner
# Sur un même support, l'union fond les zones en une.
q = pyrucast.NodeField(carre, ["q"])
tout = T | q
print([z.components() for z in tout])  # [['T', 'q'], ['T', 'q']]

# Deux supports distincts restent côte à côte, mais doivent s'accorder sur
# les nœuds qu'ils partagent.
coin = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes([d]), ["T"])
try:
    T | coin
except RuntimeError as err:
    print(err)  # incoherent NodeField: node 2, component T: 30 ≠ 0
coin.unit()[d, "T"] = 30.0
print(len(T | coin))  # 3

# add_sub ajoute une zone telle quelle ; consolidate fond celles qui
# partagent un support.
w = pyrucast.NodeField(carre, ["T"])
w.add_sub(q[0])
print(len(w), len(w.consolidate()))  # 3 2
# ANCHOR_END: node_field_combiner
assert len(pyrucast.node_field.merge(w, q)) == 2

# ── Finite element space ────────────────────────────────────────────────────


# ── fes construire ─────────────────────────────────────────

# ANCHOR: fes_construire
import pyrucast

# Un carré de côté 2 en deux triangles (zone 0), et son côté bas (zone 1).
c = pyrucast.Coords(2)
a = c.add_node([0.0, 0.0])
b = c.add_node([2.0, 0.0])
d = c.add_node([2.0, 2.0])
e = c.add_node([0.0, 2.0])
carre = pyrucast.Mesh(c, "TRI3")
carre.unit().add_cell([a, b, d])
carre.unit().add_cell([a, d, e])
bas = pyrucast.Mesh(c, "SEG2")
bas.unit().add_cell([a, b])
mesh = carre | bas

# Lagrange-1 et la quadrature de Gauss, sur chaque zone.
fes = pyrucast.FiniteElementSpace(mesh)
print(len(fes))  # 2
print(fes[0])  # SubFiniteElementSpace<TRI3, LAGRANGE1, GAUSS>: 3 Gauss point(s)
print(fes[1])  # SubFiniteElementSpace<SEG2, LAGRANGE1, GAUSS>: 2 Gauss point(s)

# Le même maillage, un point par maille : un autre espace.
centres = pyrucast.FiniteElementSpace(mesh, quadrature="REDUCED")
print(centres[0].gauss_count, centres[0].gauss_xi(0))  # 1 [0.333…, 0.333…]

# Un couple (interpolation, quadrature) par zone.
mixte = pyrucast.FiniteElementSpace.with_choices(
    mesh, [("LAGRANGE1", "GAUSS"), ("LAGRANGE1", "REDUCED")]
)
print([sub.quadrature for sub in mixte])  # ['GAUSS', 'REDUCED']

# Les zones portent maintenant un espace : leurs mailles sont figées.
print(carre.unit().is_sealed)  # True
# ANCHOR_END: fes_construire
assert len(pyrucast.FiniteElementSpace.lagrange1(mesh)) == 2
for _interpolation in ("LAGRANGE2", "HERMITE3"):
    try:
        pyrucast.FiniteElementSpace(carre, interpolation=_interpolation)
    except RuntimeError as err:
        assert "not compatible" in str(err)
    else:
        raise AssertionError(f"{_interpolation} must be refused on TRI3")
try:
    carre.unit().add_cell([b, d, e])
except RuntimeError as err:
    assert "sealed" in str(err)
else:
    raise AssertionError("a submesh captured by a space must be sealed")

# ── fes lire ───────────────────────────────────────────────

# ANCHOR: fes_lire
sub = fes[0]  # le sous-espace des triangles
print(sub.element_type, sub.interpolation, sub.quadrature)  # TRI3 LAGRANGE1 GAUSS
print(sub.ref_dim, sub.space_dim, sub.nodes_per_cell)  # 2 2 3
print(sub.cell_count, sub.gauss_count)  # 2 3

# Les tables de référence : le point de Gauss 0, son poids, N_i, ∂N_i/∂ξ_k.
print(sub.gauss_xi(0), sub.gauss_weight(0))  # [0.5, 0.0] 0.16666666666666666
print(sub.n_at_g(0))  # [0.5, 0.5, 0.0]
print(sub.dn_at_g(0))  # [-1.0, -1.0, 1.0, 0.0, 0.0, 1.0]

# La maille 0 au point 0 : J, |J| et ∂N_i/∂x_a, tirés de ses nœuds.
print(sub.jacobian(0, 0))  # [2.0, 2.0, 0.0, 2.0]
print(sub.det_jacobian(0, 0))  # 4.0
print(sub.dn_dx(0, 0))  # [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]

# Une maille vue seule : son aire est la somme des w_g |J_g|.
maille = fes.element(0, 1)  # ou sub[1]
print(sum(maille.gauss_weight(g) * maille.det_jacobian(g) for g in range(3)))  # 2.0
print(len(sub.elements()), len(fes.elements(1)))  # 2 1

# Chaque sous-espace rend sa zone : la même, pas une copie.
print(len(mesh | sub.submesh()))  # 2 : l'union ne la compte pas deux fois
# mesh() rend une zone par sous-espace, dans l'ordre.
print(len(fes.mesh()), len((fes | centres).mesh()))  # 2 4
print(len(fes | fes), len(fes | pyrucast.FiniteElementSpace(mesh)))  # 2 4
# ANCHOR_END: fes_lire
assert maille.index == 1 and maille.gauss_count == 3
assert sub[1].det_jacobian(0) == sub.det_jacobian(1, 0)
# Zone i of the rebuilt mesh is the support of subspace i, shared.
_both = fes | centres
for _i in range(len(_both)):
    assert len(_both.mesh()[_i] | _both[_i].submesh()) == 1
assert [z.element_type for z in _both.mesh()] == ["TRI3", "SEG2", "TRI3", "SEG2"]

# ── fes deplacer ───────────────────────────────────────────

# ANCHOR: fes_deplacer
# Allonger le côté bas : les deux zones voient la nouvelle position de b.
b.set_position([4.0, 0.0])
print(fes[0].det_jacobian(0, 0))  # 8.0, au lieu de 4.0
print(fes[1].det_jacobian(0, 0))  # 2.0, au lieu de 1.0
# ANCHOR_END: fes_deplacer
# |J| is a magnitude: an inverted cell measures like the upright one, a flat
# one measures zero and has no physical gradient.
b.set_position([-2.0, 0.0])
assert fes[1].det_jacobian(0, 0) == 1.0
b.set_position([2.0, 2.0])
assert fes[0].det_jacobian(0, 0) == 0.0
try:
    fes[0].dn_dx(0, 0)
except RuntimeError as err:
    assert "singular" in str(err)
else:
    raise AssertionError("a flat cell has no physical gradient")
b.set_position([2.0, 0.0])

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

# Un carré de 2 × 2 quadrangles.
c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
carre = pyrucast.mesh.extrude(pyrucast.mesh.line(a, b, 2), [0.0, 1.0], 2)
fes = pyrucast.FiniteElementSpace(carre)

# Chaque déclaration rend un modèle : un sous-modèle par sous-espace de `fes`.
thermique = pyrucast.model.heat_conduction(fes)
solide = pyrucast.model.elasticity(fes, "plane_stress")
# Une contrainte reçoit le modèle dont elle fige une inconnue : sa cible.
appui = pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([a]), solide, "u_x")

modele = thermique | solide | appui
print([sub.label() for sub in modele])  # ['HeatConduction', 'Elasticity', 'Dirichlet']
print(modele.primal_vars())  # ['T', 'u_x', 'u_y', 'lambda']
print(modele.dual_vars())  # ['q', 'f_x', 'f_y', 'imposed']
print(modele.dual_of("u_y"))  # f_y

# `modele[i]` est une vue sur un sous-modèle.
sub = modele[1]
print(sub.physics(), sub.material_components())  # ['mechanical'] ['E', 'nu']
print(modele[2].fespace(), modele[2].has_behavior)  # None False
print(len(modele.fespace()))  # 2 : l'appui n'intègre rien

# L'appui a créé son nœud-multiplicateur, à la position du nœud imposé.
m = modele.multiplier_mesh().node(0, 0, 0)
print(m.position(), m == a)  # [0.0, 0.0] False
# ANCHOR_END: model_api
assert [sub.label() for sub in modele] == ["HeatConduction", "Elasticity", "Dirichlet"]
assert modele.primal_vars() == ["T", "u_x", "u_y", "lambda"]
assert modele.dual_vars() == ["q", "f_x", "f_y", "imposed"]
assert modele.dual_of("u_y") == "f_y"
assert modele[1].physics() == ["mechanical"]
assert modele[1].material_components() == ["E", "nu"]
assert modele[2].fespace() is None and not modele[2].has_behavior
assert len(modele.fespace()) == 2
assert m.position() == [0.0, 0.0] and m != a
assert len(solide.multiplier_mesh()) == 0

# ── model natures ──────────────────────────────────────────

# ANCHOR: model_natures
# L'appui fige un déplacement : il est une contrainte, et mécanique aussi.
print(appui.unit().physics())  # ['constraint', 'mechanical']
mecanique = modele.filter("mechanical")  # l'élasticité et son appui
print([sub.label() for sub in mecanique])  # ['Elasticity', 'Dirichlet']
seule = mecanique.exclude("constraint")  # l'élasticité seule
print(seule.primal_vars())  # ['u_x', 'u_y']
# ANCHOR_END: model_natures
assert appui.unit().physics() == ["constraint", "mechanical"]
assert [sub.label() for sub in mecanique] == ["Elasticity", "Dirichlet"]
assert seule.primal_vars() == ["u_x", "u_y"]

# ── model chargement ───────────────────────────────────────

# ANCHOR: model_chargement
# Une barre [0, 1] de 4 SEG2 : conduction, source répartie, T imposée à gauche.
c = pyrucast.Coords(dim=1)
o = c.add_node([0.0])
l = c.add_node([1.0])
barre = pyrucast.mesh.line(o, l, 4)
fes = pyrucast.FiniteElementSpace(barre)
conduction = pyrucast.model.heat_conduction(fes)
source = pyrucast.model.flux(fes, conduction, "q")  # densité lue dans phi_q
gauche = pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([o]), conduction, "T")
modele = conduction | source | gauche

materiaux = pyrucast.element_field.material_field(modele, [("k", 1.0), ("phi_q", 1.0)])
K = pyrucast.matrix.stiffness(modele, materiaux)

# Les données que porte le modèle : ici la source, ∫ phi_q·N dx.
donnees = pyrucast.node_field.external_forces(modele, materiaux)
# Un flux de 1 entrant en x = 1, écrit à la main sur les nœuds de la barre.
ponctuel = pyrucast.NodeField(barre, ["q"])
ponctuel.unit().set_value(l, "q", 1.0)
# La valeur imposée par la contrainte : T = 2 en x = 0.
imposees = gauche.constraint_rhs([(o, 2.0)])

T = pyrucast.solver.solve(K, (donnees + ponctuel) | imposees)
print(T.value(l, "T"))  # 3.5 = 2 + 1 + 1/2
multiplicateur = gauche.multiplier_mesh().node(0, 0, 0)
print(T.value(multiplicateur, "lambda"))  # 2.0 : la réaction de l'appui
# ANCHOR_END: model_chargement
assert K.n_rows == 6
assert abs(T.value(l, "T") - 3.5) < 1e-12
assert abs(T.value(multiplicateur, "lambda") - 2.0) < 1e-12
