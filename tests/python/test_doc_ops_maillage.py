"""Source des exemples de `book/src/operateurs/maillage.md`.

Chaque bloc de la page vient d'ici par `{{#include …:ancre}}`. Le montage vit hors des ancres. Voir
`book/src/developper/documentation-et-tests.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
"""

import math
import os
import tempfile
import textwrap

import pyrucast


def _contour_rectangle_3d(largeur=2.0, hauteur=1.0, n=4):
    """Le même contour, mais dans un `Coords` 3-D : l'extrusion vers +z l'exige."""
    c = pyrucast.Coords(3)
    coins = [
        c.add_node(list(p) + [0.0])
        for p in [(0.0, 0.0), (largeur, 0.0), (largeur, hauteur), (0.0, hauteur)]
    ]
    contour = None
    for i in range(4):
        seg = pyrucast.mesh.line(coins[i], coins[(i + 1) % 4], n)
        contour = seg if contour is None else contour | seg
    return pyrucast.mesh.consolidate(contour)


def _contour_rectangle(largeur=2.0, hauteur=1.0, n=4):
    """Un contour SEG2 fermé, orienté CCW."""
    c = pyrucast.Coords(2)
    coins = [
        c.add_node(list(p))
        for p in [(0.0, 0.0), (largeur, 0.0), (largeur, hauteur), (0.0, hauteur)]
    ]
    contour = None
    for i in range(4):
        seg = pyrucast.mesh.line(coins[i], coins[(i + 1) % 4], n)
        contour = seg if contour is None else contour | seg
    return c, pyrucast.mesh.consolidate(contour)


# ── Ligne, extrusion, ordre quadratique ─────────────────────────────────────


# ── line et extrude ────────────────────────────────────────

# ANCHOR: line
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([4.0, 0.0])

# Ligne de 4 SEG2 entre a et b (3 nœuds intermédiaires créés).
line = pyrucast.mesh.line(a, b, 4)
print(line)  # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion en QUA4 sur 2 couches selon +y.
surf = pyrucast.mesh.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())  # ['QUA4']

# Ligne quadratique : SEG3 (nœud de milieu d'arête par élément).
line3 = pyrucast.mesh.line(a, b, 4, "SEG3")
print(line3.element_types())  # ['SEG3']
# ANCHOR_END: line
assert line.cell_count() == 4
assert surf.element_types() == ["QUA4"]
assert line3.element_types() == ["SEG3"]

# ── Balayage entre deux maillages ───────────────────────────────────────────


# ── sweep variantes ────────────────────────────────────────

c = pyrucast.Coords(2)
a0, a1 = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
b0, b1 = c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0])
mesh_a = pyrucast.mesh.line(a0, a1, 2)
mesh_b = pyrucast.mesh.line(b0, b1, 2)
# ANCHOR: sweep
tri = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI3")  # 2× plus de cellules que QUA4
qua8 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA8")
qua9 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA9")
tri6 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI6")
# ANCHOR_END: sweep
assert tri.element_types() == ["TRI3"]
assert qua8.element_types() == ["QUA8"]
assert qua9.element_types() == ["QUA9"]
assert tri6.element_types() == ["TRI6"]

# ── Maillage transfini (DALL) ───────────────────────────────────────────────


# ── transfinite ────────────────────────────────────────────

# ANCHOR: transfinite
c = pyrucast.Coords(dim=2)
p0 = c.add_node([0.0, 0.0])
p1 = c.add_node([2.0, 0.0])
p2 = c.add_node([2.0, 1.0])
p3 = c.add_node([0.0, 1.0])

side1 = pyrucast.mesh.line(p0, p1, 4)  # bas,   4 éléments
side2 = pyrucast.mesh.line(p1, p2, 2)  # droite, 2 éléments
side3 = pyrucast.mesh.line(p2, p3, 4)  # haut,  4 éléments (= side1)
side4 = pyrucast.mesh.line(p3, p0, 2)  # gauche, 2 éléments (= side2)

surf = pyrucast.mesh.transfinite(side1, side2, side3, side4)
print(surf.element_types(), surf.cell_count())  # ['QUA4'] 8
# ANCHOR_END: transfinite
assert surf.element_types() == ["QUA4"]
assert surf.cell_count() == 8

# ── Transformations : translation, rotation, symétrie ───────────────────────


def _face_et_copies():
    # ANCHOR: transformations
    import math

    import pyrucast

    # Une face TRI3 (un seul triangle) dans le plan z = 0.
    c = pyrucast.Coords(dim=3)
    face = pyrucast.Mesh(c, "TRI3")
    face.unit().add_cell(
        [
            c.add_node([1.0, 0.0, 0.0]),
            c.add_node([2.0, 0.0, 0.0]),
            c.add_node([1.0, 0.0, 1.0]),
        ]
    )

    # Copie translatée de 5 selon +z (nœuds neufs ; `face` reste intacte).
    haut = pyrucast.mesh.translate(face, [0.0, 0.0, 5.0])

    # Copie tournée de 30° autour de l'axe z passant par l'origine.
    tournee = pyrucast.mesh.rotate(face, math.pi / 6, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    # Copie symétrique dans le plan y = 0, donné par trois de ses points : la
    # moitié manquante d'une pièce maillée sur son demi-modèle (cellules remises
    # à l'endroit).
    autre_moitie = pyrucast.mesh.symmetry_plane(
        face, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    )
    # ANCHOR_END: transformations
    return face, haut, tournee, autre_moitie


# ── transformations ────────────────────────────────────────

face, haut, tournee, autre_moitie = _face_et_copies()
for m in (haut, tournee, autre_moitie):
    assert m.element_types() == ["TRI3"]
    assert m.cell_count() == 1

# ── copie ──────────────────────────────────────────────────

face, _, _, _ = _face_et_copies()
# ANCHOR: copie
# Une copie sur ses **propres** nœuds, aux mêmes endroits : les deux maillages
# ne se déplacent plus ensemble.
jumelle = pyrucast.mesh.copy(face, new_nodes=True)

# Un calque : même connectivité, **mêmes** nœuds. Il est descellé, donc de
# nouveau modifiable même si `face` a déjà servi à un calcul.
calque = face.copy(new_nodes=False)
# ANCHOR_END: copie
assert jumelle.node(0, 0, 0).id != face.node(0, 0, 0).id
assert jumelle.node(0, 0, 0).position() == face.node(0, 0, 0).position()
assert calque.node(0, 0, 0).id == face.node(0, 0, 0).id
assert calque.cell_count() == face.cell_count()

# ── sweep solid ────────────────────────────────────────────

face, _, tournee, _ = _face_et_copies()
# ANCHOR: sweep_solid
# `face` et `tournee` : la face TRI3 ci-dessus et sa copie tournée de 30°.
solide = pyrucast.mesh.sweep_solid(face, tournee, 1)
print(solide.element_types())  # ['PENTA6']
# ANCHOR_END: sweep_solid
assert solide.element_types() == ["PENTA6"]

# ── Révolution ──────────────────────────────────────────────────────────────


# ── revolve ────────────────────────────────────────────────

# ANCHOR: revolve
import math

import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([1.0, 0.0])
b = c.add_node([2.0, 0.0])

# Une couronne complète : le segment radial [1, 2] tourné d'un tour en
# 32 secteurs de QUA4 — refermée, sans couture.
rayon = pyrucast.mesh.line(a, b, 4)
couronne = pyrucast.mesh.revolve(rayon, 2 * math.pi, 32, [0.0, 0.0])
print(couronne.element_types(), couronne.cell_count())  # ['QUA4'] 128

# En 3D : un quart de tube, la section QUA4 balayée autour de l'axe z.
c3 = pyrucast.Coords(dim=3)
section = pyrucast.Mesh(c3, "QUA4")
section.unit().add_cell(
    [
        c3.add_node([1.0, 0.0, 0.0]),
        c3.add_node([2.0, 0.0, 0.0]),
        c3.add_node([2.0, 0.0, 1.0]),
        c3.add_node([1.0, 0.0, 1.0]),
    ]
)
quart = pyrucast.mesh.revolve(section, math.pi / 2, 8, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
print(quart.element_types())  # ['HEX8']
# ANCHOR_END: revolve
assert couronne.cell_count() == 128
assert quart.element_types() == ["HEX8"]

# ── Montée en ordre, changement de type ─────────────────────────────────────


# ── to quadratic ───────────────────────────────────────────

_, contour = _contour_rectangle()
# ANCHOR: to_quadratic
lin = pyrucast.mesh.triangulate_surface(contour, "TRI3", 1.0)  # maillage TRI3
quad = pyrucast.mesh.to_quadratic(lin)  # copie TRI6
print(quad.element_types())  # ['TRI6']

fes = pyrucast.FiniteElementSpace(quad, interpolation="LAGRANGE2")
# ANCHOR_END: to_quadratic
assert quad.element_types() == ["TRI6"]
assert len(fes) == 1


def _solide_penta6():
    """Un pavé : carré 3-D triangulé, extrudé selon +z."""
    c = pyrucast.Coords(3)
    coins = [c.add_node(p) for p in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(4):
        contour[0].add_cell([coins[i], coins[(i + 1) % 4]])
    surf = pyrucast.mesh.triangulate_surface(contour, "TRI3", 0.34)
    return pyrucast.mesh.extrude(surf, [0.0, 0.0, 1.0], 3)


# ── convert ────────────────────────────────────────────────

volume = pyrucast.mesh.extrude(
    pyrucast.mesh.triangulate_surface(_contour_rectangle_3d(), "QUA4", 0.5),
    [0.0, 0.0, 1.0],
    1,
)
# ANCHOR: convert
faces = pyrucast.mesh.skin(volume)  # peau en QUA4
faces = pyrucast.mesh.convert(faces, "TRI3")  # QUA4 → TRI3
print(faces.element_types())  # ['TRI3']
# ANCHOR_END: convert
assert set(faces.element_types()) == {"TRI3"}

# ── Triangulation d'une surface trouée ──────────────────────────────────────


# ── triangulate surface avec trou ──────────────────────────

# ANCHOR: triangulate_surface
import pyrucast

c = pyrucast.Coords(dim=2)

# Contour extérieur : carré 4×4 (CCW).
outer = pyrucast.Mesh(c, "SEG2")
outer_nodes = [
    c.add_node(list(p)) for p in [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
]
for i in range(4):
    outer.unit().add_cell([outer_nodes[i], outer_nodes[(i + 1) % 4]])

# Trou : carré 2×2 centré, orienté CW.
hole = pyrucast.Mesh(c, "SEG2")
hole_nodes = [
    c.add_node(list(p)) for p in [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]
]
for i in range(4):
    hole.unit().add_cell([hole_nodes[i], hole_nodes[(i + 1) % 4]])

# Composer les deux contours par l'union | (jamais +).
combined = outer | hole

# Maillage TRI3 de taille ~0.5 (aire = 16 - 4 = 12).
tri = pyrucast.mesh.triangulate_surface(combined, "TRI3", size=0.5)
print(tri.element_types(), tri.cell_count())

# Variante quad-dominante.
quad = pyrucast.mesh.triangulate_surface(combined, "QUA4", size=0.5)
print(quad.element_types())  # ['QUA4', 'TRI3'] en général
# ANCHOR_END: triangulate_surface
assert tri.cell_count() > 0
assert set(quad.element_types()) <= {"QUA4", "TRI3"}

# ── Grille orientée sur le contour ──────────────────────────────────────────


# ── grid surface ───────────────────────────────────────────

# ANCHOR: grid_surface
import pyrucast as pc

H = 0.02  # taille visée
coords = pc.Coords(2)

# Un L. Chaque côté est coupé en un nombre entier de mailles de H, donc
# tous ses nœuds tombent sur les lignes que la grille tirera des angles.
angles = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.3, 0.2), (0.3, 0.4), (0.0, 0.4)]
noeuds = [coords.add_node(list(p)) for p in angles]

contour = None
for i, a in enumerate(angles):
    b = angles[(i + 1) % len(angles)]
    n = round(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / H)
    seg = pc.mesh.line(noeuds[i], noeuds[(i + 1) % len(angles)], n)
    contour = seg if contour is None else contour | seg
contour = pc.mesh.consolidate(contour)

maillage = pc.mesh.grid_surface(contour, "QUA4", size=H)
print(maillage.element_types())  # ['QUA4'] — aucun triangle
print(maillage.cell_count())  # 450 : la grille exacte du L
# ANCHOR_END: grid_surface
assert maillage.element_types() == ["QUA4"]
assert maillage.cell_count() == 450


def _contour_en_L(H=0.02):
    """Le L de la grille : chaque côté coupé en un nombre entier de mailles."""
    coords = pyrucast.Coords(2)
    angles = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.3, 0.2), (0.3, 0.4), (0.0, 0.4)]
    noeuds = [coords.add_node(list(p)) for p in angles]
    contour = None
    for i, a in enumerate(angles):
        b = angles[(i + 1) % len(angles)]
        n = round(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / H)
        seg = pyrucast.mesh.line(noeuds[i], noeuds[(i + 1) % len(angles)], n)
        contour = seg if contour is None else contour | seg
    return pyrucast.mesh.consolidate(contour), H


# ── grid surface2 ──────────────────────────────────────────

contour, H = _contour_en_L()
pc = pyrucast
# ANCHOR: grid_surface2
maillage = pc.mesh.grid_surface2(contour, "QUA4", size=H)
# ou, en méthode :
maillage = contour.grid_surface2("QUA4", size=H)
# ANCHOR_END: grid_surface2
assert maillage.cell_count() > 0

# ── Bord et peau ────────────────────────────────────────────────────────────


# ── border ─────────────────────────────────────────────────

# ANCHOR: border
import pyrucast

c = pyrucast.Coords(dim=2)
center = c.add_node([0.0, 0.0])
disc = pyrucast.mesh.triangulate_surface(
    pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 2.0, 16), "TRI3"
)

bord = pyrucast.mesh.border(disc)
print(len(bord))  # 1  (domaine simplement connexe)
print(bord.element_types())  # ['SEG2']
print(bord.cell_counts())  # [16]
# ANCHOR_END: border
assert len(bord) == 1
assert bord.element_types() == ["SEG2"]

# ── border angle ───────────────────────────────────────────

_, contour_carre = _contour_rectangle(2.0, 2.0, 4)
# ANCHOR: border_angle
carre = pyrucast.mesh.triangulate_surface(contour_carre, "TRI3", 0.5)
aretes = pyrucast.mesh.border(carre, angle_deg=45.0)
print(len(aretes))  # 4  (les quatre côtés, arêtes ouvertes)
# ANCHOR_END: border_angle
assert len(aretes) == 4

# ── skin ───────────────────────────────────────────────────

# ANCHOR: skin
import pyrucast

# Un pavé PENTA6 : carré triangulé, extrudé selon +z.
c = pyrucast.Coords(dim=3)
coins = [c.add_node(p) for p in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour[0].add_cell([coins[i], coins[(i + 1) % 4]])
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3", 0.34)
solide = pyrucast.mesh.extrude(surf, [0.0, 0.0, 1.0], 3)  # TRI3 -> PENTA6

peau = pyrucast.mesh.skin(solide)
print(len(peau))  # 6  (deux chapeaux + quatre flancs)
print(peau.element_types())  # ['TRI3', 'TRI3', 'QUA4', 'QUA4', 'QUA4', 'QUA4']
# ANCHOR_END: skin
assert len(peau) == 6

# ── Orientation, chaînage ───────────────────────────────────────────────────


# ── orient et invert ───────────────────────────────────────

_, contour = _contour_rectangle()
# ANCHOR: orient
import pyrucast

# Une plaque trouée : contour extérieur + bord du trou, orientations quelconques.
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3")

propre = pyrucast.mesh.orient(surf)  # toutes les mailles cohérentes
trou_dedans = pyrucast.mesh.invert(propre)  # sens inversé (intérieur/extérieur)
# ANCHOR_END: orient
assert propre.cell_count() == surf.cell_count()
assert trou_dedans.cell_count() == surf.cell_count()


def _surface_triangulee(taille=0.5):
    """Une surface TRI3 prête à l'emploi, sans nommer `pyrucast` chez l'appelant.

    Un `import pyrucast` dans une ancre en fait une variable **locale** de la
    fonction de test : toute utilisation du module avant l'ancre échouerait.
    """
    _, contour = _contour_rectangle()
    return pyrucast.mesh.triangulate_surface(contour, "TRI3", taille)


# ── chain ──────────────────────────────────────────────────

surf = _surface_triangulee()
# ANCHOR: chain
import pyrucast

# Un contour tiré d'une surface : les segments sont là, mais en vrac.
bord = pyrucast.mesh.border(surf)
suite = pyrucast.mesh.chain(bord)  # ou bord.chain()

# La connectivité se lit maintenant nœud à nœud le long de la courbe.
for maille in suite[0]:
    print([n.id for n in maille])
# ANCHOR_END: chain
assert suite.cell_count() == bord.cell_count()

# ── Sélections ──────────────────────────────────────────────────────────────


# ── elements on ────────────────────────────────────────────

# ANCHOR: elements_on
import pyrucast

c = pyrucast.Coords(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0)]]

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([nodes[0], nodes[1], nodes[2]])  # cellule 0
mesh.unit().add_cell([nodes[1], nodes[3], nodes[2]])  # cellule 1

# Points = {0, 1, 2} : seule la cellule 0 a tous ses nœuds dedans.
pts = pyrucast.mesh.poi1_from_nodes([nodes[0], nodes[1], nodes[2]])

strict = pyrucast.mesh.elements_on(mesh, pts, strict=True)
print(strict.cell_count())  # 1  (cellule 0)

loose = pyrucast.mesh.elements_on(mesh, pts, strict=False)
print(loose.cell_count())  # 2  (les deux touchent un nœud de pts)
# ANCHOR_END: elements_on
assert strict.cell_count() == 1
assert loose.cell_count() == 2

# ── selections geometriques ────────────────────────────────

_, contour = _contour_rectangle(2.0, 2.0, 8)
# ANCHOR: selections
import pyrucast

# Une plaque carrée maillée en TRI3.
plaque = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.1)

# Le bord gauche (x = 0) : le plan de normale +x passant par l'origine.
gauche = pyrucast.mesh.points_on_plane(plaque, [0.0, 0.0], [1.0, 0.0])

# Les nœuds du congé : dans le disque de rayon 0.2 autour du coin rentrant.
conge = pyrucast.mesh.points_in_sphere(plaque, [1.0, 1.0], 0.2)

# La sélection sert directement de support imposé à un Dirichlet — le nuage
# POI1 est ce que `model.dirichlet` attend (cf. Contraintes / Dirichlet).
mecanique = pyrucast.model.elasticity(
    pyrucast.FiniteElementSpace(plaque), "plane_stress"
)
blocage = pyrucast.model.dirichlet(
    mecanique, "u_x", gauche, pyrucast.mesh.barycenter(gauche)
)

# La sortie POI1 est un maillage ordinaire : elle se rebranche sur les autres
# opérateurs, ici pour remonter aux éléments portés par la sélection.
bande = pyrucast.mesh.elements_on(plaque, conge, strict=True)
# ANCHOR_END: selections
assert gauche.cell_count() > 0
assert len(blocage) == 1

# ── Soudure de nœuds colocalisés ────────────────────────────────────────────


# ── merge nodes ────────────────────────────────────────────

# ANCHOR: merge_nodes
import pyrucast

# Un maillage dont l'interface porte des nœuds colocalisés mais distincts
# (deux SEG2 qui se touchent par un bout dupliqué).
c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
b2 = c.add_node([1.0, 0.0])  # superposé à b, mais nœud distinct
d = c.add_node([2.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
mesh.unit().add_cell([b2, d])

joined = pyrucast.mesh.merge_nodes(mesh, 1e-6)  # b2 est soudé sur b
# ANCHOR_END: merge_nodes
assert joined.cell_count() == 2

# ── merge nodes in place ───────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
b2, d = c.add_node([1.0, 0.0]), c.add_node([2.0, 0.0])
# ANCHOR: merge_in_place
gauche = pyrucast.mesh.line(a, b, 4)
droite = pyrucast.mesh.line(b2, d, 4)  # b2 colocalisé avec b, mais distinct

pyrucast.mesh.merge_nodes(gauche | droite, 1e-6, in_place=True)

# Les deux morceaux partagent maintenant réellement le nœud d'interface.
assert droite.node(0, 0, 0).id == b.id
# ANCHOR_END: merge_in_place


# ── Sélections sur des surfaces courbes ─────────────────────────────────────


def _tube_3d():
    """Un tube : section QUA4 tournée d'un tour autour de l'axe z."""
    c = pyrucast.Coords(3)
    section = pyrucast.Mesh(c, "QUA4")
    section.unit().add_cell(
        [
            c.add_node([5.0, 0.0, 0.0]),
            c.add_node([8.0, 0.0, 0.0]),
            c.add_node([8.0, 0.0, 10.0]),
            c.add_node([5.0, 0.0, 10.0]),
        ]
    )
    return pyrucast.mesh.revolve(
        section, 2 * math.pi, 12, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    )


tube = _tube_3d()
piece = tube

# ANCHOR: selections_courbes
# L'alésage d'un tube : la surface latérale du cylindre de rayon intérieur.
alesage = pyrucast.mesh.points_on_cylinder(tube, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 5.0)

# Un chanfrein conique (rayon 8 en z = 0, sommet fictif en z = 8).
chanfrein = pyrucast.mesh.points_on_cone(piece, [0.0, 0.0, 0.0], [0.0, 0.0, 8.0], 8.0)

# La matière autour d'une gorge torique de rayon 1 sur un cercle de rayon 5.
gorge = pyrucast.mesh.points_in_torus(piece, [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 5.0, 1.0)
# ANCHOR_END: selections_courbes

assert alesage.cell_count() > 0


# ── Tétraédrisation d'un volume ─────────────────────────────────────────────

solide_penta6 = _solide_penta6()
# `skin` rend déjà des normales **sortantes** : pas d'`invert` ici, il les
# ferait rentrer et le mailleur refuserait l'enveloppe.
enveloppe = pyrucast.mesh.convert(pyrucast.mesh.skin(solide_penta6), "TRI3")

# ANCHOR: triangulate_volume
solide = pyrucast.mesh.triangulate_volume(
    enveloppe, size=None, allow_surface_nodes=False
)
# ANCHOR_END: triangulate_volume

assert solide.element_types()[0] == "TET4"

peau = enveloppe

# ANCHOR: surface_nodes
solide = pyrucast.mesh.triangulate_volume(peau, allow_surface_nodes=True)
# ANCHOR_END: surface_nodes

# ANCHOR: surface_nodes_compte
solide = pyrucast.mesh.triangulate_volume(peau, allow_surface_nodes=True)
if solide.element_types() == ["TET4", "POI1"]:
    ajoutes = solide.cell_counts()[1]
    print(f"{ajoutes} nœud(s) posé(s) sur la peau")
# ANCHOR_END: surface_nodes_compte

assert solide.cell_count() > 0


# ── Pavage quadrangulaire, puis extrusion en hexaèdres ──────────────────────


def _contour_plaque_trouee(h=0.05):
    """Contour extérieur CCW + cercle-trou CW, chaque boucle à nombre pair de
    segments — condition pour que `all_quad` puisse aboutir."""
    # `Coords` 3-D dès le départ : l'extrusion vers +z du bloc ci-dessous
    # exige une direction à trois composantes, donc des nœuds à trois.
    c = pyrucast.Coords(3)
    coins = [
        c.add_node(list(p) + [0.0])
        for p in [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)]
    ]
    exterieur = None
    for i in range(4):
        seg = pyrucast.mesh.line(coins[i], coins[(i + 1) % 4], 8)
        exterieur = seg if exterieur is None else exterieur | seg
    centre = c.add_node([0.2, 0.2, 0.0])
    trou = pyrucast.mesh.invert(pyrucast.mesh.circle(centre, [0.0, 0.0, 1.0], 0.08, 12))
    # Chaque boucle est consolidée **séparément** : les fondre toutes les deux
    # en un seul sous-maillage produirait un nœud répété, et `pave_surface`
    # exige que chaque sous-maillage de bord soit une boucle simple.
    return pyrucast.mesh.consolidate(exterieur) | pyrucast.mesh.consolidate(trou)


contour = _contour_plaque_trouee()

# ANCHOR: pave_surface
import pyrucast as pc

# … contour extérieur CCW et cercle-trou CW, consolidés en une boucle chacun.
# Chaque boucle du contour a un nombre pair de segments, donc all_quad passe.
plaque = pc.mesh.pave_surface(contour, "QUA4", size=0.05, all_quad=True)
print(plaque.element_types())  # ['QUA4']

# Le solide prismatique vient alors gratuitement, et en hexaèdres purs.
volume = pc.mesh.extrude(plaque, [0, 0, 0.02], 2)
print(volume.element_types())  # ['HEX8']
# ANCHOR_END: pave_surface

assert plaque.element_types() == ["QUA4"]
assert volume.element_types() == ["HEX8"]


# ── Couche limite hexaédrique sur un cœur tétraédrique ──────────────────────


def _boite_hex(n=3):
    """La peau d'une boîte n³ d'hexaèdres : coque QUA4 fermée, normales sortantes."""
    coords = pyrucast.Coords(3)
    a = coords.add_node([0.0, 0.0, 0.0])
    b = coords.add_node([1.0, 0.0, 0.0])
    cc = coords.add_node([1.0, 0.0, 1.0])
    d = coords.add_node([0.0, 0.0, 1.0])
    ring = None
    for p, q in ((a, d), (d, cc), (cc, b), (b, a)):
        seg = pyrucast.mesh.line(p, q, n)
        ring = seg if ring is None else ring | seg
    face = pyrucast.mesh.pave_surface(
        pyrucast.mesh.consolidate(ring), "QUA4", all_quad=True
    )
    return pyrucast.mesh.extrude(face, [0.0, 1.0, 0.0], n)


solide = _boite_hex()

# ANCHOR: pave_volume
import pyrucast as pc

peau = pc.mesh.skin(solide)  # QUA4, normales sortantes
maille = pc.mesh.pave_volume(peau, layers=1, thickness=0.15, size=0.4)
print(dict(zip(maille.element_types(), maille.cell_counts())))
# {'HEX8': 54, 'PYRA5': 54, 'TET4': 408}
# ANCHOR_END: pave_volume

assert dict(zip(maille.element_types(), maille.cell_counts()))["HEX8"] == 54


# ── Tétraédriser une peau donnée ────────────────────────────────────────────

solide_penta6 = _solide_penta6()

# ANCHOR: triangulate_volume_taille
peau = pyrucast.mesh.convert(pyrucast.mesh.skin(solide_penta6), "TRI3")
volume = pyrucast.mesh.triangulate_volume(peau, size=0.3)
# ANCHOR_END: triangulate_volume_taille

assert volume.element_types()[0] == "TET4"


# ── Lecture d'un fichier gmsh ───────────────────────────────────────────────

# Le fichier de l'exemple est écrit dans un dossier jetable, et le module y
# bascule le temps de l'extrait : celui-ci garde donc le nom court `piece.msh`
# qu'un utilisateur écrirait. Le répertoire courant est rendu ensuite.
_MSH = textwrap.dedent(
    """\
    $MeshFormat
    2.2 0 8
    $EndMeshFormat
    $PhysicalNames
    2
    1 1 "bottom"
    2 2 "plate"
    $EndPhysicalNames
    $Nodes
    4
    1 0 0 0
    2 1 0 0
    3 1 1 0
    4 0 1 0
    $EndNodes
    $Elements
    3
    1 1 2 1 1 1 2
    2 2 2 2 2 1 2 3
    3 2 2 2 2 1 3 4
    $EndElements
    """
)
_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)
open("piece.msh", "w").write(_MSH)

# ANCHOR: read_gmsh
import pyrucast

coords = pyrucast.Coords(dim=2)
regions = pyrucast.mesh.read_gmsh(coords, "piece.msh")
# {'plate': Mesh<…>, 'bottom': Mesh<…>, …}  — ordre du fichier préservé

plate = regions["plate"]
print(plate.element_types())  # p.ex. ['TRI3']
print(plate.cell_count())
# ANCHOR_END: read_gmsh

assert plate.element_types() == ["TRI3"]
assert plate.cell_count() == 2

os.chdir(_CWD)


# ANCHOR: from_gmsh_arrays
# Le même carré, mais tel que gmsh le tend en mémoire : les tags des nœuds,
# leurs trois coordonnées chacun, puis un bloc par type d'élément dont la
# connectivité est à plat.
tags = [1, 2, 3, 4]
xyz = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
blocs = [
    (1, [1, 2], ["bottom"]),  # code 1 : SEG2
    (2, [1, 2, 3, 1, 3, 4], ["plate"]),  # code 2 : TRI3
]

coords = pyrucast.Coords(dim=2)
regions = pyrucast.mesh.from_gmsh_arrays(coords, tags, xyz, blocs)
print(regions["plate"].element_types())  # ['TRI3']
print(coords.node_count())  # 4 — un seul Coords pour les deux groupes
# ANCHOR_END: from_gmsh_arrays

assert regions["plate"].element_types() == ["TRI3"]
assert regions["plate"].cell_count() == 2
assert coords.node_count() == 4
