"""Source of the examples in `book/src/operateurs/maillage.md`.

Every block of the page comes from here through `{{#include …:anchor}}`. The scaffolding lives outside the anchors. See
`book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero return code.
"""

import math
import os
import tempfile
import textwrap

import pyrucast


def _contour_rectangle_3d(largeur=2.0, hauteur=1.0, n=4):
    """The same contour, but in a 3-D `Coords`: the extrusion towards +z requires it."""
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
    """A closed SEG2 contour, oriented CCW."""
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


# ── Line, extrusion, quadratic order ────────────────────────────────────────


# ── line and extrude ───────────────────────────────────────

# ANCHOR: line
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([4.0, 0.0])

# A line of 4 SEG2 between a and b (3 intermediate nodes created).
line = pyrucast.mesh.line(a, b, 4)
print(line)  # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion into QUA4 over 2 layers along +y.
surf = pyrucast.mesh.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())  # ['QUA4']

# Quadratic line: SEG3 (one mid-edge node per element).
line3 = pyrucast.mesh.line(a, b, 4, "SEG3")
print(line3.element_types())  # ['SEG3']
# ANCHOR_END: line
assert line.cell_count() == 4
assert surf.element_types() == ["QUA4"]
assert line3.element_types() == ["SEG3"]

# ── Sweep between two meshes ────────────────────────────────────────────────


# ── sweep variants ─────────────────────────────────────────

c = pyrucast.Coords(2)
a0, a1 = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
b0, b1 = c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0])
mesh_a = pyrucast.mesh.line(a0, a1, 2)
mesh_b = pyrucast.mesh.line(b0, b1, 2)
# ANCHOR: sweep
tri = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI3")  # 2× more cells than QUA4
qua8 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA8")
qua9 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA9")
tri6 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI6")
# ANCHOR_END: sweep
assert tri.element_types() == ["TRI3"]
assert qua8.element_types() == ["QUA8"]
assert qua9.element_types() == ["QUA9"]
assert tri6.element_types() == ["TRI6"]

# ── Transfinite mesh (DALL) ─────────────────────────────────────────────────


# ── transfinite ────────────────────────────────────────────

# ANCHOR: transfinite
c = pyrucast.Coords(dim=2)
p0 = c.add_node([0.0, 0.0])
p1 = c.add_node([2.0, 0.0])
p2 = c.add_node([2.0, 1.0])
p3 = c.add_node([0.0, 1.0])

side1 = pyrucast.mesh.line(p0, p1, 4)  # bottom, 4 elements
side2 = pyrucast.mesh.line(p1, p2, 2)  # right,  2 elements
side3 = pyrucast.mesh.line(p2, p3, 4)  # top,    4 elements (= side1)
side4 = pyrucast.mesh.line(p3, p0, 2)  # left,   2 elements (= side2)

surf = pyrucast.mesh.transfinite(side1, side2, side3, side4)
print(surf.element_types(), surf.cell_count())  # ['QUA4'] 8
# ANCHOR_END: transfinite
assert surf.element_types() == ["QUA4"]
assert surf.cell_count() == 8

# ── Transformations: translation, rotation, symmetry ────────────────────────


def _face_et_copies():
    # ANCHOR: transformations
    import math

    import pyrucast

    # A TRI3 face (a single triangle) in the z = 0 plane.
    c = pyrucast.Coords(dim=3)
    face = pyrucast.Mesh(c, "TRI3")
    face.unit().add_cell(
        [
            c.add_node([1.0, 0.0, 0.0]),
            c.add_node([2.0, 0.0, 0.0]),
            c.add_node([1.0, 0.0, 1.0]),
        ]
    )

    # Copy translated by 5 along +z (new nodes; `face` is left intact).
    haut = pyrucast.mesh.translate(face, [0.0, 0.0, 5.0])

    # Copy rotated by 30° about the z axis through the origin.
    tournee = pyrucast.mesh.rotate(face, math.pi / 6, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    # Mirror copy in the y = 0 plane, given by three of its points: the missing
    # half of a part meshed on its half-model (cells put back the right way
    # round).
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

# ── copy ───────────────────────────────────────────────────

face, _, _, _ = _face_et_copies()
# ANCHOR: copie
# A copy on its **own** nodes, at the same places: the two meshes no longer
# move together.
jumelle = pyrucast.mesh.copy(face, new_nodes=True)

# A tracing: same connectivity, **same** nodes. It is unsealed, hence editable
# again even if `face` has already been used in a computation.
calque = face.copy(new_nodes=False)
# ANCHOR_END: copie
assert jumelle.node(0, 0, 0).id != face.node(0, 0, 0).id
assert jumelle.node(0, 0, 0).position() == face.node(0, 0, 0).position()
assert calque.node(0, 0, 0).id == face.node(0, 0, 0).id
assert calque.cell_count() == face.cell_count()

# ── sweep solid ────────────────────────────────────────────

face, _, tournee, _ = _face_et_copies()
# ANCHOR: sweep_solid
# `face` and `tournee`: the TRI3 face above and its copy rotated by 30°.
solide = pyrucast.mesh.sweep_solid(face, tournee, 1)
print(solide.element_types())  # ['PENTA6']
# ANCHOR_END: sweep_solid
assert solide.element_types() == ["PENTA6"]

# ── Revolution ──────────────────────────────────────────────────────────────


# ── revolve ────────────────────────────────────────────────

# ANCHOR: revolve
import math

import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([1.0, 0.0])
b = c.add_node([2.0, 0.0])

# A complete annulus: the radial segment [1, 2] revolved a full turn into
# 32 QUA4 sectors — closed back on itself, with no seam.
rayon = pyrucast.mesh.line(a, b, 4)
couronne = pyrucast.mesh.revolve(rayon, 2 * math.pi, 32, [0.0, 0.0])
print(couronne.element_types(), couronne.cell_count())  # ['QUA4'] 128

# In 3D: a quarter of a tube, the QUA4 section swept about the z axis.
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

# ── Order raising, type change ──────────────────────────────────────────────


# ── to quadratic ───────────────────────────────────────────

_, contour = _contour_rectangle()
# ANCHOR: to_quadratic
lin = pyrucast.mesh.triangulate_surface(contour, "TRI3", 1.0)  # TRI3 mesh
quad = pyrucast.mesh.to_quadratic(lin)  # TRI6 copy
print(quad.element_types())  # ['TRI6']

fes = pyrucast.FiniteElementSpace(quad, interpolation="LAGRANGE2")
# ANCHOR_END: to_quadratic
assert quad.element_types() == ["TRI6"]
assert len(fes) == 1


def _solide_penta6():
    """A block: a triangulated 3-D square, extruded along +z."""
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
faces = pyrucast.mesh.skin(volume)  # QUA4 skin
faces = pyrucast.mesh.convert(faces, "TRI3")  # QUA4 → TRI3
print(faces.element_types())  # ['TRI3']
# ANCHOR_END: convert
assert set(faces.element_types()) == {"TRI3"}

# ── Triangulation of a surface with a hole ──────────────────────────────────


# ── triangulate surface with a hole ────────────────────────

# ANCHOR: triangulate_surface
import pyrucast

c = pyrucast.Coords(dim=2)

# Outer contour: 4×4 square (CCW).
outer = pyrucast.Mesh(c, "SEG2")
outer_nodes = [
    c.add_node(list(p)) for p in [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
]
for i in range(4):
    outer.unit().add_cell([outer_nodes[i], outer_nodes[(i + 1) % 4]])

# Hole: centred 2×2 square, oriented CW.
hole = pyrucast.Mesh(c, "SEG2")
hole_nodes = [
    c.add_node(list(p)) for p in [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]
]
for i in range(4):
    hole.unit().add_cell([hole_nodes[i], hole_nodes[(i + 1) % 4]])

# Compose the two contours through the union | (never +).
combined = outer | hole

# TRI3 mesh of size ~0.5 (area = 16 - 4 = 12).
tri = pyrucast.mesh.triangulate_surface(combined, "TRI3", size=0.5)
print(tri.element_types(), tri.cell_count())

# Quad-dominant variant.
quad = pyrucast.mesh.triangulate_surface(combined, "QUA4", size=0.5)
print(quad.element_types())  # ['QUA4', 'TRI3'] in general
# ANCHOR_END: triangulate_surface
assert tri.cell_count() > 0
assert set(quad.element_types()) <= {"QUA4", "TRI3"}

# ── Grid oriented on the contour ────────────────────────────────────────────


# ── grid surface ───────────────────────────────────────────

# ANCHOR: grid_surface
import pyrucast as pc

H = 0.02  # target size
coords = pc.Coords(2)

# An L shape. Each side is cut into a whole number of cells of size H, so all
# its nodes fall on the lines the grid will draw from the corners.
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
print(maillage.element_types())  # ['QUA4'] — not a single triangle
print(maillage.cell_count())  # 450: the exact grid of the L shape
# ANCHOR_END: grid_surface
assert maillage.element_types() == ["QUA4"]
assert maillage.cell_count() == 450


def _contour_en_L(H=0.02):
    """The L shape of the grid: each side cut into a whole number of cells."""
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
# or, as a method:
maillage = contour.grid_surface2("QUA4", size=H)
# ANCHOR_END: grid_surface2
assert maillage.cell_count() > 0

# ── Border and skin ─────────────────────────────────────────────────────────


# ── border ─────────────────────────────────────────────────

# ANCHOR: border
import pyrucast

c = pyrucast.Coords(dim=2)
center = c.add_node([0.0, 0.0])
disc = pyrucast.mesh.triangulate_surface(
    pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 2.0, 16), "TRI3"
)

bord = pyrucast.mesh.border(disc)
print(len(bord))  # 1  (simply connected domain)
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
print(len(aretes))  # 4  (the four sides, open edges)
# ANCHOR_END: border_angle
assert len(aretes) == 4

# ── skin ───────────────────────────────────────────────────

# ANCHOR: skin
import pyrucast

# A PENTA6 block: a triangulated square, extruded along +z.
c = pyrucast.Coords(dim=3)
coins = [c.add_node(p) for p in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour[0].add_cell([coins[i], coins[(i + 1) % 4]])
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3", 0.34)
solide = pyrucast.mesh.extrude(surf, [0.0, 0.0, 1.0], 3)  # TRI3 -> PENTA6

peau = pyrucast.mesh.skin(solide)
print(len(peau))  # 6  (two caps + four sides)
print(peau.element_types())  # ['TRI3', 'TRI3', 'QUA4', 'QUA4', 'QUA4', 'QUA4']
# ANCHOR_END: skin
assert len(peau) == 6

# ── Orientation, chaining ───────────────────────────────────────────────────


# ── orient and invert ──────────────────────────────────────

_, contour = _contour_rectangle()
# ANCHOR: orient
import pyrucast

# A plate with a hole: outer contour + hole border, arbitrary orientations.
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3")

propre = pyrucast.mesh.orient(surf)  # every cell made consistent
trou_dedans = pyrucast.mesh.invert(propre)  # reversed sense (inside/outside)
# ANCHOR_END: orient
assert propre.cell_count() == surf.cell_count()
assert trou_dedans.cell_count() == surf.cell_count()


def _surface_triangulee(taille=0.5):
    """A ready-to-use TRI3 surface, without naming `pyrucast` in the caller.

    An `import pyrucast` inside an anchor makes it a **local** variable of the test
    function: any use of the module before the anchor would fail.
    """
    _, contour = _contour_rectangle()
    return pyrucast.mesh.triangulate_surface(contour, "TRI3", taille)


# ── chain ──────────────────────────────────────────────────

surf = _surface_triangulee()
# ANCHOR: chain
import pyrucast

# A contour drawn from a surface: the segments are there, but in no order.
bord = pyrucast.mesh.border(surf)
suite = pyrucast.mesh.chain(bord)  # or bord.chain()

# The connectivity now reads node by node along the curve.
for maille in suite[0]:
    print([n.id for n in maille])
# ANCHOR_END: chain
assert suite.cell_count() == bord.cell_count()

# ── Selections ──────────────────────────────────────────────────────────────


# ── elements on ────────────────────────────────────────────

# ANCHOR: elements_on
import pyrucast

c = pyrucast.Coords(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0)]]

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([nodes[0], nodes[1], nodes[2]])  # cell 0
mesh.unit().add_cell([nodes[1], nodes[3], nodes[2]])  # cell 1

# Points = {0, 1, 2}: only cell 0 has all of its nodes in there.
pts = pyrucast.mesh.poi1_from_nodes([nodes[0], nodes[1], nodes[2]])

strict = pyrucast.mesh.elements_on(mesh, pts, strict=True)
print(strict.cell_count())  # 1  (cell 0)

loose = pyrucast.mesh.elements_on(mesh, pts, strict=False)
print(loose.cell_count())  # 2  (both touch a node of pts)
# ANCHOR_END: elements_on
assert strict.cell_count() == 1
assert loose.cell_count() == 2

# ── geometric selections ───────────────────────────────────

_, contour = _contour_rectangle(2.0, 2.0, 8)
# ANCHOR: selections
import pyrucast

# A square plate meshed in TRI3.
plaque = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.1)

# The left edge (x = 0): the plane of normal +x through the origin.
gauche = pyrucast.mesh.points_on_plane(plaque, [0.0, 0.0], [1.0, 0.0])

# The nodes of the fillet: inside the disc of radius 0.2 around the re-entrant corner.
conge = pyrucast.mesh.points_in_sphere(plaque, [1.0, 1.0], 0.2)

# The selection serves directly as the imposed support of a Dirichlet — the
# POI1 cloud is what `model.dirichlet` expects (cf. Constraints / Dirichlet).
mecanique = pyrucast.model.elasticity(
    pyrucast.FiniteElementSpace(plaque), "plane_stress"
)
blocage = pyrucast.model.dirichlet(
    mecanique, "u_x", gauche, pyrucast.mesh.barycenter(gauche)
)

# The POI1 output is an ordinary mesh: it plugs back into the other operators,
# here to go back up to the elements carried by the selection.
bande = pyrucast.mesh.elements_on(plaque, conge, strict=True)
# ANCHOR_END: selections
assert gauche.cell_count() > 0
assert len(blocage) == 1

# ── Welding of colocated nodes ──────────────────────────────────────────────


# ── merge nodes ────────────────────────────────────────────

# ANCHOR: merge_nodes
import pyrucast

# A mesh whose interface carries colocated but distinct nodes (two SEG2 that
# touch through a duplicated end).
c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
b2 = c.add_node([1.0, 0.0])  # on top of b, but a distinct node
d = c.add_node([2.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
mesh.unit().add_cell([b2, d])

joined = pyrucast.mesh.merge_nodes(mesh, 1e-6)  # b2 is welded onto b
# ANCHOR_END: merge_nodes
assert joined.cell_count() == 2

# ── merge nodes in place ───────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
b2, d = c.add_node([1.0, 0.0]), c.add_node([2.0, 0.0])
# ANCHOR: merge_in_place
gauche = pyrucast.mesh.line(a, b, 4)
droite = pyrucast.mesh.line(b2, d, 4)  # b2 colocated with b, but distinct

pyrucast.mesh.merge_nodes(gauche | droite, 1e-6, in_place=True)

# The two pieces now really share the interface node.
assert droite.node(0, 0, 0).id == b.id
# ANCHOR_END: merge_in_place


# ── Selections on curved surfaces ───────────────────────────────────────────


def _tube_3d():
    """A tube: a QUA4 section revolved a full turn about the z axis."""
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
# The bore of a tube: the lateral surface of the cylinder of inner radius.
alesage = pyrucast.mesh.points_on_cylinder(tube, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 5.0)

# A conical chamfer (radius 8 at z = 0, fictitious apex at z = 8).
chanfrein = pyrucast.mesh.points_on_cone(piece, [0.0, 0.0, 0.0], [0.0, 0.0, 8.0], 8.0)

# The material around a toroidal groove of radius 1 on a circle of radius 5.
gorge = pyrucast.mesh.points_in_torus(piece, [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 5.0, 1.0)
# ANCHOR_END: selections_courbes

assert alesage.cell_count() > 0


# ── Tetrahedralisation of a volume ──────────────────────────────────────────

solide_penta6 = _solide_penta6()
# `skin` already returns **outward** normals: no `invert` here, it would turn
# them inwards and the mesher would reject the envelope.
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
    print(f"{ajoutes} node(s) laid on the skin")
# ANCHOR_END: surface_nodes_compte

assert solide.cell_count() > 0


# ── Quadrangular paving, then extrusion into hexahedra ──────────────────────


def _contour_plaque_trouee(h=0.05):
    """CCW outer contour + CW hole circle, each loop with an even number of
    segments — the condition for `all_quad` to be able to succeed."""
    # A 3-D `Coords` from the start: the extrusion towards +z of the block below
    # requires a three-component direction, hence three-component nodes.
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
    # Each loop is consolidated **separately**: melting both into a single
    # sub-mesh would produce a repeated node, and `pave_surface` requires each
    # border sub-mesh to be a simple loop.
    return pyrucast.mesh.consolidate(exterieur) | pyrucast.mesh.consolidate(trou)


contour = _contour_plaque_trouee()

# ANCHOR: pave_surface
import pyrucast as pc

# … CCW outer contour and CW hole circle, each consolidated into one loop.
# Each loop of the contour has an even number of segments, so all_quad works.
plaque = pc.mesh.pave_surface(contour, "QUA4", size=0.05, all_quad=True)
print(plaque.element_types())  # ['QUA4']

# The prismatic solid then comes for free, and in pure hexahedra.
volume = pc.mesh.extrude(plaque, [0, 0, 0.02], 2)
print(volume.element_types())  # ['HEX8']
# ANCHOR_END: pave_surface

assert plaque.element_types() == ["QUA4"]
assert volume.element_types() == ["HEX8"]


# ── Hexahedral boundary layer over a tetrahedral core ───────────────────────


def _boite_hex(n=3):
    """The skin of an n³ box of hexahedra: closed QUA4 shell, outward normals."""
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

peau = pc.mesh.skin(solide)  # QUA4, outward normals
maille = pc.mesh.pave_volume(peau, layers=1, thickness=0.15, size=0.4)
print(dict(zip(maille.element_types(), maille.cell_counts())))
# {'HEX8': 54, 'PYRA5': 54, 'TET4': 408}
# ANCHOR_END: pave_volume

assert dict(zip(maille.element_types(), maille.cell_counts()))["HEX8"] == 54


# ── Tetrahedralising a given skin ───────────────────────────────────────────

solide_penta6 = _solide_penta6()

# ANCHOR: triangulate_volume_taille
peau = pyrucast.mesh.convert(pyrucast.mesh.skin(solide_penta6), "TRI3")
volume = pyrucast.mesh.triangulate_volume(peau, size=0.3)
# ANCHOR_END: triangulate_volume_taille

assert volume.element_types()[0] == "TET4"


# ── Reading a gmsh file ─────────────────────────────────────────────────────

# The file of the example is written into a throw-away directory, and the
# module switches into it for the duration of the excerpt: the latter therefore
# keeps the short name `piece.msh` a user would write. The working directory is
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
# {'plate': Mesh<…>, 'bottom': Mesh<…>, …}  — order of the file preserved

plate = regions["plate"]
print(plate.element_types())  # e.g. ['TRI3']
print(plate.cell_count())
# ANCHOR_END: read_gmsh

assert plate.element_types() == ["TRI3"]
assert plate.cell_count() == 2

os.chdir(_CWD)


# ANCHOR: from_gmsh_arrays
# The same square, but as gmsh hands it over in memory: the tags of the nodes,
# their three coordinates each, then one block per element type whose
# connectivity is flattened.
tags = [1, 2, 3, 4]
xyz = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
blocs = [
    (1, [1, 2], ["bottom"]),  # code 1: SEG2
    (2, [1, 2, 3, 1, 3, 4], ["plate"]),  # code 2: TRI3
]

coords = pyrucast.Coords(dim=2)
regions = pyrucast.mesh.from_gmsh_arrays(coords, tags, xyz, blocs)
print(regions["plate"].element_types())  # ['TRI3']
print(coords.node_count())  # 4 — a single Coords for both groups
# ANCHOR_END: from_gmsh_arrays

assert regions["plate"].element_types() == ["TRI3"]
assert regions["plate"].cell_count() == 2
assert coords.node_count() == 4
