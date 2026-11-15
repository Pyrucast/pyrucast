"""Examples the generated operator reference shows, and nothing else does.

Each anchor is named `module-function` (`node_field-restrict`): the reference
shows it under that function (see `script/book_pyapi.py`). A function whose
example already lives with a book page keeps it there; this file holds the
others. The setup lives outside the anchors, and every example is followed by
the assertions that check what it shows.

The code lives at module level, as in the other `test_doc_*.py` files: pytest
runs it at collection time, and an example that breaks is a collection error.
"""

import math
import os
import tempfile

import pyrucast


def _plaque():
    """A 2 × 1 plate of two QUA4, its FE space, and its six nodes.

    Nodes 0, 1, 2 along the bottom (y = 0), 3, 4, 5 along the top (y = 1).
    """
    c = pyrucast.Coords(2)
    n = [c.add_node([x, y]) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell([n[0], n[1], n[4], n[3]])
    mesh.unit().add_cell([n[1], n[2], n[5], n[4]])
    return c, mesh, pyrucast.FiniteElementSpace(mesh), n


def _nodal(nodes, components, value, support=None):
    """A `NodeField` on the POI1 cloud of `nodes`, `value(node)` per component."""
    support = support or pyrucast.mesh.poi1_from_nodes(nodes)
    f = pyrucast.NodeField(support, components)
    for node in nodes:
        for comp in components:
            f[0].set_value(node, comp, value(node))
    return f


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ── coords ──────────────────────────────────────────────────────────────────

_, mesh, _, n = _plaque()
# ANCHOR: coords-set
# The plate scaled by 2: positions read, multiplied, written back.
xy = pyrucast.node_field.positions(mesh)  # components X, Y
pyrucast.coords.set(xy * 2.0)
# ANCHOR_END: coords-set
assert n[5].position() == [4.0, 2.0]

_, mesh, _, n = _plaque()
u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x", "u_y"])
u[0].set_value(n[5], "u_y", 0.5)
# ANCHOR: coords-displace
# x += u on the active configuration; a node u does not move stays put.
pyrucast.coords.displace(u)
# ANCHOR_END: coords-displace
assert n[5].position() == [2.0, 1.5]
assert n[0].position() == [0.0, 0.0]


# ── node_field ──────────────────────────────────────────────────────────────

_, mesh, fes, n = _plaque()
# ANCHOR: node_field-positions
# One component per axis, one zone per submesh of the mesh.
xy = pyrucast.node_field.positions(mesh)  # also mesh.positions()
print(xy.components(), xy.value(n[5], "X"))  # ['X', 'Y'] 2.0
# ANCHOR_END: node_field-positions
assert xy.components() == ["X", "Y"]
assert xy.value(n[5], "Y") == 1.0

temperature = _nodal(n, ["T"], lambda node: 10.0 * node.position()[0])
gauche = pyrucast.mesh.poi1_from_nodes([n[0], n[3]])
# ANCHOR: node_field-restrict
# The temperature on the left edge alone; a node of the edge it lacks reads 0.
bord = pyrucast.node_field.restrict(temperature, gauche)
print(bord.node_count)  # 2
# ANCHOR_END: node_field-restrict
assert bord.node_count == 2
assert bord.value(n[3], "T") == 0.0

cible = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes([n[2], n[5]]), ["T"])
# ANCHOR: node_field-restrict_like
# The temperature read on the support and components of `cible`, so the two
# combine directly.
lu = pyrucast.node_field.restrict_like(temperature, cible)
somme = cible + lu
# ANCHOR_END: node_field-restrict_like
assert somme.value(n[2], "T") == 20.0

support = pyrucast.mesh.poi1_from_nodes(n)
a = _nodal(n, ["T"], lambda node: 1.0, support)
b = _nodal(n, ["p"], lambda node: 2.0, support)
# ANCHOR: node_field-merge
# Two fields on one support: one zone carrying both components.
ab = pyrucast.node_field.merge(a, b)  # the same as a | b
print(len(ab), ab.components())  # 1 ['T', 'p']
# ANCHOR_END: node_field-merge
assert len(ab) == 1 and sorted(ab.components()) == ["T", "p"]

# ANCHOR: node_field-consolidate
# Fuse the zones that lie on the same support.
une = pyrucast.node_field.consolidate(a | b)
# ANCHOR_END: node_field-consolidate
assert len(une) == 1

flux = pyrucast.ElementField(fes, ["q_x", "q_y"])
for cell in range(2):
    flux[0].set_cell_uniform(cell, "q_x", 1.0)
# ANCHOR: node_field-divergence
# A uniform vector q = (1, 0): its weak divergence ∫ ∇Nᵢ·q, node by node.
d = pyrucast.node_field.divergence(flux, "q")  # one component, div_q
print(d.components())  # ['div_q']
# ANCHOR_END: node_field-divergence
assert d.components() == ["div_q"]
# The integral of ∇·q over the plate is zero for a uniform q.
assert _close(d.sum("div_q"), 0.0)

bord_droit = pyrucast.Mesh(mesh.coords(), "SEG2")
bord_droit.unit().add_cell([n[2], n[5]])
conduction = pyrucast.model.heat_conduction(fes)
source = pyrucast.model.flux(pyrucast.FiniteElementSpace(bord_droit), conduction, "q")
modele = conduction | source
materiaux = pyrucast.element_field.material_field(modele, [("k", 1.0), ("phi_q", 3.0)])
# ANCHOR: node_field-external_forces
# The given data of the model: here a flux density 3 on an edge of length 1.
f_ext = pyrucast.node_field.external_forces(modele, materiaux)
print(f_ext.sum("q"))  # 3.0
# ANCHOR_END: node_field-external_forces
assert _close(f_ext.sum("q"), 3.0)

signe = _nodal(n, ["v"], lambda node: node.position()[0] - 1.0)
# ANCHOR: node_field-mask
# 1 where v ≥ 0, 0 elsewhere; the product keeps the positive part.
positif = signe * pyrucast.node_field.mask(signe, ge=0.0)
# ANCHOR_END: node_field-mask
assert positif.value(n[0], "v") == 0.0
assert positif.value(n[2], "v") == 1.0


# ── element_field ───────────────────────────────────────────────────────────

_, mesh, fes, n = _plaque()
temperature = _nodal(n, ["T"], lambda node: 10.0 * node.position()[0])
# ANCHOR: element_field-gradient
# T = 10·x: its gradient at every Gauss point is (10, 0).
grad = pyrucast.element_field.gradient(temperature, fes)
print(grad.components())  # ['grad_T_x', 'grad_T_y']
# ANCHOR_END: element_field-gradient
assert _close(grad[0].value(0, 0, "grad_T_x"), 10.0, 1e-9)
assert _close(grad[0].value(0, 0, "grad_T_y"), 0.0, 1e-9)

# ANCHOR: element_field-interp_to_gauss
# The nodal temperature carried to the Gauss points, same component.
aux_points = pyrucast.element_field.interp_to_gauss(temperature, fes)
# ANCHOR_END: element_field-interp_to_gauss
assert aux_points.components() == ["T"]
assert 0.0 < aux_points[0].value(0, 0, "T") < 10.0  # inside the first cell

modele = pyrucast.model.elasticity(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 200e3), ("nu", 0.3), ("alpha", 1e-5)]
)
chaud = pyrucast.element_field.interp_to_gauss(
    _nodal(n, ["T"], lambda node: 120.0), fes
)
# ANCHOR: element_field-thermal_strain
# Heated from 20 to 120: a free dilation α·ΔT = 10⁻³ in every direction.
eps_th = pyrucast.element_field.thermal_strain(chaud, materiaux, fes, 20.0)
print(eps_th.components())  # ['eps_xx', 'eps_xy', 'eps_yy']
# ANCHOR_END: element_field-thermal_strain
assert _close(eps_th[0].value(0, 0, "eps_xx"), 1e-3)
assert _close(eps_th[0].value(0, 0, "eps_xy"), 0.0)

sigma = pyrucast.element_field.elastic_stress(modele, eps_th, materiaux)
# ANCHOR: element_field-elastic_strain
# Back from the stress to the strain that carries it: D⁻¹:σ.
eps = pyrucast.element_field.elastic_strain(modele, sigma, materiaux)
# ANCHOR_END: element_field-elastic_strain
assert _close(eps[0].value(0, 0, "eps_xx"), 1e-3, 1e-15)

un = pyrucast.ElementField(fes, ["un"])
for cell in range(2):
    un[0].set_cell_uniform(cell, "un", 1.0)
centres = pyrucast.FiniteElementSpace(mesh, quadrature="REDUCED")
# ANCHOR: element_field-cell_integral
# ∫ 1 over each cell: its area, on a space of one point per cell.
aires = pyrucast.element_field.cell_integral(un, centres)
# ANCHOR_END: element_field-cell_integral
assert _close(aires[0].value(0, 0, "un"), 1.0)

k = pyrucast.ElementField(fes, ["k"])
young = pyrucast.ElementField(fes, ["E"])
# ANCHOR: element_field-consolidate
# Two zones on one FE space, distinct components: one zone carrying both.
materiau = pyrucast.element_field.consolidate(k | young)
print(len(k | young), len(materiau))  # 2 1
# ANCHOR_END: element_field-consolidate
assert len(materiau) == 1 and sorted(materiau.components()) == ["E", "k"]

# ANCHOR: element_field-initial_state
# The rest state of an elastic model: every stress at zero.
repos = pyrucast.element_field.initial_state(modele, materiaux)
print(repos.components())  # ['sigma_xx', 'sigma_yy', 'sigma_xy']
# ANCHOR_END: element_field-initial_state
assert repos[0].value(0, 0, "sigma_xx") == 0.0

# ANCHOR: element_field-mask
# 1 where the stress is compressive, 0 elsewhere, point by point.
comprime = pyrucast.element_field.mask(sigma, lt=0.0, components=["sigma_xx"])
# ANCHOR_END: element_field-mask
assert comprime[0].value(0, 0, "sigma_xx") == 0.0  # a free dilation: no stress

conduction = pyrucast.model.heat_conduction(fes)
gauche = pyrucast.mesh.poi1_from_nodes([n[0], n[3]])
appui = pyrucast.model.dirichlet(gauche, conduction, "T")
# ANCHOR: element_field-material_field_per_sub_model
# One list per sub-model, in order; the support needs no material.
materiaux_t = pyrucast.element_field.material_field_per_sub_model(
    conduction | appui, [[("k", 50.0)], []]
)
# ANCHOR_END: element_field-material_field_per_sub_model
assert materiaux_t.components() == ["k"]

# ANCHOR: element_field-sub_material_field
# The material of one sub-model alone, as a single zone.
zone = pyrucast.element_field.sub_material_field(conduction[0], [("k", 50.0)])
# ANCHOR_END: element_field-sub_material_field
assert zone.components() == ["k"]


# ── field: element-wise functions ───────────────────────────────────────────

_, mesh, _, n = _plaque()
x = _nodal(n, ["x"], lambda node: node.position()[0])  # 0, 1, 2 along the plate
# ANCHOR: field-abs
ecart = pyrucast.field.abs(x - 1.0)  # |x − 1|, also (x - 1.0).abs()
# ANCHOR_END: field-abs
assert ecart.value(n[0], "x") == 1.0
# ANCHOR: field-sqrt
racine = pyrucast.field.sqrt(x)  # a negative value would give nan
# ANCHOR_END: field-sqrt
assert _close(racine.value(n[2], "x"), math.sqrt(2.0))
# ANCHOR: field-exp
attenuation = pyrucast.field.exp(x * -0.5)
# ANCHOR_END: field-exp
assert _close(attenuation.value(n[2], "x"), math.exp(-1.0))
# ANCHOR: field-log
logarithme = pyrucast.field.log(x + 1.0)  # 0 gives -inf, a negative value nan
# ANCHOR_END: field-log
assert _close(logarithme.value(n[1], "x"), math.log(2.0))
# ANCHOR: field-log10
decades = pyrucast.field.log10(x * 10.0 + 1.0)
# ANCHOR_END: field-log10
assert _close(decades.value(n[2], "x"), math.log10(21.0))
# ANCHOR: field-cos
onde = pyrucast.field.cos(x * math.pi)  # radians
# ANCHOR_END: field-cos
assert _close(onde.value(n[1], "x"), -1.0)
# ANCHOR: field-sin
onde = pyrucast.field.sin(x * (math.pi / 2.0))
# ANCHOR_END: field-sin
assert _close(onde.value(n[1], "x"), 1.0)
# ANCHOR: field-tan
pente = pyrucast.field.tan(x * (math.pi / 4.0))
# ANCHOR_END: field-tan
assert _close(pente.value(n[1], "x"), 1.0)
# ANCHOR: field-sinh
s = pyrucast.field.sinh(x)
# ANCHOR_END: field-sinh
assert _close(s.value(n[1], "x"), math.sinh(1.0))
# ANCHOR: field-cosh
chainette = pyrucast.field.cosh(x - 1.0)  # a hanging chain, lowest at x = 1
# ANCHOR_END: field-cosh
assert _close(chainette.value(n[1], "x"), 1.0)
# ANCHOR: field-tanh
seuil = pyrucast.field.tanh((x - 1.0) * 10.0)  # a smooth step from −1 to 1
# ANCHOR_END: field-tanh
assert _close(seuil.value(n[1], "x"), 0.0)


# ── mesh: builders ──────────────────────────────────────────────────────────

c = pyrucast.Coords(2)
a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
# ANCHOR: mesh-from_live_nodes
# Every node of the Coords, as one POI1 zone.
tout = pyrucast.mesh.from_live_nodes(c)
# ANCHOR_END: mesh-from_live_nodes
assert tout.cell_count == 2

# ANCHOR: mesh-poi1_from_nodes
# A POI1 mesh on chosen nodes, in their order: the support of a constraint.
appui = pyrucast.mesh.poi1_from_nodes([a, b])
# ANCHOR_END: mesh-poi1_from_nodes
assert appui.element_types() == ["POI1"]

centre = a
debut, fin = c.add_node([1.0, 0.0]), c.add_node([0.0, 1.0])
# ANCHOR: mesh-arc
# A quarter circle of radius 1, from (1, 0) to (0, 1), in 8 cells.
quart = pyrucast.mesh.arc(debut, centre, fin, 8)
# ANCHOR_END: mesh-arc
assert quart.cell_count == 8

# ANCHOR: mesh-circle
# A closed circle of 16 SEG3 cells, radius 2, in the plane of normal z.
cercle = pyrucast.mesh.circle(centre, [0.0, 0.0, 1.0], 2.0, 16, "SEG3")
# ANCHOR_END: mesh-circle
assert cercle.cell_count == 16 and cercle.element_types() == ["SEG3"]

_, mesh, _, n = _plaque()
# ANCHOR: mesh-to_poi1
# The distinct nodes of the plate: six, though its two cells hold eight.
noeuds = pyrucast.mesh.to_poi1(mesh)
# ANCHOR_END: mesh-to_poi1
assert noeuds.cell_count == 6

double = mesh | pyrucast.mesh.copy(mesh, new_nodes=False)
# ANCHOR: mesh-consolidate
# Two submeshes of one type fused, the duplicate cells dropped.
seul = pyrucast.mesh.consolidate(double)
print(len(double), double.cell_count, len(seul), seul.cell_count)  # 2 4 1 2
# ANCHOR_END: mesh-consolidate
assert len(seul) == 1 and seul.cell_count == 2


# ── mesh: rigid copies ──────────────────────────────────────────────────────

_, mesh, _, n = _plaque()


def _coins(m):
    """The sorted positions of the nodes of `m`."""
    p = pyrucast.node_field.positions(m)
    return sorted(
        (round(p.value(node, "X"), 12), round(p.value(node, "Y"), 12))
        for node in [
            m.node(0, cell, k) for cell in range(m.cell_count) for k in range(4)
        ]
    )


# ANCHOR: mesh-translate
# A copy moved by (0, 3), on its own nodes; the plate does not move.
haut = pyrucast.mesh.translate(mesh, [0.0, 3.0])
# ANCHOR_END: mesh-translate
assert min(y for _, y in _coins(haut)) == 3.0
assert n[5].position() == [2.0, 1.0]

# ANCHOR: mesh-rotate
# A quarter turn about the origin, counterclockwise.
tourne = pyrucast.mesh.rotate(mesh, math.pi / 2.0, [0.0, 0.0])
# ANCHOR_END: mesh-rotate
assert max(x for x, _ in _coins(tourne)) == 0.0

# ANCHOR: mesh-symmetry_point
# Through the origin: every node goes to −x.
oppose = pyrucast.mesh.symmetry_point(mesh, [0.0, 0.0])
# ANCHOR_END: mesh-symmetry_point
assert max(x for x, _ in _coins(oppose)) == 0.0

# ANCHOR: mesh-symmetry_line
# The mirror image about the vertical line x = 0, cells reordered so that
# their Jacobians stay positive.
miroir = pyrucast.mesh.symmetry_line(mesh, [0.0, 0.0], [0.0, 1.0])
# ANCHOR_END: mesh-symmetry_line
assert min(x for x, _ in _coins(miroir)) == -2.0

c3 = pyrucast.Coords(3)
m3 = [c3.add_node(p) for p in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])]
carre = pyrucast.Mesh(c3, "QUA4")
carre.unit().add_cell(m3)
# ANCHOR: mesh-symmetry_plane
# In 3-D, through the plane z = 1 spanned by three points.
dessus = pyrucast.mesh.symmetry_plane(carre, [0, 0, 1], [1, 0, 1], [0, 1, 1])
# ANCHOR_END: mesh-symmetry_plane
assert dessus.cell_count == 1

# ANCHOR: mesh-invert
# Every cell turned over: the normals of a surface now point the other way.
retourne = pyrucast.mesh.invert(mesh)
# ANCHOR_END: mesh-invert
assert retourne.cell_count == mesh.cell_count


# ── mesh: node selection ────────────────────────────────────────────────────

_, mesh, _, n = _plaque()
# ANCHOR: mesh-points_in_sphere
# The nodes within 1.1 of the corner (0, 0): it, (1, 0) and (0, 1).
coin = pyrucast.mesh.points_in_sphere(mesh, [0.0, 0.0], 1.1)
# ANCHOR_END: mesh-points_in_sphere
assert coin.cell_count == 3
# ANCHOR: mesh-points_on_sphere
# The nodes at distance 1 from (1, 0), within the tolerance.
anneau = pyrucast.mesh.points_on_sphere(mesh, [1.0, 0.0], 1.0)
# ANCHOR_END: mesh-points_on_sphere
assert anneau.cell_count == 3
# ANCHOR: mesh-points_on_plane
# The right edge: the nodes on the plane x = 2 (a line, in 2-D).
droite = pyrucast.mesh.points_on_plane(mesh, [2.0, 0.0], [1.0, 0.0])
# ANCHOR_END: mesh-points_on_plane
assert droite.cell_count == 2
# ANCHOR: mesh-points_below_plane
# The half x ≤ 1: the side the normal (1, 0) points away from, plane included.
moitie = pyrucast.mesh.points_below_plane(mesh, [1.0, 0.0], [1.0, 0.0])
# ANCHOR_END: mesh-points_below_plane
assert moitie.cell_count == 4
# ANCHOR: mesh-points_on_line
# The diagonal through (0, 0) and (1, 1), unbounded.
diagonale = pyrucast.mesh.points_on_line(mesh, [0.0, 0.0], [1.0, 1.0])
# ANCHOR_END: mesh-points_on_line
assert diagonale.cell_count == 2
# ANCHOR: mesh-points_in_cylinder
# A thin cylinder clips the line y = 0 to the segment 0 ≤ x ≤ 1.
segment = pyrucast.mesh.points_in_cylinder(mesh, [0.0, 0.0], [1.0, 0.0], 0.1)
# ANCHOR_END: mesh-points_in_cylinder
assert segment.cell_count == 2
# ANCHOR: mesh-points_on_cylinder
# In 2-D, the lateral surface of a cylinder is its two long sides.
bords = pyrucast.mesh.points_on_cylinder(mesh, [0.0, 0.5], [2.0, 0.5], 0.5)
# ANCHOR_END: mesh-points_on_cylinder
assert bords.cell_count == 6
# ANCHOR: mesh-points_in_cone
# A cone of apex (2, 0.5) and radius 0.6 at x = 0: its radius shrinks to 0.3 at
# x = 1, so it holds the two nodes of the left edge, 0.5 from its axis.
pointe = pyrucast.mesh.points_in_cone(mesh, [0.0, 0.5], [2.0, 0.5], 0.6)
# ANCHOR_END: mesh-points_in_cone
assert pointe.cell_count == 2
# ANCHOR: mesh-points_on_cone
# The slanted sides of the cone of apex (2, 0) and radius 2 at (0, 0): the
# apex itself and (1, 1), where the radius has shrunk to 1.
flancs = pyrucast.mesh.points_on_cone(mesh, [0.0, 0.0], [2.0, 0.0], 2.0)
# ANCHOR_END: mesh-points_on_cone
assert flancs.cell_count == 2

c3 = pyrucast.Coords(3)
tore = [c3.add_node([math.cos(t), math.sin(t), 0.0]) for t in (0.0, 1.0, 2.0)]
tore.append(c3.add_node([0.0, 0.0, 0.0]))
nuage = pyrucast.mesh.poi1_from_nodes(tore)
# ANCHOR: mesh-points_in_torus
# A torus of major radius 1 around z: it holds the three points on its circle,
# not the centre.
dedans = pyrucast.mesh.points_in_torus(nuage, [0, 0, 0], [0, 0, 1], 1.0, 0.2)
# ANCHOR_END: mesh-points_in_torus
assert dedans.cell_count == 3
# ANCHOR: mesh-points_on_torus
# With a minor radius of 1, the centre lies on the tube's surface.
surface = pyrucast.mesh.points_on_torus(nuage, [0, 0, 0], [0, 0, 1], 1.0, 1.0)
# ANCHOR_END: mesh-points_on_torus
assert surface.cell_count == 1


# ── mesh: improvement ───────────────────────────────────────────────────────

c = pyrucast.Coords(2)
coins = [c.add_node(p) for p in ([0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0])]
cotes = [
    pyrucast.mesh.line(coins[i], coins[(i + 1) % 4], 4 if i % 2 == 0 else 3)
    for i in range(4)
]
contour = pyrucast.mesh.consolidate(cotes[0] | cotes[1] | cotes[2] | cotes[3])
pave = pyrucast.mesh.pave_surface(contour, "QUA4", 1.0)
# ANCHOR: mesh-regularize
# Move the interior nodes for better cells; the boundary never moves.
lisse = pyrucast.mesh.regularize(pave, sweeps=10)
# ANCHOR_END: mesh-regularize
assert lisse.cell_count == pave.cell_count
# ANCHOR: mesh-cleanup
# Fix the doublets and the wrong valences, then smooth.
propre = pyrucast.mesh.regularize(pyrucast.mesh.cleanup(pave))
# ANCHOR_END: mesh-cleanup
assert propre.cell_counts() == pave.cell_counts()  # already clean here
mixte = pyrucast.mesh.triangulate_surface(contour, "QUA4", 1.0)
# ANCHOR: mesh-merge_triangles
# The triangles a quadrangle-dominant mesh kept, removed two by two.
quads = pyrucast.mesh.merge_triangles(mixte)
# ANCHOR_END: mesh-merge_triangles
assert mixte.cell_counts()[1] == 6 and quads.cell_counts()[1] == 2


# ── mesh: gmsh and exchange ─────────────────────────────────────────────────

MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "plate"
$EndPhysicalNames
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
1 2 2 1 1 1 2 3
$EndElements
"""
# ANCHOR: mesh-read_gmsh_str
# A .msh text already in memory: one triangle in the physical group "plate".
regions = pyrucast.mesh.read_gmsh_str(pyrucast.Coords(2), MSH)
print(list(regions), regions["plate"].element_types())  # ['plate'] ['TRI3']
# ANCHOR_END: mesh-read_gmsh_str
assert list(regions) == ["plate"] and regions["plate"].cell_count == 1

# ANCHOR: mesh-element_type_from_gmsh
print(pyrucast.mesh.element_type_from_gmsh(2))  # TRI3, gmsh's 3-node triangle
# ANCHOR_END: mesh-element_type_from_gmsh
assert pyrucast.mesh.element_type_from_gmsh(2) == "TRI3"

ref = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]  # gmsh's reference triangle
# ANCHOR: mesh-gauss_to_external
# Pyrucast's rule for TRI3, carried into gmsh's reference triangle.
xi, poids = pyrucast.mesh.gauss_to_external("TRI3", ref, "gmsh")
print(len(poids), sum(poids))  # the points, and the area of the triangle: 0.5
# ANCHOR_END: mesh-gauss_to_external
assert _close(sum(poids), 0.5)
# ANCHOR: mesh-match_gauss
# Where each of pyrucast's points sits in an external rule — here the same one.
rang = pyrucast.mesh.match_gauss("TRI3", ref, xi, poids, "gmsh")
# ANCHOR_END: mesh-match_gauss
assert sorted(rang) == list(range(len(poids)))


# ── model: the continuum laws ───────────────────────────────────────────────

_, mesh, fes, n = _plaque()
# ANCHOR: model-gurson
# A porous metal: the porosity starts from f_0 in the rest state.
gurson = pyrucast.model.gurson(fes, "plane_strain")
acier = pyrucast.element_field.material_field(
    gurson,
    [
        ("E", 200e3),
        ("nu", 0.3),
        ("sigma_y", 250.0),
        ("q_1", 1.5),
        ("q_2", 1.0),
        ("q_3", 2.25),
        ("f_0", 0.001),
        ("f_c", 0.15),
        ("f_f", 0.25),
    ],
)
repos = pyrucast.element_field.initial_state(gurson, acier)
print(repos[0].value(0, 0, "porosity"))  # 0.001
# ANCHOR_END: model-gurson
assert _close(repos[0].value(0, 0, "porosity"), 0.001)

# ANCHOR: model-ottosen
# Concrete, whose strength depends on the pressure and the Lode angle.
beton = pyrucast.model.ottosen(fes, "plane_strain")
print(beton.primal_vars())  # ['u_x', 'u_y']
# ANCHOR_END: model-ottosen
assert beton.primal_vars() == ["u_x", "u_y"]

# ANCHOR: model-damage_sic_sic
# A woven composite: one damage per direction, along the material axes.
composite = pyrucast.model.damage_sic_sic(fes, "plane_stress")
# ANCHOR_END: model-damage_sic_sic
assert len(composite) == 1

# ANCHOR: model-creep_blackburn
# Primary and secondary creep: rate-dependent, it needs `dt` to integrate.
fluage = pyrucast.model.creep_blackburn(fes, "plane_strain")
# ANCHOR_END: model-creep_blackburn
assert fluage.primal_vars() == ["u_x", "u_y"]

# ANCHOR: model-creep_lemaitre
# Primary creep by strain hardening: rate-dependent, it needs `dt`.
fluage = pyrucast.model.creep_lemaitre(fes, "plane_strain")
# ANCHOR_END: model-creep_lemaitre
assert len(fluage) == 1

# ANCHOR: model-viscoplasticity_chaboche
# Kinematic hardening for cyclic loading; large rotations read Green-Lagrange.
cyclique = pyrucast.model.viscoplasticity_chaboche(
    fes, "plane_strain", strain_measure="green_lagrange"
)
# ANCHOR_END: model-viscoplasticity_chaboche
assert len(cyclique) == 1

# ANCHOR: model-viscoplasticity_lemaitre_chaboche
# The same flow, coupled to a ductile damage that drives it to rupture.
rupture = pyrucast.model.viscoplasticity_lemaitre_chaboche(fes, "plane_strain")
# ANCHOR_END: model-viscoplasticity_lemaitre_chaboche
assert len(rupture) == 1


# ── pyrucast: archive and spilling ──────────────────────────────────────────

_, mesh, _, n = _plaque()
temperature = _nodal(n, ["T"], lambda node: 10.0 * node.position()[0])
dossier = tempfile.mkdtemp()
chemin = os.path.join(dossier, "plaque.prc")
# ANCHOR: pyrucast-save
# The mesh and its field, written once each: the field's support is the mesh's.
pyrucast.save(chemin, {"plate": mesh, "temperature (°C)": temperature})
# ANCHOR_END: pyrucast-save
assert os.path.getsize(chemin) > 0

# ANCHOR: pyrucast-load
# New objects, alongside whatever the session already holds.
relu = pyrucast.load(chemin)
print(sorted(relu))  # ['plate', 'temperature (°C)']
# ANCHOR_END: pyrucast-load
assert relu["plate"].cell_count == 2
assert relu["temperature (°C)"].value(relu["plate"].node(0, 0, 2), "T") == 10.0
