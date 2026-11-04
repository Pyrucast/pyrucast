"""Source of the Python examples of `book/src/elements/`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Each element's excerpt reads its reference tables from an **empty** zone: the
Gauss points, the weights and the shape functions depend on the element type
only, not on a cell.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import math


def arrondi(valeurs):
    return [round(v, 6) + 0.0 for v in valeurs]


# ── Les degrés de liberté d'une maille ──────────────────────────────────────

# ANCHOR: ddl
import pyrucast

# Un TRI3, une maille, en élasticité plane puis en conduction.
c = pyrucast.Coords(2)
maille = pyrucast.Mesh(c, "TRI3")
maille.unit().add_cell([c.add_node(p) for p in [[0, 0], [1, 0], [0, 1]]])
fes = pyrucast.FiniteElementSpace(maille)

solide = pyrucast.model.elasticity(fes, "plane_stress")
k = pyrucast.matrix.stiffness(
    solide, pyrucast.element_field.material_field(solide, [("E", 1.0), ("nu", 0.0)])
)
print(k.n_cols)  # 6 = 3 nœuds × 2 inconnues
print(k.col_dofs()[:4])  # [(0, 'u_x'), (0, 'u_y'), (1, 'u_x'), (1, 'u_y')]

chaleur = pyrucast.model.heat_conduction(fes)
k = pyrucast.matrix.stiffness(
    chaleur, pyrucast.element_field.material_field(chaleur, [("k", 1.0)])
)
print(k.col_dofs())  # [(0, 'T'), (1, 'T'), (2, 'T')]
# ANCHOR_END: ddl

assert k.col_dofs() == [(0, "T"), (1, "T"), (2, "T")]

# ── La règle réduite ────────────────────────────────────────────────────────

# ANCHOR: reduite
# Un point au centre, de poids la mesure de l'élément de référence.
centre = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(3), "PYRA5"), quadrature="REDUCED"
)[0]
print(centre.gauss_count, centre.gauss_xi(0))  # 1 [0.0, 0.0, 0.25]
print(round(centre.gauss_weight(0), 6))  # 1.333333 = 4/3, le volume de la pyramide
# ANCHOR_END: reduite

assert centre.gauss_xi(0) == [0.0, 0.0, 0.25]

# ── SEG2 ────────────────────────────────────────────────────────────────────

# ANCHOR: seg2
import pyrucast

# Les tables de référence d'un SEG2 : une zone vide suffit.
seg2 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(1), "SEG2"))[0]
print(seg2.nodes_per_cell, seg2.gauss_count)  # 2 2

# Les deux points en ∓1/√3, de poids 1.
print([round(seg2.gauss_xi(g)[0], 6) for g in range(2)])  # [-0.57735, 0.57735]
print([seg2.gauss_weight(g) for g in range(2)])  # [1.0, 1.0]
# N = ((1 − ξ)/2, (1 + ξ)/2) au premier point ; les dérivées, constantes.
print([round(v, 6) for v in seg2.n_at_g(0)])  # [0.788675, 0.211325]
print(seg2.dn_at_g(0))  # [-0.5, 0.5]
# ANCHOR_END: seg2

assert arrondi(seg2.n_at_g(0)) == [0.788675, 0.211325]
assert seg2.dn_at_g(0) == [-0.5, 0.5]

# ── SEG3 ────────────────────────────────────────────────────────────────────

# ANCHOR: seg3
import pyrucast

# Un type quadratique demande l'interpolation de degré 2.
seg3 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(1), "SEG3"), interpolation="LAGRANGE2"
)[0]
print(seg3.nodes_per_cell, seg3.gauss_count)  # 3 3

# Les trois points en −√(3/5), 0, +√(3/5), de poids 5/9, 8/9, 5/9.
print([round(seg3.gauss_xi(g)[0], 6) for g in range(3)])  # [-0.774597, 0.0, 0.774597]
poids = [round(seg3.gauss_weight(g), 6) for g in range(3)]
print(poids)  # [0.555556, 0.888889, 0.555556]
# Au point du milieu, ξ = 0, seule la fonction du nœud 2 est non nulle.
print([i for i, v in enumerate(seg3.n_at_g(1)) if v != 0.0])  # [2]
# Les dérivées (ξ − ½, ξ + ½, −2ξ) au premier point.
print([round(v, 6) for v in seg3.dn_at_g(0)])  # [-1.274597, -0.274597, 1.549193]
# ANCHOR_END: seg3

assert arrondi(seg3.n_at_g(1)) == [0.0, 0.0, 1.0]
assert arrondi(seg3.dn_at_g(0)) == [-1.274597, -0.274597, 1.549193]

# ── TRI3 ────────────────────────────────────────────────────────────────────

# ANCHOR: tri3
import pyrucast

# Les tables de référence d'un TRI3 : une zone vide suffit.
tri3 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(2), "TRI3"))[0]
print(tri3.nodes_per_cell, tri3.gauss_count)  # 3 3

# Le point 0, au milieu du côté (0, 1), de poids 1/6.
print(tri3.gauss_xi(0), round(tri3.gauss_weight(0), 6))  # [0.5, 0.0] 0.166667
# N = (1 − ξ − η, ξ, η) en ce point.
print(tri3.n_at_g(0))  # [0.5, 0.5, 0.0]
# Les dérivées, les mêmes en tout point : (−1, −1), (1, 0), (0, 1).
print(tri3.dn_at_g(0))  # [-1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
# ANCHOR_END: tri3

assert tri3.n_at_g(0) == [0.5, 0.5, 0.0]
assert tri3.dn_at_g(0) == [-1.0, -1.0, 1.0, 0.0, 0.0, 1.0]

# ── QUA4 ────────────────────────────────────────────────────────────────────

# ANCHOR: qua4
import pyrucast

qua4 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(2), "QUA4"))[0]
print(qua4.nodes_per_cell, qua4.gauss_count)  # 4 4

# Le point 0 en (−1/√3, −1/√3), de poids 1.
print([round(x, 6) for x in qua4.gauss_xi(0)], qua4.gauss_weight(0))
# [-0.57735, -0.57735] 1.0
# N_i = ¼ (1 + ξ_i ξ)(1 + η_i η) : le nœud 0, le plus proche, pèse le plus.
print([round(v, 6) for v in qua4.n_at_g(0)])  # [0.622008, 0.166667, 0.044658, 0.166667]
# ANCHOR_END: qua4

assert arrondi(qua4.n_at_g(0)) == [0.622008, 0.166667, 0.044658, 0.166667]

# ── TET4 ────────────────────────────────────────────────────────────────────

# ANCHOR: tet4
import pyrucast

tet4 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(3), "TET4"))[0]
print(tet4.nodes_per_cell, tet4.gauss_count)  # 4 4

# Le point 0 en (α, α, α), α = (5 − √5)/20, de poids 1/24.
print([round(x, 6) for x in tet4.gauss_xi(0)], round(tet4.gauss_weight(0), 6))
# [0.138197, 0.138197, 0.138197] 0.041667
# N = (1 − ξ − η − ζ, ξ, η, ζ) : β au nœud 0, α aux trois autres.
print([round(v, 6) for v in tet4.n_at_g(0)])  # [0.58541, 0.138197, 0.138197, 0.138197]
# ANCHOR_END: tet4

assert arrondi(tet4.n_at_g(0)) == [0.58541, 0.138197, 0.138197, 0.138197]
assert abs(tet4.n_at_g(0)[0] - (5 + 3 * math.sqrt(5)) / 20) < 1e-15

# ── PYRA5 ───────────────────────────────────────────────────────────────────

# ANCHOR: pyra5
import pyrucast

pyra5 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(3), "PYRA5"))[0]
print(pyra5.nodes_per_cell, pyra5.gauss_count)  # 5 8

# Deux étages en ζ = 1/3 ∓ √10/15 ; la section rétrécit vers le sommet.
for g in (0, 1):
    print([round(x, 6) for x in pyra5.gauss_xi(g)], round(pyra5.gauss_weight(g), 6))
# [-0.506616, -0.506616, 0.122515] 0.232547
# [-0.263184, -0.263184, 0.544152] 0.100786
print(round(sum(pyra5.gauss_weight(g) for g in range(8)), 12))  # 1.333333333333 = 4/3

# Le sommet ne porte que ζ : N_4 vaut la hauteur du point.
print(round(pyra5.n_at_g(0)[4], 6))  # 0.122515
# ANCHOR_END: pyra5

assert abs(pyra5.gauss_xi(0)[2] - (1 / 3 - math.sqrt(10) / 15)) < 1e-15
assert abs(pyra5.n_at_g(0)[4] - pyra5.gauss_xi(0)[2]) < 1e-15

# ── PENTA6 ──────────────────────────────────────────────────────────────────

# ANCHOR: penta6
import pyrucast

penta6 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(3), "PENTA6"))[0]
print(penta6.nodes_per_cell, penta6.gauss_count)  # 6 6

# Les points du TRI3, à deux hauteurs ζ = ½ ∓ 1/(2√3), de poids 1/12.
print([round(x, 6) for x in penta6.gauss_xi(0)], round(penta6.gauss_weight(0), 6))
# [0.5, 0.0, 0.211325] 0.083333
# N = L_j (1 − ζ) en bas, L_j ζ en haut.
print([round(v, 6) for v in penta6.n_at_g(0)])
# [0.394338, 0.394338, 0.0, 0.105662, 0.105662, 0.0]
# ANCHOR_END: penta6

assert arrondi(penta6.n_at_g(0)) == [0.394338, 0.394338, 0.0, 0.105662, 0.105662, 0.0]

# ── HEX8 ────────────────────────────────────────────────────────────────────

# ANCHOR: hex8
import pyrucast

hex8 = pyrucast.FiniteElementSpace(pyrucast.Mesh(pyrucast.Coords(3), "HEX8"))[0]
print(hex8.nodes_per_cell, hex8.gauss_count)  # 8 8

# Le point 0 en (−1/√3, −1/√3, −1/√3), de poids 1.
print([round(x, 6) for x in hex8.gauss_xi(0)], hex8.gauss_weight(0))
# [-0.57735, -0.57735, -0.57735] 1.0
# Le nœud 0, le plus proche, et le nœud 6, à l'opposé.
n = hex8.n_at_g(0)
print(round(n[0], 6), round(n[6], 6))  # 0.490563 0.009437
# ANCHOR_END: hex8

assert (round(n[0], 6), round(n[6], 6)) == (0.490563, 0.009437)
assert abs(n[0] - (1 + 1 / math.sqrt(3)) ** 3 / 8) < 1e-15

# ── TRI6 ────────────────────────────────────────────────────────────────────

# ANCHOR: tri6
import pyrucast

tri6 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(2), "TRI6"), interpolation="LAGRANGE2"
)[0]
print(tri6.nodes_per_cell, tri6.gauss_count)  # 6 6

# Deux familles de trois points, chacune avec son poids.
print([round(tri6.gauss_weight(g), 6) for g in range(6)])
# [0.111691, 0.111691, 0.111691, 0.054976, 0.054976, 0.054976]
# Au point 0, les fonctions des sommets peuvent être négatives.
print([round(v, 6) for v in tri6.n_at_g(0)])
# [-0.08473, -0.048208, -0.048208, 0.192834, 0.79548, 0.192834]
# ANCHOR_END: tri6

assert arrondi(tri6.n_at_g(0)) == [
    -0.08473,
    -0.048208,
    -0.048208,
    0.192834,
    0.79548,
    0.192834,
]

# ── QUA8 ────────────────────────────────────────────────────────────────────

# ANCHOR: qua8
import pyrucast

qua8 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(2), "QUA8"), interpolation="LAGRANGE2"
)[0]
print(qua8.nodes_per_cell, qua8.gauss_count)  # 8 9

# Le point 4 est le centre (0, 0) : les sommets y valent −¼, les milieux ½.
print(qua8.gauss_xi(4), round(qua8.gauss_weight(4), 6))  # [0.0, 0.0] 0.790123
print(qua8.n_at_g(4))  # [-0.25, -0.25, -0.25, -0.25, 0.5, 0.5, 0.5, 0.5]
# ANCHOR_END: qua8

assert qua8.n_at_g(4) == [-0.25] * 4 + [0.5] * 4

# ── QUA9 ────────────────────────────────────────────────────────────────────

# ANCHOR: qua9
import pyrucast

qua9 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(2), "QUA9"), interpolation="LAGRANGE2"
)[0]
print(qua9.nodes_per_cell, qua9.gauss_count)  # 9 9

# Au centre, seul le nœud central est non nul : N_8 = (1 − ξ²)(1 − η²) = 1.
n = qua9.n_at_g(4)
print(qua9.gauss_xi(4), [i for i, v in enumerate(n) if abs(v) > 1e-12], n[8])
# [0.0, 0.0] [8] 1.0
# ANCHOR_END: qua9

assert arrondi(n) == [0.0] * 8 + [1.0]

# ── TET10 ───────────────────────────────────────────────────────────────────

# ANCHOR: tet10
import pyrucast

tet10 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(3), "TET10"), interpolation="LAGRANGE2"
)[0]
print(tet10.nodes_per_cell, tet10.gauss_count)  # 10 11

# Le point 0 est le centre, de poids négatif −74/5625.
print(tet10.gauss_xi(0), round(tet10.gauss_weight(0), 6))
# [0.25, 0.25, 0.25] -0.013156
print(round(sum(tet10.gauss_weight(g) for g in range(11)), 12))  # 0.166666666667 = 1/6
# Au centre, L_i = ¼ : les sommets valent −⅛, les milieux ¼.
print(tet10.n_at_g(0))
# [-0.125, -0.125, -0.125, -0.125, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
# ANCHOR_END: tet10

assert abs(tet10.gauss_weight(0) + 74 / 5625) < 1e-15
assert tet10.n_at_g(0) == [-0.125] * 4 + [0.25] * 6

# ── PENTA15 ─────────────────────────────────────────────────────────────────

# ANCHOR: penta15
import pyrucast

penta15 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(3), "PENTA15"), interpolation="LAGRANGE2"
)[0]
print(penta15.nodes_per_cell, penta15.gauss_count)  # 15 18

# Les six points du TRI6, à trois hauteurs : le premier en ζ = ½ − ½√(3/5).
print([round(x, 6) for x in penta15.gauss_xi(0)])  # [0.445948, 0.445948, 0.112702]
print(round(sum(penta15.gauss_weight(g) for g in range(18)), 12))  # 0.5
# ANCHOR_END: penta15

assert abs(penta15.gauss_xi(0)[2] - (0.5 - 0.5 * math.sqrt(0.6))) < 1e-15

# ── HEX20 ───────────────────────────────────────────────────────────────────

# ANCHOR: hex20
import pyrucast

hex20 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(3), "HEX20"), interpolation="LAGRANGE2"
)[0]
print(hex20.nodes_per_cell, hex20.gauss_count)  # 20 27

# Le point 13 est le centre : les sommets y valent −¼, les milieux ¼.
n = hex20.n_at_g(13)
print(hex20.gauss_xi(13), n[0], n[8])  # [0.0, 0.0, 0.0] -0.25 0.25
print(sum(n))  # 1.0
# ANCHOR_END: hex20

assert n == [-0.25] * 8 + [0.25] * 12

# ── HEX27 ───────────────────────────────────────────────────────────────────

# ANCHOR: hex27
import pyrucast

hex27 = pyrucast.FiniteElementSpace(
    pyrucast.Mesh(pyrucast.Coords(3), "HEX27"), interpolation="LAGRANGE2"
)[0]
print(hex27.nodes_per_cell, hex27.gauss_count)  # 27 27

# Au centre, seul le nœud 26, le centre de volume, vaut 1.
n = hex27.n_at_g(13)
non_nuls = [i for i, v in enumerate(n) if abs(v) > 1e-12]
print(hex27.gauss_xi(13), non_nuls)  # [0.0, 0.0, 0.0] [26]
# ANCHOR_END: hex27

assert arrondi(n) == [0.0] * 26 + [1.0]
