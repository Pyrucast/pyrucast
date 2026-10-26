"""Source of the examples of `book/src/mecanique/grandes-transformations.md`.

Every block of the page comes from here through `{{#include …:anchor}}`. The
setup lives outside the anchors. See `book/src/developper/documentation-et-tests.md`.

The code lives at module level, not inside test functions (mdbook does not
strip the indentation of an included excerpt): pytest runs it at collection.
"""

import math

import pyrucast as pc

POINTS = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]


def _plaque():
    """A plate of two QUA4 on [0, 2]×[0, 1], its FE space and its nodes."""
    c = pc.Coords(2)
    n = [c.add_node(p) for p in POINTS]
    mesh = pc.Mesh(c, "QUA4")
    mesh.unit().add_cell([n[0], n[1], n[4], n[3]])
    mesh.unit().add_cell([n[1], n[2], n[5], n[4]])
    return c, mesh, pc.FiniteElementSpace(mesh), n


def _mouvement(mesh, noeuds, a):
    """The displacement of the affine motion x = A X."""
    u = pc.NodeField(mesh, ["u_x", "u_y"])
    for noeud, (x, y) in zip(noeuds, POINTS, strict=True):
        u[0].set_value(noeud, "u_x", a[0][0] * x + a[0][1] * y - x)
        u[0].set_value(noeud, "u_y", a[1][0] * x + a[1][1] * y - y)
    return u


coords, maillage, fes, noeuds = _plaque()

# ANCHOR: modele
# Hooke read on the Green–Lagrange strain: Saint-Venant–Kirchhoff. The law
# reads E_* and writes S_*, the second Piola–Kirchhoff stress.
solide = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
materiaux = pc.element_field.material_field(solide, [("E", 100.0), ("nu", 0.3)])
# ANCHOR_END: modele

c60, s60 = math.cos(math.pi / 3), math.sin(math.pi / 3)
u = _mouvement(maillage, noeuds, [[c60, -s60], [s60, c60]])

# ANCHOR: rotation
# A rigid rotation of 60°: the strain, the stress and the forces are zero.
E = solide.strain(u, materiaux)  # E_xx, E_xy, E_yy
S = solide.integrate_behavior(E, materiaux)  # S_xx, S_xy, S_yy
F = pc.element_field.deformation(u, fes, "gradient")
P = pc.element_field.stress_measure(S, F, "pk1")  # P = F S
f_int = solide.internal_forces(P, u, materiaux)  # ∫ P·∇₀N dV₀
# ANCHOR_END: rotation
assert [z.components() for z in E] == [["E_xx", "E_xy", "E_yy"]]
assert [z.components() for z in S] == [["S_xx", "S_yy", "S_xy"]]
for zone in E:
    for nom in ("E_xx", "E_xy", "E_yy"):
        assert abs(zone.value(0, 0, nom)) < 1e-14
for noeud in noeuds:
    for dual in ("f_x", "f_y"):
        assert abs(f_int.value(noeud, dual)) < 1e-12
# The linear strain of the same motion is cos θ − 1 = −½ on the diagonal.
lineaire = pc.model.elasticity(fes, "plane_strain")
eps = lineaire.strain(u, materiaux)
assert abs(eps[0].value(0, 0, "eps_xx") + 0.5) < 1e-12

# A stretch, then the same stretch turned: the Cauchy stress turns with it.
etire = _mouvement(maillage, noeuds, [[1.2, 0.0], [0.0, 1.0]])
tourne = _mouvement(maillage, noeuds, [[1.2 * c60, -s60], [1.2 * s60, c60]])
u = etire

# ANCHOR: cauchy
S = solide.integrate_behavior(solide.strain(u, materiaux), materiaux)
F = pc.element_field.deformation(u, fes, "gradient")
sigma = pc.element_field.stress_measure(S, F, "cauchy")  # J⁻¹ F S Fᵀ
# ANCHOR_END: cauchy
sxx = sigma[0].value(0, 0, "sigma_xx")
S_t = solide.integrate_behavior(solide.strain(tourne, materiaux), materiaux)
F_t = pc.element_field.deformation(tourne, fes, "gradient")
sigma_t = pc.element_field.stress_measure(S_t, F_t, "cauchy")[0]
# σ' = R σ Rᵀ with σ = diag(σ_xx, σ_yy): its trace is unchanged, and
# σ'_xy = cos θ sin θ (σ_xx − σ_yy).
syy = sigma[0].value(0, 0, "sigma_yy")
assert (
    abs(sigma_t.value(0, 0, "sigma_xx") + sigma_t.value(0, 0, "sigma_yy") - sxx - syy)
    < 1e-10
)
assert abs(sigma_t.value(0, 0, "sigma_xy") - c60 * s60 * (sxx - syy)) < 1e-10
# Back to S: the conversion is invertible.
retour = pc.element_field.stress_measure(sigma, F, "pk2")
assert abs(retour[0].value(0, 0, "S_xx") - S[0].value(0, 0, "S_xx")) < 1e-12

# ANCHOR: configuration_courante
# The current configuration x = X + u, laid once in a set of its own: the
# reference stays where it is, and in_config makes it active for one block.
courante = coords.add_config("current")
pc.coords.displace_from(u, 0, courante, ["u_x", "u_y"])
with coords.in_config(courante):
    K = pc.matrix.stiffness(solide, materiaux)  # on x = X + u
    K_sigma = pc.matrix.geometric(solide, materiaux, sigma)  # K_σ with Cauchy
# ANCHOR_END: configuration_courante
assert coords.active == 0
with coords.in_config(courante):
    assert abs(noeuds[2].position()[0] - 2.4) < 1e-14
assert noeuds[2].position() == [2.0, 0.0]

# A law reading Hencky writes T_*, which stress_measure turns into S, then P.
# Plane strain, stretched 1.2 along x: H_xx = ln 1.2 alone, T_xx = (λ + 2μ) H_xx,
# S_xx = T_xx / 1.2², and the nominal stress is P_xx = T_xx / 1.2.
hencky = pc.model.elasticity(fes, "plane_strain", strain_measure="hencky")
T = hencky.integrate_behavior(hencky.strain(etire, materiaux), materiaux)
assert [z.components() for z in T] == [["T_xx", "T_yy", "T_xy"]]
F = pc.element_field.deformation(etire, fes, "gradient")
P = pc.element_field.stress_measure(T, F, "pk1")
lame, mu = 100.0 * 0.3 / (1.3 * 0.4), 100.0 / 2.6
assert abs(P[0].value(0, 0, "P_xx") - (lame + 2.0 * mu) * math.log(1.2) / 1.2) < 1e-10

# The contour of the plate, travelled counter-clockwise: its normals point out.
contour = pc.Mesh(coords, "SEG2")
for a, b in [(0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)]:
    contour.unit().add_cell([noeuds[a], noeuds[b]])

# ANCHOR: pression
# A pressure p = 1 pushing on the walls: a density along the normal of each
# cell, phi_n = −p, into every displacement row of the solid.
pression = pc.model.flux(pc.FiniteElementSpace(contour), solide, "n")
densite = pc.element_field.material_field(pression, [("phi_n", -1.0)])
with coords.in_config(courante):  # the wall where it is: x = X + u
    f_ext = pc.node_field.external_forces(pression, densite)
# ANCHOR_END: pression
# On the stretched plate the bottom wall is 2.4 long: it receives p·2.4 upwards.
bas = sum(f_ext.value(noeuds[k], "f_y") for k in (0, 1, 2))
assert abs(bas - 2.4) < 1e-12
assert abs(sum(f_ext.value(n, "f_x") for n in noeuds)) < 1e-12

# ANCHOR: axes
# A conduction whose strong axis runs along x at rest. On the plate stretched
# by 1.2 then turned by 60°, the axes turn with the part — by the rotation R of
# F = R U, not by the stretch — and the conductivities stay the material's.
conduction = pc.model.heat_conduction(fes, symmetry="orthotropic")
axes = pc.element_field.material_field(
    conduction,
    [("k_1", 10.0), ("k_2", 1.0), ("k_3", 1.0), ("V1X", 1.0), ("V1Y", 0.0)],
)
F = pc.element_field.deformation(tourne, fes, "gradient")  # on the reference
tournes = pc.element_field.rotate_frame(axes, F)  # V1 = (cos 60°, sin 60°)
# ANCHOR_END: axes
assert coords.active == 0
assert abs(tournes[0].value(0, 0, "V1X") - c60) < 1e-12
assert abs(tournes[0].value(0, 0, "V1Y") - s60) < 1e-12
assert tournes[0].value(0, 0, "k_1") == 10.0
