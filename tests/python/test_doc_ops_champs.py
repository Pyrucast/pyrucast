"""Source of the examples of `book/src/operateurs/champs.md`.

Every block of the page comes from here through `{{#include …:anchor}}`. The setup lives outside the anchors. See
`book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

import math

import pyrucast


def _plaque():
    """A plate of two QUA4, its FE space, and its nodes."""
    c = pyrucast.Coords(2)
    n = [
        c.add_node([x, y]) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)
    ]  # 0,1,2 en bas ; 3,4,5 en haut
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell([n[0], n[1], n[4], n[3]])
    mesh.unit().add_cell([n[1], n[2], n[5], n[4]])
    return c, mesh, pyrucast.FiniteElementSpace(mesh), n


def _champ_nodal(mesh, noeuds, composantes, valeurs, support=None):
    """A `NodeField` on the POI1 cloud of the given nodes.

    `support` is handed over as is from one field to the next: two calls to
    `poi1_from_nodes` build two **distinct** supports, and the reductions that
    pair node to node (`xty`) would then return zero.
    """
    support = support or pyrucast.mesh.poi1_from_nodes(noeuds)
    f = pyrucast.NodeField(support, composantes)
    for noeud, ligne in zip(noeuds, valeurs):
        for composante, v in zip(composantes, ligne):
            f[0].set_value(noeud, composante, v)
    return f


# ── Sélection : d'un champ vers un maillage ─────────────────────────────────


# ── select ─────────────────────────────────────────────────

_, mesh, fes, n = _plaque()
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
sigma = pyrucast.ElementField(fes, ["vm"])
sigma[0].set_uniform("vm", 300e6)
# ANCHOR: select
# Nodes whose temperature lies between 20 and 80 °C (inclusive bounds).
chauds = pyrucast.mesh.select(temperature, ge=20.0, le=80.0)

# Cellules dont la contrainte de von Mises dépasse un seuil (borne basse seule).
critiques = pyrucast.mesh.select(sigma, ge=250e6, components=["vm"])
# ANCHOR_END: select
assert chauds.cell_count() == 3  # 50, 30, 70
assert critiques.cell_count() == 2

# ── Mask: same structure, values rewritten ──────────────────────────────────


# ── mask ───────────────────────────────────────────────────

_, mesh, _, n = _plaque()
champ = _champ_nodal(mesh, n, ["v"], [[v] for v in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)])
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
# ANCHOR: mask
# Resets a field's negative values to zero, component by component.
positif = champ * champ.mask(ge=0.0)

# Sugar: comparisons build a mask directly.
positif = champ * (champ >= 0.0)  # the same thing
chauds = temperature > 80.0  # NodeField 0/1
# ANCHOR_END: mask
assert positif[0][n[0], "v"] == 0.0
assert positif[0][n[5], "v"] == 3.0
assert chauds[0][n[2], "T"] == 1.0

# ── Composantes : filtrer, renommer ─────────────────────────────────────────


# ── filter et rename ───────────────────────────────────────

_, mesh, fes, n = _plaque()
solution = _champ_nodal(mesh, n, ["u_x", "u_y"], [[0.1, 0.2]] * 6)
model = pyrucast.model.elasticity(fes, "plane_stress")
# ANCHOR: filter_rename
# Removes the Lagrange multipliers from a solve result.
u = solution.filter_components(model.primal_vars())

# Renames a component before exporting.
export = u.rename_component("u_x", "DX")
# ANCHOR_END: filter_rename
assert u.components() == ["u_x", "u_y"]
assert export.components() == ["DX", "u_y"]

# ── indexation ─────────────────────────────────────────────

_, mesh, _, n = _plaque()
champ = _champ_nodal(mesh, n, ["u_x", "u_y"], [[0.1, 0.2]] * 6)
node = n[0]
# ANCHOR: indexation
ux = champ["u_x"]  # == filter_components(champ, "u_x")
depl = champ[["u_x", "u_y"]]  # == filter_components(champ, ["u_x", "u_y"])
zone = champ[0]  # inchangé : la zone (SubNodeField)
val = champ[0][node, "u_x"]  # unchanged: the value at the node
# ANCHOR_END: indexation
assert ux.components() == ["u_x"]
assert depl.components() == ["u_x", "u_y"]
assert zone.node_count() == 6
assert val == 0.1

# ── alignement de composantes ──────────────────────────────

_, mesh, _, n = _plaque()
u1 = _champ_nodal(mesh, n, ["u_x", "u_y", "u_z"], [[0.1, 0.2, 0.3]] * 6)
u2 = _champ_nodal(mesh, n, ["u_x", "u_y"], [[1.0, 1.0]] * 6)
# ANCHOR: alignement
u = u1[u2.components()]  # u1 cut down to u2's set of components
# ANCHOR_END: alignement
assert u.components() == ["u_x", "u_y"]

# ── Element-wise mathematics ────────────────────────────────────────────────


# ── maths ──────────────────────────────────────────────────

_, mesh, _, n = _plaque()
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
signal = _champ_nodal(mesh, n, ["s"], [[-1.0], [2.0], [-3.0], [4.0], [-5.0], [6.0]])
# ANCHOR: maths
# Atténuation exponentielle d'un champ de température.
attenue = pyrucast.field.exp(temperature * -0.1)

# Magnitude of a field (combined with scalar field arithmetic).
amplitude = pyrucast.field.abs(signal)
# ANCHOR_END: maths
assert attenue[0][n[0], "T"] > 0.0
assert amplitude[0][n[0], "s"] == 1.0

# ── Reductions to a number ──────────────────────────────────────────────────


# ── xty ────────────────────────────────────────────────────

_, mesh, _, n = _plaque()
support = pyrucast.mesh.poi1_from_nodes(n)
forces = _champ_nodal(mesh, n, ["f_x", "f_y"], [[1.0, 2.0]] * 6, support)
deplacements = _champ_nodal(mesh, n, ["f_x", "f_y"], [[0.5, 0.5]] * 6, support)
# ANCHOR: xty
# External strain energy: work of the nodal forces in the displacement field
# (same components, same mesh).
energie = pyrucast.measure.xty(forces, deplacements)
# ANCHOR_END: xty
assert energie == 6 * (1.0 * 0.5 + 2.0 * 0.5)

# ── psca ───────────────────────────────────────────────────

_, mesh, _, n = _plaque()
vitesse = _champ_nodal(mesh, n, ["v_x", "v_y"], [[3.0, 4.0]] * 6)
# ANCHOR: psca
# Squared norm of a vector field, node by node.
norme2 = pyrucast.field.psca(vitesse, vitesse)  # one-component field, "psca"
# ANCHOR_END: psca
assert norme2[0][n[0], "psca"] == 25.0

# ── integral ───────────────────────────────────────────────

_, mesh, fes, n = _plaque()
densite = _champ_nodal(mesh, n, ["f_y"], [[1.0]] * 6)
champ_unite = _champ_nodal(mesh, n, ["u"], [[1.0]] * 6)
# ANCHOR: integral
# Resultant of a surface force density f_y on a plate (through N_i).
r_y = pyrucast.measure.integral(densite, "f_y", fespace=fes)
# Measure of the domain: ∫ 1 dΩ.
aire = pyrucast.measure.integral(champ_unite, "u", fespace=fes)
# ANCHOR_END: integral
assert abs(aire - 2.0) < 1e-12
assert abs(r_y - 2.0) < 1e-12

# ── sommes et normes ───────────────────────────────────────

_, mesh, _, n = _plaque()
forces = _champ_nodal(mesh, n, ["f_x", "f_y"], [[1.0, 2.0]] * 6)
residu = _champ_nodal(mesh, n, ["f_x", "f_y"], [[3.0, 4.0]] * 6)
# ANCHOR: sommes
# Resultant of a nodal force field, component by component.
rx = forces.sum("f_x")
ry = forces.sum("f_y")
# Squared norm of the residual, for a convergence test.
r2 = pyrucast.measure.xtx(residu)
# The same norm, restricted to the translation components alone.
r2_uy = pyrucast.measure.xtx(residu, components=["f_y"])
# Extrema of a named component…
fy_max = forces.max("f_y")
# …or, without an argument, of the whole field, components pooled.
partout = forces.min()
# ANCHOR_END: sommes
assert rx == 6.0 and ry == 12.0
assert r2 == 6 * 25.0
assert r2_uy == 6 * 16.0
assert fy_max == 2.0
assert partout == 1.0

# ── mesures de déformation ─────────────────────────────────

_, mesh, fes, n = _plaque()
# Uniform stretch λ = 1.1 along x: u_x = 0.1·x.
u = _champ_nodal(mesh, n, ["u_x", "u_y"], [[0.1 * p.position()[0], 0.0] for p in n])
# ANCHOR: mesures
# Linearized strain, the default: eps_xx, eps_xy, eps_yy.
eps = pyrucast.element_field.deformation(u, fes)
# Finite measures share the call; each has its own prefix.
E = pyrucast.element_field.deformation(u, fes, "green_lagrange")  # E_xx…
F = pyrucast.element_field.deformation(u, fes, "F")  # F_xx, F_xy, F_yx, F_yy
H = pyrucast.element_field.deformation(u, fes, "hencky")  # H_xx…
# ANCHOR_END: mesures
assert abs(eps[0].value(0, 0, "eps_xx") - 0.1) < 1e-12
assert abs(E[0].value(0, 0, "E_xx") - 0.105) < 1e-12
assert F[0].components() == ["F_xx", "F_xy", "F_yx", "F_yy"]
assert abs(F[0].value(0, 0, "F_xx") - 1.1) < 1e-12
assert abs(H[0].value(0, 0, "H_xx") - math.log(1.1)) < 1e-12

# ── opérateur élastique ────────────────────────────────────

_, mesh, fes, n = _plaque()
plasticite = pyrucast.model.plasticity_perfect(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    plasticite, [("E", 200e9), ("nu", 0.0), ("sigma_y", 1e6)]
)
dilatation = pyrucast.ElementField(fes, ["eps_xx", "eps_xy", "eps_yy"])
dilatation[0].set_uniform("eps_xx", 1e-3)
# ANCHOR: elastique
# The stress an imposed strain would cause if nothing yielded: D:ε.
sigma = pyrucast.element_field.elastic_stress(plasticite, dilatation, materiaux)
# And back: the elastic strain carrying a stress, D⁻¹:σ.
retour = pyrucast.element_field.elastic_strain(plasticite, sigma, materiaux)
# ANCHOR_END: elastique
assert abs(sigma[0].value(0, 0, "sigma_xx") - 2e8) < 1e-3  # far beyond sigma_y
assert abs(retour[0].value(0, 0, "eps_xx") - 1e-3) < 1e-15

# ── intégrale par maille, somme aux nœuds ──────────────────

_, mesh, fes, n = _plaque()
un = pyrucast.ElementField(fes, ["un"])
un[0].set_uniform("un", 1.0)
# ANCHOR: par_maille
# One point per cell, at its centroid: where per-cell values live.
centres = pyrucast.FiniteElementSpace(mesh, quadrature="REDUCED")
aires = pyrucast.element_field.cell_integral(un, centres)  # ∫ 1 = the cell's area
# Each node receives the sum over the cells around it.
volumes = pyrucast.node_field.sum_to_nodes(aires)
# ANCHOR_END: par_maille
assert abs(aires[0].value(1, 0, "un") - 1.0) < 1e-12
assert abs(volumes.value(n[1], "un") - 2.0) < 1e-12  # shared by both cells
assert abs(volumes.value(n[0], "un") - 1.0) < 1e-12

# ── B du modèle : l'entrée de chaque loi ───────────────────

_, mesh, fes, n = _plaque()
solide = pyrucast.model.elasticity(fes, "plane_strain")
couple = solide | pyrucast.model.heat_conduction(fes)
materiaux = pyrucast.element_field.material_field_per_sub_model(
    couple, [[("E", 200e9), ("nu", 0.3)], [("k", 45.0)]]
)
# u_x = 1e-3·x, u_y = 2e-3·y·x, T = 10·x + 5·y on the six nodes.
solution = _champ_nodal(
    mesh,
    n,
    ["u_x", "u_y", "T"],
    [
        [1e-3 * x, 2e-3 * y * x, 10 * x + 5 * y]
        for y in (0.0, 1.0)
        for x in (0.0, 1.0, 2.0)
    ],
)
# ANCHOR: strain_modele
# B: each law gets its own input from one solution — a strain for the solid,
# a temperature gradient for conduction.
eps = couple.strain(solution, materiaux)
etat = couple.integrate_behavior(eps, materiaux)  # the laws
f_int = couple.internal_forces(etat, solution, materiaux)  # Bᵀ: back to the nodes
# ANCHOR_END: strain_modele
assert [z.components() for z in eps] == [
    ["eps_xx", "eps_xy", "eps_yy"],
    ["grad_T_x", "grad_T_y"],
]
assert abs(eps[1].value(0, 0, "grad_T_x") - 10.0) < 1e-12
# Linear laws: Bᵀ D B u is the stiffness applied to u, to rounding.
ku = pyrucast.matrix.stiffness(couple, materiaux) * solution
for noeud in n:
    for dual in ("f_x", "f_y", "q"):
        attendu = ku.value(noeud, dual)
        assert abs(f_int.value(noeud, dual) - attendu) <= 1e-9 * (1.0 + abs(attendu))
