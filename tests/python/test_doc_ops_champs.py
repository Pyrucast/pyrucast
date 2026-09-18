"""Source of the examples of `book/src/operateurs/champs.md`.

Every block of the page comes from here through `{{#include …:anchor}}`. The setup lives outside the anchors. See
`book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

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
