"""Source of the Python examples of the `book/src/contraintes*.md` pages.

Voir `book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

import pyrucast


def _barre_elastique(n=2, dim=2):
    """A SEG2 bar. In 1-D it has one DOF per node, which is what a
    système unilatéral : en 2-D, `u_y` resterait libre et la matrice
    singular as soon as the stop releases."""
    c = pyrucast.Coords(dim)
    noeuds = [c.add_node([i / n] + [0.0] * (dim - 1)) for i in range(n + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for a, b in zip(noeuds, noeuds[1:]):
        mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    return c, mesh, fes, noeuds


def _support_et_multiplicateur(noeud):
    imposed = pyrucast.mesh.poi1_from_nodes([noeud])
    mult = pyrucast.mesh.barycenter(imposed)
    return imposed, mult


# ── A constraint's sense: the unilateral stop ───────────────────────────────


# ── butee ──────────────────────────────────────────────────

_, _, fes, noeuds = _barre_elastique()
imposed, mult = _support_et_multiplicateur(noeuds[-1])
# ANCHOR: butee
barre = pyrucast.model.truss(fes)
butee = pyrucast.model.dirichlet(barre, "u_x", imposed, mult, sense=">=")
# ANCHOR_END: butee
assert len(butee) == 1

# ── solve unilateral ───────────────────────────────────────

_, _, fes, noeuds = _barre_elastique(dim=1)
# Bar clamped on the left (equality) and stopped on the right (u_x ≥ 0): both
# are needed, a model held by its stop alone does not converge.
gauche, mult_g = _support_et_multiplicateur(noeuds[0])
droite, mult_d = _support_et_multiplicateur(noeuds[-1])
barre = pyrucast.model.truss(fes)
encastrement = pyrucast.model.dirichlet(barre, "u_x", gauche, mult_g)
butee = pyrucast.model.dirichlet(barre, "u_x", droite, mult_d, sense=">=")
model = barre | encastrement | butee
materials = pyrucast.element_field.material_field(
    model, [("E", 210_000.0), ("A", 1e-2)]
)
k = pyrucast.matrix.stiffness(model, materials)
# `constraint_rhs` is called on **one** constraint, not on the whole model:
# each brings its share of the loading.
rhs = encastrement.constraint_rhs([(noeuds[0], 0.0)]) | butee.constraint_rhs(
    [(noeuds[-1], 0.0)]
)
# ANCHOR: solve_unilateral
solution = pyrucast.solver.solve_unilateral(
    k, model, rhs
)  # method, cache, max_iter, tol
# ANCHOR_END: solve_unilateral
assert solution.node_count() > 0

# ── Loading of a constraint ─────────────────────────────────────────────────


# ── constraint rhs ─────────────────────────────────────────

_, _, fes, noeuds = _barre_elastique()
imposed, mult = _support_et_multiplicateur(noeuds[0])
conduction = pyrucast.model.heat_conduction(fes)
dual = conduction.dual_of("T")
dirichlet = pyrucast.model.dirichlet(conduction, "T", imposed, mult)
autre = pyrucast.mesh.poi1_from_nodes([noeuds[-1]])
mpc = pyrucast.model.mpc(
    conduction,
    [(autre, "T", 1.0), (imposed, "T", -1.0)],
    pyrucast.mesh.barycenter(autre),
)
noeud_contraint, u_d = noeuds[0], 1.0
noeud_terme, g = noeuds[-1], 0.5
# ANCHOR: constraint_rhs
rhs = dirichlet.constraint_rhs([(noeud_contraint, u_d)])
rhs = mpc.constraint_rhs([(noeud_terme, g)])
# ANCHOR_END: constraint_rhs
assert rhs.node_count() > 0

# ── constraint rhs by index ────────────────────────────────

_, _, fes, noeuds = _barre_elastique()
imposed, _ = _support_et_multiplicateur(noeuds[0])
autre = pyrucast.mesh.poi1_from_nodes([noeuds[-1]])
dual = pyrucast.model.heat_conduction(fes).dual_of("T")
mpc = pyrucast.model.mpc(
    conduction,
    [(autre, "T", 1.0), (imposed, "T", -1.0)],
    pyrucast.mesh.barycenter(autre),
)
index_relation, g = 0, 1.0
# ANCHOR: constraint_rhs_by_index
rhs = mpc.constraint_rhs_by_index([(index_relation, g)])
# ANCHOR_END: constraint_rhs_by_index
assert rhs.node_count() > 0

# ── Contact nœud-surface ────────────────────────────────────────────────────


def _deux_blocs(N=2):
    """Two QUA4 blocks stacked, one above the other with a gap."""
    c = pyrucast.Coords(2)

    def idx(i, j):
        return i + j * (N + 1)

    bottom = [c.add_node([i / N, j / N]) for j in range(N + 1) for i in range(N + 1)]
    top = [c.add_node([i / N, 1.0 + j / N]) for j in range(N + 1) for i in range(N + 1)]
    bas, haut = pyrucast.Mesh(c, "QUA4"), pyrucast.Mesh(c, "QUA4")
    for j in range(N):
        for i in range(N):
            bas.unit().add_cell(
                [
                    bottom[idx(i, j)],
                    bottom[idx(i + 1, j)],
                    bottom[idx(i + 1, j + 1)],
                    bottom[idx(i, j + 1)],
                ]
            )
            haut.unit().add_cell(
                [
                    top[idx(i, j)],
                    top[idx(i + 1, j)],
                    top[idx(i + 1, j + 1)],
                    top[idx(i, j + 1)],
                ]
            )
    return c, bas, haut, bottom, top, idx, N


def _bloquer(target, noeuds, var):
    imposed = pyrucast.mesh.poi1_from_nodes(noeuds)
    return pyrucast.model.dirichlet(
        target, var, imposed, pyrucast.mesh.barycenter(imposed)
    )


# ── contact ────────────────────────────────────────────────

c, bas, haut, bottom, top, idx, N = _deux_blocs()
fes = pyrucast.FiniteElementSpace(bas | haut)
# Block u_x everywhere and u_y under the lower block: without those supports
# the system is free in translation and the active set starts to cycle.
elasticite = pyrucast.model.elasticity(fes, "plane_stress")
appuis = _bloquer(elasticite, bottom + top, "u_x") | _bloquer(
    elasticite, [bottom[idx(i, 0)] for i in range(N + 1)], "u_y"
)
edge = pyrucast.Mesh(c, "SEG2")
for i in range(N):
    edge.unit().add_cell([top[idx(i, N)], top[idx(i + 1, N)]])
edge_fes = pyrucast.FiniteElementSpace(edge)
S = 1.0
# ANCHOR: contact
# Master: upper edge of the lower block, walked in −x (normal +y, towards the slave).
master = pyrucast.Mesh(c, "SEG2")
for i in reversed(range(N)):
    master.unit().add_cell([bottom[idx(i + 1, N)], bottom[idx(i, N)]])
# Slave: nodes of the upper block's lower edge.
slave = pyrucast.mesh.poi1_from_nodes([top[idx(i, 0)] for i in range(N + 1)])

contact = pyrucast.model.contact(elasticite, slave, master, ["u_x", "u_y"])
# The upper edge's pressure is a term of the model, like the contact.
charge = pyrucast.model.flux(edge_fes, elasticite, "f_y")
model = elasticite | appuis | contact | charge
materials = pyrucast.element_field.material_field(
    model, [("E", 210.0), ("nu", 0.0), ("phi_f_y", -S)]
)

rhs = pyrucast.node_field.external_forces(model, materials) | model.contact_gaps()
solution = pyrucast.solver.solve_unilateral(
    pyrucast.matrix.stiffness(model, materials), model, rhs
)
# ANCHOR_END: contact
assert solution.node_count() > 0


# ── The contact's geometric right-hand side ─────────────────────────────────

traction = pyrucast.node_field.external_forces(model, materials)

# ANCHOR: contact_gaps
rhs = traction | model.contact_gaps()
# ANCHOR_END: contact_gaps

assert rhs.node_count() > 0


# ── Dirichlet : l'exemple complet ───────────────────────────────────────────

# ANCHOR: dirichlet_complet
import pyrucast

# 1) Maillage + FE space
c = pyrucast.Coords(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.unit().add_cell([nodes[i], nodes[i + 1]])
fes = pyrucast.FiniteElementSpace(mesh)

# 2) Multiplier supports: barycenter co-locates fresh nodes.
imposed_left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
imposed_right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
mult_mesh_left = pyrucast.mesh.barycenter(imposed_left)
mult_mesh_right = pyrucast.mesh.barycenter(imposed_right)
conduction = pyrucast.model.heat_conduction(fes)
left = pyrucast.model.dirichlet(conduction, "T", imposed_left, mult_mesh_left)
right = pyrucast.model.dirichlet(conduction, "T", imposed_right, mult_mesh_right)
mult_left = mult_mesh_left.node(0, 0, 0)
mult_right = mult_mesh_right.node(0, 0, 0)

# 3) Whole model: conduction + both Dirichlet.
model = conduction | left | right
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

# 4) Loading: the `constraint_rhs` helper designates each constraint by its
#    constrained node and writes u_d at the multiplier node's imposed_T slot.
#    Both are merged with `|`.
rhs = left.constraint_rhs([(nodes[0], 0.0)]) | right.constraint_rhs([(nodes[-1], 1.0)])

# 5) Assemblage + résolution.
K = pyrucast.matrix.stiffness(model, materials)
solution = pyrucast.solver.solve(K, rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10  # T in the middle
assert abs(solution.value(mult_left, "lambda_T") - 1.0) < 1e-10  # flux on the left
# ANCHOR_END: dirichlet_complet


# ── MPC : l'exemple complet ─────────────────────────────────────────────────

# ANCHOR: mpc_complet
import pyrucast

c = pyrucast.Coords(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.unit().add_cell([nodes[i], nodes[i + 1]])
fes = pyrucast.FiniteElementSpace(mesh)

base = pyrucast.model.heat_conduction(fes)
dual = base.dual_of("T")  # "q"

# Dirichlet T(0) = 0.
imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
mult0 = pyrucast.mesh.barycenter(imposed0)
dirichlet = pyrucast.model.dirichlet(base, "T", imposed0, mult0)

# MPC 1·T(dernier) − 1·T(0) = 1.
mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
mult_mpc = pyrucast.mesh.barycenter(mesh_last)
mpc = pyrucast.model.mpc(
    conduction,
    [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
    mult_mpc,
)

model = base | dirichlet | mpc
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

# Chargement : valeur imposée de Dirichlet + second membre g de la MPC. Le
# `constraint_rhs` helper designates each relation by a node (the constrained
# node for Dirichlet, the term node for the MPC) and finds the multiplier node
# and component on its own (`imposed_T`, `mpc_rhs`). Both are merged with `|`.
rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
    [(nodes[-1], 1.0)]
)

solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10
# ANCHOR_END: mpc_complet


# ── Baignage : l'exemple complet ────────────────────────────────────────────

# ANCHOR: embedded_complet
import pyrucast

corners = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]
field = lambda c: 1.0 + 2.0 * c[0] + 3.0 * c[1] + 4.0 * c[2]

c = pyrucast.Coords(dim=3)
corner_nodes = [c.add_node(x) for x in corners]

host = pyrucast.Mesh(c, "HEX8")
host.unit().add_cell(corner_nodes)
fes = pyrucast.FiniteElementSpace(host)
base = pyrucast.model.heat_conduction(fes)

# Corners fixed to the linear field (Dirichlet).
corner_mesh = pyrucast.mesh.poi1_from_nodes(corner_nodes)
corner_mult = pyrucast.mesh.barycenter(corner_mesh)
dirichlet = pyrucast.model.dirichlet(base, "T", corner_mesh, corner_mult)

# Immersed node, tied to the host.
p = c.add_node([0.3, 0.6, 0.2])
bar = pyrucast.mesh.poi1_from_nodes([p])
embedded = pyrucast.model.embedded(base, bar, host, ["T"])
emb_mult = embedded.multiplier_mesh().node(0, 0, 0)

model = base | dirichlet | embedded
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

# Loading: the field's value at each corner, g = 0 (tie) at the immersed node.
rhs = dirichlet.constraint_rhs([(n, field(x)) for n, x in zip(corner_nodes, corners)])
rhs = rhs | embedded.constraint_rhs([(p, 0.0)])

solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
assert abs(solution.value(p, "T") - field([0.3, 0.6, 0.2])) < 1e-9  # 4.2
# ANCHOR_END: embedded_complet
