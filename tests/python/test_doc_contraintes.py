"""Source des exemples Python des pages `book/src/contraintes*.md`.

Voir `book/src/developper/documentation-et-tests.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
"""

import pyrucast


def _barre_elastique(n=2, dim=2):
    """Une barre SEG2. En 1-D elle n'a qu'un DDL par nœud, ce qu'exige un
    système unilatéral : en 2-D, `u_y` resterait libre et la matrice
    singulière dès que la butée se relâche."""
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


# ── Sens d'une contrainte : la butée unilatérale ────────────────────────────


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
# Barre encastrée à gauche (égalité) et butée à droite (u_x ≥ 0) : il faut
# les deux, un modèle qui ne tient que par sa butée ne converge pas.
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
# `constraint_rhs` s'appelle sur **une** contrainte, pas sur le modèle
# complet : chacune apporte sa part du chargement.
rhs = encastrement.constraint_rhs([(noeuds[0], 0.0)]) | butee.constraint_rhs(
    [(noeuds[-1], 0.0)]
)
# ANCHOR: solve_unilateral
solution = pyrucast.solver.solve_unilateral(
    k, model, rhs
)  # method, cache, max_iter, tol
# ANCHOR_END: solve_unilateral
assert solution.node_count() > 0

# ── Chargement d'une contrainte ─────────────────────────────────────────────


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
    """Deux blocs QUA4 empilés, l'un au-dessus de l'autre avec un jeu."""
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
# Bloquer u_x partout et u_y sous le bloc bas : sans ces appuis le système
# est libre en translation et l'ensemble actif se met à cycler.
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
# Maître : bord supérieur du bloc bas, parcouru en −x (normale +y, vers l'esclave).
master = pyrucast.Mesh(c, "SEG2")
for i in reversed(range(N)):
    master.unit().add_cell([bottom[idx(i + 1, N)], bottom[idx(i, N)]])
# Esclave : nœuds du bord inférieur du bloc haut.
slave = pyrucast.mesh.poi1_from_nodes([top[idx(i, 0)] for i in range(N + 1)])

contact = pyrucast.model.contact(elasticite, slave, master, ["u_x", "u_y"])
# La pression du bord supérieur est un terme du modèle, comme le contact.
charge = pyrucast.model.flux(edge_fes, "f_y", "mechanical")
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


# ── Le second membre géométrique du contact ─────────────────────────────────

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

# 2) Supports de multiplicateurs : barycenter colocalise des nœuds neufs.
imposed_left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
imposed_right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
mult_mesh_left = pyrucast.mesh.barycenter(imposed_left)
mult_mesh_right = pyrucast.mesh.barycenter(imposed_right)
conduction = pyrucast.model.heat_conduction(fes)
left = pyrucast.model.dirichlet(conduction, "T", imposed_left, mult_mesh_left)
right = pyrucast.model.dirichlet(conduction, "T", imposed_right, mult_mesh_right)
mult_left = mult_mesh_left.node(0, 0, 0)
mult_right = mult_mesh_right.node(0, 0, 0)

# 3) Modèle complet : conduction + les deux Dirichlet.
model = conduction | left | right
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

# 4) Chargement : le helper `constraint_rhs` désigne chaque contrainte par son
#    nœud contraint et écrit u_d au slot imposed_T du nœud-multiplicateur. On
#    fusionne les deux avec `|`.
rhs = left.constraint_rhs([(nodes[0], 0.0)]) | right.constraint_rhs([(nodes[-1], 1.0)])

# 5) Assemblage + résolution.
K = pyrucast.matrix.stiffness(model, materials)
solution = pyrucast.solver.solve(K, rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10  # T au milieu
assert abs(solution.value(mult_left, "lambda_T") - 1.0) < 1e-10  # flux à gauche
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
# helper `constraint_rhs` désigne chaque relation par un nœud (nœud contraint
# pour Dirichlet, nœud-terme pour la MPC) et retrouve seul le nœud-multiplicateur
# et la composante (`imposed_T`, `mpc_rhs`). On fusionne les deux avec `|`.
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

# Coins fixés au champ linéaire (Dirichlet).
corner_mesh = pyrucast.mesh.poi1_from_nodes(corner_nodes)
corner_mult = pyrucast.mesh.barycenter(corner_mesh)
dirichlet = pyrucast.model.dirichlet(base, "T", corner_mesh, corner_mult)

# Nœud immergé, lié à l'hôte.
p = c.add_node([0.3, 0.6, 0.2])
bar = pyrucast.mesh.poi1_from_nodes([p])
embedded = pyrucast.model.embedded(base, bar, host, ["T"])
emb_mult = embedded.multiplier_mesh().node(0, 0, 0)

model = base | dirichlet | embedded
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

# Chargement : valeur du champ à chaque coin, g = 0 (tie) au nœud immergé.
rhs = dirichlet.constraint_rhs([(n, field(x)) for n, x in zip(corner_nodes, corners)])
rhs = rhs | embedded.constraint_rhs([(p, 0.0)])

solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
assert abs(solution.value(p, "T") - field([0.3, 0.6, 0.2])) < 1e-9  # 4.2
# ANCHOR_END: embedded_complet
