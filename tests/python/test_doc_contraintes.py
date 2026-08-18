"""Source des exemples Python des pages `book/src/contraintes*.md`.

Voir `book/src/developper/documentation-et-tests.md`.
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


def test_butee():
    _, _, fes, noeuds = _barre_elastique()
    imposed, mult = _support_et_multiplicateur(noeuds[-1])
    # ANCHOR: butee
    butee = pyrucast.Model.dirichlet("u_y", "f_y", imposed, mult, sense=">=")
    # ANCHOR_END: butee
    assert len(butee) == 1


def test_solve_unilateral():
    _, _, fes, noeuds = _barre_elastique(dim=1)
    # Barre encastrée à gauche (égalité) et butée à droite (u_x ≥ 0) : il faut
    # les deux, un modèle qui ne tient que par sa butée ne converge pas.
    gauche, mult_g = _support_et_multiplicateur(noeuds[0])
    droite, mult_d = _support_et_multiplicateur(noeuds[-1])
    encastrement = pyrucast.Model.dirichlet("u_x", "f_x", gauche, mult_g)
    butee = pyrucast.Model.dirichlet("u_x", "f_x", droite, mult_d, sense=">=")
    model = pyrucast.Model.truss(fes) | encastrement | butee
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


def test_constraint_rhs():
    _, _, fes, noeuds = _barre_elastique()
    imposed, mult = _support_et_multiplicateur(noeuds[0])
    dual = pyrucast.Model.heat_conduction(fes).dual_of("T")
    dirichlet = pyrucast.Model.dirichlet("T", dual, imposed, mult)
    autre = pyrucast.mesh.poi1_from_nodes([noeuds[-1]])
    mpc = pyrucast.Model.mpc(
        [(autre, "T", dual, 1.0), (imposed, "T", dual, -1.0)],
        pyrucast.mesh.barycenter(autre),
    )
    noeud_contraint, u_d = noeuds[0], 1.0
    noeud_terme, g = noeuds[-1], 0.5
    # ANCHOR: constraint_rhs
    rhs = dirichlet.constraint_rhs([(noeud_contraint, u_d)])
    rhs = mpc.constraint_rhs([(noeud_terme, g)])
    # ANCHOR_END: constraint_rhs
    assert rhs.node_count() > 0


def test_constraint_rhs_by_index():
    _, _, fes, noeuds = _barre_elastique()
    imposed, _ = _support_et_multiplicateur(noeuds[0])
    autre = pyrucast.mesh.poi1_from_nodes([noeuds[-1]])
    dual = pyrucast.Model.heat_conduction(fes).dual_of("T")
    mpc = pyrucast.Model.mpc(
        [(autre, "T", dual, 1.0), (imposed, "T", dual, -1.0)],
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


def _bloquer(noeuds, var, dual):
    imposed = pyrucast.mesh.poi1_from_nodes(noeuds)
    return pyrucast.Model.dirichlet(
        var, dual, imposed, pyrucast.mesh.barycenter(imposed)
    )


def test_contact():
    c, bas, haut, bottom, top, idx, N = _deux_blocs()
    fes = pyrucast.FiniteElementSpace(bas | haut)
    # Bloquer u_x partout et u_y sous le bloc bas : sans ces appuis le système
    # est libre en translation et l'ensemble actif se met à cycler.
    elasticite = pyrucast.Model.elasticity(fes, "plane_stress")
    appuis = _bloquer(bottom + top, "u_x", "f_x") | _bloquer(
        [bottom[idx(i, 0)] for i in range(N + 1)], "u_y", "f_y"
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

    contact = pyrucast.Model.contact(slave, master, [("u_x", "f_x"), ("u_y", "f_y")])
    model = elasticite | appuis | contact
    materials = pyrucast.element_field.material_field(
        model, [("E", 210.0), ("nu", 0.0)]
    )

    rhs = pyrucast.node_field.flux(edge_fes[0], -S, "f_y") | model.contact_gaps()
    solution = pyrucast.solver.solve_unilateral(
        pyrucast.matrix.stiffness(model, materials), model, rhs
    )
    # ANCHOR_END: contact
    assert solution.node_count() > 0
