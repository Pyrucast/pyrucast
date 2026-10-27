"""Python tests for multi-point constraints (`model.mpc`).

MPCs impose linear relations `Σ aₖ·u(nodeₖ, varₖ) = g` via Lagrange multipliers,
on the same augmented system as Dirichlet. On a 1-D heat-conduction bar
`-u'' = 0` (analytical `u(x)` linear), a well-posed set of relations recovers the
linear solution, and the single-term MPC coincides with Dirichlet.
"""

import pyrucast
import pytest

N_ELEMS = 4
H = 1.0 / N_ELEMS
TOL = 1e-10


def _heat_bar():
    """`[0, 1]` SEG2 bar, `k = 1`. Returns (coords, nodes, fes, materials)."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i * H]) for i in range(N_ELEMS + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N_ELEMS):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)
    return c, nodes, fes, materials


def test_dual_of_finds_conjugate():
    """`dual_of` maps a primal to its conjugate dual via the positional pairing."""
    _, _, fes, _ = _heat_bar()
    model = pyrucast.model.heat_conduction(fes)
    assert model.dual_of("T") == "q"
    assert model.dual_of("nope") is None


def test_mpc_difference_relation_recovers_linear_solution():
    """`1·T(node4) − 1·T(node0) = 1` with Dirichlet `T(node0) = 0` gives u(x)=x."""
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    cible = pyrucast.model.heat_conduction(fes)
    dirichlet = pyrucast.model.dirichlet(imposed0, cible, "T")
    dir_mult = dirichlet.multiplier_mesh().node(0, 0, 0)

    base = pyrucast.model.heat_conduction(fes)
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mpc = pyrucast.model.mpc(
        [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
        base,
    )
    mpc_mult = mpc.multiplier_mesh().node(0, 0, 0)

    model = base | dirichlet | mpc

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([mpc_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed"])
    rhs[0].set_value(dir_mult, "imposed", 0.0)
    rhs[0].set_value(mpc_mult, "imposed", 1.0)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - i * H) < TOL
    assert (
        abs(solution.value(nodes[-1], "T") - solution.value(nodes[0], "T") - 1.0) < TOL
    )


def test_single_term_mpc_matches_dirichlet():
    """A single-term MPC `1·T = u_d` reproduces the equivalent Dirichlet."""

    def solve_dirichlet():
        _c, nodes, fes, materials = _heat_bar()
        left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
        cible = pyrucast.model.heat_conduction(fes)
        at_left = pyrucast.model.dirichlet(left, cible, "T")
        at_right = pyrucast.model.dirichlet(right, cible, "T")
        model = cible | at_left | at_right
        rhs = at_left.constraint_rhs([(nodes[0], 0.0)]) | at_right.constraint_rhs(
            [(nodes[-1], 1.0)]
        )
        return nodes, pyrucast.solver.solve(
            pyrucast.matrix.stiffness(model, materials), rhs
        )

    def solve_mpc():
        _c, nodes, fes, materials = _heat_bar()
        left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
        cible = pyrucast.model.heat_conduction(fes)
        at_left = pyrucast.model.dirichlet(left, cible, "T")
        relation = pyrucast.model.mpc([(right, "T", 1.0)], cible)
        model = cible | at_left | relation
        rhs = at_left.constraint_rhs([(nodes[0], 0.0)]) | relation.constraint_rhs(
            [(nodes[-1], 1.0)]
        )
        return nodes, pyrucast.solver.solve(
            pyrucast.matrix.stiffness(model, materials), rhs
        )

    nodes, dir_sol = solve_dirichlet()
    _, mpc_sol = solve_mpc()
    # Not a vacuous match: the imposed values reach the solution.
    assert abs(dir_sol.value(nodes[-1], "T") - 1.0) < TOL
    for node in nodes:
        assert abs(dir_sol.value(node, "T") - mpc_sol.value(node, "T")) < TOL


def test_constraint_rhs_helper_builds_second_member():
    """`constraint_rhs` builds the RHS from constrained / term nodes: `g` lands
    at the imposed-value component of the multiplier node, and the assembled
    solve recovers u(x)=x."""
    _, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    cible = pyrucast.model.heat_conduction(fes)
    dirichlet = pyrucast.model.dirichlet(imposed0, cible, "T")

    base = pyrucast.model.heat_conduction(fes)
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mpc = pyrucast.model.mpc(
        [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
        base,
    )

    model = base | dirichlet | mpc

    # Node keying: constrained node for Dirichlet, a term node for the MPC. The
    # multiplier node and imposed-value component are resolved by the helper.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[-1], 1.0)]
    )
    assert abs(rhs.value(mpc.multiplier_mesh().node(0, 0, 0), "imposed") - 1.0) < TOL
    assert (
        abs(rhs.value(dirichlet.multiplier_mesh().node(0, 0, 0), "imposed") - 0.0) < TOL
    )

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - i * H) < TOL


def test_constraint_rhs_by_index_matches_node_keying():
    """Keying a relation by its index gives the same field as keying by a node,
    and an out-of-range index raises."""
    _, nodes, fes, _ = _heat_bar()
    base = pyrucast.model.heat_conduction(fes)
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mpc = pyrucast.model.mpc(
        [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
        base,
    )
    mult = mpc.multiplier_mesh().node(0, 0, 0)

    by_node = mpc.constraint_rhs([(nodes[-1], 1.0)])
    by_index = mpc.constraint_rhs_by_index([(0, 1.0)])
    assert abs(by_node.value(mult, "imposed") - by_index.value(mult, "imposed")) < TOL
    assert abs(by_index.value(mult, "imposed") - 1.0) < TOL

    with pytest.raises(RuntimeError, match="relation index 1 out of range"):
        mpc.constraint_rhs_by_index([(1, 1.0)])  # only one relation


def test_constraint_rhs_rejects_bad_input():
    """The helper rejects a constraint-free model and an unconstrained node."""
    _, nodes, fes, _ = _heat_bar()
    base = pyrucast.model.heat_conduction(fes)
    with pytest.raises(RuntimeError, match="holds no constraint sub-model"):
        base.constraint_rhs([(nodes[0], 0.0)])

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    dirichlet = pyrucast.model.dirichlet(imposed0, base, "T")
    with pytest.raises(RuntimeError, match="is not constrained by this Dirichlet"):
        dirichlet.constraint_rhs([(nodes[-1], 0.0)])


def test_mpc_elimination_matches_lagrange():
    """`solve_eliminate` reproduces the Lagrange `solve` on a non-chained,
    two-term MPC: `2·T(node4) − 1·T(node2) = 1.5` with Dirichlet `T(node0) = 0`
    (disjoint slaves). Both fields coincide and the relation holds exactly."""
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    cible = pyrucast.model.heat_conduction(fes)
    dirichlet = pyrucast.model.dirichlet(imposed0, cible, "T")
    dir_mult = dirichlet.multiplier_mesh().node(0, 0, 0)

    mesh4 = pyrucast.mesh.poi1_from_nodes([nodes[4]])
    mesh2 = pyrucast.mesh.poi1_from_nodes([nodes[2]])
    mpc = pyrucast.model.mpc(
        [(mesh4, "T", 2.0), (mesh2, "T", -1.0)],
        cible,
    )
    mpc_mult = mpc.multiplier_mesh().node(0, 0, 0)

    model = pyrucast.model.heat_conduction(fes) | dirichlet | mpc

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([mpc_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed"])
    rhs[0].set_value(dir_mult, "imposed", 0.0)
    rhs[0].set_value(mpc_mult, "imposed", 1.5)

    k = pyrucast.matrix.stiffness(model, materials)
    lagrange = pyrucast.solver.solve(k, rhs)
    elim = pyrucast.solver.solve_eliminate(k, model, rhs)

    for node in nodes:
        assert abs(lagrange.value(node, "T") - elim.value(node, "T")) < TOL
    t0 = elim.value(nodes[0], "T")
    t2 = elim.value(nodes[2], "T")
    t4 = elim.value(nodes[4], "T")
    assert abs(t0) < TOL
    assert abs(2.0 * t4 - t2 - 1.5) < TOL
