"""Python tests for multi-point constraints (`Model.mpc`).

MPCs impose linear relations `Σ aₖ·u(nodeₖ, varₖ) = g` via Lagrange multipliers,
on the same augmented system as Dirichlet. On a 1-D heat-conduction bar
`-u'' = 0` (analytical `u(x)` linear), a well-posed set of relations recovers the
linear solution, and the single-term MPC coincides with Dirichlet.
"""

import pytest

import pyrucast

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
    model = pyrucast.Model.heat_conduction(fes)
    assert model.dual_of("T") == "q"
    assert model.dual_of("nope") is None


def test_mpc_difference_relation_recovers_linear_solution():
    """`1·T(node4) − 1·T(node0) = 1` with Dirichlet `T(node0) = 0` gives u(x)=x."""
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
    dir_mult = mult0.node(0, 0, 0)

    base = pyrucast.Model.heat_conduction(fes)
    dual = base.dual_of("T")  # "q"
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.mesh.barycenter(mesh_last)
    mpc = pyrucast.Model.mpc(
        [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
        mult_mpc,
    )
    mpc_mult = mult_mpc.node(0, 0, 0)

    model = base | dirichlet | mpc

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([mpc_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T", "mpc_rhs"])
    rhs[0].set_value(dir_mult, "imposed_T", 0.0)
    rhs[0].set_value(mpc_mult, "mpc_rhs", 1.0)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - i * H) < TOL
    assert (
        abs(solution.value(nodes[-1], "T") - solution.value(nodes[0], "T") - 1.0) < TOL
    )


def test_single_term_mpc_matches_dirichlet():
    """A single-term MPC `1·T = u_d` reproduces the equivalent Dirichlet."""

    def solve_dirichlet():
        c, nodes, fes, materials = _heat_bar()
        left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
        ml, mr = pyrucast.mesh.barycenter(left), pyrucast.mesh.barycenter(right)
        model = (
            pyrucast.Model.heat_conduction(fes)
            | pyrucast.Model.dirichlet("T", "q", left, ml)
            | pyrucast.Model.dirichlet("T", "q", right, mr)
        )
        rhs_mesh = pyrucast.Mesh(c, "POI1")
        rhs_mesh.unit().add_cell([ml.node(0, 0, 0)])
        rhs_mesh.unit().add_cell([mr.node(0, 0, 0)])
        rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T"])
        rhs[0].set_value(ml.node(0, 0, 0), "imposed_T", 0.0)
        rhs[0].set_value(mr.node(0, 0, 0), "imposed_T", 1.0)
        return nodes, pyrucast.solver.solve(
            pyrucast.matrix.stiffness(model, materials), rhs
        )

    def solve_mpc():
        c, nodes, fes, materials = _heat_bar()
        left = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        ml = pyrucast.mesh.barycenter(left)
        right = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
        mm = pyrucast.mesh.barycenter(right)
        model = (
            pyrucast.Model.heat_conduction(fes)
            | pyrucast.Model.dirichlet("T", "q", left, ml)
            | pyrucast.Model.mpc([(right, "T", "q", 1.0)], mm)
        )
        rhs_mesh = pyrucast.Mesh(c, "POI1")
        rhs_mesh.unit().add_cell([ml.node(0, 0, 0)])
        rhs_mesh.unit().add_cell([mm.node(0, 0, 0)])
        rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T", "mpc_rhs"])
        rhs[0].set_value(ml.node(0, 0, 0), "imposed_T", 0.0)
        rhs[0].set_value(mm.node(0, 0, 0), "mpc_rhs", 1.0)
        return nodes, pyrucast.solver.solve(
            pyrucast.matrix.stiffness(model, materials), rhs
        )

    nodes, dir_sol = solve_dirichlet()
    _, mpc_sol = solve_mpc()
    for node in nodes:
        assert abs(dir_sol.value(node, "T") - mpc_sol.value(node, "T")) < TOL


def test_constraint_rhs_helper_builds_second_member():
    """`constraint_rhs` builds the RHS from constrained / term nodes: `g` lands
    at the imposed-value component of the multiplier node, and the assembled
    solve recovers u(x)=x."""
    _, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)

    base = pyrucast.Model.heat_conduction(fes)
    dual = base.dual_of("T")
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.mesh.barycenter(mesh_last)
    mpc = pyrucast.Model.mpc(
        [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
        mult_mpc,
    )

    model = base | dirichlet | mpc

    # Node keying: constrained node for Dirichlet, a term node for the MPC. The
    # multiplier node and imposed-value component are resolved by the helper.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[-1], 1.0)]
    )
    assert abs(rhs.value(mult_mpc.node(0, 0, 0), "mpc_rhs") - 1.0) < TOL
    assert abs(rhs.value(mult0.node(0, 0, 0), "imposed_T") - 0.0) < TOL

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - i * H) < TOL


def test_constraint_rhs_by_index_matches_node_keying():
    """Keying a relation by its index gives the same field as keying by a node,
    and an out-of-range index raises."""
    _, nodes, fes, _ = _heat_bar()
    base = pyrucast.Model.heat_conduction(fes)
    dual = base.dual_of("T")
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.mesh.barycenter(mesh_last)
    mpc = pyrucast.Model.mpc(
        [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
        mult_mpc,
    )
    mult = mult_mpc.node(0, 0, 0)

    by_node = mpc.constraint_rhs([(nodes[-1], 1.0)])
    by_index = mpc.constraint_rhs_by_index([(0, 1.0)])
    assert abs(by_node.value(mult, "mpc_rhs") - by_index.value(mult, "mpc_rhs")) < TOL
    assert abs(by_index.value(mult, "mpc_rhs") - 1.0) < TOL

    with pytest.raises(Exception):
        mpc.constraint_rhs_by_index([(1, 1.0)])  # only one relation


def test_constraint_rhs_rejects_bad_input():
    """The helper rejects a constraint-free model and an unconstrained node."""
    _, nodes, fes, _ = _heat_bar()
    base = pyrucast.Model.heat_conduction(fes)
    with pytest.raises(Exception):
        base.constraint_rhs([(nodes[0], 0.0)])

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
    with pytest.raises(Exception):
        dirichlet.constraint_rhs([(nodes[-1], 0.0)])


def test_mpc_elimination_matches_lagrange():
    """`solve_eliminate` reproduces the Lagrange `solve` on a non-chained,
    two-term MPC: `2·T(node4) − 1·T(node2) = 1.5` with Dirichlet `T(node0) = 0`
    (disjoint slaves). Both fields coincide and the relation holds exactly."""
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
    dir_mult = mult0.node(0, 0, 0)

    mesh4 = pyrucast.mesh.poi1_from_nodes([nodes[4]])
    mesh2 = pyrucast.mesh.poi1_from_nodes([nodes[2]])
    mult_mpc = pyrucast.mesh.barycenter(mesh4)
    mpc = pyrucast.Model.mpc(
        [(mesh4, "T", "q", 2.0), (mesh2, "T", "q", -1.0)],
        mult_mpc,
    )
    mpc_mult = mult_mpc.node(0, 0, 0)

    model = pyrucast.Model.heat_conduction(fes) | dirichlet | mpc

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([mpc_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T", "mpc_rhs"])
    rhs[0].set_value(dir_mult, "imposed_T", 0.0)
    rhs[0].set_value(mpc_mult, "mpc_rhs", 1.5)

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
