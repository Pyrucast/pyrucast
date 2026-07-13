"""Python tests for unilateral (inequality) constraints (`sense=">="` / `"<="`)
and the active-set solver `solve_unilateral`.

A 1-D heat-conduction bar `-T'' = 0` on `[0, 1]`, `T(0) = 0` and a flux load
`q` at the right end: the unconstrained solution is `T(x) = q·x`. A unilateral
bound `T(1) ⋈ a` either releases (feasible, `λ = 0`) or holds active
(`T(1) = a`, `λ = q − a` — `≤ 0` for `≥`, `≥ 0` for `≤`).
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


def _bounded_bar(q, bound, sense):
    """`T(0) = 0`, flux `q` at the right end, `T(1) <sense> bound`.

    Returns (nodes, model, materials, rhs, uni_mult).
    """
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesher.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesher.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
    dir_mult = mult0.node(0, 0, 0)

    imposed1 = pyrucast.mesher.poi1_from_nodes([nodes[-1]])
    mult1 = pyrucast.mesher.barycenter(imposed1)
    unilateral = pyrucast.Model.dirichlet("T", "q", imposed1, mult1, sense=sense)
    uni_mult = mult1.node(0, 0, 0)

    model = pyrucast.Model.heat_conduction(fes) | dirichlet | unilateral

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([nodes[-1]])
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([uni_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["q", "imposed_T"])
    rhs[0].set_value(nodes[-1], "q", q)
    rhs[0].set_value(dir_mult, "imposed_T", 0.0)
    rhs[0].set_value(uni_mult, "imposed_T", bound)

    return nodes, model, materials, rhs, uni_mult


def _check(nodes, solution, uni_mult, slope, lam):
    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - slope * i * H) < TOL
    assert abs(solution.value(uni_mult, "lambda_T") - lam) < TOL


@pytest.mark.parametrize(
    "q,bound,sense,slope,lam",
    [
        (5.0, 2.0, ">=", 5.0, 0.0),  # feasible: released, λ = 0
        (1.0, 2.0, ">=", 2.0, -1.0),  # violated: held, λ = q − a ≤ 0
        (5.0, 2.0, "<=", 2.0, 3.0),  # violated: held, λ = q − a ≥ 0
        (1.0, 2.0, "<=", 1.0, 0.0),  # feasible: released, λ = 0
    ],
)
def test_unilateral_bound_statuses(q, bound, sense, slope, lam):
    """The four sense × status cases of a bound `T(1) ⋈ a` under a flux `q`."""
    nodes, model, materials, rhs, uni_mult = _bounded_bar(q, bound, sense)
    k = pyrucast.assemble.stiffness(model, materials)
    solution = pyrucast.solver.solve_unilateral(model, k, rhs)
    _check(nodes, solution, uni_mult, slope, lam)


def test_warm_start_survives_a_status_flip():
    """Re-solving on the same matrix warm-starts; a load flip re-iterates."""
    nodes, model, materials, rhs, uni_mult = _bounded_bar(1.0, 2.0, ">=")
    k = pyrucast.assemble.stiffness(model, materials)
    first = pyrucast.solver.solve_unilateral(model, k, rhs)
    _check(nodes, first, uni_mult, 2.0, -1.0)
    # Identical re-solve: pure warm start (cached status + factorization).
    again = pyrucast.solver.solve_unilateral(model, k, rhs)
    _check(nodes, again, uni_mult, 2.0, -1.0)
    # Stronger push on the same matrix: the relation must release from the
    # warm-started (active) status.
    rhs[0].set_value(nodes[-1], "q", 5.0)
    released = pyrucast.solver.solve_unilateral(model, k, rhs)
    _check(nodes, released, uni_mult, 5.0, 0.0)


def test_unilateral_mpc_difference_relation():
    """`T(1) − T(0) ≥ g`: active for `g = 1` (T = x), released for `g = −1`."""
    for g, slope in [(1.0, 1.0), (-1.0, 0.0)]:
        c, nodes, fes, materials = _heat_bar()

        imposed0 = pyrucast.mesher.poi1_from_nodes([nodes[0]])
        mult0 = pyrucast.mesher.barycenter(imposed0)
        dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
        dir_mult = mult0.node(0, 0, 0)

        base = pyrucast.Model.heat_conduction(fes)
        dual = base.dual_of("T")
        mesh_last = pyrucast.mesher.poi1_from_nodes([nodes[-1]])
        mesh_first = pyrucast.mesher.poi1_from_nodes([nodes[0]])
        mult_mpc = pyrucast.mesher.barycenter(mesh_last)
        mpc = pyrucast.Model.mpc(
            [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
            mult_mpc,
            sense=">=",
        )
        mpc_mult = mult_mpc.node(0, 0, 0)

        model = base | dirichlet | mpc
        rhs_mesh = pyrucast.Mesh(c, "POI1")
        rhs_mesh.unit().add_cell([dir_mult])
        rhs_mesh.unit().add_cell([mpc_mult])
        rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T", "mpc_rhs"])
        rhs[0].set_value(dir_mult, "imposed_T", 0.0)
        rhs[0].set_value(mpc_mult, "mpc_rhs", g)

        solution = pyrucast.solver.solve_unilateral(
            model, pyrucast.assemble.stiffness(model, materials), rhs
        )
        for i, node in enumerate(nodes):
            assert abs(solution.value(node, "T") - slope * i * H) < TOL
        lam = solution.value(mpc_mult, "lambda_mpc")
        if slope == 0.0:
            assert abs(lam) < TOL
        else:
            assert lam < 0.0


def test_all_equality_model_falls_back_to_plain_solve():
    """No inequality: `solve_unilateral` agrees with the plain `solve`."""
    nodes, model, materials, rhs, _ = _bounded_bar(3.0, 0.5, "=")
    k = pyrucast.assemble.stiffness(model, materials)
    a = pyrucast.solver.solve_unilateral(model, k, rhs)
    b = pyrucast.solver.solve(k, rhs)
    for node in nodes:
        assert abs(a.value(node, "T") - b.value(node, "T")) < TOL


def test_eliminate_rejects_unilateral_relations():
    """`solve_eliminate` must reject a unilateral model with a clear error."""
    _, model, materials, rhs, _ = _bounded_bar(1.0, 2.0, ">=")
    k = pyrucast.assemble.stiffness(model, materials)
    with pytest.raises(Exception, match="solve_unilateral"):
        pyrucast.solver.solve_eliminate(model, k, rhs)


def test_unknown_sense_rejected():
    """An unknown `sense` string raises a clear error at model build."""
    _, nodes, _, _ = _heat_bar()
    imposed = pyrucast.mesher.poi1_from_nodes([nodes[0]])
    mult = pyrucast.mesher.barycenter(imposed)
    with pytest.raises(Exception, match="sense"):
        pyrucast.Model.dirichlet("T", "q", imposed, mult, sense="~")
