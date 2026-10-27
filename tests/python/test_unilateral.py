"""Python tests for unilateral (inequality) constraints (`sense=">="` / `"<="`)
and the active-set loop `solve` runs when the matrix carries them.

A 1-D heat-conduction bar `-T'' = 0` on `[0, 1]`, `T(0) = 0` and a flux load
`q` at the right end: the unconstrained solution is `T(x) = q·x`. A unilateral
bound `T(1) ⋈ a` either releases (feasible, `λ = 0`) or holds active
(`T(1) = a`, `λ = q − a` — `≤ 0` for `≥`, `≥ 0` for `≤`).
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


def _bounded_bar(q, bound, sense):
    """`T(0) = 0`, flux `q` at the right end, `T(1) <sense> bound`.

    Returns (nodes, model, materials, rhs, uni_mult).
    """
    c, nodes, fes, materials = _heat_bar()

    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    conduction = pyrucast.model.heat_conduction(fes)
    dirichlet = pyrucast.model.dirichlet(imposed0, conduction, "T")
    dir_mult = dirichlet.multiplier_mesh().node(0, 0, 0)

    imposed1 = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    unilateral = pyrucast.model.dirichlet(imposed1, conduction, "T", sense=sense)
    uni_mult = unilateral.multiplier_mesh().node(0, 0, 0)

    model = pyrucast.model.heat_conduction(fes) | dirichlet | unilateral

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([nodes[-1]])
    rhs_mesh.unit().add_cell([dir_mult])
    rhs_mesh.unit().add_cell([uni_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["q", "imposed"])
    rhs[0].set_value(nodes[-1], "q", q)
    rhs[0].set_value(dir_mult, "imposed", 0.0)
    rhs[0].set_value(uni_mult, "imposed", bound)

    return nodes, model, materials, rhs, uni_mult


def _check(nodes, solution, uni_mult, slope, lam):
    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - slope * i * H) < TOL
    assert abs(solution.value(uni_mult, "lambda") - lam) < TOL


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
    k = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(k, rhs)
    _check(nodes, solution, uni_mult, slope, lam)


def test_warm_start_survives_a_status_flip():
    """Re-solving on the same matrix warm-starts; a load flip re-iterates."""
    nodes, model, materials, rhs, uni_mult = _bounded_bar(1.0, 2.0, ">=")
    k = pyrucast.matrix.stiffness(model, materials)
    first = pyrucast.solver.solve(k, rhs)
    _check(nodes, first, uni_mult, 2.0, -1.0)
    # Identical re-solve: pure warm start (cached status + factorization).
    again = pyrucast.solver.solve(k, rhs)
    _check(nodes, again, uni_mult, 2.0, -1.0)
    # Stronger push on the same matrix: the relation must release from the
    # warm-started (active) status.
    rhs[0].set_value(nodes[-1], "q", 5.0)
    released = pyrucast.solver.solve(k, rhs)
    _check(nodes, released, uni_mult, 5.0, 0.0)


def test_unilateral_mpc_difference_relation():
    """`T(1) − T(0) ≥ g`: active for `g = 1` (T = x), released for `g = −1`."""
    for g, slope in [(1.0, 1.0), (-1.0, 0.0)]:
        c, nodes, fes, materials = _heat_bar()

        imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        conduction = pyrucast.model.heat_conduction(fes)
        dirichlet = pyrucast.model.dirichlet(imposed0, conduction, "T")
        dir_mult = dirichlet.multiplier_mesh().node(0, 0, 0)

        base = pyrucast.model.heat_conduction(fes)
        mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
        mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
        mpc = pyrucast.model.mpc(
            [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
            base,
            sense=">=",
        )
        mpc_mult = mpc.multiplier_mesh().node(0, 0, 0)

        model = base | dirichlet | mpc
        rhs_mesh = pyrucast.Mesh(c, "POI1")
        rhs_mesh.unit().add_cell([dir_mult])
        rhs_mesh.unit().add_cell([mpc_mult])
        rhs = pyrucast.NodeField(rhs_mesh, ["imposed"])
        rhs[0].set_value(dir_mult, "imposed", 0.0)
        rhs[0].set_value(mpc_mult, "imposed", g)

        solution = pyrucast.solver.solve(
            pyrucast.matrix.stiffness(model, materials), rhs
        )
        for i, node in enumerate(nodes):
            assert abs(solution.value(node, "T") - slope * i * H) < TOL
        lam = solution.value(mpc_mult, "lambda")
        if slope == 0.0:
            assert abs(lam) < TOL
        else:
            assert lam < 0.0


def test_the_matrix_carries_the_sense():
    """Same bar, same load, the bound declared `>=` or `=`: `solve` reads which
    from the matrix. Feasible push (`q = 5 > 2`): the inequality releases
    (`T(1) = 5`), the equation holds (`T(1) = 2`)."""
    answers = {}
    for sense in (">=", "="):
        nodes, model, materials, rhs, _ = _bounded_bar(5.0, 2.0, sense)
        k = pyrucast.matrix.stiffness(model, materials)
        answers[sense] = pyrucast.solver.solve(k, rhs).value(nodes[-1], "T")
    assert abs(answers[">="] - 5.0) < TOL
    assert abs(answers["="] - 2.0) < TOL


def test_elimination_keeps_the_inequalities():
    """The elimination eliminates the equation `T(0) = 0` and keeps the bound
    `T(1) >= 2` on the reduced system: same field and same multiplier as the
    Lagrange path. Under `q = 1` the bound saturates: `T = 2x`, `lambda = -1`."""
    nodes, model, materials, rhs, uni_mult = _bounded_bar(1.0, 2.0, ">=")
    k = pyrucast.matrix.stiffness(model, materials)
    lagrange = pyrucast.solver.solve(k, rhs)
    elimination = pyrucast.solver.solve(k, rhs, relations="eliminate")
    for i, node in enumerate(nodes):
        assert abs(elimination.value(node, "T") - 2.0 * i * H) < TOL
        assert abs(elimination.value(node, "T") - lagrange.value(node, "T")) < TOL
    assert abs(elimination.value(uni_mult, "lambda") + 1.0) < TOL


def test_unknown_sense_rejected():
    """An unknown `sense` is a bad argument: a `ValueError` naming what is
    accepted, raised at model build."""
    _, nodes, fes, _ = _heat_bar()
    imposed = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    conduction = pyrucast.model.heat_conduction(fes)
    with pytest.raises(ValueError, match="relation sense '~'"):
        pyrucast.model.dirichlet(imposed, conduction, "T", sense="~")
