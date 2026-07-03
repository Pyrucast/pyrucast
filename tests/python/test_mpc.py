"""Python tests for multi-point constraints (`Model.mpc`).

MPCs impose linear relations `Σ aₖ·u(nodeₖ, varₖ) = g` via Lagrange multipliers,
on the same augmented system as Dirichlet. On a 1-D heat-conduction bar
`-u'' = 0` (analytical `u(x)` linear), a well-posed set of relations recovers the
linear solution, and the single-term MPC coincides with Dirichlet.
"""

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

    imposed0 = pyrucast.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", "q", imposed0, mult0)
    dir_mult = mult0.node(0, 0, 0)

    base = pyrucast.Model.heat_conduction(fes)
    dual = base.dual_of("T")  # "q"
    mesh_last = pyrucast.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.barycenter(mesh_last)
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

    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    for i, node in enumerate(nodes):
        assert abs(solution.value(node, "T") - i * H) < TOL
    assert (
        abs(solution.value(nodes[-1], "T") - solution.value(nodes[0], "T") - 1.0) < TOL
    )


def test_single_term_mpc_matches_dirichlet():
    """A single-term MPC `1·T = u_d` reproduces the equivalent Dirichlet."""

    def solve_dirichlet():
        c, nodes, fes, materials = _heat_bar()
        left = pyrucast.poi1_from_nodes([nodes[0]])
        right = pyrucast.poi1_from_nodes([nodes[-1]])
        ml, mr = pyrucast.barycenter(left), pyrucast.barycenter(right)
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
        return nodes, pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    def solve_mpc():
        c, nodes, fes, materials = _heat_bar()
        left = pyrucast.poi1_from_nodes([nodes[0]])
        ml = pyrucast.barycenter(left)
        right = pyrucast.poi1_from_nodes([nodes[-1]])
        mm = pyrucast.barycenter(right)
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
        return nodes, pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    nodes, dir_sol = solve_dirichlet()
    _, mpc_sol = solve_mpc()
    for node in nodes:
        assert abs(dir_sol.value(node, "T") - mpc_sol.value(node, "T")) < TOL
