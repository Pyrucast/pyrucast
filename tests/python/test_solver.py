"""Python tests for `pyrucast.solve` — dense LU end-to-end."""

import pyrucast


def test_poisson_1d_dirichlet_at_both_ends_recovers_linear_solution():
    """-u'' = 0 on [0, 1] with u(0) = 0, u(1) = 1. Analytical: u(x) = x.
    The Lagrange multipliers should match the boundary heat fluxes
    (+1 at the left, -1 at the right)."""
    n_elems = 4
    h = 1.0 / n_elems

    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]

    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.add_cell([nodes[i].id, nodes[i + 1].id])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(sub))
    left = pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[0].id])
    right = pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[-1].id])
    mult_left = left.multiplier_nodes()[0]
    mult_right = right.multiplier_nodes()[0]
    model.add_sub(left)
    model.add_sub(right)

    # Load: imposed values at the multiplier nodes.
    rhs_sm = pyrucast.SubMesh(c, "POI1")
    rhs_sm.add_cell([mult_left])
    rhs_sm.add_cell([mult_right])
    rhs = pyrucast.NodeField(rhs_sm, ["T"])
    rhs.set_value(mult_left, "T", 0.0)
    rhs.set_value(mult_right, "T", 1.0)

    K = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K, rhs)

    tol = 1e-10
    # T(x_i) = x_i at every node.
    for i, node in enumerate(nodes):
        expected = i * h
        got = solution.value(node.id, "T")
        assert abs(got - expected) < tol, f"node {i}: got {got}, expected {expected}"
    # Boundary fluxes (Lagrange multipliers).
    assert abs(solution.value(mult_left, "lambda_T") - 1.0) < tol
    assert abs(solution.value(mult_right, "lambda_T") + 1.0) < tol


def test_solver_singular_matrix_errors():
    """Without any Dirichlet, the bare conduction matrix is singular
    (kernel = constants)."""
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a.id, b.id])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(sub))
    K = pyrucast.stiffness(model, materials)

    rhs_sm = pyrucast.SubMesh(c, "POI1")
    rhs_sm.add_cell([a.id])
    rhs = pyrucast.NodeField(rhs_sm, ["q"])

    try:
        pyrucast.solve(K, rhs)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for singular matrix")


def test_solver_with_nonzero_neumann():
    """Heat conduction with one Dirichlet at the left and a Neumann
    source at the right. Solution: u(x) = u_d + q * x / k.
    """
    n_elems = 2
    h = 1.0 / n_elems
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.add_cell([nodes[i].id, nodes[i + 1].id])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 2.0)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(sub))
    left = pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[0].id])
    mult_left = left.multiplier_nodes()[0]
    model.add_sub(left)

    # Build a load NodeField with both components:
    #   "T" at mult_left  → 5.0 (imposed value)
    #   "q" at nodes[-1]  → 1.0 (Neumann source on the boundary row)
    load_sm = pyrucast.SubMesh(c, "POI1")
    load_sm.add_cell([nodes[-1].id])
    load_sm.add_cell([mult_left])
    rhs = pyrucast.NodeField(load_sm, ["T", "q"])
    rhs.set_value(mult_left, "T", 5.0)
    rhs.set_value(nodes[-1].id, "q", 1.0)

    K = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K, rhs)

    # Expected linear solution: u(x) = 5 + 0.5 * x (because flux = 1, k = 2 → slope = 0.5).
    tol = 1e-10
    for i, node in enumerate(nodes):
        x = i * h
        expected = 5.0 + 0.5 * x
        got = solution.value(node.id, "T")
        assert abs(got - expected) < tol, f"node {i}: got {got}, expected {expected}"
