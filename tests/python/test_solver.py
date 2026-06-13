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
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    imposed_left = pyrucast.poi1_from_nodes([nodes[0]])
    imposed_right = pyrucast.poi1_from_nodes([nodes[-1]])
    mult_mesh_left = pyrucast.barycenter(imposed_left)
    mult_mesh_right = pyrucast.barycenter(imposed_right)
    left = pyrucast.Model.dirichlet("T", "q", imposed_left, mult_mesh_left)
    right = pyrucast.Model.dirichlet("T", "q", imposed_right, mult_mesh_right)
    mult_left = mult_mesh_left.node(0, 0, 0)
    mult_right = mult_mesh_right.node(0, 0, 0)
    model = pyrucast.Model.heat_conduction(fes) + left + right

    # Load: imposed values at the multiplier nodes (slot "imposed_T").
    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([mult_left])
    rhs_mesh.unit().add_cell([mult_right])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T"])
    rhs[0].set_value(mult_left, "imposed_T", 0.0)
    rhs[0].set_value(mult_right, "imposed_T", 1.0)

    K = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K, rhs)

    tol = 1e-10
    # T(x_i) = x_i at every node.
    for i, node in enumerate(nodes):
        expected = i * h
        got = solution.value(node, "T")
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
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    model = pyrucast.Model.heat_conduction(fes)
    K = pyrucast.stiffness(model, materials)

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([a])
    rhs = pyrucast.NodeField(rhs_mesh, ["q"])

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
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 2.0)

    imposed_left = pyrucast.poi1_from_nodes([nodes[0]])
    mult_mesh_left = pyrucast.barycenter(imposed_left)
    left = pyrucast.Model.dirichlet("T", "q", imposed_left, mult_mesh_left)
    mult_left = mult_mesh_left.node(0, 0, 0)
    model = pyrucast.Model.heat_conduction(fes) + left

    # Build a load NodeField with both components:
    #   "imposed_T" at mult_left → 5.0 (imposed value)
    #   "q"         at nodes[-1] → 1.0 (Neumann source on the boundary row)
    load_mesh = pyrucast.Mesh(c, "POI1")
    load_mesh.unit().add_cell([nodes[-1]])
    load_mesh.unit().add_cell([mult_left])
    rhs = pyrucast.NodeField(load_mesh, ["imposed_T", "q"])
    rhs[0].set_value(mult_left, "imposed_T", 5.0)
    rhs[0].set_value(nodes[-1], "q", 1.0)

    K = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K, rhs)

    # Expected linear solution: u(x) = 5 + 0.5 * x (because flux = 1, k = 2 → slope = 0.5).
    tol = 1e-10
    for i, node in enumerate(nodes):
        x = i * h
        expected = 5.0 + 0.5 * x
        got = solution.value(node, "T")
        assert abs(got - expected) < tol, f"node {i}: got {got}, expected {expected}"
