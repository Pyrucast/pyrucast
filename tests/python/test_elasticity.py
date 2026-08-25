"""Python tests for linear elasticity."""

import pyrucast


def _clamp(nodes, var, dual):
    """Homogeneous Dirichlet (u = 0) on `var` over `nodes`."""
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(var, dual, imposed, multiplier)


def test_plane_stress_uniaxial_tension():
    """Unit square, traction S on the right edge ⇒ u_x = (S/E)·x, u_y = -(νS/E)·y."""
    E, NU, S, N = 210.0, 0.3, 2.0, 2
    h = 1.0 / N
    c = pyrucast.Coords(2)

    def idx(i, j):
        return j * (N + 1) + i

    grid = [c.add_node([i * h, j * h]) for j in range(N + 1) for i in range(N + 1)]
    mesh = pyrucast.Mesh(c, "QUA4")
    for j in range(N):
        for i in range(N):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    fes = pyrucast.FiniteElementSpace(mesh)

    left = [grid[idx(0, j)] for j in range(N + 1)]
    bottom = [grid[idx(i, 0)] for i in range(N + 1)]
    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = model | _clamp(left, "u_x", "f_x")
    model = model | _clamp(bottom, "u_y", "f_y")

    materials = pyrucast.element_field.material_field(model, [("E", E), ("nu", NU)])

    # Traction S on the right edge → consistent nodal forces (op flux).
    right_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        right_edge.unit().add_cell([grid[idx(N, j)], grid[idx(N, j + 1)]])
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    rhs = pyrucast.node_field.flux(right_fes[0], S, "f_x")

    K = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K, rhs)

    tol = 1e-10
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = i * h, j * h
            assert abs(solution.value(grid[idx(i, j)], "u_x") - S / E * x) < tol
            assert abs(solution.value(grid[idx(i, j)], "u_y") + NU * S / E * y) < tol


def test_elasticity_rejects_inconsistent_model():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, d])
    fes = pyrucast.FiniteElementSpace(mesh)
    # 2-D space cannot be "solid", and "nonsense" is not a model.
    for bad in ("solid", "nonsense"):
        try:
            pyrucast.model.elasticity(fes, bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"expected an error for model={bad!r}")
