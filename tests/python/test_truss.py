"""Python tests for the truss (bar) physics."""

import pyrucast


def _clamp(c, node, var, dual):
    """Homogeneous Dirichlet (u = 0) on `var` at `node` (imposed value 0)."""
    imposed = pyrucast.mesher.poi1_from_nodes([node])
    multiplier = pyrucast.mesher.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def test_truss_bar_axial_elongation():
    """Single horizontal bar, axial force F at the right end ⇒ u_x = F·L/(E·A)."""
    E, A, L, F = 210.0e9, 1.0e-4, 2.0, 1000.0
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([L, 0.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([n0, n1])
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.truss(fes)
    model = model | _clamp(c, n0, "u_x", "f_x")
    model = model | _clamp(c, n0, "u_y", "f_y")
    model = model | _clamp(c, n1, "u_y", "f_y")  # bar has no transverse stiffness

    materials = pyrucast.build.material_field(model, [("E", E), ("A", A)])

    load_mesh = pyrucast.Mesh(c, "POI1")
    load_mesh.unit().add_cell([n1])
    rhs = pyrucast.NodeField(load_mesh, ["f_x"])
    rhs[0].set_value(n1, "f_x", F)

    K = pyrucast.assemble.stiffness(model, materials)
    solution = pyrucast.solver.solve(K, rhs)

    expected = F * L / (E * A)
    assert abs(solution.value(n1, "u_x") - expected) < 1e-10 * expected
    assert abs(solution.value(n0, "u_x")) < 1e-18
    assert abs(solution.value(n1, "u_y")) < 1e-18


def test_truss_model_vars():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    model = pyrucast.Model.truss(fes)
    assert model.primal_vars() == ["u_x", "u_y"]
    assert model.dual_vars() == ["f_x", "f_y"]
