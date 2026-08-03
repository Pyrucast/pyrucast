"""Python smoke tests for structural-element mass and geometric stiffness."""

import pyrucast


def _bar(dx, dy):
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([dx, dy])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    return pyrucast.FiniteElementSpace(mesh), a, b, (dx * dx + dy * dy) ** 0.5


def test_truss_consistent_mass():
    rho, area = 2.0, 3.0
    fes, a, b, length = _bar(3.0, 4.0)  # L = 5
    model = pyrucast.Model.truss(fes)
    materials = pyrucast.element_field.material_field(
        model, [("E", 1.0), ("A", area), ("rho", rho)]
    )

    m = pyrucast.matrix.mass(model, materials)
    base = rho * area * length / 6.0
    tol = 1e-12
    assert abs(m.get(a, "f_x", a, "u_x") - 2.0 * base) < tol
    assert abs(m.get(a, "f_x", b, "u_x") - base) < tol
    assert abs(m.get(a, "f_x", a, "u_y")) < tol  # block-diagonal in components


def test_truss_geometric_transverse():
    n = 7.0
    fes, a, b, length = _bar(1.0, 0.0)  # axis x ⇒ transverse y
    model = pyrucast.Model.truss(fes)
    materials = pyrucast.element_field.material_field(model, [("E", 1.0), ("A", 1.0)])
    state = pyrucast.ElementField(fes, ["n"])
    state[0].set_uniform("n", n)

    kg = pyrucast.matrix.geometric(model, materials, state)
    tol = 1e-12
    assert abs(kg.get(a, "f_y", a, "u_y") - n / length) < tol
    assert abs(kg.get(a, "f_y", b, "u_y") + n / length) < tol
    assert abs(kg.get(a, "f_x", a, "u_x")) < tol  # axial has no geometric term
