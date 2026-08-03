"""Python tests for the Mazars damage model (the COMP brick)."""

import pyrucast


def _unit_quad():
    c = pyrucast.Coords(2)
    n = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([1.0, 1.0]),
        c.add_node([0.0, 1.0]),
    ]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(n)
    return c, n, pyrucast.FiniteElementSpace(mesh)


def _uniform_strain(c, nodes, eps_xx):
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in nodes:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    for node, (x, _y) in zip(nodes, coords):
        u[0].set_value(node, "u_x", eps_xx * x)
        u[0].set_value(node, "u_y", 0.0)
    return u


# Concrete-like parameters (MPa).
_PARAMS = [
    ("E", 30_000.0),
    ("nu", 0.2),
    ("eps_d0", 1e-4),
    ("A_t", 0.8),
    ("B_t", 20_000.0),
    ("A_c", 1.4),
    ("B_c", 1_900.0),
]


def test_mazars_undamaged_below_threshold():
    c, nodes, fes = _unit_quad()
    model = pyrucast.Model.mazars(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(model, _PARAMS)

    u = _uniform_strain(c, nodes, 1e-5)  # < eps_d0
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    sub = state[0]
    assert "damage" in sub.components()
    assert "kappa" in sub.components()
    for g in range(sub.gauss_count()):
        assert abs(sub.value(0, g, "damage")) < 1e-14


def test_mazars_damages_in_tension():
    c, nodes, fes = _unit_quad()
    model = pyrucast.Model.mazars(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(model, _PARAMS)

    u = _uniform_strain(c, nodes, 5e-4)  # > eps_d0
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    sub = state[0]
    for g in range(sub.gauss_count()):
        d = sub.value(0, g, "damage")
        assert 0.0 < d < 1.0, f"damage {d}"
        assert sub.value(0, g, "kappa") >= 5e-4 - 1e-12


def test_mazars_rejects_inconsistent_model():
    _c, _n, fes = _unit_quad()
    for bad in ("solid", "nonsense"):
        try:
            pyrucast.Model.mazars(fes, bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"expected an error for model={bad!r}")
