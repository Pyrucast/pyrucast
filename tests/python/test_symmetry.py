"""Python tests for oriented materials — orthotropic and anisotropic.

The behaviour integration (`COMP`) is the sharpest place to check a
constitutive matrix from Python: it maps a strain we choose to the stress the
law produces, with no solve in between. So each test imposes a **uniform
strain** on a single QUA4 and reads the stress back.
"""

import math

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
    """The linear displacement `u_x = eps_xx * x`, `u_y = 0`."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in nodes:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    for node, (x, _y) in zip(nodes, coords):
        u[0].set_value(node, "u_x", eps_xx * x)
        u[0].set_value(node, "u_y", 0.0)
    return u


def _stress(model, materials, c, nodes, fes, eps_xx):
    u = _uniform_strain(c, nodes, eps_xx)
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    return state[0]


# Glass/epoxy-like plane orthotropy, plus the axes along the global ones.
_ORTHO = [
    ("E_1", 200.0),
    ("E_2", 50.0),
    ("E_3", 50.0),
    ("nu_12", 0.25),
    ("nu_13", 0.25),
    ("nu_23", 0.25),
    ("G_12", 30.0),
    ("G_13", 30.0),
    ("G_23", 30.0),
    ("V1X", 1.0),
    ("V1Y", 0.0),
]


def test_orthotropic_material_contract_is_declared():
    _c, _n, fes = _unit_quad()
    model = pyrucast.model.elasticity(fes, "plane_stress", symmetry="orthotropic")
    required = model[0].material_components()
    for name in ("E_1", "E_2", "nu_12", "G_12", "V1X", "V1Y"):
        assert name in required, f"{name} missing from {required}"
    # The isotropic constants are *not* part of an orthotropic contract — that
    # disjointness is what lets two zones share a mesh.
    assert "E" not in required
    assert "nu" not in required


def test_orthotropic_stress_is_stiffer_along_the_first_axis():
    c, nodes, fes = _unit_quad()
    model = pyrucast.model.elasticity(fes, "plane_stress", symmetry="orthotropic")
    materials = pyrucast.element_field.material_field(model, _ORTHO)
    sub = _stress(model, materials, c, nodes, fes, 1e-3)
    # Uniaxial strain along x with the stiff axis along x: σ_xx must exceed what
    # the transverse modulus alone would give.
    sigma_xx = sub.value(0, 0, "sigma_xx")
    assert sigma_xx > 50.0 * 1e-3


def test_orthotropy_with_equal_constants_matches_isotropy_at_any_angle():
    """Equal constants make an orthotropic law isotropic — frame or no frame.

    This drives the full material-axis rotation from Python and compares it, term
    by term, with the isotropic law it must degenerate to.
    """
    e, nu = 210.0, 0.3
    g = e / (2.0 * (1.0 + nu))

    c, nodes, fes = _unit_quad()
    iso_model = pyrucast.model.elasticity(fes, "plane_stress")
    iso_mat = pyrucast.element_field.material_field(iso_model, [("E", e), ("nu", nu)])
    iso = _stress(iso_model, iso_mat, c, nodes, fes, 1e-3)
    expected = {
        name: iso.value(0, 0, name) for name in ("sigma_xx", "sigma_yy", "sigma_xy")
    }

    for angle_deg in (0.0, 30.0, 90.0, 137.0):
        a = math.radians(angle_deg)
        c2, nodes2, fes2 = _unit_quad()
        model = pyrucast.model.elasticity(fes2, "plane_stress", symmetry="orthotropic")
        materials = pyrucast.element_field.material_field(
            model,
            [
                ("E_1", e),
                ("E_2", e),
                ("E_3", e),
                ("nu_12", nu),
                ("nu_13", nu),
                ("nu_23", nu),
                ("G_12", g),
                ("G_13", g),
                ("G_23", g),
                ("V1X", math.cos(a)),
                ("V1Y", math.sin(a)),
            ],
        )
        sub = _stress(model, materials, c2, nodes2, fes2, 1e-3)
        for name, want in expected.items():
            got = sub.value(0, 0, name)
            assert abs(got - want) < 1e-9, f"{angle_deg}° {name}: {got} != {want}"


def test_unknown_symmetry_is_rejected():
    _c, _n, fes = _unit_quad()
    try:
        pyrucast.model.elasticity(fes, "plane_stress", symmetry="cubic")
    except ValueError as exc:
        assert "cubic" in str(exc)
        assert "orthotropic" in str(exc)
    else:  # pragma: no cover - the constructor must refuse
        raise AssertionError("an unknown symmetry must raise")


def test_orthotropic_conduction_carries_its_axes():
    _c, _n, fes = _unit_quad()
    model = pyrucast.model.heat_conduction(fes, symmetry="orthotropic")
    required = model[0].material_components()
    for name in ("k_1", "k_2", "k_3", "V1X", "V1Y"):
        assert name in required, f"{name} missing from {required}"
    assert "k" not in required


def test_anisotropic_conduction_takes_the_full_tensor():
    _c, _n, fes = _unit_quad()
    model = pyrucast.model.heat_conduction(fes, symmetry="anisotropic")
    required = model[0].material_components()
    for name in ("k_11", "k_12", "k_22", "k_33"):
        assert name in required, f"{name} missing from {required}"


def test_isotropic_stays_the_default():
    """Adding the symmetry axis must not have changed any existing call."""
    _c, _n, fes = _unit_quad()
    assert pyrucast.model.elasticity(fes, "plane_stress")[0].material_components() == [
        "E",
        "nu",
    ]
    assert pyrucast.model.heat_conduction(fes)[0].material_components() == ["k"]
