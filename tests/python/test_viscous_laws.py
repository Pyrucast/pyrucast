"""Python tests for the rate-dependent laws — creep and viscoplasticity.

What distinguishes these from the plastic laws is that the answer depends on
`dt`, so the tests turn on time rather than on a stress value.
"""

import pytest

import pyrucast

CORNERS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
]

NORTON = [("E", 150_000.0), ("nu", 0.3), ("K", 400.0), ("n", 5.0)]


def _cube():
    c = pyrucast.Coords(3)
    nodes = [c.add_node(p) for p in CORNERS]
    mesh = pyrucast.Mesh(c, "HEX8")
    mesh.unit().add_cell(nodes)
    return c, nodes, pyrucast.FiniteElementSpace(mesh)


def _strain(c, nodes, fes, exx):
    u_mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        u_mesh.unit().add_cell([n])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y", "u_z"])
    for node, p in zip(nodes, CORNERS):
        u[0].set_value(node, "u_x", exx * p[0])
        u[0].set_value(node, "u_y", 0.0)
        u[0].set_value(node, "u_z", 0.0)
    return pyrucast.element_field.deformation(u, fes)


def _von_mises(sub):
    s = [sub.value(0, 0, f"sigma_{n}") for n in ("xx", "yy", "zz", "yz", "xz", "xy")]
    mean = (s[0] + s[1] + s[2]) / 3.0
    d = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]]
    return (
        1.5
        * (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 2 * (d[3] ** 2 + d[4] ** 2 + d[5] ** 2))
    ) ** 0.5


def test_a_viscous_law_refuses_to_integrate_without_dt():
    """Integrating a creep law as if it were instantaneous must not be silent."""
    c, nodes, fes = _cube()
    model = pyrucast.model.creep_norton(fes, "full_3d")
    materials = pyrucast.element_field.material_field(model, NORTON)
    strain = _strain(c, nodes, fes, 2e-3)
    with pytest.raises(RuntimeError, match="rate-dependent"):
        pyrucast.element_field.integrate_behavior(model, strain, materials)


def test_holding_a_strain_relaxes_the_stress():
    c, nodes, fes = _cube()
    model = pyrucast.model.creep_norton(fes, "full_3d")
    materials = pyrucast.element_field.material_field(model, NORTON)
    strain = _strain(c, nodes, fes, 2e-3)

    previous = float("inf")
    for dt in (1e-4, 1e-3, 1e-2, 1e-1):
        state = pyrucast.element_field.integrate_behavior(
            model, strain, materials, dt=dt
        )
        q = _von_mises(state[0])
        assert q < previous, f"dt={dt}: {q} is not below {previous}"
        previous = q


def test_each_viscous_law_declares_its_own_material():
    _c, _n, fes = _cube()
    contracts = {
        "creep_norton": ["E", "nu", "K", "n"],
        "creep_lemaitre": ["E", "nu", "K", "N", "M"],
        "creep_blackburn": ["E", "nu", "A_1", "alpha_1", "r_1", "B_s", "beta_s"],
        "viscoplasticity_chaboche": [
            "E",
            "nu",
            "k",
            "K",
            "n",
            "C_1",
            "gamma_1",
            "b",
            "Q",
        ],
    }
    for name, expected in contracts.items():
        model = getattr(pyrucast.model, name)(fes, "full_3d")
        assert model[0].material_components() == expected, name


def test_chaboche_carries_a_back_stress_in_its_state():
    """Seven extra internal variables is what cyclic capability costs."""
    c, nodes, fes = _cube()
    model = pyrucast.model.viscoplasticity_chaboche(fes, "full_3d")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 150_000.0),
            ("nu", 0.3),
            ("k", 100.0),
            ("K", 150.0),
            ("n", 4.0),
            ("C_1", 60_000.0),
            ("gamma_1", 400.0),
            ("b", 10.0),
            ("Q", 50.0),
        ],
    )
    state = pyrucast.element_field.integrate_behavior(
        model, _strain(c, nodes, fes, 3e-3), materials, dt=1e-2
    )
    components = state[0].components()
    for name in ("X_xx", "X_yy", "X_xy", "R"):
        assert name in components, f"{name} missing from the state"
    # The back stress follows the loading direction.
    assert state[0].value(0, 0, "X_xx") > 0.0


def test_lemaitre_chaboche_adds_damage():
    c, nodes, fes = _cube()
    model = pyrucast.model.viscoplasticity_lemaitre_chaboche(fes, "full_3d")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 150_000.0),
            ("nu", 0.3),
            ("k", 100.0),
            ("K", 150.0),
            ("n", 4.0),
            ("C_1", 60_000.0),
            ("gamma_1", 400.0),
            ("b", 10.0),
            ("Q", 50.0),
            ("S", 1.0),
            ("s", 1.0),
            ("D_c", 0.3),
        ],
    )
    strain = _strain(c, nodes, fes, 4e-3)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials, dt=1e-2)
    assert "damage" in state[0].components()
    first = state[0].value(0, 0, "damage")
    assert first > 0.0

    # Damage never heals, and never exceeds D_c.
    for _ in range(6):
        state = pyrucast.element_field.integrate_behavior(
            model, strain, materials, prev=state, dt=1e-2
        )
        current = state[0].value(0, 0, "damage")
        assert current >= first - 1e-15
        assert current <= 0.3 + 1e-12
        first = current
