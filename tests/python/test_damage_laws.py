"""Python tests for the damage laws and Gurson's porous plasticity."""

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

TC = [
    ("E", 30_000.0),
    ("nu", 0.2),
    ("f_t", 3.0),
    ("f_c", 30.0),
    ("A_t", 0.9),
    ("A_c", 0.5),
]


def _cube():
    c = pyrucast.Coords(3)
    nodes = [c.add_node(p) for p in CORNERS]
    mesh = pyrucast.Mesh(c, "HEX8")
    mesh.unit().add_cell(nodes)
    return c, nodes, pyrucast.FiniteElementSpace(mesh)


def _strain(c, nodes, fes, factors):
    """A uniform strain `diag(factors)` as nodal displacements."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        u_mesh.unit().add_cell([n])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y", "u_z"])
    for node, p in zip(nodes, CORNERS):
        for a, axis in enumerate("xyz"):
            u[0].set_value(node, f"u_{axis}", factors[a] * p[a])
    return pyrucast.element_field.deformation(u, fes)


def test_damage_tc_keeps_two_independent_histories():
    c, nodes, fes = _cube()
    model = pyrucast.model.damage_tc(fes, "solid")
    materials = pyrucast.element_field.material_field(model, TC)
    state = pyrucast.element_field.integrate_behavior(
        model, _strain(c, nodes, fes, [1.5e-3, 0.0, 0.0]), materials
    )
    comps = state[0].components()
    for name in ("d_plus", "d_minus", "r_plus", "r_minus"):
        assert name in comps, f"{name} missing"
    # Tension damages; compression does not.
    assert state[0].value(0, 0, "d_plus") > 0.1
    assert state[0].value(0, 0, "d_minus") < 1e-12


def test_sic_sic_damages_by_direction():
    c, nodes, fes = _cube()
    model = pyrucast.model.damage_sic_sic(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 230_000.0),
            ("nu", 0.2),
            ("eps_0_1", 5e-4),
            ("eps_c_1", 2e-3),
            ("d_max_1", 0.6),
            ("eps_0_2", 5e-4),
            ("eps_c_2", 2e-3),
            ("d_max_2", 0.6),
            ("eps_0_3", 5e-4),
            ("eps_c_3", 2e-3),
            ("d_max_3", 0.6),
            ("V1X", 1.0),
            ("V1Y", 0.0),
            ("V1Z", 0.0),
            ("V2X", 0.0),
            ("V2Y", 1.0),
            ("V2Z", 0.0),
        ],
    )
    state = pyrucast.element_field.integrate_behavior(
        model, _strain(c, nodes, fes, [2e-3, 0.0, 0.0]), materials
    )
    assert state[0].value(0, 0, "d_1") > 0.05
    assert state[0].value(0, 0, "d_2") < 1e-12
    assert state[0].value(0, 0, "d_3") < 1e-12


def test_gurson_exposes_and_grows_its_porosity():
    c, nodes, fes = _cube()
    model = pyrucast.model.gurson(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 200_000.0),
            ("nu", 0.3),
            ("sigma_y", 400.0),
            ("q_1", 1.5),
            ("q_2", 1.0),
            ("q_3", 2.25),
            ("f_0", 0.001),
            ("f_c", 0.15),
            ("f_f", 0.25),
        ],
    )
    # Triaxial tension: voids open where the pressure pulls them apart.
    strain = _strain(c, nodes, fes, [5e-3, 5e-3, 5e-3])
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    assert "porosity" in state[0].components()
    first = state[0].value(0, 0, "porosity")
    assert first > 0.001, "the porosity must start from f_0 and grow"

    state = pyrucast.element_field.integrate_behavior(
        model, strain, materials, prev=state
    )
    assert state[0].value(0, 0, "porosity") >= first, "the porosity must not shrink"


def test_mazars_kept_its_name_and_its_single_scalar():
    _c, _n, fes = _cube()
    model = pyrucast.model.mazars(fes, "solid")
    assert model[0].material_components() == [
        "E",
        "nu",
        "eps_d0",
        "A_t",
        "B_t",
        "A_c",
        "B_c",
    ]
