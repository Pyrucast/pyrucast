"""Python tests for the yield laws beyond perfect plasticity.

The behaviour integration is where a law shows what it is, so each test drives a
single element to a chosen strain and reads the stress back — no solve in
between.
"""

import pyrucast


def _cube():
    """A single HEX8 unit cube and its FE space."""
    c = pyrucast.Coords(3)
    corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    nodes = [c.add_node(p) for p in corners]
    mesh = pyrucast.Mesh(c, "HEX8")
    mesh.unit().add_cell(nodes)
    return c, nodes, corners, pyrucast.FiniteElementSpace(mesh)


def _stress(model, materials, c, nodes, corners, fes, exx):
    """The stress at the first Gauss point under a uniaxial strain `exx`."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        u_mesh.unit().add_cell([n])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y", "u_z"])
    for node, p in zip(nodes, corners):
        u[0].set_value(node, "u_x", exx * p[0])
        u[0].set_value(node, "u_y", 0.0)
        u[0].set_value(node, "u_z", 0.0)
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    return state[0]


def _von_mises(sub):
    s = [sub.value(0, 0, f"sigma_{n}") for n in ("xx", "yy", "zz", "yz", "xz", "xy")]
    mean = (s[0] + s[1] + s[2]) / 3.0
    d = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]]
    return (
        1.5
        * (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 2 * (d[3] ** 2 + d[4] ** 2 + d[5] ** 2))
    ) ** 0.5


def test_each_law_declares_its_own_material():
    _c, _n, _p, fes = _cube()
    contracts = {
        "plasticity_perfect": ["E", "nu", "sigma_y"],
        "plasticity_isotropic": ["E", "nu", "sigma_y", "H"],
        "drucker_prager": ["E", "nu", "friction", "k", "psi"],
        "ottosen": ["E", "nu", "a", "b", "k_1", "k_2", "sigma_c"],
    }
    for name, expected in contracts.items():
        model = getattr(pyrucast.Model, name)(fes, "solid")
        assert model[0].material_components() == expected, name


def test_isotropic_hardening_satisfies_consistency():
    c, nodes, corners, fes = _cube()
    model = pyrucast.Model.plasticity_isotropic(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model, [("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0), ("H", 5_000.0)]
    )
    sub = _stress(model, materials, c, nodes, corners, fes, 0.02)
    p = sub.value(0, 0, "p")
    assert p > 0.0
    assert abs(_von_mises(sub) - (200.0 + 5_000.0 * p)) < 1e-6 * (200.0 + 5_000.0 * p)


def test_drucker_prager_lands_on_its_cone():
    c, nodes, corners, fes = _cube()
    model = pyrucast.Model.drucker_prager(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [("E", 20_000.0), ("nu", 0.2), ("friction", 0.3), ("k", 30.0), ("psi", 0.1)],
    )
    sub = _stress(model, materials, c, nodes, corners, fes, 0.02)
    trace = sum(sub.value(0, 0, f"sigma_{n}") for n in ("xx", "yy", "zz"))
    f = _von_mises(sub) + 0.3 * trace - 30.0
    assert abs(f) < 1e-5


def test_ottosen_is_far_weaker_in_tension_than_compression():
    c, nodes, corners, fes = _cube()
    model = pyrucast.Model.ottosen(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 30_000.0),
            ("nu", 0.2),
            ("a", 1.2759),
            ("b", 3.1962),
            ("k_1", 11.7365),
            ("k_2", 0.9801),
            ("sigma_c", 30.0),
        ],
    )
    trace = lambda sub: sum(  # noqa: E731
        sub.value(0, 0, f"sigma_{n}") for n in ("xx", "yy", "zz")
    )
    tension = trace(_stress(model, materials, c, nodes, corners, fes, 0.01))
    compression = trace(_stress(model, materials, c, nodes, corners, fes, -0.01))
    assert abs(compression) > 3.0 * abs(tension)


def test_the_perfect_law_kept_its_behaviour_under_its_new_name():
    """`plasticity` became `plasticity_perfect`; only the name changed."""
    c, nodes, corners, fes = _cube()
    model = pyrucast.Model.plasticity_perfect(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model, [("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)]
    )
    sub = _stress(model, materials, c, nodes, corners, fes, 0.02)
    # Perfect plasticity caps the equivalent stress at the yield stress.
    assert abs(_von_mises(sub) - 200.0) < 1e-6
    assert not hasattr(pyrucast.Model, "plasticity")
