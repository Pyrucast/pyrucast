"""Large displacements in the step-by-step loop (``pyrucast.stepping``).

The laws read a large-transformation measure (``strain_measure``); the loop
reads it off them. Every test has an answer in closed form, independent of
the loop: a rigid rotation that strains nothing, the hydrostatic state a
follower pressure leaves, the force of a perfectly plastic bar in Hencky
strain, the heat flux through a stretched bar.
"""

import math

import pyrucast as pc
import pytest


def _grid(nx, ny, length, height):
    """An nx×ny grid of QUA4 on [0, length]×[0, height]:
    (coords, mesh, fes, node(i, j), position(i, j))."""
    c = pc.Coords(2)

    def position(i, j):
        return [i * length / nx, j * height / ny]

    grid = [c.add_node(position(i, j)) for j in range(ny + 1) for i in range(nx + 1)]

    def node(i, j):
        return grid[j * (nx + 1) + i]

    mesh = pc.Mesh(c, "QUA4")
    for j in range(ny):
        for i in range(nx):
            mesh.unit().add_cell(
                [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            )
    return c, mesh, pc.FiniteElementSpace(mesh), node, position


def _fix(target, nodes, variable, dim=2):
    """``variable`` held on ``nodes``; returns the constraint and the support
    its imposed value goes on."""
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multipliers = pc.mesh.translate(imposed, [0.0] * dim)
    return pc.model.dirichlet(target, variable, imposed, multipliers), multipliers


def _on(support, component, value):
    field = pc.NodeField(support, [component])
    field[0].add_to_component(component, value)
    return field


def _turned_edge(measure):
    """A free 4×1 strip whose left edge is turned by θ(t) = t·π/2, one
    constraint per node and direction: (data, node, position, nx, ny)."""
    nx, ny = 4, 2
    _, _mesh, fes, node, position = _grid(nx, ny, 4.0, 1.0)
    solid = pc.model.elasticity(fes, "plane_strain", strain_measure=measure)
    model = solid
    held = []
    for j in range(ny + 1):
        for variable in ("u_x", "u_y"):
            constraint, support = _fix(solid, [node(0, j)], variable)
            model = model | constraint
            held.append((j, variable, support))

    def turned(theta):
        loads = None
        for j, variable, support in held:
            x, y = position(0, j)
            if variable == "u_x":
                value = math.cos(theta) * x - math.sin(theta) * y - x
            else:
                value = math.sin(theta) * x + math.cos(theta) * y - y
            field = _on(support, "imposed_" + variable, value)
            loads = field if loads is None else loads | field
        return loads

    times = [0.25 * k for k in range(5)]
    data = {
        "model": model,
        "times": times,
        "materials": pc.element_field.material_field(
            model, [("E", 1000.0), ("nu", 0.3)]
        ),
        "loads": pc.Evolution([(t, turned(t * math.pi / 2)) for t in times]),
        "precision": 1e-10,
        "strain_precision": 1e-12,
    }
    return data, node, position, nx, ny


def test_a_free_strip_turned_by_one_edge_rotates_without_stress():
    """Turned by 90° in four steps, the strip rotates rigidly: every node lands
    at R·X and nothing is stressed."""
    data, node, position, nx, ny = _turned_edge("green_lagrange")
    pc.stepping.step_by_step(data)
    last = data["results"][-1]
    assert all(record["converged"] for record in data["results"])
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = position(i, j)
            u = last["displacement"]
            assert u.value(node(i, j), "u_x") == pytest.approx(-y - x, abs=1e-9)
            assert u.value(node(i, j), "u_y") == pytest.approx(x - y, abs=1e-9)
    for name in ("S_xx", "S_yy", "S_xy"):
        assert last["behaviour"].abs().max(name) < 1e-8
        assert last["cauchy"].abs().max("sigma" + name[1:]) < 1e-8


def test_the_linear_strain_of_the_same_motion_is_not_zero():
    """The contrast: read on the linear strain, the same imposed rotation
    builds a stress of the order of E (ε = cos θ − 1 = −1 at 90°)."""
    data, *_ = _turned_edge("linear")
    pc.stepping.step_by_step(data)
    assert data["results"][-1]["behaviour"].abs().max("sigma_xx") > 100.0


def test_a_follower_pressure_leaves_a_hydrostatic_cauchy_stress():
    """A quarter of a square under a pressure p on its two free walls, which
    follows them as they move: the Cauchy stress is −p I exactly, whatever the
    strain (here several per cent)."""
    p = 10.0
    c, _mesh, fes, node, _ = _grid(2, 2, 1.0, 1.0)
    solid = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
    left, _ = _fix(solid, [node(0, j) for j in range(3)], "u_x")
    bottom, _ = _fix(solid, [node(i, 0) for i in range(3)], "u_y")
    walls = pc.Mesh(c, "SEG2")
    # Counter-clockwise: the normals point out of the material.
    for a, b in [
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 2), (1, 2)),
        ((1, 2), (0, 2)),
    ]:
        walls.unit().add_cell([node(*a), node(*b)])
    pressure = pc.model.flux(pc.FiniteElementSpace(walls), solid, "n")
    model = solid | left | bottom | pressure

    def materials(phi):
        return pc.element_field.material_field(
            model, [("E", 100.0), ("nu", 0.3), ("phi_n", phi)]
        )

    data = {
        "model": model,
        "times": [0.0, 0.5, 1.0],
        "materials": pc.Evolution([(0.0, materials(0.0)), (1.0, materials(-p))]),
        "precision": 1e-10,
        "strain_precision": 1e-12,
    }
    pc.stepping.step_by_step(data)
    last = data["results"][-1]
    assert last["converged"]
    sigma = last["cauchy"]
    for name in ("sigma_xx", "sigma_yy"):
        assert sigma.min(name) == pytest.approx(-p, rel=1e-9)
        assert sigma.max(name) == pytest.approx(-p, rel=1e-9)
    assert sigma.abs().max("sigma_xy") < 1e-9 * p
    # Several per cent of strain: well outside small displacements.
    assert last["displacement"].value(node(2, 2), "u_x") < -0.05


def test_a_perfectly_plastic_bar_in_hencky_strain_pulls_with_sigma_y_over_lambda():
    """A cube on sliding supports, pulled to λ = 1.5 along x: once plastic,
    the Hencky dual is T = σ_y along the bar, hence S = σ_y/λ² and a nominal
    force A₀ P = A₀ σ_y / λ — whatever the lateral contraction."""
    sigma_y = 1.0
    c = pc.Coords(3)
    corners = [
        [x, y, z] for z in (0.0, 1.0) for (x, y) in [(0, 0), (1, 0), (1, 1), (0, 1)]
    ]
    nodes = [c.add_node([float(v) for v in p]) for p in corners]
    mesh = pc.Mesh(c, "HEX8")
    mesh.unit().add_cell(nodes)
    fes = pc.FiniteElementSpace(mesh)
    bar = pc.model.plasticity_perfect(fes, "full_3d", strain_measure="hencky")

    def face(axis, value):
        return [n for n, p in zip(nodes, corners, strict=True) if p[axis] == value]

    model = bar
    for axis, variable in enumerate(("u_x", "u_y", "u_z")):
        model = model | _fix(bar, face(axis, 0.0), variable, dim=3)[0]
    pull, pull_support = _fix(bar, face(0, 1.0), "u_x", dim=3)
    model = model | pull
    data = {
        "model": model,
        "times": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "materials": pc.element_field.material_field(
            model, [("E", 1000.0), ("nu", 0.3), ("sigma_y", sigma_y)]
        ),
        "loads": pc.Evolution(
            [
                (0.0, _on(pull_support, "imposed_u_x", 0.0)),
                (1.0, _on(pull_support, "imposed_u_x", 0.5)),
            ]
        ),
    }
    pc.stepping.step_by_step(data)
    for record in data["results"][2:]:
        stretch = 1.0 + 0.5 * record["time"]
        assert record["converged"]
        held = sum(record["reactions"].value(n, "f_x") for n in face(0, 1.0))
        assert abs(held) == pytest.approx(sigma_y / stretch, rel=1e-4)
        assert record["behaviour"].max("T_xx") == pytest.approx(sigma_y, rel=1e-4)


def test_heat_runs_on_the_deformed_part():
    """A strip stretched to λ = 1.5 along x (ν = 0, so its height stays), held
    at 0 and 100 at its ends: once stretched, the heat crosses a longer strip,
    and the gradient — hence the flux k ∇T — is ΔT / (λ L₀)."""
    k, length, stretch = 3.0, 2.0, 1.5
    _, _mesh, fes, node, _ = _grid(4, 1, length, 0.5)
    conduction = pc.model.heat_conduction(fes)
    solid = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
    left = [node(0, j) for j in range(2)]
    right = [node(4, j) for j in range(2)]
    cold, cold_support = _fix(conduction, left, "T")
    hot, hot_support = _fix(conduction, right, "T")
    pull, pull_support = _fix(solid, right, "u_x")
    model = (
        conduction
        | solid
        | cold
        | hot
        | _fix(solid, left, "u_x")[0]
        | _fix(solid, [node(i, 0) for i in range(5)], "u_y")[0]
        | pull
    )
    fluxes = []
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0],
        "materials": pc.element_field.material_field(
            model, [("k", k), ("E", 100.0), ("nu", 0.0)]
        ),
        "loads": _on(cold_support, "imposed_T", 0.0)
        | _on(hot_support, "imposed_T", 100.0)
        | _on(pull_support, "imposed_u_x", (stretch - 1.0) * length),
        "hooks": {
            "after_heat": lambda _data, _state, heat: fluxes.append(heat.behaviour)
        },
    }
    pc.stepping.step_by_step(data)
    # The first step's heat runs on the part at rest, the second on the
    # stretched one (the configuration the mechanics reached before it).
    # The conduction law writes k ∇T (`flux_x`), the gradient of the
    # configuration it runs on.
    at_rest, stretched = fluxes
    assert at_rest.min("flux_x") == pytest.approx(k * 100.0 / length, rel=1e-9)
    assert stretched.min("flux_x") == pytest.approx(
        k * 100.0 / (stretch * length), rel=1e-6
    )
    assert stretched.max("flux_x") == pytest.approx(
        k * 100.0 / (stretch * length), rel=1e-6
    )


def test_what_large_displacements_do_not_offer_is_refused():
    _, _, fes, node, _ = _grid(2, 1, 2.0, 1.0)
    solid = pc.model.elasticity(fes, "plane_strain")
    materials = pc.element_field.material_field(solid, [("E", 1.0), ("nu", 0.3)])
    base = {"model": solid, "times": [0.0, 1.0], "materials": materials}
    # Large displacements come from the laws, not from a switch.
    with pytest.raises(ValueError, match="strain_measure"):
        pc.stepping.step_by_step(base | {"large_displacements": True})
    # K_σ belongs to large displacements.
    with pytest.raises(ValueError, match="geometric_stiffness"):
        pc.stepping.step_by_step(base | {"geometric_stiffness": True})
    # A law in large transformations beside a law in small strains.
    gl = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
    c = fes.mesh().coords()
    bars = pc.Mesh(c, "SEG2")
    bars.unit().add_cell([node(0, 1), node(2, 1)])
    truss = pc.model.truss(pc.FiniteElementSpace(bars))
    mixed = gl | truss
    with pytest.raises(ValueError, match="same strain measure"):
        pc.stepping.step_by_step(
            {
                "model": mixed,
                "times": [0.0, 1.0],
                "materials": pc.element_field.material_field(
                    mixed, [("E", 1.0), ("nu", 0.3), ("A", 1.0)]
                ),
            }
        )
