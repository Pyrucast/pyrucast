"""Large displacements in the step-by-step loop (``pyrucast.stepping``).

The laws read a large-transformation measure (``strain_measure``); the loop
reads it off them. Every test has an answer in closed form, independent of
the loop: a rigid rotation that strains nothing, the hydrostatic state a
follower pressure leaves, the force of a perfectly plastic bar in Hencky
strain, the heat through a stretched bar, an oriented conduction that turns
with the part, an exchange on a stretched face, a truss snapping through,
a taut string under a weight.
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


def _fix(target, nodes, variable):
    """``variable`` held on ``nodes``; returns the constraint and the support
    its imposed value goes on."""
    constraint = pc.model.dirichlet(pc.mesh.poi1_from_nodes(nodes), target, variable)
    return constraint, constraint.multiplier_mesh()


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
            field = _on(support, "imposed", value)
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


def _cauchy(record, fes):
    """The Cauchy stress of a record, computed from what it holds: the
    results keep only the Lagrangian stress the laws write."""
    gradient = pc.element_field.deformation(record["displacement"], fes, "gradient")
    return pc.element_field.stress_measure(record["behaviour"], gradient, "cauchy")


def test_a_free_strip_turned_by_one_edge_rotates_without_stress():
    """Turned by 90° in four steps, the strip rotates rigidly: every node lands
    at R·X and nothing is stressed."""
    data, node, position, nx, ny = _turned_edge("green_lagrange")
    pc.stepping.step_by_step(data)
    last = data["results"][-1]
    sigma = _cauchy(last, data["model"][0:1].fespace())
    assert all(record["converged"] for record in data["results"])
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = position(i, j)
            u = last["displacement"]
            assert u.value(node(i, j), "u_x") == pytest.approx(-y - x, abs=1e-9)
            assert u.value(node(i, j), "u_y") == pytest.approx(x - y, abs=1e-9)
    for name in ("S_xx", "S_yy", "S_xy"):
        assert last["behaviour"].abs().max(name) < 1e-8
        assert sigma.abs().max("sigma" + name[1:]) < 1e-8


def test_computing_again_reuses_the_current_configuration():
    """A second computation on the same model finds the current
    configuration the first one added, rather than adding another — and
    gives the same answer, the set being laid again from the reference."""
    data, node, *_ = _turned_edge("green_lagrange")
    coords = data["model"][0:1].fespace().mesh().coords()
    pc.stepping.step_by_step(data)
    after_one = coords.names()
    first = data["results"][-1]["displacement"].value(node(4, 2), "u_x")
    pc.stepping.step_by_step(data)
    assert coords.names() == after_one
    assert after_one.count("stepping:current") == 1
    assert coords.active == 0
    again = data["results"][-1]["displacement"].value(node(4, 2), "u_x")
    assert again == pytest.approx(first, abs=1e-12)


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
    sigma = _cauchy(last, fes)
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
        model = model | _fix(bar, face(axis, 0.0), variable)[0]
    pull, pull_support = _fix(bar, face(0, 1.0), "u_x")
    model = model | pull
    data = {
        "model": model,
        "times": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "materials": pc.element_field.material_field(
            model, [("E", 1000.0), ("nu", 0.3), ("sigma_y", sigma_y)]
        ),
        "loads": pc.Evolution(
            [
                (0.0, _on(pull_support, "imposed", 0.0)),
                (1.0, _on(pull_support, "imposed", 0.5)),
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


def test_a_stretched_bar_conducts_as_the_longer_bar_it_is():
    """A strip stretched to λ = 1.5 along x (ν = 0, so its height stays), held
    at 0 and 100 at its ends. The conductivity is the material's, whatever its
    shape: the heat runs through the stretched strip, longer, and the flux the
    law writes is the true one, kΔT / L₀ at rest and kΔT / (λL₀) stretched."""
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
        "loads": _on(cold_support, "imposed", 0.0)
        | _on(hot_support, "imposed", 100.0)
        | _on(pull_support, "imposed", (stretch - 1.0) * length),
        "hooks": {
            "after_heat": lambda _data, _state, heat: fluxes.append(heat.behaviour)
        },
    }
    pc.stepping.step_by_step(data)
    # The first step's heat runs on the part at rest, the second on the
    # stretched one (the configuration the mechanics reached before it).
    for flux, current_length in zip(fluxes, (length, stretch * length), strict=True):
        expected = k * 100.0 / current_length
        assert flux.min("flux_x") == pytest.approx(expected, rel=1e-9)
        assert flux.max("flux_x") == pytest.approx(expected, rel=1e-9)


def _turned_with_heat(axis_angle):
    """The strip of ``_turned_edge``, conducting heat along an oriented
    conductivity (k₁ = 10 along an axis at ``axis_angle`` from the strip,
    k₂ = 1), held at 0 on its left end and 100 on its right one; returns
    (data, node, nx, ny, temperatures) — the temperature of every heat step."""
    data, node, _position, nx, ny = _turned_edge("green_lagrange")
    fes = data["model"][0:1].fespace()
    conduction = pc.model.heat_conduction(fes, symmetry="orthotropic")
    cold, cold_support = _fix(conduction, [node(0, j) for j in range(ny + 1)], "T")
    hot, hot_support = _fix(conduction, [node(nx, j) for j in range(ny + 1)], "T")
    data["model"] = data["model"] | conduction | cold | hot
    data["materials"] = pc.element_field.material_field(
        data["model"],
        [
            ("E", 1000.0),
            ("nu", 0.3),
            ("k_1", 10.0),
            ("k_2", 1.0),
            ("k_3", 1.0),
            ("V1X", math.cos(axis_angle)),
            ("V1Y", math.sin(axis_angle)),
        ],
    )
    held = _on(cold_support, "imposed", 0.0) | _on(hot_support, "imposed", 100.0)
    data["loads"] = pc.Evolution(
        [(t, data["loads"].interpolate(t) | held) for t in data["times"]]
    )
    temperatures = []
    data["hooks"] = {
        "after_heat": lambda _data, _state, heat: temperatures.append(heat.temperature)
    }
    return data, node, nx, ny, temperatures


def test_an_oriented_conduction_turns_with_the_part():
    """Material axes at 30° from the strip, so that the heat does not run
    straight along it: the temperature varies across the strip. Turned rigidly
    (the last heat step runs on the strip turned by 67.5°), the part carries
    its axes along — turned by the rotation R of F = R U, k = R k₀ Rᵀ — and
    every node keeps the temperature it had at rest."""
    data, node, nx, ny, temperatures = _turned_with_heat(math.pi / 6)
    pc.stepping.step_by_step(data)
    assert all(record["converged"] for record in data["results"])
    at_rest, turned = temperatures[0], temperatures[-1]
    # The anisotropy shows: across the middle of the strip, T is not uniform.
    middle = [at_rest.value(node(nx // 2, j), "T") for j in range(ny + 1)]
    assert max(middle) - min(middle) > 1.0
    for i in range(nx + 1):
        for j in range(ny + 1):
            assert turned.value(node(i, j), "T") == pytest.approx(
                at_rest.value(node(i, j), "T"), abs=1e-9
            )


def _cooled_strip(scheme, stretch_x, stretch_y, materials, held=None, start=None):
    """A 4×2 strip of QUA4 on [0, 2]×[0, 1], cooled by convection (h = 2,
    ambient 0) on its right face and stretched to ``stretch_x`` along,
    ``stretch_y`` across (ν = 0, sliding supports) by its first mechanical
    step; ``materials`` adds to the convection's and the solid's, ``held`` is
    the temperature of its left end, ``start`` its uniform initial one. Runs
    two steps of length 1 and returns (node, the temperature of each heat
    step): the first on the part at rest, the second on the stretched one."""
    length, height = 2.0, 1.0
    c, mesh, fes, node, _ = _grid(4, 2, length, height)
    conduction = pc.model.heat_conduction(fes)
    face = pc.Mesh(c, "SEG2")
    for j in range(2):
        face.unit().add_cell([node(4, j), node(4, j + 1)])
    cooling = pc.model.boundary_transfer(pc.FiniteElementSpace(face), conduction, ["T"])
    solid = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
    along, along_support = _fix(solid, [node(4, j) for j in range(3)], "u_x")
    across, across_support = _fix(solid, [node(i, 2) for i in range(5)], "u_y")
    model = (
        conduction
        | cooling
        | solid
        | _fix(solid, [node(0, j) for j in range(3)], "u_x")[0]
        | _fix(solid, [node(i, 0) for i in range(5)], "u_y")[0]
        | along
        | across
    )
    loads = _on(along_support, "imposed", (stretch_x - 1.0) * length) | _on(
        across_support, "imposed", (stretch_y - 1.0) * height
    )
    if held is not None:
        hold, hold_support = _fix(conduction, [node(0, j) for j in range(3)], "T")
        model = model | hold
        loads = loads | _on(hold_support, "imposed", held)
    temperatures = []
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0],
        "materials": pc.element_field.material_field(
            model,
            [("h_T", 2.0), ("a_ext_T", 0.0), ("E", 100.0), ("nu", 0.0)] + materials,
        ),
        "loads": loads,
        "heat_scheme": scheme,
        "hooks": {
            "after_heat": lambda _data, _state, heat: temperatures.append(
                heat.temperature
            )
        },
    }
    if start is not None:
        initial = pc.NodeField(mesh, ["T"])
        for zone in initial:
            zone.add_to_component("T", start)
        data["initial"] = {"temperature": initial}
    pc.stepping.step_by_step(data)
    assert all(record["converged"] for record in data["results"])
    return node, temperatures


@pytest.mark.parametrize("scheme", ["nonlinear", "linear"])
def test_the_heat_runs_through_the_deformed_part(scheme):
    """A strip held at T₀ = 100 on its left end and cooled by convection on its
    right one, then stretched to λ = 1.5 along and μ = 1.2 across: both act on
    the part as it is, the conduction through G_c = k μH₀ / (λL₀), the exchange
    through G_h = h μH₀. In series, the face sits at T₀ G_c / (G_c + G_h).
    Conducting on the reference (G_c = k H₀ / L₀) or exchanging there
    (G_h = h H₀) would put it elsewhere. The linear scheme solves once: its
    operator must be exact."""
    k, h, length, height, stretch_x, stretch_y = 3.0, 2.0, 2.0, 1.0, 1.5, 1.2
    node, temperatures = _cooled_strip(
        scheme, stretch_x, stretch_y, [("k", k)], held=100.0
    )
    for temperature, (lx, ly) in zip(
        temperatures, ((1.0, 1.0), (stretch_x, stretch_y)), strict=True
    ):
        conductance = k * ly * height / (lx * length)
        expected = 100.0 * conductance / (conductance + h * ly * height)
        for j in range(3):
            assert temperature.value(node(4, j), "T") == pytest.approx(
                expected, rel=1e-9
            )


@pytest.mark.parametrize("scheme", ["nonlinear", "linear"])
def test_the_capacity_stays_on_the_reference(scheme):
    """The same strip, insulated but for its cooled face, starting uniform at
    T₀ = 100 with a conductivity large enough to stay uniform: each implicit
    step gives T ← T C / (C + G_h Δt). The exchange works on the face as it is,
    G_h = h μH₀, and the capacity is the mass's, C = ρ c V₀ whatever the
    stretch — mass conservation, ρ c dv = ρ₀ c dV₀. On the stretched part
    (λμ V₀) the second step would cool less."""
    h, length, height, stretch_x, stretch_y = 2.0, 2.0, 1.0, 1.5, 1.2
    node, temperatures = _cooled_strip(
        scheme,
        stretch_x,
        stretch_y,
        [("k", 1e9), ("rho", 1.0), ("cp", 1.0)],
        start=100.0,
    )
    capacity = 1.0 * 1.0 * length * height
    expected = 100.0
    for temperature, ly in zip(temperatures, (1.0, stretch_y), strict=True):
        expected *= capacity / (capacity + h * ly * height)
        for i in range(5):
            for j in range(3):
                assert temperature.value(node(i, j), "T") == pytest.approx(
                    expected, rel=1e-6
                )


def _bars(points, cells):
    """Bars in Green–Lagrange between ``points``: (nodes, mesh, bars)."""
    c = pc.Coords(2)
    nodes = [c.add_node(p) for p in points]
    mesh = pc.Mesh(c, "SEG2")
    for a, b in cells:
        mesh.unit().add_cell([nodes[a], nodes[b]])
    bars = pc.model.truss(pc.FiniteElementSpace(mesh), strain_measure="green_lagrange")
    return nodes, mesh, bars


def test_a_von_mises_truss_snaps_through_under_automatic_control():
    """Two bars from (∓a, 0) to the apex (0, h), pushed down at the apex. With
    w the apex deflection, l² = a² + (h − w)² and L² = a² + h², each bar
    carries S_n = EA (l² − L²)/2L² along the chord, and the load is
    P(w) = −2 S_n (h − w)/L: up to P_max at h − w = h/√3, down through zero
    when the bars lie flat, to −P_max, then up again. A size measure that
    grows with w — the apex deflection, signed — takes the automatic control
    through both limit points, as Cast3M's own test redefines AUTOPILO."""
    a, h, young, area, p_end = 1.0, 0.2, 1000.0, 1.0, 5.0
    (left, apex, right), _, bars = _bars(
        [[-a, 0.0], [0.0, h], [a, 0.0]], [(0, 1), (2, 1)]
    )
    model = bars
    for nodes, variable in (
        ([left, right], "u_x"),
        ([left, right], "u_y"),
        ([apex], "u_x"),
    ):
        model = model | _fix(bars, nodes, variable)[0]
    push = pc.NodeField(pc.mesh.poi1_from_nodes([apex]), ["f_y"])
    push[0].set_value(apex, "f_y", -p_end)
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": pc.element_field.material_field(
            model, [("E", young), ("A", area)]
        ),
        # Past the first limit point the load falls below zero: so does the time.
        "loads": pc.Evolution([(-1.0, push * -1.0), (1.0, push)]),
        "automatic": True,
        "automatic_target": 0.01,
        "precision": 1e-10,
    }
    # Each step lands where the control puts it, not at a time to save: the
    # path is read as it goes.
    path = []
    data["hooks"] = {
        "automatic_measure": lambda d, s, du, el: -du.value(apex, "u_y"),
        "after_step": lambda d, s: path.append(
            (-s.solution.value(apex, "u_y"), p_end * s.time, s.converged)
        ),
    }
    pc.stepping.step_by_step(data)

    rest = math.hypot(a, h)
    p_max = 2.0 * young * area * h**3 / (3.0 * math.sqrt(3.0) * rest**3)

    def load(w):
        return young * area * (h * h - (h - w) ** 2) * (h - w) / rest**3

    assert all(converged for _, _, converged in path)
    for w, p, _ in path:
        assert p == pytest.approx(load(w), abs=1e-8 * p_max)
    # Over the top, through the flat bars, back from the bottom, to the end.
    assert max(p for w, p, _ in path if w < h) == pytest.approx(p_max, rel=1e-2)
    assert min(p for _, p, _ in path) == pytest.approx(-p_max, rel=1e-2)
    assert path[-1][1] == pytest.approx(p_end)
    assert path[-1][0] > 2.0 * h


@pytest.mark.parametrize("geometric", [True, False])
def test_a_taut_string_carries_a_weight_through_its_geometric_stiffness(geometric):
    """A string of two bars, its ends pulled apart by δ before the weight
    hangs from its middle. Straight, it has no transverse stiffness but the
    one its tension gives: without K_σ the operator is singular. With it, the
    deflection w is the exact equilibrium 2 S_n w / L = P, with
    S_n = EA ((L + δ)² + w² − L²)/2L², solved here by a scalar Newton."""
    half, delta, young, area, weight = 1.0, 0.01, 1000.0, 1.0, 2.0
    (left, middle, right), mesh, string = _bars(
        [[-half, 0.0], [0.0, 0.0], [half, 0.0]], [(0, 1), (1, 2)]
    )
    held_left, at_left = _fix(string, [left], "u_x")
    held_right, at_right = _fix(string, [right], "u_x")
    model = string | held_left | held_right | _fix(string, [left, right], "u_y")[0]
    # Taut before it is loaded: the initial state is the stretched string.
    stretch = pc.NodeField(mesh, ["u_x", "u_y"])
    stretch[0].set_value(left, "u_x", -delta)
    stretch[0].set_value(right, "u_x", delta)
    own = pc.element_field.material_field(string, [("E", young), ("A", area)])
    tension = string.integrate_behavior(string.strain(stretch, own), own)
    pulled = _on(at_left, "imposed", -delta) | _on(at_right, "imposed", delta)
    hung = pc.NodeField(pc.mesh.poi1_from_nodes([middle]), ["f_y"])
    hung[0].set_value(middle, "f_y", -weight)
    data = {
        "model": model,
        "times": [0.0, 0.25, 0.5, 0.75, 1.0],
        "materials": pc.element_field.material_field(
            model, [("E", young), ("A", area)]
        ),
        "loads": pc.Evolution([(0.0, pulled | hung * 0.0), (1.0, pulled | hung)]),
        "initial": {"displacement": stretch, "behaviour": tension},
        "geometric_stiffness": geometric,
        "precision": 1e-10,
    }
    if not geometric:
        with pytest.raises(RuntimeError, match="singular"):
            pc.stepping.step_by_step(data)
        return
    pc.stepping.step_by_step(data)

    def deflection(p):
        w = 0.1
        for _ in range(50):
            s = young * area * ((half + delta) ** 2 + w * w - half**2) / (2.0 * half**2)
            ds = young * area * w / half**2
            w -= (2.0 * s * w / half - p) / (2.0 * (s + w * ds) / half)
        return w

    for record in data["results"][1:]:
        w = -record["displacement"].value(middle, "u_y")
        assert w == pytest.approx(deflection(weight * record["time"]), rel=1e-8)


def _ramp(times, pull):
    """A bar in Green–Lagrange from the anchor P(0, −1) to the node S(0.5, 0)
    resting on a rigid floor, flat from A(−1, 0) to B(1, 0) then rising along
    the ramp B–C, C(3, 1): the floor is held, its normals point up. S is pulled
    to the right by a fixed force growing with time to ``pull``. Returns
    (data, s, slave contact model)."""
    c = pc.Coords(2)
    anchor, s = c.add_node([0.0, -1.0]), c.add_node([0.5, 0.0])
    a, b, top = c.add_node([-1.0, 0.0]), c.add_node([1.0, 0.0]), c.add_node([3.0, 1.0])
    bar = pc.Mesh(c, "SEG2")
    bar.unit().add_cell([anchor, s])
    bars = pc.model.truss(pc.FiniteElementSpace(bar), strain_measure="green_lagrange")
    floor = pc.Mesh(c, "SEG2")
    floor.unit().add_cell([b, a])
    floor.unit().add_cell([top, b])
    contact = pc.model.contact(
        pc.mesh.poi1_from_nodes([s]), floor, bars, ["u_x", "u_y"]
    )
    model = bars | contact
    for variable in ("u_x", "u_y"):
        model = model | _fix(bars, [anchor, a, b, top], variable)[0]
    force = pc.NodeField(pc.mesh.poi1_from_nodes([s]), ["f_x"])
    force[0].set_value(s, "f_x", pull)
    data = {
        "model": model,
        "times": times,
        "materials": pc.element_field.material_field(model, [("E", 1.0), ("A", 1.0)]),
        "loads": pc.Evolution([(0.0, force * 0.0), (1.0, force)]),
        "precision": 1e-10,
    }
    return data, s


_ANCHOR = (0.0, -1.0)
_REST = math.dist(_ANCHOR, (0.5, 0.0))
_RAMP = (2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0))


def _bar_force(q):
    """The bar's axial force at S = q, pulling S toward the anchor: Cauchy
    force n = λ S_n with S_n = EA (l² − L²)/2L², λ = l/L, EA = 1."""
    l = math.dist(_ANCHOR, q)
    n = (l / _REST) * (l * l - _REST**2) / (2.0 * _REST**2)
    return n * (_ANCHOR[0] - q[0]) / l, n * (_ANCHOR[1] - q[1]) / l


def _root(f, lo, hi):
    """The root of ``f`` between ``lo`` and ``hi``, by bisection."""
    f_lo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _on_the_flat(pull):
    """Where S stands if the floor were flat everywhere: horizontal balance
    along y = 0 — what a contact frozen on its first facet would give."""
    x = _root(lambda x: pull + _bar_force((x, 0.0))[0], 0.5, 5.0)
    return x, 0.0


def _on_the_ramp(pull):
    """Where S stands on the ramp B + ψ t: the balance along the ramp, the
    floor pushing only along its normal."""

    def along(psi):
        q = (1.0 + psi * _RAMP[0], psi * _RAMP[1])
        fx, fy = _bar_force(q)
        return (pull + fx) * _RAMP[0] + fy * _RAMP[1]

    psi = _root(along, 0.0, 3.0)
    return 1.0 + psi * _RAMP[0], psi * _RAMP[1]


@pytest.mark.parametrize("times", [[0.0, 0.1, 0.2, 0.8, 1.0], [0.0, 1.0]])
def test_a_node_pulled_along_a_kinked_floor_climbs_the_ramp(times):
    """S slides along the flat, then past the kink B onto the ramp: the
    contact is updated each time the operator is rebuilt, and checked once
    more at convergence. Under the final pull it stands where the balance
    along the ramp puts it — not where a contact frozen on the flat would —,
    on the ramp (no gap), the reaction normal to the ramp. One case takes the
    whole pull in a single step, the kink crossed within it. In the other, the
    first correction of the first step throws S past the kink: the contact
    updated there holds it on the ramp, and only the check at convergence
    brings it back to the flat, where it belongs."""
    pull = 1.0
    data, s = _ramp(times, pull)
    pc.stepping.step_by_step(data)
    for record in data["results"][1:]:
        f = pull * record["time"]
        flat_end = -_bar_force((1.0, 0.0))[0]
        expected = _on_the_flat(f) if f < flat_end else _on_the_ramp(f)
        u = record["displacement"]
        at = (0.5 + u.value(s, "u_x"), u.value(s, "u_y"))
        assert at[0] == pytest.approx(expected[0], abs=1e-7)
        assert at[1] == pytest.approx(expected[1], abs=1e-7)
    # The last state, on the ramp: on it, pushed only along its normal, and
    # far from where the frozen contact would have left it.
    x, y = at
    assert y == pytest.approx((x - 1.0) * _RAMP[1] / _RAMP[0], abs=1e-9)
    reaction = record["reactions"]
    rx, ry = reaction.value(s, "f_x"), reaction.value(s, "f_y")
    assert abs(rx * _RAMP[0] + ry * _RAMP[1]) < 1e-8 * math.hypot(rx, ry)
    assert math.hypot(rx, ry) > 0.1
    frozen = _on_the_flat(pull)
    assert abs(x - frozen[0]) > 0.1


def _pressed_square(depth):
    """A unit square in Green–Lagrange (plane strain, ν = 0), its bottom held,
    its top edge pressed down by ``depth``: (data, fes)."""
    _, _, fes, node, _ = _grid(1, 1, 1.0, 1.0)
    solid = pc.model.elasticity(fes, "plane_strain", strain_measure="green_lagrange")
    held, _ = _fix(solid, [node(0, 0), node(1, 0)], "u_y")
    anchored, _ = _fix(solid, [node(0, 0)], "u_x")
    pressed, at = _fix(solid, [node(0, 1), node(1, 1)], "u_y")
    model = solid | held | anchored | pressed
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": pc.element_field.material_field(model, [("E", 1.0), ("nu", 0.0)]),
        "loads": pc.Evolution(
            [(0.0, _on(at, "imposed", 0.0)), (1.0, _on(at, "imposed", -depth))]
        ),
        "precision": 1e-10,
    }
    return data, fes


def test_a_cell_turned_inside_out_stops_the_step():
    """Pressed through its own base, the square turns inside out: J = det F
    < 0, where no law means anything. The iterate is rejected and the step
    fails, saying so — a Green–Lagrange law alone would have computed on.
    Pressed to half its height instead, it converges, and its volume ratio,
    measured after the fact, is the stretch, J = 1/2 (ν = 0)."""
    data, _ = _pressed_square(1.5)
    with pytest.raises(RuntimeError, match="inside out"):
        pc.stepping.step_by_step(data)

    data, fes = _pressed_square(0.5)
    pc.stepping.step_by_step(data)
    u = data["results"][-1]["displacement"].filter_components(["u_x", "u_y"])
    jacobian = pc.element_field.deformation(u, fes, "jacobian")
    assert jacobian.min("J") == pytest.approx(0.5, rel=1e-9)
    assert jacobian.max("J") == pytest.approx(0.5, rel=1e-9)


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
