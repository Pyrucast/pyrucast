"""Contact, load control, adaptive steps, coupling passes and hooks of the
step-by-step (``pyrucast.stepping``), each against a closed-form answer."""

import pyrucast as pc
import pytest

E, NU = 1000.0, 0.25
NX, NY, L, H = 4, 2, 2.0, 1.0


def _strip():
    c = pc.Coords(2)
    grid = [
        c.add_node([i * L / NX, j * H / NY])
        for j in range(NY + 1)
        for i in range(NX + 1)
    ]

    def node(i, j):
        return grid[j * (NX + 1) + i]

    mesh = pc.Mesh(c, "QUA4")
    for j in range(NY):
        for i in range(NX):
            mesh.unit().add_cell(
                [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            )
    return mesh, pc.FiniteElementSpace(mesh), node


def _fix(target, nodes, variable, sense=None):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multipliers = pc.mesh.translate(imposed, [0.0, 0.0])
    return pc.model.dirichlet(
        target, variable, imposed, multipliers, sense
    ), multipliers


def _edge_traction(nodes, component, total):
    """Nodal forces of a uniform traction of resultant ``total`` along an edge
    of equal segments: half a share at each end."""
    support = pc.mesh.poi1_from_nodes(nodes)
    field = pc.NodeField(support, [component])
    share = total / (len(nodes) - 1)
    for k, n in enumerate(nodes):
        field[0].set_value(
            n, component, share * (0.5 if k in (0, len(nodes) - 1) else 1.0)
        )
    return field


def _value(support, component, value):
    field = pc.NodeField(support, [component])
    field[0].add_to_component(component, value)
    return field


def _hung_strip():
    """A strip hanging from its top edge (held in y, and in x at one corner),
    pulled down by a uniform traction on its bottom edge."""
    mesh, fes, node = _strip()
    solid = pc.model.elasticity(fes, "plane_stress")
    top = [node(i, NY) for i in range(NX + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    model = solid | _fix(solid, top, "u_y")[0] | _fix(solid, [node(0, NY)], "u_x")[0]
    materials = pc.element_field.material_field(model, [("E", E), ("nu", NU)])
    return mesh, node, solid, model, materials, bottom


def test_contact_stops_the_strip_on_the_floor():
    """Uniform tension σ = t stretches the strip by t H / E until its bottom
    meets the floor a gap g below; from then on the bottom stays at −g and
    the floor takes the rest, (t − E g / H)·L."""
    gap = 0.02
    _mesh, _node, solid, model, materials, bottom = _hung_strip()
    floor, floor_support = _fix(solid, bottom, "u_y", ">=")
    model = model | floor
    pull = _edge_traction(bottom, "f_y", -1.0 * L)
    floor_at = _value(floor_support, "imposed_u_y", -gap)
    data = {
        "model": model,
        "times": [0.0, 0.25, 0.5, 0.75, 1.0],
        "materials": materials,
        "loads": pc.Evolution(
            [(0.0, pull * 0.0 | floor_at), (1.0, pull * 40.0 | floor_at)]
        ),
    }
    pc.stepping.step_by_step(data)
    for record in data["results"][1:]:
        traction = 40.0 * record["time"]
        free = -traction * H / E
        if free > -gap:
            expected, carried = free, 0.0
        else:
            expected, carried = -gap, (traction - E * gap / H) * L
        for n in bottom:
            assert record["displacement"].value(n, "u_y") == pytest.approx(
                expected, rel=1e-6
            )
        floor_force = sum(record["reactions"].value(n, "f_y") for n in bottom)
        assert floor_force == pytest.approx(carried, rel=1e-6, abs=1e-9)


def _pulled_strip():
    """A plane-stress strip held at x = 0 (in x) and y = 0 (in y), with a unit
    traction on its right edge: uniform uniaxial σ_xx."""
    mesh, fes, node = _strip()
    solid = pc.model.elasticity(fes, "plane_stress")
    left = [node(0, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    model = solid | _fix(solid, left, "u_x")[0] | _fix(solid, bottom, "u_y")[0]
    materials = pc.element_field.material_field(model, [("E", E), ("nu", NU)])
    right = [node(NX, j) for j in range(NY + 1)]
    return mesh, node, model, materials, _edge_traction(right, "f_x", H)


def test_automatic_control_cuts_a_large_step_and_lands_on_the_target():
    """A step straining 5e-3 under a control aiming at 1e-3 per step is cut
    into several; the target is reached all the same, with the linear
    answer u_x = σ x / E."""
    _mesh, node, model, materials, unit = _pulled_strip()
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": materials,
        "loads": pc.Evolution([(0.0, unit * 0.0), (1.0, unit * 5.0)]),
        "automatic": True,
        "automatic_target": 1e-3,
        "save_times": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    steps = []
    data["hooks"] = {"after_step": lambda d, s: steps.append(s.time)}
    pc.stepping.step_by_step(data)
    assert len(steps) >= 4
    assert steps[-1] == pytest.approx(1.0)
    last = data["results"][-1]
    assert last["time"] == pytest.approx(1.0)
    assert last["displacement"].value(node(NX, 1), "u_x") == pytest.approx(
        5.0 * L / E, rel=1e-6
    )


def test_indirect_control_advances_the_strain_by_the_target():
    """Under indirect control the pilot load grows so that each step adds
    Δτ to the strain along its own direction: the first step along
    (1, 1, 1) — ε_xx + ε_yy = (1 − ν) σ / E = Δτ —, the next ones along ε
    itself — ‖ε‖ grows by Δτ, ‖ε‖ = √(1 + ν²) σ / E."""
    _mesh, _node, model, materials, unit = _pulled_strip()
    target = 1e-3
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0, 3.0],
        "materials": materials,
        "indirect_control": {"load": unit, "increment": target},
    }
    pc.stepping.step_by_step(data)
    norm = (1.0 + NU**2) ** 0.5
    first = E * target / (1.0 - NU)
    expected = [0.0, first, first + E * target / norm, first + 2.0 * E * target / norm]
    for record, sigma in zip(data["results"], expected, strict=True):
        assert record["load_multiplier"] == pytest.approx(sigma, rel=1e-6)
        if record["behaviour"] is not None:
            assert record["behaviour"].max("sigma_xx") == pytest.approx(sigma, rel=1e-6)


def test_adaptive_steps_reach_the_targets_of_a_plastic_pull():
    """With adaptive steps the instants between two targets are the solver's
    choice; what is recorded at the targets is the plastic answer."""
    sigma_y = 2.0
    _mesh, fes, node = _strip()
    solid = pc.model.plasticity_perfect(fes, "plane_stress")
    left = [node(0, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    right = [node(NX, j) for j in range(NY + 1)]
    pull, support = _fix(solid, right, "u_x")
    model = solid | _fix(solid, left, "u_x")[0] | _fix(solid, bottom, "u_y")[0] | pull
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", sigma_y)]
    )
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0],
        "materials": materials,
        "loads": pc.Evolution(
            [
                (0.0, _value(support, "imposed_u_x", 0.0)),
                (2.0, _value(support, "imposed_u_x", 8e-3 * L)),
            ]
        ),
        "adaptive_step": True,
    }
    pc.stepping.step_by_step(data)
    assert [r["time"] for r in data["results"]] == [0.0, 1.0, 2.0]
    for record in data["results"][1:]:
        assert record["behaviour"].max("sigma_xx") == pytest.approx(sigma_y, rel=1e-4)
        assert record["behaviour"].max("p") == pytest.approx(
            4e-3 * record["time"] - sigma_y / E, rel=1e-3
        )


def test_hooks_see_every_step_and_may_add_a_load():
    """A load added by ``extra_mechanical_load`` acts like a given one; the
    after-step hook sees each step once."""
    _mesh, node, model, materials, unit = _pulled_strip()
    seen = []
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0],
        "materials": materials,
        "hooks": {
            "extra_mechanical_load": lambda d, s, t: unit * t,
            "after_step": lambda d, s: seen.append(s.time),
        },
    }
    pc.stepping.step_by_step(data)
    assert seen == [1.0, 2.0]
    assert data["results"][-1]["displacement"].value(
        node(NX, 0), "u_x"
    ) == pytest.approx(2.0 * L / E, rel=1e-9)


def test_coupling_passes_settle_on_the_one_pass_answer():
    """Without a feedback of the mechanics on the heat, a second pass gives the
    same temperature: the coherence test stops there, on the same result."""
    _mesh, fes, node = _strip()
    conduction = pc.model.heat_conduction(fes)
    solid = pc.model.elasticity(fes, "plane_stress")
    left = [node(0, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    hot, hot_support = _fix(conduction, left, "T")
    model = (
        conduction
        | solid
        | hot
        | _fix(solid, left, "u_x")[0]
        | _fix(solid, bottom, "u_y")[0]
    )
    materials = pc.element_field.material_field(
        model, [("k", 1.0), ("E", E), ("nu", NU), ("alpha", 1e-5)]
    )
    answers = []
    for coherence in (False, True):
        data = {
            "model": model,
            "times": [0.0, 1.0],
            "materials": materials,
            "loads": _value(hot_support, "imposed_T", 50.0),
            "coherence": coherence,
        }
        pc.stepping.step_by_step(data)
        answers.append(data["results"][-1]["displacement"].value(node(NX, NY), "u_x"))
    assert answers[1] == pytest.approx(answers[0], rel=1e-12)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"friction": True}, "friction"),
        ({"coherence": True, "adaptive_step": True}, "coherence"),
        ({"heat_scheme": "explicit"}, "heat_scheme"),
        ({"hooks": {"automatic_measure": lambda *_: 1.0}}, "automatic_measure"),
    ],
)
def test_options_refuse_what_they_cannot_do(extra, message):
    _mesh, _node, model, materials, _unit = _pulled_strip()
    data = {"model": model, "times": [0.0, 1.0], "materials": materials, **extra}
    with pytest.raises(ValueError, match=message):
        pc.stepping.step_by_step(data)
