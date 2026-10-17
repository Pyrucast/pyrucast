"""Analytic cases of the heat step-by-step (``pyrucast.stepping``).

The transient schemes are checked on a problem with a single free
temperature, where each scheme is a scalar recurrence written out in the test
from the assembled capacity and conductance; the steady ones against their
closed-form profiles.
"""

import itertools

import pytest

import pyrucast as pc

SIGMA_SB = 5.670374419e-8


def _rod(n_cells=2, length=2.0):
    """A 1-D rod of SEG2 on [0, length]: (mesh, fes, nodes)."""
    c = pc.Coords(1)
    nodes = [c.add_node([length * i / n_cells]) for i in range(n_cells + 1)]
    mesh = pc.Mesh(c, "SEG2")
    for a, b in itertools.pairwise(nodes):
        mesh.unit().add_cell([a, b])
    return mesh, pc.FiniteElementSpace(mesh), nodes


def _held(target, nodes, variable):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multipliers = pc.mesh.translate(imposed, [0.0])
    return pc.model.dirichlet(target, variable, imposed, multipliers), multipliers


def _entry(matrix, mesh, node):
    """The diagonal entry of a nodal matrix at ``node``: the product with the
    unit field at that node, read back there."""
    unit = pc.NodeField(mesh, ["T"])
    unit[0].set_value(node, "T", 1.0)
    return (matrix * unit).value(node, "q")


def _single_dof(scheme, theta=1.0):
    """A rod held at T = 0 at both ends, its middle at T = 1: one free
    temperature. Returns the run and the scalars ``c`` (capacity) and ``k``
    (conductance) of that temperature."""
    mesh, fes, nodes = _rod()
    conduction = pc.model.heat_conduction(fes)
    ends, _ = _held(conduction, [nodes[0], nodes[-1]], "T")
    model = conduction | ends
    materials = pc.element_field.material_field(
        conduction, [("k", 3.0), ("rho", 2.0), ("cp", 5.0)]
    )
    start = pc.NodeField(mesh, ["T"])
    start[0].set_value(nodes[1], "T", 1.0)
    data = {
        "model": model,
        "times": [0.1 * k for k in range(6)],
        "materials": materials,
        "initial": {"temperature": start},
        "heat_scheme": scheme,
        "theta": theta,
    }
    pc.stepping.step_by_step(data)
    k = _entry(pc.matrix.stiffness(conduction, materials), mesh, nodes[1])
    c = _entry(pc.matrix.mass(conduction, materials), mesh, nodes[1])
    middle = [r["temperature"].value(nodes[1], "T") for r in data["results"]]
    return middle, c, k


@pytest.mark.parametrize("scheme", ["nonlinear", "linear"])
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_the_theta_scheme_is_its_recurrence(scheme, theta):
    """(c/Δt + θk) Tₙ₊₁ = (c/Δt − (1−θ)k) Tₙ for the free temperature."""
    middle, c, k = _single_dof(scheme, theta)
    dt = 0.1
    expected = [1.0]
    for _ in range(5):
        expected.append(
            expected[-1] * (c / dt - (1.0 - theta) * k) / (c / dt + theta * k)
        )
    for got, want in zip(middle, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_dupont_is_its_three_level_recurrence():
    """(c/Δt + (½+λ)k) T₂ = (c/Δt + (2λ−½)k) T₁ − λk T₀, from the second step on —
    with the true previous temperature, step after step."""
    middle, c, k = _single_dof("dupont")
    dt, lam = 0.1, 0.25
    # The first step has no previous one: it is the θ-scheme (θ = 1).
    expected = [1.0, c / dt / (c / dt + k)]
    for _ in range(4):
        t1, t0 = expected[-1], expected[-2]
        expected.append(
            ((c / dt + (2.0 * lam - 0.5) * k) * t1 - lam * k * t0)
            / (c / dt + (0.5 + lam) * k)
        )
    for got, want in zip(middle, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def _plate():
    """A 4×1 QUA4 strip on [0, 2]×[0, 0.5]: (mesh, fes, node(i, j))."""
    c = pc.Coords(2)
    grid = [c.add_node([0.5 * i, 0.5 * j]) for j in range(2) for i in range(5)]

    def node(i, j):
        return grid[j * 5 + i]

    mesh = pc.Mesh(c, "QUA4")
    for i in range(4):
        mesh.unit().add_cell([node(i, 0), node(i + 1, 0), node(i + 1, 1), node(i, 1)])
    return c, mesh, pc.FiniteElementSpace(mesh), node


def test_steady_conduction_between_two_temperatures_is_linear():
    _, _, fes, node = _plate()
    conduction = pc.model.heat_conduction(fes)
    left, left_support = _held2(conduction, [node(0, 0), node(0, 1)])
    right, right_support = _held2(conduction, [node(4, 0), node(4, 1)])
    loads = _value(left_support, 20.0) | _value(right_support, 120.0)
    data = {
        "model": conduction | left | right,
        "times": [0.0, 1.0],
        "materials": pc.element_field.material_field(conduction, [("k", 2.0)]),
        "loads": loads,
    }
    pc.stepping.step_by_step(data)
    temperature = data["results"][-1]["temperature"]
    for i in range(5):
        for j in range(2):
            assert temperature.value(node(i, j), "T") == pytest.approx(20.0 + 25.0 * i)


def _held2(target, nodes):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multipliers = pc.mesh.translate(imposed, [0.0, 0.0])
    return pc.model.dirichlet(target, "T", imposed, multipliers), multipliers


def _value(support, value):
    field = pc.NodeField(support, ["imposed_T"])
    field[0].add_to_component("imposed_T", value)
    return field


def test_steady_radiation_balances_the_conducted_flux():
    """T(0) imposed, the far end radiating to T∞: the profile is linear and
    its end temperature solves k (T₀ − T_L)/L = σε (T_L⁴ − T∞⁴) — solved here by
    a scalar Newton, independent of the finite elements."""
    k, emis, t0, t_inf, length = 5.0, 0.8, 900.0, 300.0, 2.0
    c, mesh, fes, node = _plate()
    conduction = pc.model.heat_conduction(fes)
    edge = pc.Mesh(c, "SEG2")
    edge.unit().add_cell([node(4, 0), node(4, 1)])
    radiation = pc.model.radiation(pc.FiniteElementSpace(edge), conduction)
    left, left_support = _held2(conduction, [node(0, 0), node(0, 1)])
    model = conduction | radiation | left
    materials = pc.element_field.material_field_per_sub_model(
        model, [[("k", k)], [("emis", emis), ("T_inf", t_inf)], []]
    )
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": materials,
        "loads": _value(left_support, t0),
        "initial": {"temperature": _uniform(mesh, t0)},
        "precision": 1e-8,
    }
    pc.stepping.step_by_step(data)
    end = t0
    for _ in range(50):
        f = k * (t0 - end) / length - SIGMA_SB * emis * (end**4 - t_inf**4)
        df = -k / length - 4.0 * SIGMA_SB * emis * end**3
        end -= f / df
    temperature = data["results"][-1]["temperature"]
    assert data["results"][-1]["converged"]
    for j in range(2):
        assert temperature.value(node(4, j), "T") == pytest.approx(end, rel=1e-7)
        assert temperature.value(node(2, j), "T") == pytest.approx(
            0.5 * (t0 + end), rel=1e-7
        )


def _uniform(mesh, value):
    field = pc.NodeField(mesh, ["T"])
    for zone in field:
        zone.add_to_component("T", value)
    return field
