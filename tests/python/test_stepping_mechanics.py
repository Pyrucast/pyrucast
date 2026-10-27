"""Analytic cases of the mechanical step-by-step (``pyrucast.stepping``).

Each test sets a problem whose answer is known in closed form — or a property
the scheme must hold, such as Newmark's energy conservation — and checks the
loop against it, never against its own output.
"""

import pyrucast as pc
import pytest

E, NU, ALPHA = 200_000.0, 0.3, 1e-5
T_REF = 20.0
NX, NY, L, H = 4, 2, 4.0, 1.0


def _strip(nx=NX, ny=NY, length=L, height=H):
    """An nx×ny grid of QUA4 on [0, length]×[0, height]: (mesh, fes, node(i, j))."""
    c = pc.Coords(2)
    grid = [
        c.add_node([i * length / nx, j * height / ny])
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]

    def node(i, j):
        return grid[j * (nx + 1) + i]

    mesh = pc.Mesh(c, "QUA4")
    for j in range(ny):
        for i in range(nx):
            mesh.unit().add_cell(
                [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            )
    return mesh, pc.FiniteElementSpace(mesh), node


def _fix(target, nodes, variable, sense="="):
    """``variable`` held on ``nodes``, one multiplier per node; returns the
    constraint and its multiplier support (where an imposed value goes)."""
    constraint = pc.model.dirichlet(
        pc.mesh.poi1_from_nodes(nodes), target, variable, sense
    )
    return constraint, constraint.multiplier_mesh()


def _on(support, component, value):
    """A load field: ``value`` on ``component`` at every node of ``support``."""
    field = pc.NodeField(support, [component])
    field[0].add_to_component(component, value)
    return field


def _supported_strip(law="elasticity", **kinematics):
    """A plane-stress strip held at u_x = 0 on the left and u_y = 0 at the
    bottom: free to stretch and contract, rigid motions removed."""
    mesh, fes, node = _strip(**kinematics)
    if law == "elasticity":
        solid = pc.model.elasticity(fes, "plane_stress")
    else:
        solid = pc.model.plasticity_perfect(fes, "plane_stress")
    left = [node(0, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    model = solid | _fix(solid, left, "u_x")[0] | _fix(solid, bottom, "u_y")[0]
    return mesh, fes, node, solid, model


def test_free_thermal_expansion_is_stress_free():
    _mesh, fes, node = _strip()
    thermal = pc.model.heat_conduction(fes)
    solid = pc.model.elasticity(fes, "plane_stress")
    left = [node(0, j) for j in range(NY + 1)]
    right = [node(NX, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    hot, hot_support = _fix(thermal, left + right, "T")
    model = (
        thermal
        | solid
        | hot
        | _fix(solid, left, "u_x")[0]
        | _fix(solid, bottom, "u_y")[0]
    )
    materials = pc.element_field.material_field(
        model, [("k", 1.0), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": materials,
        "loads": _on(hot_support, "imposed", 120.0),
        "t_ref": T_REF,
    }
    pc.stepping.step_by_step(data)
    last = data["results"][-1]
    assert last["converged"]
    for i in range(NX + 1):
        for j in range(NY + 1):
            x, y = i * L / NX, j * H / NY
            assert last["temperature"].value(node(i, j), "T") == pytest.approx(120.0)
            assert last["displacement"].value(node(i, j), "u_x") == pytest.approx(
                ALPHA * 100.0 * x, abs=1e-12
            )
            assert last["displacement"].value(node(i, j), "u_y") == pytest.approx(
                ALPHA * 100.0 * y, abs=1e-12
            )
    assert last["behaviour"].abs().max("sigma_xx") < 1e-6


def _solid(fes, law):
    """A plane-stress solid: linear elastic, or perfectly plastic."""
    if law == "elasticity":
        return pc.model.elasticity(fes, "plane_stress")
    return pc.model.plasticity_perfect(fes, "plane_stress")


def _clamped_bar(law="elasticity"):
    """A strip held in x at both ends and in y at the bottom: heated, it can
    only build a uniform σ_xx = −E α ΔT."""
    mesh, fes, node = _strip()
    solid = _solid(fes, law)
    ends = [node(0, j) for j in range(NY + 1)] + [node(NX, j) for j in range(NY + 1)]
    bottom = [node(i, 0) for i in range(NX + 1)]
    model = solid | _fix(solid, ends, "u_x")[0] | _fix(solid, bottom, "u_y")[0]
    return mesh, model


def _uniform(mesh, component, value):
    field = pc.NodeField(mesh, [component])
    for zone in field:
        zone.add_to_component(component, value)
    return field


def test_a_clamped_bar_carries_the_thermal_stress_of_a_prescribed_temperature():
    mesh, model = _clamped_bar()
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    heating = pc.Evolution(
        [(0.0, _uniform(mesh, "T", T_REF)), (1.0, _uniform(mesh, "T", 220.0))]
    )
    data = {
        "model": model,
        "times": [0.0, 0.5, 1.0],
        "materials": materials,
        "temperature": heating,
        "t_ref": T_REF,
    }
    pc.stepping.step_by_step(data)
    for record, heated in zip(data["results"], (0.0, 100.0, 200.0), strict=True):
        sigma = record["behaviour"]
        if heated == 0.0:
            assert sigma is None
            continue
        assert sigma.max("sigma_xx") == pytest.approx(-E * ALPHA * heated, rel=1e-9)
        assert sigma.min("sigma_xx") == pytest.approx(-E * ALPHA * heated, rel=1e-9)
        assert sigma.abs().max("sigma_yy") < 1e-6


@pytest.mark.parametrize("law", ["elasticity", "plasticity"])
def test_moduli_depending_on_temperature_keep_the_stress_secant(law):
    """E(T) read from a law: the stress stays the secant one, −E(T) α (T − T_ref),
    at every step — for an elastic law and for a plastic one below its yield
    stress, each computing its stress from its elastic strain with the moduli
    of the end of the step."""
    mesh, model = _clamped_bar(law)
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA), ("sigma_y", 1e6)]
    )
    young = pc.Evolution(
        [(T_REF, E), (320.0, 0.5 * E)],
        abscissa_type="T",
        ordinate_type="E",
        out_of_range="extrapolate",
    )
    heating = pc.Evolution(
        [(0.0, _uniform(mesh, "T", T_REF)), (3.0, _uniform(mesh, "T", 320.0))]
    )
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0, 3.0],
        "materials": materials,
        "material_laws": young,
        "temperature": heating,
        "t_ref": T_REF,
    }
    pc.stepping.step_by_step(data)
    for record in data["results"][1:]:
        heated = 100.0 * record["time"]
        modulus = E * (1.0 - 0.5 * heated / 300.0)
        assert record["converged"]
        expected = -modulus * ALPHA * heated
        assert record["behaviour"].min("sigma_xx") == pytest.approx(expected, rel=1e-9)
        assert record["behaviour"].max("sigma_xx") == pytest.approx(expected, rel=1e-9)


def test_heating_then_pulling_equals_pulling_then_heating():
    """A secant law forgets the path: a bar heated (its modulus halving) then
    pulled ends with the stress of the bar pulled then heated,
    E(T)·(ε − α ΔT). An incremental law would carry the stress built under
    the other modulus."""
    young = pc.Evolution(
        [(T_REF, E), (220.0, 0.5 * E)],
        abscissa_type="T",
        ordinate_type="E",
        out_of_range="extrapolate",
    )
    finals = []
    for heat_first in (True, False):
        mesh, _fes, node, solid, model = _supported_strip()
        right = [node(NX, j) for j in range(NY + 1)]
        pull, support = _fix(solid, right, "u_x")
        model = model | pull
        materials = pc.element_field.material_field(
            model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
        )
        cold, hot = _uniform(mesh, "T", T_REF), _uniform(mesh, "T", 220.0)
        rest = _on(support, "imposed", 0.0)
        pulled = _on(support, "imposed", 3e-3 * L)
        if heat_first:
            temperature = pc.Evolution([(0.0, cold), (1.0, hot), (2.0, hot)])
            loads = pc.Evolution([(0.0, rest), (1.0, rest), (2.0, pulled)])
        else:
            temperature = pc.Evolution([(0.0, cold), (1.0, cold), (2.0, hot)])
            loads = pc.Evolution([(0.0, rest), (1.0, pulled), (2.0, pulled)])
        data = {
            "model": model,
            "times": [0.0, 0.5, 1.0, 1.5, 2.0],
            "materials": materials,
            "material_laws": young,
            "temperature": temperature,
            "loads": loads,
            "t_ref": T_REF,
        }
        pc.stepping.step_by_step(data)
        finals.append(data["results"][-1]["behaviour"].max("sigma_xx"))
    expected = 0.5 * E * (3e-3 - ALPHA * 200.0)
    assert finals[0] == pytest.approx(expected, rel=1e-9)
    assert finals[1] == pytest.approx(expected, rel=1e-9)


def test_an_elastoplastic_bar_pulled_by_its_end_yields_at_sigma_y():
    """Uniaxial tension under an imposed end displacement: σ = E ε up to
    σ_y, then σ = σ_y and the plastic strain takes the rest."""
    sigma_y = 200.0
    _mesh, _fes, node, solid, model = _supported_strip("plasticity")
    right = [node(NX, j) for j in range(NY + 1)]
    pull, pull_support = _fix(solid, right, "u_x")
    model = model | pull
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", sigma_y)]
    )
    stretch = pc.Evolution(
        [
            (0.0, _on(pull_support, "imposed", 0.0)),
            (1.0, _on(pull_support, "imposed", 4e-3 * L)),
        ]
    )
    data = {
        "model": model,
        "times": [0.0, 0.1, 0.2, 0.5, 1.0],
        "materials": materials,
        "loads": stretch,
    }
    pc.stepping.step_by_step(data)
    for record in data["results"][1:]:
        strain = 4e-3 * record["time"]
        stress = min(E * strain, sigma_y)
        assert record["converged"]
        # To the equilibrium precision (1e-4, as Cast3M's default).
        assert record["behaviour"].max("sigma_xx") == pytest.approx(stress, rel=1e-4)
        assert record["behaviour"].min("sigma_xx") == pytest.approx(stress, rel=1e-4)
        assert record["behaviour"].max("p") == pytest.approx(
            max(strain - sigma_y / E, 0.0), abs=1e-4 * strain
        )
        # The support pulls back with the force the section carries.
        held = sum(record["reactions"].value(node(0, j), "f_x") for j in range(NY + 1))
        assert held == pytest.approx(-stress * H, rel=1e-4)


def test_newmark_conserves_the_energy_of_a_free_vibration():
    """Average acceleration (γ = ½, β = ¼) conserves ½ vᵀMv + ½ uᵀKu exactly
    for a linear free vibration: every step keeps the kinetic energy given at
    the start."""
    mesh, _fes, node, solid, model = _supported_strip()
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("rho", 8e-9)]
    )
    data = {
        "model": model,
        "times": [k * 2e-6 for k in range(8)],
        "materials": materials,
        "dynamics": True,
        "initial": {"velocity": _ramp(mesh, node)},
    }
    pc.stepping.step_by_step(data)
    stiffness = pc.matrix.stiffness(solid, materials)
    mass = pc.matrix.mass(solid, materials)
    given = 0.5 * _quadratic(mass, _ramp(mesh, node))
    assert given > 0.0
    for record in data["results"][1:]:
        energy = 0.5 * _quadratic(mass, record["velocity"]) + 0.5 * _quadratic(
            stiffness, record["displacement"]
        )
        assert energy == pytest.approx(given, rel=1e-9)
    # And it does move: the displacement is not zero at the end.
    assert data["results"][-1]["displacement"].abs().max("u_x") > 0.0


def _ramp(mesh, node):
    """A velocity u̇_x = x (zero on the held edge), the shape of the first mode."""
    field = pc.NodeField(mesh, ["u_x", "u_y"])
    x = pc.node_field.positions(mesh, ["X"])
    for i in range(NX + 1):
        for j in range(NY + 1):
            field[0].set_value(node(i, j), "u_x", x.value(node(i, j), "X") * 1e3)
    return field


def _quadratic(matrix, field):
    """``xᵀ A x`` for a nodal ``field``: the product, brought back on the
    field's support, dotted with it."""
    product = matrix * field
    renamed = product.rename_component("f_x", "u_x").rename_component("f_y", "u_y")
    return pc.measure.xty(pc.node_field.restrict_like(renamed, field), field)
