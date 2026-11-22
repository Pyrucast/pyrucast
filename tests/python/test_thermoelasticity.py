"""Python tests for uncoupled thermomechanics (thermal strain, Cast3M EPTH).

A prescribed temperature ΔT produces a thermal strain ε_th = α·ΔT. In small
strain the stiffness stays elastic; the thermal term only enters the load
(f_th = ∫ Bᵀ D ε_th) and the post-treated stress σ = D:(ε − ε_th). Two closed
forms on a plane-stress bar heated by ΔT:

- fully constrained in x (both ends clamped) ⇒ σ_xx = −E·α·ΔT, σ_yy = 0;
- free to expand (rollers only) ⇒ σ ≈ 0 and u = α·ΔT·(x, y).

Then the out-of-plane direction, on a unit QUA4 in plane strain with E = 1,
ν = 0.3 and α·ΔT = 1, computed by ``stepping.step_by_step``: the long body
holds its thickness, so its dilation along z is blocked and comes back in the
plane — ε_xx = (1+ν)/(1−ν) held in y, 1+ν free, with σ_zz the stress that
holds it.
"""

import pyrucast
import pytest

E, NU, ALPHA = 210_000.0, 0.3, 1e-5
T_REF, DT = 20.0, 100.0
NX, NY, L, H = 4, 2, 4.0, 1.0


def _clamp(target, nodes, var):
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    return pyrucast.model.dirichlet(imposed, target, var)


def _bar():
    """NX×NY grid of QUA4 over [0,L]×[0,H]. Returns (coords, grid, fes, idx)."""
    c = pyrucast.Coords(2)
    hx, hy = L / NX, H / NY

    def idx(i, j):
        return j * (NX + 1) + i

    grid = [c.add_node([i * hx, j * hy]) for j in range(NY + 1) for i in range(NX + 1)]
    mesh = pyrucast.Mesh(c, "QUA4")
    for j in range(NY):
        for i in range(NX):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    return c, grid, pyrucast.FiniteElementSpace(mesh), idx


def _uniform_temperature(c, grid, fes, value):
    """Per-element temperature field 'T' = value everywhere (nodal → Gauss)."""
    t_mesh = pyrucast.Mesh(c, "POI1")
    for node in grid:
        t_mesh.unit().add_cell([node])
    t_nodal = pyrucast.NodeField(t_mesh, ["T"])
    for node in grid:
        t_nodal[0].set_value(node, "T", value)
    return pyrucast.element_field.interp_to_gauss(t_nodal, fes)


def _thermal_load(model, materials, eps_th, fes):
    """Equivalent nodal thermal load f_th = ∫ Bᵀ D ε_th (BSIG of σ_th = D:ε_th).

    A **load**, not a residual: it applies the continuum `Bᵀ` to a prescribed
    stress, so it goes through the model-free operator rather than the one that
    forms `Σ f_int` from a solution.
    """
    sig_th = pyrucast.element_field.integrate_behavior(
        model, eps_th, materials, pyrucast.element_field.initial_state(model, materials)
    )
    # The tensor's divergence, renamed into dual rows: the operator is
    # geometric, this is where we say these are forces.
    return (
        pyrucast.node_field.divergence(sig_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
    )


def _displacement(solution, c, grid):
    """Clean (u_x, u_y) field over `grid`, dropped multiplier DOFs — `deformation`
    wants exactly space_dim components (the solve output also carries the Lagrange
    multipliers)."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in grid:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    for node in grid:
        u[0].set_value(node, "u_x", solution.value(node, "u_x"))
        u[0].set_value(node, "u_y", solution.value(node, "u_y"))
    return u


def test_fully_constrained_bar_thermal_stress():
    """Both x-ends clamped, heated by ΔT ⇒ σ_xx = −E·α·ΔT, σ_yy = 0."""
    c, grid, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = model | _clamp(model, left, "u_x")
    model = model | _clamp(model, right, "u_x")
    model = model | _clamp(model, bottom, "u_y")

    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    eps_th = pyrucast.element_field.thermal_strain(
        model, _uniform_temperature(c, grid, fes, T_REF + DT), materials, T_REF
    )

    f_th = _thermal_load(model, materials, eps_th, fes)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), f_th)
    u = _displacement(solution, c, grid)

    # Real stress σ = D:(ε(u) − ε_th).
    sigma = pyrucast.element_field.integrate_behavior(
        model,
        pyrucast.element_field.deformation(u, fes) - eps_th,
        materials,
        pyrucast.element_field.initial_state(model, materials),
    )
    expected = -E * ALPHA * DT
    for zone in range(len(sigma)):
        sub = sigma[zone]
        for g in range(sub.gauss_count):
            for cell in range(sub.cell_count):
                assert abs(sub.value(cell, g, "sigma_xx") - expected) < 1e-6 * abs(
                    expected
                )
                assert abs(sub.value(cell, g, "sigma_yy")) < 1e-6 * abs(expected)


def test_free_bar_expands_without_stress():
    """Rollers only (x on left, y on bottom): free dilation u = α·ΔT·(x, y), σ ≈ 0."""
    c, grid, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = model | _clamp(model, left, "u_x")
    model = model | _clamp(model, bottom, "u_y")

    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    eps_th = pyrucast.element_field.thermal_strain(
        model, _uniform_temperature(c, grid, fes, T_REF + DT), materials, T_REF
    )

    f_th = _thermal_load(model, materials, eps_th, fes)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), f_th)
    u = _displacement(solution, c, grid)

    hx, hy = L / NX, H / NY
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = i * hx, j * hy
            assert abs(u.value(grid[idx(i, j)], "u_x") - ALPHA * DT * x) < 1e-9
            assert abs(u.value(grid[idx(i, j)], "u_y") - ALPHA * DT * y) < 1e-9

    # Free expansion ⇒ (near) zero stress.
    sigma = pyrucast.element_field.integrate_behavior(
        model,
        pyrucast.element_field.deformation(u, fes) - eps_th,
        materials,
        pyrucast.element_field.initial_state(model, materials),
    )
    for zone in range(len(sigma)):
        sub = sigma[zone]
        for g in range(sub.gauss_count):
            for cell in range(sub.cell_count):
                for comp in ("sigma_xx", "sigma_yy", "sigma_xy"):
                    assert abs(sub.value(cell, g, comp)) < 1e-6


# ── Out of plane: plane strain holds the thickness ──────────────────────────

NU_Z = 0.3
#: hold → (u_x of the right side, {stress: value}), plane strain, E = α·ΔT = 1.
OUT_OF_PLANE = {
    "y": (
        (1 + NU_Z) / (1 - NU_Z),
        {
            "sigma_xx": 0.0,
            "sigma_yy": -1 / (1 - NU_Z),
            "sigma_zz": -1 / (1 - NU_Z),
            "sigma_xy": 0.0,
        },
    ),
    "free": (
        1 + NU_Z,
        {"sigma_xx": 0.0, "sigma_yy": 0.0, "sigma_zz": -1.0, "sigma_xy": 0.0},
    ),
}


def _unit_square(hold):
    """One QUA4 on [0, 1]² in plane strain, held in x on its left side and in y
    on its bottom (``hold="free"``) or on every node (``"y"``).
    Returns ``(mesh, nodes, model)``."""
    c = pyrucast.Coords(2)
    nodes = [c.add_node(xy) for xy in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(nodes)
    solid = pyrucast.model.elasticity(pyrucast.FiniteElementSpace(mesh), "plane_strain")
    left, bottom = [nodes[0], nodes[3]], [nodes[0], nodes[1]]
    held_y = bottom if hold == "free" else nodes
    model = solid | _clamp(solid, left, "u_x") | _clamp(solid, held_y, "u_y")
    return mesh, nodes, model


def _heated(mesh, value):
    field = pyrucast.NodeField(mesh, ["T"])
    for zone in field:
        zone.add_to_component("T", value)
    return field


def _check(u, stress, nodes, expected):
    u_x, sigma = expected
    for node in (nodes[1], nodes[2]):
        assert u.value(node, "u_x") == pytest.approx(u_x, rel=1e-12)
    for name, value in sigma.items():
        assert stress.max(name) == pytest.approx(value, abs=1e-12)
        assert stress.min(name) == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("hold", sorted(OUT_OF_PLANE))
def test_a_heated_plane_strain_body_dilates_out_of_plane(hold):
    mesh, nodes, model = _unit_square(hold)
    materials = pyrucast.element_field.material_field(
        model, [("E", 1.0), ("nu", NU_Z), ("alpha", 1.0)]
    )
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": materials,
        "temperature": pyrucast.Evolution(
            [(0.0, _heated(mesh, 0.0)), (1.0, _heated(mesh, 1.0))]
        ),
        "t_ref": 0.0,
    }
    pyrucast.stepping.step_by_step(data)
    last = data["results"][-1]
    _check(last["displacement"], last["behaviour"], nodes, OUT_OF_PLANE[hold])
