"""Python tests for uncoupled thermomechanics (thermal strain, Cast3M EPTH).

A prescribed temperature ΔT produces a thermal strain ε_th = α·ΔT. In small
strain the stiffness stays elastic; the thermal term only enters the load
(f_th = ∫ Bᵀ D ε_th) and the post-treated stress σ = D:(ε − ε_th). Two closed
forms on a plane-stress bar heated by ΔT:

- fully constrained in x (both ends clamped) ⇒ σ_xx = −E·α·ΔT, σ_yy = 0;
- free to expand (rollers only) ⇒ σ ≈ 0 and u = α·ΔT·(x, y).
"""

import pyrucast

E, NU, ALPHA = 210_000.0, 0.3, 1e-5
T_REF, DT = 20.0, 100.0
NX, NY, L, H = 4, 2, 4.0, 1.0


def _clamp(nodes, var, dual):
    imposed = pyrucast.poi1_from_nodes(nodes)
    multiplier = pyrucast.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


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
    return pyrucast.interp_to_gauss(t_nodal, fes)


def _thermal_load(model, materials, eps_th):
    """Equivalent nodal thermal load f_th = ∫ Bᵀ D ε_th (BSIG of σ_th = D:ε_th)."""
    sig_th = pyrucast.integrate_behavior(model, eps_th, materials)
    return pyrucast.internal_forces(model, sig_th)


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

    model = pyrucast.Model.elasticity(fes, "plane_stress")
    model = model | _clamp(left, "u_x", "f_x")
    model = model | _clamp(right, "u_x", "f_x")
    model = model | _clamp(bottom, "u_y", "f_y")

    materials = pyrucast.material_field(model, [("E", E), ("nu", NU), ("alpha", ALPHA)])
    eps_th = pyrucast.thermal_strain(
        _uniform_temperature(c, grid, fes, T_REF + DT), materials, fes, T_REF
    )

    f_th = _thermal_load(model, materials, eps_th)
    solution = pyrucast.solve(pyrucast.stiffness(model, materials), f_th)
    u = _displacement(solution, c, grid)

    # Real stress σ = D:(ε(u) − ε_th).
    sigma = pyrucast.integrate_behavior(
        model, pyrucast.deformation(u, fes) - eps_th, materials
    )
    expected = -E * ALPHA * DT
    for zone in range(len(sigma)):
        sub = sigma[zone]
        for g in range(sub.gauss_count()):
            for cell in range(sub.cell_count()):
                assert abs(sub.value(cell, g, "sigma_xx") - expected) < 1e-6 * abs(
                    expected
                )
                assert abs(sub.value(cell, g, "sigma_yy")) < 1e-6 * abs(expected)


def test_free_bar_expands_without_stress():
    """Rollers only (x on left, y on bottom): free dilation u = α·ΔT·(x, y), σ ≈ 0."""
    c, grid, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    model = pyrucast.Model.elasticity(fes, "plane_stress")
    model = model | _clamp(left, "u_x", "f_x")
    model = model | _clamp(bottom, "u_y", "f_y")

    materials = pyrucast.material_field(model, [("E", E), ("nu", NU), ("alpha", ALPHA)])
    eps_th = pyrucast.thermal_strain(
        _uniform_temperature(c, grid, fes, T_REF + DT), materials, fes, T_REF
    )

    f_th = _thermal_load(model, materials, eps_th)
    solution = pyrucast.solve(pyrucast.stiffness(model, materials), f_th)
    u = _displacement(solution, c, grid)

    hx, hy = L / NX, H / NY
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = i * hx, j * hy
            assert abs(u.value(grid[idx(i, j)], "u_x") - ALPHA * DT * x) < 1e-9
            assert abs(u.value(grid[idx(i, j)], "u_y") - ALPHA * DT * y) < 1e-9

    # Free expansion ⇒ (near) zero stress.
    sigma = pyrucast.integrate_behavior(
        model, pyrucast.deformation(u, fes) - eps_th, materials
    )
    for zone in range(len(sigma)):
        sub = sigma[zone]
        for g in range(sub.gauss_count()):
            for cell in range(sub.cell_count()):
                for comp in ("sigma_xx", "sigma_yy", "sigma_xy"):
                    assert abs(sub.value(cell, g, comp)) < 1e-6
