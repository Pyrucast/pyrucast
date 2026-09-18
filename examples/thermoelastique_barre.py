"""Thermomécanique non couplée — barre chauffée (contraintes planes).

Physique
--------
An imposed temperature ΔT generates a free thermal strain
`ε_th = α·(T − T_ref)`. En petites déformations la rigidité reste élastique ;
the thermal term acts only on the right-hand side (equivalent thermal load
`f_th = ∫ Bᵀ D ε_th`) and on the real stress
`σ = D:(ε(u) − ε_th)`. Uncoupled: the mechanics does not feed back on the thermics.

Bricks composed by hand (no "all-in-one" operator):
`interp_to_gauss` (T nodale → points de Gauss), `thermal_strain` (EPTH),
`integrate_behavior` (σ = D:ε), `internal_forces` (BSIG), `solve`, puis
`deformation` and a field subtraction for ε_mech = ε(u) − ε_th.

`alpha` travels through the material field (`material_field`) as an
**optional** component of the elasticity, beside `E`/`nu`.

Two regimes on the same bar
------------------------------
- **bloquée** (deux bords en x encastrés) : `σ_xx = −E·α·ΔT`, `σ_yy = 0` ;
- **libre** (appuis simples) : dilatation `u = α·ΔT·(x, y)`, `σ ≈ 0`.

Lancement ::

    maturin develop --features extension-module
    python examples/thermoelastique_barre.py
"""

import pyrucast

E, NU, ALPHA = 210_000.0, 0.3, 1e-5
T_REF, DT = 20.0, 100.0
NX, NY, L, H = 4, 2, 4.0, 1.0


def _clamp(target, nodes, var):
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def _bar():
    """An NX×NY grid of QUA4 on [0,L]×[0,H]. Returns (coords, grid, fes, idx)."""
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
    """A temperature field 'T' = value everywhere, carried at the Gauss points."""
    t_mesh = pyrucast.Mesh(c, "POI1")
    for node in grid:
        t_mesh.unit().add_cell([node])
    t_nodal = pyrucast.NodeField(t_mesh, ["T"])
    for node in grid:
        t_nodal[0].set_value(node, "T", value)
    return pyrucast.element_field.interp_to_gauss(t_nodal, fes)


def _displacement(solution, c, grid):
    """Extracts a clean (u_x, u_y) field (without the Lagrange multipliers)."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in grid:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    for node in grid:
        u[0].set_value(node, "u_x", solution.value(node, "u_x"))
        u[0].set_value(node, "u_y", solution.value(node, "u_y"))
    return u


def _solve_thermal(model, materials, fes, c, grid):
    """ε_th → charge thermique → u → σ = D:(ε(u) − ε_th)."""
    eps_th = pyrucast.element_field.thermal_strain(
        _uniform_temperature(c, grid, fes, T_REF + DT), materials, fes, T_REF
    )
    sig_th = pyrucast.element_field.integrate_behavior(model, eps_th, materials)
    # A **load**, not a residual: the divergence of the prescribed tensor, hence
    # the geometric operator. They are then renamed into dual rows — that is
    # where, and only where, these numbers become forces.
    f_th = (
        pyrucast.node_field.divergence(sig_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
    )
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), f_th)
    u = _displacement(solution, c, grid)
    sigma = pyrucast.element_field.integrate_behavior(
        model, pyrucast.element_field.deformation(u, fes) - eps_th, materials
    )
    return u, sigma


def main() -> None:
    # ── Régime bloqué : σ_xx = −E·α·ΔT ──────────────────────────────────────
    c, grid, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = (
        model
        | _clamp(model, left, "u_x")
        | _clamp(model, right, "u_x")
        | _clamp(model, bottom, "u_y")
    )
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    _u, sigma = _solve_thermal(model, materials, fes, c, grid)
    expected = -E * ALPHA * DT
    sub = sigma[0]
    sxx = sub.value(0, 0, "sigma_xx")
    print(f"Bloquée : σ_xx = {sxx:12.4f}  (attendu {expected:.4f} = −E·α·ΔT)")
    assert abs(sxx - expected) < 1e-6 * abs(expected)

    # ── Régime libre : dilatation u = α·ΔT·(x, y), σ ≈ 0 ────────────────────
    c, grid, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = model | _clamp(model, left, "u_x") | _clamp(model, bottom, "u_y")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    u, sigma = _solve_thermal(model, materials, fes, c, grid)
    tip = grid[idx(NX, NY)]
    ux, uy = u.value(tip, "u_x"), u.value(tip, "u_y")
    print(
        f"Libre   : u(coin) = ({ux:.6e}, {uy:.6e})  (attendu ({ALPHA * DT * L:.6e}, {ALPHA * DT * H:.6e}))"
    )
    assert abs(ux - ALPHA * DT * L) < 1e-9 and abs(uy - ALPHA * DT * H) < 1e-9
    assert abs(sigma[0].value(0, 0, "sigma_xx")) < 1e-6

    print("\nOK: blocked bar → σ_xx = −E·α·ΔT; free bar → expansion without stress.")


if __name__ == "__main__":
    main()
