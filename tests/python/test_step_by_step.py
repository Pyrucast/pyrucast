"""Tests de la couche Python haut niveau ``step_by_step`` (thermo-mécanique).

``step_by_step`` splits the model per physics, solves at each step the thermics
(stationnaire) puis la mécanique non linéaire (Newton modifié + Anderson), le
couplage étant faible (température → déformation thermique).

Cas de validation : plaque plane chauffée uniformément et libre de se dilater
(appuis simples). Solution analytique fermée :

* thermics — imposed temperature on the left/right edges at ``T_HOT`` ⇒ a
  **uniform** field ``T = T_HOT`` (no source, conduction);
* mechanics — free expansion ``u = α·ΔT·(x, y)`` with ``ΔT = T_HOT − T_REF`` and
  contrainte quasi nulle.
"""

import pyrucast as pc

E, NU, ALPHA = 210_000.0, 0.3, 1e-5
K_COND = 1.0
T_REF, T_HOT = 20.0, 120.0  # ΔT = 100
NX, NY, L, H = 4, 2, 4.0, 1.0


def _clamp(target, nodes, var):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multiplier = pc.mesh.barycenter(imposed)
    return pc.model.dirichlet(target, var, imposed, multiplier)


def _bar():
    """An NX×NY grid of QUA4 on [0,L]×[0,H]. Returns (coords, grid, mesh, fes, idx)."""
    c = pc.Coords(2)
    hx, hy = L / NX, H / NY

    def idx(i, j):
        return j * (NX + 1) + i

    grid = [c.add_node([i * hx, j * hy]) for j in range(NY + 1) for i in range(NX + 1)]
    mesh = pc.Mesh(c, "QUA4")
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
    return c, grid, mesh, pc.FiniteElementSpace(mesh), idx


def test_step_by_step_free_thermal_expansion():
    c, grid, mesh, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    # Thermics: every left+right node pinned at T_HOT (one multiplier per node
    # through `translate` ⇒ a uniform temperature, not merely on average).
    th_nodes = left + right
    th_imposed = pc.mesh.poi1_from_nodes(th_nodes)
    th_mult = pc.mesh.translate(th_imposed, [0.0, 0.0])
    thermal = pc.model.heat_conduction(fes)
    thermal_dir = pc.model.dirichlet(thermal, "T", th_imposed, th_mult)

    # Modèle complet : conduction + élasticité (contraintes planes) + Dirichlet.
    # Mechanics: simple supports (u_x=0 on the left, u_y=0 at the bottom, value 0
    # charge) ⇒ dilatation libre.
    mecanique = pc.model.elasticity(fes, "plane_stress")
    model = (
        thermal
        | mecanique
        | thermal_dir
        | _clamp(mecanique, left, "u_x")
        | _clamp(mecanique, bottom, "u_y")
    )

    materials = pc.element_field.material_field(
        model, [("k", K_COND), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # Unioned load: imposed_T = T_HOT on every thermal multiplier
    # (aucune charge mécanique — dilatation libre).
    loads = pc.NodeField(th_mult, ["imposed_T"])
    loads[0].add_to_component("imposed_T", T_HOT)

    # fespace and mesh are deduced from the model: only `model` is required.
    data = {
        "times": [0.0, 1.0],
        "model": model,
        "loads": loads,
        "materials": materials,
        "t_ref": T_REF,
    }

    out = pc.thermomechanics.step_by_step(data)

    # The same dictionary is returned, filled in.
    assert out is data
    results = data["results"]
    assert len(results) == 2

    last = results[-1]
    assert last["converged"], f"non convergé : {last}"

    # Thermique : température uniforme T_HOT.
    temperature = last["temperature"]
    for node in grid:
        assert abs(temperature.value(node, "T") - T_HOT) < 1e-9

    # Mécanique : dilatation libre u = α·ΔT·(x, y).
    displacement = last["displacement"]
    dT = T_HOT - T_REF
    hx, hy = L / NX, H / NY
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = i * hx, j * hy
            node = grid[idx(i, j)]
            assert abs(displacement.value(node, "u_x") - ALPHA * dT * x) < 1e-7
            assert abs(displacement.value(node, "u_y") - ALPHA * dT * y) < 1e-7

    # Near-zero stress (free expansion). The materials keep their two zones
    # (thermal + mechanical): every operator resolves its own by component,
    # without consolidation.
    sigma = pc.element_field.integrate_behavior(
        model.filter("mechanical"),
        pc.element_field.deformation(displacement, fes)
        - pc.element_field.thermal_strain(
            pc.element_field.interp_to_gauss(
                pc.node_field.restrict(temperature, mesh), fes
            ),
            materials,
            fes,
            T_REF,
        ),
        materials,
    )
    for zone in range(len(sigma)):
        sub = sigma[zone]
        for g in range(sub.gauss_count()):
            for cell in range(sub.cell_count()):
                for comp in ("sigma_xx", "sigma_yy", "sigma_xy"):
                    assert abs(sub.value(cell, g, comp)) < 1e-4


def test_step_by_step_returns_history_per_time():
    """The result list has one item per instant, in order."""
    c, grid, mesh, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    th_imposed = pc.mesh.poi1_from_nodes(
        left + [grid[idx(NX, j)] for j in range(NY + 1)]
    )
    th_mult = pc.mesh.translate(th_imposed, [0.0, 0.0])
    thermal = pc.model.heat_conduction(fes)
    mecanique = pc.model.elasticity(fes, "plane_stress")
    model = (
        thermal
        | mecanique
        | pc.model.dirichlet(thermal, "T", th_imposed, th_mult)
        | _clamp(mecanique, left, "u_x")
        | _clamp(mecanique, bottom, "u_y")
    )
    materials = pc.element_field.material_field(
        model, [("k", K_COND), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # Temperature history rising from T_REF to T_HOT (a field-valued Evolution).
    cold = pc.NodeField(th_mult, ["imposed_T"])
    cold[0].add_to_component("imposed_T", T_REF)
    hot = pc.NodeField(th_mult, ["imposed_T"])
    hot[0].add_to_component("imposed_T", T_HOT)
    loads = pc.Evolution([(0.0, cold), (1.0, hot)], out_of_range="clamp")

    times = [0.0, 0.5, 1.0]
    data = {
        "times": times,
        "model": model,
        "loads": loads,
        "materials": materials,
        "t_ref": T_REF,
    }
    pc.thermomechanics.step_by_step(data)

    results = data["results"]
    assert [r["time"] for r in results] == times

    # The expansion's deflection grows with the imposed temperature (monotone).
    tip = grid[idx(NX, NY)]
    prev = -1.0
    for r in results:
        ux = r["displacement"].value(tip, "u_x")
        assert ux >= prev - 1e-12
        prev = ux


def test_thermal_step_conducts_linearly_between_two_imposed_edges():
    """Steady conduction, T_REF imposed on the left edge and T_HOT on the right:
    no source, so T is linear in x — T = T_REF + (T_HOT − T_REF)·x/L."""
    c, grid, mesh, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]

    thermal = pc.model.heat_conduction(fes)
    model, loads = thermal, None
    for edge, value in ((left, T_REF), (right, T_HOT)):
        imposed = pc.mesh.poi1_from_nodes(edge)
        multiplier = pc.mesh.translate(imposed, [0.0, 0.0])
        model = model | pc.model.dirichlet(thermal, "T", imposed, multiplier)
        load = pc.NodeField(multiplier, ["imposed_T"])
        load[0].add_to_component("imposed_T", value)
        loads = load if loads is None else loads | load
    materials = pc.element_field.material_field(model, [("k", K_COND)])

    temperature = pc.thermomechanics.thermal_step(model, materials, loads)
    hx = L / NX
    for j in range(NY + 1):
        for i in range(NX + 1):
            expected = T_REF + (T_HOT - T_REF) * i * hx / L
            assert abs(temperature.value(grid[idx(i, j)], "T") - expected) < 1e-9


def test_mechanical_step_clamped_bar_carries_the_thermal_stress():
    """Both x-ends clamped, heated uniformly by ΔT: σ_xx = −E·α·ΔT, σ_yy = 0
    (plane stress, free in y)."""
    c, grid, mesh, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    mecanique = pc.model.elasticity(fes, "plane_stress")
    model = (
        mecanique
        | _clamp(mecanique, left, "u_x")
        | _clamp(mecanique, right, "u_x")
        | _clamp(mecanique, bottom, "u_y")
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    temperature = pc.NodeField(pc.mesh.poi1_from_nodes(grid), ["T"])
    temperature[0].add_to_component("T", T_HOT)
    # The residual norm is taken on the free nodes: on a clamped one it is the
    # support reaction, which equilibrium does not drive to zero.
    clamped = {n.id for n in left + right + bottom}
    free = pc.mesh.poi1_from_nodes([n for n in grid if n.id not in clamped])

    u, state, info = pc.thermomechanics.mechanical_step(
        model,
        fes,
        mesh,
        materials,
        pc.NodeField(mesh, ["f_x", "f_y"]),
        temperature,
        pc.NodeField(mesh, ["u_x", "u_y"]),
        None,
        1.0,
        t_ref=T_REF,
        free_mesh=free,
    )
    assert info["converged"], info
    expected = -E * ALPHA * (T_HOT - T_REF)
    sub = state[0]
    for cell in range(sub.cell_count()):
        for g in range(sub.gauss_count()):
            assert abs(sub.value(cell, g, "sigma_xx") - expected) < 1e-6
            assert abs(sub.value(cell, g, "sigma_yy")) < 1e-6
