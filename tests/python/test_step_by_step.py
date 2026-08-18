"""Tests de la couche Python haut niveau ``step_by_step`` (thermo-mécanique).

``step_by_step`` découpe le modèle par physique, résout à chaque pas la thermique
(stationnaire) puis la mécanique non linéaire (Newton modifié + Anderson), le
couplage étant faible (température → déformation thermique).

Cas de validation : plaque plane chauffée uniformément et libre de se dilater
(appuis simples). Solution analytique fermée :

* thermique — température imposée sur les bords gauche/droit à ``T_HOT`` ⇒ champ
  **uniforme** ``T = T_HOT`` (pas de source, conduction) ;
* mécanique — dilatation libre ``u = α·ΔT·(x, y)`` avec ``ΔT = T_HOT − T_REF`` et
  contrainte quasi nulle.
"""

import pyrucast as pc

E, NU, ALPHA = 210_000.0, 0.3, 1e-5
K_COND = 1.0
T_REF, T_HOT = 20.0, 120.0  # ΔT = 100
NX, NY, L, H = 4, 2, 4.0, 1.0


def _clamp(nodes, var, dual):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multiplier = pc.mesh.barycenter(imposed)
    return pc.Model.dirichlet(var, dual, imposed, multiplier)


def _bar():
    """Grille NX×NY de QUA4 sur [0,L]×[0,H]. Renvoie (coords, grid, mesh, fes, idx)."""
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

    # Thermique : chaque nœud gauche+droit épinglé à T_HOT (un multiplicateur par
    # nœud via `translate` ⇒ température uniforme, pas seulement en moyenne).
    th_nodes = left + right
    th_imposed = pc.mesh.poi1_from_nodes(th_nodes)
    th_mult = pc.mesh.translate(th_imposed, [0.0, 0.0])
    thermal_dir = pc.Model.dirichlet("T", "q", th_imposed, th_mult)

    # Modèle complet : conduction + élasticité (contraintes planes) + Dirichlet.
    # Mécanique : appuis simples (u_x=0 à gauche, u_y=0 en bas, valeur 0 ⇒ pas de
    # charge) ⇒ dilatation libre.
    model = (
        pc.Model.heat_conduction(fes)
        | pc.Model.elasticity(fes, "plane_stress")
        | thermal_dir
        | _clamp(left, "u_x", "f_x")
        | _clamp(bottom, "u_y", "f_y")
    )

    materials = pc.element_field.material_field(
        model, [("k", K_COND), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # Charge unionée : imposed_T = T_HOT sur tous les multiplicateurs thermiques
    # (aucune charge mécanique — dilatation libre).
    loads = pc.NodeField(th_mult, ["imposed_T"])
    loads[0].add_to_component("imposed_T", T_HOT)

    # fespace et maillage sont déduits du modèle : seul `model` est requis.
    data = {
        "times": [0.0, 1.0],
        "model": model,
        "loads": loads,
        "materials": materials,
        "t_ref": T_REF,
    }

    out = pc.thermomechanics.step_by_step(data)

    # Le même dictionnaire est renvoyé, complété.
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

    # Contrainte quasi nulle (dilatation libre). Les matériaux gardent leurs deux
    # zones (thermique + mécanique) : chaque opérateur résout la sienne par
    # composante, sans consolidation.
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
    """La liste des résultats a un élément par instant, dans l'ordre."""
    c, grid, mesh, fes, idx = _bar()
    left = [grid[idx(0, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    th_imposed = pc.mesh.poi1_from_nodes(
        left + [grid[idx(NX, j)] for j in range(NY + 1)]
    )
    th_mult = pc.mesh.translate(th_imposed, [0.0, 0.0])
    model = (
        pc.Model.heat_conduction(fes)
        | pc.Model.elasticity(fes, "plane_stress")
        | pc.Model.dirichlet("T", "q", th_imposed, th_mult)
        | _clamp(left, "u_x", "f_x")
        | _clamp(bottom, "u_y", "f_y")
    )
    materials = pc.element_field.material_field(
        model, [("k", K_COND), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # Histoire de température montant de T_REF à T_HOT (Evolution à valeur champ).
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

    # La flèche de dilatation croît avec la température imposée (monotone).
    tip = grid[idx(NX, NY)]
    prev = -1.0
    for r in results:
        ux = r["displacement"].value(tip, "u_x")
        assert ux >= prev - 1e-12
        prev = ux
