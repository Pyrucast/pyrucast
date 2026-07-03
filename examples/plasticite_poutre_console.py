"""Poutre console élasto-plastique — Newton « maison » au-dessus de pyrucast.

Version Python de `examples/plasticite_poutre_console.rs` (mêmes briques, même
algorithme). Elle privilégie la lisibilité ; pour le **bench de parallélisme**,
préférer l'exemple Rust (pur, sans surcoût interpréteur).

Physique
--------
Continuum 2-D en contraintes planes, petites déformations. Plasticité de von
Mises parfaite (retour radial J2, sans écrouissage) : la contrainte équivalente
est plafonnée à `sigma_y`. Poutre encastrée à gauche (`u_x = u_y = 0`), cisaillée
vers le bas sur la face droite. On monte la charge par incréments ; au-delà de la
première plastification, une zone plastique se développe près de l'encastrement
et la flèche s'écarte de la réponse linéaire.

Rôle de pyrucast vs. rôle de l'exemple
--------------------------------------
pyrucast ne connaît PAS Newton. Il fournit les opérateurs ponctuels :

- `stiffness` : la rigidité **élastique** `K` (opérateur d'itération) ;
- `deformation` : la déformation `ε = ½(∇u + ∇uᵀ)` aux points de Gauss ;
- `integrate_behavior` (Cast3m `COMP`) : la loi au point — retour radial, qui
  rend `σ` et l'état plastique mis à jour (`VAR0` → `VAR1`) ;
- `internal_forces` (Cast3m `BSIG`) : les forces internes `∫ Bᵀ σ dΩ` ;
- `solve` : la résolution linéaire (LU creux, factorisation en cache).

L'exemple assemble sa propre boucle de Newton : résidu `r = F_ext − F_int`,
incrément `δu = K⁻¹ r`, `u ← u + δu`, et le portage de l'état interne d'un pas
au suivant. C'est un **Newton modifié** (opérateur constant = `K` élastique) :
`K` est assemblé et factorisé une seule fois.

Lancement ::

    maturin develop --release
    python examples/plasticite_poutre_console.py

Variables d'environnement : `PYRUCAST_NX`, `PYRUCAST_NY` (mailles en long / en
hauteur), `PYRUCAST_NSTEPS` (pas de charge), `PYRUCAST_PMAX` (charge finale).
"""

import os

import pyrucast

# Composantes de l'état interne plastique portées d'un pas au suivant (VAR) :
# déformation plastique 3-D (tenseur, 6) + déformation plastique cumulée `p`.
STATE_COMPONENTS = [
    "eps_p_xx", "eps_p_yy", "eps_p_zz", "eps_p_yz", "eps_p_xz", "eps_p_xy", "p",
]


def _clamp(nodes, variable, dual):
    """Sous-modèle Dirichlet encastrant `variable` (dual `dual`) sur `nodes`."""
    imposed = pyrucast.poi1_from_nodes(nodes)
    multiplier = pyrucast.barycenter(imposed)
    return pyrucast.Model.dirichlet(variable, dual, imposed, multiplier)


def _behavior_input(strain, state, fes):
    """Fusionne, point de Gauss par point, la déformation totale `ε` et l'état
    plastique `VAR0` en une entrée pour `integrate_behavior`."""
    strain_sub, state_sub = strain[0], state[0]
    strain_comps = strain_sub.components()
    state_comps = state_sub.components()
    inp = pyrucast.ElementField(fes, strain_comps + state_comps)
    inp_sub = inp[0]
    for cell in range(inp_sub.cell_count()):
        for g in range(inp_sub.gauss_count()):
            for name in strain_comps:
                inp_sub[cell, g, name] = strain_sub[cell, g, name]
            for name in state_comps:
                inp_sub[cell, g, name] = state_sub[cell, g, name]
    return inp


def _extract_state(out, fes):
    """Nouvel état plastique `VAR1` (les `STATE_COMPONENTS`) extrait de la sortie
    de comportement convergée — le `VAR0` du pas suivant."""
    out_sub = out[0]
    state = pyrucast.ElementField(fes, STATE_COMPONENTS)
    state_sub = state[0]
    for cell in range(out_sub.cell_count()):
        for g in range(out_sub.gauss_count()):
            for name in STATE_COMPONENTS:
                state_sub[cell, g, name] = out_sub[cell, g, name]
    return state


def _plastic_diagnostics(state):
    """(p_max, nombre de points de Gauss plastifiés) — `p > 0` marque un point."""
    sub = state[0]
    p_max, n_plastic = 0.0, 0
    for cell in range(sub.cell_count()):
        for g in range(sub.gauss_count()):
            p = sub[cell, g, "p"]
            if p > 1e-12:
                n_plastic += 1
            p_max = max(p_max, p)
    return p_max, n_plastic


def main():
    # ── Paramètres (matériau acier, géométrie, chargement) ──────────────────
    young, nu, sigma_y = 210_000.0, 0.3, 250.0
    length, height = 10.0, 1.0
    nx = int(os.environ.get("PYRUCAST_NX", 24))
    ny = int(os.environ.get("PYRUCAST_NY", 6))
    nsteps = int(os.environ.get("PYRUCAST_NSTEPS", 10))
    p_max_load = float(os.environ.get("PYRUCAST_PMAX", 5.0))

    print(
        f"Poutre console plastique : {nx}×{ny} QUA4  (L={length}, H={height}), "
        f"E={young}, ν={nu}, σy={sigma_y}"
    )
    print(f"Chargement : 0 → {p_max_load} en {nsteps} pas (Newton modifié, K élastique)\n")

    # ── Maillage : grille de nœuds (j en hauteur, i en long), cellules QUA4 ──
    c = pyrucast.Coords(2)
    grid = [
        [c.add_node([i * length / nx, j * height / ny]) for i in range(nx + 1)]
        for j in range(ny + 1)
    ]
    mesh = pyrucast.Mesh(c, "QUA4")
    cells = mesh.unit()
    for j in range(ny):
        for i in range(nx):
            cells.add_cell([
                grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i],
            ])
    fes = pyrucast.FiniteElementSpace(mesh)

    # Ensembles de nœuds : bord gauche (encastré), bout (mi-hauteur).
    left_nodes = [grid[j][0] for j in range(ny + 1)]
    right_nodes = [grid[j][nx] for j in range(ny + 1)]
    tip = grid[ny // 2][nx]

    # ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    model = pyrucast.Model.plasticity(fes, "plane_stress")
    model = model | _clamp(left_nodes, "u_x", "f_x")
    model = model | _clamp(left_nodes, "u_y", "f_y")
    materials = pyrucast.material_field(
        model, [("E", young), ("nu", nu), ("sigma_y", sigma_y)]
    )

    # Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée une
    # fois ; `solve` met la factorisation en cache et la réutilise.
    k = pyrucast.stiffness(model, materials)

    # ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    #    droite, réparti en efforts nodaux cohérents (op `flux`). ──────────────
    right_edge = pyrucast.Mesh(c, "SEG2")
    edge_cells = right_edge.unit()
    for j in range(ny):
        edge_cells.add_cell([grid[j][nx], grid[j + 1][nx]])
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    load_unit = pyrucast.flux(right_fes[0], -1.0, "f_y")
    load_unit_norm = sum(load_unit.value(n, "f_y") ** 2 for n in right_nodes) ** 0.5

    # ── État de la simulation (persistant entre les pas) ────────────────────
    u = pyrucast.NodeField(mesh, ["u_x", "u_y"])   # déplacement cumulé, nul au départ
    u_sub = u[0]
    state = pyrucast.ElementField(fes, STATE_COMPONENTS)  # VAR0, nul au premier pas

    # ── Boucle sur les pas de charge ────────────────────────────────────────
    # Newton modifié (opérateur = K élastique) : convergence linéaire, donc lente
    # sur la branche plastique. Plafond d'itérations haut, résidu relatif 1e-6.
    max_newton = 200
    print(f"{'pas':>4} {'P':>8} {'iter':>6} {'flèche u_y':>14} {'p_max':>14} {'n_plast':>8}")

    prev_defl = 0.0
    any_plasticity = False

    for step in range(1, nsteps + 1):
        load_p = p_max_load * step / nsteps
        factor = load_p  # densité −1 ⇒ effort total = −load_p (vers le bas)
        ext_norm = abs(factor) * load_unit_norm
        tol = 1e-6 * ext_norm + 1e-12

        iters = 0
        last_out = None
        res_norm = float("inf")

        for _ in range(max_newton):
            # ε(u) → entrée de comportement (ε + VAR0) → σ, VAR1 (COMP).
            strain = pyrucast.deformation(u, fes)
            out = pyrucast.integrate_behavior(model, _behavior_input(strain, state, fes), materials)
            # Forces internes F_int = ∫ Bᵀ σ dΩ (BSIG).
            f_int = pyrucast.internal_forces(model, out)

            # Résidu r = F_ext − F_int (F_ext = facteur · charge sur la face droite),
            # et sa norme sur les DDL LIBRES (les nœuds encastrés portent la réaction).
            residual = pyrucast.NodeField(mesh, ["f_x", "f_y"])
            r_sub = residual[0]
            free_sq = 0.0
            for j in range(ny + 1):
                for i in range(nx + 1):
                    node = grid[j][i]
                    rx = -f_int.value(node, "f_x")
                    fext_y = factor * load_unit.value(node, "f_y") if i == nx else 0.0
                    ry = fext_y - f_int.value(node, "f_y")
                    r_sub[node, "f_x"] = rx
                    r_sub[node, "f_y"] = ry
                    if i != 0:
                        free_sq += rx * rx + ry * ry
            res_norm = free_sq ** 0.5
            last_out = out

            if res_norm <= tol:
                break
            # δu = K⁻¹ r (K élastique, factorisation en cache), puis u ← u + δu.
            du = pyrucast.solve(k, residual)
            for row in grid:
                for node in row:
                    u_sub[node, "u_x"] = u_sub[node, "u_x"] + du.value(node, "u_x")
                    u_sub[node, "u_y"] = u_sub[node, "u_y"] + du.value(node, "u_y")
            iters += 1

        converged = res_norm <= tol

        # Commit de l'état : VAR0 ← VAR1 (extrait de la sortie convergée).
        state = _extract_state(last_out, fes)

        # Diagnostics du pas.
        p_max_val, n_plastic = _plastic_diagnostics(state)
        defl = u.value(tip, "u_y")
        any_plasticity = any_plasticity or n_plastic > 0
        flag = "" if converged else "  (résidu résiduel)"
        print(f"{step:>4} {load_p:>8.3f} {iters:>6} {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}")

        # La flèche croît (en valeur absolue, vers le bas) avec la charge.
        assert abs(defl) >= abs(prev_defl) - 1e-9, f"flèche non monotone au pas {step}"
        prev_defl = defl

    # Au-delà de la première plastification, une zone plastique doit apparaître.
    p_first_yield = sigma_y * (height * height / 6.0) / length
    if p_max_load > p_first_yield:
        assert any_plasticity, (
            f"P_max={p_max_load} dépasse la première plastification "
            f"(≈{p_first_yield:.2f}) mais aucun point plastique détecté"
        )
        print(f"\nOK : plastification développée (P_max={p_max_load} > P_élastique≈{p_first_yield:.2f}).")
    else:
        print(f"\nOK : réponse restée élastique (P_max={p_max_load} ≤ ≈{p_first_yield:.2f}).")


if __name__ == "__main__":
    main()
