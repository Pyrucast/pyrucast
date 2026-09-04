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
- `solve` : la résolution linéaire (LU creux, factorisation en cache) ;
- l'**arithmétique de champs** (`+ - *`), l'**union** (`|`) et `restrict_like`
  (reprojection d'un champ sur le support/composantes d'un autre), qui
  remplacent toute boucle nodale : `residual = f_ext - f_int`, `u = u + du` ;
- une `Evolution` à valeur champ pour l'**histoire de chargement** : la charge
  de chaque pas est interpolée au pseudo-temps (`load_evo.interpolate(t)`).

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


def _plastic_diagnostics(state):
    """(p_max, nombre de points de Gauss plastifiés) — `p > 0` marque un point.

    Sans boucle : `p_max` par `max`, le comptage en masquant la composante `p`
    en 0/1 (bande « > 1e-12 ») puis en la sommant."""
    p_max = state.max("p")
    masked = state.mask(gt=1e-12, components=["p"])
    n_plastic = round(masked.sum("p"))
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
    print(
        f"Chargement : 0 → {p_max_load} en {nsteps} pas (Newton modifié, K élastique)\n"
    )

    # ── Maillage : bords SEG2 gauche/droit puis balayage en QUA4 ────────────
    c = pyrucast.Coords(2)
    pt_a = c.add_node([0.0, 0.0])
    pt_b = c.add_node([0.0, height])
    pt_c = c.add_node([length, 0.0])
    pt_d = c.add_node([length, height])
    left_edge = pyrucast.mesh.line(pt_a, pt_b, ny)
    right_edge = pyrucast.mesh.line(pt_c, pt_d, ny)
    mesh = pyrucast.mesh.sweep(left_edge, right_edge, nx)
    fes = pyrucast.FiniteElementSpace(mesh)

    # Nœud du bout (mi-hauteur) et maillage POI1 des nœuds LIBRES (X > 0) —
    # support cible pour la norme du résidu sur les seuls DDL libres.
    tip = mesh.nearest_node([length, height / 2.0])
    coords_field = pyrucast.node_field.positions(mesh, ["X"])
    free_mesh = pyrucast.mesh.select(coords_field, ge=length / nx / 2.0)
    imposed_mesh = pyrucast.mesh.to_poi1(left_edge)
    multiplier = pyrucast.mesh.translate(imposed_mesh, [0.0, 0.0])

    # ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    model = pyrucast.model.plasticity_perfect(fes, "plane_stress")
    model = model | pyrucast.model.dirichlet(model, "u_x", imposed_mesh, multiplier)
    model = model | pyrucast.model.dirichlet(model, "u_y", imposed_mesh, multiplier)

    # ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    #    droite, en efforts nodaux cohérents. C'est un terme du modèle : il le
    #    rejoint, sa densité rejoint le matériau. ─────────────────────────────
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    model = model | pyrucast.model.flux(right_fes, model, "f_y")
    materials = pyrucast.element_field.material_field(
        model, [("E", young), ("nu", nu), ("sigma_y", sigma_y), ("phi_f_y", -1.0)]
    )
    load_unit = pyrucast.node_field.external_forces(model, materials)

    # Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée une
    # fois ; `solve` met la factorisation en cache et la réutilise.
    k = pyrucast.matrix.stiffness(model, materials)

    # ── Histoire de chargement : une Evolution à valeur CHAMP, tabulée en
    #    pseudo-temps t ∈ [0, 1]. Deux keyframes du champ d'effort nodal — nul en
    #    t=0, complet (`p_max · charge_unitaire`) en t=1 — sur le MÊME support.
    #    La charge de chaque pas est lue par interpolation linéaire. ────────────
    zero_frame = load_unit * 0.0
    full_frame = load_unit * p_max_load
    load_evo = pyrucast.Evolution(
        [(0.0, zero_frame), (1.0, full_frame)], out_of_range="clamp"
    )

    # ── État de la simulation (persistant entre les pas) ────────────────────
    u = pyrucast.NodeField(mesh, ["u_x", "u_y"])  # déplacement cumulé, nul au départ
    # L'état au repos : `None` au premier pas, où A est la configuration de
    # référence. L'opérateur le matérialise lui-même, avec **toutes** les
    # composantes que la loi relit ensuite — σ(A), ε(A), l'état interne — là où
    # une liste écrite à la main en oublie.
    state = None

    # ── Boucle sur les pas de charge ────────────────────────────────────────
    # Newton modifié (opérateur = K élastique) : convergence linéaire, donc lente
    # sur la branche plastique. Plafond d'itérations haut, résidu relatif 1e-6.
    max_newton = 200
    print(
        f"{'pas':>4} {'P':>8} {'iter':>6} {'flèche u_y':>14} {'p_max':>14} {'n_plast':>8}"
    )

    prev_defl = 0.0
    any_plasticity = False

    for step in range(1, nsteps + 1):
        # Pseudo-temps du pas ∈ ]0, 1] ; la charge externe en découle par
        # interpolation de l'Evolution (champ d'effort nodal du pas).
        t = step / nsteps
        load_p = p_max_load * t  # cisaillement nominal au bout (pour l'affichage)
        load_scaled = load_evo.interpolate(t)
        # Norme de la charge du pas (échelle relative du résidu) : xᵀx du champ.
        ext_norm = pyrucast.measure.xtx(load_scaled) ** 0.5
        tol = 1e-6 * ext_norm + 1e-12

        iters = 0
        last_out = None
        res_norm = float("inf")

        for _ in range(max_newton):
            # ε(u) → entrée de comportement (ε | VAR0) → σ, VAR1 (COMP).
            strain = pyrucast.element_field.deformation(u, fes)
            out = pyrucast.element_field.integrate_behavior(
                model, strain, materials, prev=state
            )
            # Forces internes F_int = ∫ Bᵀ σ dΩ (BSIG).
            f_int = pyrucast.node_field.internal_forces(model, out, u, materials)

            # Résidu r = F_ext − F_int et sa norme sur les DDL **libres**, sans
            # aucune boucle nodale — tout par les opérateurs et primitives :
            # - `f_ext` = charge externe du pas reprojetée sur le support ET les
            #   composantes de `f_int` (`restrict_like`) : `f_x` (=0) et `f_y` ;
            # - `residual = f_ext − f_int` via l'opérateur `-` ;
            # - la norme se lit sur les seuls nœuds libres : `residual` `restrict`é
            #   à `free_mesh` puis `xtx` (les nœuds encastrés portent la réaction).
            f_ext = pyrucast.node_field.restrict_like(load_scaled, f_int)
            residual = f_ext - f_int
            res_norm = (
                pyrucast.measure.xtx(pyrucast.node_field.restrict(residual, free_mesh))
                ** 0.5
            )
            last_out = out

            if res_norm <= tol:
                break
            # δu = K⁻¹ r (K élastique, factorisation en cache). δu porte les DDL
            # primaux ET duaux (multiplicateurs). Son support coïncide déjà avec
            # celui de u (même compagnon POI1 caché de `to_poi1`, partagé par `solve`
            # et `NodeField(mesh)`) ; `restrict_like` ne sert qu'à filtrer les
            # composantes duales — sinon `u + δu` recopierait les multiplicateurs
            # dans u par union. Puis u ← u + δu.
            du = pyrucast.solver.solve(k, residual)
            u = u + pyrucast.node_field.restrict_like(du, u)
            iters += 1

        converged = res_norm <= tol

        # Commit de l'état : VAR0 ← VAR1. La sortie de comportement convergée
        # porte, en plus de l'état plastique (`eps_p_*`, `p`), les contraintes
        # (`sig_*`) ; on la reporte telle quelle comme nouveau VAR0. La loi lit
        # ses entrées par nom, donc les composantes surnuméraires sont ignorées.
        state = last_out

        # Diagnostics du pas.
        p_max_val, n_plastic = _plastic_diagnostics(state)
        defl = u.value(tip, "u_y")
        any_plasticity = any_plasticity or n_plastic > 0
        flag = "" if converged else "  (résidu résiduel)"
        print(
            f"{step:>4} {load_p:>8.3f} {iters:>6} {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        )

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
        print(
            f"\nOK : plastification développée (P_max={p_max_load} > P_élastique≈{p_first_yield:.2f})."
        )
    else:
        print(
            f"\nOK : réponse restée élastique (P_max={p_max_load} ≤ ≈{p_first_yield:.2f})."
        )


if __name__ == "__main__":
    main()
