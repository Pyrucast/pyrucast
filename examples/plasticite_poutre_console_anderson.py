"""Poutre console élasto-plastique — Newton modifié **accéléré par
l'accélération d'Anderson (m = 3)** au-dessus de pyrucast.

Version Python de `examples/plasticite_poutre_console_anderson.rs` (mêmes briques,
même algorithme). Variante de `plasticite_poutre_console.py` : même physique, même
maillage, mêmes opérateurs pyrucast. Seule la boucle non linéaire change — on garde
l'original **gelé** comme référence pour comparer les résultats (les flèches et la
plasticité doivent coïncider ; seul le nombre d'itérations doit chuter).

Newton modifié = point fixe préconditionné
------------------------------------------
Comme dans l'exemple d'origine, l'opérateur d'itération est la rigidité
**élastique** `K` (assemblée + factorisée une fois, cache de `solve`). L'itération
`u ← u + K⁻¹r(u)` est un **point fixe préconditionné** : la « direction résidu »
`g(u) = K⁻¹ r(u)` s'annule à convergence — c'est le résidu naturel du point fixe,
**déjà calculé** à chaque itération (`du = solve(k, residual)`).

Le prix de l'opérateur constant est une convergence seulement **linéaire** sur la
branche plastique (beaucoup d'itérations). L'accélération d'Anderson exploite
l'historique des `m = 3` derniers couples `(u, g)` pour extrapoler un pas bien
meilleur, **sans réévaluer la loi de comportement** : le petit moindre-carré ne
manipule que des produits scalaires de champs déjà en main.

Accélération d'Anderson (m = 3)
-------------------------------
À l'itération `k`, avec l'historique des `m ≤ 3` derniers `(uᵢ, gᵢ)` (le plus
récent en tête) :

1. Différences `ΔGⱼ = g − g_hist`, `ΔUⱼ = u − u_hist`.
2. Moindre-carré `min_γ ‖g − Σⱼ γⱼ ΔGⱼ‖²` → équations normales `(ΔGᵀΔG) γ =
   ΔGᵀg`, dont toutes les entrées sont des produits scalaires **sur les DDL
   libres** (mêmes DDL que la norme du résidu), régularisées façon Tikhonov.
3. Petit solve dense `m×m` (m ≤ 3, élimination de Gauss).
4. Pas extrapolé `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.

Garde-fou de descente : on évalue le résidu du candidat d'Anderson et on ne le
retient que s'il réduit **strictement** le résidu courant ; sinon on prend le pas
de Newton pur `u + g` (dont le résidu sera évalué gratuitement au tour suivant) et
on vide l'historique. Anderson ne peut donc jamais dégrader la convergence.

Lancement ::

    maturin develop --release
    python examples/plasticite_poutre_console_anderson.py

Variables d'environnement : `PYRUCAST_NX`, `PYRUCAST_NY` (mailles en long /
hauteur), `PYRUCAST_NSTEPS` (pas de charge), `PYRUCAST_PMAX` (charge finale).
"""

import os

import pyrucast

# Composantes de l'état interne plastique portées d'un pas au suivant (VAR) :
# déformation plastique 3-D (tenseur, 6) + déformation plastique cumulée `p`.
STATE_COMPONENTS = [
    "eps_p_xx",
    "eps_p_yy",
    "eps_p_zz",
    "eps_p_yz",
    "eps_p_xz",
    "eps_p_xy",
    "p",
]

# Profondeur de l'historique d'Anderson (nombre de couples `(u, g)` gardés).
ANDERSON_DEPTH = 3


def _plastic_diagnostics(state):
    """(p_max, nombre de points de Gauss plastifiés) — `p > 0` marque un point."""
    p_max = state.max("p")
    masked = pyrucast.field.mask(state, gt=1e-12, components=["p"])
    n_plastic = round(masked.sum("p"))
    return p_max, n_plastic


def _solve_small_spd(a, b):
    """Résout un petit système dense **symétrique** `A x = b` (`m ≤ 3`) par
    élimination de Gauss avec pivot partiel. Renvoie `None` si `A` est singulière
    (pivot ~ 0) — l'appelant retombe alors sur le Newton pur."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        # Pivot partiel.
        pivot = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r
        if abs(a[pivot][col]) < 1e-30:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        # Élimination.
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for cc in range(col, n):
                a[r][cc] -= factor * a[col][cc]
            b[r] -= factor * b[col]
    # Remontée.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x


def _anderson_step(u, g, history, free_mesh):
    """Correction d'Anderson `Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)` à soustraire au pas de Newton pur
    `u + g` : `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.

    Les `γ` résolvent le moindre-carré `min ‖g − Σⱼ γⱼ ΔGⱼ‖²` sur les DDL
    **libres** (`free_mesh`), via les équations normales `(ΔGᵀΔG) γ = ΔGᵀg`
    régularisées (Tikhonov). Retourne `None` si l'historique est vide ou si le
    petit système dégénère (l'appelant retombe alors sur le Newton pur).

    Tout passe par les opérateurs de champ (`-`, `xty`, `restrict`) : les produits
    scalaires sont les seules réductions, aucune évaluation de la loi."""
    m = len(history)
    if m == 0:
        return None

    # Différences ΔUⱼ = u − u_hist, ΔGⱼ = g − g_hist (vers l'itéré courant).
    du_diffs = [u - u_hist for (u_hist, _) in history]
    dg_diffs = [g - g_hist for (_, g_hist) in history]

    # ΔG restreints aux DDL libres (support des produits scalaires du résidu).
    dg_free = [pyrucast.field.restrict(d, free_mesh) for d in dg_diffs]
    g_free = pyrucast.field.restrict(g, free_mesh)

    # Équations normales (ΔGᵀΔG) γ = ΔGᵀg (petit système m×m symétrique).
    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    trace = 0.0
    for i in range(m):
        for j in range(i, m):
            v = pyrucast.field.xty(dg_free[i], dg_free[j])
            a[i][j] = v
            a[j][i] = v
        trace += a[i][i]
        b[i] = pyrucast.field.xty(dg_free[i], g_free)
    if trace <= 0.0:
        return None  # directions dégénérées
    # Régularisation de Tikhonov : + λ·(trace/m) sur la diagonale.
    lam = 1e-10 * trace / m
    for i in range(m):
        a[i][i] += lam

    gamma = _solve_small_spd(a, b)
    if gamma is None:
        return None

    # Correction Σⱼ γⱼ (ΔUⱼ + ΔGⱼ), assemblée par opérateurs de champ.
    corr = None
    for j, gj in enumerate(gamma):
        term = (du_diffs[j] + dg_diffs[j]) * gj
        corr = term if corr is None else corr + term
    return corr


def main():
    # ── Paramètres (matériau acier, géométrie, chargement) ──────────────────
    young, nu, sigma_y = 210_000.0, 0.3, 250.0
    length, height = 10.0, 1.0
    nx = int(os.environ.get("PYRUCAST_NX", 24))
    ny = int(os.environ.get("PYRUCAST_NY", 6))
    nsteps = int(os.environ.get("PYRUCAST_NSTEPS", 10))
    p_max_load = float(os.environ.get("PYRUCAST_PMAX", 5.0))

    print(
        f"Poutre console plastique (Anderson m={ANDERSON_DEPTH}) : {nx}×{ny} QUA4  "
        f"(L={length}, H={height}), E={young}, ν={nu}, σy={sigma_y}"
    )
    print(
        f"Chargement : 0 → {p_max_load} en {nsteps} pas "
        f"(Newton modifié + accélération d'Anderson)\n"
    )

    # ── Maillage : bords SEG2 gauche/droit puis balayage en QUA4 ────────────
    c = pyrucast.Coords(2)
    pt_a = c.add_node([0.0, 0.0])
    pt_b = c.add_node([0.0, height])
    pt_c = c.add_node([length, 0.0])
    pt_d = c.add_node([length, height])
    left_edge = pyrucast.mesher.line_seg2(pt_a, pt_b, ny)
    right_edge = pyrucast.mesher.line_seg2(pt_c, pt_d, ny)
    mesh = pyrucast.mesher.sweep_qua4(left_edge, right_edge, nx)
    fes = pyrucast.FiniteElementSpace(mesh)

    # Nœud du bout (mi-hauteur) et maillage POI1 des nœuds LIBRES (X > 0).
    tip = mesh.nearest_node([length, height / 2.0])
    coords_field = pyrucast.field.coordinates(mesh, ["X"])
    free_mesh = pyrucast.field.select(coords_field, ge=length / nx / 2.0)

    # ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    model = pyrucast.Model.plasticity(fes, "plane_stress")
    imposed_mesh = pyrucast.mesher.to_poi1(left_edge)
    multiplier = pyrucast.mesher.translate(imposed_mesh, [0.0, 0.0])
    model = model | pyrucast.Model.dirichlet("u_x", "f_x", imposed_mesh, multiplier)
    model = model | pyrucast.Model.dirichlet("u_y", "f_y", imposed_mesh, multiplier)
    materials = pyrucast.build.material_field(
        model, [("E", young), ("nu", nu), ("sigma_y", sigma_y)]
    )

    # Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée une
    # fois ; `solve` met la factorisation en cache et la réutilise.
    k = pyrucast.assemble.stiffness(model, materials)

    # ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    #    droite, réparti en efforts nodaux cohérents (op `flux`). ──────────────
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    load_unit = pyrucast.assemble.flux(right_fes[0], -1.0, "f_y")

    # ── Histoire de chargement : Evolution à valeur CHAMP (t ∈ [0, 1]) ───────
    zero_frame = load_unit * 0.0
    full_frame = load_unit * p_max_load
    load_evo = pyrucast.Evolution(
        [(0.0, zero_frame), (1.0, full_frame)], out_of_range="clamp"
    )

    # ── État de la simulation (persistant entre les pas) ────────────────────
    u = pyrucast.NodeField(mesh, ["u_x", "u_y"])  # déplacement cumulé, nul au départ
    state = pyrucast.ElementField(fes, STATE_COMPONENTS)  # VAR0, nul au premier pas

    # ── Boucle sur les pas de charge ────────────────────────────────────────
    max_newton = 200
    print(
        f"{'pas':>4} {'P':>8} {'iter':>6} {'andrs':>6} "
        f"{'flèche u_y':>14} {'p_max':>14} {'n_plast':>8}"
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
        ext_norm = pyrucast.field.xtx(load_scaled) ** 0.5
        tol = 1e-6 * ext_norm + 1e-12

        # Résidu (et sortie de comportement) à un déplacement d'essai `u` :
        # ε(u) → COMP → BSIG → r = F_ext − F_int, plus la norme sur les DDL libres.
        def residual_at(u):
            strain = pyrucast.field.deformation(u, fes)
            out = pyrucast.behavior.integrate_behavior(model, strain | state, materials)
            f_int = pyrucast.assemble.internal_forces(model, out)
            f_ext = pyrucast.field.restrict_like(load_scaled, f_int)
            residual = f_ext - f_int
            free_res = (
                pyrucast.field.xtx(pyrucast.field.restrict(residual, free_mesh)) ** 0.5
            )
            return residual, free_res, out

        iters = 0
        n_anderson = 0  # combien de pas ont réellement été accélérés
        last_out = None
        res_norm = float("inf")

        # Historique d'Anderson : couples (u, g=K⁻¹r) du pas courant, le plus
        # récent en tête. Vidé au début de chaque pas de charge.
        history = []

        while True:
            # Résidu au déplacement courant (= point fixe `g = K⁻¹r`).
            residual, res_norm, out = residual_at(u)
            last_out = out

            if res_norm <= tol or iters >= max_newton:
                break

            # Direction résidu g = K⁻¹ r (K élastique, cache de factorisation). Le
            # support de δu coïncide déjà avec celui de u (même compagnon POI1 caché
            # de `to_poi1`) ; `restrict_like` ne filtre que les composantes duales
            # (multiplicateurs) — sinon elles se recopieraient dans u par union.
            du = pyrucast.solver.solve(k, residual)
            g = pyrucast.field.restrict_like(du, u)

            # Snapshot du couple (u, g) courant AVANT de bouger (`u + 0.0` = copie
            # indépendante) — source des différences d'Anderson au tour suivant.
            u_snapshot = u + 0.0
            pure_step = u + g  # pas de Newton modifié (référence)

            # Candidat Anderson (si l'historique porte au moins un couple) :
            # extrapolation sur les m derniers (u, g). Garde-fou de descente : on
            # ne le retient que s'il réduit **strictement** le résidu courant.
            chose_anderson = False
            next_u = None
            if history:
                corr = _anderson_step(u, g, history, free_mesh)
                if corr is not None:
                    u_acc = pure_step - corr
                    _, res_acc, _ = residual_at(u_acc)
                    if res_acc < res_norm:
                        next_u = u_acc
                        chose_anderson = True

            # Historique : si Anderson a été retenu, on empile et tronque à la
            # profondeur ; sinon on repart proprement (historique vidé).
            if chose_anderson:
                n_anderson += 1
                history.insert(0, (u_snapshot, g))
                del history[ANDERSON_DEPTH:]
            else:
                history = [(u_snapshot, g)]
            u = next_u if next_u is not None else pure_step
            iters += 1

        converged = res_norm <= tol

        # Commit de l'état : VAR0 ← VAR1 (sortie de comportement convergée ; la loi
        # lit ses entrées par nom, les composantes surnuméraires sont ignorées).
        state = last_out

        # Diagnostics du pas.
        p_max_val, n_plastic = _plastic_diagnostics(state)
        defl = u.value(tip, "u_y")
        any_plasticity = any_plasticity or n_plastic > 0
        flag = "" if converged else "  (résidu résiduel)"
        print(
            f"{step:>4} {load_p:>8.3f} {iters:>6} {n_anderson:>6} "
            f"{defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
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
