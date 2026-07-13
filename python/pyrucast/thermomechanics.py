"""Orchestration thermo-mécanique **pas-à-pas** (Python pur au-dessus de pyrucast).

Couche haut niveau assemblée uniquement à partir des opérateurs déjà exposés par
l'extension : aucune boucle n'est écrite en Rust. Couplage **faible, sens unique**
thermo→méca et thermique **stationnaire par pas** (la librairie n'a pas de terme
transitoire) — la dépendance au temps vient des charges / matériaux interpolés.

Trois fonctions :

* :func:`step_by_step` — mise en donnée (découpe du modèle par physique) puis boucle
  sur les instants ; à chaque pas appelle :func:`thermal_step` puis
  :func:`mechanical_step`, et complète le dictionnaire d'entrée avec les résultats.
* :func:`thermal_step` — une résolution thermique stationnaire ``K_th·T = charges``.
* :func:`mechanical_step` — résolution **non linéaire** (Newton modifié préconditionné
  par la rigidité élastique, **accéléré par Anderson**) de la mécanique, la
  température du pas entrant comme déformation thermique ``ε_th``.

Le chargement et les matériaux sont chacun **un seul champ unioné** : ``loads`` porte à
la fois les composantes thermiques (``q`` / ``imposed_T``) et mécaniques (``f_*`` /
``imposed_u``) ; ``materials`` porte ``k``/``h`` (thermique) et ``E``/``nu``/``alpha``
(mécanique). Chaque étape ne lit que ce dont elle a besoin : ``solve`` n'échantillonne
le second membre qu'aux DDL de sa matrice et ignore les composantes surnuméraires.
"""

from . import _pyrucast as pc

__all__ = ["step_by_step", "thermal_step", "mechanical_step"]

# Profondeur d'historique d'Anderson par défaut (nombre de couples (u, g) gardés).
_ANDERSON_DEPTH = 3


# ─────────────────────────────────────────────────────────────────────────────
# Découpe du modèle par physique
# ─────────────────────────────────────────────────────────────────────────────
def _constrained_variable(sub):
    """Variable contrainte par un sous-modèle de Lagrange (Dirichlet) : le primal
    d'une contrainte est ``lambda_<var>`` — on renvoie ``<var>`` (ou ``None``)."""
    for name in sub.primal_vars():
        if name.startswith("lambda_"):
            return name[len("lambda_") :]
    return None


def _split_model(model):
    """Sépare ``model`` en (thermique, mécanique). ``Model.filter`` partage les
    handles (pas de copie), donc les zones matériau / fespace coïncident avec le
    modèle complet. Les contraintes (exclues par ``filter`` des physiques) sont
    ré-unionées à la physique dont elles contraignent une variable primale."""
    thermal = model.filter("thermal")
    mechanical = model.filter("mechanical")
    thermal_vars = set(thermal.primal_vars())
    mechanical_vars = set(mechanical.primal_vars())

    constraints = model.filter("constraint")
    for i in range(len(constraints)):
        sub = constraints[i]
        var = _constrained_variable(sub)
        if var in thermal_vars:
            thermal = thermal | sub
        elif var in mechanical_vars:
            mechanical = mechanical | sub
        else:
            # Contrainte non rattachable (variable inconnue des deux physiques) :
            # défaut mécanique — le cas thermo-mécanique usuel n'y tombe pas.
            mechanical = mechanical | sub
    return thermal, mechanical


def _interpolate(spec, t):
    """Valeur d'un champ éventuellement tabulé dans le temps : ``spec.interpolate(t)``
    si ``spec`` est une ``Evolution``, sinon ``spec`` tel quel (champ constant)."""
    if isinstance(spec, pc.Evolution):
        return spec.interpolate(t)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Petit solveur dense pour les équations normales d'Anderson (m ≤ 3)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_small_spd(a, b):
    """Résout un petit système dense **symétrique** ``A x = b`` (``m ≤ 3``) par
    élimination de Gauss à pivot partiel. Renvoie ``None`` si ``A`` est singulière
    (pivot ~ 0) — l'appelant retombe alors sur le Newton pur."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r
        if abs(a[pivot][col]) < 1e-30:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for cc in range(col, n):
                a[r][cc] -= factor * a[col][cc]
            b[r] -= factor * b[col]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x


def _anderson_correction(u, g, history, free_mesh):
    """Correction d'Anderson ``Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`` à soustraire au pas de Newton pur
    ``u + g`` : ``u_acc = u + g − correction``. Les ``γ`` résolvent le moindre-carré
    ``min ‖g − Σⱼ γⱼ ΔGⱼ‖²`` sur les DDL libres via les équations normales
    ``(ΔGᵀΔG) γ = ΔGᵀg`` régularisées (Tikhonov). ``None`` si l'historique est vide
    ou si le petit système dégénère. Tout passe par les opérateurs de champ."""
    m = len(history)
    if m == 0:
        return None

    du_diffs = [u - u_hist for (u_hist, _) in history]
    dg_diffs = [g - g_hist for (_, g_hist) in history]

    dg_free = [pc.restrict(d, free_mesh) for d in dg_diffs]
    g_free = pc.restrict(g, free_mesh)

    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    trace = 0.0
    for i in range(m):
        for j in range(i, m):
            v = pc.xty(dg_free[i], dg_free[j])
            a[i][j] = v
            a[j][i] = v
        trace += a[i][i]
        b[i] = pc.xty(dg_free[i], g_free)
    if trace <= 0.0:
        return None
    lam = 1e-10 * trace / m
    for i in range(m):
        a[i][i] += lam

    gamma = _solve_small_spd(a, b)
    if gamma is None:
        return None

    correction = None
    for j, gj in enumerate(gamma):
        term = (du_diffs[j] + dg_diffs[j]) * gj
        correction = term if correction is None else correction + term
    return correction


# ─────────────────────────────────────────────────────────────────────────────
# Étape thermique (stationnaire)
# ─────────────────────────────────────────────────────────────────────────────
def thermal_step(thermal_model, materials, loads):
    """Une résolution thermique **stationnaire** : assemble ``K_th`` (conduction +
    éventuel terme de film de convection) et résout ``K_th·T = loads``.

    ``loads`` est le champ de charges unioné : ``solve`` n'y lit que les lignes
    thermiques (``q`` en source de Neumann, ``imposed_T`` pour un Dirichlet de
    température) et ignore les composantes mécaniques. Renvoie le ``NodeField``
    solution (température ``T`` + multiplicateurs éventuels)."""
    k = pc.stiffness(thermal_model, materials)
    return pc.solve(k, loads)


# ─────────────────────────────────────────────────────────────────────────────
# Étape mécanique (non linéaire, Newton modifié + Anderson, thermo-couplée)
# ─────────────────────────────────────────────────────────────────────────────
def mechanical_step(
    mechanical_model,
    fespace,
    mesh,
    materials,
    loads,
    temperature,
    u,
    state_prev,
    dt,
    *,
    t_ref=0.0,
    free_mesh=None,
    anderson_depth=_ANDERSON_DEPTH,
    max_newton=200,
    tol_rel=1e-6,
    reference=0.0,
    stiffness_matrix=None,
):
    """Résout la mécanique non linéaire du pas et renvoie ``(u, out, info)``.

    Newton **modifié** : l'opérateur d'itération est la rigidité **élastique** ``k``
    (assemblée une fois par pas, factorisation mise en cache par ``solve``) ;
    l'itération ``u ← u + k⁻¹ r(u)`` est un point fixe préconditionné, **accéléré par
    Anderson** (historique ``m = anderson_depth``, garde-fou de descente).

    Le couplage thermique est **faible, sens unique** : la température ``temperature``
    du pas donne la déformation thermique ``ε_th`` (via ``thermal_strain``), retirée
    de la déformation totale avant l'intégration de la loi de comportement. L'état
    interne du pas précédent (``state_prev``, ``None`` au premier pas) est passé comme
    ``prev`` — prédicteur incrémental ``σ(A) + C:Δε``. ``out`` (contraintes + VAR + ε)
    convergé est renvoyé pour servir de ``prev`` au pas suivant.
    """
    k = (
        stiffness_matrix
        if stiffness_matrix is not None
        else pc.stiffness(mechanical_model, materials)
    )
    support = free_mesh if free_mesh is not None else mesh

    # Déformation thermique du pas (champ aux points de Gauss). ``thermal_strain``
    # prend une température **aux points de Gauss** : on passe le champ nodal par
    # ``interp_to_gauss`` (restreint au maillage pour ne garder que « T »).
    eps_th = None
    if temperature is not None:
        t_gauss = pc.interp_to_gauss(pc.restrict(temperature, mesh), fespace)
        eps_th = pc.thermal_strain(t_gauss, materials, fespace, t_ref)

    def residual_at(u_trial):
        strain = pc.deformation(u_trial, fespace)
        mech_eps = strain - eps_th if eps_th is not None else strain
        out = pc.integrate_behavior(
            mechanical_model, mech_eps, materials, prev=state_prev, dt=dt
        )
        f_int = pc.internal_forces(mechanical_model, out)
        f_ext = pc.restrict_like(loads, f_int)
        residual = f_ext - f_int
        free_res = pc.xtx(pc.restrict(residual, support)) ** 0.5
        return residual, free_res, out

    history = []
    iters = 0
    n_anderson = 0
    last_out = None
    res_norm = float("inf")
    # Échelle de référence du critère de Newton : le **maximum courant** du résidu
    # initial des pas (déséquilibre de charge, thermique et/ou externe). La prendre
    # globale — et non le résidu initial du *pas* — évite qu'un pas déjà à
    # l'équilibre (charge inchangée) impose une tolérance sous le plancher de
    # round-off du solveur ; c'est aussi robuste pour une charge purement thermique
    # (contrainte nulle, aucune échelle d'effort propre au pas).
    ref = reference
    tol = None

    while True:
        residual, res_norm, out = residual_at(u)
        last_out = out
        if tol is None:
            ref = max(ref, res_norm)
            tol = tol_rel * ref + 1e-12
        if res_norm <= tol or iters >= max_newton:
            break

        # Direction résidu g = k⁻¹ r (k élastique, factorisation cachée), reprojetée
        # sur le support / les composantes de u.
        g = pc.restrict_like(pc.solve(k, residual), u)
        u_snapshot = u + 0.0  # copie indépendante avant de bouger
        pure_step = u + g

        chose_anderson = False
        next_u = None
        if history:
            correction = _anderson_correction(u, g, history, support)
            if correction is not None:
                u_acc = pure_step - correction
                _, res_acc, _ = residual_at(u_acc)
                if res_acc < res_norm:
                    next_u = u_acc
                    chose_anderson = True

        if chose_anderson:
            n_anderson += 1
            history.insert(0, (u_snapshot, g))
            del history[anderson_depth:]
        else:
            history = [(u_snapshot, g)]
        u = next_u if next_u is not None else pure_step
        iters += 1

    info = {
        "iters": iters,
        "anderson": n_anderson,
        "converged": res_norm <= tol,
        "res_norm": res_norm,
        "reference": ref,
    }
    return u, last_out, info


# ─────────────────────────────────────────────────────────────────────────────
# Boucle pas-à-pas
# ─────────────────────────────────────────────────────────────────────────────
def step_by_step(data):
    """Calcule une suite de pas de temps thermo-mécaniques et **complète** ``data``.

    ``data`` est un dictionnaire :

    ``times``      liste des instants (``list[float]``).
    ``model``      ``Model`` complet (thermique + mécanique + Dirichlet), sur ``fespace``.
    ``fespace``    ``FiniteElementSpace`` continu partagé thermique / mécanique.
    ``mesh``       ``Mesh`` continu (support du déplacement cumulé).
    ``loads``      ``NodeField`` ou ``Evolution`` unioné (``q``/``imposed_T`` + ``f_*``/``imposed_u``).
    ``materials``  ``ElementField`` ou ``Evolution`` unioné (``k``/``h`` + ``E``/``nu``/``alpha``).
    ``t_ref``      (opt.) température de référence pour ``ε_th`` (défaut ``0.0``).
    ``free_mesh``  (opt.) ``Mesh`` des DDL libres pour la norme de résidu (recommandé
                   en présence de Dirichlet ; défaut : ``mesh`` complet).
    ``anderson_depth`` / ``max_newton`` / ``tol_rel`` — (opt.) réglages du solveur méca.

    En retour, ``data["results"]`` est une liste (un élément par instant) de dicts
    ``{"time", "temperature", "displacement", "state", "mech_iters",
    "mech_anderson", "converged"}``. ``data`` (le même objet) est renvoyé.
    """
    model = data["model"]
    fespace = data["fespace"]
    mesh = data["mesh"]
    times = list(data["times"])
    loads_spec = data["loads"]
    materials_spec = data["materials"]

    t_ref = data.get("t_ref", 0.0)
    free_mesh = data.get("free_mesh")
    anderson_depth = data.get("anderson_depth", _ANDERSON_DEPTH)
    max_newton = data.get("max_newton", 200)
    tol_rel = data.get("tol_rel", 1e-6)

    thermal_model, mechanical_model = _split_model(model)
    has_thermal = len(thermal_model) > 0
    # Composantes de déplacement (domaine mécanique seul, **avant** contraintes :
    # `primal_vars` d'un modèle avec Dirichlet inclut les multiplicateurs
    # `lambda_*` que `deformation` refuse).
    displacement_vars = model.filter("mechanical").primal_vars()
    has_mechanical = len(displacement_vars) > 0

    # État persistant entre les pas.
    u = pc.NodeField(mesh, displacement_vars) if has_mechanical else None
    state_prev = None
    reference = 0.0  # échelle de résidu (max courant), partagée entre les pas

    results = []
    prev_t = times[0] if times else 0.0
    for i, t in enumerate(times):
        dt = t - prev_t
        # Matériaux du pas, **consolidés** : `material_field` produit une zone par
        # sous-modèle, donc thermique et mécanique bâtis sur la même fespace créent
        # deux zones sur ce support — que `stiffness` ne saurait départager.
        # `consolidate` les fusionne en une zone portant l'union des composantes
        # (`k` + `E`/`nu`/`alpha`), lue par chaque physique selon ses besoins.
        materials_t = pc.consolidate(_interpolate(materials_spec, t))
        loads_t = _interpolate(loads_spec, t)

        temperature = (
            thermal_step(thermal_model, materials_t, loads_t) if has_thermal else None
        )

        info = {}
        if has_mechanical:
            u, state_prev, info = mechanical_step(
                mechanical_model,
                fespace,
                mesh,
                materials_t,
                loads_t,
                temperature,
                u,
                state_prev,
                dt,
                t_ref=t_ref,
                free_mesh=free_mesh,
                anderson_depth=anderson_depth,
                max_newton=max_newton,
                tol_rel=tol_rel,
                reference=reference,
            )
            reference = info.get("reference", reference)

        results.append(
            {
                "time": t,
                "temperature": temperature,
                "displacement": u,
                "state": state_prev,
                "mech_iters": info.get("iters"),
                "mech_anderson": info.get("anderson"),
                "converged": info.get("converged"),
            }
        )
        prev_t = t

    data["results"] = results
    return data
