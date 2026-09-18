"""**Step-by-step** thermo-mechanical orchestration (pure Python above pyrucast).

A high-level layer assembled solely from the operators the extension already
exposes: no loop is written in Rust. **Weak, one-way** thermal→mechanical
coupling, and **steady-state per step** thermics (the library has no transient
term) — time dependence comes from the interpolated loads / materials.

Trois fonctions :

* :func:`step_by_step` — setup (splitting the model per physics) then a loop
  over the instants; at each step it calls :func:`thermal_step` then
  :func:`mechanical_step`, and fills the input dictionary with the results.
* :func:`thermal_step` — one steady thermal solve ``K_th·T = loads``.
* :func:`mechanical_step` — résolution **non linéaire** (Newton modifié préconditionné
  by the elastic stiffness, **Anderson accelerated**) of the mechanics, the
  step's temperature entering as thermal strain ``ε_th``.

Loading and materials are each **a single unioned field**: ``loads`` carries
both the thermal components (``q`` / ``imposed_T``) and the mechanical ones
``imposed_u``) ; ``materials`` porte ``k``/``h`` (thermique) et ``E``/``nu``/``alpha``
(``f_*`` / …). Each step reads only what it needs: ``solve`` samples the
right-hand side only at its matrix's DOFs and ignores the extra components.
"""

from . import _pyrucast as pc

__all__ = ["step_by_step", "thermal_step", "mechanical_step"]

# Default Anderson history depth (number of (u, g) pairs kept).
_ANDERSON_DEPTH = 3


# ─────────────────────────────────────────────────────────────────────────────
# Splitting the model per physics
# ─────────────────────────────────────────────────────────────────────────────
def _constrained_variable(sub):
    """Variable constrained by a Lagrange sub-model (Dirichlet): a constraint's
    primal is ``lambda_<var>`` — we return ``<var>`` (or ``None``)."""
    for name in sub.primal_vars():
        if name.startswith("lambda_"):
            return name[len("lambda_") :]
    return None


def _split_model(model):
    """Splits ``model`` into (thermal, mechanical). ``Model.filter`` shares the
    handles (no copy), so the material / fespace zones coincide with the full
    model. The constraints (excluded from the physics by ``filter``) are
    unioned back into the physics whose primal variable they constrain."""
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
            # Unattachable constraint (variable unknown to both physics):
            # mechanical by default — the usual case never falls here.
            mechanical = mechanical | sub
    return thermal, mechanical


def _interpolate(spec, t):
    """Value of a field possibly tabulated in time: ``spec.interpolate(t)`` if
    ``spec`` is an ``Evolution``, else ``spec`` as is (constant field)."""
    if isinstance(spec, pc.Evolution):
        return spec.interpolate(t)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Small dense solver for Anderson's normal equations (m ≤ 3)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_small_spd(a, b):
    """Résout un petit système dense **symétrique** ``A x = b`` (``m ≤ 3``) par
    Gauss elimination with partial pivoting. Returns ``None`` if ``A`` is
    singular (pivot ~ 0) — the caller then falls back on plain Newton."""
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
    """Anderson correction ``Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`` to subtract from the plain
    Newton step ``u + g``: ``u_acc = u + g − correction``. The ``γ`` solve the
    least squares ``min ‖g − Σⱼ γⱼ ΔGⱼ‖²`` on the free DOFs through the normal
    equations ``(ΔGᵀΔG) γ = ΔGᵀg``, regularized (Tikhonov). ``None`` if the
    history is empty or the small system degenerates. All through field operators."""
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
    """One **steady** thermal solve: assembles ``K_th`` (conduction +
    éventuel terme de film de convection) et résout ``K_th·T = loads``.

    ``loads`` is the unioned load field: ``solve`` reads only the thermal rows
    there (``q`` as a Neumann source, ``imposed_T`` for a temperature
    Dirichlet) and ignores the mechanical components. Returns the ``NodeField``
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
    """Solves the step's nonlinear mechanics and returns ``(u, out, info)``.

    **Modified** Newton: the iteration operator is the **elastic** stiffness
    ``k`` (assembled once per step, factorization cached by ``solve``); the
    iteration ``u ← u + k⁻¹ r(u)`` is a preconditioned fixed point,
    Anderson** (historique ``m = anderson_depth``, garde-fou de descente).

    The thermal coupling is **weak and one-way**: the step's ``temperature``
    gives the thermal strain ``ε_th`` (through ``thermal_strain``), removed
    de la déformation totale avant l'intégration de la loi de comportement. L'état
    internal state of the previous step (``state_prev``, ``None`` at the first
    ``prev`` — prédicteur incrémental ``σ(A) + C:Δε``. ``out`` (contraintes + VAR + ε)
    step) is passed as ``prev`` for the next step.
    """
    k = (
        stiffness_matrix
        if stiffness_matrix is not None
        else pc.stiffness(mechanical_model, materials)
    )
    support = free_mesh if free_mesh is not None else mesh

    # The step's thermal strain (a field at the Gauss points).
    # ``thermal_strain`` takes a temperature **at the Gauss points**: the nodal
    # field goes through ``interp_to_gauss`` (restricted to the mesh to keep "T").
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
        f_int = pc.internal_forces(mechanical_model, out, u_trial, materials)
        f_ext = pc.restrict_like(loads, f_int)
        residual = f_ext - f_int
        free_res = pc.xtx(pc.restrict(residual, support)) ** 0.5
        return residual, free_res, out

    history = []
    iters = 0
    n_anderson = 0
    last_out = None
    res_norm = float("inf")
    # Reference scale of the Newton criterion: the **running maximum** of the
    # steps' initial residual (load imbalance, thermal and/or external). Taking
    # it global — rather than the *step's* initial residual — keeps a step
    # already at equilibrium (unchanged load) from imposing a tolerance below
    # the solver's round-off floor; it is also robust for a purely thermal load
    # (zero stress, no force scale of its own for the step).
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
        # on u's support / components.
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
# Step-by-step loop
# ─────────────────────────────────────────────────────────────────────────────
def step_by_step(data):
    """Computes a series of thermo-mechanical time steps and **fills** ``data``.

    ``data`` is a dictionary:

    ``times``      list of instants (``list[float]``).
    ``model``      ``Model`` complet (thermique + mécanique + Dirichlet). La fespace
                   and the mesh are **deduced** from it (``Model.fespace()`` /
                   ``FiniteElementSpace.mesh()``) — no need to supply them.
    ``loads``      unioned ``NodeField`` or ``Evolution`` (``q``/``imposed_T`` + ``f_*``/``imposed_u``).
    ``materials``  unioned ``ElementField`` or ``Evolution`` (``k``/``h`` + ``E``/``nu``/``alpha``).
    ``t_ref``      (opt.) reference temperature for ``ε_th`` (default ``0.0``).
    ``free_mesh``  (opt.) ``Mesh`` of the free DOFs for the residual norm (advised
                   en présence de Dirichlet ; défaut : maillage mécanique complet).
    ``anderson_depth`` / ``max_newton`` / ``tol_rel`` — (opt.) mechanical solver settings.

    On return, ``data["results"]`` is a list (one item per instant) of dicts
    ``{"time", "temperature", "displacement", "state", "mech_iters",
    "mech_anderson", "converged"}``. ``data`` (the same object) is returned.
    """
    model = data["model"]
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
    # The mechanical domain alone (without the constraints): `primal_vars` of a
    # model with Dirichlet includes the `lambda_*` multipliers that
    # `deformation` refuses. Fespace and mesh are deduced from this domain.
    mechanical_domain = model.filter("mechanical")
    displacement_vars = mechanical_domain.primal_vars()
    has_mechanical = len(displacement_vars) > 0

    # Fespace / mesh deduced from the model (no separate argument required).
    if has_mechanical:
        fespace = mechanical_domain.fespace()
        mesh = fespace.mesh()
        u = pc.NodeField(
            mesh, displacement_vars
        )  # cumulative displacement, zero at first
    else:
        fespace = mesh = u = None
    state_prev = None
    reference = 0.0  # residual scale (running max), shared across the steps

    results = []
    prev_t = times[0] if times else 0.0
    for i, t in enumerate(times):
        dt = t - prev_t
        # `material_field` produces one zone per sub-model: thermal and
        # mechanical built on the same fespace leave two zones (disjoint
        # components) side by side. No need to merge them — the operators
        # (`stiffness`, `integrate_behavior`, `thermal_strain`) resolve their
        # material zone through the components they require.
        materials_t = _interpolate(materials_spec, t)
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
