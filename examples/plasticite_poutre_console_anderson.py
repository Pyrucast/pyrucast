"""Elasto-plastic cantilever beam — modified Newton **accelerated by Anderson
acceleration (m = 3)** on top of pyrucast.

Python version of `examples/plasticite_poutre_console_anderson.rs` (same building
blocks, same algorithm). A variant of `plasticite_poutre_console.py`: same
physics, same mesh, same pyrucast operators. Only the non-linear loop changes —
the original is kept **frozen** as the reference to compare results against (the
deflections and the plasticity must coincide; only the iteration count should
drop).

Modified Newton = preconditioned fixed point
--------------------------------------------
As in the original example, the iteration operator is the **elastic** stiffness
`K` (assembled + factorised once, `solve`'s cache). The iteration
`u ← u + K⁻¹r(u)` is a **preconditioned fixed point**: the "residual direction"
`g(u) = K⁻¹ r(u)` vanishes at convergence — it is the natural residual of the
fixed point, **already computed** at every iteration (`du = solve(k, residual)`).

The price of the constant operator is a merely **linear** convergence on the
plastic branch (many iterations). Anderson acceleration exploits the history of
the last `m = 3` pairs `(u, g)` to extrapolate a much better step, **without
re-evaluating the behaviour law**: the small least-squares problem only handles
dot products of fields already in hand.

Anderson acceleration (m = 3)
-----------------------------
At iteration `k`, with the history of the last `m ≤ 3` pairs `(uᵢ, gᵢ)` (most
recent first):

1. Differences `ΔGⱼ = g − g_hist`, `ΔUⱼ = u − u_hist`.
2. Least squares `min_γ ‖g − Σⱼ γⱼ ΔGⱼ‖²` → normal equations `(ΔGᵀΔG) γ =
   ΔGᵀg`, whose entries are all dot products **over the free DOFs** (the same
   DOFs as the residual norm), Tikhonov-regularised.
3. Small dense `m×m` solve (m ≤ 3, Gaussian elimination).
4. Extrapolated step `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.

Descent safeguard: the residual of the Anderson candidate is evaluated and the
candidate is kept only if it **strictly** reduces the current residual;
otherwise the pure Newton step `u + g` is taken (whose residual will be
evaluated for free at the next round) and the history is cleared. Anderson can
therefore never degrade the convergence.

Run with ::

    maturin develop --release
    python examples/plasticite_poutre_console_anderson.py

Environment variables: `PYRUCAST_NX`, `PYRUCAST_NY` (cells along the length /
through the height), `PYRUCAST_NSTEPS` (load steps), `PYRUCAST_PMAX` (final load).
"""

import os

import pyrucast

# Components of the plastic internal state carried from one step to the next
# (VAR): 3-D plastic strain (tensor, 6) + cumulated plastic strain `p`.
STATE_COMPONENTS = [
    "eps_p_xx",
    "eps_p_yy",
    "eps_p_zz",
    "eps_p_yz",
    "eps_p_xz",
    "eps_p_xy",
    "p",
]

# Depth of the Anderson history (number of `(u, g)` pairs kept).
ANDERSON_DEPTH = 3


def _plastic_diagnostics(state):
    """(p_max, number of yielded Gauss points) — `p > 0` marks a point."""
    p_max = state.max("p")
    masked = state.mask(gt=1e-12, components=["p"])
    n_plastic = round(masked.sum("p"))
    return p_max, n_plastic


def _solve_small_spd(a, b):
    """Solves a small dense **symmetric** system `A x = b` (`m ≤ 3`) by Gaussian
    elimination with partial pivoting. Returns `None` if `A` is singular
    (pivot ~ 0) — the caller then falls back on the pure Newton step."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        # Partial pivoting.
        pivot = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r
        if abs(a[pivot][col]) < 1e-30:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        # Elimination.
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for cc in range(col, n):
                a[r][cc] -= factor * a[col][cc]
            b[r] -= factor * b[col]
    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x


def _anderson_step(u, g, history, free_mesh):
    """Anderson correction `Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)` to subtract from the pure Newton
    step `u + g`: `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.

    The `γ` solve the least-squares problem `min ‖g − Σⱼ γⱼ ΔGⱼ‖²` over the
    **free** DOFs (`free_mesh`), through the regularised (Tikhonov) normal
    equations `(ΔGᵀΔG) γ = ΔGᵀg`. Returns `None` if the history is empty or if
    the small system degenerates (the caller then falls back on the pure Newton
    step).

    Everything goes through the field operators (`-`, `xty`, `restrict`): the dot
    products are the only reductions, and the law is never evaluated."""
    m = len(history)
    if m == 0:
        return None

    # Differences ΔUⱼ = u − u_hist, ΔGⱼ = g − g_hist (towards the current iterate).
    du_diffs = [u - u_hist for (u_hist, _) in history]
    dg_diffs = [g - g_hist for (_, g_hist) in history]

    # ΔG restricted to the free DOFs (support of the residual dot products).
    dg_free = [pyrucast.node_field.restrict(d, free_mesh) for d in dg_diffs]
    g_free = pyrucast.node_field.restrict(g, free_mesh)

    # Normal equations (ΔGᵀΔG) γ = ΔGᵀg (small symmetric m×m system).
    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    trace = 0.0
    for i in range(m):
        for j in range(i, m):
            v = pyrucast.measure.xty(dg_free[i], dg_free[j])
            a[i][j] = v
            a[j][i] = v
        trace += a[i][i]
        b[i] = pyrucast.measure.xty(dg_free[i], g_free)
    if trace <= 0.0:
        return None  # degenerate directions
    # Tikhonov regularisation: + λ·(trace/m) on the diagonal.
    lam = 1e-10 * trace / m
    for i in range(m):
        a[i][i] += lam

    gamma = _solve_small_spd(a, b)
    if gamma is None:
        return None

    # Correction Σⱼ γⱼ (ΔUⱼ + ΔGⱼ), assembled through the field operators.
    corr = None
    for j, gj in enumerate(gamma):
        term = (du_diffs[j] + dg_diffs[j]) * gj
        corr = term if corr is None else corr + term
    return corr


def main():
    # ── Parameters (steel material, geometry, loading) ──────────────────────
    young, nu, sigma_y = 210_000.0, 0.3, 250.0
    length, height = 10.0, 1.0
    nx = int(os.environ.get("PYRUCAST_NX", 24))
    ny = int(os.environ.get("PYRUCAST_NY", 6))
    nsteps = int(os.environ.get("PYRUCAST_NSTEPS", 10))
    p_max_load = float(os.environ.get("PYRUCAST_PMAX", 5.0))

    print(
        f"Plastic cantilever beam (Anderson m={ANDERSON_DEPTH}): {nx}×{ny} QUA4  "
        f"(L={length}, H={height}), E={young}, ν={nu}, σy={sigma_y}"
    )
    print(
        f"Loading: 0 → {p_max_load} in {nsteps} steps "
        f"(modified Newton + Anderson acceleration)\n"
    )

    # ── Mesh: left/right SEG2 edges, then a sweep into QUA4 ─────────────────
    c = pyrucast.Coords(2)
    pt_a = c.add_node([0.0, 0.0])
    pt_b = c.add_node([0.0, height])
    pt_c = c.add_node([length, 0.0])
    pt_d = c.add_node([length, height])
    left_edge = pyrucast.mesh.line(pt_a, pt_b, ny)
    right_edge = pyrucast.mesh.line(pt_c, pt_d, ny)
    mesh = pyrucast.mesh.sweep(left_edge, right_edge, nx)
    fes = pyrucast.FiniteElementSpace(mesh)

    # Tip node (mid-height) and POI1 mesh of the FREE nodes (X > 0).
    tip = mesh.nearest_node([length, height / 2.0])
    coords_field = pyrucast.node_field.positions(mesh, ["X"])
    free_mesh = pyrucast.mesh.select(coords_field, ge=length / nx / 2.0)

    # ── Model: plasticity (plane stress) + clamped end (Dirichlet) ──────────
    model = pyrucast.model.plasticity_perfect(fes, "plane_stress")
    imposed_mesh = pyrucast.mesh.to_poi1(left_edge)
    multiplier = pyrucast.mesh.translate(imposed_mesh, [0.0, 0.0])
    model = model | pyrucast.model.dirichlet(model, "u_x", imposed_mesh, multiplier)
    model = model | pyrucast.model.dirichlet(model, "u_y", imposed_mesh, multiplier)
    # ── Reference load: unit shear (density −1) on the right face, as
    #    consistent nodal forces. A term of the model. ────────────────────────
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    model = model | pyrucast.model.flux(right_fes, model, "f_y")

    materials = pyrucast.element_field.material_field(
        model,
        [("E", young), ("nu", nu), ("sigma_y", sigma_y), ("phi_f_y", -1.0)],
    )
    load_unit = pyrucast.node_field.external_forces(model, materials)

    # ELASTIC stiffness: iteration operator of the modified Newton. Assembled
    # once; `solve` caches the factorisation and reuses it.
    k = pyrucast.matrix.stiffness(model, materials)

    # ── Loading history: an Evolution with FIELD values (t ∈ [0, 1]) ────────
    zero_frame = load_unit * 0.0
    full_frame = load_unit * p_max_load
    load_evo = pyrucast.Evolution(
        [(0.0, zero_frame), (1.0, full_frame)], out_of_range="clamp"
    )

    # ── Simulation state (persistent across the steps) ──────────────────────
    u = pyrucast.NodeField(
        mesh, ["u_x", "u_y"]
    )  # accumulated displacement, zero at first
    # The state at rest: `None` at the first step, where A is the reference
    # configuration. The operator materialises it itself, with **every** component
    # the law reads back afterwards — σ(A), ε(A), the internal state — where a
    # hand-written list forgets some.
    state = None

    # ── Loop over the load steps ────────────────────────────────────────────
    max_newton = 200
    print(
        f"{'step':>4} {'P':>8} {'iter':>6} {'andrs':>6} "
        f"{'deflection u_y':>14} {'p_max':>14} {'n_plast':>8}"
    )

    prev_defl = 0.0
    any_plasticity = False

    for step in range(1, nsteps + 1):
        # Pseudo-time of the step ∈ ]0, 1]; the external load follows from it by
        # interpolation of the Evolution (nodal force field of the step).
        t = step / nsteps
        load_p = p_max_load * t  # nominal shear at the tip (for the display)
        load_scaled = load_evo.interpolate(t)
        # Norm of the step load (relative scale of the residual): xᵀx of the field.
        ext_norm = pyrucast.measure.xtx(load_scaled) ** 0.5
        tol = 1e-6 * ext_norm + 1e-12

        # Residual (and behaviour output) at a trial displacement `u`:
        # ε(u) → COMP → BSIG → r = F_ext − F_int, plus the norm on the free DOFs.
        def residual_at(u):
            strain = pyrucast.element_field.deformation(u, fes)
            out = pyrucast.element_field.integrate_behavior(
                model, strain, materials, prev=state
            )
            f_int = pyrucast.node_field.internal_forces(model, out, u, materials)
            f_ext = pyrucast.node_field.restrict_like(load_scaled, f_int)
            residual = f_ext - f_int
            free_res = (
                pyrucast.measure.xtx(pyrucast.node_field.restrict(residual, free_mesh))
                ** 0.5
            )
            return residual, free_res, out

        iters = 0
        n_anderson = 0  # how many steps were actually accelerated
        last_out = None
        res_norm = float("inf")

        # Anderson history: pairs (u, g=K⁻¹r) of the current step, most recent
        # first. Cleared at the beginning of each load step.
        history = []

        while True:
            # Residual at the current displacement (= fixed point `g = K⁻¹r`).
            residual, res_norm, out = residual_at(u)
            last_out = out

            if res_norm <= tol or iters >= max_newton:
                break

            # Residual direction g = K⁻¹ r (elastic K, factorisation cache). The
            # support of δu already coincides with that of u (same hidden POI1
            # companion from `to_poi1`); `restrict_like` only filters out the dual
            # components (multipliers) — otherwise they would be copied into u by
            # union.
            du = pyrucast.solver.solve(k, residual)
            g = pyrucast.node_field.restrict_like(du, u)

            # Snapshot of the current (u, g) pair BEFORE moving (`u + 0.0` = an
            # independent copy) — source of the Anderson differences next round.
            u_snapshot = u + 0.0
            pure_step = u + g  # modified Newton step (reference)

            # Anderson candidate (if the history carries at least one pair):
            # extrapolation over the last m (u, g). Descent safeguard: it is kept
            # only if it **strictly** reduces the current residual.
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

            # History: if Anderson was kept, push and truncate to the depth;
            # otherwise start over cleanly (history cleared).
            if chose_anderson:
                n_anderson += 1
                history.insert(0, (u_snapshot, g))
                del history[ANDERSON_DEPTH:]
            else:
                history = [(u_snapshot, g)]
            u = next_u if next_u is not None else pure_step
            iters += 1

        converged = res_norm <= tol

        # State commit: VAR0 ← VAR1 (converged behaviour output; the law reads its
        # inputs by name, so the extra components are ignored).
        state = last_out

        # Diagnostics of the step.
        p_max_val, n_plastic = _plastic_diagnostics(state)
        defl = u.value(tip, "u_y")
        any_plasticity = any_plasticity or n_plastic > 0
        flag = "" if converged else "  (residual left over)"
        print(
            f"{step:>4} {load_p:>8.3f} {iters:>6} {n_anderson:>6} "
            f"{defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        )

        # The deflection grows (in absolute value, downwards) with the load.
        assert abs(defl) >= abs(prev_defl) - 1e-9, (
            f"non-monotonic deflection at step {step}"
        )
        prev_defl = defl

    # Beyond first yield, a plastic zone must appear.
    p_first_yield = sigma_y * (height * height / 6.0) / length
    if p_max_load > p_first_yield:
        assert any_plasticity, (
            f"P_max={p_max_load} exceeds first yield "
            f"(≈{p_first_yield:.2f}) but no plastic point was detected"
        )
        print(
            f"\nOK: plasticity developed (P_max={p_max_load} > P_elastic≈{p_first_yield:.2f})."
        )
    else:
        print(
            f"\nOK: the response stayed elastic (P_max={p_max_load} ≤ ≈{p_first_yield:.2f})."
        )


if __name__ == "__main__":
    main()
