//! Elasto-plastic cantilever beam — **instrumented variant** of
//! [`plasticite_poutre_console_anderson`] for performance evaluation.
//!
//! Physics, mesh, operators and non-linear loop **identical** to the Anderson
//! example (frozen as the reference): this file differs from it only by
//! `AtomicU64` timing counters around each sub-operator of the hot loop. At the
//! end, it prints a **profile of the cumulated times**, breaking the cost down
//! between `deformation`, the `strain | state` merge, the behaviour law, the
//! internal forces, the residual, the solve and the Anderson acceleration. The
//! budget of the loop is closed ("unaccounted" entry ≈ 0).
//!
//! Runs like the original example (`PYRUCAST_NX`, `PYRUCAST_NY`,
//! `PYRUCAST_NSTEPS`, `PYRUCAST_PMAX`). For the complementary sampling profile
//! (samply + `addr2line`), see `examples/profil_anderson.sh`.
//!
//! Physics
//! -------
//! 2-D plane-stress continuum, small strains. Perfect von Mises plasticity
//! (J2 radial return, no hardening): the equivalent stress is capped at
//! `sigma_y`. Beam clamped at the left end (`u_x = u_y = 0`), sheared
//! downwards on the right face. The load is raised by increments.
//!
//! Modified Newton = preconditioned fixed point
//! --------------------------------------------
//! As in the original example, the iteration operator is the **elastic**
//! stiffness `K` (assembled + factorised once, `solve`'s cache). The iteration
//! `u ← u + K⁻¹r(u)` is a **preconditioned fixed point**: the "residual
//! direction" `g(u) = K⁻¹ r(u)` vanishes at convergence — it is the natural
//! residual of the fixed point, and it is **already computed** at every
//! iteration (`du = solve(&k, &residual)`).
//!
//! The price of the constant operator is a merely **linear** convergence on
//! the plastic branch (many iterations). Anderson acceleration exploits the
//! history of the last `m = 3` pairs `(u, g)` to extrapolate a much better
//! step, **without re-evaluating the behaviour law**: the small least-squares
//! problem only handles dot products of fields already in hand.
//!
//! Anderson acceleration (m = 3)
//! -----------------------------
//! At iteration `k`, with the history of the last `m ≤ 3` pairs `(uᵢ, gᵢ)`
//! (most recent first):
//!
//! 1. Differences `ΔGⱼ = g_{k-j+1} − g_{k-j}`, `ΔUⱼ = u_{k-j+1} − u_{k-j}`.
//! 2. Least squares `min_γ ‖g_k − Σⱼ γⱼ ΔGⱼ‖²` → normal system `(ΔGᵀΔG) γ =
//!    ΔGᵀg_k`, whose entries are all dot products **over the free DOFs** (the
//!    same DOFs as the residual norm), Tikhonov-regularised.
//! 3. Small dense `m×m` solve (m ≤ 3, Gaussian elimination).
//! 4. Extrapolated step `u_acc = u_k + g_k − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.
//!
//! Descent safeguard: the residual of the Anderson candidate **and** that of
//! the pure modified-Newton step `u_k + g_k` are both evaluated, and the
//! **better of the two** is kept. Anderson can therefore never degrade the
//! convergence relative to the original modified Newton; if it is rejected,
//! the history is cleared.

// The parentheses in `|| (&u + &g)` are necessary: without them, `|| &u + &g`
// parses as `(|| &u) + &g` (E0369). The lint is therefore silenced for this file.
#![allow(unused_parens)]

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::Band;
use pyrucast::atoms::ElementType;
use pyrucast::atoms::Node;
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::evolution::{
    Evolution, Interpolated, OutOfRange, SubEvolution, SubValue,
};
use pyrucast::containers::field::{Field, SubField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::Mesh;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::behavior::integrate;
use pyrucast::ops::element_field::deformation;
use pyrucast::ops::element_field::mask;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::matrix::stiffness;
use pyrucast::ops::mesh::select_nodes;
use pyrucast::ops::mesh::{line, sweep, to_poi1, translate};
use pyrucast::ops::model;
use pyrucast::ops::node_field::{external_forces, internal_forces};
use pyrucast::ops::node_field::{positions, restrict, restrict_like};
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

/// Depth of the Anderson history (number of `(u, g)` pairs kept).
const ANDERSON_DEPTH: usize = 3;

// ── Profiling: cumulated time per sub-operator (ns), thread-safe ───────────
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static T_DEFORM: AtomicU64 = AtomicU64::new(0);
static T_BEHAVIOR: AtomicU64 = AtomicU64::new(0);
static T_FINT: AtomicU64 = AtomicU64::new(0);
static T_RESIDUAL: AtomicU64 = AtomicU64::new(0);
static T_SOLVE: AtomicU64 = AtomicU64::new(0);
static T_ANDERSON: AtomicU64 = AtomicU64::new(0);
static T_FIELDOPS: AtomicU64 = AtomicU64::new(0); // restrict_like, map_all, +/- on fields
static T_DIAG: AtomicU64 = AtomicU64::new(0); // plastic_diagnostics
static T_RESID_WALL: AtomicU64 = AtomicU64::new(0); // residual_at end to end
static N_RESID_EVAL: AtomicU64 = AtomicU64::new(0);

fn tic<T>(acc: &AtomicU64, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    acc.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    // ── Parameters (steel material, geometry, loading) ──────────────────────
    let (young, nu, sigma_y) = (210_000.0_f64, 0.3_f64, 250.0_f64);
    let (length, height) = (10.0_f64, 1.0_f64);
    let nx = env_usize("PYRUCAST_NX", 40);
    let ny = env_usize("PYRUCAST_NY", 8);
    let nsteps = env_usize("PYRUCAST_NSTEPS", 10);
    let p_max = env_f64("PYRUCAST_PMAX", 5.0); // final shear force at the tip

    println!(
        "Plastic cantilever beam (Anderson m={ANDERSON_DEPTH}): {nx}×{ny} QUA4  \
         (L={length}, H={height}), E={young}, ν={nu}, σy={sigma_y}"
    );
    println!("Loading: 0 → {p_max} in {nsteps} steps (modified Newton + Anderson acceleration)\n");

    // ── Mesh: grid of nodes (j through the height, i along the length), QUA4 cells ──
    println!(
        "▸ Mesh: {} nodes, {} QUA4 cells…",
        (nx + 1) * (ny + 1),
        nx * ny
    );
    let t_mesh0 = Instant::now();
    let coords = Handle::new(Coords::new(2)?);
    let pt_a = Node::create_in(coords.clone(), &[0., 0.])?;
    let pt_b = Node::create_in(coords.clone(), &[0., height])?;
    let pt_c = Node::create_in(coords.clone(), &[length, 0.])?;
    let pt_d = Node::create_in(coords.clone(), &[length, height])?;
    let left_edge = line(&pt_a, &pt_b, ny, ElementType::SEG2)?;
    let right_edge = line(&pt_c, &pt_d, ny, ElementType::SEG2)?;
    let mesh = sweep(&left_edge, &right_edge, nx, ElementType::QUA4)?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let t_mesh = t_mesh0.elapsed();

    // Useful node sets: tip (mid-height), and a POI1 mesh of the FREE nodes
    // (outside the clamped end) — target support for measuring the residual norm
    // on the free DOFs only (`restrict` + `xtx`). The free nodes are those with a
    // strictly positive X coordinate (band on the `positions` field).
    let tip = &mesh.nearest_node(&[length, height / 2.])?;
    let coords_field = positions(&mesh, Some(vec!["X".into()]))?;
    let free_mesh = select_nodes(
        &coords_field,
        &Band::new(Some(length / nx as f64 / 2.), None, None, None)?,
        None,
    )?;

    // ── Model: plasticity (plane stress) + clamped end (Dirichlet) ──────────
    println!("▸ Model: J2 plasticity (plane stress) + clamped end…");
    let mut model = model::plasticity_perfect(&fes, Kinematics::PlaneStress)?;
    let imposed_mesh = to_poi1(&left_edge)?;
    let multiplier = translate(&imposed_mesh, &[0., 0.])?;
    model = model.union(&model::dirichlet(
        &model,
        "u_x",
        &imposed_mesh,
        &multiplier,
        Default::default(),
    )?)?;
    model = model.union(&model::dirichlet(
        &model,
        "u_y",
        &imposed_mesh,
        &multiplier,
        Default::default(),
    )?)?;

    // The reference load is a term of the model: it joins the model, and its
    // density joins the material.
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    let model = model.union(&pyrucast::ops::model::flux(
        &right_fes,
        &model,
        "f_y".into(),
    )?)?;
    let materials = material_field(
        &model,
        &[
            ("E", young),
            ("nu", nu),
            ("sigma_y", sigma_y),
            ("phi_f_y", -1.0),
        ],
    )?;

    // ELASTIC stiffness: iteration operator of the modified Newton. Assembled
    // once; `solve` caches the factorisation and reuses it at every
    // forward/back substitution.
    println!("▸ Assembling the elastic stiffness K…");
    let t_k0 = Instant::now();
    let k = stiffness(&model, &materials)?;
    let t_k = t_k0.elapsed();

    // ── Reference load: unit shear (density −1) on the right face, spread into
    //    consistent nodal forces (∫ density·N dΓ, `flux` op). ─────────────────
    println!("▸ Reference load + loading history…");
    // `external_forces` returns an aggregate; the loading history is tabulated
    // zone by zone.
    let load_unit = external_forces(&model, &materials)?.get(0)?.read().clone();

    // ── Loading history: an Evolution with FIELD values, tabulated against the
    //    pseudo-time t ∈ [0, 1]. Two keyframes of the nodal force field — zero at
    //    t=0, complete (`p_max · unit_load`) at t=1 — on the SAME support. ─────
    let zero_frame = load_unit.map_all(|_| 0.0);
    let full_frame = load_unit.map_all(|v| v * p_max);
    let load_curve = SubEvolution::new(
        vec![
            (0.0, SubValue::Node(zero_frame)),
            (1.0, SubValue::Node(full_frame)),
        ],
        OutOfRange::Clamp,
    )?;
    let mut load_evo = Evolution::default();
    load_evo.add_sub(Handle::new(load_curve))?;

    // ── Simulation state (persistent across the steps) ──────────────────────
    // Accumulated displacement u (u_x, u_y on every node), initially zero.
    let mut u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
    // Converged state of the previous step (VAR0 = `prev`): `None` at the first step.
    let mut state: Option<ElementField> = None;

    // ── Loop over the load steps ────────────────────────────────────────────
    let max_newton = 200;
    println!("▸ Solving: {nsteps} load steps (modified Newton + Anderson)\n");
    println!(
        "{:>4} {:>8} {:>6} {:>6} {:>14} {:>14} {:>8}",
        "step", "P", "iter", "andrs", "deflection u_y", "p_max", "n_plast"
    );

    let mut prev_defl = 0.0_f64;
    let mut any_plasticity = false;

    let t_loop0 = Instant::now();
    for step in 1..=nsteps {
        // Pseudo-time of the step ∈ ]0, 1]; the external load follows from it by
        // interpolation of the Evolution (nodal force field of the step).
        let t = step as f64 / nsteps as f64;
        let load_p = p_max * t; // nominal shear at the tip (for the display)
        let Interpolated::Node(load_scaled) = load_evo.interpolate(t, None)? else {
            unreachable!("evolution with nodal values")
        };
        // Norm of the step load (relative scale of the residual): xᵀx of the field.
        let ext_norm = load_scaled.xtx().sqrt();
        let tol = 1e-6 * ext_norm + 1e-12;

        // Modified Newton + Anderson: iterate until the residual (out-of-balance
        // forces at the free DOFs) is negligible. `last_state` keeps the
        // converged behaviour output, source of the new VAR0.
        let mut iters = 0;
        let mut n_anderson = 0; // how many steps were actually accelerated
        let mut last_state: Option<ElementField>;
        let mut res_norm;

        // Anderson history: pairs (u, g=K⁻¹r) of the current step, most recent
        // first. Cleared at the beginning of each load step.
        let mut history: Vec<(NodeField, NodeField)> = Vec::with_capacity(ANDERSON_DEPTH + 1);

        // Residual (and converged internal forces) at a trial displacement `u`:
        // ε(u) → COMP → BSIG → r = F_ext − F_int, plus the norm on the free DOFs.
        // No nodal loop (field operators only).
        let residual_at = |u: &NodeField| -> Result<(NodeField, f64, ElementField)> {
            tic(&T_RESID_WALL, || {
                N_RESID_EVAL.fetch_add(1, Ordering::Relaxed);
                let strain = tic(&T_DEFORM, || deformation(u, &fes))?;
                // Behaviour A→B: ε(B) directly, state of A in `prev`.
                let out = tic(&T_BEHAVIOR, || {
                    integrate(&model, &strain, state.as_ref(), &materials, None)
                })?;
                let f_int = tic(&T_FINT, || internal_forces(&model, &out, u, &materials))?;
                // Residual r = F_ext − F_int and its norm on the free DOFs.
                let (residual, free_res) = tic(&T_RESIDUAL, || -> Result<_> {
                    let f_ext = restrict_like(&load_scaled, &f_int)?;
                    let residual = (&f_ext - &f_int)?;
                    let free_res = restrict(&residual, &free_mesh)?.xtx().sqrt();
                    Ok((residual, free_res))
                })?;
                Ok((residual, free_res, out))
            })
        };

        loop {
            // Residual at the current displacement (= fixed point `g = K⁻¹r`).
            let (residual, cur_res, out) = residual_at(&u)?;
            res_norm = cur_res;
            last_state = Some(out);

            if res_norm <= tol || iters >= max_newton {
                break;
            }

            // Residual direction g = K⁻¹ r (elastic K, factorisation cache). The
            // support of δu already coincides with that of u (same hidden POI1
            // companion from `to_poi1`); `restrict_like` only filters out the dual
            // components (multipliers) — otherwise they would be copied into u by
            // union.
            let du = tic(&T_SOLVE, || solve(&k, &residual))?;
            let g = tic(&T_FIELDOPS, || restrict_like(&du, &u))?;

            // Snapshot of the current (u, g) pair BEFORE moving — source of the
            // Anderson differences at the next round. `map_all(|v| v)` = deep copy
            // of the aggregate (NodeField is not Clone at the aggregate level).
            let u_snapshot = tic(&T_FIELDOPS, || u.map_all(|v| v))?;
            let pure_step = tic(&T_FIELDOPS, || (&u + &g))?; // modified Newton step

            // Anderson candidate (if the history carries at least one pair):
            // extrapolation over the last m (u, g). Descent safeguard: it is kept
            // only if it **strictly** reduces the current residual `cur_res` —
            // otherwise the pure step is taken, whose residual will be evaluated
            // for free at the top of the next round (no wasted evaluation).
            let mut chose_anderson = false;
            let mut next_u = None;
            if !history.is_empty()
                && let Some(corr) =
                    tic(&T_ANDERSON, || anderson_step(&u, &g, &history, &free_mesh))?
            {
                let u_acc = tic(&T_FIELDOPS, || (&pure_step - &corr))?;
                let (_, res_acc, _) = residual_at(&u_acc)?;
                if res_acc < cur_res {
                    next_u = Some(u_acc);
                    chose_anderson = true;
                }
            }

            // History: if Anderson was kept, push and truncate to the depth;
            // otherwise start over cleanly (history cleared) so as not to drag
            // along directions that do not help.
            if chose_anderson {
                n_anderson += 1;
                history.insert(0, (u_snapshot, g));
                history.truncate(ANDERSON_DEPTH);
            } else {
                history.clear();
                history.push((u_snapshot, g));
            }
            u = next_u.unwrap_or(pure_step);
            iters += 1;
        }
        let converged = res_norm <= tol;

        // State commit: `prev` ← VAR1. The converged output carries the complete
        // state of B and becomes the `prev` (state of A) of the next step.
        let committed = last_state.take().expect("at least one residual evaluation");

        // Diagnostics of the step.
        let (p_max_val, n_plastic) = tic(&T_DIAG, || plastic_diagnostics(&committed))?;
        state = Some(committed);
        let defl = u.value(tip.id(), "u_y")?;
        any_plasticity |= n_plastic > 0;
        let flag = if converged {
            ""
        } else {
            "  (residual left over)"
        };
        println!(
            "{step:>4} {load_p:>8.3} {iters:>6} {n_anderson:>6} \
             {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        );

        // The deflection grows (in absolute value, downwards) with the load.
        assert!(
            defl.abs() >= prev_defl.abs() - 1e-9,
            "non-monotonic deflection at step {step}"
        );
        prev_defl = defl;
    }
    let t_loop = t_loop0.elapsed();

    // ── Profiling report ────────────────────────────────────────────────────
    let ms = |ns: u64| ns as f64 / 1e6;
    let n_eval = N_RESID_EVAL.load(Ordering::Relaxed);
    let deform = T_DEFORM.load(Ordering::Relaxed);
    let behav = T_BEHAVIOR.load(Ordering::Relaxed);
    let fint = T_FINT.load(Ordering::Relaxed);
    let resid = T_RESIDUAL.load(Ordering::Relaxed);
    let slv = T_SOLVE.load(Ordering::Relaxed);
    let andr = T_ANDERSON.load(Ordering::Relaxed);
    let fops = T_FIELDOPS.load(Ordering::Relaxed);
    let diag = T_DIAG.load(Ordering::Relaxed);
    let resid_wall = T_RESID_WALL.load(Ordering::Relaxed);
    let resid_total = deform + behav + fint + resid;
    let resid_inner_gap = resid_wall.saturating_sub(resid_total);
    let accounted = resid_wall + slv + andr + fops + diag;
    let unaccounted = (t_loop.as_nanos() as u64).saturating_sub(accounted);
    println!("\n──────── Profile (cumulated times) ────────");
    println!(
        "  mesh + fes            : {:>9.1} ms",
        ms(t_mesh.as_nanos() as u64)
    );
    println!(
        "  assembly of K         : {:>9.1} ms",
        ms(t_k.as_nanos() as u64)
    );
    println!(
        "  solution loop         : {:>9.1} ms",
        ms(t_loop.as_nanos() as u64)
    );
    println!("  ── of which, inside the loop:");
    println!(
        "     residual_at (×{n_eval:>4})  : {:>9.1} ms wall",
        ms(resid_wall)
    );
    println!(
        "        └ deform {:.0} + behaviour {:.0} + f_int {:.0} + residual {:.0} + inner gap {:.0}",
        ms(deform),
        ms(behav),
        ms(fint),
        ms(resid),
        ms(resid_inner_gap)
    );
    println!("     solve K⁻¹r           : {:>9.1} ms", ms(slv));
    println!("     anderson_step        : {:>9.1} ms", ms(andr));
    println!("     field ops (restrict/copy/±): {:>4.1} ms", ms(fops));
    println!("     plastic diagnostics   : {:>9.1} ms", ms(diag));
    println!("     unaccounted           : {:>9.1} ms", ms(unaccounted));
    println!("────────────────────────────────────────");

    // Beyond first yield, a plastic zone must appear.
    let p_first_yield = sigma_y * (height * height / 6.0) / length;
    if p_max > p_first_yield {
        assert!(
            any_plasticity,
            "P_max={p_max} exceeds first yield (≈{p_first_yield:.2}) \
             but no plastic point was detected"
        );
        println!("\nOK: plasticity developed (P_max={p_max} > P_elastic≈{p_first_yield:.2}).");
    } else {
        println!("\nOK: the response stayed elastic (P_max={p_max} ≤ ≈{p_first_yield:.2}).");
    }
    Ok(())
}

/// Anderson acceleration step: from the current displacement `u`, its residual
/// direction `g = K⁻¹r(u)`, and the history of the last `m ≤ 3` pairs
/// `(uᵢ, gᵢ)` (most recent first), computes the **correction**
/// `Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)` to subtract from the pure Newton step `u + g`:
///
/// `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.
///
/// The `γ` solve the least-squares problem `min ‖g − Σⱼ γⱼ ΔGⱼ‖²` over the
/// **free** DOFs (`free_mesh`), through the regularised (Tikhonov) normal
/// equations `(ΔGᵀΔG) γ = ΔGᵀg`. Returns `None` if the history is empty or if
/// the small system degenerates (the caller then falls back on the pure
/// Newton step).
///
/// Everything goes through the field operators (`-`, `dot_field`, `restrict`):
/// the dot products are the only reductions, and the behaviour law is never
/// evaluated.
fn anderson_step(
    u: &NodeField,
    g: &NodeField,
    history: &[(NodeField, NodeField)],
    free_mesh: &Mesh,
) -> Result<Option<NodeField>> {
    let m = history.len();
    if m == 0 {
        return Ok(None);
    }

    // Differences ΔUⱼ = u_{present} − u_{history}, ΔGⱼ = g − g_{history}.
    // (A convention equivalent to successive differences up to a global sign,
    // absorbed by γ; here the differences are taken towards the current iterate.)
    let mut du_diffs: Vec<NodeField> = Vec::with_capacity(m);
    let mut dg_diffs: Vec<NodeField> = Vec::with_capacity(m);
    for (u_hist, g_hist) in history {
        du_diffs.push((u - u_hist)?);
        dg_diffs.push((g - g_hist)?);
    }

    // ΔG restricted to the free DOFs (support of the residual dot products).
    let dg_free: Vec<NodeField> = dg_diffs
        .iter()
        .map(|d| restrict(d, free_mesh))
        .collect::<Result<_>>()?;
    let g_free = restrict(g, free_mesh)?;

    // Normal equations (ΔGᵀΔG) γ = ΔGᵀg (small symmetric m×m system).
    let mut a = vec![vec![0.0_f64; m]; m];
    let mut b = vec![0.0_f64; m];
    let mut trace = 0.0;
    for i in 0..m {
        for j in i..m {
            let v = dg_free[i].dot_field(&dg_free[j])?;
            a[i][j] = v;
            a[j][i] = v;
        }
        trace += a[i][i];
        b[i] = dg_free[i].dot_field(&g_free)?;
    }
    if trace <= 0.0 {
        return Ok(None); // degenerate directions
    }
    // Tikhonov regularisation: + λ·(trace/m) on the diagonal.
    let lambda = 1e-10 * trace / m as f64;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += lambda;
    }

    let Some(gamma) = solve_small_spd(a, b) else {
        return Ok(None);
    };

    // Correction Σⱼ γⱼ (ΔUⱼ + ΔGⱼ), assembled through the field operators.
    let mut corr: Option<NodeField> = None;
    for (j, gj) in gamma.iter().enumerate() {
        let term = (&(&du_diffs[j] + &dg_diffs[j])? * *gj)?;
        corr = Some(match corr {
            None => term,
            Some(acc) => (&acc + &term)?,
        });
    }
    Ok(corr)
}

/// Solves a small dense **symmetric** system `A x = b` (`m ≤ 3`) by Gaussian
/// elimination with partial pivoting. Returns `None` if `A` is singular
/// (pivot ~ 0) — the caller then falls back on the pure Newton step.
fn solve_small_spd(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivoting.
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-30 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // Elimination (the pivot row is copied to avoid the double borrow).
        let pivot_row = a[col].clone();
        let b_pivot = b[col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot_row[col];
            for (dst, &src) in a[r].iter_mut().zip(pivot_row.iter()).skip(col) {
                *dst -= factor * src;
            }
            b[r] -= factor * b_pivot;
        }
    }
    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// `(p_max, number of yielded Gauss points)` of the current state (`p > 0`
/// marks a plastic point).
fn plastic_diagnostics(state: &ElementField) -> Result<(f64, usize)> {
    let p_max = Field::max(state, Some("p"))?;
    let band = Band::new(None, Some(1e-12), None, None)?;
    let masked = mask(state, &band, Some(vec!["p".to_string()]))?;
    let n_plastic = Field::sum(&masked, "p")?.round() as usize;
    Ok((p_max, n_plastic))
}
