//! Elasto-plastic cantilever beam — modified Newton **accelerated by Anderson
//! acceleration (m = 3)** on top of the pyrucast building blocks.
//!
//! A variant of [`plasticite_poutre_console`]: same physics, same mesh, same
//! pyrucast operators. Only the non-linear loop changes — the original is kept
//! **frozen** as the reference executable to compare results against (the
//! deflections and the plasticity must coincide; only the iteration count
//! should drop).
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
    let coords = Handle::new(Coords::new(2)?);
    let pt_a = Node::create_in(coords.clone(), &[0., 0.])?;
    let pt_b = Node::create_in(coords.clone(), &[0., height])?;
    let pt_c = Node::create_in(coords.clone(), &[length, 0.])?;
    let pt_d = Node::create_in(coords.clone(), &[length, height])?;
    let left_edge = line(&pt_a, &pt_b, ny, ElementType::SEG2)?;
    let right_edge = line(&pt_c, &pt_d, ny, ElementType::SEG2)?;
    let mesh = sweep(&left_edge, &right_edge, nx, ElementType::QUA4)?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

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
    let k = stiffness(&model, &materials)?;

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
    // Converged state of the previous step (VAR0 = `prev`): `None` at the first
    // step — A is then the reference configuration (σ(A)=0, ε(A)=0).
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
            // ε(u)=ε(B), state of A in `prev` → σ, VAR1 (COMP) → F_int (BSIG).
            let strain = deformation(u, &fes)?;
            let out = integrate(&model, &strain, state.as_ref(), &materials, None)?;
            let f_int = internal_forces(&model, &out, u, &materials)?;
            // Residual r = F_ext − F_int and its norm on the free DOFs (field
            // operators only: `restrict_like`, `-`, `restrict`, `xtx`).
            let f_ext = restrict_like(&load_scaled, &f_int)?;
            let residual = (&f_ext - &f_int)?;
            let free_res = restrict(&residual, &free_mesh)?.xtx().sqrt();
            Ok((residual, free_res, out))
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
            let du = solve(&k, &residual)?;
            let g = restrict_like(&du, &u)?;

            // Snapshot of the current (u, g) pair BEFORE moving — source of the
            // Anderson differences at the next round. `map_all(|v| v)` = deep copy
            // of the aggregate (NodeField is not Clone at the aggregate level).
            let u_snapshot = u.map_all(|v| v)?;
            let pure_step = (&u + &g)?; // modified Newton step (reference)

            // Anderson candidate (if the history carries at least one pair):
            // extrapolation over the last m (u, g). Descent safeguard: it is kept
            // only if it **strictly** reduces the current residual `cur_res` —
            // otherwise the pure step is taken, whose residual will be evaluated
            // for free at the top of the next round (no wasted evaluation).
            let mut chose_anderson = false;
            let mut next_u = None;
            if !history.is_empty()
                && let Some(corr) = anderson_step(&u, &g, &history, &free_mesh)?
            {
                let u_acc = (&pure_step - &corr)?;
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
        // state of B (σ(B), ε_p(B), p(B), ε(B)) and becomes the `prev` (state of
        // A) of the next step (the law reads its inputs by name).
        let committed = last_state.take().expect("at least one residual evaluation");

        // Diagnostics of the step.
        let (p_max_val, n_plastic) = plastic_diagnostics(&committed)?;
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
