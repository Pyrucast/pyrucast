//! Elasto-plastic cantilever beam — a hand-rolled Newton loop on top of the
//! pyrucast building blocks (Rust example, written for the parallelism bench).
//!
//! Physics
//! -------
//! 2-D plane-stress continuum, small strains. Perfect von Mises plasticity
//! (J2 radial return, no hardening): the equivalent stress is capped at
//! `sigma_y`. Beam clamped at the left end (`u_x = u_y = 0`), sheared
//! downwards on the right face. The load is raised by increments; beyond
//! first yield a plastic zone develops near the clamped end and the
//! deflection departs from the linear response.
//!
//! What pyrucast does vs. what the example does
//! --------------------------------------------
//! **pyrucast** knows NOTHING about Newton. It provides the pointwise
//! operators:
//!
//! - [`stiffness`]: the **elastic** stiffness `K` (iteration operator);
//! - [`deformation`]: the strain `ε = ½(∇u + ∇uᵀ)` at the Gauss points;
//! - [`integrate`] (Cast3m `COMP`): the behaviour law at the point — here the
//!   radial return, which yields `σ` and the updated plastic state
//!   (`VAR0` → `VAR1`);
//! - [`internal_forces`] (Cast3m `BSIG`): the internal forces `∫ Bᵀ σ dΩ`;
//! - [`solve`]: the linear solve (faer sparse LU, cached factorisation);
//! - **field arithmetic** (`+ - * /`) and [`restrict_like`] (reprojection of a
//!   field onto the support/components of another), which replace every nodal
//!   loop: `residual = &f_ext - &f_int`, `u = (&u + &δu_reprojected)?`;
//! - an [`Evolution`] with field values for the **loading history**: the load
//!   of each step is interpolated at the pseudo-time ([`Evolution::interpolate`]).
//!
//! **The example** assembles its own Newton loop out of those blocks:
//! residual `r = F_ext − F_int`, increment `δu = K⁻¹ r`, `u ← u + δu`, and the
//! carry-over of the internal state from one load step to the next. This is a
//! **modified Newton** (constant operator = elastic `K`): `K` is assembled and
//! factorised once, each iteration only redoes a forward/back substitution
//! ([`solve`]'s factorisation cache). No loop over the nodes: the whole balance
//! goes through the field operators and the library primitives.
//!
//! Parallelism bench
//! -----------------
//! The parallelised hot loops (assembly, `deformation`, `integrate`,
//! `internal_forces`) are re-evaluated at every Newton iteration. Vary the mesh
//! size and the thread count:
//!
//! ```text
//! RAYON_NUM_THREADS=1 PYRUCAST_NX=200 PYRUCAST_NY=40 \
//!     cargo run --release --example plasticite_poutre_console
//! RAYON_NUM_THREADS=8 PYRUCAST_NX=200 PYRUCAST_NY=40 \
//!     cargo run --release --example plasticite_poutre_console
//! ```
//!
//! Environment variables: `PYRUCAST_NX`, `PYRUCAST_NY` (cells along the length /
//! through the height), `PYRUCAST_NSTEPS` (load steps), `PYRUCAST_PMAX` (final
//! load, shear force at the tip).

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
        "Plastic cantilever beam: {nx}×{ny} QUA4  (L={length}, H={height}), \
         E={young}, ν={nu}, σy={sigma_y}"
    );
    println!("Loading: 0 → {p_max} in {nsteps} steps (modified Newton, elastic K)\n");

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

    // grid[j][i]: node at (x = i·L/nx, y = j·H/ny).
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Useful node sets: left edge (clamped), tip (mid-height), and a POI1 mesh of
    // the FREE nodes (outside the clamped end) — target support for measuring the
    // residual norm on the free DOFs only (`restrict` + `xtx`).
    let tip_id = &mesh.nearest_node(&[length, height / 2.])?;
    let coords_field = positions(&mesh, Some(vec!["X".into()]))?;
    let free_mesh = select_nodes(
        &coords_field,
        &Band::new(Some(length / nx as f64 / 2.), None, None, None)?,
        None,
    )?;
    let imposed_mesh = to_poi1(&left_edge)?;
    let multiplier = translate(&imposed_mesh, &[0., 0.])?;

    // ── Model: plasticity (plane stress) + clamped end (Dirichlet) ──────────
    println!("▸ Model: J2 plasticity (plane stress) + clamped end…");
    let mut model = model::plasticity_perfect(&fes, Kinematics::PlaneStress)?;

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
    //    t=0, complete (`p_max · unit_load`) at t=1 — on the SAME support (both
    //    derived from the same sub-field, a condition of the interpolation). The
    //    load of each step is read by linear interpolation,
    //    `load_evo.interpolate(t)`. A non-linear history would only add
    //    keyframes. ─────────────────────────────────────────────────────────────
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
    // Modified Newton (operator = elastic K): linear convergence, hence slow on
    // the plastic branch. Iterations are capped high and we aim at a relative
    // residual of 1e-6 (largely enough here).
    let max_newton = 200;
    println!("▸ Solving: {nsteps} load steps (modified Newton)\n");
    println!(
        "{:>4} {:>8} {:>6} {:>14} {:>14} {:>8}",
        "step", "P", "iter", "deflection u_y", "p_max", "n_plast"
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

        // Modified Newton: iterate until the residual (out-of-balance forces at
        // the free DOFs) is negligible. `last_state` keeps the converged
        // behaviour output, source of the new VAR0.
        let mut iters = 0;
        let mut last_state: Option<ElementField> = None;
        let mut res_norm = f64::INFINITY;

        for _ in 0..max_newton {
            // ε(u) = ε(B), state of A in `prev` → σ, VAR1 (COMP), A→B rise.
            let strain = deformation(&u, &fes)?;
            let out = integrate(&model, &strain, state.as_ref(), &materials, None)?;
            // Internal forces F_int = ∫ Bᵀ σ dΩ (BSIG).
            let f_int = internal_forces(&model, &out, &u, &materials)?;

            // Residual r = F_ext − F_int and its norm on the **free** DOFs, with
            // no nodal loop at all — everything through the operators and the
            // primitives:
            // - `f_ext` = external load of the step (`load_scaled`, already
            //   scaled on `f_y`) reprojected onto the support AND the components
            //   of `f_int` (`restrict_like`): components `f_x` (=0) and `f_y`;
            // - `residual = f_ext − f_int` through the `-` operator;
            // - the norm is read on the free nodes only: `residual` `restrict`ed
            //   to `free_mesh` then `xtx` (the clamped nodes carry the reaction).
            let f_ext = restrict_like(&load_scaled, &f_int)?;
            let residual = (f_ext - f_int)?;
            res_norm = restrict(&residual, &free_mesh)?.xtx().sqrt();
            last_state = Some(out);

            if res_norm <= tol {
                break;
            }
            // δu = K⁻¹ r (elastic K, factorisation cache). δu carries the primal
            // AND the dual DOFs (Lagrange multipliers). Its support already
            // coincides with that of u (same hidden POI1 companion from
            // `to_poi1`, shared by `solve` and `NodeField::new(&mesh)`); so
            // `restrict_like` only serves to **filter out the dual components** —
            // otherwise `u + δu` would copy the multipliers into u by union.
            // Then u ← u + δu through `+`.
            let du = solve(&k, &residual)?;
            u = (&u + &restrict_like(&du, &u)?)?;
            iters += 1;
        }
        let converged = res_norm <= tol;

        // State commit: `prev` ← VAR1. The converged behaviour output carries the
        // complete state of B (σ(B), ε_p(B), p(B), ε(B)) and becomes the `prev`
        // (state of A) of the next step. The law reads its inputs by name, so the
        // extra components are ignored.
        let committed = last_state.take().expect("at least one iteration");

        // Diagnostics of the step.
        let (p_max_val, n_plastic) = plastic_diagnostics(&committed)?;
        state = Some(committed);
        let defl = u.value(tip_id.id(), "u_y")?;
        any_plasticity |= n_plastic > 0;
        let flag = if converged {
            ""
        } else {
            "  (residual left over)"
        };
        println!(
            "{step:>4} {load_p:>8.3} {iters:>6} {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        );

        // The deflection grows (in absolute value, downwards) with the load.
        assert!(
            defl.abs() >= prev_defl.abs() - 1e-9,
            "non-monotonic deflection at step {step}"
        );
        prev_defl = defl;
    }

    // Beyond first yield, a plastic zone must appear.
    // (Elastic bounds: first yield around P ≈ σy·I/(c·L).)
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

/// `(p_max, number of yielded Gauss points)` of the current state (`p > 0`
/// marks a plastic point). Without a loop: `p_max` through [`Field::max`], the
/// count by masking the `p` component into 0/1 (band "> 1e-12") then summing it
/// ([`Field::sum`]).
fn plastic_diagnostics(state: &ElementField) -> Result<(f64, usize)> {
    let p_max = Field::max(state, Some("p"))?;
    let band = Band::new(None, Some(1e-12), None, None)?;
    let masked = mask(state, &band, Some(vec!["p".to_string()]))?;
    let n_plastic = Field::sum(&masked, "p")?.round() as usize;
    Ok((p_max, n_plastic))
}
