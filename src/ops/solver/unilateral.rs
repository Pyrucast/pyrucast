//! Unilateral (inequality) constraint solve by the **active-set** (status)
//! method — the operator for models whose constraints carry a non-equality
//! [`RelationSense`].
//!
//! A unilateral relation `Σₖ aₖ·u(nodeₖ,varₖ) ≥ g` (or `≤ g`) obeys the KKT
//! complementarity conditions instead of a plain equation: either the relation
//! is **active** (it holds as an equality and its multiplier `λ` carries the
//! reaction) or it is **inactive** (the gap `C·u − g` is on the feasible side
//! and `λ = 0`). Which relations are active is not known in advance; the status
//! method iterates on it:
//!
//! 1. start from a trial active set (every inequality active, or the previous
//!    converged status when cached — a *warm start*);
//! 2. solve the saddle-point system with the **inactive** constraint rows
//!    replaced by `λ_r = 0` (the matrix keeps its size; only values change);
//! 3. check the signs: an active relation whose `λ` pulls (sign infeasible for
//!    its sense) is released; an inactive relation whose gap penetrates is
//!    activated;
//! 4. no status change ⇒ converged (the finite loop of the classical status
//!    method); otherwise refactorize and repeat.
//!
//! # Sign convention
//!
//! The assembled saddle-point reads `K·u + Cᵀ·λ = f`, `C·u = g` (see
//! [`constraint_block_pair`](crate::models)); against the KKT multiplier `μ ≥ 0`
//! of a `≥` constraint (`K·u − Cᵀ·μ = f`) this gives `λ = −μ`. Hence, in the
//! solution field (same convention as the Lagrange path):
//!
//! - `GreaterEqual` — an active relation has `λ ≤ 0`; released when `λ > tol`;
//! - `LessEqual` — an active relation has `λ ≥ 0`; released when `λ < −tol`.
//!
//! An inactive relation reports `λ = 0` exactly (its row is the identity on the
//! multiplier DOF).
//!
//! # Method-neutral input
//!
//! Like [`eliminate`](super::eliminate), the constraint structure is read from
//! [`Constraint::relations()`](crate::models::Constraint::relations) — the same
//! seam the Lagrange blocks are built from — so the user's mesh-per-term input
//! is never re-parsed here. Equality relations are left untouched (their rows
//! are always enforced); a model with **no** inequality falls back to a plain
//! [`lu::solve`].

use crate::containers::matrix::Matrix;
use crate::containers::mesh::NodeId;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::models::RelationSense;
use crate::store::read;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use super::lu::{self, lu_solve_vec, SolveMethod, SolveOptions, SparseLu};

type NamedDof = (NodeId, String);

/// How the active-set loop assembles each status's system.
///
/// Both variants explore the **same** status trajectory (identical KKT checks,
/// same converged result) — they differ only in how the linear system of a
/// status is factorized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActiveSetMethod {
    /// Factorize the **inequality-free base** `A` (every unilateral relation
    /// released) **once**, cache it on the matrix, and obtain each status by a
    /// dense Schur/Woodbury update (the *Delassus* operator) — no sparse
    /// refactorization per status change. Falls back automatically to
    /// [`Refactorize`](Self::Refactorize) when `A` is singular (a structure that
    /// does not hold without contact, e.g. a body merely resting on a support).
    #[default]
    SchurComplement,
    /// Refactorize the full sparse saddle-point at each status change. The
    /// original method: robust to a singular base, but pays one sparse
    /// factorization per iteration.
    Refactorize,
}

/// Options for [`solve_with_options`]. `method` / `cache` mirror
/// [`SolveOptions`]; `max_iter` bounds the status loop; `tol` is the
/// complementarity tolerance on both the multiplier sign and the gap sign
/// (absolute, in the units of the reaction / of the relation).
#[derive(Clone, Copy, Debug)]
pub struct UnilateralOptions {
    /// Direct back-end for each status iteration's system.
    pub method: SolveMethod,
    /// How each status's system is factorized (Schur base reuse vs. refactorize).
    pub active_set: ActiveSetMethod,
    /// Cache the active-set state (layout + last converged status and, per
    /// method, its factorization / the base factorization and Schur columns) on
    /// the matrix: a re-solve warm-starts from the previous status and reuses the
    /// cached factors. Cleared automatically when the matrix changes.
    pub cache: bool,
    /// Maximum number of status iterations.
    pub max_iter: usize,
    /// Sign tolerance for releasing (`λ` past `tol`) and activating (gap past
    /// `−tol`) a relation.
    pub tol: f64,
}

impl Default for UnilateralOptions {
    fn default() -> Self {
        Self {
            method: SolveMethod::Lu,
            active_set: ActiveSetMethod::default(),
            cache: true,
            max_iter: 100,
            tol: 1e-10,
        }
    }
}

/// One inequality relation, resolved against the matrix DOF layout.
struct Inequality {
    /// Row index of the constraint equation `(multiplier_node, imposed_value)`.
    row: usize,
    /// Column index of the multiplier DOF `(multiplier_node, multiplier_var)`.
    lambda_col: usize,
    /// `(column, coefficient)` of every term — evaluates the gap `C·u − g`.
    term_cols: Vec<(usize, f64)>,
    /// `GreaterEqual` or `LessEqual` (equalities are not collected).
    sense: RelationSense,
}

/// The Schur/Delassus artifacts cached across solves (matrix-invariant): the LU
/// of the inequality-free base `A` and, built lazily, the columns
/// `xᵣ = A⁻¹·e_row(r)` needed to update `A` for each active relation.
struct SchurCache {
    /// LU of the base `A` (all unilateral relations released).
    base: Arc<SparseLu>,
    /// `xᵣ = A⁻¹·e_row(r)`, keyed by the inequality's constraint-row index;
    /// filled on first use of relation `r` and reused by later solves.
    columns: HashMap<usize, Arc<Vec<f64>>>,
    /// Last converged status, reused as the Schur path's warm start.
    warm: Option<Vec<bool>>,
}

/// The Schur base's lifecycle on a given matrix: not yet attempted, established
/// singular (⇒ the Schur path permanently falls back), or ready with its cache.
enum SchurSlot {
    Untried,
    Singular,
    Ready(SchurCache),
}

/// The active-set state cached transparently on the [`Matrix`] (same `dyn Any`
/// slot as [`lu::Factorization`] / the elimination's condensation; cleared on
/// any matrix mutation). Holds the resolved layout plus, per method, the warm
/// start and cached factors.
struct ActiveSetState {
    /// The matrix DOF layout (rows gather the rhs, columns name the solution).
    row_dofs: Vec<NamedDof>,
    col_dofs: Vec<NamedDof>,
    /// The resolved inequality relations, in model order (deterministic).
    inequalities: Vec<Inequality>,
    /// [`Refactorize`](ActiveSetMethod::Refactorize) warm start: last converged
    /// status (`true` = active, one per inequality) and the LU of its system.
    last: Mutex<Option<(Vec<bool>, Arc<SparseLu>)>>,
    /// [`SchurComplement`](ActiveSetMethod::SchurComplement) base + columns cache.
    schur: Mutex<SchurSlot>,
}

/// Solve `model`'s system with unilateral constraints by the active-set method,
/// using the default options. `matrix` is the assembled saddle-point stiffness
/// of `model` (as produced by [`crate::ops::assemble::stiffness`]); `rhs` is
/// the load field (the right-hand sides `g` live at the multiplier nodes'
/// imposed-value slots).
///
/// A model with no inequality relation falls back to a plain [`lu::solve`].
pub fn solve(model: &Model, matrix: &Matrix, rhs: &NodeField) -> Result<NodeField> {
    solve_inner(model, matrix, rhs, &UnilateralOptions::default(), &NoCancel)
}

/// Like [`solve`] but with explicit [`UnilateralOptions`] (back-end, cache,
/// iteration bound, sign tolerance).
pub fn solve_with_options(
    model: &Model,
    matrix: &Matrix,
    rhs: &NodeField,
    options: &UnilateralOptions,
) -> Result<NodeField> {
    solve_inner(model, matrix, rhs, options, &NoCancel)
}

/// Like [`solve`], but polls `cancel` at each status iteration so the call can
/// be stopped early (returning [`PyrucastError::Interrupted`]). Same granularity
/// as [`lu::solve_cancellable`]: each factorization is a single library call and
/// is not interrupted mid-way.
pub fn solve_cancellable(
    model: &Model,
    matrix: &Matrix,
    rhs: &NodeField,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(model, matrix, rhs, &UnilateralOptions::default(), cancel)
}

/// [`solve_cancellable`] with explicit [`UnilateralOptions`] — the full form the
/// Python binding routes to.
pub fn solve_cancellable_with_options(
    model: &Model,
    matrix: &Matrix,
    rhs: &NodeField,
    options: &UnilateralOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(model, matrix, rhs, options, cancel)
}

fn solve_inner(
    model: &Model,
    matrix: &Matrix,
    rhs: &NodeField,
    options: &UnilateralOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    let SolveMethod::Lu = options.method;
    cancel.check()?;

    // ── Step 1 — obtain the active-set state (cached or fresh) ─────────
    let state: Arc<ActiveSetState> = if options.cache {
        match matrix.cached_factorization::<ActiveSetState>() {
            Some(s) => s,
            None => {
                let s = Arc::new(build_state(model, matrix)?);
                matrix.store_factorization(s.clone());
                s
            }
        }
    } else {
        Arc::new(build_state(model, matrix)?)
    };

    // No inequality ⇒ the saddle-point is linear; solve it as-is.
    if state.inequalities.is_empty() {
        let opts = SolveOptions {
            method: options.method,
            cache: options.cache,
        };
        return lu::solve_cancellable_with_options(matrix, rhs, &opts, cancel);
    }
    cancel.check()?;

    // ── Step 2 — full rhs (g at the constraint rows) + base CSC ────────
    let b_full = rhs.gather(&state.row_dofs)?;
    let csc = matrix.to_csc()?;

    // ── Step 3 — run the chosen active-set method ──────────────────────
    // The Schur path reuses one base factorization; it falls back to the
    // refactorizing path (Ok(None)) when the base is singular.
    if options.active_set == ActiveSetMethod::SchurComplement {
        if let Some(solution) = solve_schur(&state, &csc, &b_full, rhs, options, cancel)? {
            return Ok(solution);
        }
    }
    solve_refactorize(&state, &csc, &b_full, rhs, options, cancel)
}

/// The KKT sign test on one status's solution `x`: release every active relation
/// whose reaction `λ` pulls, activate every inactive one whose gap penetrates.
/// Returns whether the status changed. Shared by both active-set methods.
fn update_status(
    inequalities: &[Inequality],
    status: &mut [bool],
    x: &[f64],
    b_full: &[f64],
    tol: f64,
) -> bool {
    let mut changed = false;
    for (ineq, active) in inequalities.iter().zip(status.iter_mut()) {
        if *active {
            // Active: the reaction must not pull. λ = −μ (see module doc).
            let lambda = x[ineq.lambda_col];
            let release = match ineq.sense {
                RelationSense::GreaterEqual => lambda > tol,
                RelationSense::LessEqual => lambda < -tol,
                RelationSense::Equality => unreachable!("equalities are not collected"),
            };
            if release {
                *active = false;
                changed = true;
            }
        } else {
            // Inactive: the gap C·u − g must stay on the feasible side.
            let cu: f64 = ineq.term_cols.iter().map(|&(c, a)| a * x[c]).sum();
            let gap = cu - b_full[ineq.row];
            let violated = match ineq.sense {
                RelationSense::GreaterEqual => gap < -tol,
                RelationSense::LessEqual => gap > tol,
                RelationSense::Equality => unreachable!("equalities are not collected"),
            };
            if violated {
                *active = true;
                changed = true;
            }
        }
    }
    changed
}

/// The original method: refactorize the full sparse saddle-point at each status
/// change, warm-started from the cached converged status + its LU.
fn solve_refactorize(
    state: &ActiveSetState,
    csc: &nalgebra_sparse::CscMatrix<f64>,
    b_full: &[f64],
    rhs: &NodeField,
    options: &UnilateralOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    // Initial status: warm start (status + its factorization) or all-active.
    let (mut status, mut lu_current): (Vec<bool>, Option<Arc<SparseLu>>) =
        match state.last.lock().as_ref() {
            Some((s, f)) => (s.clone(), Some(f.clone())),
            None => (vec![true; state.inequalities.len()], None),
        };

    let n = state.row_dofs.len();
    for _iter in 0..options.max_iter {
        cancel.check()?;

        // Factorize the current status's system (reused when warm-started).
        let lu = match lu_current.take() {
            Some(f) => f,
            None => Arc::new(factorize_status(csc, &state.inequalities, &status)?),
        };

        // Solve with g zeroed at the inactive constraint rows (λ_r = 0).
        let mut b = b_full.to_vec();
        for (ineq, &active) in state.inequalities.iter().zip(&status) {
            if !active {
                b[ineq.row] = 0.0;
            }
        }
        let x = lu_solve_vec(&lu, &b);
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PyrucastError::Message(
                "solve_unilateral: LU failed (matrix is singular — check that the \
                 structure is supported when every unilateral relation releases)"
                    .into(),
            ));
        }
        cancel.check()?;

        if !update_status(&state.inequalities, &mut status, &x, b_full, options.tol) {
            // Converged: remember the status + factors for the next warm start.
            if options.cache {
                *state.last.lock() = Some((status, lu));
            }
            debug_assert_eq!(x.len(), n);
            return NodeField::from_dof_values(rhs.coords()?, &state.col_dofs, &x);
        }
    }

    Err(PyrucastError::Message(format!(
        "solve_unilateral: the active set did not converge in {} iterations \
         (cycling — try a looser tol, or check the model)",
        options.max_iter
    )))
}

/// The Schur/Delassus method: factorize the inequality-free base `A` once, then
/// obtain each status by a dense Woodbury update, reusing `A`'s factors and the
/// cached columns `A⁻¹·e_row(r)`.
///
/// Returns `Ok(None)` (asking the caller to fall back to
/// [`solve_refactorize`]) when the base is singular or any base solve goes
/// non-finite; `Ok(Some(_))` on a converged solution.
fn solve_schur(
    state: &ActiveSetState,
    csc: &nalgebra_sparse::CscMatrix<f64>,
    b_full: &[f64],
    rhs: &NodeField,
    options: &UnilateralOptions,
    cancel: &dyn Cancel,
) -> Result<Option<NodeField>> {
    let n = state.row_dofs.len();

    // ── Base `A` (all relations released), factorized once and cached ──
    let mut slot = state.schur.lock();
    match &*slot {
        SchurSlot::Singular => return Ok(None),
        SchurSlot::Ready(_) => {}
        SchurSlot::Untried => {
            let all_inactive = vec![false; state.inequalities.len()];
            // A singular base is a legitimate outcome (the structure needs
            // contact to hold): mark it and let the caller refactorize. faer's
            // pivoted LU may factor a singular matrix into *finite* garbage
            // without erroring, so a round-trip solve confirms non-singularity.
            match factorize_status(csc, &state.inequalities, &all_inactive) {
                Ok(base) if base_is_nonsingular(&base, csc, &state.inequalities, n) => {
                    *slot = SchurSlot::Ready(SchurCache {
                        base: Arc::new(base),
                        columns: HashMap::new(),
                        warm: None,
                    });
                }
                _ => {
                    *slot = SchurSlot::Singular;
                    return Ok(None);
                }
            }
        }
    }
    let cache = match &mut *slot {
        SchurSlot::Ready(c) => c,
        _ => unreachable!("base just established Ready"),
    };
    let base = cache.base.clone();

    // Warm start from the last converged status (else all-active).
    let mut status = cache
        .warm
        .clone()
        .unwrap_or_else(|| vec![true; state.inequalities.len()]);

    for _iter in 0..options.max_iter {
        cancel.check()?;

        let Some(x) = schur_status_solve(
            &base,
            &mut cache.columns,
            &state.inequalities,
            &status,
            b_full,
            n,
        ) else {
            // A non-finite base/Woodbury solve ⇒ fall back to refactorization.
            return Ok(None);
        };
        cancel.check()?;

        if !update_status(&state.inequalities, &mut status, &x, b_full, options.tol) {
            if options.cache {
                cache.warm = Some(status);
            }
            let field = NodeField::from_dof_values(rhs.coords()?, &state.col_dofs, &x)?;
            return Ok(Some(field));
        }
    }

    Err(PyrucastError::Message(format!(
        "solve_unilateral: the active set did not converge in {} iterations \
         (cycling — try a looser tol, or check the model)",
        options.max_iter
    )))
}

/// `A·x`, where `A` is the released base: the assembled matrix `csc` with every
/// inequality row replaced by the identity on its multiplier column (`λᵣ = 0`).
fn base_matvec(
    csc: &nalgebra_sparse::CscMatrix<f64>,
    inequalities: &[Inequality],
    x: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0; csc.nrows()];
    let col_offsets = csc.col_offsets();
    let row_indices = csc.row_indices();
    let values = csc.values();
    for col in 0..csc.ncols() {
        let xc = x[col];
        if xc != 0.0 {
            for k in col_offsets[col]..col_offsets[col + 1] {
                y[row_indices[k]] += values[k] * xc;
            }
        }
    }
    // The base row of every inequality is the identity at its λ column.
    for ineq in inequalities {
        y[ineq.row] = x[ineq.lambda_col];
    }
    y
}

/// Whether the released base `A` is non-singular, by a round-trip solve:
/// `A⁻¹·(A·1)` must recover `1`. A rigid mode (the structure floats without
/// contact) makes `A` singular; faer then returns a solution off by a large
/// multiple of the null vector, which this catches (finite garbage included).
fn base_is_nonsingular(
    base: &SparseLu,
    csc: &nalgebra_sparse::CscMatrix<f64>,
    inequalities: &[Inequality],
    n: usize,
) -> bool {
    let ones = vec![1.0; n];
    let b = base_matvec(csc, inequalities, &ones);
    let y = lu_solve_vec(base, &b);
    y.iter().all(|v| v.is_finite()) && y.iter().all(|&v| (v - 1.0).abs() < 1e-6)
}

/// The cached Schur column `xᵣ = A⁻¹·e_row(r)`, computed on first use.
fn schur_column(
    base: &SparseLu,
    columns: &mut HashMap<usize, Arc<Vec<f64>>>,
    row: usize,
    n: usize,
) -> Arc<Vec<f64>> {
    if let Some(x) = columns.get(&row) {
        return x.clone();
    }
    let mut e = vec![0.0; n];
    e[row] = 1.0;
    let x = Arc::new(lu_solve_vec(base, &e));
    columns.insert(row, x.clone());
    x
}

/// Solve one status's saddle-point system by the Woodbury identity around the
/// released base `A`. In `A` a released relation carries the identity row
/// `λᵣ = 0` (a `1` at its multiplier column `λcol`); activating it restores the
/// real constraint row `Cᵣ`. That is a rank-`k` row update
/// `A + Σ e_row(r)·(Cᵣ − e_λcol(r))ᵀ`, so
///
/// ```text
/// x = A⁻¹·b − X·(I + Vᵀ X)⁻¹·(Vᵀ A⁻¹ b)
/// ```
///
/// with `X = [A⁻¹·e_row(r)]` (cached columns) and `Vᵣ = Cᵣ − e_λcol(r)` (so
/// `Vᵣᵀ v = Cᵣ·v − v[λcol]`). The `k×k` system (`I + Vᵀ X` — the active
/// Delassus operator) is dense.
///
/// Returns `None` when a solve is non-finite (⇒ the caller falls back).
fn schur_status_solve(
    base: &SparseLu,
    columns: &mut HashMap<usize, Arc<Vec<f64>>>,
    inequalities: &[Inequality],
    status: &[bool],
    b_full: &[f64],
    n: usize,
) -> Option<Vec<f64>> {
    // b with g zeroed at the inactive constraint rows (λ_r = 0).
    let mut b = b_full.to_vec();
    for (ineq, &active) in inequalities.iter().zip(status) {
        if !active {
            b[ineq.row] = 0.0;
        }
    }
    let y0 = lu_solve_vec(base, &b);
    if y0.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let active: Vec<usize> = (0..inequalities.len()).filter(|&i| status[i]).collect();
    let k = active.len();
    if k == 0 {
        return Some(y0);
    }

    // Cache the columns of the active relations.
    let xs: Vec<Arc<Vec<f64>>> = active
        .iter()
        .map(|&i| schur_column(base, columns, inequalities[i].row, n))
        .collect();
    if xs.iter().any(|x| x.iter().any(|v| !v.is_finite())) {
        return None;
    }

    // Dense G = I + Vᵀ X and small rhs = Vᵀ y0, with Vᵢ = Cᵢ − e_row(i).
    let mut g = DMatrix::<f64>::zeros(k, k);
    let mut rhs_small = DVector::<f64>::zeros(k);
    for (i, &ai) in active.iter().enumerate() {
        let ineq_i = &inequalities[ai];
        // Vᵢᵀ v = Cᵢ·v − v[λcolᵢ] (the identity the base carries at λcolᵢ).
        let ci_y0: f64 = ineq_i.term_cols.iter().map(|&(c, a)| a * y0[c]).sum();
        rhs_small[i] = ci_y0 - y0[ineq_i.lambda_col];
        for (j, xj) in xs.iter().enumerate() {
            let ci_xj: f64 = ineq_i.term_cols.iter().map(|&(c, a)| a * xj[c]).sum();
            let mut v = ci_xj - xj[ineq_i.lambda_col];
            if i == j {
                v += 1.0;
            }
            g[(i, j)] = v;
        }
    }

    let d = g.lu().solve(&rhs_small)?;
    if d.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // x = y0 − Σⱼ dⱼ·xⱼ.
    let mut x = y0;
    for (j, xj) in xs.iter().enumerate() {
        let dj = d[j];
        for (xi, &xij) in x.iter_mut().zip(xj.iter()) {
            *xi -= dj * xij;
        }
    }
    Some(x)
}

/// Resolve the model's inequality relations against the matrix DOF layout.
/// Equality relations are skipped (their rows are always enforced as-is).
fn build_state(model: &Model, matrix: &Matrix) -> Result<ActiveSetState> {
    let row_dofs = matrix.row_dofs()?;
    let col_dofs = matrix.col_dofs()?;
    if row_dofs.len() != col_dofs.len() {
        return Err(PyrucastError::Message(format!(
            "solve_unilateral: matrix must be square; got {}×{}",
            row_dofs.len(),
            col_dofs.len()
        )));
    }
    let row_of: HashMap<&NamedDof, usize> = row_dofs.iter().zip(0..).collect();
    let col_of: HashMap<&NamedDof, usize> = col_dofs.iter().zip(0..).collect();

    let mut inequalities = Vec::new();
    for h in model {
        let sub = read(h)?;
        let kind = sub.as_kind();
        let Some(constraint) = kind.as_constraint() else {
            continue;
        };
        // The multiplier (primal) conjugate to each imposed-value (dual) slot,
        // by the positional pairing every constraint declares (cf. `dual_of`).
        let primals = kind.primal_vars();
        let duals = kind.dual_vars();
        for rel in constraint.relations()? {
            if rel.sense == RelationSense::Equality {
                continue;
            }
            let row = *row_of
                .get(&(rel.multiplier_node, rel.imposed_value.clone()))
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "solve_unilateral: constraint row ({:?}, '{}') is not a row of \
                         the matrix — was it assembled from this model?",
                        rel.multiplier_node, rel.imposed_value
                    ))
                })?;
            let multiplier_var = duals
                .iter()
                .position(|d| d == &rel.imposed_value)
                .map(|i| primals[i].clone())
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "solve_unilateral: '{}' is not a dual variable of {}",
                        rel.imposed_value,
                        kind.label()
                    ))
                })?;
            let lambda_col = *col_of
                .get(&(rel.multiplier_node, multiplier_var.clone()))
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "solve_unilateral: multiplier DOF ({:?}, '{}') is not a column \
                         of the matrix",
                        rel.multiplier_node, multiplier_var
                    ))
                })?;
            let mut term_cols = Vec::with_capacity(rel.terms.len());
            for term in &rel.terms {
                let col = *col_of
                    .get(&(term.node, term.variable.clone()))
                    .ok_or_else(|| {
                        PyrucastError::Message(format!(
                            "solve_unilateral: term ({:?}, '{}') is not a column of the \
                             matrix",
                            term.node, term.variable
                        ))
                    })?;
                term_cols.push((col, term.coefficient));
            }
            inequalities.push(Inequality {
                row,
                lambda_col,
                term_cols,
                sense: rel.sense,
            });
        }
    }

    Ok(ActiveSetState {
        row_dofs,
        col_dofs,
        inequalities,
        last: Mutex::new(None),
        schur: Mutex::new(SchurSlot::Untried),
    })
}

/// Test-only counter of sparse saddle-point factorizations, used to assert that
/// the Schur path factorizes the base **once** where the refactorizing path pays
/// one per status iteration.
#[cfg(test)]
pub(crate) static FACTORIZE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Factorize the system of one status: the base saddle-point with, for every
/// **inactive** inequality, its constraint row replaced by the identity on the
/// multiplier DOF (`λ_r = 0`). The `Cᵀ` column needs no clearing — `λ_r = 0`
/// zeroes its contribution exactly.
fn factorize_status(
    csc: &nalgebra_sparse::CscMatrix<f64>,
    inequalities: &[Inequality],
    status: &[bool],
) -> Result<SparseLu> {
    #[cfg(test)]
    FACTORIZE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n = csc.nrows();
    // Rows to blank: the constraint rows of the inactive inequalities.
    let mut blanked = vec![false; n];
    for (ineq, &active) in inequalities.iter().zip(status) {
        if !active {
            blanked[ineq.row] = true;
        }
    }
    let col_offsets = csc.col_offsets();
    let row_indices = csc.row_indices();
    let values = csc.values();
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(values.len());
    for col in 0..csc.ncols() {
        for k in col_offsets[col]..col_offsets[col + 1] {
            let row = row_indices[k];
            if !blanked[row] {
                triplets.push(Triplet::new(row, col, values[k]));
            }
        }
    }
    for (ineq, &active) in inequalities.iter().zip(status) {
        if !active {
            triplets.push(Triplet::new(ineq.row, ineq.lambda_col, 1.0));
        }
    }
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).map_err(|e| {
        PyrucastError::Message(format!("solve_unilateral: sparse build failed: {e:?}"))
    })?;
    a.sp_lu().map_err(|e| {
        PyrucastError::Message(format!("solve_unilateral: LU failed (singular?): {e:?}"))
    })
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::element_field::{ElementField, SubElementField};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::matrix::Matrix;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::model::{Model, SubModel};
    use crate::containers::node_field::SubNodeField;
    use crate::ops::assemble::stiffness;
    use crate::ops::mesher::barycenter;
    use crate::store::insert;
    use std::sync::atomic::Ordering;

    /// A 2-element heat bar on `[0, 1]` (`k = 1`) with `T(0) = 0` (equality) and
    /// **two loose lower bounds** `T(x₁) ≥ −10`, `T(x₂) ≥ −20`, driven by a
    /// positive flux `q` at the right end. The unconstrained solution `T = q·x`
    /// is positive, so **both bounds release** — a two-status-iteration problem
    /// starting from the all-active warm start.
    fn two_loose_bounds(q: f64) -> (Model, ElementField, Matrix, NodeField, Vec<Node>) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=2)
            .map(|i| Node::create_in(coords.clone(), &[i as f64 * 0.5]).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[nodes[0].id(), nodes[1].id()]).unwrap();
        mesh.add_cell(&[nodes[1].id(), nodes[2].id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();

        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();

        let poi1 = |n: &Node| {
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(n)).unwrap())
        };

        // Equality T(0) = 0.
        let imp0 = poi1(&nodes[0]);
        let m0 = barycenter(&imp0).unwrap();
        let dir = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &imp0,
            &m0,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        let dir_mult = dir.multiplier_nodes().unwrap()[0];
        model.add_sub(insert(dir)).unwrap();

        // Two loose lower bounds at nodes 1 and 2.
        let mut bound_mults = Vec::new();
        for node in [&nodes[1], &nodes[2]] {
            let imp = poi1(node);
            let mm = barycenter(&imp).unwrap();
            let b = SubModel::dirichlet(
                "T".into(),
                "q".into(),
                &imp,
                &mm,
                None,
                None,
                RelationSense::GreaterEqual,
            )
            .unwrap();
            bound_mults.push(b.multiplier_nodes().unwrap()[0]);
            model.add_sub(insert(b)).unwrap();
        }

        // rhs: flux q at the right node, imposed values at every multiplier slot.
        let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
        rhs_sm.add_cell(&[nodes[2].id()]).unwrap();
        rhs_sm.add_cell(&[dir_mult]).unwrap();
        rhs_sm.add_cell(&[bound_mults[0]]).unwrap();
        rhs_sm.add_cell(&[bound_mults[1]]).unwrap();
        let rhs_sm = insert(rhs_sm);
        let mut rhs =
            SubNodeField::from_poi1(&rhs_sm, vec!["q".into(), "imposed_T".into()]).unwrap();
        rhs.set_value(nodes[2].id(), "q", q).unwrap();
        rhs.set_value(dir_mult, "imposed_T", 0.0).unwrap();
        rhs.set_value(bound_mults[0], "imposed_T", -10.0).unwrap();
        rhs.set_value(bound_mults[1], "imposed_T", -20.0).unwrap();
        let rhs = NodeField::from_sub(rhs);

        let k = stiffness(&model, &materials).unwrap();
        (model, materials, k, rhs, nodes)
    }

    fn opts(active_set: ActiveSetMethod) -> UnilateralOptions {
        UnilateralOptions {
            active_set,
            ..Default::default()
        }
    }

    /// Serialize the tests that read the process-global [`FACTORIZE_CALLS`]
    /// counter (the default test harness runs them on parallel threads).
    static COUNT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The Schur path factorizes the base **once**, where the refactorizing path
    /// pays one sparse factorization per status iteration (here two: all-active,
    /// then all-inactive). Both reach the same solution.
    #[test]
    fn schur_factorizes_the_base_once() {
        let _guard = COUNT_LOCK.lock().unwrap();
        // Refactorize: fresh matrix, count the factorizations (all-active +
        // all-inactive = 2).
        let (model, _mat, k_ref, rhs, nodes) = two_loose_bounds(5.0);
        FACTORIZE_CALLS.store(0, Ordering::Relaxed);
        let sol_ref =
            solve_with_options(&model, &k_ref, &rhs, &opts(ActiveSetMethod::Refactorize)).unwrap();
        let n_ref = FACTORIZE_CALLS.load(Ordering::Relaxed);
        assert!(
            n_ref >= 2,
            "refactorize pays one factorization per iteration, got {n_ref}"
        );

        // Schur: fresh matrix, only the base is factorized (1 call).
        let (model_s, _m2, k_schur, rhs_s, _n2) = two_loose_bounds(5.0);
        FACTORIZE_CALLS.store(0, Ordering::Relaxed);
        let sol_schur =
            solve_with_options(&model_s, &k_schur, &rhs_s, &opts(ActiveSetMethod::SchurComplement))
                .unwrap();
        let n_schur = FACTORIZE_CALLS.load(Ordering::Relaxed);
        assert_eq!(n_schur, 1, "schur factorizes only the inequality-free base");

        // Both agree with the analytical T = 5·x (both bounds released).
        for (i, node) in nodes.iter().enumerate() {
            let expected = 5.0 * i as f64 * 0.5;
            let a = sol_ref.value(node.id(), "T").unwrap();
            let b = sol_schur.value(node.id(), "T").unwrap();
            assert!((a - expected).abs() < 1e-10);
            assert!((a - b).abs() < 1e-12, "schur and refactorize must agree");
        }
    }

    /// A re-solve on the same matrix reuses the cached base — **zero**
    /// factorization the second time — and warm-starts to the same solution.
    #[test]
    fn schur_resolve_reuses_the_cached_base() {
        let _guard = COUNT_LOCK.lock().unwrap();
        let (model, _mat, k, rhs, _nodes) = two_loose_bounds(5.0);
        FACTORIZE_CALLS.store(0, Ordering::Relaxed);
        let first =
            solve_with_options(&model, &k, &rhs, &opts(ActiveSetMethod::SchurComplement)).unwrap();
        assert_eq!(FACTORIZE_CALLS.load(Ordering::Relaxed), 1);

        // Second solve: base cached on the matrix, nothing refactorized.
        let again =
            solve_with_options(&model, &k, &rhs, &opts(ActiveSetMethod::SchurComplement)).unwrap();
        assert_eq!(
            FACTORIZE_CALLS.load(Ordering::Relaxed),
            1,
            "the cached base must be reused (no new factorization)"
        );
        for node in &_nodes {
            assert_eq!(
                first.value(node.id(), "T").unwrap(),
                again.value(node.id(), "T").unwrap()
            );
        }
    }
}
