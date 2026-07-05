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
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use super::lu::{self, lu_solve_vec, SolveMethod, SolveOptions, SparseLu};

type NamedDof = (NodeId, String);

/// Options for [`solve_with_options`]. `method` / `cache` mirror
/// [`SolveOptions`]; `max_iter` bounds the status loop; `tol` is the
/// complementarity tolerance on both the multiplier sign and the gap sign
/// (absolute, in the units of the reaction / of the relation).
#[derive(Clone, Copy, Debug)]
pub struct UnilateralOptions {
    /// Direct back-end for each status iteration's system.
    pub method: SolveMethod,
    /// Cache the active-set state (layout + last converged status and its
    /// factorization) on the matrix: a re-solve warm-starts from the previous
    /// status and, when the status is confirmed, skips the refactorization.
    /// Cleared automatically when the matrix changes.
    pub cache: bool,
    /// Maximum number of status iterations (each one factorization).
    pub max_iter: usize,
    /// Sign tolerance for releasing (`λ` past `tol`) and activating (gap past
    /// `−tol`) a relation.
    pub tol: f64,
}

impl Default for UnilateralOptions {
    fn default() -> Self {
        Self {
            method: SolveMethod::Lu,
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

/// The active-set state cached transparently on the [`Matrix`] (same `dyn Any`
/// slot as [`lu::Factorization`] / the elimination's condensation; cleared on
/// any matrix mutation). Holds the resolved layout plus the last converged
/// status and its factorization for warm starts.
struct ActiveSetState {
    /// The matrix DOF layout (rows gather the rhs, columns name the solution).
    row_dofs: Vec<NamedDof>,
    col_dofs: Vec<NamedDof>,
    /// The resolved inequality relations, in model order (deterministic).
    inequalities: Vec<Inequality>,
    /// Last converged status (`true` = active, one per inequality) and the LU
    /// of its system, reused as warm start by the next solve.
    last: Mutex<Option<(Vec<bool>, Arc<SparseLu>)>>,
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

    // ── Step 3 — initial status: warm start or all-active ──────────────
    let (mut status, mut lu_current): (Vec<bool>, Option<Arc<SparseLu>>) =
        match state.last.lock().as_ref() {
            Some((s, f)) => (s.clone(), Some(f.clone())),
            None => (vec![true; state.inequalities.len()], None),
        };

    // ── Step 4 — status loop ────────────────────────────────────────────
    let n = state.row_dofs.len();
    for _iter in 0..options.max_iter {
        cancel.check()?;

        // Factorize the current status's system (reused when warm-started).
        let lu = match lu_current.take() {
            Some(f) => f,
            None => Arc::new(factorize_status(&csc, &state.inequalities, &status)?),
        };

        // Solve with g zeroed at the inactive constraint rows (λ_r = 0).
        let mut b = b_full.clone();
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

        // Check the KKT signs and flip every violating status at once.
        let mut changed = false;
        for (ineq, active) in state.inequalities.iter().zip(status.iter_mut()) {
            if *active {
                // Active: the reaction must not pull. λ = −μ (see module doc).
                let lambda = x[ineq.lambda_col];
                let release = match ineq.sense {
                    RelationSense::GreaterEqual => lambda > options.tol,
                    RelationSense::LessEqual => lambda < -options.tol,
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
                    RelationSense::GreaterEqual => gap < -options.tol,
                    RelationSense::LessEqual => gap > options.tol,
                    RelationSense::Equality => unreachable!("equalities are not collected"),
                };
                if violated {
                    *active = true;
                    changed = true;
                }
            }
        }

        if !changed {
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
    })
}

/// Factorize the system of one status: the base saddle-point with, for every
/// **inactive** inequality, its constraint row replaced by the identity on the
/// multiplier DOF (`λ_r = 0`). The `Cᵀ` column needs no clearing — `λ_r = 0`
/// zeroes its contribution exactly.
fn factorize_status(
    csc: &nalgebra_sparse::CscMatrix<f64>,
    inequalities: &[Inequality],
    status: &[bool],
) -> Result<SparseLu> {
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
