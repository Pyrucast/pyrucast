//! Python wrapper for the solver operation in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `NodeField` result).

use crate::ops::solver::{ActiveSetMethod, RelationMethod, SolveMethod, SolveOptions, Verbosity};
use crate::py::matrix::PyMatrix;
use crate::py::node_field::PyNodeField;
use crate::py::signals::PySignals;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// Solve the linear system `A·x = b` for `x` (sparse direct, faer).
///
/// The matrix carries everything the solve needs: its relation rows, their
/// coefficients and their sense (`=`, `>=`, `<=`), declared by the constraints
/// that built them — no model is passed alongside. As soon as it carries an
/// inequality, an active-set loop decides which relations are saturated: an
/// active relation holds as an equation (its `lambda` is `<= 0` for `>=`, `>= 0`
/// for `<=`), an inactive one reports `lambda = 0`.
///
/// # Parameters
///
/// - `matrix`: the assembled system `A`.
/// - `rhs`: the right-hand side `b`, read at every row of `matrix`; a row no zone
///   of `rhs` defines reads zero. The imposed value of a constraint relation
///   lives at its multiplier node, component `imposed`.
/// - `method`: the factorization. `"lu"` works on any square matrix;
///   `"cholesky"` asks that the system be symmetric **and** positive definite —
///   never a system with multipliers, usually one reduced by elimination.
///   `"auto"` attempts the Cholesky when the system to factorize declares itself
///   symmetric and holds no multiplier, and takes the LU when it does not or
///   when the Cholesky refuses it (a structure past its buckling load). The
///   active-set loop factorizes by LU whatever is asked.
/// - `relations`: how the constraint relations are enforced. `"lagrange"`: by
///   their multipliers, the system solved as assembled. `"eliminate"`: each
///   equality relation eliminates one unknown and the smaller system left,
///   without multipliers, is solved; an inequality is not eliminated, it stays
///   on the reduced system and is resolved by the same active-set loop. Scope of
///   `"eliminate"`: each relation eliminates an unknown of its own, which
///   appears in no other equality relation. `"auto"`: `"eliminate"` when every
///   equality relation is within that scope, `"lagrange"` otherwise. Every
///   choice gives the same solution.
/// - `active_set`: how each status of the active-set loop is factorized, read
///   only when `matrix` carries an inequality. `"schur"` factorizes the
///   inequality-free base once and updates it per status, falling back on
///   `"refactorize"` when that base is singular; `"refactorize"` factorizes the
///   full system at each status change.
/// - `cache`: reuse what a previous solve stored on `matrix` — the
///   factorization, the condensation, the active-set state and its last status —
///   and store what this one computes. The store is cleared when the matrix
///   changes. A matrix keeps one of them at a time, and a factorization serves
///   the `method` it was made with only: another `relations` or `method`
///   computes afresh and replaces it. `"auto"` takes whatever is stored, so
///   what it chose on the first solve of a matrix is kept for the next ones.
///   `False` neither reads nor fills it.
/// - `max_iter`: the bound of the active-set loop, in status changes.
/// - `tol`: the sign tolerance of the active-set loop, on the multiplier and on
///   the gap.
/// - `verbosity`: how much the solve reports on stdout: `"silent"`, `"brief"` or
///   `"detailed"`.
///
/// # Returns
///
/// The solution, read at the columns of `matrix`: one zone per column support of
/// its blocks, each on the support of the block itself. Two solves of the same
/// model therefore share their supports, and their fields combine zone by zone.
/// The multiplier of every relation comes back at its node, component `lambda`,
/// whether the relations were kept or eliminated.
///
/// # Raises
///
/// `RuntimeError` if `matrix` has no assembled values, if the LU meets an
/// exactly zero pivot (an unknown nothing holds at all), if `"cholesky"` meets a
/// matrix that is not positive definite, if a relation is out of the scope of
/// `"eliminate"`, or if the active-set loop has not converged after `max_iter`
/// status changes. `ValueError` for an unknown `method`, `relations`,
/// `active_set` or `verbosity`. `KeyboardInterrupt` on `Ctrl+C`.
///
/// # Notes
///
/// `Ctrl+C` is honoured between the phases of the solve and at each status
/// change; a factorization is a single library call and is not interrupted
/// mid-way.
///
/// A matrix that is singular only up to rounding is not always reported: a
/// structure left free to move as a rigid body, its boundary conditions
/// forgotten, can come back as a finite but absurdly large solution. Check the
/// boundary conditions when the magnitudes are implausible.
#[py_op(method_on = PyMatrix)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (
    matrix,
    rhs,
    method = SolveMethod::Lu,
    relations = RelationMethod::Lagrange,
    active_set = ActiveSetMethod::SchurComplement,
    cache = true,
    max_iter = 100,
    tol = 1e-10,
    verbosity = Verbosity::Silent,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve(
    py: Python<'_>,
    matrix: PyRef<PyMatrix>,
    rhs: PyRef<PyNodeField>,
    method: SolveMethod,
    relations: RelationMethod,
    active_set: ActiveSetMethod,
    cache: bool,
    max_iter: usize,
    tol: f64,
    verbosity: Verbosity,
) -> PyResult<PyNodeField> {
    let options = SolveOptions {
        method,
        relations,
        active_set,
        cache,
        max_iter,
        tol,
        verbosity,
    };
    let solution = crate::ops::solver::solve_cancellable_with_options(
        &matrix.inner,
        &rhs.inner,
        &options,
        &PySignals(py),
    )?;
    Ok(PyNodeField { inner: solution })
}
