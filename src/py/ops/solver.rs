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
/// `matrix` is the assembled system `A`; `rhs` is the right-hand side `b` as a
/// `NodeField`, read at every row DOF of the matrix (a DOF no zone defines
/// reads zero). The imposed value of a constraint relation lives at its
/// multiplier node, component `imposed`. The matrix carries everything the
/// solve needs: its relation rows, their coefficients and their sense (`=`,
/// `>=`, `<=`), declared by the constraints that built them — no model is
/// passed alongside.
///
/// `relations` says how the constraint relations are enforced:
///
/// - `"lagrange"` (default) — by their multipliers, the saddle-point solved as
///   assembled; the multipliers come back in the solution as the reactions.
///   As soon as the matrix carries an inequality, the **active-set** loop
///   decides which relations are saturated: an active relation holds as an
///   equation (its `lambda` is `<= 0` for `>=`, `>= 0` for `<=`), an inactive
///   one reports `lambda = 0`;
/// - `"eliminate"` — each equality relation eliminates one unknown, and the
///   smaller system left, without multipliers, is solved; each eliminated
///   unknown's reaction is returned in its dual row. An inequality is not
///   eliminated: it stays on the reduced system, resolved by the same
///   active-set loop, its `lambda` returned at its multiplier node. Scope:
///   non-chained, disjoint slaves.
///
/// `method` selects the factorization: `"lu"` (default) works on any square
/// matrix, `"cholesky"` asks that the system be symmetric **and** positive
/// definite — never a Lagrange saddle-point, usually a system reduced by
/// elimination. The active-set loop factorizes by LU whatever is asked.
///
/// `active_set` selects how each status of the active-set loop is factorized:
/// `"schur"` (default) factorizes the inequality-free base once and updates it
/// per status (falling back when that base is singular), `"refactorize"`
/// refactorizes the full system at each status change. `max_iter` (default
/// `100`) bounds the status loop; `tol` (default `1e-10`) is its sign
/// tolerance on the multiplier and on the gap.
///
/// `cache` (default `True`) reuses what is stored transparently on the matrix —
/// the factorization, the condensation, the active-set state and its last
/// status — cleared automatically when the matrix changes. `verbosity` says
/// how much the solve reports on stdout: `"silent"` (default), `"brief"` or
/// `"detailed"`.
///
/// A `Ctrl+C` is honoured at the solver's phase boundaries and at each status
/// iteration; a factorization itself is a single library call and is not
/// interrupted mid-way.
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
