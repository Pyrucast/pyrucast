//! Python wrapper for the solver operation in [`crate::ops::solver`].
//!
//! Free functions solving a linear system into a [`PyNodeField`] and finding
//! eigenmodes, each mode a `dict`. Kept here — mirroring `src/ops/solver/` —
//! per the `py/ops/` convention (their identity is the *operation*, not their
//! result).

use crate::ops::solver::{
    ActiveSetMethod, Mode, Normalization, RelationMethod, SolveMethod, SolveOptions, Verbosity,
};
use crate::py::matrix::PyMatrix;
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use crate::py::node_field::PyNodeField;
use crate::py::signals::PySignals;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
///   never a system with multipliers, usually one reduced by elimination. The
///   active-set loop factorizes by LU whatever is asked.
/// - `relations`: how the constraint relations are enforced. `"lagrange"`: by
///   their multipliers, the system solved as assembled. `"eliminate"`: each
///   equality relation eliminates one unknown and the smaller system left,
///   without multipliers, is solved; an inequality is not eliminated, it stays
///   on the reduced system and is resolved by the same active-set loop. Scope of
///   `"eliminate"`: each relation eliminates an unknown of its own, which
///   appears in no other equality relation.
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
///   computes afresh and replaces it. `False` neither reads nor fills it.
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

/// The `count` eigenvalues of `a x = λ b x` nearest `shift`, and their modes.
///
/// Vibrations take the stiffness and the mass (`λ = ω²`), buckling the
/// stiffness and the geometric stiffness. The relations `a` carries (the
/// multipliers of `dirichlet`, `mpc`…) hold in every mode.
///
/// The modes are those of the shift-inverted operator `(a − shift·b)⁻¹ b`:
/// `a − shift·b` is factorized once, as `solve` does, and a Krylov-Schur
/// iteration runs over the unknowns `b` carries. Each mode is then completed
/// (unknowns without mass, multipliers) and checked by its residual; modes of a
/// common eigenvalue come out `b`-orthogonal.
///
/// # Parameters
///
/// - `a`, `b`: assembled symmetric matrices; `a` or `b` is positive definite on
///   the unknowns the relations leave free (the stiffness, or the mass).
/// - `count`: how many modes; at most the number of unknowns `b` carries.
/// - `shift`: the value near which to look, in the units of `λ` (`ω²` for
///   vibrations). A free structure has `λ = 0`: give it a negative shift.
/// - `normalization`: `"mass"` scales each shape to `φᵀbφ = ±1`, so that, projected
///   on the modes, `b` becomes the identity and `a` the diagonal of the
///   eigenvalues; `"max"` scales its value of largest magnitude to `1`. Either way,
///   that value is positive.
///
/// # Returns
///
/// The modes, in increasing order of `λ`. A mode is a `dict`:
///
/// - `"node"`: its own `Node`, at the origin of the matrices' `Coords`: it carries
///   the mode's coordinate `alpha` and its force `f_alpha` in what is projected
///   on the modes.
/// - `"support"`: the POI1 `Mesh` shared by the modes of the list, whose nodes are
///   exactly theirs. A slice or a union of lists needs its own:
///   `s = pyrucast.mesh.poi1_from_nodes([m["node"] for m in part])`, then
///   `[{**m, "support": s} for m in part]`.
/// - `"eigenvalue"`: `λ`.
/// - `"shape"`: the shape, a `NodeField` on the column supports of `a` (those
///   `solve` returns), without the multipliers.
/// - `"pairs"`: the `(primal, dual)` names of the shape, as `xty` takes them.
/// - `"generalized_mass"`: `φᵀbφ` of the scaled shape.
///
/// # Raises
///
/// `RuntimeError` if `a` or `b` has no assembled values or is not symmetric, if
/// `a` carries a unilateral relation, if `a − shift·b` is singular, or if a mode
/// does not converge. `ValueError` for an unknown `normalization`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (a, b, count, shift = 0.0, normalization = Normalization::Mass))]
pub fn eigen<'py>(
    py: Python<'py>,
    a: PyRef<PyMatrix>,
    b: PyRef<PyMatrix>,
    count: usize,
    shift: f64,
    normalization: Normalization,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let (a, b) = (&a.inner, &b.inner);
    let modes = py.detach(|| crate::ops::solver::eigen(a, b, count, shift, normalization))?;
    modes
        .into_iter()
        .map(|mode| mode_to_dict(py, mode))
        .collect()
}

/// A mode as Python sees it: a `dict` holding the mode's handles.
pub(crate) fn mode_to_dict(py: Python<'_>, mode: Mode) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("node", Py::new(py, PyNode::from_node(mode.node))?)?;
    dict.set_item(
        "support",
        Py::new(
            py,
            PyMesh {
                inner: mode.support,
            },
        )?,
    )?;
    dict.set_item("eigenvalue", mode.eigenvalue)?;
    if let Some(value) = mode.eigenvalue_imag {
        dict.set_item("eigenvalue_imag", value)?;
    }
    dict.set_item("shape", Py::new(py, PyNodeField { inner: mode.shape })?)?;
    if let Some(shape) = mode.shape_imag {
        dict.set_item("shape_imag", Py::new(py, PyNodeField { inner: shape })?)?;
    }
    dict.set_item("pairs", mode.pairs)?;
    dict.set_item("generalized_mass", mode.generalized_mass)?;
    Ok(dict)
}
