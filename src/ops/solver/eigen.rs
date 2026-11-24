//! Eigenmodes, as lists of [`Mode`]s: [`eigen`] for `A x = λ B x` on sparse
//! matrices, [`quadratic_eigen`] for the complex modes of a small dense
//! `[K + iωC − ω²M] X = 0`.
//!
//! A mode carries everything its consumers need: its own node, the POI1
//! support shared by the modes of its list, its eigenvalue, its shape, the
//! `(primal, dual)` pairs of that shape and its generalized mass. The node of
//! a mode carries its modal coordinate [`MODAL_COORDINATE`] and its modal
//! force [`MODAL_FORCE`]: what is projected on a list of modes lives on their
//! support, and is read back node by node.
//!
//! ```
//! # use pyrucast::aggregate::Aggregate;
//! # use pyrucast::atoms::{ElementType, Node};
//! # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
//! # use pyrucast::containers::mesh::SubMesh;
//! # use pyrucast::coords::Coords;
//! # use pyrucast::handle::Handle;
//! # use pyrucast::ops::solver::{self, Normalization};
//! # let coords = Handle::new(Coords::new(1).unwrap());
//! # let p = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! # let support = {
//! #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
//! #     sm.add_cell(&[p.id()]).unwrap();
//! #     Handle::new(sm)
//! # };
//! # let point = |value: f64| -> pyrucast::Result<Matrix> {
//! #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
//! #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
//! #     z.add_entry(p.id(), "f", p.id(), "u", value)?;
//! #     let mut m = Matrix::empty();
//! #     m.add_sub(Handle::new(z))?;
//! #     m.assemble()?;
//! #     Ok(m)
//! # };
//! // A mass of 40 on a spring of 4e5: one mode, λ = k/m = 1e4.
//! let (k, m) = (point(4e5)?, point(40.0)?);
//! let modes = solver::eigen(&k, &m, 1, 0.0, Normalization::Mass)?;
//! assert!((modes[0].eigenvalue - 1e4).abs() < 1e-8);
//! // Normalized by the mass: φ = 1/√40, of generalized mass 1.
//! assert!((modes[0].shape.value(p.id(), "u")? - 40f64.sqrt().recip()).abs() < 1e-12);
//! assert!((modes[0].generalized_mass - 1.0).abs() < 1e-12);
//! # Ok::<(), pyrucast::PyrucastError>(())
//! ```

use std::collections::HashMap;

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{gevd_real, gevd_scratch};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::matrix_free::eigen::{partial_eigen, partial_eigen_scratch, PartialEigenParams};
use faer::matrix_free::LinOp;
use faer::reborrow::ReborrowMut;
use faer::{c64, ColRef, Mat, MatMut, MatRef, Par, Side};

use super::lu::Factorization;
use super::SolveOptions;
use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, Node};
use crate::containers::field::Field;
use crate::containers::matrix::{dof_key, dof_node, dof_var, DofKey, Matrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::{IMPOSED, MULTIPLIER};

/// The primal unknown a mode's node carries: the mode's coordinate in a
/// projected problem.
///
/// ```
/// assert_eq!(pyrucast::ops::solver::MODAL_COORDINATE, "alpha");
/// ```
pub const MODAL_COORDINATE: &str = "alpha";

/// The dual of [`MODAL_COORDINATE`]: the modal force, the rows of a projected
/// matrix.
///
/// ```
/// assert_eq!(pyrucast::ops::solver::MODAL_FORCE, "f_alpha");
/// ```
pub const MODAL_FORCE: &str = "f_alpha";

/// How [`eigen`] scales each mode. The Python default is `"mass"`.
///
/// ```
/// # use pyrucast::named::Named;
/// # use pyrucast::ops::solver::Normalization;
/// assert_eq!(Normalization::parse("max")?, Normalization::Max);
/// assert_eq!(Normalization::Mass.name(), "mass");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// `φᵀBφ = ±1`: projected on the modes, `b` becomes the identity and `a`
    /// the diagonal of the eigenvalues.
    Mass,
    /// The value of largest magnitude of each shape is `1`.
    Max,
}

impl crate::named::Named for Normalization {
    const LABEL: &'static str = "normalization";
    const VALUES: &'static [Self] = &[Self::Mass, Self::Max];

    fn name(self) -> &'static str {
        match self {
            Self::Mass => "mass",
            Self::Max => "max",
        }
    }
}

/// One mode of a list, self-contained: its members are handles, so a mode
/// costs no copy of its shape.
///
/// - `node` is the mode's identity: everything projected on the list is read
///   at this node, components [`MODAL_COORDINATE`] and [`MODAL_FORCE`].
/// - `support` is the POI1 mesh shared by every mode of the list, whose nodes
///   are exactly theirs. A slice or a union of lists needs its own:
///   `mesh::poi1_from_nodes` on the modes' nodes.
/// - `shape` lies on the column supports of the matrix the modes come from,
///   the supports a solve of that matrix returns, without the multipliers.
/// - `pairs` are the `(primal, dual)` names of the shape, as `xty` takes them.
/// - `eigenvalue_imag` and `shape_imag` are the imaginary parts of a complex
///   mode, present together; both are `None` for a real mode.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::solver::{self, Normalization};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let p = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[p.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let point = |value: f64| -> pyrucast::Result<Matrix> {
/// #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
/// #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
/// #     z.add_entry(p.id(), "f", p.id(), "u", value)?;
/// #     let mut m = Matrix::empty();
/// #     m.add_sub(Handle::new(z))?;
/// #     m.assemble()?;
/// #     Ok(m)
/// # };
/// let modes = solver::eigen(&point(4e5)?, &point(40.0)?, 1, 0.0, Normalization::Max)?;
/// let mode = &modes[0];
/// // Its node is the one node of its support, and its shape pairs u with f.
/// assert_eq!(mode.support.unit()?.read().connectivity(), &[mode.node.id()]);
/// assert_eq!(mode.pairs, vec![("u".to_string(), "f".to_string())]);
/// assert!(mode.eigenvalue_imag.is_none() && mode.shape_imag.is_none());
/// // Scaled to a largest value of 1, its generalized mass is the mass.
/// assert!((mode.generalized_mass - 40.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Debug)]
pub struct Mode {
    /// The mode's own node, at the origin of the matrices' `Coords`.
    pub node: Node,
    /// The POI1 mesh shared by the modes of the list.
    pub support: Mesh,
    /// λ, or the real part of a complex eigenvalue.
    pub eigenvalue: f64,
    /// The imaginary part of a complex eigenvalue; `None` for a real mode.
    pub eigenvalue_imag: Option<f64>,
    /// The shape, or the real part of a complex one.
    pub shape: NodeField,
    /// The imaginary part of a complex shape; `None` for a real mode.
    pub shape_imag: Option<NodeField>,
    /// The `(primal, dual)` pairs of the shape.
    pub pairs: Vec<(String, String)>,
    /// `φᵀBφ` of the scaled shape.
    pub generalized_mass: f64,
}

/// The relative residual a mode must reach, `‖A x − λ B x‖ / (‖A x‖ + |λ|
/// ‖B x‖)`, once refined.
const RESIDUAL_TOLERANCE: f64 = 1e-8;

/// The tolerance handed to the Krylov-Schur iteration.
const KRYLOV_TOLERANCE: f64 = 1e-12;

/// The `count` eigenvalues of `a x = λ b x` nearest `shift`, and their modes,
/// in increasing order of λ.
///
/// Vibrations take the stiffness and the mass (λ = ω²), buckling the
/// stiffness and the geometric stiffness. The relations `a` carries (the
/// multipliers of `dirichlet`, `mpc`…) hold in every mode.
///
/// The modes are those of the shift-inverted operator `(a − shift·b)⁻¹ b`,
/// found by a Krylov-Schur iteration over the unknowns `b` carries (its
/// columns): `a − shift·b` is factorized once, as `solve` does. One more solve
/// per mode then gives the whole vector, unknowns without mass and multipliers
/// included, and a Rayleigh-Ritz step on the modes found makes those of a
/// common eigenvalue `b`-orthogonal. Every mode is checked by its residual.
///
/// # Parameters
///
/// - `a`, `b`: assembled symmetric matrices; `a` or `b` is positive definite
///   on the unknowns the relations leave free (the stiffness, or the mass).
/// - `count`: how many modes; at most the number of columns of `b`.
/// - `shift`: the value near which to look, in the units of λ (ω² for
///   vibrations; `0.0` in Python). A free structure has λ = 0: give it a
///   negative shift.
/// - `normalization`: how each shape is scaled ([`Normalization`]; `"mass"`
///   in Python). The value of largest magnitude of a shape is positive.
///
/// # Errors
///
/// - `a` or `b` not assembled, or not symmetric;
/// - `a` carries a unilateral relation;
/// - `a − shift·b` is singular;
/// - a mode does not converge.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::solver::{self, Normalization};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..2).map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap()).collect();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     for x in &n { sm.add_cell(&[x.id()]).unwrap(); }
/// #     Handle::new(sm)
/// # };
/// # let matrix = |entries: &[(usize, usize, f64)]| -> pyrucast::Result<Matrix> {
/// #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
/// #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
/// #     for &(i, j, v) in entries { z.add_entry(n[i].id(), "f", n[j].id(), "u", v)?; }
/// #     let mut m = Matrix::empty();
/// #     m.add_sub(Handle::new(z))?;
/// #     m.assemble()?;
/// #     Ok(m)
/// # };
/// // Two unit masses, each on a spring of 2 to the ground, a spring of 1
/// // between them: λ = 2 (in phase) and λ = 4 (in opposition).
/// let k = matrix(&[(0, 0, 3.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 3.0)])?;
/// let m = matrix(&[(0, 0, 1.0), (1, 1, 1.0)])?;
/// let modes = solver::eigen(&k, &m, 2, 0.0, Normalization::Mass)?;
/// assert!((modes[0].eigenvalue - 2.0).abs() < 1e-9 && (modes[1].eigenvalue - 4.0).abs() < 1e-9);
/// // Near 3.5, the nearer one comes alone.
/// let near = solver::eigen(&k, &m, 1, 3.5, Normalization::Mass)?;
/// assert!((near[0].eigenvalue - 4.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn eigen(
    a: &Matrix,
    b: &Matrix,
    count: usize,
    shift: f64,
    normalization: Normalization,
) -> Result<Vec<Mode>> {
    for (m, name) in [(a, "a"), (b, "b")] {
        if !m.is_assembled() {
            return Err(PyrucastError::Message(format!(
                "eigen: `{name}` has no assembled values; call assemble() first"
            )));
        }
        if !m.symmetric() {
            return Err(PyrucastError::Message(format!(
                "eigen: `{name}` is not symmetric; the real modes of a x = λ b x need a \
                 symmetric pair"
            )));
        }
    }
    if a.has_inequality() {
        return Err(PyrucastError::Message(
            "eigen: `a` carries a unilateral relation (≥ or ≤), which has no eigenproblem".into(),
        ));
    }
    let c = (a - &(b * shift))?;
    let (b_rows, b_cols) = reindex(b, &c)?;
    let count = count.min(b_cols.len());
    if count == 0 {
        return Ok(Vec::new());
    }
    let (a_rows, a_cols) = reindex(a, &c)?;
    let n_c = c.n_rows()?;
    let fact = Factorization::new(&c, &SolveOptions::default())?;
    let op = ShiftInvert {
        fact: &fact,
        b,
        b_rows: &b_rows,
        b_cols: &b_cols,
        n_c,
    };

    // A singular `a − shift·b` factorizes with a zero pivot, and its first
    // product is not finite: refused before the iteration runs on it.
    let v0 = starting_vector(b_cols.len());
    let mut probe = vec![0.0; n_c];
    op.solve_from(&v0, &mut probe);
    if probe.iter().any(|v| !v.is_finite()) {
        return Err(PyrucastError::Message(format!(
            "eigen: a − shift·b is singular (shift = {shift}); a free structure has λ = 0, \
             give it a negative shift"
        )));
    }

    let par = Par::Seq;
    let params = PartialEigenParams::default();
    let mut buffer = MemBuffer::new(partial_eigen_scratch(&op, count, par, params));
    let mut vectors = Mat::<c64>::zeros(b_cols.len(), count);
    let mut thetas = vec![c64::new(0.0, 0.0); count];
    let info = partial_eigen(
        vectors.as_mut(),
        &mut thetas,
        &op,
        ColRef::from_slice(&v0),
        KRYLOV_TOLERANCE,
        par,
        MemStack::new(&mut buffer),
        params,
    );
    if info.n_converged_eigen < count {
        return Err(PyrucastError::Message(format!(
            "eigen: {} of the {count} modes converged",
            info.n_converged_eigen
        )));
    }

    // The whole vector of each mode, on the columns of `c`: x = (a − σb)⁻¹ b v / θ.
    // The real eigenvalues of a symmetric pair come with real vectors.
    let mut xs: Vec<Vec<f64>> = Vec::with_capacity(count);
    for (i, theta) in thetas.iter().enumerate() {
        let v: Vec<f64> = vectors.col(i).iter().map(|z| z.re).collect();
        let mut x = vec![0.0; n_c];
        op.solve_from(&v, &mut x);
        x.iter_mut().for_each(|e| *e /= theta.re);
        xs.push(x);
    }

    // Rayleigh-Ritz on the modes found: Â y = λ B̂ y, with Â = Xᵀ(A X) and
    // B̂ = Xᵀ(B X) — the products pair each row with the column of its primal.
    // Modes of a common eigenvalue come out b-orthogonal.
    let pairs = pairs_of(&[a]);
    let partners = row_partners(&c, &pairs)?;
    let gather = |x: &[f64], cols: &[usize]| -> Vec<f64> { cols.iter().map(|&k| x[k]).collect() };
    let mut axs = Vec::with_capacity(count);
    let mut bxs = Vec::with_capacity(count);
    for x in &xs {
        axs.push(scatter_product(a, &a_rows, &gather(x, &a_cols), n_c)?);
        bxs.push(scatter_product(b, &b_rows, &gather(x, &b_cols), n_c)?);
    }
    let a_hat = Mat::<f64>::from_fn(count, count, |i, j| paired_dot(&partners, &axs[j], &xs[i]));
    let b_hat = Mat::<f64>::from_fn(count, count, |i, j| paired_dot(&partners, &bxs[j], &xs[i]));
    let (lambdas, y) = rayleigh_ritz(&a_hat, &b_hat)?;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&i, &j| lambdas[i].total_cmp(&lambdas[j]));

    let combine = |vs: &[Vec<f64>], k: usize| -> Vec<f64> {
        let mut out = vec![0.0; n_c];
        for (j, v) in vs.iter().enumerate() {
            let w = y[(j, k)];
            out.iter_mut().zip(v).for_each(|(o, e)| *o += w * e);
        }
        out
    };
    let multiplier_cols = multiplier_columns(&c)?;
    let primal_names: Vec<String> = c
        .dof_vars()
        .iter()
        .filter(|v| v.as_str() != MULTIPLIER)
        .cloned()
        .collect();
    let (nodes, zone) = new_support(&c, count)?;
    let mut modes = Vec::with_capacity(count);
    for (rank, &k) in order.iter().enumerate() {
        let lambda = lambdas[k];
        let (mut x, ax, bx) = (combine(&xs, k), combine(&axs, k), combine(&bxs, k));
        let residual = ax
            .iter()
            .zip(&bx)
            .map(|(p, q)| (p - lambda * q).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = norm(&ax) + lambda.abs() * norm(&bx);
        if residual > RESIDUAL_TOLERANCE * scale {
            return Err(PyrucastError::Message(format!(
                "eigen: mode {rank} (λ = {lambda}) did not converge: relative residual {:e}",
                residual / scale
            )));
        }
        let g = paired_dot(&partners, &bx, &x);
        let largest = x
            .iter()
            .zip(&multiplier_cols)
            .filter(|&(_, is_multiplier)| !is_multiplier)
            .map(|(v, _)| *v)
            .fold(0.0, |m: f64, v| if v.abs() > m.abs() { v } else { m });
        let s = match normalization {
            Normalization::Mass => largest.signum() / g.abs().sqrt(),
            Normalization::Max => 1.0 / largest,
        };
        x.iter_mut().for_each(|e| *e *= s);
        modes.push(Mode {
            node: nodes[rank].clone(),
            support: support_mesh(&zone)?,
            eigenvalue: lambda,
            eigenvalue_imag: None,
            shape: c
                .field_from_col_values(&x)?
                .filter_components(primal_names.clone()),
            shape_imag: None,
            pairs: pairs.clone(),
            generalized_mass: g * s * s,
        });
    }
    Ok(modes)
}

/// The complex modes of `[K + iωC − ω²M] X = 0`, for small dense problems,
/// sorted by `|ω|`, then by `Re ω`.
///
/// `q(t) = X e^{iωt}`: `Re ω` is the damped angular frequency and `Im ω` the
/// decay rate, positive for a damped mode. The problem is written in first
/// order, `z = [x ; iωx]`, and solved by faer's QZ on dense `2n × 2n`
/// matrices — `n` the unknowns of the three matrices together. Every finite
/// eigenvalue comes out: an oscillating mode with its mirror `−ω̄`, of
/// conjugate shape.
///
/// It is meant for matrices projected on real modes (`project_on_basis`): the
/// shapes then hold modal coordinates, which `recombine` brings back to the
/// physical unknowns, part by part, with the real modes.
///
/// # Parameters
///
/// - `mass`, `stiffness`, `damping`: assembled matrices without relations. No
///   damping is the empty matrix, `Matrix::default()` (`None` in Python).
/// - `normalization`: `"mass"` scales each shape to `XᴴMX = 1`, `"max"` its
///   value of largest modulus to `1` ([`Normalization`]; `"mass"` in Python).
///   Either way, that value is real and positive.
///
/// # Errors
///
/// - a matrix carries relations (multipliers have no mass, and would give
///   infinite eigenvalues): project it first;
/// - a matrix has no assembled values;
/// - the three matrices together are not square;
/// - the QZ iteration does not converge.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::solver::{self, Normalization};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let p = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[p.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let point = |value: f64| -> pyrucast::Result<Matrix> {
/// #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
/// #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
/// #     z.add_entry(p.id(), "f", p.id(), "u", value)?;
/// #     let mut m = Matrix::empty();
/// #     m.add_sub(Handle::new(z))?;
/// #     m.assemble()?;
/// #     Ok(m)
/// # };
/// // m = 40, k = 4e5, c = 400 (5 % of the critical damping):
/// // ω = ±√(4km − c²)/2m + i·c/2m = ±99.8749 + 5i.
/// let modes = solver::quadratic_eigen(&point(40.0)?, &point(4e5)?, &point(400.0)?,
///                                     Normalization::Mass)?;
/// let damped = (6.4e7f64 - 1.6e5).sqrt() / 80.0;
/// assert!((modes[1].eigenvalue - damped).abs() < 1e-9);
/// assert!((modes[1].eigenvalue_imag.unwrap() - 5.0).abs() < 1e-9);
/// assert!((modes[0].eigenvalue + damped).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn quadratic_eigen(
    mass: &Matrix,
    stiffness: &Matrix,
    damping: &Matrix,
    normalization: Normalization,
) -> Result<Vec<Mode>> {
    for (m, name) in [
        (mass, "mass"),
        (stiffness, "stiffness"),
        (damping, "damping"),
    ] {
        if m.dof_vars().iter().any(|v| v == MULTIPLIER) {
            return Err(PyrucastError::Message(format!(
                "quadratic_eigen: `{name}` carries relations, whose multipliers have no mass; \
                 project it on real modes first"
            )));
        }
        if !m.is_empty() && !m.is_assembled() {
            return Err(PyrucastError::Message(format!(
                "quadratic_eigen: `{name}` has no assembled values; call assemble() first"
            )));
        }
    }
    let mut d = (stiffness + mass)?;
    if !damping.is_empty() {
        d = (&d + damping)?;
    }
    let n = d.n_cols()?;
    if d.n_rows()? != n {
        return Err(PyrucastError::Message(format!(
            "quadratic_eigen: the matrices make a {}×{n} system, not a square one",
            d.n_rows()?
        )));
    }
    let dense = |m: &Matrix| -> Result<Mat<f64>> {
        let mut out = Mat::<f64>::zeros(n, n);
        if m.is_empty() {
            return Ok(out);
        }
        let (rows, cols) = reindex(m, &d)?;
        let (offsets, columns, values) = m.csr_arrays()?;
        for (r, &row) in rows.iter().enumerate() {
            for k in offsets[r]..offsets[r + 1] {
                out[(row, cols[columns[k]])] += values[k];
            }
        }
        Ok(out)
    };
    let (m_d, k_d, c_d) = (dense(mass)?, dense(stiffness)?, dense(damping)?);

    // P z = s Q z with z = [x ; s x] and s = iω.
    let mut p = Mat::<f64>::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
        (true, true) => 0.0,
        (true, false) => f64::from(u8::from(j - n == i)),
        (false, true) => -k_d[(i - n, j)],
        (false, false) => -c_d[(i - n, j - n)],
    });
    let mut q = Mat::<f64>::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
        (true, true) => f64::from(u8::from(i == j)),
        (false, false) => m_d[(i - n, j - n)],
        _ => 0.0,
    });
    let par = Par::Seq;
    let mut alpha_re = Diag::<f64>::zeros(2 * n);
    let mut alpha_im = Diag::<f64>::zeros(2 * n);
    let mut beta = Diag::<f64>::zeros(2 * n);
    let mut right = Mat::<f64>::zeros(2 * n, 2 * n);
    // faer 0.24.4's `gevd_scratch` leaves out what the right eigenvectors take
    // — two norms of `2n` and two `2n × 2` work matrices, held together — which
    // only the smallest problems notice: it is reserved here.
    let norms = faer::linalg::temp_mat_scratch::<f64>(2 * n, 1);
    let vectors = faer::linalg::temp_mat_scratch::<f64>(2 * n, 2);
    let mut buffer = MemBuffer::new(
        gevd_scratch::<f64>(
            2 * n,
            ComputeEigenvectors::No,
            ComputeEigenvectors::Yes,
            par,
            Default::default(),
        )
        .or(StackReq::all_of(&[norms, norms, vectors, vectors])),
    );
    gevd_real(
        p.as_mut(),
        q.as_mut(),
        alpha_re.as_mut(),
        alpha_im.as_mut(),
        beta.as_mut(),
        None,
        Some(right.as_mut()),
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    )
    .map_err(|e| {
        PyrucastError::Message(format!("quadratic_eigen: the QZ iteration failed: {e:?}"))
    })?;

    // Each finite eigenvalue with its x = the first n components of z, as
    // (Re ω, Im ω, Re x, Im x). A complex pair occupies two columns of
    // `right`, the real and the imaginary part of the first eigenvalue's
    // vector; the second eigenvalue and vector are their conjugates, written
    // from the first (faer 0.24.4 does not scale the second `α` by its `β`).
    let mut found: Vec<(f64, f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(2 * n);
    let mut i = 0;
    while i < 2 * n {
        let pair = alpha_im[i] != 0.0;
        if beta[i] != 0.0 {
            let column = |j: usize| -> Vec<f64> { (0..n).map(|r| right[(r, j)]).collect() };
            let (x_re, x_im) = if pair {
                (column(i), column(i + 1))
            } else {
                (column(i), vec![0.0; n])
            };
            // ω = −i·s, s = (α_re + i α_im)/β.
            found.push((
                alpha_im[i] / beta[i],
                -alpha_re[i] / beta[i],
                x_re.clone(),
                x_im.clone(),
            ));
            if pair {
                let conjugate = x_im.iter().map(|v| -v).collect();
                found.push((
                    -alpha_im[i] / beta[i],
                    -alpha_re[i] / beta[i],
                    x_re,
                    conjugate,
                ));
            }
        }
        i += if pair { 2 } else { 1 };
    }
    found.sort_by(|a, b| {
        a.0.hypot(a.1)
            .total_cmp(&b.0.hypot(b.1))
            .then(a.0.total_cmp(&b.0))
    });

    let pairs = pairs_of(&[stiffness, mass, damping]);
    let partners = row_partners(&d, &pairs)?;
    let (nodes, zone) = new_support(&d, found.len())?;
    let mut modes = Vec::with_capacity(found.len());
    for (rank, (omega_re, omega_im, mut x_re, mut x_im)) in found.into_iter().enumerate() {
        // XᴴMX, the rows paired with the columns of their primals.
        let m_re = dense_product(&m_d, &x_re);
        let m_im = dense_product(&m_d, &x_im);
        let h: f64 = partners
            .iter()
            .map(|&(r, c)| x_re[c] * m_re[r] + x_im[c] * m_im[r])
            .sum();
        let (re, im) = x_re
            .iter()
            .zip(&x_im)
            .fold((0.0f64, 0.0f64), |(a, b), (&u, &v)| {
                if u.hypot(v) > a.hypot(b) {
                    (u, v)
                } else {
                    (a, b)
                }
            });
        let modulus = re.hypot(im);
        // The factor that makes the largest component real and positive,
        // then scales: conj(z)/|z|/√h for "mass", 1/z for "max".
        let (f_re, f_im) = match normalization {
            Normalization::Mass => {
                let s = modulus * h.abs().sqrt();
                (re / s, -im / s)
            }
            Normalization::Max => {
                let s = modulus * modulus;
                (re / s, -im / s)
            }
        };
        for (u, v) in x_re.iter_mut().zip(x_im.iter_mut()) {
            (*u, *v) = (*u * f_re - *v * f_im, *u * f_im + *v * f_re);
        }
        modes.push(Mode {
            node: nodes[rank].clone(),
            support: support_mesh(&zone)?,
            eigenvalue: omega_re,
            eigenvalue_imag: Some(omega_im),
            shape: d.field_from_col_values(&x_re)?,
            shape_imag: Some(d.field_from_col_values(&x_im)?),
            pairs: pairs.clone(),
            generalized_mass: h * (f_re * f_re + f_im * f_im),
        });
    }
    Ok(modes)
}

/// `m · x` for a dense square `m`.
fn dense_product(m: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    (0..m.nrows())
        .map(|r| (0..m.ncols()).map(|c| m[(r, c)] * x[c]).sum())
        .collect()
}

/// The POI1 zone shared by `modes`, and the rank in it of each mode's node,
/// once the list is checked: what the operators that project on modes, or
/// recombine them, rely on. `modes` is not empty.
///
/// # Errors
///
/// - the support is not one POI1 zone;
/// - two modes carry different supports, or the support's nodes are not
///   exactly the modes' — a slice or a union of lists that kept its old
///   support: the message gives the way to give it its own;
/// - a mode is complex, or has no pairs.
pub(crate) fn real_support_zone(modes: &[Mode], op: &str) -> Result<(Handle<SubMesh>, Vec<usize>)> {
    const RECIPE: &str = "give the list its own support, on the modes' nodes: \
        `s = mesh.poi1_from_nodes([m[\"node\"] for m in modes])`, then \
        `[{**m, \"support\": s} for m in modes]`";
    let zone = modes[0].support.unit()?;
    let guard = zone.read();
    if guard.element_type() != ElementType::POI1 {
        return Err(PyrucastError::Message(format!(
            "{op}: the support of the modes is a {} mesh, not a POI1 one",
            guard.element_type()
        )));
    }
    let index = guard.node_index();
    let coords = guard.coords();
    let mut taken = vec![false; index.len()];
    let mut ranks = Vec::with_capacity(modes.len());
    for (i, mode) in modes.iter().enumerate() {
        if mode.eigenvalue_imag.is_some() || mode.shape_imag.is_some() {
            return Err(PyrucastError::Message(format!(
                "{op}: mode {i} is complex; only real modes are projected on and recombined"
            )));
        }
        if mode.pairs.is_empty() {
            return Err(PyrucastError::Message(format!(
                "{op}: mode {i} has no pairs"
            )));
        }
        let same = mode.support.len() == 1 && mode.support[0].same_object(&zone);
        let rank = mode
            .node
            .coords()
            .same_object(&coords)
            .then(|| index.get(&mode.node.id()).copied())
            .flatten();
        match rank {
            Some(r) if same && !taken[r] => {
                taken[r] = true;
                ranks.push(r);
            }
            _ => {
                return Err(PyrucastError::Message(format!(
                    "{op}: the modes do not share one support whose nodes are exactly theirs \
                     (mode {i}); {RECIPE}"
                )))
            }
        }
    }
    if ranks.len() != index.len() {
        return Err(PyrucastError::Message(format!(
            "{op}: the support of the modes holds {} nodes for {} modes; {RECIPE}",
            index.len(),
            ranks.len()
        )));
    }
    drop(guard);
    Ok((zone, ranks))
}

/// The shift-inverted operator `v ↦ ((a − σb)⁻¹ b v)` restricted to the
/// columns of `b`: the Krylov vectors hold only the unknowns `b` carries, so
/// the multipliers, at the scale of the structure's stiffness, never enter
/// their scalar products.
struct ShiftInvert<'a> {
    /// The factorization of `c = a − σb`.
    fact: &'a Factorization,
    b: &'a Matrix,
    /// The rank in `c` of each row of `b`.
    b_rows: &'a [usize],
    /// The rank in `c` of each column of `b`.
    b_cols: &'a [usize],
    /// The size of `c`.
    n_c: usize,
}

impl ShiftInvert<'_> {
    /// `x = c⁻¹ (b v)`, on the columns of `c`, written into `x`.
    fn solve_from(&self, v: &[f64], x: &mut [f64]) {
        let y = self
            .b
            .mul_values(v)
            .expect("b is assembled: checked before the iteration");
        x.fill(0.0);
        for (&row, value) in self.b_rows.iter().zip(y) {
            x[row] = value;
        }
        self.fact.solve_in_place(x);
    }
}

impl std::fmt::Debug for ShiftInvert<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ShiftInvert({} unknowns of b, {} of a − σb)",
            self.b_cols.len(),
            self.n_c
        )
    }
}

impl LinOp<f64> for ShiftInvert<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<f64>(self.b_cols.len() + self.n_c)
    }

    fn nrows(&self) -> usize {
        self.b_cols.len()
    }

    fn ncols(&self) -> usize {
        self.b_cols.len()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        _par: Par,
        stack: &mut MemStack,
    ) {
        let n_b = self.b_cols.len();
        // The two buffers of a product, carved once from the iteration's stack.
        let (mut buffer, _) = stack.make_with(n_b + self.n_c, |_| 0.0f64);
        let (v, x) = buffer.split_at_mut(n_b);
        for j in 0..rhs.ncols() {
            v.iter_mut()
                .zip(rhs.col(j).iter())
                .for_each(|(d, s)| *d = *s);
            self.solve_from(v, x);
            out.rb_mut()
                .col_mut(j)
                .iter_mut()
                .zip(self.b_cols)
                .for_each(|(o, &k)| *o = x[k]);
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

/// A starting vector of the Krylov iteration: pseudo-random entries in
/// `[−1, 1)` from a fixed seed (splitmix64). The run is deterministic, and no
/// accidental symmetry of the vector hides a mode from the Krylov space, as a
/// vector of ones would on a symmetric structure.
fn starting_vector(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// The eigenvalues `λ` of the small symmetric pencil `a_hat y = λ b_hat y`, and
/// its eigenvectors `y`, columns of the returned matrix.
///
/// The pencil is reduced by the Cholesky factor `L` of its definite member —
/// `b_hat` for vibrations, `a_hat` when `b_hat` is not (buckling) — to the
/// symmetric `L⁻¹ M L⁻ᵀ`, whose orthonormal eigenvectors `z` give
/// `y = L⁻ᵀ z`: vectors of a common eigenvalue are orthogonal for both
/// members.
fn rayleigh_ritz(a_hat: &Mat<f64>, b_hat: &Mat<f64>) -> Result<(Vec<f64>, Mat<f64>)> {
    let (llt, other, reciprocal) = match b_hat.llt(Side::Lower) {
        Ok(llt) => (llt, a_hat, false),
        Err(_) => (
            a_hat.llt(Side::Lower).map_err(|_| {
                PyrucastError::Message(
                    "eigen: neither a nor b is positive definite on the modes found".into(),
                )
            })?,
            b_hat,
            true,
        ),
    };
    let l = llt.L();
    // L⁻¹ M L⁻ᵀ = (L⁻¹ (L⁻¹ M)ᵀ)ᵀ, M being symmetric.
    let mut reduced = other.to_owned();
    solve_lower_triangular_in_place(l, reduced.as_mut(), Par::Seq);
    let mut reduced = reduced.transpose().to_owned();
    solve_lower_triangular_in_place(l, reduced.as_mut(), Par::Seq);
    let evd = reduced.self_adjoint_eigen(Side::Lower).map_err(|e| {
        PyrucastError::Message(format!("eigen: the Rayleigh-Ritz step failed: {e:?}"))
    })?;
    let mut y = evd.U().to_owned();
    solve_upper_triangular_in_place(l.transpose(), y.as_mut(), Par::Seq);
    let s = evd.S();
    let lambdas = (0..s.dim())
        .map(|i| if reciprocal { 1.0 / s[i] } else { s[i] })
        .collect();
    Ok((lambdas, y))
}

/// `m · x`, with `x` on the columns of `m`, scattered onto the rows of a
/// larger matrix by `rows` (the rank there of each row of `m`).
fn scatter_product(m: &Matrix, rows: &[usize], x: &[f64], n: usize) -> Result<Vec<f64>> {
    let y = m.mul_values(x)?;
    let mut out = vec![0.0; n];
    for (&row, value) in rows.iter().zip(y) {
        out[row] = value;
    }
    Ok(out)
}

/// `Σ y[row] · x[col]` over the `(row, col)` partners of a matrix: the product
/// of a row vector and a column vector of it.
fn paired_dot(partners: &[(usize, usize)], y: &[f64], x: &[f64]) -> f64 {
    partners.iter().map(|&(r, c)| y[r] * x[c]).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|e| e * e).sum::<f64>().sqrt()
}

/// The rank in `onto` of each row and of each column of `from`, in `from`'s
/// order. `onto` is a sum that contains `from`, so every key is found.
pub(super) fn reindex(from: &Matrix, onto: &Matrix) -> Result<(Vec<usize>, Vec<usize>)> {
    let (from_rows, from_cols) = from.dof_key_orders()?;
    let (onto_rows, onto_cols) = onto.dof_key_orders()?;
    let onto_vars = onto.dof_vars();
    let slot: HashMap<&str, u32> = onto_vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i as u32))
        .collect();
    // `from`'s variable slots, resolved once in `onto`'s table.
    let to_onto: Vec<u32> = from.dof_vars().iter().map(|v| slot[v.as_str()]).collect();
    let rank = |keys: &[DofKey], onto_keys: &[DofKey]| -> Vec<usize> {
        let index: HashMap<DofKey, usize> =
            onto_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        keys.iter()
            .map(|&k| index[&dof_key(dof_node(k), to_onto[dof_var(k) as usize])])
            .collect()
    };
    Ok((rank(&from_rows, &onto_rows), rank(&from_cols, &onto_cols)))
}

/// The `(primal, dual)` pairs of the blocks of `matrices`: each block whose
/// dual and primal names are as many pairs them in order, as the models
/// declare them. The relations' multiplier and imposed value pair nothing.
pub(super) fn pairs_of(matrices: &[&Matrix]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for m in matrices {
        for h in m.iter() {
            let block = h.read();
            if block.dual_vars().len() != block.primal_vars().len() {
                continue;
            }
            for (primal, dual) in block.primal_vars().iter().zip(block.dual_vars()) {
                let pair = (primal.clone(), dual.clone());
                if primal != MULTIPLIER && dual != IMPOSED && !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            }
        }
    }
    pairs
}

/// The `(row, column)` partners of `m`: for each row whose dual `pairs` names,
/// the column of its primal at the same node.
pub(super) fn row_partners(m: &Matrix, pairs: &[(String, String)]) -> Result<Vec<(usize, usize)>> {
    let (rows, cols) = m.dof_key_orders()?;
    let vars = m.dof_vars();
    let slot: HashMap<&str, u32> = vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i as u32))
        .collect();
    let mut primal_of: HashMap<u32, u32> = HashMap::new();
    for (primal, dual) in pairs {
        if let (Some(&d), Some(&p)) = (slot.get(dual.as_str()), slot.get(primal.as_str())) {
            primal_of.entry(d).or_insert(p);
        }
    }
    let col_index: HashMap<DofKey, usize> = cols.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    Ok(rows
        .iter()
        .enumerate()
        .filter_map(|(r, &k)| {
            let p = *primal_of.get(&dof_var(k))?;
            Some((r, *col_index.get(&dof_key(dof_node(k), p))?))
        })
        .collect())
}

/// Whether each column of `m` is a multiplier.
fn multiplier_columns(m: &Matrix) -> Result<Vec<bool>> {
    let (_, cols) = m.dof_key_orders()?;
    let vars = m.dof_vars();
    Ok(cols
        .iter()
        .map(|&k| vars[dof_var(k) as usize] == MULTIPLIER)
        .collect())
}

/// `count` new nodes at the origin of `matrix`'s `Coords`, and the POI1 zone
/// that holds them, in order: the nodes and the support of a list of modes.
pub(super) fn new_support(matrix: &Matrix, count: usize) -> Result<(Vec<Node>, Handle<SubMesh>)> {
    let coords = matrix
        .iter()
        .next()
        .ok_or_else(|| PyrucastError::Message("eigen: the matrix has no block".into()))?
        .read()
        .coords();
    let origin = vec![0.0; coords.read().dim() as usize];
    let nodes = (0..count)
        .map(|_| Node::create_in(coords.clone(), &origin))
        .collect::<Result<Vec<Node>>>()?;
    let zone = crate::ops::mesh::poi1_from_nodes(&nodes)?.unit()?;
    Ok((nodes, zone))
}

/// A mesh of the one zone `zone`: what each mode of a list holds, the zone's
/// handle shared.
pub(super) fn support_mesh(zone: &Handle<SubMesh>) -> Result<Mesh> {
    let mut mesh = Mesh::default();
    mesh.add_sub(zone.clone())?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    use crate::coords::Coords;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::models::RelationSense;
    use crate::ops::{element_field, matrix, mesh, model};

    /// A line of `n` unit SEG2 cells and its `n + 1` nodes.
    fn line(dim: u8, n: usize) -> (Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(dim).unwrap());
        let nodes: Vec<Node> = (0..=n)
            .map(|i| {
                let mut p = vec![0.0; dim as usize];
                p[0] = i as f64;
                Node::create_in(coords.clone(), &p).unwrap()
            })
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        for i in 0..n {
            sm.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        (Mesh::from_submesh(sm), nodes)
    }

    /// The stiffness of a chain of unit springs (heat conduction, k = 1), held
    /// at its first node by a relation of `sense`.
    fn chain_stiffness(n: usize, sense: RelationSense) -> (Matrix, Vec<Node>) {
        let (line, nodes) = line(1, n);
        let fes = FiniteElementSpace::lagrange1(&line).unwrap();
        let chain = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let held = mesh::poi1_from_nodes(&nodes[..1]).unwrap();
        let held = model::dirichlet(&held, &chain, "T", sense).unwrap();
        let both = chain.union(&held).unwrap();
        let materials = element_field::material_field(&both, &[("k", 1.0)]).unwrap();
        (matrix::stiffness(&both, &materials).unwrap(), nodes)
    }

    /// A diagonal "mass" `values` on `nodes`, rows `q`, columns `T`.
    fn lumped(nodes: &[Node], values: &[f64], symmetry: Symmetry) -> Matrix {
        let support = mesh::poi1_from_nodes(nodes).unwrap().unit().unwrap();
        let mut z = SubMatrix::new(
            support.clone(),
            support,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            symmetry,
        )
        .unwrap();
        for (node, &v) in nodes.iter().zip(values) {
            z.add_entry(node.id(), "q", node.id(), "T", v).unwrap();
        }
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(z)).unwrap();
        m.assemble().unwrap();
        m
    }

    #[test]
    fn eigen_holds_relations_and_massless_unknowns() {
        // Four nodes, the first held, a unit mass on nodes 1 and 3, none on 2.
        // Condensing T2 = (T1 + T3)/2 leaves K = [[1.5, −0.5], [−0.5, 0.5]] and
        // M = I: λ = 1 ∓ √½.
        let (k, n) = chain_stiffness(3, RelationSense::Equality);
        let m = lumped(&[n[1].clone(), n[3].clone()], &[1.0, 1.0], Symmetry::Full);
        let modes = eigen(&k, &m, 2, 0.0, Normalization::Mass).unwrap();
        let expected = [1.0 - 0.5f64.sqrt(), 1.0 + 0.5f64.sqrt()];
        for (mode, lambda) in modes.iter().zip(expected) {
            assert!(
                (mode.eigenvalue - lambda).abs() < 1e-10,
                "{} ≠ {lambda}",
                mode.eigenvalue
            );
            let t = |i: usize| mode.shape.value(n[i].id(), "T").unwrap();
            assert!(t(0).abs() < 1e-12, "the relation holds");
            assert!(
                (t(2) - 0.5 * (t(1) + t(3))).abs() < 1e-12,
                "static condensation"
            );
            assert_eq!(
                mode.shape.components(),
                vec!["T".to_string()],
                "no multiplier"
            );
            assert!((mode.generalized_mass - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn eigen_shift_selects_the_nearest_modes() {
        // A fixed-free chain of three unit masses: λₖ = 4 sin²((2k − 1)π/14).
        let (k, n) = chain_stiffness(3, RelationSense::Equality);
        let m = lumped(&n[1..], &[1.0; 3], Symmetry::Full);
        let lambda = |k: f64| {
            4.0 * ((2.0 * k - 1.0) * std::f64::consts::PI / 14.0)
                .sin()
                .powi(2)
        };
        // 2 lies between λ₂ = 1.555 and λ₃ = 3.247, nearer both than λ₁ = 0.198.
        let modes = eigen(&k, &m, 2, 2.0, Normalization::Mass).unwrap();
        assert!((modes[0].eigenvalue - lambda(2.0)).abs() < 1e-10);
        assert!((modes[1].eigenvalue - lambda(3.0)).abs() < 1e-10);
    }

    #[test]
    fn eigen_double_eigenvalue_gives_b_orthogonal_shapes() {
        // A cantilever of square section bends alike in its two planes.
        let (line, nodes) = line(3, 10);
        let fes = FiniteElementSpace::new(&line, Interpolation::Hermite3).unwrap();
        let beam = model::bernoulli(&fes).unwrap();
        let clamp = mesh::poi1_from_nodes(&nodes[..1]).unwrap();
        let hold = |v: &str| model::dirichlet(&clamp, &beam, v, RelationSense::Equality).unwrap();
        let vars = beam.primal_vars();
        let mut held = beam.union(&hold(&vars[0])).unwrap();
        for v in &vars[1..] {
            held = held.union(&hold(v)).unwrap();
        }
        let values = [
            ("E", 2e11),
            ("A", 0.01),
            ("I_y", 1e-5),
            ("I_z", 1e-5),
            ("J", 2e-5),
            ("G", 8e10),
            ("rho", 7800.0),
        ];
        let materials = element_field::material_field(&held, &values).unwrap();
        let k = matrix::stiffness(&held, &materials).unwrap();
        let m = matrix::mass(&beam, &materials).unwrap();
        let modes = eigen(&k, &m, 2, 0.0, Normalization::Mass).unwrap();
        let (a, b) = (&modes[0], &modes[1]);
        assert!(
            (a.eigenvalue / b.eigenvalue - 1.0).abs() < 1e-8,
            "{} {}",
            a.eigenvalue,
            b.eigenvalue
        );
        let pairs: Vec<(&str, &str)> = a
            .pairs
            .iter()
            .map(|(p, d)| (p.as_str(), d.as_str()))
            .collect();
        let cross = a
            .shape
            .dot_field(&m.mul_field(&b.shape).unwrap(), Some(&pairs));
        assert!(cross.abs() < 1e-12, "φ₁ᵀMφ₂ = {cross}");
    }

    #[test]
    fn eigen_normalizations() {
        // The point of vibr10: m = 40, k = 4e5.
        let (_, n) = line(1, 1);
        let k = lumped(&n[..1], &[4e5], Symmetry::Full);
        let m = lumped(&n[..1], &[40.0], Symmetry::Full);
        let by_mass = &eigen(&k, &m, 1, 0.0, Normalization::Mass).unwrap()[0];
        assert!(
            (by_mass.shape.value(n[0].id(), "T").unwrap() - 0.158_113_883_008_418_97).abs() < 1e-15
        );
        assert!((by_mass.generalized_mass - 1.0).abs() < 1e-14);
        let by_max = &eigen(&k, &m, 1, 0.0, Normalization::Max).unwrap()[0];
        assert_eq!(by_max.shape.value(n[0].id(), "T").unwrap(), 1.0);
        assert!((by_max.generalized_mass - 40.0).abs() < 1e-12);
    }

    #[test]
    fn eigen_refusals_and_counts() {
        let (k, n) = chain_stiffness(3, RelationSense::Equality);
        let m = lumped(&n[1..], &[1.0; 3], Symmetry::Full);
        let skew = lumped(&n[1..], &[1.0; 3], Symmetry::None);
        let err = eigen(&k, &skew, 1, 0.0, Normalization::Mass)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`b` is not symmetric"), "{err}");
        let (floor, _) = chain_stiffness(3, RelationSense::GreaterEqual);
        let err = eigen(&floor, &m, 1, 0.0, Normalization::Mass)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unilateral"), "{err}");
        // Without its relation, the chain is free: λ = 0 makes `a` singular.
        let (line, _) = line(1, 3);
        let fes = FiniteElementSpace::lagrange1(&line).unwrap();
        let free = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let materials = element_field::material_field(&free, &[("k", 1.0)]).unwrap();
        let free = matrix::stiffness(&free, &materials).unwrap();
        let err = eigen(&free, &mass_of(&free), 1, 0.0, Normalization::Mass)
            .unwrap_err()
            .to_string();
        assert!(err.contains("singular"), "{err}");
        assert!(eigen(&k, &m, 0, 0.0, Normalization::Mass)
            .unwrap()
            .is_empty());
        assert_eq!(eigen(&k, &m, 5, 0.0, Normalization::Mass).unwrap().len(), 3);
    }

    /// The point of vibr10, k and m on the unknown `T`, `c` the damping.
    fn oscillator(c: f64) -> (Matrix, Matrix, Matrix) {
        let (_, n) = line(1, 1);
        let k = lumped(&n[..1], &[4e5], Symmetry::Full);
        let m = lumped(&n[..1], &[40.0], Symmetry::Full);
        let damping = if c == 0.0 {
            Matrix::empty()
        } else {
            lumped(&n[..1], &[c], Symmetry::Full)
        };
        (m, k, damping)
    }

    #[test]
    fn quadratic_eigen_damped_oscillator() {
        // c = 2ζ√(km) with ζ = 0.05: ω = ±99.8749 + 5i, of modulus √(k/m) = 100.
        let (m, k, c) = oscillator(400.0);
        let modes = quadratic_eigen(&m, &k, &c, Normalization::Mass).unwrap();
        assert_eq!(modes.len(), 2);
        let damped = (6.4e7f64 - 1.6e5).sqrt() / 80.0;
        for (mode, re) in modes.iter().zip([-damped, damped]) {
            assert!((mode.eigenvalue - re).abs() < 1e-9, "{}", mode.eigenvalue);
            assert!((mode.eigenvalue_imag.unwrap() - 5.0).abs() < 1e-9);
            assert!((mode.generalized_mass - 1.0).abs() < 1e-12);
            let node = mode
                .shape
                .iter()
                .next()
                .unwrap()
                .read()
                .support()
                .read()
                .connectivity()[0];
            let x = mode.shape.value(node, "T").unwrap();
            let y = mode.shape_imag.as_ref().unwrap().value(node, "T").unwrap();
            assert!(
                (x - 40f64.sqrt().recip()).abs() < 1e-12 && y == 0.0,
                "{x} {y}"
            );
        }
    }

    #[test]
    fn quadratic_eigen_without_damping_matches_eigen() {
        // Two unit masses on springs of 2, coupled by 1: λ = 2 and 4, so
        // ω = ±√2 and ±2, the shapes real.
        let (_, n) = line(1, 1);
        let support = mesh::poi1_from_nodes(&n).unwrap().unit().unwrap();
        let mut z = SubMatrix::new(
            support.clone(),
            support,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        )
        .unwrap();
        for (i, j, v) in [(0, 0, 3.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 3.0)] {
            z.add_entry(n[i].id(), "q", n[j].id(), "T", v).unwrap();
        }
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(z)).unwrap();
        k.assemble().unwrap();
        let m = lumped(&n, &[1.0, 1.0], Symmetry::Full);
        let complex = quadratic_eigen(&m, &k, &Matrix::empty(), Normalization::Mass).unwrap();
        let real = eigen(&k, &m, 2, 0.0, Normalization::Mass).unwrap();
        let omegas: Vec<f64> = complex.iter().map(|mode| mode.eigenvalue).collect();
        let s2 = 2f64.sqrt();
        for (w, expected) in omegas.iter().zip([-s2, s2, -2.0, 2.0]) {
            assert!((w - expected).abs() < 1e-12, "{omegas:?}");
        }
        for (mode, same) in complex.iter().zip([&real[0], &real[0], &real[1], &real[1]]) {
            assert!(mode.eigenvalue_imag.unwrap().abs() < 1e-12);
            for node in &n {
                let x = mode.shape.value(node.id(), "T").unwrap();
                let y = mode
                    .shape_imag
                    .as_ref()
                    .unwrap()
                    .value(node.id(), "T")
                    .unwrap();
                assert!(
                    (x - same.shape.value(node.id(), "T").unwrap()).abs() < 1e-12
                        && y.abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn quadratic_eigen_refuses_relations() {
        let (k, n) = chain_stiffness(3, RelationSense::Equality);
        let m = lumped(&n[1..], &[1.0; 3], Symmetry::Full);
        let err = quadratic_eigen(&m, &k, &Matrix::empty(), Normalization::Mass)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`stiffness` carries relations"), "{err}");
    }

    /// A unit mass on every row node of `k`.
    fn mass_of(k: &Matrix) -> Matrix {
        let support = k.iter().next().unwrap().read().row_support().clone();
        let nodes: Vec<Node> = support
            .read()
            .connectivity()
            .iter()
            .map(|&id| Node::acquire(support.read().coords(), id).unwrap())
            .collect();
        lumped(&nodes, &vec![1.0; nodes.len()], Symmetry::Full)
    }
}
