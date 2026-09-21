//! Sparse direct linear solver: `A · x = b`.
//!
//! This module bridges the abstract [`Matrix`] / [`NodeField`] objects and the
//! sparse linear algebra of [`faer`]. It exposes a single free function —
//! [`solve`] — that:
//!
//! 1. hands faer the assembled CSR **borrowed**: one counting-sort transpose
//!    into column-major arrays, wrapped as a `SparseColMatRef`. The matrix is
//!    never materialised in another form;
//! 2. reads a right-hand-side vector out of the `NodeField`, one entry
//!    per **row DOF** of the matrix (zones resolved first-found; missing
//!    entries default to `0.0`);
//! 3. runs a multithreaded sparse factorization — **LU** with partial pivoting
//!    by default, **Cholesky** when the caller asks for it and the matrix is
//!    symmetric positive definite;
//! 4. wraps the solution back into a fresh single-zone `NodeField`
//!    indexed by the **column DOFs** of the matrix.
//!
//! The factorization is **reusable**: it is cached inside the `Matrix`
//! ([`SolveOptions::cache`]), so a Newton loop or a multi-load-case run pays
//! for it once and only redoes descent / back-substitution afterwards —
//! *factor once, solve many*. [`SolveMethod`] is the seam that already carries
//! LU and Cholesky, and through which an iterative back-end could be selected
//! later without touching the call sites.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::field::SubField;
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::coords::Coords;
//! use pyrucast::containers::element_field::{ElementField, SubElementField};
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::model::{Model, SubModel};
//! use pyrucast::atoms::Node;
//! use pyrucast::containers::node_field::{NodeField, SubNodeField};
//! use pyrucast::ops::matrix;
//! use pyrucast::ops::mesh;
//! use pyrucast::ops::solver::lu::solve;
//! use pyrucast::handle::Handle;
//!
//! // 1-D Poisson on [0, 1] with one SEG2 element, k = 1.
//! let coords = Handle::new(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.get(0).unwrap();
//! let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
//! mat.set_uniform("k", 1.0).unwrap();
//! let mut materials = ElementField::empty();
//! materials.add_sub(Handle::new(mat)).unwrap();
//!
//! let mut model = Model::empty();
//! model
//!     .add_sub(Handle::new(SubModel::heat_conduction(sub).unwrap()))
//!     .unwrap();
//! // Dirichlet at both ends: imposed POI1 meshes + colocated multiplier
//! // supports minted by `barycenter`.
//! let imposed_a = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
//! let imposed_b = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&b)).unwrap());
//! let mult_mesh_a = mesh::barycenter(&imposed_a).unwrap();
//! let mult_mesh_b = mesh::barycenter(&imposed_b).unwrap();
//! let dir_a = SubModel::dirichlet(&model, "T", &imposed_a, &mult_mesh_a, Default::default()).unwrap();
//! let dir_b = SubModel::dirichlet(&model, "T", &imposed_b, &mult_mesh_b, Default::default()).unwrap();
//! let mult_a = dir_a.multiplier_nodes()[0];
//! let mult_b = dir_b.multiplier_nodes()[0];
//! model.add_sub(Handle::new(dir_a)).unwrap();
//! model.add_sub(Handle::new(dir_b)).unwrap();
//!
//! // Load: imposed values T_a = 0, T_b = 1 at the multiplier nodes (slot "imposed_T").
//! let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
//! load_sm.add_cell(&[mult_a]).unwrap();
//! load_sm.add_cell(&[mult_b]).unwrap();
//! let load_sm_h = Handle::new(load_sm);
//! let mut rhs = SubNodeField::from_poi1(&load_sm_h, vec!["imposed_T".into()]).unwrap();
//! rhs.set_value(mult_a, "imposed_T", 0.0).unwrap();
//! rhs.set_value(mult_b, "imposed_T", 1.0).unwrap();
//! let rhs = NodeField::from_sub(rhs);
//!
//! let k = pyrucast::ops::matrix::stiffness(&model, &materials).unwrap();
//! let solution = solve(&k, &rhs).unwrap();
//! // Solution: T(a) = 0, T(b) = 1, λ_a = +1, λ_b = -1 (boundary fluxes).
//! assert!((solution.value(a.id(), "T").unwrap() - 0.0).abs() < 1e-12);
//! assert!((solution.value(b.id(), "T").unwrap() - 1.0).abs() < 1e-12);
//! ```

use crate::containers::matrix::{DofKey, Matrix};
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::LltError;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use std::sync::Arc;

/// A faer sparse LU factorization over `usize` indices and `f64` values — the
/// direct back-end shared by [`Factorization`] (full saddle-point system) and
/// [`crate::ops::solver::eliminate`] (reduced condensed system).
pub(crate) type SparseLu = faer::sparse::linalg::solvers::Lu<usize, f64>;

/// A faer sparse Cholesky (`L·Lᵀ`) factorization, for the matrices that admit
/// one.
pub(crate) type SparseLlt = faer::sparse::linalg::solvers::Llt<usize, f64>;

/// The factorization a matrix turned out to admit.
///
/// `L·Lᵀ` stores **one** triangle of factors where `L·U` stores two, searches no
/// pivot, and is ordered by AMD rather than COLAMD — the ordering suited to a
/// symmetric pattern. It is worth attempting whenever the matrix is symmetric,
/// and it is only an attempt: positive-definiteness is a numerical property that
/// depends on the boundary conditions, so nothing can declare it in advance.
/// An unconstrained stiffness is only *semi*-definite, and a Lagrange
/// saddle-point is symmetric **indefinite** — never positive definite.
///
/// Both variants answer `solve_in_place` identically: faer implements
/// `Solve` for anything that implements `SolveCore`.
pub(crate) enum Factored {
    /// `L·Lᵀ` — the matrix was symmetric positive definite.
    Cholesky(SparseLlt),
    /// `L·U` with partial pivoting — everything else.
    Lu(SparseLu),
}

impl Factored {
    /// Solve `A·x = b` in place against this factorization.
    pub(crate) fn solve_in_place(&self, b: &mut [f64]) {
        let n = b.len();
        let x = faer::MatMut::from_column_major_slice_mut(b, n, 1);
        match self {
            Factored::Cholesky(llt) => llt.solve_in_place(x),
            Factored::Lu(lu) => lu.solve_in_place(x),
        }
    }

    /// [`solve_in_place`](Self::solve_in_place) for a caller that owns no buffer.
    pub(crate) fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        self.solve_in_place(&mut x);
        x
    }

    /// Which factorization ran — `"cholesky"` or `"lu"`.
    ///
    /// The one way to observe the attempt's outcome without reading printed
    /// text: what the tests assert on, and what the verbose modes report.
    pub(crate) fn method(&self) -> &'static str {
        match self {
            Factored::Cholesky(_) => "cholesky",
            Factored::Lu(_) => "lu",
        }
    }
}

/// Transpose a square CSR into CSC arrays, by counting sort.
///
/// Columns come out **sorted** for free: the outer loop walks rows in
/// increasing order, so each column receives its row indices in increasing
/// order — which is exactly the invariant [`SymbolicSparseColMatRef::new_checked`]
/// demands. Nothing is sorted, nothing is deduplicated: a CSR that already
/// holds each `(row, col)` once transposes into a CSC that does too.
///
/// Sequential on purpose. The scatter pass carries one cursor per column, which
/// a parallel split would have to partition and merge, for an O(nnz) pass that
/// costs a fraction of the factorization it feeds.
fn transpose_to_csc(
    n: usize,
    offsets: &[usize],
    cols: &[usize],
    vals: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = cols.len();
    let mut col_ptr = vec![0usize; n + 1];
    for &c in cols {
        col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut cursor = col_ptr[..n].to_vec();
    let mut row_idx = vec![0usize; nnz];
    let mut out = vec![0.0f64; nnz];
    for r in 0..n {
        for k in offsets[r]..offsets[r + 1] {
            let j = cols[k];
            let p = cursor[j];
            row_idx[p] = r;
            out[p] = vals[k];
            cursor[j] = p + 1;
        }
    }
    (col_ptr, row_idx, out)
}

/// Factorize a square matrix, given in **CSR** form (faer). The single place the
/// → faer handover lives, so both the Lagrange and the elimination solvers share
/// one implementation — and one reading of the symmetry flag.
///
/// `symmetric` is what the caller guarantees about the **array**, and it decides
/// two things: whether the CSR arrays can be handed over as their own CSC
/// without a turnaround, and whether a Cholesky may be attempted at all.
///
/// Takes the three CSR arrays **borrowed** rather than a `CscMatrix`, and hands
/// faer a borrowed [`SparseColMatRef`] rather than an owned matrix built from
/// triplets. What that removes, in the order it used to happen: the CSR→CSC
/// materialisation, the `Vec<Triplet>` it was unfolded into (24 bytes per
/// non-zero), the index-sorting array faer allocates to put that back in order,
/// and the owned `SparseColMat` at the end. All four used to be **live at once
/// during `sp_lu`** — the moment the factorization is allocating its own
/// factors, which is the worst possible time to be holding four copies of the
/// matrix. Only the counting-sort transpose below remains.
///
/// The caller's arrays must satisfy what faer checks: offsets non-decreasing,
/// and each row's column indices strictly increasing (sorted, no duplicate).
/// Both the assembled pattern ([`crate::ops::scatter::build_pattern`]) and
/// `nalgebra_sparse::CsrMatrix` guarantee this.
pub(crate) fn factorize_csr(
    n: usize,
    offsets: &[usize],
    cols: &[usize],
    vals: &[f64],
    symmetric: bool,
    options: &SolveOptions,
) -> Result<Factored> {
    let start = std::time::Instant::now();
    if n == 0 {
        return Err(PyrucastError::Message("solve: matrix is empty".into()));
    }
    if offsets.len() != n + 1 {
        return Err(PyrucastError::Message(format!(
            "solve: matrix must be square; got {} row offset(s) for {n} column(s)",
            offsets.len()
        )));
    }
    // A symmetric matrix is its own CSC — `CSR(A)` is bit-for-bit `CSC(Aᵀ)`, and
    // `Aᵀ = A` — so the turnaround, and the second full copy of the matrix that
    // comes with it, are skipped. This is the whole point of carrying the flag:
    // the peak alongside the factorization drops from two copies to one.
    //
    // KNOWN DEFECT, being fixed next: the flag describes the **operator**, while
    // what is needed here is the **array**. The global row and column orders are
    // collected in two independent walks over the blocks, so a constraint that
    // introduces two new nodes at once — `embedded`, whose immersed node the
    // physics never numbered — has them discovered in opposite order on the two
    // sides, and lands the multiplier at row `k` against the immersed node at
    // column `k`. The array is then a permutation away from symmetric, and this
    // short-circuit quietly factorizes the transpose.
    let transposed = (!symmetric).then(|| transpose_to_csc(n, offsets, cols, vals));
    let (col_ptr, row_idx, values) = match &transposed {
        Some((ptr, idx, v)) => (&ptr[..], &idx[..], &v[..]),
        None => (offsets, cols, vals),
    };
    let out = match options.method {
        SolveMethod::Lu => Factored::Lu(lu_of(n, col_ptr, row_idx, values)?),
        SolveMethod::Cholesky => {
            // faer cannot catch this one: a Cholesky reads a single triangle, so
            // on a non-symmetric matrix it does not fail — it quietly uses the
            // half it was handed. The refusal has to come from here.
            if !symmetric {
                return Err(PyrucastError::Message(
                    "solve: Cholesky needs a symmetric matrix, and this one does not \
                     declare itself symmetric. A Cholesky reads only one triangle, so it \
                     would not fail on the other half — it would quietly use the wrong \
                     one. Solve with the LU, or fix the model's symmetry declaration."
                        .into(),
                ));
            }
            Factored::Cholesky(cholesky_of(n, col_ptr, row_idx, values).map_err(|e| {
                PyrucastError::Message(match e {
                    LltError::Numeric(
                        faer::linalg::cholesky::llt::factor::LltError::NonPositivePivot { index },
                    ) => format!(
                        "solve: Cholesky refused this matrix at pivot {index} — it is not \
                         positive definite. A Lagrange saddle-point never is; solve it with \
                         the LU, or eliminate the constraints first."
                    ),
                    other => format!("solve: Cholesky could not be set up: {other:?}"),
                })
            })?)
        }
    };
    report(options, n, cols.len(), &out, start);
    Ok(out)
}

/// Account for one factorization, at the level the caller asked for.
fn report(
    options: &SolveOptions,
    n: usize,
    nnz: usize,
    factored: &Factored,
    start: std::time::Instant,
) {
    if let Verbosity::Silent = options.verbosity {
        return;
    }
    print!("solve: {n} DOF, {nnz} nnz, {}", factored.method());
    if let Verbosity::Detailed = options.verbosity {
        print!(" — factorized in {:.2?}", start.elapsed());
    }
    println!();
}

/// `L·Lᵀ` of a symmetric matrix given as borrowed CSC arrays.
///
/// Only one triangle is read — `Lower` and `Upper` are interchangeable here,
/// both being present and equal in a matrix this crate assembles, so the choice
/// is arbitrary.
fn cholesky_of(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
) -> std::result::Result<SparseLlt, LltError> {
    let symbolic = SymbolicSparseColMatRef::<usize>::new_checked(n, n, col_ptr, None, row_idx);
    SparseColMatRef::<usize, f64>::new(symbolic, values).sp_cholesky(faer::Side::Lower)
}

/// `L·U` of a matrix given as borrowed CSC arrays.
fn lu_of(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<SparseLu> {
    // `new_checked` walks the arrays once, allocating nothing, and panics on a
    // malformed one. Kept rather than its unchecked twin: one O(nnz) pass buys
    // an invariant the factorization would otherwise trust blindly, for a
    // fraction of what the factorization itself costs.
    let symbolic = SymbolicSparseColMatRef::<usize>::new_checked(n, n, col_ptr, None, row_idx);
    SparseColMatRef::<usize, f64>::new(symbolic, values)
        .sp_lu()
        .map_err(|e| PyrucastError::Message(format!("solve: LU failed (singular?): {e:?}")))
}

/// Factorize a square matrix already held as **CSC** arrays, borrowed.
///
/// The seam [`factorize_csr`] lands on, and the entry point for a caller that
/// builds its column-major form itself (the active-set solver, which rebuilds a
/// modified system per status — see [`crate::ops::solver::unilateral`]).
pub(crate) fn factorize_csc_arrays(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
) -> Result<SparseLu> {
    lu_of(n, col_ptr, row_idx, values)
}

/// Solve `A·x = b` for one right-hand side against a computed [`SparseLu`]
/// (descent / back-substitution only), **in place**. Shared by both direct
/// solvers.
///
/// faer solves in place, so `b` is both the right-hand side and the result.
/// Wrapping the caller's buffer as a one-column matrix rather than copying it
/// into a fresh `faer::Mat` spares two vectors of length `n` and the two
/// element-by-element passes that used to fill and drain them.
pub(crate) fn lu_solve_in_place(lu: &SparseLu, b: &mut [f64]) {
    let n = b.len();
    let x = faer::MatMut::from_column_major_slice_mut(b, n, 1);
    lu.solve_in_place(x);
}

/// [`lu_solve_in_place`] for a caller that owns no buffer yet.
pub(crate) fn lu_solve_vec(lu: &SparseLu, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    lu_solve_in_place(lu, &mut x);
    x
}

/// Which factorization a solve should run. The caller chooses; nothing is
/// attempted and retried behind their back, because the case that fails is the
/// common one — every multiplier boundary condition assembles a saddle-point —
/// and the failed attempt would be paid on each factorization. The enum leaves
/// room for an iterative back-end without changing the `solve` call sites.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::{SolveMethod, SolveOptions};
/// # use pyrucast::ops::model;
/// // The LU takes any square matrix, and is what a solve runs unless told
/// // otherwise.
/// assert_eq!(SolveMethod::default(), SolveMethod::Lu);
/// let o = SolveOptions { method: SolveMethod::Lu, cache: false, ..Default::default() };
/// assert!(solver::lu::solve_with_options(&k, &charge, &o).is_ok());
///
/// // This one is clamped by a Lagrange multiplier: symmetric, but **indefinite**.
/// // Cholesky refuses it, and says where.
/// let c = SolveOptions { method: SolveMethod::Cholesky, cache: false, ..Default::default() };
/// let err = solver::lu::solve_with_options(&k, &charge, &c).unwrap_err().to_string();
/// assert!(err.contains("positive definite"), "{err}");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolveMethod {
    /// Sparse LU with partial pivoting (faer, multithreaded). Works on any
    /// square matrix, and is what a solve does unless told otherwise.
    #[default]
    Lu,
    /// Sparse Cholesky (`L·Lᵀ`). Half the factors of an LU, no pivot search,
    /// AMD ordering instead of COLAMD — but only a **symmetric positive
    /// definite** matrix admits one, and asking for it is asserting that.
    ///
    /// A non-positive pivot comes back as an error naming where it refused: a
    /// Lagrange saddle-point is symmetric but indefinite and will always refuse,
    /// whereas the system an [elimination](crate::ops::solver::eliminate)
    /// reduces it to carries no multiplier DOF and usually accepts.
    Cholesky,
}

/// How much a solve says about what it did.
///
/// A solve is an **action**, not an object: there is nothing to inspect with
/// [`Dump`](crate::dump::Dump) afterwards. What is useful is for the solve to
/// account for itself as it goes — which method it settled on, and what it
/// cost. Printed to stdout, as `Dump::dump` already does; this crate carries no
/// logging dependency.
///
/// ```
/// # use pyrucast::ops::solver::lu::Verbosity;
/// # use pyrucast::named::Named;
/// assert_eq!(Verbosity::default(), Verbosity::Silent);
/// assert_eq!(Verbosity::parse("brief")?, Verbosity::Brief);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verbosity {
    /// Say nothing.
    #[default]
    Silent,
    /// One line per solve: size, non-zero count, and the method used.
    Brief,
    /// The same, plus what each phase cost.
    ///
    /// Not the size of the factors: faer's `SymbolicLlt` keeps its inner
    /// symbolic private, so the `len_val` that would give it exactly is out of
    /// reach without dropping to the low-level API.
    Detailed,
}

impl crate::named::Named for Verbosity {
    const LABEL: &'static str = "verbosity";
    const VALUES: &'static [Self] = &[Verbosity::Silent, Verbosity::Brief, Verbosity::Detailed];

    fn name(self) -> &'static str {
        match self {
            Verbosity::Silent => "silent",
            Verbosity::Brief => "brief",
            Verbosity::Detailed => "detailed",
        }
    }
}

impl crate::named::Named for SolveMethod {
    const LABEL: &'static str = "solver method";
    const VALUES: &'static [Self] = &[SolveMethod::Lu, SolveMethod::Cholesky];

    fn name(self) -> &'static str {
        match self {
            SolveMethod::Lu => "lu",
            SolveMethod::Cholesky => "cholesky",
        }
    }
}

/// Options for [`solve_with_options`]. Defaults: sparse LU, silent, with the
/// reusable factorization cache enabled.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::{SolveMethod, SolveOptions};
/// # use pyrucast::ops::model;
/// // By default, sparse LU with the factorization cache on: the first
/// // `solve` factorizes, the next ones only run the substitutions.
/// let d = SolveOptions::default();
/// assert!(d.cache);
/// assert!(k.cached_factorization::<solver::lu::Factorization>().is_none());
/// solver::lu::solve_with_options(&k, &charge, &d)?;
/// assert!(k.cached_factorization::<solver::lu::Factorization>().is_some());
///
/// // `cache: false` factorizes afresh and **does not touch** the cache.
/// let sans = SolveOptions { method: SolveMethod::Lu, cache: false, ..Default::default() };
/// solver::lu::solve_with_options(&k, &charge, &sans)?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    /// Direct method to use.
    pub method: SolveMethod,
    /// Reuse / populate the matrix's cached factorization. When `true` (default)
    /// the first solve factorizes and caches; later solves on the **same**
    /// matrix reuse the factors (descent/back-substitution only). When `false`,
    /// factorize fresh and do not touch the cache.
    pub cache: bool,
    /// How much the solve says about what it did.
    pub verbosity: Verbosity,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            method: SolveMethod::Lu,
            cache: true,
            verbosity: Verbosity::Silent,
        }
    }
}

/// A reusable sparse factorization of a [`Matrix`], plus the DOF layout
/// needed to map a right-hand side in and a solution out. Cached transparently
/// inside the `Matrix` (see [`SolveOptions::cache`]); derived, non-serialized
/// state — never persisted.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::Factorization;
/// # use std::sync::Arc;
/// # use pyrucast::ops::model;
/// // A **derived** state, never persisted: it carries the LU and the DOF
/// // layout that makes it usable. It is not called directly — `solve` drops
/// // it in the matrix's cache and picks it up again at the next solve.
/// k.store_factorization(Arc::new(Factorization::new(&k, &Default::default())?));
/// assert!(k.cached_factorization::<Factorization>().is_some());
/// let u = solver::lu::solve(&k, &charge)?;
/// assert!(u.node_count()? > 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct Factorization {
    factored: Factored,
    /// The row numbering the factorization is valid for, in packed form: a
    /// materialised `(NodeId, String)` list would hold one heap allocation per
    /// degree of freedom for the whole life of the cached factorization.
    /// Shared with the matrix it was computed from, not copied: the numbering
    /// outlives neither of them separately, and a factorization cached for the
    /// life of a solve loop has no reason to hold its own eight bytes per DOF.
    vars: Arc<Vec<String>>,
    row_keys: Arc<Vec<DofKey>>,
}

impl Factorization {
    /// Factorize `matrix` (must be square and finalized) with the method
    /// `options` names.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::RelationSense;
    /// # use pyrucast::ops::{element_field, matrix, mesh, solver};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let n: Vec<Node> = (0..3)
    /// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
    /// #     .collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let cible = model::heat_conduction(&fes).unwrap();
    /// # let modele = model::heat_conduction(&fes).unwrap()
    /// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
    /// #                              RelationSense::Equality).unwrap()).unwrap();
    /// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
    /// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
    /// #                                      vec!["imposed_T".into()]).unwrap();
    /// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
    /// // The matrix must be **square and finalized**.
    /// assert!(solver::lu::Factorization::new(&k, &Default::default()).is_ok());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(matrix: &Matrix, options: &SolveOptions) -> Result<Self> {
        let vars = matrix.dof_vars();
        let row_keys = matrix.row_dof_keys()?;
        let n_cols = matrix.col_dof_keys()?.len();
        if row_keys.len() != n_cols {
            return Err(PyrucastError::Message(format!(
                "solve: matrix must be square; got {}×{}",
                row_keys.len(),
                n_cols
            )));
        }
        // Borrowed straight from the assembled CSR: the matrix is handed to
        // faer without ever being materialised in another form — and, when the
        // matrix declares itself symmetric, without even being turned around.
        let (offsets, cols, vals) = matrix.csr_arrays()?;
        let factored = factorize_csr(
            row_keys.len(),
            offsets,
            cols,
            vals,
            matrix.symmetric(),
            options,
        )?;
        Ok(Self {
            factored,
            vars,
            row_keys,
        })
    }

    /// Solve `A·x = b` for one right-hand side (descent/back-substitution only).
    fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        self.factored.solve_vec(b)
    }

    /// Which factorization this one turned out to be — `"cholesky"` or `"lu"`.
    ///
    /// The one way to observe the attempt's outcome without reading printed
    /// text; the verbose modes report it, tests assert on it.
    #[cfg(test)]
    pub(crate) fn method(&self) -> &'static str {
        self.factored.method()
    }
}

/// Solve `matrix · x = rhs` using the default options (sparse LU, cached).
///
/// `matrix` must be square (`n_rows == n_cols ≥ 1`). The `rhs` `NodeField` is
/// read at every row DOF of the matrix, through the aggregate (first zone
/// defining the pair wins); missing entries (no zone defines that
/// `(node, component)`) default to `0.0`.
///
/// The returned `NodeField` has one zone per distinct **column support** of the
/// matrix's blocks, each zone sharing that block's own POI1 support handle and
/// carrying its primal variables (see [`Matrix::field_from_col_values`]) — no
/// support submesh is rebuilt, and the output aligns by
/// [`same_support`](crate::containers::field::SubField::same_support) with
/// any other field on those supports. On a Lagrange-constrained model this
/// includes a zone for the multipliers (the reactions).
///
/// Uninterruptible convenience form; see [`solve_cancellable`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// // A bar clamped on the left at 100 °C, with no other load: the whole bar
/// // is at 100 °C. The solution also carries the **reaction** at the
/// // multiplier node — hence two zones.
/// let u = solver::lu::solve(&k, &charge)?;
/// assert!((u.get(0)?.read().value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
/// // One zone per support of the model: the bar, the imposed nodes, and the
/// // multiplier carrying the **reaction**.
/// assert_eq!(u.len(), 3);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve(matrix: &Matrix, rhs: &NodeField) -> Result<NodeField> {
    solve_inner(matrix, rhs, &SolveOptions::default(), &NoCancel)
}

/// Like [`solve`] but with explicit [`SolveOptions`] (method / factorization
/// cache).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::{SolveMethod, SolveOptions};
/// # use pyrucast::ops::model;
/// // La forme explicite : méthode et cache de factorisation.
/// let o = SolveOptions { method: SolveMethod::Lu, cache: true, ..Default::default() };
/// let u = solver::lu::solve_with_options(&k, &charge, &o)?;
/// assert!((u.get(0)?.read().value(n[0].id(), "T")? - 100.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_with_options(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
) -> Result<NodeField> {
    solve_inner(matrix, rhs, options, &NoCancel)
}

/// Like [`solve`], but polls `cancel` at each phase boundary so the call can be
/// stopped early (returning [`PyrucastError::Interrupted`]).
///
/// **Granularity.** The sparse factorization (faer's `sp_lu` / `solve_in_place`)
/// is a single library call with no cooperative checkpoint, so it is **not**
/// interrupted mid-way: `cancel` is polled *around* the heavy steps (vector
/// assembly, before factorization, result write-back). A `Ctrl+C` therefore
/// lands at the next phase boundary, not inside the factorization itself. When
/// the factorization is already cached, only the (cheap) substitution runs.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use std::sync::atomic::{AtomicBool, Ordering};
/// # use pyrucast::ops::model;
/// // The token is polled **around** the heavy stages: the sparse
/// // factorization is a library call with no cooperative stopping point.
/// let stop = AtomicBool::new(false);
/// assert!(solver::lu::solve_cancellable(&k, &charge, &stop).is_ok());
/// // Token armed in advance: the stop lands at the first phase boundary.
/// stop.store(true, Ordering::Relaxed);
/// assert!(solver::lu::solve_cancellable(&k, &charge, &stop).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable(
    matrix: &Matrix,
    rhs: &NodeField,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(matrix, rhs, &SolveOptions::default(), cancel)
}

/// [`solve_cancellable`] with explicit [`SolveOptions`] — the full form the
/// Python binding routes to.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let cible = model::heat_conduction(&fes).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet(&cible, "T", &impose, &mult,
/// #                              RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::SolveOptions;
/// # use std::sync::atomic::AtomicBool;
/// # use pyrucast::ops::model;
/// // The full form, the one the Python binding routes to.
/// let stop = AtomicBool::new(false);
/// let u = solver::lu::solve_cancellable_with_options(
///     &k, &charge, &SolveOptions::default(), &stop)?;
/// assert!(u.node_count()? > 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable_with_options(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(matrix, rhs, options, cancel)
}

fn solve_inner(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    cancel.check()?;

    // ── Step 1 — obtain the factorization (cached or fresh) ────────────
    let fact: Arc<Factorization> = if options.cache {
        match matrix.cached_factorization::<Factorization>() {
            Some(f) => f,
            None => {
                let f = Arc::new(Factorization::new(matrix, options)?);
                matrix.store_factorization(f.clone());
                f
            }
        }
    } else {
        Arc::new(Factorization::new(matrix, options)?)
    };
    cancel.check()?;

    // ── Step 2 — build the b vector at the row DOFs ────────────────────
    // "No zone defines this DOF" means "no imposed value here" — zero.
    let b = rhs.gather_keys(&fact.row_keys, &fact.vars)?;

    // ── Step 3 — substitution ──────────────────────────────────────────
    let x = fact.solve_vec(&b);
    // A singular matrix factorizes with a zero pivot; the back-substitution
    // then divides by it, yielding non-finite entries. Flag it like the old
    // dense solver did, instead of returning a garbage field.
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PyrucastError::Message(
            "solve: LU failed (matrix is singular)".into(),
        ));
    }
    cancel.check()?;

    // ── Step 4 — wrap the solution into a NodeField on the blocks' supports ──
    // One zone per distinct column support, sharing the block's own POI1 handle
    // (no submesh rebuilt); `x` is in `fact.col_dofs` order, which is the
    // assembled column order (`Factorization::new` reads `matrix.col_dofs()`).
    matrix.field_from_col_values(&x)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::element_field::SubElementField;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::matrix::Symmetry;
    use crate::containers::mesh::Mesh;
    use crate::containers::mesh::SubMesh;
    use crate::containers::model::{Model, SubModel};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// 1-D Poisson `-u'' = 0` on `[0, 1]` with `u(0) = 0` and `u(1) = 1`,
    /// discretized with `n` SEG2 elements. The analytical solution is
    /// `u(x) = x`. Lagrange multipliers at the boundary represent the
    /// boundary heat flux: `+1` at x=0, `-1` at x=1 (outward normal
    /// conventions).
    #[test]
    fn poisson_1d_dirichlet_at_both_ends_recovers_linear_solution() {
        let n_elems = 4;
        let h = 1.0 / n_elems as f64;
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=n_elems)
            .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]).unwrap())
            .collect();

        // Mesh.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        for i in 0..n_elems {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();

        // k = 1 uniform.
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = crate::containers::element_field::ElementField::empty();
        materials.add_sub(Handle::new(mat)).unwrap();

        // Model.
        let mut model = Model::empty();
        model
            .add_sub(Handle::new(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let imposed_left =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&nodes[0])).unwrap());
        let imposed_right = Mesh::from_submesh(
            SubMesh::poi1_from_nodes(std::slice::from_ref(&nodes[n_elems])).unwrap(),
        );
        let mult_mesh_left = crate::ops::mesh::barycenter(&imposed_left).unwrap();
        let mult_mesh_right = crate::ops::mesh::barycenter(&imposed_right).unwrap();
        let left_dir = SubModel::dirichlet(
            &model,
            "T",
            &imposed_left,
            &mult_mesh_left,
            Default::default(),
        )
        .unwrap();
        let right_dir = SubModel::dirichlet(
            &model,
            "T",
            &imposed_right,
            &mult_mesh_right,
            Default::default(),
        )
        .unwrap();
        let mult_left = left_dir.multiplier_nodes()[0];
        let mult_right = right_dir.multiplier_nodes()[0];
        model.add_sub(Handle::new(left_dir)).unwrap();
        model.add_sub(Handle::new(right_dir)).unwrap();

        // Build rhs: T_left = 0 at mult_left, T_right = 1 at mult_right
        // (imposed value goes to the "imposed_T" slot).
        let mut rhs_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        rhs_sm.add_cell(&[mult_left]).unwrap();
        rhs_sm.add_cell(&[mult_right]).unwrap();
        let rhs_sm_h = Handle::new(rhs_sm);
        let mut rhs = SubNodeField::from_poi1(&rhs_sm_h, vec!["imposed_T".into()]).unwrap();
        rhs.set_value(mult_left, "imposed_T", 0.0).unwrap();
        rhs.set_value(mult_right, "imposed_T", 1.0).unwrap();
        let rhs = NodeField::from_sub(rhs);

        // Assemble + solve.
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        let solution = solve(&k, &rhs).unwrap();

        // Verify T at every node equals its physical coordinate.
        let tol = 1e-10;
        for (i, node) in nodes.iter().enumerate() {
            let expected = i as f64 * h;
            let got = solution.value(node.id(), "T").unwrap();
            assert!(
                (got - expected).abs() < tol,
                "T at node {i}: got {got}, expected {expected}"
            );
        }
        // Verify the boundary fluxes (Lagrange multipliers).
        let lambda_left = solution.value(mult_left, "lambda_T").unwrap();
        let lambda_right = solution.value(mult_right, "lambda_T").unwrap();
        assert!(
            (lambda_left - 1.0).abs() < tol,
            "lambda at left: got {lambda_left}, expected 1.0"
        );
        assert!(
            (lambda_right + 1.0).abs() < tol,
            "lambda at right: got {lambda_right}, expected -1.0"
        );
    }

    /// Singular matrix (Neumann everywhere) should produce a clean error.
    #[test]
    fn singular_matrix_yields_error() {
        // 2-node SEG2 with no Dirichlet → K is the discrete Laplacian
        // [[1, -1], [-1, 1]], singular (kernel = constants).
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = crate::containers::element_field::ElementField::empty();
        materials.add_sub(Handle::new(mat)).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(Handle::new(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();

        // Build a tiny non-empty rhs on a real node so we can find the
        // Coords of the result SubNodeField.
        let mut rhs_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        rhs_sm.add_cell(&[a.id()]).unwrap();
        let rhs_sm_h = Handle::new(rhs_sm);
        let rhs =
            NodeField::from_sub(SubNodeField::from_poi1(&rhs_sm_h, vec!["q".into()]).unwrap());
        // K is singular ⇒ solve must err.
        assert!(solve(&k, &rhs).is_err());
    }

    /// The solution's zones live on the matrix blocks' **own** column supports
    /// (`same_object`), so consecutive solves — and any block-shaped field —
    /// align by support instead of falling into merge passthrough. Two solves
    /// on the same matrix share the very same support handles.
    #[test]
    fn solution_zones_share_the_blocks_column_supports() {
        // 1-D Poisson with one Dirichlet end (multi-block: K + C/Cᵀ).
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..3)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        for i in 0..2 {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = crate::containers::element_field::ElementField::empty();
        materials.add_sub(Handle::new(mat)).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(Handle::new(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        // Ground both ends so K-with-constraints is nonsingular.
        for end in [0usize, 2] {
            let imposed = Mesh::from_submesh(
                SubMesh::poi1_from_nodes(std::slice::from_ref(&nodes[end])).unwrap(),
            );
            let mult = crate::ops::mesh::barycenter(&imposed).unwrap();
            model
                .add_sub(Handle::new(
                    SubModel::dirichlet(&model, "T", &imposed, &mult, Default::default()).unwrap(),
                ))
                .unwrap();
        }
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        let rhs_sm = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nodes[0].id()]).unwrap();
            Handle::new(sm)
        };
        let rhs = NodeField::from_sub(SubNodeField::from_poi1(&rhs_sm, vec!["q".into()]).unwrap());

        let sol_a = solve(&k, &rhs).unwrap();
        let sol_b = solve(&k, &rhs).unwrap();
        sol_a.check().unwrap();

        // Each zone's support is one of the blocks' column supports.
        let block_col_supports: Vec<_> = k.iter().map(|h| h.read().col_support().clone()).collect();
        assert!(sol_a.len() > 1, "multi-block model ⇒ multi-zone solution");
        for zh in &sol_a {
            let zone_support = zh.read().support();
            assert!(
                block_col_supports
                    .iter()
                    .any(|bs| bs.same_object(&zone_support)),
                "zone support must be one of the blocks' col supports"
            );
        }
        // Consecutive solves share the same support handles (same_object),
        // so their arithmetic aligns by support.
        for (za, zb) in sol_a.iter().zip(sol_b.iter()) {
            let sa = za.read().support();
            let sb = zb.read().support();
            assert!(sa.same_object(&sb));
        }
    }

    #[test]
    fn rectangular_matrix_yields_error() {
        use crate::containers::matrix::{DofOrdering, SubMatrix};
        // 2-row support, 1-col support → 2×1 rectangular block.
        let coords = Handle::new(Coords::new(1).unwrap());
        let r0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let r1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c0 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut row_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        row_sm.add_cell(&[r0.id()]).unwrap();
        row_sm.add_cell(&[r1.id()]).unwrap();
        let mut col_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        col_sm.add_cell(&[c0.id()]).unwrap();
        let mut block = SubMatrix::new(
            Handle::new(row_sm),
            Handle::new(col_sm),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        block.add_entry(r0.id(), "q", c0.id(), "T", 1.0).unwrap();
        block.add_entry(r1.id(), "q", c0.id(), "T", 1.0).unwrap();
        // 2 rows × 1 col — rectangular.
        let mut m = crate::containers::matrix::Matrix::empty();
        m.add_sub(Handle::new(block)).unwrap();
        m.finalize().unwrap();

        // Build a minimal rhs on the same coords.
        let mut rhs_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        rhs_sm.add_cell(&[r0.id()]).unwrap();
        let rhs = NodeField::from_sub(
            SubNodeField::from_poi1(&Handle::new(rhs_sm), vec!["q".into()]).unwrap(),
        );
        assert!(solve(&m, &rhs).is_err());
    }

    /// A solvable 1×1 system `2·T = b` and a rhs carrying `b` at the row DOF.
    fn tiny_system() -> (
        crate::containers::matrix::Matrix,
        NodeField,
        crate::atoms::NodeId,
    ) {
        use crate::containers::matrix::{DofOrdering, SubMatrix};
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            Handle::new(sm)
        };
        let mut block = SubMatrix::new(
            sm.clone(),
            sm.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        block.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
        let mut m = crate::containers::matrix::Matrix::empty();
        m.add_sub(Handle::new(block)).unwrap();
        m.finalize().unwrap();

        let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
        rhs.set_value(a.id(), "q", 6.0).unwrap();
        (m, NodeField::from_sub(rhs), a.id())
    }

    /// A symmetric **positive definite** matrix: a 5-point Laplacian with a
    /// strictly dominant diagonal, the same shape `benches/parallel.rs`
    /// factorizes. Cholesky must take it — and the solution must be right.
    fn spd_grid(n: usize) -> (crate::containers::matrix::Matrix, NodeField) {
        use crate::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
        let coords = Handle::new(Coords::new(2).unwrap());
        let ids: Vec<crate::atoms::NodeId> = (0..=n)
            .flat_map(|j| (0..=n).map(move |i| (i, j)))
            .map(|(i, j)| {
                Node::create_in(coords.clone(), &[i as f64, j as f64])
                    .unwrap()
                    .id()
            })
            .collect();
        let at = |i: usize, j: usize| ids[j * (n + 1) + i];
        let sm = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for &id in &ids {
                sm.add_cell(&[id]).unwrap();
            }
            Handle::new(sm)
        };
        let mut block = SubMatrix::new(
            sm.clone(),
            sm.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        for j in 0..=n {
            for i in 0..=n {
                let c = at(i, j);
                let mut neighbours = Vec::with_capacity(4);
                if i > 0 {
                    neighbours.push(at(i - 1, j));
                }
                if i < n {
                    neighbours.push(at(i + 1, j));
                }
                if j > 0 {
                    neighbours.push(at(i, j - 1));
                }
                if j < n {
                    neighbours.push(at(i, j + 1));
                }
                for &m in &neighbours {
                    block.add_entry(c, "q", m, "T", -1.0).unwrap();
                }
                // diag = #neighbours + 1 ⇒ strictly dominant ⇒ positive definite.
                block
                    .add_entry(c, "q", c, "T", neighbours.len() as f64 + 1.0)
                    .unwrap();
            }
        }
        let mut m = crate::containers::matrix::Matrix::empty();
        m.add_sub(Handle::new(block)).unwrap();
        m.finalize().unwrap();

        let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
        for &id in &ids {
            rhs.set_value(id, "q", 1.0).unwrap();
        }
        (m, NodeField::from_sub(rhs))
    }

    /// Asking for a Cholesky on a matrix that admits one.
    #[test]
    fn an_spd_system_is_factorized_by_cholesky() {
        let (m, _) = spd_grid(4);
        assert!(m.symmetric(), "the fixture must declare itself symmetric");
        let f = Factorization::new(&m, &cholesky()).unwrap();
        assert_eq!(f.method(), "cholesky");
    }

    fn cholesky() -> SolveOptions {
        SolveOptions {
            method: SolveMethod::Cholesky,
            ..Default::default()
        }
    }

    /// A Cholesky reads one triangle, so on a non-symmetric matrix it does not
    /// fail — it quietly uses the half it was handed. faer cannot catch that,
    /// so the refusal comes from us, before anything is factorized.
    #[test]
    fn cholesky_refuses_a_matrix_that_declares_no_symmetry() {
        let (m, _, _) = tiny_system();
        assert!(!m.symmetric());
        let Err(err) = Factorization::new(&m, &cholesky()) else {
            panic!("a matrix declaring no symmetry must be refused");
        };
        assert!(err.to_string().contains("symmetric"), "{err}");
    }

    /// A conduction bar clamped at one end by a **Lagrange multiplier** — the
    /// shape every clamped beam of this repo assembles, and the reason a
    /// Cholesky cannot be the default: symmetric, and indefinite.
    fn clamped_bar() -> crate::containers::matrix::Matrix {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=3)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        for i in 0..3 {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();

        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = crate::containers::element_field::ElementField::empty();
        materials.add_sub(Handle::new(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&nodes[0])).unwrap());
        let mult = crate::ops::mesh::barycenter(&imposed).unwrap();
        let dir = SubModel::dirichlet(&model, "T", &imposed, &mult, Default::default()).unwrap();
        model.add_sub(Handle::new(dir)).unwrap();

        crate::ops::matrix::stiffness(&model, &materials).unwrap()
    }

    /// A Lagrange saddle-point — what every clamped beam in this repo assembles
    /// — is symmetric but **indefinite**: the zero block at its bottom right
    /// forbids positive-definiteness whatever the physics. Cholesky refuses it,
    /// and says where.
    #[test]
    fn a_lagrange_saddle_point_is_refused_by_cholesky() {
        let m = clamped_bar();
        assert!(
            m.symmetric(),
            "a stiffness with its Dirichlet pair is symmetric"
        );
        // The LU takes it without blinking.
        let f = Factorization::new(&m, &SolveOptions::default()).unwrap();
        assert_eq!(f.method(), "lu");

        let Err(err) = Factorization::new(&m, &cholesky()) else {
            panic!("an indefinite saddle-point must be refused by Cholesky");
        };
        let err = err.to_string();
        assert!(err.contains("positive definite"), "{err}");
        assert!(err.contains("pivot"), "the message must say where: {err}");
    }

    /// The two methods answer the same question. On a matrix that admits both,
    /// they must agree — to rounding, being two different factorizations.
    #[test]
    fn both_methods_agree_on_a_matrix_that_admits_both() {
        let (m, rhs) = spd_grid(3);
        let node = m.row_mesh().unwrap().node(0, 0, 0).unwrap().id();
        let by_lu = solve_with_options(&m, &rhs, &SolveOptions::default()).unwrap();
        let by_llt = solve_with_options(&m, &rhs, &cholesky()).unwrap();
        let (a, b) = (
            by_lu.value(node, "T").unwrap(),
            by_llt.value(node, "T").unwrap(),
        );
        assert!((a - b).abs() <= 1e-10 * (1.0 + a.abs()), "{a} vs {b}");
    }

    #[test]
    fn solve_cancellable_stops_on_preset_flag() {
        use std::sync::atomic::AtomicBool;
        let (m, rhs, _a) = tiny_system();
        let flag = AtomicBool::new(true);
        let err = solve_cancellable(&m, &rhs, &flag).unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    fn solve_cancellable_completes_when_not_cancelled() {
        use std::sync::atomic::AtomicBool;
        let (m, rhs, a) = tiny_system();
        let flag = AtomicBool::new(false);
        let sol = solve_cancellable(&m, &rhs, &flag).unwrap();
        // 2·T = 6 ⇒ T = 3.
        assert!((sol.value(a, "T").unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn empty_matrix_yields_error() {
        let m = crate::containers::matrix::Matrix::empty();
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        let rhs = NodeField::from_sub(
            SubNodeField::from_poi1(&Handle::new(sm), vec!["q".into()]).unwrap(),
        );
        assert!(solve(&m, &rhs).is_err());
    }

    #[test]
    fn factorization_cache_reused_then_invalidated_on_change() {
        use crate::containers::matrix::{DofOrdering, SubMatrix};
        let (mut m, rhs, a) = tiny_system();

        // Nothing cached before the first solve.
        assert!(m.cached_factorization::<Factorization>().is_none());

        // First solve factorizes + caches; 2·T = 6 ⇒ T = 3.
        let s1 = solve(&m, &rhs).unwrap();
        assert!((s1.value(a, "T").unwrap() - 3.0).abs() < 1e-12);
        assert!(m.cached_factorization::<Factorization>().is_some());

        // Second solve reuses the cached factorization: identical result.
        let s2 = solve(&m, &rhs).unwrap();
        assert_eq!(s1.value(a, "T").unwrap(), s2.value(a, "T").unwrap());

        // Mutating the matrix invalidates the cache.
        let coords = Handle::new(Coords::new(1).unwrap());
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[b.id()]).unwrap();
            Handle::new(sm)
        };
        let mut block = SubMatrix::new(
            sm.clone(),
            sm.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        block.add_entry(b.id(), "q", b.id(), "T", 4.0).unwrap();
        m.add_sub(Handle::new(block)).unwrap();
        assert!(m.cached_factorization::<Factorization>().is_none());
    }

    #[test]
    fn solve_with_cache_disabled_does_not_populate() {
        let (m, rhs, a) = tiny_system();
        let opts = SolveOptions {
            method: SolveMethod::Lu,
            cache: false,
            ..Default::default()
        };
        let s = solve_with_options(&m, &rhs, &opts).unwrap();
        assert!((s.value(a, "T").unwrap() - 3.0).abs() < 1e-12);
        assert!(m.cached_factorization::<Factorization>().is_none());
    }
}
