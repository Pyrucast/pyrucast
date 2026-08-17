//! Sparse direct linear solver: `A · x = b`.
//!
//! This module bridges the abstract [`Matrix`] / [`NodeField`] objects and the
//! sparse linear algebra of [`faer`]. It exposes a single free function —
//! [`solve`] — that:
//!
//! 1. converts the assembled `Matrix` to a CSC (`nalgebra_sparse`), then to a
//!    faer `SparseColMat`;
//! 2. reads a right-hand-side vector out of the `NodeField`, one entry
//!    per **row DOF** of the matrix (zones resolved first-found; missing
//!    entries default to `0.0`);
//! 3. runs a multithreaded **sparse LU** factorization with partial pivoting;
//! 4. wraps the solution back into a fresh single-zone `NodeField`
//!    indexed by the **column DOFs** of the matrix.
//!
//! The factorization is **reusable**: it is cached inside the `Matrix`
//! ([`SolveOptions::cache`]), so a Newton loop or a multi-load-case run pays
//! for it once and only redoes descent / back-substitution afterwards —
//! *factor once, solve many*. [`SolveMethod`] is the seam through which another
//! back-end (iterative, preconditioned) could be selected later without
//! touching the call sites.
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
//! use pyrucast::store::Handle;
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
//! let dir_a = SubModel::dirichlet("T".into(), "q".into(), &imposed_a, &mult_mesh_a, None, None, Default::default()).unwrap();
//! let dir_b = SubModel::dirichlet("T".into(), "q".into(), &imposed_b, &mult_mesh_b, None, None, Default::default()).unwrap();
//! let mult_a = dir_a.multiplier_nodes().unwrap()[0];
//! let mult_b = dir_b.multiplier_nodes().unwrap()[0];
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

use crate::atoms::NodeId;
use crate::containers::matrix::Matrix;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra_sparse::CscMatrix;
use std::sync::Arc;

/// A faer sparse LU factorization over `usize` indices and `f64` values — the
/// direct back-end shared by [`Factorization`] (full saddle-point system) and
/// [`crate::ops::solver::eliminate`] (reduced condensed system).
pub(crate) type SparseLu = faer::sparse::linalg::solvers::Lu<usize, f64>;

/// Factorize a square CSC matrix with sparse LU (faer). The single place the
/// nalgebra-sparse CSC → faer `SparseColMat` → `sp_lu` conversion lives, so both
/// the Lagrange and the elimination solvers share one implementation.
pub(crate) fn factorize_csc(csc: &CscMatrix<f64>) -> Result<SparseLu> {
    let n = csc.nrows();
    if n == 0 {
        return Err(PyrucastError::Message("solve: matrix is empty".into()));
    }
    if csc.ncols() != n {
        return Err(PyrucastError::Message(format!(
            "solve: matrix must be square; got {}×{}",
            n,
            csc.ncols()
        )));
    }
    // Build a faer sparse matrix from the (duplicate-summed) CSC. Each
    // (row, col) appears once, so the triplet form is exact.
    let col_offsets = csc.col_offsets();
    let row_indices = csc.row_indices();
    let values = csc.values();
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(values.len());
    for col in 0..n {
        for k in col_offsets[col]..col_offsets[col + 1] {
            triplets.push(Triplet::new(row_indices[k], col, values[k]));
        }
    }
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|e| PyrucastError::Message(format!("solve: sparse build failed: {e:?}")))?;
    a.sp_lu()
        .map_err(|e| PyrucastError::Message(format!("solve: LU failed (singular?): {e:?}")))
}

/// Solve `A·x = b` for one right-hand side against a computed [`SparseLu`]
/// (descent / back-substitution only). Shared by both direct solvers.
pub(crate) fn lu_solve_vec(lu: &SparseLu, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = faer::Mat::<f64>::zeros(n, 1);
    for (i, &v) in b.iter().enumerate() {
        x[(i, 0)] = v;
    }
    lu.solve_in_place(&mut x);
    (0..n).map(|i| x[(i, 0)]).collect()
}

/// Direct solver method. Today only sparse LU (faer); the enum leaves room to
/// force another backend (iterative, …) in the future without changing the
/// `solve` call sites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolveMethod {
    /// Sparse LU with partial pivoting (faer, multithreaded).
    #[default]
    Lu,
}

/// Options for [`solve_with_options`]. Defaults: sparse LU with the reusable
/// factorization cache enabled.
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    /// Direct method to use.
    pub method: SolveMethod,
    /// Reuse / populate the matrix's cached factorization. When `true` (default)
    /// the first solve factorizes and caches; later solves on the **same**
    /// matrix reuse the factors (descent/back-substitution only). When `false`,
    /// factorize fresh and do not touch the cache.
    pub cache: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            method: SolveMethod::Lu,
            cache: true,
        }
    }
}

/// A reusable sparse LU factorization of a [`Matrix`], plus the DOF layout
/// needed to map a right-hand side in and a solution out. Cached transparently
/// inside the `Matrix` (see [`SolveOptions::cache`]); derived, non-serialized
/// state — never persisted.
pub struct Factorization {
    lu: SparseLu,
    row_dofs: Vec<(NodeId, String)>,
}

impl Factorization {
    /// Factorize `matrix` (must be square and finalized) with sparse LU.
    pub fn new(matrix: &Matrix) -> Result<Self> {
        let row_dofs = matrix.row_dofs()?;
        let col_dofs = matrix.col_dofs()?;
        if row_dofs.len() != col_dofs.len() {
            return Err(PyrucastError::Message(format!(
                "solve: matrix must be square; got {}×{}",
                row_dofs.len(),
                col_dofs.len()
            )));
        }
        let lu = factorize_csc(&matrix.to_csc()?)?;
        Ok(Self { lu, row_dofs })
    }

    /// Solve `A·x = b` for one right-hand side (descent/back-substitution only).
    fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        lu_solve_vec(&self.lu, b)
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
pub fn solve(matrix: &Matrix, rhs: &NodeField) -> Result<NodeField> {
    solve_inner(matrix, rhs, &SolveOptions::default(), &NoCancel)
}

/// Like [`solve`] but with explicit [`SolveOptions`] (method / factorization
/// cache).
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
pub fn solve_cancellable(
    matrix: &Matrix,
    rhs: &NodeField,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(matrix, rhs, &SolveOptions::default(), cancel)
}

/// [`solve_cancellable`] with explicit [`SolveOptions`] — the full form the
/// Python binding routes to.
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
    let SolveMethod::Lu = options.method;
    cancel.check()?;

    // ── Step 1 — obtain the factorization (cached or fresh) ────────────
    let fact: Arc<Factorization> = if options.cache {
        match matrix.cached_factorization::<Factorization>() {
            Some(f) => f,
            None => {
                let f = Arc::new(Factorization::new(matrix)?);
                matrix.store_factorization(f.clone());
                f
            }
        }
    } else {
        Arc::new(Factorization::new(matrix)?)
    };
    cancel.check()?;

    // ── Step 2 — build the b vector at the row DOFs ────────────────────
    // "No zone defines this DOF" means "no imposed value here" — zero.
    let b = rhs.gather(&fact.row_dofs)?;

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
    use crate::containers::mesh::Mesh;
    use crate::containers::mesh::SubMesh;
    use crate::containers::model::{Model, SubModel};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::store::Handle;

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
            "T".into(),
            "q".into(),
            &imposed_left,
            &mult_mesh_left,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        let right_dir = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &imposed_right,
            &mult_mesh_right,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        let mult_left = left_dir.multiplier_nodes().unwrap()[0];
        let mult_right = right_dir.multiplier_nodes().unwrap()[0];
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
                    SubModel::dirichlet(
                        "T".into(),
                        "q".into(),
                        &imposed,
                        &mult,
                        None,
                        None,
                        Default::default(),
                    )
                    .unwrap(),
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
            false,
        )
        .unwrap();
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
    fn tiny_system() -> (crate::containers::matrix::Matrix, NodeField, NodeId) {
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
            false,
        )
        .unwrap();
        block.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
        let mut m = crate::containers::matrix::Matrix::empty();
        m.add_sub(Handle::new(block)).unwrap();
        m.finalize().unwrap();

        let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
        rhs.set_value(a.id(), "q", 6.0).unwrap();
        (m, NodeField::from_sub(rhs), a.id())
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
            false,
        )
        .unwrap();
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
        };
        let s = solve_with_options(&m, &rhs, &opts).unwrap();
        assert!((s.value(a, "T").unwrap() - 3.0).abs() < 1e-12);
        assert!(m.cached_factorization::<Factorization>().is_none());
    }
}
