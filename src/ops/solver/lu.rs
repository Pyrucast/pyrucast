//! Dense linear solver: `A · x = b`.
//!
//! This module bridges the abstract [`Matrix`] / [`NodeField`] objects
//! and the dense linear algebra of [`nalgebra`]. It exposes a single
//! free function — [`solve`] — that:
//!
//! 1. converts the assembled `Matrix` to a dense [`nalgebra::DMatrix`];
//! 2. reads a right-hand-side vector out of the `NodeField`, one entry
//!    per **row DOF** of the matrix (missing entries default to `0.0`);
//! 3. runs a standard LU factorization (`nalgebra::DMatrix::lu`);
//! 4. wraps the solution back into a fresh `NodeField` indexed by the
//!    **column DOFs** of the matrix.
//!
//! This is the minimal harness needed to validate the assembly of
//! [`crate::containers::model::Model`] end-to-end (Poisson 1-D, etc.). A richer
//! `LinearSolver` trait (iterative methods, sparse direct factorization,
//! preconditioners) belongs to Phase 3 of the roadmap and will sit on
//! top of `nalgebra-sparse` (CSR / CSC views are already available on
//! `Matrix`).
//!
//! # Example
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::containers::mesh::configuration::Configuration;
//! use pyrucast::containers::element_field::SubElementField;
//! use pyrucast::containers::mesh::element_type::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::model::{Model, SubModel};
//! use pyrucast::containers::mesh::node::Node;
//! use pyrucast::containers::node_field::NodeField;
//! use pyrucast::ops::solver::lu::solve;
//! use pyrucast::store::insert;
//!
//! // 1-D Poisson on [0, 1] with one SEG2 element, k = 1.
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.subspace(0).unwrap();
//! let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
//! mat.set_uniform("k", 1.0).unwrap();
//!
//! let mut model = Model::empty();
//! model
//!     .add_sub(insert(SubModel::heat_conduction(sub, insert(mat))))
//!     .unwrap();
//! let dir_a = SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![a.id()]).unwrap();
//! let dir_b = SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![b.id()]).unwrap();
//! let mult_a = dir_a.multiplier_nodes()[0];
//! let mult_b = dir_b.multiplier_nodes()[0];
//! model.add_sub(insert(dir_a)).unwrap();
//! model.add_sub(insert(dir_b)).unwrap();
//!
//! // Load: imposed values T_a = 0, T_b = 1 at the multiplier nodes.
//! let mut load_sm = SubMesh::new(cfg.clone(), ElementType::POI1);
//! load_sm.add_cell(&[mult_a]).unwrap();
//! load_sm.add_cell(&[mult_b]).unwrap();
//! let load_sm_h = insert(load_sm);
//! let mut rhs = NodeField::from_poi1(&load_sm_h, vec!["T".into()]).unwrap();
//! rhs.set_value(mult_a, "T", 0.0).unwrap();
//! rhs.set_value(mult_b, "T", 1.0).unwrap();
//!
//! let k = model.stiffness().unwrap();
//! let solution = solve(&k, &rhs).unwrap();
//! // Solution: T(a) = 0, T(b) = 1, λ_a = +1, λ_b = -1 (boundary fluxes).
//! assert!((solution.value(a.id(), "T").unwrap() - 0.0).abs() < 1e-12);
//! assert!((solution.value(b.id(), "T").unwrap() - 1.0).abs() < 1e-12);
//! ```

use crate::containers::mesh::configuration::NodeId;
use crate::containers::mesh::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::containers::matrix::Matrix;
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::NodeField;
use crate::store::insert;
use nalgebra::DVector;

/// Solve `matrix · x = rhs` using dense LU factorization.
///
/// `matrix` must be square (`n_rows == n_cols ≥ 1`). The `rhs`
/// `NodeField` is read at every row DOF of the matrix; missing entries
/// (component absent in `rhs.components()`, or node not in
/// `rhs`'s support) default to `0.0`.
///
/// The returned `NodeField` lives on the column-DOF nodes of the
/// matrix (a POI1 submesh built on the fly) and exposes one component
/// per distinct column field name.
pub fn solve(matrix: &Matrix, rhs: &NodeField) -> Result<NodeField> {
    if matrix.n_rows() != matrix.n_cols() {
        return Err(PyrucastError::Message(format!(
            "solve: matrix must be square; got {}×{}",
            matrix.n_rows(),
            matrix.n_cols()
        )));
    }
    let n = matrix.n_rows();
    if n == 0 {
        return Err(PyrucastError::Message("solve: matrix is empty".into()));
    }

    // ── Step 1 — build the b vector ────────────────────────────────────
    let mut b = DVector::<f64>::zeros(n);
    let rhs_components: Vec<String> = rhs.components().to_vec();
    for (i, dof) in matrix.row_dofs().iter().enumerate() {
        let field_name = matrix.field_name(dof.field_idx);
        if rhs_components.iter().any(|c| c == field_name) {
            // value() errors if `dof.node_id` isn't in the rhs's
            // support — we treat that as "no imposed value here", i.e.
            // zero.
            if let Ok(v) = rhs.value(dof.node_id, field_name) {
                b[i] = v;
            }
        }
    }

    // ── Step 2 — LU factorization + solve ──────────────────────────────
    let a = matrix.to_dmatrix();
    let lu = a.lu();
    let x = lu.solve(&b).ok_or_else(|| {
        PyrucastError::Message("solve: LU failed (matrix is singular)".into())
    })?;

    // ── Step 3 — wrap the solution into a fresh NodeField ──────────────
    let cfg = rhs.configuration();

    // Unique col nodes in first-seen order.
    let mut unique_nodes: Vec<NodeId> = Vec::new();
    for dof in matrix.col_dofs() {
        if !unique_nodes.contains(&dof.node_id) {
            unique_nodes.push(dof.node_id);
        }
    }
    // Unique col field names in first-seen order.
    let mut unique_components: Vec<String> = Vec::new();
    for dof in matrix.col_dofs() {
        let name = matrix.field_name(dof.field_idx).to_string();
        if !unique_components.contains(&name) {
            unique_components.push(name);
        }
    }

    // POI1 submesh over the col nodes — provides the support of the
    // resulting NodeField. The submesh and the field both end up in the
    // store; they cascade-decref the nodes correctly when dropped.
    let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    for nid in &unique_nodes {
        sm.add_cell(&[*nid])?;
    }
    let sm_h = insert(sm);

    let mut result = NodeField::from_poi1(&sm_h, unique_components)?;
    for (i, dof) in matrix.col_dofs().iter().enumerate() {
        let field_name = matrix.field_name(dof.field_idx).to_string();
        result.set_value(dof.node_id, &field_name, x[i])?;
    }
    Ok(result)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::configuration::Configuration;
    use crate::containers::element_field::SubElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::containers::model::{Model, SubModel};
    use crate::containers::mesh::node::Node;
    use crate::store::{insert};

    /// 1-D Poisson `-u'' = 0` on `[0, 1]` with `u(0) = 0` and `u(1) = 1`,
    /// discretized with `n` SEG2 elements. The analytical solution is
    /// `u(x) = x`. Lagrange multipliers at the boundary represent the
    /// boundary heat flux: `+1` at x=0, `-1` at x=1 (outward normal
    /// conventions).
    #[test]
    fn poisson_1d_dirichlet_at_both_ends_recovers_linear_solution() {
        let n_elems = 4;
        let h = 1.0 / n_elems as f64;
        let cfg = insert(Configuration::new(1).unwrap());
        let nodes: Vec<Node> = (0..=n_elems)
            .map(|i| Node::create_in(cfg.clone(), &[i as f64 * h]).unwrap())
            .collect();

        // Mesh.
        let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
        for i in 0..n_elems {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();

        // k = 1 uniform.
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mat_h = insert(mat);

        // Model.
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub, mat_h)))
            .unwrap();
        let left_dir =
            SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![nodes[0].id()])
                .unwrap();
        let right_dir = SubModel::dirichlet(
            cfg.clone(),
            "T".into(),
            "q".into(),
            vec![nodes[n_elems].id()],
        )
        .unwrap();
        let mult_left = left_dir.multiplier_nodes()[0];
        let mult_right = right_dir.multiplier_nodes()[0];
        model.add_sub(insert(left_dir)).unwrap();
        model.add_sub(insert(right_dir)).unwrap();

        // Build rhs: T_left = 0 at mult_left, T_right = 1 at mult_right.
        let mut rhs_sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        rhs_sm.add_cell(&[mult_left]).unwrap();
        rhs_sm.add_cell(&[mult_right]).unwrap();
        let rhs_sm_h = insert(rhs_sm);
        let mut rhs = NodeField::from_poi1(&rhs_sm_h, vec!["T".into()]).unwrap();
        rhs.set_value(mult_left, "T", 0.0).unwrap();
        rhs.set_value(mult_right, "T", 1.0).unwrap();

        // Assemble + solve.
        let k = model.stiffness().unwrap();
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
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mat_h = insert(mat);
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub, mat_h)))
            .unwrap();
        let k = model.stiffness().unwrap();

        // Build a tiny non-empty rhs on a real node so we can find the
        // Configuration of the result NodeField.
        let mut rhs_sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        rhs_sm.add_cell(&[a.id()]).unwrap();
        let rhs_sm_h = insert(rhs_sm);
        let rhs = NodeField::from_poi1(&rhs_sm_h, vec!["q".into()]).unwrap();
        // K is singular ⇒ solve must err.
        assert!(solve(&k, &rhs).is_err());
    }

    #[test]
    fn rectangular_matrix_yields_error() {
        let mut m = Matrix::new(false);
        m.add_entry(NodeId(0), "q", NodeId(0), "T", 1.0);
        m.add_entry(NodeId(1), "q", NodeId(0), "T", 1.0);
        // 2 rows × 1 col — rectangular.

        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        let rhs = NodeField::from_poi1(&insert(sm), vec!["q".into()]).unwrap();
        assert!(solve(&m, &rhs).is_err());
    }

    #[test]
    fn empty_matrix_yields_error() {
        let m = Matrix::new(false);
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        let rhs = NodeField::from_poi1(&insert(sm), vec!["q".into()]).unwrap();
        assert!(solve(&m, &rhs).is_err());
    }
}
