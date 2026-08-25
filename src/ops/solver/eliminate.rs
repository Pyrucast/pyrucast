//! Constraint imposition by **master/slave elimination** (condensation) — the
//! alternative to the Lagrange-multiplier path ([`super::lu`]).
//!
//! A constraint relation `Σₖ aₖ·u(nodeₖ,varₖ) = g` is enforced by picking one
//! term as the **slave** `s` and expressing its DOF from the others (the
//! **masters**):
//!
//! ```text
//! u_s = (g − Σ_{k≠s} aₖ·u_k) / a_s
//! ```
//!
//! Collecting every relation yields a global transformation `u = T·û + u₀`
//! (`û` = the retained/free DOFs, `T` the `n_phys × n_free` prolongation, `u₀`
//! the inhomogeneous part carrying the right-hand sides `g`). The system reduces
//! to `K̂·û = f̂` with
//!
//! ```text
//! K̂ = Tᵀ K T          f̂ = Tᵀ (f − K·u₀)
//! ```
//!
//! solved by the same sparse LU back-end as the Lagrange path (on a *smaller*,
//! definite matrix — no multiplier DOFs), then prolonged back `u = T·û + u₀`.
//! The multiplier-equivalent **reaction** is recovered in post-processing as the
//! residual `−(K·u − f)` at each slave's dual row (`= aₛ·λ`).
//!
//! # Method-neutral input
//!
//! The constraint structure is read from [`Constraint::relations()`](crate::models::Constraint::relations)
//! — the same seam the Lagrange path builds its `C`/`Cᵀ` blocks from — so the
//! user's mesh-per-term input is never re-parsed here. The physics stiffness `K`
//! is taken *numerically* from the assembled saddle-point [`Matrix`]: the
//! physics×physics sub-block is clean `K` (the multiplier nodes are minted fresh,
//! so they never coincide with a physics node — the DOF partition is exact).
//!
//! # v1 scope: non-chained, disjoint slaves
//!
//! Each relation eliminates exactly one slave DOF, and a slave DOF may not appear
//! in any other relation (neither as another slave nor as a master). This covers
//! periodicity and is validated with a clear error. Chained relations (a master
//! that is itself a slave elsewhere) are out of scope for v1.

use crate::atoms::NodeId;
use crate::containers::matrix::Matrix;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::lu::{self, factorize_csc, lu_solve_vec, SolveOptions, SparseLu};

type NamedDof = (NodeId, String);

/// One eliminated slave DOF and the data needed to build `u₀` and recover its
/// reaction. The masters are folded into `T` at build time, so they are not kept.
struct SlaveInfo {
    /// Physics-DOF index (shared row/column index) of the slave.
    phys_idx: usize,
    /// The slave's constrained node (for the reaction output).
    slave_node: NodeId,
    /// The slave's dual variable — the row that carries its reaction.
    target_dual: String,
    /// The slave coefficient `aₛ`.
    a_s: f64,
    /// Multiplier node whose load slot holds this relation's `g`.
    multiplier_node: NodeId,
    /// Component name of `g` in the load field (the constraint's dual variable).
    imposed_value_slot: String,
}

/// The condensation of a model's constraints against an assembled matrix — the
/// expensive artifact cached transparently on the [`Matrix`] (exactly like
/// [`super::lu::Factorization`]; cleared on any matrix mutation).
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
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #                              None, None, RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// // L'artefact coûteux de l'élimination, mis en cache sur la matrice tout
/// // comme la factorisation LU — et vidé dès que la matrice change.
/// # use pyrucast::ops::solver::eliminate::Condensation;
/// # use pyrucast::ops::model;
/// assert!(k.cached_factorization::<Condensation>().is_none());
/// solver::eliminate::solve(&k, &modele, &charge)?;
/// assert!(k.cached_factorization::<Condensation>().is_some());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct Condensation {
    /// Prolongation `T` (`n_phys × n_free`): `u_phys = T·û + u₀`.
    t: CsrMatrix<f64>,
    /// `Tᵀ` (`n_free × n_phys`), kept to avoid re-transposing at each solve.
    tt: CsrMatrix<f64>,
    /// Physics stiffness `K` (`n_phys × n_phys`), for `K·u₀` and the reactions.
    k_phys: CsrMatrix<f64>,
    /// LU factorization of the reduced `K̂ = Tᵀ K T` (`n_free × n_free`).
    reduced: SparseLu,
    /// Physics dual (row) DOFs, in shared physics-index order — where `f` is read.
    phys_row_dofs: Vec<NamedDof>,
    /// Physics primal (col) DOFs, in shared physics-index order — the output `u`.
    phys_col_dofs: Vec<NamedDof>,
    /// One entry per eliminated slave.
    slaves: Vec<SlaveInfo>,
}

/// Solve `model`'s system by master/slave elimination, using the default options
/// (reduced sparse LU, condensation cached). `matrix` is the assembled
/// saddle-point stiffness of `model` (as produced by
/// [`crate::ops::matrix::stiffness`]); `rhs` is the load field (its `g` values
/// live at the multiplier nodes' imposed-value slots).
///
/// A model with no constraint falls back to a plain [`lu::solve`].
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
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #                              None, None, RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// // La voie **alternative** à Lagrange : on élimine les esclaves au lieu
/// // d'agrandir le système. Même solution, système plus petit.
/// let par_elimination = solver::eliminate::solve(&k, &modele, &charge)?;
/// let par_lagrange = solver::lu::solve(&k, &charge)?;
/// let lu = |f: &NodeField| f.get(0).unwrap().read().value(n[2].id(), "T").unwrap();
/// assert!((lu(&par_elimination) - lu(&par_lagrange)).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve(matrix: &Matrix, model: &Model, rhs: &NodeField) -> Result<NodeField> {
    solve_inner(matrix, model, rhs, &SolveOptions::default(), &NoCancel)
}

/// Like [`solve`] but with explicit [`SolveOptions`] (`method` selects the direct
/// back-end for the *reduced* system; `cache` toggles the condensation cache).
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
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #                              None, None, RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::SolveOptions;
/// # use pyrucast::ops::model;
/// // `method` choisit le moteur direct du système **réduit** ; `cache`
/// // pilote la condensation.
/// let u = solver::eliminate::solve_with_options(
///     &k, &modele, &charge, &SolveOptions::default())?;
/// assert!((u.get(0)?.read().value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_with_options(
    matrix: &Matrix,
    model: &Model,
    rhs: &NodeField,
    options: &SolveOptions,
) -> Result<NodeField> {
    solve_inner(matrix, model, rhs, options, &NoCancel)
}

/// Like [`solve`], but polls `cancel` at each phase boundary so the call can be
/// stopped early (returning [`PyrucastError::Interrupted`]). Same granularity as
/// [`lu::solve_cancellable`]: the heavy library calls are not interrupted
/// mid-way, only around them.
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
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #                              None, None, RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use std::sync::atomic::{AtomicBool, Ordering};
/// # use pyrucast::ops::model;
/// let stop = AtomicBool::new(false);
/// assert!(solver::eliminate::solve_cancellable(&k, &modele, &charge, &stop).is_ok());
/// stop.store(true, Ordering::Relaxed);
/// assert!(solver::eliminate::solve_cancellable(&k, &modele, &charge, &stop).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable(
    matrix: &Matrix,
    model: &Model,
    rhs: &NodeField,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(matrix, model, rhs, &SolveOptions::default(), cancel)
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
/// # let modele = model::heat_conduction(&fes).unwrap()
/// #     .union(&model::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #                              None, None, RelationSense::Equality).unwrap()).unwrap();
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// # let k = matrix::stiffness(&modele, &materiaux).unwrap();
/// # let charge = NodeField::from_submesh(&mult.get(0).unwrap(),
/// #                                      vec!["imposed_T".into()]).unwrap();
/// # charge.get(0).unwrap().write().add_to_component("imposed_T", 100.0).unwrap();
/// # use pyrucast::ops::solver::lu::SolveOptions;
/// # use std::sync::atomic::AtomicBool;
/// # use pyrucast::ops::model;
/// let stop = AtomicBool::new(false);
/// let u = solver::eliminate::solve_cancellable_with_options(
///     &k, &modele, &charge, &SolveOptions::default(), &stop)?;
/// assert!(u.node_count()? > 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable_with_options(
    matrix: &Matrix,
    model: &Model,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_inner(matrix, model, rhs, options, cancel)
}

fn solve_inner(
    matrix: &Matrix,
    model: &Model,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    cancel.check()?;

    // No constraint ⇒ elimination is a no-op; solve the matrix as-is.
    if !has_constraint(model)? {
        return lu::solve_cancellable_with_options(matrix, rhs, options, cancel);
    }

    // ── Step 1 — obtain the condensation (cached or fresh) ─────────────
    let cond: Arc<Condensation> = if options.cache {
        match matrix.cached_factorization::<Condensation>() {
            Some(c) => c,
            None => {
                let c = Arc::new(build_condensation(model, matrix)?);
                matrix.store_factorization(c.clone());
                c
            }
        }
    } else {
        Arc::new(build_condensation(model, matrix)?)
    };
    cancel.check()?;

    // ── Step 2 — build u₀ from the right-hand sides g ──────────────────
    let n_phys = cond.phys_col_dofs.len();
    let g_dofs: Vec<NamedDof> = cond
        .slaves
        .iter()
        .map(|s| (s.multiplier_node, s.imposed_value_slot.clone()))
        .collect();
    let gs = rhs.gather(&g_dofs)?;
    let mut u0 = DVector::<f64>::zeros(n_phys);
    for (s, &g) in cond.slaves.iter().zip(&gs) {
        u0[s.phys_idx] = g / s.a_s;
    }

    // ── Step 3 — reduce the right-hand side: f̂ = Tᵀ (f − K·u₀) ─────────
    let f_phys = DVector::from_vec(rhs.gather(&cond.phys_row_dofs)?);
    let rhs_full = &f_phys - &cond.k_phys * &u0;
    let rhs_hat = &cond.tt * &rhs_full;
    cancel.check()?;

    // ── Step 4 — solve the reduced system and prolong ──────────────────
    let u_hat = lu_solve_vec(&cond.reduced, rhs_hat.as_slice());
    if u_hat.iter().any(|v| !v.is_finite()) {
        return Err(PyrucastError::Message(
            "solve: reduced LU failed (matrix is singular)".into(),
        ));
    }
    let u_phys = &cond.t * &DVector::from_vec(u_hat) + &u0;
    cancel.check()?;

    // ── Step 5 — reactions: −(K·u − f) at each slave's dual row ─────────
    let resid = &cond.k_phys * &u_phys - &f_phys;

    // ── Step 6 — assemble the solution NodeField ───────────────────────
    // Primal u at every physics node, plus each slave's reaction in its dual row
    // (same shape as the Lagrange solution: primal field + a dual reaction).
    //
    // Unlike `lu`/`unilateral`, this output cannot be wrapped by
    // `Matrix::field_from_col_values`: its DOF set is not the matrix columns
    // (multiplier columns are condensed out, and the reactions land on **dual**
    // variables at the slave nodes — row-flavoured DOFs). Keep the explicit
    // materialised support here.
    let mut out_dofs = cond.phys_col_dofs.clone();
    let mut out_vals: Vec<f64> = u_phys.iter().copied().collect();
    for s in &cond.slaves {
        out_dofs.push((s.slave_node, s.target_dual.clone()));
        out_vals.push(-resid[s.phys_idx]);
    }
    NodeField::from_dof_values(rhs.coords()?, &out_dofs, &out_vals)
}

/// Whether `model` carries at least one constraint sub-model.
fn has_constraint(model: &Model) -> Result<bool> {
    for h in model {
        if h.read().as_kind().as_constraint().is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Build the [`Condensation`]: partition physics vs multiplier DOFs, extract the
/// physics stiffness, read the relations, pick slaves, assemble `T` and factorize
/// the reduced matrix.
fn build_condensation(model: &Model, matrix: &Matrix) -> Result<Condensation> {
    // ── Collect (relation, imposed_value_slot) from every constraint ────
    // The slot is carried **per relation** (`Relation::imposed_value`), so a
    // multi-component constraint (`Embedded`) reads each component's own g slot.
    let mut relations: Vec<(crate::models::Relation, String)> = Vec::new();
    for h in model {
        let sub = h.read();
        if let Some(constraint) = sub.as_kind().as_constraint() {
            for rel in constraint.relations()? {
                // Condensation enforces its relations unconditionally — a
                // unilateral relation needs the active-set solver instead.
                if rel.sense != crate::models::RelationSense::Equality {
                    return Err(PyrucastError::Message(format!(
                        "elimination: relation at multiplier node {:?} is unilateral \
                         ('{}') — solve it with solve_unilateral",
                        rel.multiplier_node, rel.sense
                    )));
                }
                let slot = rel.imposed_value.clone();
                relations.push((rel, slot));
            }
        }
    }

    // ── Partition the matrix DOFs: multiplier nodes vs physics ─────────
    let mult_nodes: HashSet<NodeId> = relations.iter().map(|(r, _)| r.multiplier_node).collect();
    let full_row = matrix.row_dofs()?;
    let full_col = matrix.col_dofs()?;

    // Physics DOFs, in first-seen (matrix) order. Row (dual) and column (primal)
    // must line up node-for-node so a single physics index `k` names the
    // conjugate pair — required for T to apply on both the primal (u) and the
    // dual (f) sides. Validate that alignment.
    let mut phys_row_dofs = Vec::new();
    let mut full_row_to_phys = vec![usize::MAX; full_row.len()];
    for (i, dof) in full_row.iter().enumerate() {
        if !mult_nodes.contains(&dof.0) {
            full_row_to_phys[i] = phys_row_dofs.len();
            phys_row_dofs.push(dof.clone());
        }
    }
    let mut phys_col_dofs = Vec::new();
    let mut full_col_to_phys = vec![usize::MAX; full_col.len()];
    let mut pos_of_col: HashMap<NamedDof, usize> = HashMap::new();
    for (j, dof) in full_col.iter().enumerate() {
        if !mult_nodes.contains(&dof.0) {
            let idx = phys_col_dofs.len();
            full_col_to_phys[j] = idx;
            pos_of_col.insert(dof.clone(), idx);
            phys_col_dofs.push(dof.clone());
        }
    }
    let n_phys = phys_col_dofs.len();
    if phys_row_dofs.len() != n_phys {
        return Err(PyrucastError::Message(format!(
            "elimination: physics block is not square ({} dual rows vs {} primal cols)",
            phys_row_dofs.len(),
            n_phys
        )));
    }
    for k in 0..n_phys {
        if phys_row_dofs[k].0 != phys_col_dofs[k].0 {
            return Err(PyrucastError::Message(
                "elimination: physics row/column DOFs are not conjugate-aligned; \
                 this physics is unsupported by the elimination solver"
                    .into(),
            ));
        }
    }

    // ── Extract K_phys from the assembled CSR ──────────────────────────
    let csr = matrix.to_csr()?;
    let mut k_coo = CooMatrix::<f64>::new(n_phys, n_phys);
    for (r, c, &v) in csr.triplet_iter() {
        let (pr, pc) = (full_row_to_phys[r], full_col_to_phys[c]);
        if pr != usize::MAX && pc != usize::MAX {
            k_coo.push(pr, pc, v);
        }
    }
    let k_phys = CsrMatrix::from(&k_coo);

    // ── Pass 1: pick slaves + validate the non-chained/disjoint scope ──
    struct SlaveRecord {
        s_idx: usize,
        a_s: f64,
        masters: Vec<(usize, f64)>,
        slave_node: NodeId,
        target_dual: String,
        multiplier_node: NodeId,
        imposed_value_slot: String,
    }
    let mut records: Vec<SlaveRecord> = Vec::with_capacity(relations.len());
    let mut slave_set: HashSet<usize> = HashSet::new();
    let mut master_set: HashSet<usize> = HashSet::new();
    for (rel, slot) in &relations {
        // Resolve every term to a physics index.
        let mut resolved: Vec<(usize, &crate::models::ConstraintTerm)> =
            Vec::with_capacity(rel.terms.len());
        for term in &rel.terms {
            let idx = *pos_of_col
                .get(&(term.node, term.variable.clone()))
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "elimination: term ({:?}, '{}') is not a physics DOF of the matrix",
                        term.node, term.variable
                    ))
                })?;
            resolved.push((idx, term));
        }
        // Slave = the **available** term (not already a slave or a master of
        // another relation) with the largest |coefficient| — the numerically
        // safest denominator that also keeps shared masters as masters. If no
        // term is available, every DOF of this relation is already constrained:
        // that is a chained/over-constrained system, out of v1 scope.
        let (slave_pos, &(s_idx, slave_term)) = resolved
            .iter()
            .enumerate()
            .filter(|(_, (idx, _))| !slave_set.contains(idx) && !master_set.contains(idx))
            .max_by(|a, b| {
                a.1 .1
                    .coefficient
                    .abs()
                    .total_cmp(&b.1 .1.coefficient.abs())
            })
            .ok_or_else(|| {
                PyrucastError::Message(format!(
                    "elimination: relation at multiplier node {:?} has no eliminable \
                     DOF — all its terms are already constrained (chaining is out of \
                     scope, v1)",
                    rel.multiplier_node
                ))
            })?;
        let a_s = slave_term.coefficient;
        if a_s == 0.0 {
            return Err(PyrucastError::Message(format!(
                "elimination: relation at multiplier node {:?} has no term with a \
                 nonzero coefficient",
                rel.multiplier_node
            )));
        }
        let mut masters = Vec::with_capacity(resolved.len() - 1);
        for (p, &(m_idx, term)) in resolved.iter().enumerate() {
            if p == slave_pos {
                continue;
            }
            if slave_set.contains(&m_idx) {
                return Err(PyrucastError::Message(format!(
                    "elimination: master DOF ({:?}, '{}') is a slave of another \
                     relation — chaining is out of scope (v1)",
                    term.node, term.variable
                )));
            }
            masters.push((m_idx, term.coefficient));
        }
        slave_set.insert(s_idx);
        master_set.extend(masters.iter().map(|(m, _)| *m));
        records.push(SlaveRecord {
            s_idx,
            a_s,
            masters,
            slave_node: slave_term.node,
            target_dual: slave_term.target_dual.clone(),
            multiplier_node: rel.multiplier_node,
            imposed_value_slot: slot.clone(),
        });
    }

    // ── Free-DOF numbering (physics minus slaves) ──────────────────────
    let mut phys_to_free = vec![None; n_phys];
    let mut n_free = 0;
    for (p, slot) in phys_to_free.iter_mut().enumerate() {
        if !slave_set.contains(&p) {
            *slot = Some(n_free);
            n_free += 1;
        }
    }

    // ── Build T (n_phys × n_free): identity on free rows, master combo on
    //    slave rows (u_s = −Σ (a_k/a_s) u_master) ────────────────────────
    let mut t_coo = CooMatrix::<f64>::new(n_phys, n_free);
    for (p, slot) in phys_to_free.iter().enumerate() {
        if let Some(q) = slot {
            t_coo.push(p, *q, 1.0);
        }
    }
    for rec in &records {
        for &(m_idx, a_k) in &rec.masters {
            // Masters are free by construction (validated non-chained).
            let q = phys_to_free[m_idx].expect("master is a free DOF");
            t_coo.push(rec.s_idx, q, -a_k / rec.a_s);
        }
    }
    let t = CsrMatrix::from(&t_coo);
    let tt = t.transpose();

    // ── Reduced K̂ = Tᵀ K T, factorized ────────────────────────────────
    let khat: CsrMatrix<f64> = &(&tt * &k_phys) * &t;
    let reduced = factorize_csc(&CscMatrix::from(&khat))?;

    let slaves = records
        .into_iter()
        .map(|r| SlaveInfo {
            phys_idx: r.s_idx,
            slave_node: r.slave_node,
            target_dual: r.target_dual,
            a_s: r.a_s,
            multiplier_node: r.multiplier_node,
            imposed_value_slot: r.imposed_value_slot,
        })
        .collect();

    Ok(Condensation {
        t,
        tt,
        k_phys,
        reduced,
        phys_row_dofs,
        phys_col_dofs,
        slaves,
    })
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::matrix::{DofOrdering, SubMatrix};
    use crate::containers::mesh::SubMesh;
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A constraint-free model must route through the plain LU solver: a
    /// standalone `2·T = 6` system (no constraints) solves to `T = 3`.
    #[test]
    fn empty_model_falls_back_to_plain_solve() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
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
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(block)).unwrap();
        m.finalize().unwrap();

        let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
        rhs.set_value(a.id(), "q", 6.0).unwrap();
        let rhs = NodeField::from_sub(rhs);

        let model = Model::empty();
        let sol = solve(&m, &model, &rhs).unwrap();
        assert!((sol.value(a.id(), "T").unwrap() - 3.0).abs() < 1e-12);
    }
}
