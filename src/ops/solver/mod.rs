//! Linear solvers for an assembled [`Matrix`] and a right-hand side.
//!
//! One entry, [`solve`]: the matrix carries everything the solve needs. A
//! constraint relation is a row named [`IMPOSED`](crate::models::IMPOSED), its
//! coefficients are that row, its multiplier is the column
//! [`MULTIPLIER`](crate::models::MULTIPLIER) of the same node, and its sense
//! (`=`, `≥`, `≤`) is declared by the block that built it
//! ([`SubMatrix::row_sense`](crate::containers::matrix::SubMatrix::row_sense)).
//! No model is passed alongside, so none can contradict the matrix.
//!
//! [`SolveOptions::relations`] chooses how the relations are enforced:
//!
//! - [`RelationMethod::Lagrange`] (default) — the saddle-point as assembled,
//!   factorized by a sparse LU (or Cholesky); as soon as the matrix carries an
//!   inequality, the **active-set** method decides which relations are
//!   saturated ([`ActiveSetMethod`]);
//! - [`RelationMethod::Eliminate`] — every equality relation eliminates one
//!   unknown (master/slave condensation), and the smaller system left carries
//!   no multiplier but those of the inequalities, which the same active-set
//!   loop resolves on it;
//! - [`RelationMethod::Auto`] — the elimination when every relation is within
//!   its scope, the multipliers otherwise.
//!
//! Whatever is costly — the factorization, the condensation, the active-set
//! state — is cached on the matrix ([`SolveOptions::cache`]) and cleared when
//! the matrix changes. What is cached also records the path that made it:
//! under `auto`, a matrix solved again goes down that path again.
//!
//! ```
//! # use pyrucast::models::symmetry::MaterialSymmetry;
//! # use pyrucast::aggregate::Aggregate;
//! # use pyrucast::atoms::{ElementType, Node};
//! # use pyrucast::containers::field::SubField;
//! # use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! # use pyrucast::containers::mesh::{Mesh, SubMesh};
//! # use pyrucast::containers::node_field::NodeField;
//! # use pyrucast::coords::Coords;
//! # use pyrucast::handle::Handle;
//! # use pyrucast::models::RelationSense;
//! # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
//! # let coords = Handle::new(Coords::new(1).unwrap());
//! # let n: Vec<Node> = (0..3)
//! #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
//! #     .collect();
//! # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
//! # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
//! # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
//! # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
//! let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
//! let modele = barre.union(&model::dirichlet(&impose, &barre, "T", RelationSense::Equality)?)?;
//! # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
//! let k = matrix::stiffness(&modele, &materiaux)?;
//! # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
//! #                                      vec!["imposed".into()])?;
//! # charge.get(0)?.write().add_to_component("imposed", 100.0)?;
//! // A bar held at 100 °C on its left end, with no other load: 100 °C
//! // everywhere — by multipliers or by elimination alike.
//! let u = solver::solve(&k, &charge)?;
//! assert!((u.value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
//! let par_elimination = solver::solve_with_options(&k, &charge, &solver::SolveOptions {
//!     relations: solver::RelationMethod::Eliminate, ..Default::default() })?;
//! assert!((par_elimination.value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
//! # Ok::<(), pyrucast::PyrucastError>(())
//! ```

use crate::containers::matrix::Matrix;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::interrupt::{Cancel, NoCancel};

mod eliminate;
mod lu;
pub mod methods;
mod relations;
mod scaling;
mod unilateral;

pub use lu::{SolveMethod, Verbosity};
pub use unilateral::ActiveSetMethod;

/// How a solve enforces the constraint relations of the matrix.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::named::Named;
/// # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
/// # use pyrucast::ops::solver::{RelationMethod, SolveOptions};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// # let modele = barre.union(&model::dirichlet(&impose, &barre, "T", RelationSense::Equality)?)?;
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
/// # let k = matrix::stiffness(&modele, &materiaux)?;
/// # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
/// #                                      vec!["imposed".into()])?;
/// # charge.get(0)?.write().add_to_component("imposed", 100.0)?;
/// assert_eq!(RelationMethod::default(), RelationMethod::Lagrange);
/// assert_eq!(RelationMethod::parse("eliminate")?, RelationMethod::Eliminate);
/// // Either way, the same solution: the unknowns, and each relation's
/// // multiplier at its node.
/// let lagrange = solver::solve(&k, &charge)?;
/// let elimine = solver::solve_with_options(&k, &charge, &SolveOptions {
///     relations: RelationMethod::Eliminate, ..Default::default() })?;
/// let m = modele.multiplier_mesh()?.get(0)?.read().connectivity()[0];
/// assert!((elimine.value(m, "lambda")? - lagrange.value(m, "lambda")?).abs() < 1e-9);
///
/// // `auto` eliminates this one — a single relation, an unknown of its own —
/// // and gives the same solution again.
/// assert_eq!(RelationMethod::parse("auto")?, RelationMethod::Auto);
/// let auto = solver::solve_with_options(&k, &charge, &SolveOptions {
///     relations: RelationMethod::Auto, ..Default::default() })?;
/// assert!((auto.value(m, "lambda")? - lagrange.value(m, "lambda")?).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RelationMethod {
    /// Lagrange multipliers: the saddle-point is solved as assembled, the
    /// multipliers come back in the solution at their nodes.
    #[default]
    Lagrange,
    /// Master/slave elimination: each equality relation expresses one unknown
    /// from the others, and the reduced system — smaller, without multipliers,
    /// usually positive definite — is solved. Each eliminated relation's
    /// multiplier is recovered from its slave's row and returned at its node:
    /// the solution is the Lagrange one, the same DOFs on the same supports.
    /// An inequality is not eliminated: it stays on the
    /// reduced system with its multiplier, for the active-set loop. Scope:
    /// non-chained, disjoint slaves.
    Eliminate,
    /// The elimination when every equality relation is within its scope — an
    /// unknown of its own, appearing in no other equality relation — and the
    /// multipliers otherwise: a relation chained to another is left to them
    /// instead of being refused. Either way, the same solution.
    ///
    /// Decided when the matrix is first solved: what that solve caches — a
    /// condensation, or a factorization of the saddle-point — records the
    /// path, and the next solve of the same matrix takes it again.
    Auto,
}

impl crate::named::Named for RelationMethod {
    const LABEL: &'static str = "relation method";
    const VALUES: &'static [Self] = &[
        RelationMethod::Lagrange,
        RelationMethod::Eliminate,
        RelationMethod::Auto,
    ];

    fn name(self) -> &'static str {
        match self {
            RelationMethod::Lagrange => "lagrange",
            RelationMethod::Eliminate => "eliminate",
            RelationMethod::Auto => "auto",
        }
    }
}

/// Options for [`solve_with_options`]. The defaults are those of the Python
/// face: `method="lu"`, `relations="lagrange"`, `active_set="schur"`,
/// `cache=True`, `max_iter=100`, `tol=1e-10`, `verbosity="silent"`.
///
/// ```
/// # use pyrucast::ops::solver::{ActiveSetMethod, RelationMethod, SolveMethod, SolveOptions};
/// let d = SolveOptions::default();
/// assert_eq!(d.method, SolveMethod::Lu);
/// assert_eq!(d.relations, RelationMethod::Lagrange);
/// assert_eq!(d.active_set, ActiveSetMethod::SchurComplement);
/// assert!(d.cache && d.max_iter == 100 && d.tol == 1e-10);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    /// Direct factorization: LU, or Cholesky for a symmetric positive definite
    /// system (a saddle-point never is; a system reduced by elimination
    /// usually is), or `Auto` to let the solve find out. The active-set loop
    /// factorizes by LU whatever is asked.
    pub method: SolveMethod,
    /// How the constraint relations are enforced.
    pub relations: RelationMethod,
    /// How each status of the active-set loop is factorized — read only when
    /// the matrix carries inequalities.
    pub active_set: ActiveSetMethod,
    /// Reuse / populate what is cached on the matrix: the factorization, the
    /// condensation or the active-set state (with its last converged status as
    /// a warm start). A factorization, alone or inside a condensation, is
    /// reused for the `method` it was made with only — or for `Auto`, which
    /// takes whichever is there. When `false`, everything is computed afresh
    /// and the cache is not touched.
    pub cache: bool,
    /// Maximum number of status iterations of the active-set loop.
    pub max_iter: usize,
    /// Sign tolerance of the active-set loop, on the multiplier (release) and
    /// on the gap (activation), in the units of the reaction and of the
    /// relation.
    pub tol: f64,
    /// How much the solve says about what it did.
    pub verbosity: Verbosity,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            method: SolveMethod::Lu,
            relations: RelationMethod::Lagrange,
            active_set: ActiveSetMethod::SchurComplement,
            cache: true,
            max_iter: 100,
            tol: 1e-10,
            verbosity: Verbosity::Silent,
        }
    }
}

/// Solve `matrix · x = rhs` with the default options.
///
/// `matrix` is assembled and square; `rhs` is read at every row DOF of the
/// matrix, a DOF no zone defines reading zero — the imposed value of a
/// relation lives at its multiplier node, component `imposed`. A relation is
/// enforced by its multiplier, and an inequality relation (`≥`, `≤`) by the
/// active-set loop: an active relation holds as an equation and its multiplier
/// has the sign of a push (`λ ≤ 0` for `≥`, `λ ≥ 0` for `≤`: the reaction is
/// `−Cᵀλ`), an inactive one reports `λ = 0`.
///
/// The solution has one zone per distinct column support of the matrix's
/// blocks, each sharing that block's POI1 support — the multipliers included.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// // A floor: T(0) ≥ 100 °C, nothing else.
/// let plancher = model::dirichlet(&impose, &barre, "T", RelationSense::GreaterEqual)?;
/// let modele = barre.union(&plancher)?;
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
/// let k = matrix::stiffness(&modele, &materiaux)?;
/// # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
/// #                                      vec!["imposed".into()])?;
/// # charge.get(0)?.write().add_to_component("imposed", 100.0)?;
/// // The matrix knows the relation is an inequality: the active set finds it
/// // saturated, and the bar sits on its floor.
/// let u = solver::solve(&k, &charge)?;
/// assert!((u.value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve(matrix: &Matrix, rhs: &NodeField) -> Result<NodeField> {
    solve_cancellable_with_options(matrix, rhs, &SolveOptions::default(), &NoCancel)
}

/// Like [`solve`], with explicit [`SolveOptions`].
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
/// # use pyrucast::ops::solver::{SolveMethod, SolveOptions};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// # let modele = barre.union(&model::dirichlet(&impose, &barre, "T", RelationSense::Equality)?)?;
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
/// # let k = matrix::stiffness(&modele, &materiaux)?;
/// # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
/// #                                      vec!["imposed".into()])?;
/// # charge.get(0)?.write().add_to_component("imposed", 100.0)?;
/// // A Lagrange saddle-point is symmetric but indefinite: Cholesky refuses it,
/// // and says so.
/// let cholesky = SolveOptions { method: SolveMethod::Cholesky, cache: false, ..Default::default() };
/// let err = solver::solve_with_options(&k, &charge, &cholesky).unwrap_err().to_string();
/// assert!(err.contains("positive definite"), "{err}");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_with_options(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
) -> Result<NodeField> {
    solve_cancellable_with_options(matrix, rhs, options, &NoCancel)
}

/// Like [`solve`], but polls `cancel` at each phase boundary so the call can be
/// stopped early (returning [`PyrucastError::Interrupted`](crate::PyrucastError::Interrupted)).
///
/// **Granularity.** A sparse factorization is a single library call with no
/// cooperative checkpoint: `cancel` is polled *around* the heavy steps and at
/// each status iteration of the active-set loop, so a `Ctrl+C` lands at the
/// next phase boundary, not inside a factorization.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
/// # use std::sync::atomic::{AtomicBool, Ordering};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// # let modele = barre.union(&model::dirichlet(&impose, &barre, "T", RelationSense::Equality)?)?;
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
/// # let k = matrix::stiffness(&modele, &materiaux)?;
/// # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
/// #                                      vec!["imposed".into()])?;
/// let stop = AtomicBool::new(false);
/// assert!(solver::solve_cancellable(&k, &charge, &stop).is_ok());
/// // A token armed in advance: the stop lands at the first phase boundary.
/// stop.store(true, Ordering::Relaxed);
/// assert!(solver::solve_cancellable(&k, &charge, &stop).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable(
    matrix: &Matrix,
    rhs: &NodeField,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    solve_cancellable_with_options(matrix, rhs, &SolveOptions::default(), cancel)
}

/// [`solve_cancellable`] with explicit [`SolveOptions`] — the full form the
/// Python binding routes to.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::RelationSense;
/// # use pyrucast::ops::{element_field, matrix, mesh, model, solver};
/// # use pyrucast::ops::solver::{ActiveSetMethod, SolveOptions};
/// # use std::sync::atomic::AtomicBool;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = (0..3)
/// #     .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
/// #     .collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # for i in 0..2 { sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap(); }
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let barre = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// # let modele = barre.union(&model::dirichlet(&impose, &barre, "T", RelationSense::GreaterEqual)?)?;
/// # let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
/// # let k = matrix::stiffness(&modele, &materiaux)?;
/// # let charge = NodeField::from_submesh(&modele.multiplier_mesh()?.get(0)?,
/// #                                      vec!["imposed".into()])?;
/// # charge.get(0)?.write().add_to_component("imposed", 100.0)?;
/// let stop = AtomicBool::new(false);
/// let refacto = SolveOptions { active_set: ActiveSetMethod::Refactorize, ..Default::default() };
/// let u = solver::solve_cancellable_with_options(&k, &charge, &refacto, &stop)?;
/// assert!((u.value(n[2].id(), "T")? - 100.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn solve_cancellable_with_options(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    cancel.check()?;
    match options.relations {
        // An explicit `relations` decides the path, with the blocks, before any
        // cache is looked at. Each path keeps its own artefact in the matrix's
        // single cache slot: another path on the same matrix replaces it.
        RelationMethod::Lagrange => by_multipliers(matrix, rhs, options, cancel),
        RelationMethod::Eliminate => eliminate::solve(matrix, rhs, options, cancel),
        // `auto` reads the path off what the slot holds: a factorization or an
        // active-set state was made by multipliers, a condensation by
        // eliminating — `eliminate::solve` takes that one up itself. On an
        // empty slot the elimination is tried, and hands over to the
        // multipliers when a relation is out of its scope.
        RelationMethod::Auto
            if options.cache
                && (matrix.cached_factorization::<lu::Factorization>().is_some()
                    || matrix
                        .cached_factorization::<unilateral::ActiveSetState>()
                        .is_some()) =>
        {
            by_multipliers(matrix, rhs, options, cancel)
        }
        RelationMethod::Auto => eliminate::solve(matrix, rhs, options, cancel),
    }
}

/// The multipliers path: the saddle-point as assembled, through the active-set
/// loop as soon as the matrix carries an inequality.
fn by_multipliers(
    matrix: &Matrix,
    rhs: &NodeField,
    options: &SolveOptions,
    cancel: &dyn Cancel,
) -> Result<NodeField> {
    if matrix.has_inequality() {
        unilateral::solve(matrix, rhs, options, cancel)
    } else {
        lu::solve(matrix, rhs, options, cancel)
    }
}
