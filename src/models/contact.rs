//! Node-to-surface contact constraint via Lagrange multipliers — the
//! **unilateral** sibling of [`Embedded`](crate::models::embedded::Embedded).
//!
//! A `Contact` sub-model prevents the nodes of a **slave** mesh from
//! penetrating a **master** surface mesh. Each slave node `s` is paired at
//! build time with its closest master facet
//! ([`project_points`](crate::ops::geom::project_points())): projection weights
//! `Nᵢ(ξ)`, unit facet normal `n` and initial signed gap `g₀`. Linearised
//! around the initial geometry (small displacements, fixed pairing and
//! normal), non-penetration reads, per slave node,
//!
//! ```text
//! g₀ + n·u(s) − Σᵢ Nᵢ(ξ)·n·u(masterᵢ) ≥ 0
//! ```
//!
//! i.e. one **unilateral relation** `Σ coeffs·u ≥ −g₀` per slave node, whose
//! multiplier `λ ≤ 0` carries the contact reaction (the force in the solution
//! field is `−λ·n` on the slave, spread as `+λ·Nᵢ·n` on the master facet). The
//! model is solved by the active-set operator
//! [`solve_unilateral`](crate::ops::solver::unilateral).
//!
//! # Orientation and right-hand side
//!
//! The master surface must be **consistently oriented** with its normal
//! pointing toward the slave body (see
//! [`project_points`](crate::ops::geom::project_points())): the initial gap `g₀`
//! is then positive when separated, negative when penetrating. The relation's
//! right-hand side is `−g₀`; the helper
//! [`Model::contact_gaps`](crate::containers::model::Model::contact_gaps)
//! builds that load field so the user never computes it by hand. Omitting it
//! treats every pair as initially touching (`g₀ = 0`).
//!
//! # Variables
//!
//! One relation per slave node, all sharing this sub-model's variable pair:
//!
//! - `multiplier` (default `lambda_contact`) — the primal on the multiplier
//!   node: the contact reaction intensity (`≤ 0` while touching, `0` when
//!   separated);
//! - `imposed_value` (default `contact_gap`) — the dual: the constraint row and
//!   the slot for the right-hand side `−g₀`.
//!
//! `components` names the displacement components and their duals in ambient
//! order (e.g. `[("u_x","f_x"), ("u_y","f_y")]` in 2-D) — exactly one pair per
//! space dimension, since the normal couples them all in one scalar relation.

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::{
    Constraint, ConstraintTerm, Contribution, MatrixKind, Physics, Relation, RelationSense,
    SubModelKind,
};
use crate::ops::geom::project_points;
use crate::ops::mesh::barycenter;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Default multiplier (primal) name — the contact reaction intensity.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # use pyrucast::models::contact;
/// // Dérivés plutôt que tabulés : `multiplier` et `imposed_value` laissés
/// // à `None` prennent ces valeurs.
/// assert_eq!(
///     (contact::default_multiplier().as_str(),
///      contact::default_imposed_value().as_str()),
///     ("lambda_contact", "contact_gap"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn default_multiplier() -> String {
    "lambda_contact".to_string()
}

/// Default imposed-value (dual) name — the constraint row and `−g₀` slot.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # use pyrucast::models::contact;
/// // Dérivés plutôt que tabulés : `multiplier` et `imposed_value` laissés
/// // à `None` prennent ces valeurs.
/// assert_eq!(
///     (contact::default_multiplier().as_str(),
///      contact::default_imposed_value().as_str()),
///     ("lambda_contact", "contact_gap"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn default_imposed_value() -> String {
    "contact_gap".to_string()
}

/// The master binding of one slave node: the closest facet's nodes, the
/// projection weights `Nᵢ(ξ)`, the unit facet normal and the initial gap.
#[derive(Serialize, Deserialize)]
struct Pairing {
    nodes: Vec<NodeId>,
    weights: Vec<f64>,
    normal: Vec<f64>,
    gap: f64,
}

/// One displacement component and the dual row its reaction lands in.
#[derive(Serialize, Deserialize)]
struct Component {
    variable: String,
    target_dual: String,
}

/// Node-to-surface contact constraint (frictionless, small displacements).
///
/// See the module documentation. Built by pairing each slave node with its
/// closest master facet once, at construction; the pairing, the normal and the
/// weights are then **fixed** (linearised contact).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # use pyrucast::models::contact::{self, Contact};
/// # let mut maitre = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # maitre.add_cell(&[n[0].id(), n[1].id()])?;
/// # let master = Mesh::from_submesh(maitre);
/// # let slave = mesh::poi1_from_nodes(&n[2..3])?;
/// // Une relation **unilatérale** par nœud esclave, appariée dès la
/// // construction à sa facette maître la plus proche.
/// let c = Contact::new(&slave, &master,
///     vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
///     None, None)?;
/// assert_eq!(c.relations()?.len(), 1);
/// assert_eq!(c.gaps().len(), 1);
/// assert_eq!(c.relations()?[0].imposed_value, contact::default_imposed_value());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize)]
pub struct Contact {
    /// POI1 mesh of the slave nodes, one cell per relation (its cell `r` is
    /// the slave node of relation `r`).
    slave_mesh: Mesh,
    /// POI1 mesh of the fresh multiplier nodes, colocated with the slave nodes
    /// (paired cell-for-cell).
    multiplier_mesh: Mesh,
    /// POI1 support listing every node the constraint blocks reference — the
    /// slave nodes and every paired master node — deduplicated.
    support_mesh: Mesh,
    /// Master binding per slave node (same order as `slave_mesh`'s cells).
    pairings: Vec<Pairing>,
    /// Displacement components in ambient order (one per space dimension).
    components: Vec<Component>,
    /// This sub-model's primal — the contact reaction (e.g. `lambda_contact`).
    multiplier: String,
    /// This sub-model's dual — constraint row + `−g₀` slot (e.g. `contact_gap`).
    imposed_value: String,
}

impl Contact {
    /// Build a contact constraint preventing the nodes of `slave` from
    /// penetrating the oriented `master` surface, coupling the displacement
    /// `components` (one `(variable, target_dual)` pair per space dimension,
    /// in ambient order — find each dual with
    /// [`Model::dual_of`](crate::containers::model::Model::dual_of)).
    ///
    /// `multiplier` / `imposed_value` default to `lambda_contact` /
    /// `contact_gap` when `None`.
    ///
    /// # Errors
    ///
    /// - `components` is empty or does not match the space dimension;
    /// - `slave` and `master` do not share a `Coords`;
    /// - `master` is not a surface mesh (facet dim ≠ `sdim − 1`), or is empty.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # use pyrucast::models::contact::{self, Contact};
    /// # let mut maitre = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # maitre.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let master = Mesh::from_submesh(maitre);
    /// # let slave = mesh::poi1_from_nodes(&n[2..3])?;
    /// // Une relation **unilatérale** par nœud esclave, appariée dès la
    /// // construction à sa facette maître la plus proche.
    /// let c = Contact::new(&slave, &master,
    ///     vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
    ///     None, None)?;
    /// assert_eq!(c.relations()?.len(), 1);
    /// assert_eq!(c.gaps().len(), 1);
    /// assert_eq!(c.relations()?[0].imposed_value, contact::default_imposed_value());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        slave: &Mesh,
        master: &Mesh,
        components: Vec<(String, String)>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        // Slave and master must live in the same Coords (node ids are relative).
        let coords = slave.coords()?;
        let master_coords = master.coords()?;
        if !coords.same_object(&master_coords) {
            return Err(PyrucastError::Message(
                "Contact: slave and master meshes must share a Coords".into(),
            ));
        }
        let sdim = coords.read().dim() as usize;
        if components.len() != sdim {
            return Err(PyrucastError::Message(format!(
                "Contact: {} component(s) for a {sdim}-D space — the normal couples \
                 every displacement component, give one (variable, dual) pair per \
                 dimension, in ambient order",
                components.len()
            )));
        }

        // The slave support: one POI1 cell per unique slave node.
        let slave_ids = unique_nodes(slave)?;
        if slave_ids.is_empty() {
            return Err(PyrucastError::Message(
                "Contact: slave mesh carries no node".into(),
            ));
        }
        let slave_mesh =
            Mesh::from_submesh(SubMesh::poi1_from_node_ids(coords.clone(), &slave_ids)?);

        // Physical coordinates of the slave nodes, then project them.
        let points: Vec<Vec<f64>> = {
            let c = coords.read();
            slave_ids
                .iter()
                .map(|&n| Ok(c.position(n)?.to_vec()))
                .collect::<Result<_>>()?
        };
        let projections = project_points(master, &points)?;
        let pairings: Vec<Pairing> = projections
            .into_iter()
            .map(|p| Pairing {
                nodes: p.nodes,
                weights: p.weights,
                normal: p.normal,
                gap: p.gap,
            })
            .collect();

        // Fresh multiplier nodes, colocated with the slave nodes.
        let multiplier_mesh = barycenter(&slave_mesh)?;

        // The block support: slave nodes ∪ every paired master node, dedup'd.
        let mut support_ids = slave_ids.clone();
        let mut seen: HashSet<NodeId> = support_ids.iter().copied().collect();
        for p in &pairings {
            for &n in &p.nodes {
                if seen.insert(n) {
                    support_ids.push(n);
                }
            }
        }
        let support_mesh = Mesh::from_submesh(SubMesh::poi1_from_node_ids(coords, &support_ids)?);

        let components = components
            .into_iter()
            .map(|(variable, target_dual)| Component {
                variable,
                target_dual,
            })
            .collect();

        Ok(Self {
            slave_mesh,
            multiplier_mesh,
            support_mesh,
            pairings,
            components,
            multiplier: multiplier.unwrap_or_else(default_multiplier),
            imposed_value: imposed_value.unwrap_or_else(default_imposed_value),
        })
    }

    /// The initial signed gaps `g₀`, one per relation (slave node), in relation
    /// order — positive when separated, negative when penetrating. The
    /// relation's right-hand side is `−g₀` (see
    /// [`Model::contact_gaps`](crate::containers::model::Model::contact_gaps)).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # use pyrucast::models::contact::{self, Contact};
    /// # let mut maitre = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # maitre.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let master = Mesh::from_submesh(maitre);
    /// # let slave = mesh::poi1_from_nodes(&n[2..3])?;
    /// // Une relation **unilatérale** par nœud esclave, appariée dès la
    /// // construction à sa facette maître la plus proche.
    /// let c = Contact::new(&slave, &master,
    ///     vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
    ///     None, None)?;
    /// assert_eq!(c.relations()?.len(), 1);
    /// assert_eq!(c.gaps().len(), 1);
    /// assert_eq!(c.relations()?[0].imposed_value, contact::default_imposed_value());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gaps(&self) -> Vec<f64> {
        self.pairings.iter().map(|p| p.gap).collect()
    }

    /// The single POI1 submesh of multiplier nodes.
    fn multiplier_sm(&self) -> Result<Handle<SubMesh>> {
        self.multiplier_mesh.get(0)
    }

    /// The single POI1 support submesh of the constraint blocks.
    fn support_sm(&self) -> Result<Handle<SubMesh>> {
        self.support_mesh.get(0)
    }

    /// Slave node of relation `r`.
    fn slave_node(&self, r: usize) -> Result<NodeId> {
        Ok(self.slave_mesh.get(0)?.read().connectivity()[r])
    }

    /// Multiplier node of relation `r`.
    fn multiplier_node(&self, r: usize) -> Result<NodeId> {
        Ok(self.multiplier_mesh.get(0)?.read().connectivity()[r])
    }
}

impl SubModelKind for Contact {
    fn primal_vars(&self) -> Vec<String> {
        vec![self.multiplier.clone()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.imposed_value.clone()]
    }

    /// Its relations, applied to the solution: the reaction `Cᵀ λ` on the
    /// constrained nodes, and `C·u` on its own row.
    fn internal_force_contribution(&self) -> Vec<crate::models::ResidualContribution> {
        vec![crate::models::ResidualContribution::Relations]
    }

    fn as_constraint(&self) -> Option<&dyn Constraint> {
        Some(self)
    }

    /// Like the other constraints, `Contact` fills its `C` / `Cᵀ` blocks
    /// directly (no `stiffness_layout`). One block pair for the whole
    /// sub-model: the row of relation `r` couples **every** displacement
    /// component through the normal — the slave node carries `+n_c`, each
    /// master node `−Nᵢ·n_c`.
    fn contributions(
        &self,
        kind: MatrixKind,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        // A constraint only enters the global (stiffness) matrix — no
        // mass/geometric/tangent term.
        if kind != MatrixKind::Stiffness {
            return Ok(Vec::new());
        }
        let mult_sm = self.multiplier_sm()?;
        let support_sm = self.support_sm()?;
        let variables: Vec<String> = self.components.iter().map(|c| c.variable.clone()).collect();
        let duals: Vec<String> = self
            .components
            .iter()
            .map(|c| c.target_dual.clone())
            .collect();

        // C: rows (multiplier, imposed_value), cols (support, every variable).
        let mut c = SubMatrix::new(
            mult_sm.clone(),
            support_sm.clone(),
            vec![self.imposed_value.clone()],
            variables,
            DofOrdering::NodesThenVars,
            false,
        )?;
        // Cᵀ: rows (support, every dual), cols (multiplier, multiplier).
        let mut ct = SubMatrix::new(
            support_sm,
            mult_sm,
            duals,
            vec![self.multiplier.clone()],
            DofOrdering::NodesThenVars,
            false,
        )?;
        for (r, p) in self.pairings.iter().enumerate() {
            let m = self.multiplier_node(r)?;
            let s = self.slave_node(r)?;
            for (comp, &n_c) in self.components.iter().zip(&p.normal) {
                // +n_c on the slave node.
                c.add_entry(m, &self.imposed_value, s, &comp.variable, n_c)?;
                ct.add_entry(s, &comp.target_dual, m, &self.multiplier, n_c)?;
                // −Nᵢ·n_c on each master node.
                for (node, w) in p.nodes.iter().zip(p.weights.iter()) {
                    c.add_entry(m, &self.imposed_value, *node, &comp.variable, -w * n_c)?;
                    ct.add_entry(*node, &comp.target_dual, m, &self.multiplier, -w * n_c)?;
                }
            }
        }
        Ok(vec![Contribution::Literal(vec![c, ct])])
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Constraint]
    }

    fn label(&self) -> &'static str {
        "Contact"
    }

    fn display(&self) -> String {
        format!(
            "SubModel<Contact>: {} slave node(s), {} component(s)",
            self.pairings.len(),
            self.components.len()
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let mut out = format!(
            "SubModel<Contact>\n  primal var(s): {} (multiplier)\n  dual var(s):   {} \
             (imposed value / −g₀)\n  slave nodes: {}\n  components:",
            self.multiplier,
            self.imposed_value,
            self.pairings.len()
        );
        for c in &self.components {
            out.push_str(&format!("\n    {} (dual {})", c.variable, c.target_dual));
        }
        out
    }
}

impl Constraint for Contact {
    fn multiplier_mesh(&self) -> &Mesh {
        &self.multiplier_mesh
    }

    /// One **unilateral** [`Relation`] (`≥`) per slave node: the slave terms
    /// with coefficients `+n_c` (one per component), then one term per
    /// (master node × component) with coefficient `−Nᵢ·n_c`.
    fn relations(&self) -> Result<Vec<Relation>> {
        let mut relations = Vec::with_capacity(self.pairings.len());
        for (r, p) in self.pairings.iter().enumerate() {
            let m = self.multiplier_node(r)?;
            let s = self.slave_node(r)?;
            let mut terms = Vec::with_capacity((1 + p.nodes.len()) * self.components.len());
            for (comp, &n_c) in self.components.iter().zip(&p.normal) {
                terms.push(ConstraintTerm {
                    node: s,
                    variable: comp.variable.clone(),
                    target_dual: comp.target_dual.clone(),
                    coefficient: n_c,
                });
                for (node, w) in p.nodes.iter().zip(p.weights.iter()) {
                    terms.push(ConstraintTerm {
                        node: *node,
                        variable: comp.variable.clone(),
                        target_dual: comp.target_dual.clone(),
                        coefficient: -w * n_c,
                    });
                }
            }
            relations.push(Relation {
                multiplier_node: m,
                imposed_value: self.imposed_value.clone(),
                terms,
                sense: RelationSense::GreaterEqual,
            });
        }
        Ok(relations)
    }
}

/// Unique node ids of a mesh, in order of first appearance across submeshes.
fn unique_nodes(mesh: &Mesh) -> Result<Vec<NodeId>> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut out: Vec<NodeId> = Vec::new();
    for sm in mesh {
        for &nid in sm.read().connectivity() {
            if seen.insert(nid) {
                out.push(nid);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A slave node above a single SEG2 master (2-D): pairing, relation
    /// structure, sense and gap.
    fn seg_and_node(height: f64) -> (Mesh, Mesh) {
        let coords = Handle::new(Coords::new(2).unwrap());
        // Master from (0,0) to (2,0), normal (0,−1) (tangent +x).
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mut msm = SubMesh::new(coords.clone(), ElementType::SEG2);
        msm.add_cell(&[a.id(), b.id()]).unwrap();
        let master = Mesh::from_submesh(msm);
        // Slave node above the middle.
        let s = Node::create_in(coords.clone(), &[0.5, height]).unwrap();
        let mut ssm = SubMesh::new(coords, ElementType::POI1);
        ssm.add_cell(&[s.id()]).unwrap();
        let slave = Mesh::from_submesh(ssm);
        (slave, master)
    }

    #[test]
    fn relations_couple_all_components_through_the_normal() {
        let (slave, master) = seg_and_node(0.25);
        let contact = Contact::new(
            &slave,
            &master,
            vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
            None,
            None,
        )
        .unwrap();

        let rels = contact.relations().unwrap();
        assert_eq!(rels.len(), 1);
        let rel = &rels[0];
        assert_eq!(rel.sense, RelationSense::GreaterEqual);
        // (1 slave + 2 master nodes) × 2 components.
        assert_eq!(rel.terms.len(), 6);
        // Normal (0,−1): u_x coefficients vanish, u_y are ∓.
        let sum_abs_x: f64 = rel
            .terms
            .iter()
            .filter(|t| t.variable == "u_x")
            .map(|t| t.coefficient.abs())
            .sum();
        assert!(sum_abs_x < 1e-12);
        let slave_y = rel
            .terms
            .iter()
            .find(|t| t.variable == "u_y" && t.coefficient < 0.0)
            .expect("slave term −1 on u_y");
        assert!((slave_y.coefficient + 1.0).abs() < 1e-12);
        // Master weights sum to +1 on u_y (−(−1)·Nᵢ), gap = −0.25 (behind n).
        let master_y: f64 = rel
            .terms
            .iter()
            .filter(|t| t.variable == "u_y" && t.coefficient > 0.0)
            .map(|t| t.coefficient)
            .sum();
        assert!((master_y - 1.0).abs() < 1e-12);
        assert!((contact.gaps()[0] + 0.25).abs() < 1e-12);
    }

    #[test]
    fn component_count_must_match_dimension() {
        let (slave, master) = seg_and_node(0.25);
        assert!(Contact::new(
            &slave,
            &master,
            vec![("u_y".into(), "f_y".into())], // 1 component in 2-D
            None,
            None,
        )
        .is_err());
    }
}
