//! Multi-point constraint (MPC) via Lagrange multipliers.
//!
//! An MPC sub-model imposes one or more **linear relations** between degrees of
//! freedom: for each relation `r`,
//!
//! ```text
//! Σₖ aₖ · u(nodeₖ, varₖ) = g
//! ```
//!
//! It is a *constraint*, not a volumetric physics: it carries no material and no
//! constitutive law, and it creates no node (it never mutates the
//! [`Coords`](crate::containers::mesh::Coords)). It is the **generalisation of
//! [`Dirichlet`](crate::models::dirichlet::Dirichlet)**, which is the single-term
//! relation `1·u = u_d`.
//!
//! # Mesh-per-term layout
//!
//! Each term is a [`MpcTerm`] = (POI1 mesh, primal variable, dual variable,
//! scalar coefficient). All term meshes and the `multiplier_mesh` are paired
//! **element-for-element**: relation `r` links the `r`-th cell of *every* term
//! mesh to the `r`-th multiplier node. So a periodicity between two surfaces of
//! `N` nodes is `N` relations at once, vectorised over the cells (no per-node
//! loop). Consistent ordering of the paired meshes is the **user's
//! responsibility** (cell `r` of each mesh must be geometric partners).
//!
//! # Variables
//!
//! Like `Dirichlet`, an MPC owns a pair of variables shared by all its relations:
//!
//! - `multiplier` (default `lambda_mpc`) — this sub-model's **primal**: the
//!   Lagrange multiplier `λ`, an unknown of the augmented system, whose solved
//!   value is the **reaction**;
//! - `imposed_value` (default `mpc_rhs`) — this sub-model's **dual**: the
//!   constraint-equation row, and the **slot** at which the user writes the
//!   right-hand side `g` in the load `SubNodeField` (on the multiplier node).
//!
//! Each term's `target_dual` is the dual (residual) variable of the term's own
//! physics — the row into which its reaction `aₖ·λ` is added. Find it easily with
//! [`Model::dual_of`](crate::containers::model::Model::dual_of).

use crate::aggregate::Aggregate;
use crate::containers::element_field::SubElementField;
use crate::containers::mesh::{ElementType, Mesh, NodeId};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{
    constraint_block_pair, Constraint, ConstraintTerm, Contribution, Relation, SubModelKind,
};
use crate::store::read;
use serde::{Deserialize, Serialize};

/// Default multiplier (primal) name shared by every relation of an MPC.
pub fn default_multiplier() -> String {
    "lambda_mpc".to_string()
}

/// Default imposed-value (dual) name — the constraint-equation row and `g` slot.
pub fn default_imposed_value() -> String {
    "mpc_rhs".to_string()
}

/// One term `coefficient · u(node, variable)` shared by every relation of an
/// [`Mpc`]. Its mesh is POI1: cell `r` holds the node of relation `r`.
#[derive(Serialize, Deserialize)]
pub struct MpcTerm {
    /// POI1 mesh, paired element-for-element with the other terms and the
    /// multiplier mesh (cell `r` = the node of relation `r` for this term).
    pub(crate) mesh: Mesh,
    /// Constrained primal variable (a *column* of the target's stiffness), e.g. `"u_x"`.
    pub(crate) variable: String,
    /// The term physics's dual; the *row* where the reaction `coefficient·λ`
    /// lands (e.g. `"f_x"`). Find it with `Model::dual_of`.
    pub(crate) target_dual: String,
    /// Scalar coefficient `aₖ` (v1).
    pub(crate) coefficient: f64,
}

impl MpcTerm {
    /// Build a term. `mesh` must be POI1 (one node per relation); it is shared
    /// (its submeshes are increfed so the nodes stay alive).
    pub fn new(
        mesh: &Mesh,
        variable: String,
        target_dual: String,
        coefficient: f64,
    ) -> Result<Self> {
        Ok(Self {
            mesh: share(mesh)?,
            variable,
            target_dual,
            coefficient,
        })
    }
}

/// Multi-point constraint imposed via Lagrange multipliers.
///
/// See the module documentation for the meaning of the two variable names, the
/// mesh-per-term layout and the element-for-element pairing. The right-hand side
/// `g` is **not** stored here: the user supplies it through the load
/// `SubNodeField` at the multiplier node's `imposed_value` component (default
/// `g = 0`).
#[derive(Serialize, Deserialize)]
pub struct Mpc {
    /// The terms summed on the left-hand side of every relation (at least one).
    pub(crate) terms: Vec<MpcTerm>,
    /// POI1 mesh of the multiplier nodes (one node `λ` per relation).
    pub(crate) multiplier_mesh: Mesh,
    /// This sub-model's primal — the Lagrange multiplier (e.g. `"lambda_mpc"`).
    pub(crate) multiplier: String,
    /// This sub-model's dual — constraint row + `g` slot (e.g. `"mpc_rhs"`).
    pub(crate) imposed_value: String,
}

impl Mpc {
    /// Build a multi-point constraint imposing, for each relation `r`,
    /// `Σₖ aₖ·u(nodeₖ, varₖ) = g` — the `r`-th cell of every term mesh paired
    /// with the `r`-th multiplier node.
    ///
    /// `multiplier` / `imposed_value` default to `lambda_mpc` / `mpc_rhs` when
    /// `None`. Every term mesh and `multiplier_mesh` must be POI1, share one
    /// [`Coords`](crate::containers::mesh::Coords), and pair element-for-element
    /// (same number of submeshes, same cell count per submesh).
    pub fn new(
        terms: Vec<MpcTerm>,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        if terms.is_empty() {
            return Err(PyrucastError::Message(
                "Mpc: at least one term is required".into(),
            ));
        }
        if multiplier_mesh.cell_count()? == 0 {
            return Err(PyrucastError::Message(
                "Mpc: multiplier_mesh must carry at least one node".into(),
            ));
        }
        let n_sub = multiplier_mesh.len();
        // NodeIds are Coords-relative: every mesh must share it.
        let coords_m = multiplier_mesh.coords()?;

        // Reference: multiplier submesh element types + cell counts (all POI1).
        let mut mult_counts = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            let mult_sm = multiplier_mesh.get(i)?;
            let (met, mcount) = {
                let s = read(&mult_sm)?;
                (s.element_type(), s.cell_count())
            };
            if met != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Mpc: multiplier_mesh submesh {i} must be POI1, got {met}"
                )));
            }
            mult_counts.push(mcount);
        }

        // Every term: shared Coords, same submesh count, POI1 + equal cell counts.
        for (t_idx, t) in terms.iter().enumerate() {
            let coords_t = t.mesh.coords()?;
            if coords_t.index() != coords_m.index()
                || coords_t.generation() != coords_m.generation()
            {
                return Err(PyrucastError::Message(format!(
                    "Mpc: term {t_idx} mesh and multiplier_mesh must share a Coords"
                )));
            }
            if t.mesh.len() != n_sub {
                return Err(PyrucastError::Message(format!(
                    "Mpc: multiplier_mesh has {n_sub} submesh(es) but term {t_idx} has {}",
                    t.mesh.len()
                )));
            }
            for i in 0..n_sub {
                let term_sm = t.mesh.get(i)?;
                let (tet, tcount) = {
                    let s = read(&term_sm)?;
                    (s.element_type(), s.cell_count())
                };
                if tet != ElementType::POI1 {
                    return Err(PyrucastError::Message(format!(
                        "Mpc: term {t_idx} submesh {i} must be POI1, got {tet}"
                    )));
                }
                if tcount != mult_counts[i] {
                    return Err(PyrucastError::Message(format!(
                        "Mpc: term {t_idx} submesh {i} has {tcount} node(s) but \
                         multiplier has {}",
                        mult_counts[i]
                    )));
                }
            }
        }

        Ok(Self {
            terms,
            multiplier_mesh: share(multiplier_mesh)?,
            multiplier: multiplier.unwrap_or_else(default_multiplier),
            imposed_value: imposed_value.unwrap_or_else(default_imposed_value),
        })
    }
}

impl SubModelKind for Mpc {
    fn primal_vars(&self) -> Vec<String> {
        vec![self.multiplier.clone()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.imposed_value.clone()]
    }

    fn as_constraint(&self) -> Option<&dyn Constraint> {
        Some(self)
    }

    /// Like `Dirichlet`, MPC has no [`stiffness_layout`](SubModelKind::stiffness_layout)
    /// (nothing is integrated on a cell): it returns its filled `C` / `Cᵀ` blocks
    /// directly, one pair per `(submesh, term)`, via the shared
    /// `constraint_block_pair` builder (coefficient `aₖ` instead of `1`). All
    /// terms of a relation share the same multiplier node and `imposed_value` row,
    /// which is what sums them into one equation.
    fn contributions(
        &self,
        _material: Option<&crate::store::Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        let n_sub = self.multiplier_mesh.len();
        let mut blocks = Vec::with_capacity(n_sub * self.terms.len() * 2);
        for i in 0..n_sub {
            let mult_sm = self.multiplier_mesh.get(i)?;
            for t in &self.terms {
                let term_sm = t.mesh.get(i)?;
                let (c_block, ct_block) = constraint_block_pair(
                    &mult_sm,
                    &term_sm,
                    &t.variable,
                    &t.target_dual,
                    &self.multiplier,
                    &self.imposed_value,
                    t.coefficient,
                )?;
                blocks.push(c_block);
                blocks.push(ct_block);
            }
        }
        Ok(vec![Contribution::Literal(blocks)])
    }

    fn label(&self) -> &'static str {
        "Mpc"
    }

    fn display(&self) -> String {
        let n = self.multiplier_mesh.cell_count().unwrap_or(0);
        format!(
            "SubModel<Mpc>: {} relation(s), {} term(s) each",
            n,
            self.terms.len()
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = self.multiplier_mesh.cell_count().unwrap_or(0);
        let mut out = format!(
            "SubModel<Mpc>\n  primal var(s): {} (multiplier)\n  dual var(s):   {} \
             (imposed value / g)\n  relations: {n}\n  terms:",
            self.multiplier, self.imposed_value,
        );
        for t in &self.terms {
            out.push_str(&format!(
                "\n    {:+} · {} (dual {})",
                t.coefficient, t.variable, t.target_dual
            ));
        }
        out
    }
}

impl Constraint for Mpc {
    fn multiplier_mesh(&self) -> &Mesh {
        &self.multiplier_mesh
    }

    /// One [`Relation`] per multiplier node, each with one [`ConstraintTerm`] per
    /// [`MpcTerm`]. Flattens the mesh-per-term layout into the method-neutral view
    /// a future master/slave elimination consumes.
    fn relations(&self) -> Result<Vec<Relation>> {
        let n_sub = self.multiplier_mesh.len();
        let mut relations = Vec::new();
        for i in 0..n_sub {
            let mult_nodes: Vec<NodeId> =
                read(&self.multiplier_mesh.get(i)?)?.connectivity().to_vec();
            // One node list per term for this submesh (relation r = index r).
            let term_nodes: Vec<Vec<NodeId>> = self
                .terms
                .iter()
                .map(|t| Ok(read(&t.mesh.get(i)?)?.connectivity().to_vec()))
                .collect::<Result<Vec<_>>>()?;
            for (r, mult) in mult_nodes.iter().enumerate() {
                let terms = self
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(k, t)| ConstraintTerm {
                        node: term_nodes[k][r],
                        variable: t.variable.clone(),
                        target_dual: t.target_dual.clone(),
                        coefficient: t.coefficient,
                    })
                    .collect();
                relations.push(Relation {
                    multiplier_node: *mult,
                    imposed_value: self.imposed_value.clone(),
                    terms,
                });
            }
        }
        Ok(relations)
    }
}

/// Clone a mesh by sharing its submeshes — increfs each submesh handle so the
/// nodes stay alive for the lifetime of the sub-model.
fn share(mesh: &Mesh) -> Result<Mesh> {
    let mut out = Mesh::empty();
    for sm in mesh {
        out.add_sub(sm.clone())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, Node, SubMesh};
    use crate::ops::mesher::barycenter;
    use crate::store::insert;

    fn poi1(node: &Node) -> Mesh {
        Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node)).unwrap())
    }

    #[test]
    fn empty_terms_rejected() {
        let coords = insert(Coords::new(1).unwrap());
        let n = Node::create_in(coords, &[0.0]).unwrap();
        let mult = barycenter(&poi1(&n)).unwrap();
        assert!(Mpc::new(vec![], &mult, None, None).is_err());
    }

    #[test]
    fn mismatched_cell_count_rejected() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        // Term mesh with two nodes, but the multiplier mesh has one → rejected.
        let mut term_sm = SubMesh::new(coords, ElementType::POI1);
        term_sm.add_cell(&[a.id()]).unwrap();
        term_sm.add_cell(&[b.id()]).unwrap();
        let term_mesh = Mesh::from_submesh(term_sm);
        let mult = barycenter(&poi1(&a)).unwrap();
        let term = MpcTerm::new(&term_mesh, "T".into(), "q".into(), 1.0).unwrap();
        assert!(Mpc::new(vec![term], &mult, None, None).is_err());
    }

    #[test]
    fn relations_flatten_terms() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords, &[1.0]).unwrap();
        let mesh_a = poi1(&a);
        let mesh_b = poi1(&b);
        let mult = barycenter(&mesh_a).unwrap();
        let terms = vec![
            MpcTerm::new(&mesh_a, "T".into(), "q".into(), 2.0).unwrap(),
            MpcTerm::new(&mesh_b, "T".into(), "q".into(), -3.0).unwrap(),
        ];
        let mpc = Mpc::new(terms, &mult, None, None).unwrap();

        // Default variable names.
        assert_eq!(mpc.primal_vars(), vec!["lambda_mpc".to_string()]);
        assert_eq!(mpc.dual_vars(), vec!["mpc_rhs".to_string()]);

        // One relation carrying both terms, in order, with their coefficients.
        let rels = mpc.relations().unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].terms.len(), 2);
        assert_eq!(rels[0].terms[0].node, a.id());
        assert_eq!(rels[0].terms[0].coefficient, 2.0);
        assert_eq!(rels[0].terms[1].node, b.id());
        assert_eq!(rels[0].terms[1].coefficient, -3.0);
    }
}
