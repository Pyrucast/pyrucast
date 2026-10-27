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
//! constitutive law. It is the **generalisation of
//! [`Dirichlet`](crate::models::dirichlet::Dirichlet)**, which is the single-term
//! relation `1·u = u_d`.
//!
//! # Mesh-per-term layout
//!
//! Each term is a `(mesh, variable, coefficient)` triple, its mesh POI1. All
//! term meshes are paired **element-for-element**: relation `r` links the
//! `r`-th cell of *every* term mesh. So a periodicity between two surfaces of
//! `N` nodes is `N` relations at once, vectorised over the cells (no per-node
//! loop). Consistent ordering of the paired meshes is the **user's
//! responsibility** (cell `r` of each mesh must be geometric partners).
//!
//! Each relation gets a multiplier node of its own, minted at construction by
//! the [`barycenter`](crate::ops::mesh::barycenter()) mesher beside the node of
//! the first term.
//!
//! # Variables
//!
//! Like `Dirichlet`, an MPC owns the pair of variables of every constraint
//! relation:
//!
//! - [`MULTIPLIER`] — this sub-model's **primal**: the Lagrange multiplier `λ`,
//!   an unknown of the augmented system, whose solved value is the
//!   **reaction**;
//! - [`IMPOSED`] — this sub-model's **dual**: the constraint-equation row, and
//!   the **slot** at which the right-hand side `g` is written in the load
//!   `SubNodeField` (on the multiplier node).
//!
//! Each term's variable is a primal of the target model, whose dual — the row
//! into which the reaction `aₖ·λ` is added — is read from it.

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::containers::mesh::Mesh;
use crate::containers::model::Model;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::transfer::{constrained_physics, constraint_physics};
use crate::models::{
    constraint_block_pair, Constraint, ConstraintTerm, Contribution, MatrixKind, Physics, Relation,
    RelationSense, SubModelKind, IMPOSED, MULTIPLIER,
};
use crate::ops::mesh::barycenter;
use serde::{Deserialize, Serialize};

/// One term `coefficient · u(node, variable)` of every relation of an [`Mpc`],
/// as stored: its POI1 mesh (cell `r` holds the node of relation `r`), its
/// variable and the target's dual row its reaction lands in.
#[derive(Serialize, Deserialize)]
struct Term {
    mesh: Mesh,
    /// Constrained primal variable (a *column* of the target's stiffness), e.g. `"u_x"`.
    variable: String,
    /// The target's dual of `variable`; the *row* where the reaction
    /// `coefficient·λ` lands (e.g. `"f_x"`).
    target_dual: String,
    coefficient: f64,
}

/// Multi-point constraint imposed via Lagrange multipliers.
///
/// See the module documentation for the meaning of the two variable names and
/// the mesh-per-term layout. The right-hand side `g` is **not** stored here:
/// the user supplies it through the load `SubNodeField` at the multiplier
/// node's [`IMPOSED`] component (default `g = 0`).
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # use pyrucast::models::mpc::Mpc;
/// # let a = mesh::poi1_from_nodes(&n[..1])?;
/// # let b = mesh::poi1_from_nodes(&n[1..2])?;
/// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
/// // "Both nodes have the same temperature": Σ aₖ·uₖ = g, with one λ per
/// // relation.
/// let m = Mpc::new(&cible, &[(&a, "T", 1.0), (&b, "T", -1.0)], RelationSense::Equality)?;
/// assert_eq!(m.relations()?.len(), 1);
/// assert_eq!(m.relations()?[0].terms.len(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize)]
pub struct Mpc {
    /// The terms summed on the left-hand side of every relation (at least one).
    terms: Vec<Term>,
    /// POI1 mesh of the multiplier nodes, minted at construction (one node `λ`
    /// per relation).
    multiplier_mesh: Mesh,
    /// Equality (default) or unilateral inequality (`Σ aₖ·uₖ ≥ g` / `≤ g`),
    /// solved by the active-set operator.
    sense: RelationSense,
    /// Nature shared by every term's physics, `None` when the relations tie
    /// several physics together.
    target_physics: Option<Physics>,
}

impl Mpc {
    /// Build a multi-point constraint on `target` imposing, for each relation
    /// `r`, `Σₖ aₖ·u(nodeₖ, varₖ) ⋈ g` (`⋈` is `sense`, `=` by default) — the
    /// `r`-th cell of every term mesh, each term given as
    /// `(mesh, variable, coefficient)`.
    ///
    /// Every variable is a primal of `target`, whose dual row is read from it;
    /// to tie two bodies modelled apart, pass their union (`a | b`). Every term
    /// mesh must be POI1, share one [`Coords`](crate::coords::Coords), and pair
    /// element-for-element (same number of submeshes, same cell count per
    /// submesh).
    ///
    /// A non-equality `sense` turns the relations unilateral (enforced only
    /// while active); such a model is solved by the active-set operator
    /// [`unilateral`](crate::ops::solver::unilateral).
    ///
    /// ```
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # use pyrucast::models::mpc::Mpc;
    /// # let a = mesh::poi1_from_nodes(&n[..1])?;
    /// # let b = mesh::poi1_from_nodes(&n[1..2])?;
    /// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
    /// let m = Mpc::new(&cible, &[(&a, "T", 1.0), (&b, "T", -1.0)], RelationSense::Equality)?;
    /// // One multiplier node per relation, minted beside the first term's node.
    /// assert_eq!(m.multiplier_mesh().node(0, 0, 0)?.position()?, vec![0.0, 0.0]);
    /// // A variable the target does not assemble is refused.
    /// assert!(Mpc::new(&cible, &[(&a, "u_x", 1.0)], RelationSense::Equality).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(target: &Model, terms: &[(&Mesh, &str, f64)], sense: RelationSense) -> Result<Self> {
        let Some(&(first, _, _)) = terms.first() else {
            return Err(PyrucastError::Message(
                "Mpc: at least one term is required".into(),
            ));
        };
        if first.cell_count() == 0 {
            return Err(PyrucastError::Message(
                "Mpc: the terms must carry at least one relation".into(),
            ));
        }
        let n_sub = first.len();
        // NodeIds are Coords-relative: every mesh must share it.
        let coords = first.coords()?;
        let counts: Vec<usize> = first.into_iter().map(|sm| sm.read().cell_count()).collect();

        // Every term: a primal of the target, shared Coords, same submesh count,
        // POI1 + equal cell counts.
        let mut stored = Vec::with_capacity(terms.len());
        for (t_idx, &(mesh, variable, coefficient)) in terms.iter().enumerate() {
            let target_dual = target.dual_of(variable).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "Mpc: term {t_idx}: `{variable}` is not a primal of the model it \
                     constrains — it declares {:?}",
                    target.primal_vars()
                ))
            })?;
            if !mesh.coords()?.same_object(&coords) {
                return Err(PyrucastError::Message(format!(
                    "Mpc: term {t_idx} mesh must share the Coords of term 0"
                )));
            }
            if mesh.len() != n_sub {
                return Err(PyrucastError::Message(format!(
                    "Mpc: term 0 has {n_sub} submesh(es) but term {t_idx} has {}",
                    mesh.len()
                )));
            }
            for (i, term_sm) in mesh.into_iter().enumerate() {
                let (tet, tcount) = {
                    let s = term_sm.read();
                    (s.element_type(), s.cell_count())
                };
                if tet != ElementType::POI1 {
                    return Err(PyrucastError::Message(format!(
                        "Mpc: term {t_idx} submesh {i} must be POI1, got {tet}"
                    )));
                }
                if tcount != counts[i] {
                    return Err(PyrucastError::Message(format!(
                        "Mpc: term {t_idx} submesh {i} has {tcount} node(s) but term 0 has {}",
                        counts[i]
                    )));
                }
            }
            stored.push(Term {
                mesh: share(mesh)?,
                variable: variable.to_string(),
                target_dual,
                coefficient,
            });
        }

        // One nature when every term constrains the same physics; a relation
        // across two physics belongs to neither.
        let target_physics = constrained_physics(target, terms.iter().map(|&(_, v, _)| v));

        Ok(Self {
            terms: stored,
            // One fresh node per relation, beside the first term's node.
            multiplier_mesh: barycenter(first)?,
            sense,
            target_physics,
        })
    }
}

impl SubModelKind for Mpc {
    fn primal_vars(&self) -> Vec<String> {
        vec![MULTIPLIER.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![IMPOSED.to_string()]
    }

    /// Its relations, applied to the solution: the reaction `Cᵀ λ` on the
    /// constrained nodes, and `C·u` on its own row.
    fn internal_force_contribution(&self) -> Vec<crate::models::ResidualContribution> {
        vec![crate::models::ResidualContribution::Relations]
    }

    fn as_constraint(&self) -> Option<&dyn Constraint> {
        Some(self)
    }

    /// Like `Dirichlet`, MPC has no [`stiffness_layout`](SubModelKind::stiffness_layout)
    /// (nothing is integrated on a cell): it returns its filled `C` / `Cᵀ` blocks
    /// directly, one pair per `(submesh, term)`, via the shared
    /// `constraint_block_pair` builder (coefficient `aₖ` instead of `1`). All
    /// terms of a relation share the same multiplier node, hence the same
    /// [`IMPOSED`] row, which is what sums them into one equation.
    fn contributions(
        &self,
        kind: MatrixKind,
        _material: Option<&crate::handle::Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        // A constraint only enters the global (stiffness) matrix — no mass or
        // geometric term. A tangent request reaches it as a stiffness one
        // (`tangent_is_stiffness`), so it never arrives here.
        if kind != MatrixKind::Stiffness {
            return Ok(Vec::new());
        }
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
                    t.coefficient,
                )?;
                blocks.push(c_block);
                blocks.push(ct_block);
            }
        }
        Ok(vec![Contribution::Literal(blocks)])
    }

    /// Linear relations: their tangent blocks **are** their stiffness blocks.
    fn tangent_is_stiffness(&self) -> bool {
        true
    }

    fn physics(&self) -> &'static [Physics] {
        constraint_physics(self.target_physics)
    }

    fn label(&self) -> &'static str {
        "Mpc"
    }

    fn display(&self) -> String {
        let n = self.multiplier_mesh.cell_count();
        format!(
            "SubModel<Mpc>: {} relation(s), {} term(s) each",
            n,
            self.terms.len()
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = self.multiplier_mesh.cell_count();
        let mut out = format!(
            "SubModel<Mpc>\n  primal var(s): {MULTIPLIER} (multiplier)\n  dual var(s):   \
             {IMPOSED} (imposed value / g)\n  sense: {}\n  relations: {n}\n  terms:",
            self.sense,
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
    /// term. Flattens the mesh-per-term layout into the method-neutral view the
    /// elimination path consumes.
    fn relations(&self) -> Result<Vec<Relation>> {
        let n_sub = self.multiplier_mesh.len();
        let mut relations = Vec::new();
        for i in 0..n_sub {
            let mult = self.multiplier_mesh.get(i)?.read();
            // One **guard** per term for this submesh, all held together, so the
            // relation at index r reads the r-th node of each in place. Two terms
            // may name the same submesh: two read guards on one object, which the
            // lock allows (see `handle.rs`, Concurrency).
            let term_guards: Vec<crate::handle::ReadGuard<crate::containers::mesh::SubMesh>> = self
                .terms
                .iter()
                .map(|t| Ok(t.mesh.get(i)?.read()))
                .collect::<Result<Vec<_>>>()?;
            for (r, mult_node) in mult.connectivity().iter().enumerate() {
                let terms = self
                    .terms
                    .iter()
                    .zip(&term_guards)
                    .map(|(t, g)| ConstraintTerm {
                        node: g.connectivity()[r],
                        variable: t.variable.clone(),
                        target_dual: t.target_dual.clone(),
                        coefficient: t.coefficient,
                    })
                    .collect();
                relations.push(Relation {
                    multiplier_node: *mult_node,
                    terms,
                    sense: self.sense,
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
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::models::symmetry::MaterialSymmetry;

    /// A conduction on a segment: the target a relation constrains.
    fn cible_conduction(coords: &Handle<Coords>, a: &Node) -> Model {
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = crate::containers::finite_element_space::FiniteElementSpace::lagrange1(
            &Mesh::from_submesh(sm),
        )
        .unwrap();
        crate::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap()
    }

    fn poi1(node: &Node) -> Mesh {
        Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node)).unwrap())
    }

    #[test]
    fn empty_terms_rejected() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let cible = cible_conduction(&coords, &n);
        assert!(Mpc::new(&cible, &[], Default::default()).is_err());
    }

    #[test]
    fn mismatched_cell_count_rejected() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        // Second term with two nodes, but the first has one → rejected.
        let mut term_sm = SubMesh::new(coords.clone(), ElementType::POI1);
        term_sm.add_cell(&[a.id()]).unwrap();
        term_sm.add_cell(&[b.id()]).unwrap();
        let two = Mesh::from_submesh(term_sm);
        let one = poi1(&a);
        let cible = cible_conduction(&coords, &a);
        assert!(Mpc::new(
            &cible,
            &[(&one, "T", 1.0), (&two, "T", 1.0)],
            Default::default()
        )
        .is_err());
    }

    #[test]
    fn relations_flatten_terms() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mesh_a = poi1(&a);
        let mesh_b = poi1(&b);
        let cible = cible_conduction(&coords, &a);
        let mpc = Mpc::new(
            &cible,
            &[(&mesh_a, "T", 2.0), (&mesh_b, "T", -3.0)],
            Default::default(),
        )
        .unwrap();

        // The names of every constraint relation.
        assert_eq!(mpc.primal_vars(), vec![MULTIPLIER.to_string()]);
        assert_eq!(mpc.dual_vars(), vec![IMPOSED.to_string()]);

        // One relation carrying both terms, in order, with their coefficients,
        // and a multiplier node of its own.
        let rels = mpc.relations().unwrap();
        assert_eq!(rels.len(), 1);
        assert_ne!(rels[0].multiplier_node, a.id());
        assert_ne!(rels[0].multiplier_node, b.id());
        assert_eq!(rels[0].terms.len(), 2);
        assert_eq!(rels[0].terms[0].node, a.id());
        assert_eq!(rels[0].terms[0].coefficient, 2.0);
        assert_eq!(rels[0].terms[1].node, b.id());
        assert_eq!(rels[0].terms[1].coefficient, -3.0);
    }
}
