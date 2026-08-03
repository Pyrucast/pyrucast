//! Embedded (immersed) constraint via Lagrange multipliers.
//!
//! An `Embedded` sub-model ties the field of every node of an **immersed** mesh
//! (a bar, a cable, a set of points) to the interpolation of a **host** mesh at
//! the same physical location. For each immersed node `p` lodged in the host
//! cell of shape functions `Nᵢ(ξ)`, and for each constrained component `c`,
//!
//! ```text
//! u_c(p) − Σᵢ Nᵢ(ξ_p)·u_c(hostᵢ) = g_c   (default g_c = 0: a rigid tie)
//! ```
//!
//! It is the archetype of a bar « baignée » in a volume: the bar nodes follow
//! the volumic displacement field, without the two meshes sharing nodes. Like
//! [`Dirichlet`](crate::models::dirichlet::Dirichlet) and
//! [`Mpc`](crate::models::mpc::Mpc) it is a *constraint* — no material, no
//! constitutive law — and it **creates no node beyond its multipliers** (the
//! immersed and host nodes are the user's, referenced in place).
//!
//! # Construction
//!
//! The coupling weights `Nᵢ(ξ_p)` are computed **once, at build time**, by
//! locating each immersed node in the host mesh
//! ([`crate::ops::geom::locate_points`]). An immersed node that lands in no
//! host cell is an error: the immersed mesh must lie within the host. The
//! immersed and host meshes must share one
//! [`Coords`](crate::coords::Coords) (their node ids are
//! Coords-relative). Fresh colocated multiplier nodes are minted with the
//! [`barycenter`](crate::ops::mesher::barycenter()) mesher, one per immersed node.
//!
//! # Variables
//!
//! One relation per (immersed node × component). All components share **one**
//! multiplier node per immersed node, each carrying its own multiplier /
//! imposed-value variable:
//!
//! - `variable` (e.g. `"u_x"`) — the constrained primal, a column shared by the
//!   immersed physics and the host physics;
//! - `target_dual` (e.g. `"f_x"`) — the dual (residual) row of that variable,
//!   where the reaction lands (find it with
//!   [`Model::dual_of`](crate::containers::model::Model::dual_of));
//! - `multiplier` (default `lambda_<variable>`) — this sub-model's primal on the
//!   multiplier node (the reaction);
//! - `imposed_value` (default `imposed_<variable>`) — this sub-model's dual: the
//!   constraint row and the slot for the right-hand side `g_c`.

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{
    Constraint, ConstraintTerm, Contribution, MatrixKind, Physics, Relation, RelationSense,
    SubModelKind,
};
use crate::ops::geom::locate_points;
use crate::ops::mesher::barycenter;
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Default reference-domain tolerance for locating immersed nodes in the host.
const DEFAULT_TOL: f64 = 1e-6;

/// Default multiplier (primal) name for a constrained variable: `lambda_<v>`.
pub fn default_multiplier(variable: &str) -> String {
    format!("lambda_{variable}")
}

/// Default imposed-value (dual) name for a constrained variable: `imposed_<v>`.
pub fn default_imposed_value(variable: &str) -> String {
    format!("imposed_{variable}")
}

/// The host binding of one immersed node: the host cell's nodes and the shape
/// weights `Nᵢ(ξ)` at the immersed node, `weights[i]` pairing `nodes[i]`.
#[derive(Serialize, Deserialize)]
struct HostBinding {
    nodes: Vec<NodeId>,
    weights: Vec<f64>,
}

/// One constrained component: its primal variable, dual row, and the two
/// variable names this sub-model owns on the multiplier node.
#[derive(Serialize, Deserialize)]
struct Component {
    variable: String,
    target_dual: String,
    multiplier: String,
    imposed_value: String,
}

/// Embedded (immersed) constraint tying immersed nodes to a host interpolation.
///
/// See the module documentation. The right-hand side `g` defaults to `0` (a
/// rigid tie); a non-zero `g` is supplied through the load `SubNodeField` at the
/// multiplier node's `imposed_value` component, as for `Dirichlet` / `Mpc`.
#[derive(Serialize, Deserialize)]
pub struct Embedded {
    /// POI1 mesh of the immersed nodes, one cell per relation-group (its cell
    /// `r` is the immersed node of relation-group `r`).
    immersed_mesh: Mesh,
    /// POI1 mesh of the fresh multiplier nodes, colocated with the immersed
    /// nodes (paired cell-for-cell).
    multiplier_mesh: Mesh,
    /// POI1 support listing every node the constraint blocks reference — the
    /// immersed nodes and every host node — deduplicated. Row/col support of the
    /// `Cᵀ` / `C` blocks.
    constrained_mesh: Mesh,
    /// Host binding per immersed node (same order as `immersed_mesh`'s cells).
    hosts: Vec<HostBinding>,
    /// Constrained components (at least one).
    components: Vec<Component>,
}

impl Embedded {
    /// Build an embedded constraint tying each node of `immersed` to the host
    /// interpolation of `host` at that node, for every `(variable, target_dual)`
    /// in `components`.
    ///
    /// `multipliers` / `imposed_values`, when `Some`, override the per-component
    /// defaults `lambda_<variable>` / `imposed_<variable>` (must then match
    /// `components` in length). `tol` is the reference-domain slack of the point
    /// location (default `1e-6`).
    ///
    /// # Errors
    ///
    /// - `components` is empty, or an override length mismatches;
    /// - `immersed` and `host` do not share a `Coords`;
    /// - an immersed node lies in no host cell.
    pub fn new(
        immersed: &Mesh,
        host: &Mesh,
        components: Vec<(String, String)>,
        multipliers: Option<Vec<String>>,
        imposed_values: Option<Vec<String>>,
        tol: Option<f64>,
    ) -> Result<Self> {
        if components.is_empty() {
            return Err(PyrucastError::Message(
                "Embedded: at least one component is required".into(),
            ));
        }
        if let Some(m) = &multipliers {
            if m.len() != components.len() {
                return Err(PyrucastError::Message(
                    "Embedded: multipliers length must match components".into(),
                ));
            }
        }
        if let Some(iv) = &imposed_values {
            if iv.len() != components.len() {
                return Err(PyrucastError::Message(
                    "Embedded: imposed_values length must match components".into(),
                ));
            }
        }

        // Immersed and host must live in the same Coords (node ids are relative).
        let coords = immersed.coords()?;
        let host_coords = host.coords()?;
        if coords.index() != host_coords.index() || coords.generation() != host_coords.generation()
        {
            return Err(PyrucastError::Message(
                "Embedded: immersed and host meshes must share a Coords".into(),
            ));
        }

        // The immersed support: one POI1 cell per unique immersed node.
        let immersed_ids = unique_nodes(immersed)?;
        if immersed_ids.is_empty() {
            return Err(PyrucastError::Message(
                "Embedded: immersed mesh carries no node".into(),
            ));
        }
        let immersed_mesh =
            Mesh::from_submesh(SubMesh::poi1_from_node_ids(coords.clone(), &immersed_ids)?);

        // Physical coordinates of the immersed nodes, then locate them.
        let points: Vec<Vec<f64>> = {
            let c = read(&coords)?;
            immersed_ids
                .iter()
                .map(|&n| Ok(c.coord(n)?.to_vec()))
                .collect::<Result<_>>()?
        };
        let tol = tol.unwrap_or(DEFAULT_TOL);
        let located = locate_points(host, &points, tol)?;

        let mut hosts = Vec::with_capacity(immersed_ids.len());
        let mut missing = Vec::new();
        for (i, loc) in located.into_iter().enumerate() {
            match loc {
                Some(l) => hosts.push(HostBinding {
                    nodes: l.nodes,
                    weights: l.weights,
                }),
                None => missing.push(i),
            }
        }
        if !missing.is_empty() {
            return Err(PyrucastError::Message(format!(
                "Embedded: {} immersed node(s) lie outside the host mesh (first at \
                 index {}); the immersed mesh must be contained in the host",
                missing.len(),
                missing[0]
            )));
        }

        // Fresh multiplier nodes, colocated with the immersed nodes.
        let multiplier_mesh = barycenter(&immersed_mesh)?;

        // The block support: immersed nodes ∪ every host node, deduplicated.
        let mut support_ids = immersed_ids.clone();
        let mut seen: HashSet<NodeId> = support_ids.iter().copied().collect();
        for h in &hosts {
            for &n in &h.nodes {
                if seen.insert(n) {
                    support_ids.push(n);
                }
            }
        }
        let constrained_mesh =
            Mesh::from_submesh(SubMesh::poi1_from_node_ids(coords, &support_ids)?);

        // Materialise the components with their (defaulted) variable names.
        let multipliers = multipliers.unwrap_or_else(|| {
            components
                .iter()
                .map(|(v, _)| default_multiplier(v))
                .collect()
        });
        let imposed_values = imposed_values.unwrap_or_else(|| {
            components
                .iter()
                .map(|(v, _)| default_imposed_value(v))
                .collect()
        });
        let components = components
            .into_iter()
            .zip(multipliers)
            .zip(imposed_values)
            .map(
                |(((variable, target_dual), multiplier), imposed_value)| Component {
                    variable,
                    target_dual,
                    multiplier,
                    imposed_value,
                },
            )
            .collect();

        Ok(Self {
            immersed_mesh,
            multiplier_mesh,
            constrained_mesh,
            hosts,
            components,
        })
    }

    /// The single POI1 submesh of multiplier nodes (paired cell-for-cell with
    /// the immersed nodes and the host bindings).
    fn multiplier_sm(&self) -> Result<Handle<SubMesh>> {
        self.multiplier_mesh.get(0)
    }

    /// The single POI1 support submesh of the constraint blocks.
    fn support_sm(&self) -> Result<Handle<SubMesh>> {
        self.constrained_mesh.get(0)
    }

    /// Immersed node of relation-group `r`.
    fn immersed_node(&self, r: usize) -> Result<NodeId> {
        Ok(read(&self.immersed_mesh.get(0)?)?.connectivity()[r])
    }

    /// Multiplier node of relation-group `r`.
    fn multiplier_node(&self, r: usize) -> Result<NodeId> {
        Ok(read(&self.multiplier_mesh.get(0)?)?.connectivity()[r])
    }
}

impl SubModelKind for Embedded {
    fn primal_vars(&self) -> Vec<String> {
        self.components
            .iter()
            .map(|c| c.multiplier.clone())
            .collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        self.components
            .iter()
            .map(|c| c.imposed_value.clone())
            .collect()
    }

    fn as_constraint(&self) -> Option<&dyn Constraint> {
        Some(self)
    }

    /// Like the other constraints, `Embedded` fills its `C` / `Cᵀ` blocks
    /// directly (no `stiffness_layout`). One block pair per component: the
    /// immersed node carries coefficient `+1`, each host node its shape weight
    /// `−Nᵢ`, so every relation reads `u_c(p) − Σᵢ Nᵢ·u_c(hostᵢ) = g_c`.
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
        let n = self.hosts.len();

        let mut blocks = Vec::with_capacity(self.components.len() * 2);
        for comp in &self.components {
            // C: rows (multiplier, imposed_value), cols (constrained, variable).
            let mut c = SubMatrix::new(
                mult_sm.clone(),
                support_sm.clone(),
                vec![comp.imposed_value.clone()],
                vec![comp.variable.clone()],
                DofOrdering::NodesThenVars,
                false,
            )?;
            // Cᵀ: rows (constrained, target_dual), cols (multiplier, multiplier).
            let mut ct = SubMatrix::new(
                support_sm.clone(),
                mult_sm.clone(),
                vec![comp.target_dual.clone()],
                vec![comp.multiplier.clone()],
                DofOrdering::NodesThenVars,
                false,
            )?;
            for r in 0..n {
                let m = self.multiplier_node(r)?;
                let p = self.immersed_node(r)?;
                // +1 on the immersed node.
                c.add_entry(m, &comp.imposed_value, p, &comp.variable, 1.0)?;
                ct.add_entry(p, &comp.target_dual, m, &comp.multiplier, 1.0)?;
                // −Nᵢ on each host node.
                let h = &self.hosts[r];
                for (node, w) in h.nodes.iter().zip(h.weights.iter()) {
                    c.add_entry(m, &comp.imposed_value, *node, &comp.variable, -w)?;
                    ct.add_entry(*node, &comp.target_dual, m, &comp.multiplier, -w)?;
                }
            }
            blocks.push(c);
            blocks.push(ct);
        }
        Ok(vec![Contribution::Literal(blocks)])
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Constraint]
    }

    fn label(&self) -> &'static str {
        "Embedded"
    }

    fn display(&self) -> String {
        format!(
            "SubModel<Embedded>: {} immersed node(s), {} component(s)",
            self.hosts.len(),
            self.components.len()
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let mut out = format!(
            "SubModel<Embedded>\n  immersed nodes: {}\n  components:",
            self.hosts.len()
        );
        for c in &self.components {
            out.push_str(&format!(
                "\n    {} (dual {}, multiplier {}, imposed {})",
                c.variable, c.target_dual, c.multiplier, c.imposed_value
            ));
        }
        out
    }
}

impl Constraint for Embedded {
    fn multiplier_mesh(&self) -> &Mesh {
        &self.multiplier_mesh
    }

    /// One [`Relation`] per (immersed node × component): the immersed term with
    /// coefficient `+1`, then one host term per host node with coefficient
    /// `−Nᵢ`. This is the method-neutral view an elimination path consumes.
    fn relations(&self) -> Result<Vec<Relation>> {
        let n = self.hosts.len();
        let mut relations = Vec::with_capacity(n * self.components.len());
        for comp in &self.components {
            for r in 0..n {
                let m = self.multiplier_node(r)?;
                let p = self.immersed_node(r)?;
                let mut terms = Vec::with_capacity(1 + self.hosts[r].nodes.len());
                terms.push(ConstraintTerm {
                    node: p,
                    variable: comp.variable.clone(),
                    target_dual: comp.target_dual.clone(),
                    coefficient: 1.0,
                });
                let h = &self.hosts[r];
                for (node, w) in h.nodes.iter().zip(h.weights.iter()) {
                    terms.push(ConstraintTerm {
                        node: *node,
                        variable: comp.variable.clone(),
                        target_dual: comp.target_dual.clone(),
                        coefficient: -w,
                    });
                }
                relations.push(Relation {
                    multiplier_node: m,
                    imposed_value: comp.imposed_value.clone(),
                    terms,
                    sense: RelationSense::default(),
                });
            }
        }
        Ok(relations)
    }
}

/// Unique node ids of a mesh, in order of first appearance across submeshes.
fn unique_nodes(mesh: &Mesh) -> Result<Vec<NodeId>> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut out: Vec<NodeId> = Vec::new();
    for sm in mesh {
        for &nid in read(sm)?.connectivity() {
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
    use crate::store::insert;

    /// Build a unit HEX8 host and a two-node bar through its interior; check the
    /// weights and the relation structure.
    fn hex_and_bar() -> (Mesh, Mesh, Handle<Coords>) {
        let coords = insert(Coords::new(3).unwrap());
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let hids: Vec<_> = corners
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut hsm = SubMesh::new(coords.clone(), ElementType::HEX8);
        hsm.add_cell(&hids).unwrap();
        let host = Mesh::from_submesh(hsm);

        // Bar with two interior nodes.
        let b0 = Node::create_in(coords.clone(), &[0.25, 0.5, 0.5]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[0.75, 0.5, 0.5]).unwrap();
        let mut bsm = SubMesh::new(coords.clone(), ElementType::SEG2);
        bsm.add_cell(&[b0.id(), b1.id()]).unwrap();
        let bar = Mesh::from_submesh(bsm);
        (bar, host, coords)
    }

    #[test]
    fn empty_components_rejected() {
        let (bar, host, _c) = hex_and_bar();
        assert!(Embedded::new(&bar, &host, vec![], None, None, None).is_err());
    }

    #[test]
    fn node_outside_host_rejected() {
        let coords = insert(Coords::new(3).unwrap());
        // A degenerate 1-node "host" made of a bar cell far from the immersed pt.
        let h0 = Node::create_in(coords.clone(), &[10.0, 10.0, 10.0]).unwrap();
        let h1 = Node::create_in(coords.clone(), &[11.0, 10.0, 10.0]).unwrap();
        let mut hsm = SubMesh::new(coords.clone(), ElementType::SEG2);
        hsm.add_cell(&[h0.id(), h1.id()]).unwrap();
        let host = Mesh::from_submesh(hsm);
        let p = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let mut bsm = SubMesh::new(coords, ElementType::POI1);
        bsm.add_cell(&[p.id()]).unwrap();
        let bar = Mesh::from_submesh(bsm);
        assert!(Embedded::new(
            &bar,
            &host,
            vec![("u_x".into(), "f_x".into())],
            None,
            None,
            None
        )
        .is_err());
    }

    #[test]
    fn relations_have_immersed_plus_host_terms() {
        let (bar, host, _c) = hex_and_bar();
        let emb = Embedded::new(
            &bar,
            &host,
            vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
            None,
            None,
            None,
        )
        .unwrap();

        // Two components × two immersed nodes = four relations.
        assert_eq!(emb.primal_vars(), vec!["lambda_u_x", "lambda_u_y"]);
        assert_eq!(emb.dual_vars(), vec!["imposed_u_x", "imposed_u_y"]);
        let rels = emb.relations().unwrap();
        assert_eq!(rels.len(), 4);

        // Each relation: +1 immersed term then 8 host terms (HEX8), weights sum 1.
        for rel in &rels {
            assert_eq!(rel.terms.len(), 1 + 8);
            assert_eq!(rel.terms[0].coefficient, 1.0);
            let host_sum: f64 = rel.terms[1..].iter().map(|t| -t.coefficient).sum();
            assert!((host_sum - 1.0).abs() < 1e-9, "host weights sum to 1");
        }

        // Two block pairs (one per component).
        let contribs = emb.contributions(MatrixKind::Stiffness, None).unwrap();
        match &contribs[0] {
            Contribution::Literal(blocks) => assert_eq!(blocks.len(), 4),
            _ => panic!("expected literal blocks"),
        }
    }
}
