//! Node-to-surface unilateral contact.

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::containers::node_field::NodeField;
use crate::error::Result;

/// Node-to-surface contact `Model` (a single sub-model) preventing the
/// nodes of `slave` from penetrating the oriented `master` surface.
/// Parent-level operator — see [`SubModel::contact`] and
/// [`contact::Contact::new`](crate::models::contact::Contact::new) for the
/// pairing, the normal coupling and the errors. Solve with
/// [`solve`](crate::ops::solver::solve).
///
/// The supports come first — `slave`, which carries the relations, then
/// `master`, onto which it projects — and the model they constrain after them.
///
/// ```
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::models::{Physics, RelationSense};
/// # use pyrucast::ops::mesh;
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mut maitre = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # maitre.add_cell(&[n[0].id(), n[1].id()])?;
/// # let master = Mesh::from_submesh(maitre);
/// # let slave = mesh::poi1_from_nodes(&n[2..3])?;
/// # let cible = pyrucast::ops::model::elasticity(
/// #     &FiniteElementSpace::lagrange1(&maillage)?,
/// #     pyrucast::models::tensor::Kinematics::PlaneStress, MaterialSymmetry::Isotropic, StrainMeasure::Linear)?;
/// let m = model::contact(&slave, &master, &cible,
///     vec!["u_x".into(), "u_y".into()])?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn contact(
    slave: &Mesh,
    master: &Mesh,
    target: &Model,
    variables: Vec<String>,
) -> Result<Model> {
    single(SubModel::contact(slave, master, target, variables)?)
}

/// The model with its contacts **updated** to the positions `X + u` — `X` the
/// active configuration, `u` read in `displacement` on each contact's
/// variables (a node or a component it lacks reads zero). Every other
/// sub-model is shared as it is; a model without contact comes back unchanged.
///
/// Each slave node is paired again with its closest master facet on `X + u`
/// (projection weights, normal), and the relation becomes the gap linearised
/// there: its right-hand side, the load the contact carries, is
/// `−g̃ = −n·(X_s − Σ Nᵢ X_mᵢ)`, the gap of the new pairing measured on `X`,
/// so that `C·u ≥ −g̃` is written in total displacement from `X` like every
/// other relation. The slave mesh and the **multiplier nodes** are kept: the
/// unknowns `lambda` and the rows `imposed` stay the same, only the
/// coefficients of the rows and their right-hand side change. At `u = 0`, the
/// contact as it was built.
///
/// Its coefficients are the exact derivative of the gap at `X + u`: for a
/// closest-point projection, `δg = n·(δx_s − Σ Nᵢ δx_mᵢ)` — the projected
/// point slides along the facet, orthogonally to `n`, and `n` stays unit.
/// What is neglected is the second order, the geometric stiffness of the
/// contact. A master facet collapsed by the displacement is not detected here:
/// its normal is non-finite.
///
/// ```
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, IMPOSED};
/// # use pyrucast::ops::{mesh, model, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// // A floor of two segments, A(0,0)–B(1,0)–C(2,0), its normal up (each
/// // segment run right to left), and a slave node above A–B.
/// # let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap();
/// # let (a, b, c, s) = (p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(0.5, 1.0));
/// # let mut floor = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # floor.add_cell(&[b.id(), a.id()])?;
/// # floor.add_cell(&[c.id(), b.id()])?;
/// # let master = Mesh::from_submesh(floor);
/// # let slave = mesh::poi1_from_nodes(&[s.clone()])?;
/// # let mut bars = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # bars.add_cell(&[a.id(), s.id()])?;
/// # let cible = model::truss(&FiniteElementSpace::lagrange1(&Mesh::from_submesh(bars))?,
/// #     StrainMeasure::Linear)?;
/// let contact = model::contact(&slave, &master, &cible, vec!["u_x".into(), "u_y".into()])?;
/// # let gap = |m: &pyrucast::containers::model::Model| -> f64 {
/// #     let mult = m.multiplier_mesh().unwrap().node(0, 0, 0).unwrap().id();
/// #     node_field::external_forces(m, &pyrucast::containers::element_field::ElementField::empty())
/// #         .unwrap().value(mult, IMPOSED).unwrap()
/// # };
/// # let terms = |m: &pyrucast::containers::model::Model| -> Vec<_> {
/// #     m.get(0).unwrap().read().as_kind().as_constraint().unwrap().relations().unwrap()[0]
/// #         .terms.iter().filter(|t| t.coefficient != 0.0).map(|t| t.node).collect()
/// # };
/// // At u = 0, the contact as it was built: the gap is 1, on A–B.
/// let rest = NodeField::empty();
/// assert_eq!(gap(&model::update_contact(&contact, &rest)), gap(&contact));
/// // The slave slid by (1, −0.7), to (1.5, 0.3): it now projects on B–C, and
/// // the relation reads the gap of that pairing on the positions at rest.
/// let mut u = NodeField::new(&mesh::poi1_from_nodes(&[s.clone()])?,
///     vec!["u_x".into(), "u_y".into()])?;
/// u.get(0)?.write().set_value(s.id(), "u_x", 1.0)?;
/// u.get(0)?.write().set_value(s.id(), "u_y", -0.7)?;
/// let moved = model::update_contact(&contact, &u);
/// assert!(terms(&moved).contains(&c.id()) && !terms(&moved).contains(&a.id()));
/// assert!((gap(&moved) + 1.0).abs() < 1e-12);
/// // The same multiplier: the same unknown.
/// assert_eq!(moved.multiplier_mesh()?.node(0, 0, 0)?.id(),
///            contact.multiplier_mesh()?.node(0, 0, 0)?.id());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn update_contact(model: &Model, displacement: &NodeField) -> Model {
    model.map_contacts(|contact| contact.updated(displacement))
}
