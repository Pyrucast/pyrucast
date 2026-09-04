//! Node-to-surface unilateral contact.

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;

/// Node-to-surface contact `Model` (a single sub-model) preventing the
/// nodes of `slave` from penetrating the oriented `master` surface.
/// Parent-level operator — see [`SubModel::contact`] and
/// [`contact::Contact::new`](crate::models::contact::Contact::new) for the
/// pairing, the normal coupling and the errors. Solve with
/// [`unilateral`](crate::ops::solver::unilateral).
///
/// ```
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
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let mut maitre = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # maitre.add_cell(&[n[0].id(), n[1].id()])?;
/// # let master = Mesh::from_submesh(maitre);
/// # let slave = mesh::poi1_from_nodes(&n[2..3])?;
/// # let cible = pyrucast::ops::model::elasticity(
/// #     &FiniteElementSpace::lagrange1(&maillage)?,
/// #     pyrucast::models::tensor::Kinematics::PlaneStress)?;
/// let m = model::contact(&cible, &slave, &master,
///     vec!["u_x".into(), "u_y".into()])?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn contact(
    target: &Model,
    slave: &Mesh,
    master: &Mesh,
    variables: Vec<String>,
) -> Result<Model> {
    single(SubModel::contact(target, slave, master, variables)?)
}
