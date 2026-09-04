//! Multi-point constraint (linear relation between DOFs).

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::mpc::MpcTerm;
use crate::models::RelationSense;

/// Multi-point constraint `Model` (a single sub-model) imposing a linear
/// relation per relation via Lagrange multipliers. Parent-level operator —
/// see [`SubModel::mpc`] and
/// [`mpc::Mpc::new`](crate::models::mpc::Mpc::new) for the mesh-per-term
/// layout and the two variable names.
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
/// # let cible = pyrucast::ops::model::heat_conduction(&fes).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # use pyrucast::models::mpc::MpcTerm;
/// # let a = mesh::poi1_from_nodes(&n[..1])?;
/// # let b = mesh::poi1_from_nodes(&n[1..2])?;
/// let m = model::mpc(
///     vec![MpcTerm::new(&cible, &a, "T", 1.0)?,
///          MpcTerm::new(&cible, &b, "T", -1.0)?],
///     &mult, RelationSense::Equality)?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn mpc(terms: Vec<MpcTerm>, multiplier_mesh: &Mesh, sense: RelationSense) -> Result<Model> {
    single(SubModel::mpc(terms, multiplier_mesh, sense)?)
}
