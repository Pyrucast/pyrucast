//! Multi-point constraint (linear relation between DOFs).

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::RelationSense;

/// Multi-point constraint `Model` (a single sub-model) on `target`, imposing
/// one linear relation per cell of the term meshes via Lagrange multipliers,
/// each term a `(mesh, variable, coefficient)` triple. Parent-level operator —
/// see [`SubModel::mpc`] and
/// [`mpc::Mpc::new`](crate::models::mpc::Mpc::new) for the mesh-per-term
/// layout, the variable names and the errors.
///
/// The terms carry the support, so they come first, then the model they
/// constrain. The Python face defaults `sense` to `RelationSense::Equality`.
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
/// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let a = mesh::poi1_from_nodes(&n[..1])?;
/// # let b = mesh::poi1_from_nodes(&n[1..2])?;
/// let m = model::mpc(&[(&a, "T", 1.0), (&b, "T", -1.0)], &cible, RelationSense::Equality)?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn mpc(terms: &[(&Mesh, &str, f64)], target: &Model, sense: RelationSense) -> Result<Model> {
    single(SubModel::mpc(terms, target, sense)?)
}
