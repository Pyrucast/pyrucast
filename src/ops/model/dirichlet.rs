//! Imposed value on a variable (Lagrange multiplier).

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::RelationSense;

/// Dirichlet `Model` (a single sub-model) constraining `variable`, a primal of
/// `target`, on the nodes of `imposed_mesh` via Lagrange multipliers, one per
/// node on a node of its own. Parent-level operator — see
/// [`SubModel::dirichlet`] and
/// [`Dirichlet::new`](crate::models::dirichlet::Dirichlet::new) for the
/// variable names and the errors.
///
/// The support comes first, then the model it constrains, as for every
/// operator of [`crate::ops::model`]. The Python face defaults `sense` to
/// `RelationSense::Equality`.
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
/// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
/// // The whole model of a thermal problem: the physics, then the support.
/// let m = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?.union(
///     &model::dirichlet(&impose, &cible, "T", RelationSense::Equality)?)?;
/// assert_eq!(m.len(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn dirichlet(
    imposed_mesh: &Mesh,
    target: &Model,
    variable: &str,
    sense: RelationSense,
) -> Result<Model> {
    single(SubModel::dirichlet(imposed_mesh, target, variable, sense)?)
}
