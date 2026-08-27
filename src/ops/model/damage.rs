//! Damage, by law.

use super::spanning;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::damage::law::DamageLaw;
use crate::models::tensor::Kinematics;

/// Damage `Model` spanning **every** subspace of `fes`, with an explicit
/// law. Parent-level operator; the material each law needs is
/// supplied at assembly / integration time.
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
/// # use pyrucast::models::damage::law::DamageLaw;
/// let m = model::damage_with_law(&fes, Kinematics::PlaneStress, DamageLaw::Mazars)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn damage_with_law(
    fes: &FiniteElementSpace,
    kinematics: Kinematics,
    law: DamageLaw,
) -> Result<Model> {
    spanning(fes, |zone| SubModel::damage_with_law(zone, kinematics, law))
}
