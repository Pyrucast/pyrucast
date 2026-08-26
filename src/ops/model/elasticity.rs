//! Linear elasticity.

use super::spanning;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::elasticity::ElasticityModel;
use crate::models::symmetry::MaterialSymmetry;

/// Linear-elasticity `Model` spanning **every** subspace of `fes` (same
/// 2-D/3-D `model` for all). Parent-level operator; material
/// (`E`, `nu`) is supplied at assembly time.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::ElasticityModel;
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
/// let m = model::elasticity(&fes, ElasticityModel::PlaneStress)?;
/// assert_eq!(m.dual_vars()?, vec!["f_x".to_string(), "f_y".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elasticity(fes: &FiniteElementSpace, model: ElasticityModel) -> Result<Model> {
    elasticity_with_symmetry(fes, model, MaterialSymmetry::Isotropic)
}

/// Linear-elasticity `Model` spanning **every** subspace of `fes`, with an
/// explicit material symmetry. Parent-level operator; the elastic
/// constants and, for an oriented material, its axes are supplied at assembly
/// time.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::ElasticityModel;
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
/// let m = model::elasticity_with_symmetry(
///     &fes, ElasticityModel::PlaneStress, MaterialSymmetry::Orthotropic)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elasticity_with_symmetry(
    fes: &FiniteElementSpace,
    model: ElasticityModel,
    symmetry: MaterialSymmetry,
) -> Result<Model> {
    spanning(fes, |zone| {
        SubModel::elasticity_with_symmetry(zone, model, symmetry)
    })
}
