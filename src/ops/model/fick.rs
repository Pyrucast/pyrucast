//! Fickian diffusion of a named species.

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;
use crate::models::symmetry::MaterialSymmetry;

/// Fickian-diffusion `Model` spanning **every** subspace of `fes`,
/// **isotropic**. Parent-level operator; the diffusivity is
/// supplied at assembly time.
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
/// let m = model::fick(&fes, "H2")?;
/// assert_eq!(m.primal_vars()?, vec!["c_H2".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn fick(fes: &FiniteElementSpace, species: &str) -> Result<Model> {
    fick_with_symmetry(fes, MaterialSymmetry::Isotropic, species)
}

/// Fickian-diffusion `Model` spanning **every** subspace of `fes`, with an
/// explicit material symmetry (the same for all). The diffusivity (`D`, or
/// `D_1…` / `D_11…` plus the material axes) is supplied at assembly time.
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
/// let m = model::fick_with_symmetry(&fes, MaterialSymmetry::Orthotropic, "H2")?;
/// assert_eq!(m.dual_vars()?, vec!["j_H2".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn fick_with_symmetry(
    fes: &FiniteElementSpace,
    symmetry: MaterialSymmetry,
    species: &str,
) -> Result<Model> {
    let mut model = Model::empty();
    for sub in fes {
        model.add_sub(Handle::new(SubModel::fick_with_symmetry(
            sub.clone(),
            symmetry,
            species,
        )?))?;
    }
    Ok(model)
}
