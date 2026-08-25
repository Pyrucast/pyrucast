//! Damage, by law.

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;
use crate::models::damage::DamageLaw;
use crate::models::elasticity::ElasticityModel;

/// Mazars-damage `Model` spanning **every** subspace of `fes` (same 2-D/3-D
/// `model` for all). Parent-level operator; material
/// (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at assembly
/// / integration time.
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
/// let m = model::mazars(&fes, ElasticityModel::PlaneStress)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn mazars(fes: &FiniteElementSpace, model: ElasticityModel) -> Result<Model> {
    damage_with_law(fes, model, DamageLaw::Mazars)
}

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
/// # use pyrucast::models::damage::DamageLaw;
/// let m = model::damage_with_law(&fes, ElasticityModel::PlaneStress, DamageLaw::Mazars)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn damage_with_law(
    fes: &FiniteElementSpace,
    model: ElasticityModel,
    law: DamageLaw,
) -> Result<Model> {
    let mut out = Model::empty();
    for sub in fes {
        out.add_sub(Handle::new(SubModel::damage_with_law(
            sub.clone(),
            model,
            law,
        )?))?;
    }
    Ok(out)
}
