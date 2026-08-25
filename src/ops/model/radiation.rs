//! Radiative exchange on a boundary.

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;

/// Radiation-to-infinity `Model` spanning **every** subspace of a *boundary*
/// `fes`. Parent-level operator; the emissivity and far-field
/// temperature are supplied at assembly time.
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
/// // Le rayonnement s'unione à la conduction : mêmes DDL, donc mêmes blocs.
/// let m = model::heat_conduction(&fes)?.union(&model::radiation(&fes)?)?;
/// assert_eq!(m.primal_vars()?, vec!["T".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn radiation(fes: &FiniteElementSpace) -> Result<Model> {
    let mut model = Model::empty();
    for sub in fes {
        model.add_sub(Handle::new(SubModel::radiation(sub.clone())?))?;
    }
    Ok(model)
}
