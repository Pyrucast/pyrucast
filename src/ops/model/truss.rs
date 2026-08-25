//! Bar (truss) elements.

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;

/// Truss / bar `Model` spanning **every** subspace of `fes` — one
/// [`SubModel::Truss`] per
/// [`SubFiniteElementSpace`](crate::containers::finite_element_space::SubFiniteElementSpace).
/// Parent-level operator; material (`E`, `A`) is supplied at assembly time.
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
/// # let mut b = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # b.add_cell(&[n[0].id(), n[1].id()])?;
/// # let barres = FiniteElementSpace::lagrange1(&Mesh::from_submesh(b))?;
/// let m = model::truss(&barres)?;
/// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn truss(fes: &FiniteElementSpace) -> Result<Model> {
    let mut model = Model::empty();
    for sub in fes {
        model.add_sub(Handle::new(SubModel::truss(sub.clone())?))?;
    }
    Ok(model)
}
