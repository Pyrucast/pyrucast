//! Shells (Kirchhoff-Love / Reissner-Mindlin).

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;
use crate::models::shell::ShellModel;

/// Shell `Model` spanning **every** subspace of a *surface* `fes`.
/// Parent-level operator; material is supplied at assembly time.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::shell::ShellModel;
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// let m = model::shell(&fes, ShellModel::Thick)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn shell(fes: &FiniteElementSpace, model: ShellModel) -> Result<Model> {
    let mut out = Model::empty();
    for sub in fes {
        out.add_sub(Handle::new(SubModel::shell(sub.clone(), model)?))?;
    }
    Ok(out)
}
