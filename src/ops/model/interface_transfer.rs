//! Transfer between two facing interfaces.

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::Physics;

/// Interface-exchange `Model` between two conforming boundary FE spaces —
/// one [`SubModel::InterfaceTransfer`] per subspace pair, taken in order.
/// Parent-level operator; the coefficients `h_<primal>` are
/// supplied at assembly time.
///
/// The two spaces must hold the **same number** of subspaces: an interface
/// pairs zone with zone, and a mismatch is a modelling error, not something
/// to resolve by convention.
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
/// # let mut bord = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # bord.add_cell(&[n[0].id(), n[1].id()])?;
/// # let fes_bord = FiniteElementSpace::lagrange1(&Mesh::from_submesh(bord))?;
/// let m = model::interface_transfer(
///     &fes_bord, &fes_bord, vec![("T".into(), "q".into())], Physics::Thermal, 1e-6)?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn interface_transfer(
    side_a: &FiniteElementSpace,
    side_b: &FiniteElementSpace,
    components: Vec<(String, String)>,
    physics: Physics,
    tol: f64,
) -> Result<Model> {
    if side_a.len() != side_b.len() {
        return Err(PyrucastError::Message(format!(
            "interface_transfer: the two sides must hold the same number of FE subspaces \
             — {} facing {}",
            side_a.len(),
            side_b.len()
        )));
    }
    let mut model = Model::empty();
    for (a, b) in side_a.into_iter().zip(side_b) {
        model.add_sub(Handle::new(SubModel::interface_transfer(
            a.clone(),
            b.clone(),
            components.clone(),
            physics,
            tol,
        )?))?;
    }
    Ok(model)
}
