//! Immersed nodes bound to a host interpolation.

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;

/// Embedded (immersed) constraint `Model` (a single sub-model) tying each
/// node of `immersed` to the interpolation of `host` at that node, for every
/// `(variable, target_dual)` in `components`. Parent-level operator
/// — see [`SubModel::embedded`] and
/// [`embedded::Embedded::new`](crate::models::embedded::Embedded::new) for the
/// coupling weights, the per-component variable names and the errors.
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
/// # let mut barre = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # let p: Vec<_> = [[0.25, 0.25], [0.5, 0.25]].iter()
/// #     .map(|q| Node::create_in(coords.clone(), q).unwrap()).collect();
/// # barre.add_cell(&[p[0].id(), p[1].id()])?;
/// # let immergee = Mesh::from_submesh(barre);
/// let m = model::embedded(
///     &immergee, &maillage,
///     vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
///     None, None, None)?;
/// assert_eq!(m.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn embedded(
    immersed: &Mesh,
    host: &Mesh,
    components: Vec<(String, String)>,
    multipliers: Option<Vec<String>>,
    imposed_values: Option<Vec<String>>,
    tol: Option<f64>,
) -> Result<Model> {
    single(SubModel::embedded(
        immersed,
        host,
        components,
        multipliers,
        imposed_values,
        tol,
    )?)
}
