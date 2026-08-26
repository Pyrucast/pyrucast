//! Imposed value on a variable (Lagrange multiplier).

use super::single;
use crate::containers::mesh::Mesh;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::RelationSense;

/// Dirichlet `Model` (a single sub-model) constraining `imposed_variable`
/// on the nodes of `imposed_mesh` via Lagrange multipliers carried by
/// `multiplier_mesh`. Parent-level operator — see
/// [`SubModel::dirichlet`] for the semantics of the four variable names
/// and the two meshes.
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
/// // Le modèle complet d'un problème thermique : la physique, puis l'appui.
/// let m = model::heat_conduction(&fes)?.union(
///     &model::dirichlet("T".into(), "q".into(), &impose, &mult,
///                       None, None, RelationSense::Equality)?)?;
/// assert_eq!(m.len(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dirichlet(
    imposed_variable: String,
    target_dual: String,
    imposed_mesh: &Mesh,
    multiplier_mesh: &Mesh,
    multiplier: Option<String>,
    imposed_value: Option<String>,
    sense: RelationSense,
) -> Result<Model> {
    single(SubModel::dirichlet(
        imposed_variable,
        target_dual,
        imposed_mesh,
        multiplier_mesh,
        multiplier,
        imposed_value,
        sense,
    )?)
}
