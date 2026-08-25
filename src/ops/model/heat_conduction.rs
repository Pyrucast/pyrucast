//! Heat conduction (Fourier).

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;
use crate::models::symmetry::MaterialSymmetry;

/// Heat-conduction `Model` spanning **every** subspace of `fes` — one
/// [`SubModel::HeatConduction`] sub-model per
/// [`SubFiniteElementSpace`](crate::containers::finite_element_space::SubFiniteElementSpace).
///
/// This is the parent-level operator (see `CONVENTIONS.md`):
/// it consumes the FE-space *parent* and returns a `Model`, so the
/// caller never builds a `SubModel` by hand. A single-subspace `fes`
/// yields the unit case; several subspaces yield one zone each.
/// Compose heterogeneous physics with `union` (Python `|`), e.g.
/// `model::heat_conduction(&fes)?.union(&model::dirichlet(...)?)?`.
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
/// // Un sous-modèle par sous-espace : le modèle couvre **tout** l'espace EF.
/// let m = model::heat_conduction(&fes)?;
/// assert_eq!(m.len(), fes.len());
/// assert_eq!(m.primal_vars()?, vec!["T".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn heat_conduction(fes: &FiniteElementSpace) -> Result<Model> {
    heat_conduction_with_symmetry(fes, MaterialSymmetry::Isotropic)
}

/// Heat-conduction `Model` spanning **every** subspace of `fes`, with an
/// explicit material symmetry. Parent-level operator; the
/// conductivity (`k`, or `k_1…` / `k_11…` plus the material axes) is supplied
/// at assembly time.
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
/// let m = model::heat_conduction_with_symmetry(&fes, MaterialSymmetry::Orthotropic)?;
/// assert_eq!(m.len(), fes.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn heat_conduction_with_symmetry(
    fes: &FiniteElementSpace,
    symmetry: MaterialSymmetry,
) -> Result<Model> {
    let mut model = Model::empty();
    for sub in fes {
        model.add_sub(Handle::new(SubModel::heat_conduction_with_symmetry(
            sub.clone(),
            symmetry,
        )?))?;
    }
    Ok(model)
}
