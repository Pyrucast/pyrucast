//! External nodal forces of a `Model` — the **right** side of the balance
//! `Σ f_int = Σ f_ext`.
//!
//! The counterpart of
//! [`crate::ops::node_field::internal_forces`](fn@crate::ops::node_field::internal_forces),
//! and with it the nodal mirror of [`crate::ops::matrix`]: where an assembly
//! asks every sub-model for its blocks of `∂r/∂u`, these two ask every
//! sub-model for its term of `r`, on one side of the equals sign or the other.
//!
//! Splitting the two sides is what keeps signs out of the physics files. An
//! author writes `∫ Bᵀ σ` on the internal side and `∫ N φ` on the external one,
//! both positive, exactly as the weak form reads; the single subtraction that
//! turns them into a residual lives in the caller. Whether a term belongs left
//! or right is a question of physics — the side of the equals sign it sits on —
//! not of bookkeeping.
//!
//! A physics whose term is entirely a response to `u` (elasticity, conduction,
//! a bar) has nothing here and declares nothing. The ambient of a boundary
//! transfer and a distributed flux load do.

use crate::aggregate::Aggregate;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::Result;

/// External nodal forces of `model` — the right side of `Σ f_int = Σ f_ext`.
///
/// Asks every sub-model for its terms on this side, through
/// [`SubModelKind::external_force_contribution`](crate::models::SubModelKind::external_force_contribution),
/// and gathers them into one [`NodeField`], one zone per contributing term in
/// model order. A model whose every term is a response to `u` yields an empty
/// field — which is the honest answer, not a failure.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{model, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// let modele = model::heat_conduction(&fes)?;
/// // La conduction seule n'a aucun terme à droite du signe égal : le champ
/// // extérieur est vide, et c'est la réponse juste.
/// assert_eq!(node_field::external_forces(&modele)?.len(), 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn external_forces(model: &Model) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for h in model {
        let contribution = h.read().as_kind().external_force_contribution();
        for sub in &contribution {
            out.add_sub(sub.clone())?;
        }
    }
    Ok(out)
}
