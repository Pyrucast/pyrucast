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
use crate::containers::element_field::ElementField;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::{kernel, MatrixKind, ResidualContribution};

/// External nodal forces of `model` — the right side of `Σ f_int = Σ f_ext`.
///
/// Asks every sub-model for its terms on this side, through
/// [`SubModelKind::external_force_contribution`](crate::models::SubModelKind::external_force_contribution),
/// resolves the material each integrated term reads — exactly as
/// [`crate::ops::matrix::assemble_kind()`] resolves it before asking for a
/// block — drives the kernel and gathers the result. One zone per contributing
/// term, in model order.
///
/// A model whose every term is a response to `u` yields an empty field, which is
/// the honest answer rather than a failure.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::Physics;
/// # use pyrucast::ops::{model, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = ElementField::new(&fes, vec!["k".into()])?;
/// # mat.get(0)?.write().set_uniform("k", 1.0)?;
/// let volume = model::heat_conduction(&fes)?;
/// // La conduction seule n'a aucun terme à droite du signe égal : le champ
/// // revient vide, et c'est la réponse juste.
/// assert_eq!(node_field::external_forces(&volume, &mat)?.len(), 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn external_forces(model: &Model, materials: &ElementField) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for sub_h in model {
        let built = {
            let sub = sub_h.read();
            let kind = sub.as_kind();
            let mut zones = Vec::new();
            for c in kind.external_force_contribution() {
                match c {
                    ResidualContribution::Literal(field) => {
                        zones.extend(field.iter().cloned());
                    }
                    ResidualContribution::Computed(layout) => {
                        // The operator resolves what the term reads, as
                        // `assemble_kind` resolves the material before asking
                        // for a block. The sub-model resolved nothing.
                        let domain = kind.as_domain().ok_or_else(|| {
                            PyrucastError::Message(format!(
                                "{}: declares an integrated external term but no material \
                                 subspace to read its density on",
                                kind.label()
                            ))
                        })?;
                        let mat = materials.sub_for_fespace_with(
                            &domain.material_fespace(),
                            &domain.material_components(),
                        )?;
                        let guard = mat.read();
                        // Resolved once for the zone, before the parallel region.
                        let lay = domain.element_layout(MatrixKind::Stiffness, &guard, None)?;
                        zones.push(Handle::new(kernel::scatter_to_nodes(
                            &layout.fespaces,
                            &layout.support,
                            layout.dual_vars,
                            |geoms, fe| kind.external_force_element(geoms, &guard, &lay, fe),
                        )?));
                    }
                }
            }
            zones
        };
        for zone in built {
            // `r = Σ rᵢ` est une **somme**, pas un empilement : deux termes
            // peuvent charger le même nœud dans la même composante, et une
            // vue d'agrégat en choisirait un au lieu de les ajouter. `+` fait
            // l'union des supports et somme ce qui se recouvre.
            let mut one = NodeField::empty();
            one.add_sub(zone)?;
            out = (&out + &one)?;
        }
    }
    Ok(out)
}
