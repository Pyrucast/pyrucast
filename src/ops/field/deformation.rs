//! Deformation measure of a displacement field at the Gauss points — the
//! mechanical counterpart of [`crate::ops::field::gradient`](fn@crate::ops::field::gradient).
//!
//! Only the **linearized** (small-strain) measure `ε = ½(∇u + ∇uᵀ)` is
//! implemented for now; non-linear measures (Green–Lagrange, …) will join it
//! here under the same `(displacement, fespace) → ElementField` shape, so a
//! caller chooses *which* deformation to feed
//! [`crate::ops::behavior::integrate`].

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::Field;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::ops::field::gradient::AXES;
use crate::store::{insert, read};

/// Linearized (small-strain) deformation `ε = ½(∇u + ∇uᵀ)` of a displacement
/// field `u` at the Gauss points of every subspace of `fespace`.
///
/// `u` must carry exactly `space_dim` components, taken **in order** as the
/// displacement along x, y, z. The result is the symmetric strain tensor in
/// **tensor** convention (`eps_xy = ½(∂u_x/∂y + ∂u_y/∂x)`, *not* engineering
/// shear `γ`), with one component `eps_<ai><aj>` per independent entry
/// `i ≤ j`, in order `eps_xx, eps_xy, …, eps_yy, …`.
///
/// Runs on the shared parallel driver `models::kernel::nodal_pointwise`,
/// like [`crate::ops::field::gradient`](fn@crate::ops::field::gradient).
pub fn deformation(u: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(u)?;
    let view = u.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        let space_dim = read(sub)?.space_dim();
        if components.len() != space_dim {
            return Err(PyrucastError::Message(format!(
                "deformation: the displacement field carries {} component(s) but the FE \
                 space is {}-D — expected exactly one displacement component per axis",
                components.len(),
                space_dim
            )));
        }
        // Independent strain entries eps_<ai><aj> for i ≤ j, in the order
        // eps_xx, eps_xy, …, eps_yy, … — matching the tensor convention.
        let mut names = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        let mut pairs = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        for i in 0..space_dim {
            for j in i..space_dim {
                names.push(format!("eps_{}{}", AXES[i], AXES[j]));
                pairs.push((i, j));
            }
        }
        // Point kernel: ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i) with ∂u_i/∂x_j = Σ_k u_i(k)·∂N_k/∂x_j.
        let sf = kernel::nodal_pointwise(sub, &view, names, |geom, field, g, out| {
            let dn_dx = geom.dn_dx(g)?;
            let ids = geom.node_ids();
            let sd = geom.space_dim;
            for (c, &(i, j)) in pairs.iter().enumerate() {
                let mut dij = 0.0; // ∂u_i/∂x_j
                let mut dji = 0.0; // ∂u_j/∂x_i
                for k in 0..geom.n_nodes {
                    let u_i = field.value(ids[k], &components[i])?;
                    let u_j = field.value(ids[k], &components[j])?;
                    dij += u_i * dn_dx[k * sd + j];
                    dji += u_j * dn_dx[k * sd + i];
                }
                out[c] = 0.5 * (dij + dji);
            }
            Ok(())
        })?;
        out.add_sub(insert(sf))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::store::{insert, read};

    /// Linear displacement `u_x = 2x + 0.5y`, `u_y = 0.1x + 3y` on a TRI3.
    /// ⇒ ε_xx = 2, ε_yy = 3, ε_xy = ½(0.5 + 0.1) = 0.3.
    #[test]
    fn linear_displacement_gives_constant_strain() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        // u_x = 2x + 0.5y, u_y = 0.1x + 3y at the three nodes.
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 2.0 * x + 0.5 * y).unwrap();
            u.set_value(n.id(), "u_y", 0.1 * x + 3.0 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        {
            let s = read(&strain.get(0).unwrap()).unwrap();
            assert_eq!(
                s.components(),
                &[
                    "eps_xx".to_string(),
                    "eps_xy".to_string(),
                    "eps_yy".to_string()
                ]
            );
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "eps_xx").unwrap() - 2.0).abs() < 1e-12);
                assert!((s.value(0, g, "eps_yy").unwrap() - 3.0).abs() < 1e-12);
                assert!((s.value(0, g, "eps_xy").unwrap() - 0.3).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rejects_displacement_with_wrong_component_count() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // 2-D space but a single displacement component.
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let u = NodeField::from_sub(SubNodeField::from_poi1(&support, vec!["u_x".into()]).unwrap());
        let err = deformation(&u, &fes).unwrap_err();
        assert!(format!("{err}").contains("one displacement component per axis"));
    }
}
