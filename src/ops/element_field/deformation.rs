//! Deformation measure of a displacement field at the Gauss points — the
//! mechanical counterpart of [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient).
//!
//! Only the **linearized** (small-strain) measure `ε = ½(∇u + ∇uᵀ)` is
//! implemented for now; non-linear measures (Green–Lagrange, …) will join it
//! here under the same `(displacement, fespace) → ElementField` shape, so a
//! caller chooses *which* deformation to feed
//! [`crate::ops::element_field::behavior::integrate`].

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::Field;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;
use crate::ops::element_field::gradient::AXES;

/// Linearized (small-strain) deformation `ε = ½(∇u + ∇uᵀ)` of a displacement
/// field `u` at the Gauss points of every subspace of `fespace`.
///
/// `u` must carry exactly `space_dim` components, taken **in order** as the
/// displacement along x, y, z. The result is the symmetric strain tensor in
/// **tensor** convention (`eps_xy = ½(∂u_x/∂y + ∂u_y/∂x)`, *not* engineering
/// shear `γ`), with one component `eps_<ai><aj>` per independent entry
/// `i ≤ j`, in order `eps_xx, eps_xy, …, eps_yy, …`.
///
/// On an [axisymmetric](crate::coords::Coords::axisymmetric) subspace
/// a fourth component `eps_zz` is appended: the **hoop** strain
/// `ε_θθ = u_r / r`, which the meridian gradient cannot express.
///
/// Runs on the shared parallel driver `models::kernel::nodal_pointwise`,
/// like [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // ε = ½(∇u + ∇uᵀ) aux points de Gauss. Une **translation** d'ensemble
/// // ne déforme rien.
/// let u = NodeField::from_submesh(&support.get(0)?, vec!["u_x".into(), "u_y".into()])?;
/// u.get(0)?.write().add_to_component("u_x", 0.5)?;
/// let eps = element_field::deformation(&u, &fes)?;
/// assert!(eps.get(0)?.read().value(0, 0, "eps_xx")?.abs() < 1e-12);
/// // Un étirement uniforme en x, lui, se lit tel quel : u_x = 0,1·x donne
/// // ε_xx = 0,1.
/// # let x = node_field::positions(&maillage, None)?;
/// let etire = NodeField::from_submesh(&support.get(0)?, vec!["u_x".into(), "u_y".into()])?;
/// for i in 0..3 {
///     let v = x.get(0)?.read().value(n[i].id(), "X")? * 0.1;
///     etire.get(0)?.write().set_value(n[i].id(), "u_x", v)?;
/// }
/// let eps = element_field::deformation(&etire, &fes)?;
/// assert!((eps.get(0)?.read().value(0, 0, "eps_xx")? - 0.1).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn deformation(u: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(u)?;
    let view = u.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        let (space_dim, axisymmetric) = {
            let s = sub.read();
            (s.space_dim(), s.is_axisymmetric())
        };
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
        // On a body of revolution the hoop strain ε_θθ = u_r / r is a fourth,
        // independent component. It is appended last and named `eps_zz` (Cast3M's
        // naming, x = r and y = z), which is free of clashes since the meridian
        // plane only spans xx, xy and yy. Consumers read by name.
        if axisymmetric {
            names.push("eps_zz".to_string());
        }
        // Point kernel: ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i) with ∂u_i/∂x_j = Σ_k u_i(k)·∂N_k/∂x_j.
        // `dofs` holds the cell's displacements, node-major: the driver
        // gathered them once, so the Gauss loop is arithmetic alone.
        let nc = components.len();
        let sf = kernel::nodal_pointwise(sub, &view, &components, names, |geom, g, dofs, out| {
            let dn_dx = geom.dn_dx(g)?;
            let sd = geom.space_dim;
            for (c, &(i, j)) in pairs.iter().enumerate() {
                let mut dij = 0.0; // ∂u_i/∂x_j
                let mut dji = 0.0; // ∂u_j/∂x_i
                for k in 0..geom.n_nodes {
                    dij += dofs[k * nc + i] * dn_dx[k * sd + j];
                    dji += dofs[k * nc + j] * dn_dx[k * sd + i];
                }
                out[c] = 0.5 * (dij + dji);
            }
            if axisymmetric {
                // ε_θθ = u_r / r, with u_r interpolated at the Gauss point.
                let n = geom.field_n_at_g(g)?;
                let mut u_r = 0.0;
                for k in 0..geom.n_nodes {
                    u_r += dofs[k * nc] * n[k];
                }
                out[pairs.len()] = u_r / geom.radius(g)?;
            }
            Ok(())
        })?;
        out.add_sub(Handle::new(sf))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Linear displacement `u_x = 2x + 0.5y`, `u_y = 0.1x + 3y` on a TRI3.
    /// ⇒ ε_xx = 2, ε_yy = 3, ε_xy = ½(0.5 + 0.1) = 0.3.
    #[test]
    fn linear_displacement_gives_constant_strain() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        // u_x = 2x + 0.5y, u_y = 0.1x + 3y at the three nodes.
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 2.0 * x + 0.5 * y).unwrap();
            u.set_value(n.id(), "u_y", 0.1 * x + 3.0 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        {
            let s = strain.get(0).unwrap().read();
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // 2-D space but a single displacement component.
        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let u = NodeField::from_sub(SubNodeField::from_poi1(&support, vec!["u_x".into()]).unwrap());
        let err = deformation(&u, &fes).unwrap_err();
        assert!(format!("{err}").contains("one displacement component per axis"));
    }
}
