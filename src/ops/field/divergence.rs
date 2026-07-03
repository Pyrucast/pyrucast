//! Weak divergence of a per-element **vector** field — the adjoint of
//! [`crate::ops::field::gradient`](fn@crate::ops::field::gradient).
//!
//! Where `gradient` maps a nodal field to its gradient at the Gauss points
//! (`∇f = Σ_i f_i ∇N_i`), `divergence` maps a per-element vector field `F`
//! (given at the Gauss points) back to a [`NodeField`]:
//!
//! ```text
//! d_i = ∫_Ω ∇N_i · F dΩ  ≈  Σ_cell Σ_g (∇N_i · F)|_g · |J|_g · w_g
//! ```
//!
//! accumulated per node. This is the **discrete divergence operator** `Bᵀ`,
//! the transpose of the gradient operator: for a nodal field `f`, it satisfies
//! `⟨∇f, F⟩ = ⟨f, div F⟩`. By integration by parts it equals
//! `∫ N_i (∇·F) − ∮ N_i F·n` — the consistent (weak) nodal divergence, no mass
//! solve. The input field must carry exactly `space_dim` components, taken in
//! order as `F_x, F_y, F_z`; each input subspace yields one output zone with a
//! single `"div"` component.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::models::kernel::{self, CellGeom};
use crate::store::{insert, read, Handle};

/// Weak divergence `div F` of a per-element vector `field` (see the module
/// docs), as a [`NodeField`] — one zone per input subspace, component `"div"`.
///
/// This is the scalar case of the `Bᵀ` scatter, so it delegates to the shared
/// driver [`crate::models::kernel::scatter_to_nodes`] (with `B` = the gradient
/// operator): it inherits the driver's colour-parallel, per-thread-scratch
/// scatter rather than duplicating the loop.
pub fn divergence(field: &ElementField) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for sub in field {
        out.add_sub(insert(subspace_divergence(sub)?))?;
    }
    Ok(out)
}

/// Weak divergence on a single subspace, via the `Bᵀ` driver.
fn subspace_divergence(field: &Handle<SubElementField>) -> Result<SubNodeField> {
    let f = read(field)?;
    let fespace = f.support();
    let (submesh, space_dim) = {
        let s = read(&fespace)?;
        (s.submesh(), s.space_dim())
    };
    let n_comps = f.components().len();
    if n_comps != space_dim {
        return Err(PyrucastError::Message(format!(
            "divergence: the field carries {} component(s) but the FE space is {}-D — \
             expected one vector component per axis",
            n_comps, space_dim
        )));
    }

    // The field guard is captured by the element closure (borrowed in place) and
    // held across the parallel scatter.
    let support = insert(read(&submesh)?.to_poi1()?);
    kernel::scatter_to_nodes(
        std::slice::from_ref(&fespace),
        &support,
        vec!["div".to_string()],
        |geoms, fe| divergence_element(geoms, &f, fe),
    )
}

/// Element kernel of the weak divergence: `fe[i] = Σ_g (∇N_i · F_g) |J| w`, the
/// single output component `"div"` per node (`n_dual = 1`). It is the transpose
/// of the gradient, so it plugs into the
/// [`crate::models::kernel::scatter_to_nodes`] driver exactly like a physics'
/// `internal_force_element`.
fn divergence_element(geoms: &[CellGeom], field: &SubElementField, fe: &mut [f64]) -> Result<()> {
    let geom = &geoms[0];
    let d = geom.space_dim;
    let comps = field.components();
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?; // [i * d + a]
        let det_j_w = geom.det_j_w(g)?;
        for i in 0..geom.n_nodes {
            let mut grad_dot_f = 0.0;
            for (a, comp) in comps.iter().enumerate() {
                grad_dot_f += dn[i * d + a] * field.value(geom.cell, g, comp)?;
            }
            fe[i] += grad_dot_f * det_j_w;
        }
    }
    Ok(())
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::field::Field;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};

    /// 1-D, two SEG2 on `[0, 2]`, uniform field `F = (a)`: the weak divergence
    /// telescopes to `[−a, 0, +a]` — zero at the interior node (div of a
    /// constant field), `±a` flux at the ends.
    #[test]
    fn divergence_of_uniform_1d_field_telescopes() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let a = 3.0;
        let mut field = SubElementField::new(fes.get(0).unwrap(), vec!["Fx".into()]).unwrap();
        field.set_uniform("Fx", a).unwrap();
        let mut ef = ElementField::empty();
        ef.add_sub(insert(field)).unwrap();

        let div = divergence(&ef).unwrap();
        let view = div.view().unwrap();
        let tol = 1e-12;
        assert!((view.value(n0.id(), "div").unwrap() + a).abs() < tol); // −a
        assert!(view.value(n1.id(), "div").unwrap().abs() < tol); //  0
        assert!((view.value(n2.id(), "div").unwrap() - a).abs() < tol); // +a
    }

    /// Adjoint property `⟨∇f, F⟩ = ⟨f, div F⟩` on a 2-D TRI3.
    #[test]
    fn divergence_is_adjoint_of_gradient() {
        use crate::containers::node_field::SubNodeField;
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // Nodal field f and a per-element vector field F.
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        f.set_value(a.id(), "f", 0.5).unwrap();
        f.set_value(b.id(), "f", 1.5).unwrap();
        f.set_value(c.id(), "f", -2.0).unwrap();
        let f = NodeField::from_sub(f);

        let mut ff =
            SubElementField::new(fes.get(0).unwrap(), vec!["Fx".into(), "Fy".into()]).unwrap();
        ff.set_uniform("Fx", 1.3).unwrap();
        ff.set_uniform("Fy", -0.7).unwrap();
        let mut ef = ElementField::empty();
        ef.add_sub(insert(ff)).unwrap();

        // ⟨∇f, F⟩ over the Gauss points (with |J|·w).
        let grad = crate::ops::field::gradient(&f, &fes).unwrap();
        let gsub = read(&grad.get(0).unwrap()).unwrap();
        let s = read(&fes.get(0).unwrap()).unwrap();
        let mut lhs = 0.0;
        for cell in 0..s.cell_count().unwrap() {
            for g in 0..s.gauss_count() {
                let w = s.det_jacobian(cell, g).unwrap() * s.gauss_weight(g).unwrap();
                lhs += (gsub.value(cell, g, "grad_f_x").unwrap() * 1.3
                    + gsub.value(cell, g, "grad_f_y").unwrap() * -0.7)
                    * w;
            }
        }

        // ⟨f, div F⟩ over the nodes.
        let div = divergence(&ef).unwrap();
        let dview = div.view().unwrap();
        let fview = f.view().unwrap();
        let mut rhs = 0.0;
        for nid in [a.id(), b.id(), c.id()] {
            rhs += fview.value(nid, "f").unwrap() * dview.value(nid, "div").unwrap();
        }
        assert!((lhs - rhs).abs() < 1e-12, "adjoint: {lhs} ≠ {rhs}");
    }

    #[test]
    fn wrong_component_count_is_rejected() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        // 1 component on a 2-D space (needs 2).
        let field = SubElementField::new(fes.get(0).unwrap(), vec!["Fx".into()]).unwrap();
        let mut ef = ElementField::empty();
        ef.add_sub(insert(field)).unwrap();
        assert!(divergence(&ef).is_err());
    }
}
