//! Weak divergence of a per-element **vector** field — the adjoint of
//! [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient).
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
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::kernel::{self, CellGeom};

/// Weak divergence `div F` of a per-element vector `field` (see the module
/// docs), as a [`NodeField`] — one zone per input subspace, component `"div"`.
///
/// This is the scalar case of the `Bᵀ` scatter, so it delegates to the shared
/// driver [`crate::models::kernel::scatter_to_nodes`] (with `B` = the gradient
/// operator): it inherits the driver's colour-parallel, per-thread-scratch
/// scatter rather than duplicating the loop.
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
/// // L'opérateur Bᵀ : d'un champ aux points de Gauss vers un champ nodal.
/// // Un flux **uniforme** a une divergence nodale de somme nulle — ce qui
/// // ne sort par un nœud entre y par un autre.
/// # let mut q = ElementField::new(&fes, vec!["q_x".into(), "q_y".into()])?;
/// # q.get(0)?.write().set_uniform("q_x", 1.0)?;
/// let d = node_field::divergence(&q)?;
/// assert_eq!(d.get(0)?.read().components(), &["div".to_string()]);
/// let total: f64 = (0..3)
///     .map(|i| d.get(0).unwrap().read().value(n[i].id(), "div").unwrap())
///     .sum();
/// assert!(total.abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn divergence(field: &ElementField) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for sub in field {
        out.add_sub(Handle::new(subspace_divergence(sub)?))?;
    }
    Ok(out)
}

/// Weak divergence on a single subspace, via the `Bᵀ` driver.
fn subspace_divergence(field: &Handle<SubElementField>) -> Result<SubNodeField> {
    let f = field.read();
    let fespace = f.support();
    let (submesh, space_dim) = {
        let s = fespace.read();
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
    // Les composantes du champ, résolues **une fois pour la zone** : le noyau
    // ci-dessous tranche des lignes et indexe, il ne compare plus un seul nom.
    let comps = f.components().to_vec();
    let refs: Vec<&str> = comps.iter().map(String::as_str).collect();
    let lay = f.resolve_components(&refs, "flux")?;

    let support = submesh.read().to_poi1()?;
    kernel::scatter_to_nodes(
        std::slice::from_ref(&fespace),
        &support,
        vec!["div".to_string()],
        |geoms, fe| divergence_element(geoms, &f, &lay, fe),
    )
}

/// Element kernel of the weak divergence: `fe[i] = Σ_g (∇N_i · F_g) |J| w`, the
/// single output component `"div"` per node (`n_dual = 1`). It is the transpose
/// of the gradient, so it plugs into the
/// [`crate::models::kernel::scatter_to_nodes`] driver exactly like a physics'
/// `internal_force_element`.
fn divergence_element(
    geoms: &[CellGeom],
    field: &SubElementField,
    lay: &[u32],
    fe: &mut [f64],
) -> Result<()> {
    let geom = &geoms[0];
    let d = geom.space_dim;
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    for g in 0..geom.n_gauss {
        let dn = &mut dn_buf[..geom.n_nodes * d]; // [i * d + a]
        geom.dn_dx(g, dn)?;
        let det_j_w = geom.det_j_w(g);
        // La ligne du point, tranchée une fois : elle était relue par nom, et
        // rebornée, pour chaque composante de chaque nœud de chaque point.
        let row = field.row(geom.cell, g);
        for i in 0..geom.n_nodes {
            let mut grad_dot_f = 0.0;
            for a in 0..d {
                grad_dot_f += dn[i * d + a] * row[lay[a] as usize];
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
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;

    /// 1-D, two SEG2 on `[0, 2]`, uniform field `F = (a)`: the weak divergence
    /// telescopes to `[−a, 0, +a]` — zero at the interior node (div of a
    /// constant field), `±a` flux at the ends.
    #[test]
    fn divergence_of_uniform_1d_field_telescopes() {
        let coords = Handle::new(Coords::new(1).unwrap());
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
        ef.add_sub(Handle::new(field)).unwrap();

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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // Nodal field f and a per-element vector field F.
        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
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
        ef.add_sub(Handle::new(ff)).unwrap();

        // ⟨∇f, F⟩ over the Gauss points (with |J|·w).
        let grad = crate::ops::element_field::gradient(&f, &fes).unwrap();
        let gsub = grad.get(0).unwrap().read();
        let s = fes.get(0).unwrap().read();
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        // 1 component on a 2-D space (needs 2).
        let field = SubElementField::new(fes.get(0).unwrap(), vec!["Fx".into()]).unwrap();
        let mut ef = ElementField::empty();
        ef.add_sub(Handle::new(field)).unwrap();
        assert!(divergence(&ef).is_err());
    }
}
