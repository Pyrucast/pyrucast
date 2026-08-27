//! Gradient of a nodal field at the Gauss points — purely geometric, no
//! model/physics involved (it depends only on the FE space and the field).
//!
//! `∇f = Σ_i f_i ∇N_i` evaluated cell-by-cell, Gauss-point by Gauss-point.
//! The shared `subspace_gradients` core also backs
//! [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation) (the symmetric gradient).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::Field;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::kernel::MAX_CELL_DOFS;

/// Axis suffixes for spatial directions (`x`, `y`, `z`).
pub(crate) const AXES: [&str; 3] = ["x", "y", "z"];

/// Gradient `∇f` of a node `field` at the Gauss points of every subspace of
/// `fespace`.
///
/// Geometric and physics-agnostic: each component of `field` is
/// differentiated w.r.t. every spatial axis, giving an [`ElementField`] with
/// one component `grad_<name>_<axis>` per (input component, axis) pair, in
/// order component-major then axis (`grad_T_x`, …). Feed the result to
/// [`crate::ops::element_field::behavior::integrate`] as the deformation input of a physics
/// whose behaviour consumes a gradient (heat conduction).
///
/// `∇f = Σ_i f_i ∇N_i` evaluated cell-by-cell, Gauss point by Gauss point, via
/// the shared parallel driver `models::kernel::nodal_pointwise`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // ∇T aux points de Gauss. Sur ce triangle T croît de 10 à 50 sur deux
/// // unités en x : le gradient vaut 20.
/// let g = element_field::gradient(&temp, &fes)?;
/// assert!((g.get(0)?.read().value(0, 0, "grad_T_x")? - 20.0).abs() < 1e-9);
/// assert!((g.get(0)?.read().value(0, 0, "grad_T_y")? - 40.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn gradient(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(field)?;
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        let space_dim = sub.read().space_dim();
        let mut names = Vec::with_capacity(components.len() * space_dim);
        for comp in &components {
            for a in 0..space_dim {
                names.push(format!("grad_{comp}_{}", AXES[a]));
            }
        }
        // Point kernel: ∂(comp)/∂x_a = Σ_i f_i · ∂N_i/∂x_a, laid out
        // component-major then axis (grad_<comp>_x, grad_<comp>_y, …).
        let nc = components.len();
        let sf = kernel::nodal_pointwise(sub, &view, &components, names, |geom, g, dofs, out| {
            let sd = geom.space_dim;
            let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
            let dn_dx = &mut dn_buf[..geom.n_nodes * sd];
            geom.dn_dx(g, dn_dx)?;
            let mut oc = 0;
            for c in 0..nc {
                for a in 0..sd {
                    let mut grad = 0.0;
                    for i in 0..geom.n_nodes {
                        grad += dofs[i * nc + c] * dn_dx[i * sd + a];
                    }
                    out[oc] = grad;
                    oc += 1;
                }
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

    /// 1-D linear field `T(x) = x` on a single SEG2 ⇒ `∇T = 1`.
    #[test]
    fn gradient_of_linear_field_on_seg2() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut t = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        t.set_value(a.id(), "T", 0.0).unwrap();
        t.set_value(b.id(), "T", 2.0).unwrap(); // T = x
        let t = NodeField::from_sub(t);

        let grad = gradient(&t, &fes).unwrap();
        assert_eq!(grad.len(), 1);
        {
            let s = grad.get(0).unwrap().read();
            assert_eq!(s.components(), &["grad_T_x".to_string()]);
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_T_x").unwrap() - 1.0).abs() < 1e-12);
            }
        }
    }

    /// 2-D linear field `f = 2x + 3y` on a TRI3 ⇒ `∇f = (2, 3)`.
    #[test]
    fn gradient_2d_linear_field_on_tri3() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        f.set_value(a.id(), "f", 0.0).unwrap(); // 2·0 + 3·0
        f.set_value(b.id(), "f", 2.0).unwrap(); // 2·1 + 3·0
        f.set_value(c.id(), "f", 3.0).unwrap(); // 2·0 + 3·1
        let f = NodeField::from_sub(f);

        let grad = gradient(&f, &fes).unwrap();
        {
            let s = grad.get(0).unwrap().read();
            assert_eq!(
                s.components(),
                &["grad_f_x".to_string(), "grad_f_y".to_string()]
            );
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_f_x").unwrap() - 2.0).abs() < 1e-12);
                assert!((s.value(0, g, "grad_f_y").unwrap() - 3.0).abs() < 1e-12);
            }
        }
    }
}
