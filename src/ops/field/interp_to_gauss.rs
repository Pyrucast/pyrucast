//! Interpolation of a nodal field to the Gauss points — the value counterpart
//! of [`crate::ops::field::gradient`](fn@crate::ops::field::gradient) (which
//! takes the derivatives). Purely geometric, no model/physics involved.
//!
//! `f(ξ_g) = Σ_i f_i N_i(ξ_g)` evaluated cell-by-cell, Gauss point by Gauss
//! point. It turns a **per-node** [`NodeField`] into a **per-element**
//! [`ElementField`] carrying the same component names, ready to feed operators
//! that expect Gauss-point data (e.g.
//! [`crate::ops::field::thermal_strain`](fn@crate::ops::field::thermal_strain)).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::Field;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::models::kernel;
use crate::store::insert;

/// Interpolate a nodal `field` to the Gauss points of every subspace of
/// `fespace`: `f(ξ_g) = Σ_i f_i N_i(ξ_g)`.
///
/// The result is an [`ElementField`] carrying the **same component names** as
/// the input, one value per `(cell, Gauss point)`. Runs on the shared parallel
/// driver `models::kernel::nodal_pointwise`.
pub fn interp_to_gauss(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(field)?;
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        // Point kernel: value at Gauss g = Σ_i N_i(g) · f_i, per component.
        let sf = kernel::nodal_pointwise(sub, &view, components.clone(), |geom, field, g, out| {
            let shape = geom.n_at_g(g)?;
            let ids = geom.node_ids();
            for (c, comp) in components.iter().enumerate() {
                let mut v = 0.0;
                for i in 0..geom.n_nodes {
                    v += shape[i] * field.value(ids[i], comp)?;
                }
                out[c] = v;
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
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::store::{insert, read};

    /// Constant nodal field ⇒ same value at every Gauss point, same component
    /// names as the input.
    #[test]
    fn constant_field_is_reproduced_at_every_gauss_point() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut t = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        for n in [&a, &b, &c] {
            t.set_value(n.id(), "T", 42.0).unwrap();
        }
        let t = NodeField::from_sub(t);

        let elem = interp_to_gauss(&t, &fes).unwrap();
        let s = read(&elem.get(0).unwrap()).unwrap();
        assert_eq!(s.components(), &["T".to_string()]);
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "T").unwrap() - 42.0).abs() < 1e-12);
        }
    }

    /// Linear nodal field `T(x) = 1 + 3x` on a SEG2: the shape-function
    /// interpolation is exact, so `T` at each Gauss point equals `1 + 3·x_g`.
    #[test]
    fn linear_field_interpolated_exactly_on_seg2() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut t = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        t.set_value(a.id(), "T", 1.0).unwrap(); // 1 + 3·0
        t.set_value(b.id(), "T", 7.0).unwrap(); // 1 + 3·2
        let t = NodeField::from_sub(t);

        let elem = interp_to_gauss(&t, &fes).unwrap();
        let s = read(&elem.get(0).unwrap()).unwrap();
        // The average of the two Gauss values equals the midpoint value 1 + 3·1 = 4.
        let mean: f64 = (0..s.gauss_count())
            .map(|g| s.value(0, g, "T").unwrap())
            .sum::<f64>()
            / s.gauss_count() as f64;
        assert!((mean - 4.0).abs() < 1e-12, "mean = {mean}");
        // Every Gauss value stays inside the nodal range [1, 7].
        for g in 0..s.gauss_count() {
            let v = s.value(0, g, "T").unwrap();
            assert!((1.0..=7.0).contains(&v), "v = {v}");
        }
    }

    /// Multiple components are interpolated independently, and every subspace of
    /// a multi-zone FE space gets its own zone.
    #[test]
    fn multi_component_multi_subspace() {
        let coords = insert(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_qua).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(
            SubMesh::poi1_from_nodes(&[n0.clone(), n1.clone(), n2.clone(), n3.clone()]).unwrap(),
        );
        let mut f = SubNodeField::from_poi1(&support, vec!["a".into(), "b".into()]).unwrap();
        for n in [&n0, &n1, &n2, &n3] {
            f.set_value(n.id(), "a", 5.0).unwrap();
            f.set_value(n.id(), "b", -2.0).unwrap();
        }
        let f = NodeField::from_sub(f);

        let elem = interp_to_gauss(&f, &fes).unwrap();
        assert_eq!(elem.len(), 2);
        for zone in 0..2 {
            let s = read(&elem.get(zone).unwrap()).unwrap();
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "a").unwrap() - 5.0).abs() < 1e-12);
                assert!((s.value(0, g, "b").unwrap() + 2.0).abs() < 1e-12);
            }
        }
    }
}
