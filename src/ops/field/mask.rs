//! Value-range masking — turn a field into a 0/1 indicator of the **same
//! shape**, testing each value against a `[lower, upper]` band, component by
//! component (Cast3M's `MASQUE`).
//!
//! Unlike [`select`](super::select), which extracts the passing part of the
//! support into a `Mesh`, `mask` keeps the field's exact structure (same
//! zones, same support, same components) and only rewrites the values:
//! `1.0` where the band holds, `0.0` where it does not. The result is
//! therefore multipliable term by term with the input — `field *
//! mask(field, ge=0)` zeroes the out-of-band values, component by
//! component.
//!
//! There is **no** AND across components here (that is [`select`](super::select)'s job):
//! each value stands on its own, so the mask is per component.
//!
//! # Component filter
//!
//! - `components = None` ⇒ every component is tested (gets a real 0/1);
//! - `components = Some(list)` ⇒ only those components are tested; the
//!   others are left at `1.0` (identity for the product), so masking a
//!   subset of components leaves the rest untouched. A zone that does not
//!   carry a listed component is left all-`1.0` (the filter cannot apply).
//!
//! # Bounds
//!
//! The `[lower, upper]` band comes from a shared [`Band`],
//! built from the four comparison bounds `ge` / `gt` / `le` / `lt` — each
//! side open, inclusive, or strict.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::{Field, SubField};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;

use super::band::Band;
use super::select::components_to_test;

/// Rewrite `sub` into its 0/1 mask: each **tested** component becomes `1.0`
/// inside the band and `0.0` outside; untested components stay `1.0`.
///
/// Components are the innermost storage axis for both field flavours, so a
/// single flat pass with `idx % ncomp` handles nodes and Gauss points alike.
fn mask_sub<S>(sub: &S, band: &Band, components: &Option<Vec<String>>) -> S
where
    S: SubField + Clone,
{
    let ncomp = sub.component_count();
    // Which component indices are tested? `None` ⇒ the zone lacks a
    // requested component — nothing to test, leave it identity (all 1.0).
    let mut tested = vec![false; ncomp];
    if let Some(idx) = components_to_test(sub.components(), components) {
        for i in idx {
            tested[i] = true;
        }
    }
    let mut out = sub.clone();
    for (i, v) in out.values_mut().iter_mut().enumerate() {
        *v = if tested[i % ncomp] {
            f64::from(band.contains(*v))
        } else {
            1.0
        };
    }
    out
}

/// Per-component 0/1 mask of `field` against `band`, zone by zone. Returns a
/// [`NodeField`] with the **same structure** (zones, support, components) as
/// the input. See the [module documentation](self) for the component-filter
/// semantics.
pub fn mask_nodes(
    field: &NodeField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<NodeField> {
    field.map_subs(|s| Ok(mask_sub(s, band, &components)))
}

/// Per-component 0/1 mask of `field` against `band`, zone by zone (one value
/// per Gauss point × component). Returns an [`ElementField`] with the same
/// structure as the input.
pub fn mask_cells(
    field: &ElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<ElementField> {
    field.map_subs(|s| Ok(mask_sub(s, band, &components)))
}

/// Single-zone [`mask_nodes`].
pub fn mask_sub_nodes(
    sub: &SubNodeField,
    band: &Band,
    components: Option<Vec<String>>,
) -> SubNodeField {
    mask_sub(sub, band, &components)
}

/// Single-zone [`mask_cells`].
pub fn mask_sub_cells(
    sub: &SubElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> SubElementField {
    mask_sub(sub, band, &components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::store::{insert, read, write};

    /// Single-zone POI1 NodeField over `n` 1-D nodes; returns (nodes, field).
    fn poi1_field(n: usize, components: Vec<String>) -> (Vec<Node>, NodeField) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in &nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        let field = NodeField::from_sub(SubNodeField::from_poi1(&insert(sm), components).unwrap());
        (nodes, field)
    }

    /// Flat 0/1 values of the (single) zone of a mask field.
    fn values(field: &NodeField) -> Vec<f64> {
        read(&field.get(0).unwrap()).unwrap().values().to_vec()
    }

    #[test]
    fn mask_keeps_structure_and_flags_band() {
        let (_n, f) = poi1_field(5, vec!["T".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            for i in 0..5 {
                s.set(i, 0, i as f64 * 10.0).unwrap(); // 0,10,20,30,40
            }
        }
        // 10 <= T <= 30 → nodes 1,2,3 flagged.
        let m = mask_nodes(
            &f,
            &Band::new(Some(10.0), None, Some(30.0), None).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(m.len(), 1, "same number of zones as input");
        assert_eq!(values(&m), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn mask_strict_bounds() {
        let (_n, f) = poi1_field(3, vec!["T".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set(0, 0, 0.0).unwrap();
            s.set(1, 0, 5.0).unwrap();
            s.set(2, 0, 9.0).unwrap();
        }
        // v > 5 (gt) → only the last node (5 excluded).
        let m = mask_nodes(&f, &Band::new(None, Some(5.0), None, None).unwrap(), None).unwrap();
        assert_eq!(values(&m), vec![0.0, 0.0, 1.0]);
        // v <= 5 (le) → the first two (5 included).
        let m = mask_nodes(&f, &Band::new(None, None, Some(5.0), None).unwrap(), None).unwrap();
        assert_eq!(values(&m), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn mask_is_per_component_no_and() {
        let (_n, f) = poi1_field(2, vec!["U".into(), "V".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            // node0: U=1 V=9 ; node1: U=9 V=1
            s.set(0, 0, 1.0).unwrap();
            s.set(0, 1, 9.0).unwrap();
            s.set(1, 0, 9.0).unwrap();
            s.set(1, 1, 1.0).unwrap();
        }
        // 0 <= * <= 5: each component independent (no AND across U,V).
        let m = mask_nodes(
            &f,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            None,
        )
        .unwrap();
        // layout node → component: [U0,V0, U1,V1]
        assert_eq!(values(&m), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn mask_component_filter_leaves_others_identity() {
        let (_n, f) = poi1_field(2, vec!["U".into(), "V".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set(0, 0, 9.0).unwrap(); // U out of band
            s.set(0, 1, 9.0).unwrap(); // V out of band
            s.set(1, 0, 1.0).unwrap();
            s.set(1, 1, 1.0).unwrap();
        }
        // Test U only: V stays 1.0 (identity), U gets a real 0/1.
        let m = mask_nodes(
            &f,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            Some(vec!["U".into()]),
        )
        .unwrap();
        // [U0=0, V0=1, U1=1, V1=1]
        assert_eq!(values(&m), vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_zone_missing_component_is_identity() {
        let (_n, f) = poi1_field(3, vec!["T".into()]);
        // Filter on "P" which the zone lacks → everything stays 1.0.
        let m = mask_nodes(
            &f,
            &Band::new(Some(0.0), None, Some(0.0), None).unwrap(),
            Some(vec!["P".into()]),
        )
        .unwrap();
        assert_eq!(values(&m), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_cells_per_gauss() {
        let coords = insert(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut mesh = Mesh::empty();
        let tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(tri).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["s".into()]).unwrap();
        {
            let mut s = write(&ef.get(0).unwrap()).unwrap();
            s.set_uniform("s", 2.0).unwrap();
            s.set_value(0, 0, "s", 100.0).unwrap(); // one Gauss point out of band
        }
        let m = mask_cells(
            &ef,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            None,
        )
        .unwrap();
        let mr = read(&m.get(0).unwrap()).unwrap();
        // The masked-out Gauss point is 0.0, every other one is 1.0 —
        // per-Gauss, no all-must-pass collapse (that is `select`'s rule).
        assert_eq!(mr.get(0, 0, 0).unwrap(), 0.0);
        for g in 1..mr.gauss_count() {
            assert_eq!(mr.get(0, g, 0).unwrap(), 1.0);
        }
    }
}
