//! Consolidate a [`NodeField`]: fuse zones sharing the **same support**.
//!
//! Sub-fields defined on the *same* support `SubMesh` (matched by handle
//! identity, [`crate::handle::Handle::same_object`]) are fused into a single
//! [`SubNodeField`] carrying the **union of their components**. A component
//! defined by several of those sub-fields must hold the **same** value at
//! every shared node (exact comparison) — anything else is an error. Zones
//! on distinct supports stay separate.
//!
//! This is the finalization step of the node-field union (`a | b`,
//! [`crate::aggregate::Aggregate::merge`]): after the union deduplicates by
//! handle, `consolidate` collapses the remaining zones that share a
//! support.
//!
//! Beyond per-support fusion, a final cross-zone check runs
//! ([`NodeField::check`]): a node shared by sub-fields on *different*
//! supports must still agree on any common component.
//!
//! A support group reduced to a single sub-field is **shared** (same handle,
//! no copy) in the result.

use crate::aggregate::Aggregate;
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;
use crate::handle::Handle;
use crate::parallel::*;

/// Fuse the zones of `field` that share the same support `SubMesh`.
///
/// See the module documentation. Errors if two zones on the same support
/// disagree on a shared `(node, component)` value, or if the global
/// cross-zone check fails.
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
/// // Fusionne en une zone les champs qui partagent un support.
/// let flux = champ(vec!["q".into()]);
/// let deux = node_field::merge(&temp, &flux)?;
/// assert_eq!(node_field::consolidate(&deux)?.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn consolidate(field: &NodeField) -> Result<NodeField> {
    // Group the zone handles by support identity, first-seen order. Only the
    // support handle is read here, never the values: a group of one must cost
    // nothing.
    let mut groups: Vec<Vec<Handle<SubNodeField>>> = Vec::new();
    for h in field {
        let support = h.read().support();
        match groups
            .iter_mut()
            .find(|g| g[0].read().support().same_object(&support))
        {
            Some(g) => g.push(h.clone()),
            None => groups.push(vec![h.clone()]),
        }
    }

    let mut out = NodeField::default();
    for group in &groups {
        // A group of one keeps its zone: shared, never copied. Snapshotting it
        // would copy a whole displacement field to hand it back unchanged.
        if let [single] = group.as_slice() {
            out.add_sub(single.clone())?;
            continue;
        }

        // Component lists only — a few names per zone. The value buffers stay
        // where they are, and the fused zone is built *before* the long read
        // guards: its support seals itself, and no guard must span that.
        let comps: Vec<Vec<String>> = group
            .iter()
            .map(|h| h.read().components().to_vec())
            .collect();

        // Union of the group's components, first-seen order across the zones.
        let mut components: Vec<String> = Vec::new();
        for cs in &comps {
            for c in cs {
                if !components.contains(c) {
                    components.push(c.clone());
                }
            }
        }

        // All zones in the group share the support, hence the same node list
        // and the same row order: the fused buffer lines up positionally with
        // every source buffer.
        let support = group[0].read().support();
        let mut fused = SubNodeField::from_support(&support, components.clone())?;
        let out_nc = components.len();
        // The node ids serve the error message alone; read them once, before
        // the guards and outside the parallel pass.
        let nodes = fused.node_ids();

        // Resolve, **once per zone**, where each of its components lands in the
        // fused layout and whether this zone is the one that writes it. Looking
        // the name up at every node instead would re-prove, once per value, a
        // property of the zone.
        let mut written = vec![false; out_nc];
        let plans: Vec<Vec<(usize, usize, bool)>> = comps
            .iter()
            .map(|cs| {
                cs.iter()
                    .enumerate()
                    .map(|(ci, comp)| {
                        let oc = components
                            .iter()
                            .position(|c| c == comp)
                            .expect("the union of the group's components contains each of them");
                        let first = !written[oc];
                        written[oc] = true;
                        (ci, oc, first)
                    })
                    .collect()
            })
            .collect();

        // Read the group's zones for the whole fusion — concurrent shared
        // locks, no copy.
        let zones: Vec<_> = group.iter().map(|h| h.read()).collect();

        // One flat pass per zone, parallel over the node rows: each row of the
        // fused buffer is written by one task alone. A component several zones
        // carry must agree at every node — checked by index, in the same pass.
        for (z, plan) in zones.iter().zip(&plans) {
            let src_nc = z.component_count();
            let src = z.values();
            fused
                .values_mut()
                .par_chunks_mut(out_nc)
                .with_min_len((MIN_PARALLEL_LEN / out_nc.max(1)).max(1))
                .enumerate()
                .try_for_each(|(row, dst)| -> Result<()> {
                    for &(ci, oc, first) in plan {
                        let v = src[row * src_nc + ci];
                        if first {
                            dst[oc] = v;
                        } else if dst[oc] != v {
                            return Err(crate::error::PyrucastError::Message(format!(
                                "incoherent NodeField on shared support: node {}, \
                                 component {}: {} \u{2260} {}",
                                nodes[row], components[oc], dst[oc], v
                            )));
                        }
                    }
                    Ok(())
                })?;
        }
        drop(zones);
        out.add_sub(Handle::new(fused))?;
    }

    // Cross-support coherence: a node shared by zones on *different* supports
    // must still agree on any common component.
    out.check()?;
    Ok(out)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::Field;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Single-zone POI1 field over `nodes`, sharing the support handle `sm`.
    fn field_on(sm: &Handle<SubMesh>, components: Vec<String>) -> NodeField {
        NodeField::from_sub(SubNodeField::from_poi1(sm, components).unwrap())
    }

    /// Two-zone field built **without** triggering the union's finalize, so
    /// `consolidate` can be exercised directly. Each sub is a fresh handle.
    fn two_zone(a: &NodeField, b: &NodeField) -> NodeField {
        let mut f = NodeField::default();
        f.add_sub(a.get(0).unwrap()).unwrap();
        f.add_sub(b.get(0).unwrap()).unwrap();
        f
    }

    fn poi1(coords: &Handle<Coords>, nodes: &[&Node]) -> Handle<SubMesh> {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        Handle::new(sm)
    }

    #[test]
    fn same_support_distinct_components_fuse() {
        // Two zones on the *same* POI1 support, components ["T"] and ["P"].
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = poi1(&coords, &[&n0, &n1]);

        let a = field_on(&sm, vec!["T".into()]);
        let b = field_on(&sm, vec!["P".into()]);
        a.get(0)
            .unwrap()
            .write()
            .set_value(n0.id(), "T", 5.0)
            .unwrap();
        b.get(0)
            .unwrap()
            .write()
            .set_value(n1.id(), "P", 9.0)
            .unwrap();

        let c = consolidate(&two_zone(&a, &b)).unwrap();
        assert_eq!(c.len(), 1, "same support ⇒ one fused zone");
        assert_eq!(Field::components(&c).unwrap(), vec!["T", "P"]);
        assert_eq!(c.value(n0.id(), "T").unwrap(), 5.0);
        assert_eq!(c.value(n1.id(), "P").unwrap(), 9.0);
    }

    #[test]
    fn same_support_shared_component_must_agree() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = poi1(&coords, &[&n0]);
        let a = field_on(&sm, vec!["T".into()]);
        let b = field_on(&sm, vec!["T".into()]);
        a.get(0)
            .unwrap()
            .write()
            .set_value(n0.id(), "T", 1.0)
            .unwrap();
        b.get(0)
            .unwrap()
            .write()
            .set_value(n0.id(), "T", 2.0)
            .unwrap();
        // Distinct handles, same support: the union's finalize fuses them
        // and detects the diverging T — so `|` itself errors.
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn same_support_shared_component_agreeing_is_ok() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = poi1(&coords, &[&n0]);
        let a = field_on(&sm, vec!["T".into(), "P".into()]);
        let b = field_on(&sm, vec!["T".into()]);
        a.get(0)
            .unwrap()
            .write()
            .set_value(n0.id(), "T", 7.0)
            .unwrap();
        b.get(0)
            .unwrap()
            .write()
            .set_value(n0.id(), "T", 7.0)
            .unwrap();
        // Agreeing shared component → `|` succeeds and fuses.
        let c = a.union(&b).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.value(n0.id(), "T").unwrap(), 7.0);
    }

    #[test]
    fn distinct_supports_stay_separate_and_share_handles() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm_a = poi1(&coords, &[&n0]);
        let sm_b = poi1(&coords, &[&n1]);
        let a = field_on(&sm_a, vec!["T".into()]);
        let b = field_on(&sm_b, vec!["P".into()]);
        let f = two_zone(&a, &b);
        let c = consolidate(&f).unwrap();
        assert_eq!(c.len(), 2);
        // Singleton groups: handles shared, not copied.
        assert!(c.get(0).unwrap().same_object(&f.get(0).unwrap()));
        assert!(c.get(1).unwrap().same_object(&f.get(1).unwrap()));
    }

    #[test]
    fn cross_support_shared_node_checked() {
        // Two distinct POI1 supports that both include node n (different
        // submeshes, n shared); a diverging T at n is still an error.
        let coords = Handle::new(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let m = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm_a = poi1(&coords, &[&n, &m]);
        let sm_b = poi1(&coords, &[&n]);
        let a = field_on(&sm_a, vec!["T".into()]);
        let b = field_on(&sm_b, vec!["T".into()]);
        a.get(0)
            .unwrap()
            .write()
            .set_value(n.id(), "T", 1.0)
            .unwrap();
        b.get(0)
            .unwrap()
            .write()
            .set_value(n.id(), "T", 2.0)
            .unwrap();
        // Different supports → no fusion, but the cross-support check in the
        // union's finalize still catches the diverging shared node.
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn empty_field_consolidates_to_empty() {
        let c = consolidate(&NodeField::default()).unwrap();
        assert!(c.is_empty());
    }
}
