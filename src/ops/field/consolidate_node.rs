//! Consolidate a [`NodeField`]: fuse zones sharing the **same support**.
//!
//! Sub-fields defined on the *same* support `SubMesh` (matched by handle
//! identity, [`crate::store::Handle::same_slot`]) are fused into a single
//! [`SubNodeField`] carrying the **union of their components**. A component
//! defined by several of those sub-fields must hold the **same** value at
//! every shared node (exact comparison) — anything else is an error. Zones
//! on distinct supports stay separate.
//!
//! This is the finalization step of the node-field union (`a | b`,
//! [`crate::aggregate::Aggregate::merge`]): after the union deduplicates by
//! handle, `consolidate_node` collapses the remaining zones that share a
//! support.
//!
//! Beyond per-support fusion, a final cross-zone check runs
//! ([`NodeField::check`]): a node shared by sub-fields on *different*
//! supports must still agree on any common component.
//!
//! A support group reduced to a single sub-field is **shared** (same handle,
//! no copy) in the result.

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;
use crate::store::{insert, read, Handle};

/// Fuse the zones of `field` that share the same support `SubMesh`.
///
/// See the module documentation. Errors if two zones on the same support
/// disagree on a shared `(node, component)` value, or if the global
/// cross-zone check fails.
pub fn consolidate_node(field: &NodeField) -> Result<NodeField> {
    // Snapshot every sub once (one lock each, never nested), keeping its
    // support handle for grouping and the singleton-sharing case.
    struct Snap {
        handle: Handle<SubNodeField>,
        support: Handle<crate::containers::mesh::SubMesh>,
        nodes: Vec<NodeId>,
        components: Vec<String>,
        values: Vec<f64>,
    }
    let mut snaps: Vec<Snap> = Vec::with_capacity(field.len());
    for h in field {
        let (support, nodes, components, values) = {
            let s = read(h)?;
            (
                s.support(),
                s.nodes().to_vec(),
                s.components().to_vec(),
                s.values().to_vec(),
            )
        };
        snaps.push(Snap {
            handle: h.clone(),
            support,
            nodes,
            components,
            values,
        });
    }

    // Group sub indices by support handle identity, first-seen order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, snap) in snaps.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|idxs| snaps[idxs[0]].support.same_slot(&snap.support))
        {
            Some(idxs) => idxs.push(i),
            None => groups.push(vec![i]),
        }
    }

    let mut out = NodeField::default();
    for idxs in &groups {
        if let [single] = idxs.as_slice() {
            // Nothing to fuse: share the sub-field as-is.
            out.add_sub(snaps[*single].handle.clone())?;
            continue;
        }

        // Union of the group's components, first-seen order across the subs.
        let mut components: Vec<String> = Vec::new();
        for &i in idxs {
            for c in &snaps[i].components {
                if !components.contains(c) {
                    components.push(c.clone());
                }
            }
        }

        // All subs in the group share the same support, hence the same node
        // list; build the fused sub on that very support (shared handle).
        let support = snaps[idxs[0]].support.clone();
        let mut fused = SubNodeField::from_support(&support, components)?;

        // Fill from every sub; a component shared by several subs must agree
        // at each node (exact comparison) — first writer sets, later writers
        // must match.
        let mut seen: std::collections::HashSet<(NodeId, String)> =
            std::collections::HashSet::new();
        for &i in idxs {
            let snap = &snaps[i];
            let ncomp = snap.components.len();
            for (ni, &nid) in snap.nodes.iter().enumerate() {
                for (ci, comp) in snap.components.iter().enumerate() {
                    let v = snap.values[ni * ncomp + ci];
                    if seen.insert((nid, comp.clone())) {
                        fused.set_value(nid, comp, v)?;
                    } else {
                        let existing = fused.value(nid, comp)?;
                        if existing != v {
                            return Err(crate::error::PyrucastError::Message(format!(
                                "incoherent NodeField on shared support: node {}, \
                                 component {}: {} ≠ {}",
                                nid, comp, existing, v
                            )));
                        }
                    }
                }
            }
        }
        out.add_sub(insert(fused))?;
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
    use crate::store::{insert, write, Handle};

    /// Single-zone POI1 field over `nodes`, sharing the support handle `sm`.
    fn field_on(sm: &Handle<SubMesh>, components: Vec<String>) -> NodeField {
        NodeField::from_sub(SubNodeField::from_poi1(sm, components).unwrap())
    }

    /// Two-zone field built **without** triggering the union's finalize, so
    /// `consolidate_node` can be exercised directly. Each sub is a fresh handle.
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
        insert(sm)
    }

    #[test]
    fn same_support_distinct_components_fuse() {
        // Two zones on the *same* POI1 support, components ["T"] and ["P"].
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = poi1(&coords, &[&n0, &n1]);

        let a = field_on(&sm, vec!["T".into()]);
        let b = field_on(&sm, vec!["P".into()]);
        write(&a.get(0).unwrap())
            .unwrap()
            .set_value(n0.id(), "T", 5.0)
            .unwrap();
        write(&b.get(0).unwrap())
            .unwrap()
            .set_value(n1.id(), "P", 9.0)
            .unwrap();

        let c = consolidate_node(&two_zone(&a, &b)).unwrap();
        assert_eq!(c.len(), 1, "same support ⇒ one fused zone");
        assert_eq!(Field::components(&c).unwrap(), vec!["T", "P"]);
        assert_eq!(c.value(n0.id(), "T").unwrap(), 5.0);
        assert_eq!(c.value(n1.id(), "P").unwrap(), 9.0);
    }

    #[test]
    fn same_support_shared_component_must_agree() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = poi1(&coords, &[&n0]);
        let a = field_on(&sm, vec!["T".into()]);
        let b = field_on(&sm, vec!["T".into()]);
        write(&a.get(0).unwrap())
            .unwrap()
            .set_value(n0.id(), "T", 1.0)
            .unwrap();
        write(&b.get(0).unwrap())
            .unwrap()
            .set_value(n0.id(), "T", 2.0)
            .unwrap();
        // Distinct handles, same support: the union's finalize fuses them
        // and detects the diverging T — so `|` itself errors.
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn same_support_shared_component_agreeing_is_ok() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let sm = poi1(&coords, &[&n0]);
        let a = field_on(&sm, vec!["T".into(), "P".into()]);
        let b = field_on(&sm, vec!["T".into()]);
        write(&a.get(0).unwrap())
            .unwrap()
            .set_value(n0.id(), "T", 7.0)
            .unwrap();
        write(&b.get(0).unwrap())
            .unwrap()
            .set_value(n0.id(), "T", 7.0)
            .unwrap();
        // Agreeing shared component → `|` succeeds and fuses.
        let c = a.union(&b).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.value(n0.id(), "T").unwrap(), 7.0);
    }

    #[test]
    fn distinct_supports_stay_separate_and_share_handles() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm_a = poi1(&coords, &[&n0]);
        let sm_b = poi1(&coords, &[&n1]);
        let a = field_on(&sm_a, vec!["T".into()]);
        let b = field_on(&sm_b, vec!["P".into()]);
        let f = two_zone(&a, &b);
        let c = consolidate_node(&f).unwrap();
        assert_eq!(c.len(), 2);
        // Singleton groups: handles shared, not copied.
        assert_eq!(c.get(0).unwrap().index(), f.get(0).unwrap().index());
        assert_eq!(c.get(1).unwrap().index(), f.get(1).unwrap().index());
    }

    #[test]
    fn cross_support_shared_node_checked() {
        // Two distinct POI1 supports that both include node n (different
        // submeshes, n shared); a diverging T at n is still an error.
        let coords = insert(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let m = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm_a = poi1(&coords, &[&n, &m]);
        let sm_b = poi1(&coords, &[&n]);
        let a = field_on(&sm_a, vec!["T".into()]);
        let b = field_on(&sm_b, vec!["T".into()]);
        write(&a.get(0).unwrap())
            .unwrap()
            .set_value(n.id(), "T", 1.0)
            .unwrap();
        write(&b.get(0).unwrap())
            .unwrap()
            .set_value(n.id(), "T", 2.0)
            .unwrap();
        // Different supports → no fusion, but the cross-support check in the
        // union's finalize still catches the diverging shared node.
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn empty_field_consolidates_to_empty() {
        let c = consolidate_node(&NodeField::default()).unwrap();
        assert!(c.is_empty());
    }
}
