//! Consolidate a [`NodeField`]: merge its zones « au plus juste ».
//!
//! Sub-fields sharing the **same component set** (order ignored) are
//! fused into a single [`SubNodeField`] over the union of their nodes —
//! interface nodes duplicated across zones are stored once. Sub-fields
//! with distinct component sets stay separate: nothing is densified, no
//! `0.0` is invented for a `(node, component)` pair no zone defines.
//!
//! Coherence is verified first ([`NodeField::check`]): consolidating a
//! field whose zones disagree on a shared node is an error, never a
//! silent first-wins pick.
//!
//! A group reduced to a single sub-field is **shared** (same handle, no
//! copy) in the result.

use crate::aggregate::Aggregate;
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::containers::mesh::NodeId;
use crate::error::Result;
use crate::store::{insert, read};

/// Merge the zones of `field` that share the same component set.
///
/// See the module documentation. Errors if `field` is incoherent
/// (diverging duplicated interface values) or empty.
pub fn consolidate(field: &NodeField) -> Result<NodeField> {
    field.check()?;
    let cfg = field.configuration()?;

    // Snapshot every sub once (one lock each, never nested), keeping its
    // handle for the singleton-group sharing case.
    struct Snap {
        handle: crate::store::Handle<SubNodeField>,
        nodes: Vec<NodeId>,
        components: Vec<String>,
        values: Vec<f64>,
    }
    let mut snaps: Vec<Snap> = Vec::with_capacity(field.len());
    for h in field {
        let (nodes, components, values) = {
            let s = read(h)?;
            (s.nodes().to_vec(), s.components().to_vec(), s.values().to_vec())
        };
        snaps.push(Snap { handle: h.clone(), nodes, components, values });
    }

    // Group sub indices by component *set* (sorted key), first-seen order.
    let mut groups: Vec<(Vec<String>, Vec<usize>)> = Vec::new();
    for (i, snap) in snaps.iter().enumerate() {
        let mut key = snap.components.clone();
        key.sort();
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, idxs)) => idxs.push(i),
            None => groups.push((key, vec![i])),
        }
    }

    let mut out = NodeField::default();
    for (_, idxs) in &groups {
        if let [single] = idxs.as_slice() {
            // Nothing to fuse: share the sub-field as-is.
            out.add_sub(snaps[*single].handle.clone())?;
            continue;
        }
        // Component order of the group's first sub; union of the nodes.
        let components = snaps[idxs[0]].components.clone();
        let mut nodes: Vec<NodeId> = Vec::new();
        for &i in idxs {
            for &nid in &snaps[i].nodes {
                if !nodes.contains(&nid) {
                    nodes.push(nid);
                }
            }
        }
        let mut fused =
            SubNodeField::new_with_nodes(cfg.clone(), nodes, components.clone())?;
        // Fill order is irrelevant: check() guaranteed duplicates agree.
        for &i in idxs {
            let snap = &snaps[i];
            let ncomp = snap.components.len();
            for (ni, &nid) in snap.nodes.iter().enumerate() {
                for (ci, comp) in snap.components.iter().enumerate() {
                    fused.set_value(nid, comp, snap.values[ni * ncomp + ci])?;
                }
            }
        }
        out.add_sub(insert(fused))?;
    }
    Ok(out)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Configuration, ElementType, Mesh, Node, SubMesh};
    use crate::store::{insert, write, Handle};

    /// Two TRI3 zones sharing an interface edge (nodes n1, n2).
    fn two_zone_mesh() -> (Handle<Configuration>, Vec<Node>, Mesh) {
        let cfg = insert(Configuration::new(2).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        (cfg, vec![n0, n1, n2, n3], mesh)
    }

    #[test]
    fn same_components_fuse_into_one_sub() {
        let (_cfg, nodes, mesh) = two_zone_mesh();
        let f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set_value(nodes[0].id(), "T", 1.0).unwrap();
            s.set_value(nodes[1].id(), "T", 2.0).unwrap();
        }
        {
            let mut s = write(&f.get(1).unwrap()).unwrap();
            s.set_value(nodes[1].id(), "T", 2.0).unwrap(); // interface: same value
            s.set_value(nodes[3].id(), "T", 4.0).unwrap();
        }

        let c = consolidate(&f).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.node_count().unwrap(), 4);
        // interface nodes stored once
        assert_eq!(read(&c.get(0).unwrap()).unwrap().node_count(), 4);
        assert_eq!(c.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(c.value(nodes[1].id(), "T").unwrap(), 2.0);
        assert_eq!(c.value(nodes[3].id(), "T").unwrap(), 4.0);
    }

    #[test]
    fn incoherent_field_errors() {
        let (_cfg, nodes, mesh) = two_zone_mesh();
        let f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_value(nodes[1].id(), "T", 1.0)
            .unwrap();
        write(&f.get(1).unwrap())
            .unwrap()
            .set_value(nodes[1].id(), "T", 2.0)
            .unwrap();
        assert!(consolidate(&f).is_err());
    }

    #[test]
    fn distinct_component_sets_stay_separate_and_share_handles() {
        let (_cfg, _nodes, mesh) = two_zone_mesh();
        let f = NodeField::with(&mesh, &[vec!["T".into()], vec!["P".into()]]).unwrap();
        let c = consolidate(&f).unwrap();
        assert_eq!(c.len(), 2);
        // Singleton groups: handles shared, not copied.
        assert_eq!(c.get(0).unwrap().index(), f.get(0).unwrap().index());
        assert_eq!(c.get(1).unwrap().index(), f.get(1).unwrap().index());
    }

    #[test]
    fn component_set_ignores_order_keeps_first_subs_order() {
        let (_cfg, nodes, mesh) = two_zone_mesh();
        let f = NodeField::with(
            &mesh,
            &[
                vec!["UX".into(), "UY".into()],
                vec!["UY".into(), "UX".into()],
            ],
        )
        .unwrap();
        write(&f.get(1).unwrap())
            .unwrap()
            .set_value(nodes[3].id(), "UY", 9.0)
            .unwrap();
        let c = consolidate(&f).unwrap();
        assert_eq!(c.len(), 1);
        // first sub's order
        assert_eq!(
            read(&c.get(0).unwrap()).unwrap().components(),
            &["UX", "UY"]
        );
        assert_eq!(c.value(nodes[3].id(), "UY").unwrap(), 9.0);
    }

    #[test]
    fn overlapping_component_sets_checked_globally() {
        // ["T"] and ["T", "P"] are different sets (no fusion), but a
        // diverging shared T at the interface is still an error.
        let (_cfg, nodes, mesh) = two_zone_mesh();
        let f = NodeField::with(&mesh, &[vec!["T".into()], vec!["T".into(), "P".into()]])
            .unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_value(nodes[1].id(), "T", 1.0)
            .unwrap();
        write(&f.get(1).unwrap())
            .unwrap()
            .set_value(nodes[1].id(), "T", 2.0)
            .unwrap();
        assert!(consolidate(&f).is_err());
    }

    #[test]
    fn empty_field_errors() {
        assert!(consolidate(&NodeField::default()).is_err());
    }
}
