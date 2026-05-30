use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};

/// Merge two node fields into one defined on the union of their supports.
///
/// The result carries the union of both component lists (those of `a`
/// first, then `b`'s extras) over the union of both node lists (same
/// ordering convention). At a `(node, component)` pair present in only one
/// field, that field's value is kept; present in neither, `0.0`.
///
/// Errors if both fields hold a **different** value at the same
/// `(node, component)` pair, or if they are attached to different
/// `Configuration`s. Equal values at shared points are kept as-is.
pub fn merge(a: &NodeField, b: &NodeField) -> Result<NodeField> {
    a.check_compatible(b)?;
    let (components, nodes) = a.union_layout(b);
    let mut result = NodeField::new_with_nodes(a.configuration(), nodes.clone(), components.clone())?;
    for (ni, &nid) in nodes.iter().enumerate() {
        for (ci, comp) in components.iter().enumerate() {
            let va = a.component_value_opt(nid, comp);
            let vb = b.component_value_opt(nid, comp);
            let v = match (va, vb) {
                (None, None) => 0.0,
                (Some(x), None) => x,
                (None, Some(y)) => y,
                (Some(x), Some(y)) if x == y => x,
                (Some(x), Some(y)) => {
                    return Err(PyrucastError::Message(format!(
                        "merge: conflicting values at node {}, \
                         component \"{}\": {} vs {}",
                        nid, comp, x, y
                    )))
                }
            };
            result.set(ni, ci, v)?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::SubMesh;
    use crate::store::insert;

    /// POI1 field over the given nodes (all attached to `cfg`).
    fn poi1_field(
        cfg: &crate::store::Handle<Configuration>,
        nodes: &[&Node],
        components: Vec<String>,
    ) -> NodeField {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        for nd in nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        NodeField::from_poi1(&insert(sm), components).unwrap()
    }

    #[test]
    fn merge_compatible() {
        // a: [na, nb] T = [5, 3] ; b: [nb, nc] T = [3, 9] (nb shared, equal).
        let cfg = insert(Configuration::new(1).unwrap());
        let na = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let nc = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let mut a = poi1_field(&cfg, &[&na, &nb], vec!["T".into()]);
        let mut b = poi1_field(&cfg, &[&nb, &nc], vec!["T".into()]);
        a.set(0, 0, 5.0).unwrap();
        a.set(1, 0, 3.0).unwrap();
        b.set(0, 0, 3.0).unwrap();
        b.set(1, 0, 9.0).unwrap();

        let c = merge(&a, &b).unwrap();
        assert_eq!(c.node_count(), 3);
        assert_eq!(c.value(na.id(), "T").unwrap(), 5.0);
        assert_eq!(c.value(nb.id(), "T").unwrap(), 3.0);
        assert_eq!(c.value(nc.id(), "T").unwrap(), 9.0);
    }

    #[test]
    fn merge_conflict_errors() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let mut a = poi1_field(&cfg, &[&n], vec!["T".into()]);
        let mut b = poi1_field(&cfg, &[&n], vec!["T".into()]);
        a.set(0, 0, 1.0).unwrap();
        b.set(0, 0, 2.0).unwrap();
        assert!(merge(&a, &b).is_err());
    }

    #[test]
    fn merge_incompatible_cfg_errors() {
        let cfg1 = insert(Configuration::new(1).unwrap());
        let cfg2 = insert(Configuration::new(1).unwrap());
        let n1 = Node::create_in(cfg1.clone(), &[0.0]).unwrap();
        let n2 = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        let a = poi1_field(&cfg1, &[&n1], vec!["T".into()]);
        let b = poi1_field(&cfg2, &[&n2], vec!["T".into()]);
        assert!(merge(&a, &b).is_err());
    }
}
