use crate::aggregate::Aggregate;
use crate::containers::node_field::NodeField;
use crate::error::Result;

/// Merge two node fields « au plus juste »: the union of their zones.
///
/// This is exactly [`Aggregate::union`] (Python's `a | b`): zones sharing
/// the same support `SubMesh` are fused over the union of their components,
/// and a `(node, component)` pair stored by both must hold the **same**
/// value (exact comparison) — anything else is an error. Distinct supports
/// stay separate; nothing is densified.
///
/// Errors on a value conflict or if the fields are attached to different
/// `Configuration`s.
pub fn merge(a: &NodeField, b: &NodeField) -> Result<NodeField> {
    a.union(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::field::Field;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::SubMesh;
    use crate::containers::node_field::SubNodeField;
    use crate::store::insert;

    /// Single-zone POI1 field over the given nodes (attached to `cfg`).
    fn poi1_field(
        cfg: &crate::store::Handle<Configuration>,
        nodes: &[&Node],
        components: Vec<String>,
    ) -> NodeField {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        for nd in nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        NodeField::from_sub(SubNodeField::from_poi1(&insert(sm), components).unwrap())
    }

    #[test]
    fn merge_compatible() {
        // a: [na, nb] T = [5, 3] ; b: [nb, nc] T = [3, 9] (nb shared, equal).
        let cfg = insert(Configuration::new(1).unwrap());
        let na = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let nc = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let a = poi1_field(&cfg, &[&na, &nb], vec!["T".into()]);
        let b = poi1_field(&cfg, &[&nb, &nc], vec!["T".into()]);
        {
            let mut s = crate::store::write(&a.get(0).unwrap()).unwrap();
            s.set(0, 0, 5.0).unwrap();
            s.set(1, 0, 3.0).unwrap();
        }
        {
            let mut s = crate::store::write(&b.get(0).unwrap()).unwrap();
            s.set(0, 0, 3.0).unwrap();
            s.set(1, 0, 9.0).unwrap();
        }

        let c = merge(&a, &b).unwrap();
        // Different support SubMeshes ⇒ no fusion (fusion is by support
        // handle, not by component set); the shared interface node nb is
        // checked and agrees.
        assert_eq!(c.len(), 2, "distinct supports stay separate");
        assert_eq!(c.node_count().unwrap(), 3);
        assert_eq!(c.value(na.id(), "T").unwrap(), 5.0);
        assert_eq!(c.value(nb.id(), "T").unwrap(), 3.0);
        assert_eq!(c.value(nc.id(), "T").unwrap(), 9.0);
    }

    #[test]
    fn merge_distinct_component_sets_stay_separate() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let a = poi1_field(&cfg, &[&n], vec!["T".into()]);
        let b = poi1_field(&cfg, &[&n], vec!["P".into()]);
        let c = merge(&a, &b).unwrap();
        assert_eq!(c.len(), 2, "distinct supports ⇒ separate zones");
        assert_eq!(Field::components(&c).unwrap(), vec!["T", "P"]);
    }

    #[test]
    fn merge_conflict_errors() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let a = poi1_field(&cfg, &[&n], vec!["T".into()]);
        let b = poi1_field(&cfg, &[&n], vec!["T".into()]);
        crate::store::write(&a.get(0).unwrap())
            .unwrap()
            .set(0, 0, 1.0)
            .unwrap();
        crate::store::write(&b.get(0).unwrap())
            .unwrap()
            .set(0, 0, 2.0)
            .unwrap();
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
