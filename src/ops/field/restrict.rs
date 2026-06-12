use crate::aggregate::Aggregate;
use crate::containers::field::Field;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::store::insert;

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new [`NodeField`] with one zone per submesh of `mesh`, each
/// supported on the distinct nodes of its zone, carrying the union of
/// `field`'s components. Values are read through the aggregate (first
/// zone of `field` defining the pair wins); a node of `mesh` that
/// `field` does not cover is assigned `0.0`; a node of `field` absent
/// from `mesh` is dropped.
///
/// Errors if `mesh` is attached to a different `Configuration` than
/// `field`.
pub fn restrict(field: &NodeField, mesh: &Mesh) -> Result<NodeField> {
    let mesh_cfg = mesh.configuration()?;
    let field_cfg = field.configuration()?;
    if mesh_cfg.index() != field_cfg.index() || mesh_cfg.generation() != field_cfg.generation() {
        return Err(PyrucastError::Message(
            "restrict: mesh is not attached to the same Configuration".into(),
        ));
    }

    let components = Field::components(field)?;
    let view = field.view()?;
    let mut out = NodeField::default();
    for sm in mesh {
        let mut sub = SubNodeField::from_support(sm, components.clone())?;
        let nodes = sub.nodes().to_vec();
        for nid in nodes {
            for comp in &components {
                if let Some(v) = view.value_opt(nid, comp) {
                    sub.set_value(nid, comp, v)?;
                }
            }
        }
        out.add_sub(insert(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::SubMesh;
    use crate::store::{insert, write};

    /// Build a single-zone POI1 field on `n` fresh 1-D nodes;
    /// returns (cfg, nodes, field).
    fn poi1_field(n: usize, components: Vec<String>) -> (
        crate::store::Handle<Configuration>,
        Vec<Node>,
        NodeField,
    ) {
        let cfg = insert(Configuration::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(cfg.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        for nd in &nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        let field = NodeField::from_sub(
            SubNodeField::from_poi1(&insert(sm), components).unwrap(),
        );
        (cfg, nodes, field)
    }

    #[test]
    fn restrict_subset() {
        let (cfg, nodes, f) = poi1_field(3, vec!["T".into(), "P".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set(0, 0, 1.0).unwrap();
            s.set(1, 0, 2.0).unwrap();
            s.set(2, 0, 3.0).unwrap();
        }

        // Mesh with only nodes[0] and nodes[2].
        let mut m = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nodes[2].id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.node_count().unwrap(), 2);
        assert_eq!(Field::components(&r).unwrap(), vec!["T", "P"]);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 3.0);
        assert_eq!(r.value(nodes[0].id(), "P").unwrap(), 0.0); // absent → 0
        assert_eq!(r.value_opt(nodes[1].id(), "T").unwrap(), None); // dropped
    }

    #[test]
    fn restrict_node_absent_from_field_gives_zero() {
        let (cfg, nodes, f) = poi1_field(1, vec!["T".into()]);
        write(&f.get(0).unwrap()).unwrap().set(0, 0, 7.0).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();

        // Mesh contains nb which is NOT in the field.
        let mut m = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nb.id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.node_count().unwrap(), 2);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 7.0);
        assert_eq!(r.value(nb.id(), "T").unwrap(), 0.0);
    }

    #[test]
    fn restrict_one_zone_per_mesh_submesh() {
        let (cfg, nodes, f) = poi1_field(2, vec!["T".into()]);
        let mut mesh = Mesh::empty();
        for nd in &nodes {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[nd.id()]).unwrap();
            mesh.add_sub(insert(sm)).unwrap();
        }
        let r = restrict(&f, &mesh).unwrap();
        assert_eq!(r.len(), 2, "one zone per submesh of the target mesh");
    }

    #[test]
    fn restrict_incompatible_cfg_errors() {
        let (_cfg1, _nodes1, f) = poi1_field(1, vec!["T".into()]);
        // A mesh attached to a *different* Configuration.
        let cfg2 = insert(Configuration::new(1).unwrap());
        let n2 = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        let mut m2 = Mesh::from_submesh(SubMesh::new(cfg2.clone(), ElementType::POI1));
        m2.add_cell(&[n2.id()]).unwrap();
        assert!(restrict(&f, &m2).is_err());
    }
}
