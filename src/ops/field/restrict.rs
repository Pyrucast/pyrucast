use crate::containers::mesh::configuration::NodeId;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::store::with;

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new [`NodeField`] with the same components as `field`,
/// supported on the **unique** nodes of `mesh` in order of first
/// appearance. A node of `mesh` that `field` does not cover is assigned
/// `0.0`; a node of `field` absent from `mesh` is dropped.
///
/// Errors if `mesh` is attached to a different `Configuration` than
/// `field`.
pub fn restrict(field: &NodeField, mesh: &Mesh) -> Result<NodeField> {
    let mesh_cfg = mesh.configuration()?;
    let field_cfg = field.configuration();
    if mesh_cfg.index() != field_cfg.index() || mesh_cfg.generation() != field_cfg.generation() {
        return Err(PyrucastError::Message(
            "restrict: mesh is not attached to the same Configuration".into(),
        ));
    }

    // Unique nodes of the mesh, in order of first appearance.
    let mut mesh_nodes: Vec<NodeId> = Vec::new();
    for sm in mesh {
        let connectivity = with(sm, |s| s.connectivity().to_vec())?;
        for nid in connectivity {
            if !mesh_nodes.contains(&nid) {
                mesh_nodes.push(nid);
            }
        }
    }

    let ncomp = field.component_count();
    let mut result =
        NodeField::new_with_nodes(field_cfg, mesh_nodes.clone(), field.components().to_vec())?;
    for (ni, &nid) in mesh_nodes.iter().enumerate() {
        if let Some(src_ni) = field.index_of(nid) {
            for ci in 0..ncomp {
                result.set(ni, ci, field.get(src_ni, ci)?)?;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::configuration::Configuration;
    use crate::containers::mesh::element_type::ElementType;
    use crate::containers::mesh::node::Node;
    use crate::containers::mesh::SubMesh;
    use crate::store::insert;

    /// Build a POI1 field on `n` fresh 1-D nodes; returns (cfg, nodes, field).
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
        let field = NodeField::from_poi1(&insert(sm), components).unwrap();
        (cfg, nodes, field)
    }

    #[test]
    fn restrict_subset() {
        let (cfg, nodes, mut f) = poi1_field(3, vec!["T".into(), "P".into()]);
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 0, 2.0).unwrap();
        f.set(2, 0, 3.0).unwrap();

        // Mesh with only nodes[0] and nodes[2].
        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nodes[2].id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.node_count(), 2);
        assert_eq!(r.components(), &["T", "P"]);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 3.0);
        assert_eq!(r.value(nodes[0].id(), "P").unwrap(), 0.0); // absent → 0
    }

    #[test]
    fn restrict_node_absent_from_field_gives_zero() {
        let (cfg, nodes, mut f) = poi1_field(1, vec!["T".into()]);
        f.set(0, 0, 7.0).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();

        // Mesh contains nb which is NOT in the field.
        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nb.id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.node_count(), 2);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 7.0);
        assert_eq!(r.value(nb.id(), "T").unwrap(), 0.0);
    }

    #[test]
    fn restrict_incompatible_cfg_errors() {
        let (_cfg1, _nodes1, f) = poi1_field(1, vec!["T".into()]);
        // A mesh attached to a *different* Configuration.
        let cfg2 = insert(Configuration::new(1).unwrap());
        let n2 = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        let mut m2 = Mesh::with_element_type(cfg2.clone(), ElementType::POI1);
        m2.add_cell(&[n2.id()]).unwrap();
        assert!(restrict(&f, &m2).is_err());
    }
}
