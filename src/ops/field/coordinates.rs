use crate::error::{PyrucastError, Result};
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::NodeField;
use crate::store::{insert, with};

/// Map a coordinate-component name (`"X"`, `"Y"`, `"Z"`) to its axis index.
fn axis_index(name: &str) -> Option<usize> {
    match name {
        "X" => Some(0),
        "Y" => Some(1),
        "Z" => Some(2),
        _ => None,
    }
}

/// Build a [`NodeField`] carrying the coordinates of every node of `mesh`.
///
/// The field has one component per requested axis (`"X"`, `"Y"`, `"Z"`),
/// each holding that node's coordinate in the Configuration's active
/// coordinate set. `components = None` requests all the axes the mesh's
/// `Configuration` actually has: `["X"]` in 1-D, `["X", "Y"]` in 2-D,
/// `["X", "Y", "Z"]` in 3-D.
///
/// If `mesh` is already entirely POI1 its nodes are read directly;
/// otherwise it is first converted with [`crate::ops::mesher::to_poi1`].
/// Either way the field support is the **unique** nodes of `mesh`, in
/// order of first appearance.
///
/// Errors if `mesh` has no submeshes, if a requested component is not one
/// of `"X"` / `"Y"` / `"Z"`, or if it names an axis the Configuration does
/// not have (e.g. `"Z"` on a 2-D mesh).
pub fn coordinates(mesh: &Mesh, components: Option<Vec<String>>) -> Result<NodeField> {
    let cfg = mesh.configuration()?;
    let dim = with(&cfg, |c| c.dim())? as usize;

    // Default component list = the axes present in this dimension.
    let components = match components {
        Some(c) => c,
        None => ["X", "Y", "Z"][..dim.min(3)]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };

    // Resolve and validate every requested component up front.
    let axes: Vec<usize> = components
        .iter()
        .map(|name| {
            let a = axis_index(name).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "coordinates: unknown component \"{name}\" (expected X, Y or Z)"
                ))
            })?;
            if a >= dim {
                return Err(PyrucastError::Message(format!(
                    "coordinates: component \"{name}\" needs dimension ≥ {}, \
                     but the Configuration is {dim}-D",
                    a + 1
                )));
            }
            Ok(a)
        })
        .collect::<Result<_>>()?;

    // Gather the unique node list. POI1 mesh ⇒ read it directly; otherwise
    // convert to POI1 first (function 1), then read the conversion.
    let all_poi1 = mesh
        .element_types()?
        .iter()
        .all(|&et| et == ElementType::POI1);
    let nodes = if all_poi1 {
        unique_nodes(mesh)?
    } else {
        unique_nodes(&crate::ops::mesher::to_poi1(mesh)?)?
    };

    // Build a single POI1 support holding those nodes, then the field.
    let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    for &nid in &nodes {
        sm.add_cell(&[nid])?;
    }
    let support = insert(sm);
    let mut field = NodeField::from_poi1(&support, components)?;

    // Read all coordinates under a single Configuration lock, then fill.
    let coords: Vec<Vec<f64>> = with(&cfg, |c| -> Result<Vec<Vec<f64>>> {
        nodes
            .iter()
            .map(|&nid| c.coord(nid).map(|s| s.to_vec()))
            .collect()
    })??;
    for (ni, coord) in coords.iter().enumerate() {
        for (ci, &axis) in axes.iter().enumerate() {
            field.set(ni, ci, coord[axis])?;
        }
    }

    Ok(field)
}

/// Unique nodes used across every submesh of `mesh`, in order of first
/// appearance.
fn unique_nodes(mesh: &Mesh) -> Result<Vec<NodeId>> {
    let mut nodes: Vec<NodeId> = Vec::new();
    for sm in mesh {
        let conn = with(sm, |s| s.connectivity().to_vec())?;
        for nid in conn {
            if !nodes.contains(&nid) {
                nodes.push(nid);
            }
        }
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::Mesh;
    use crate::store::insert;

    #[test]
    fn poi1_mesh_coordinates_xyz() {
        let cfg = insert(Configuration::new(3).unwrap());
        let a = Node::create_in(cfg.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[4.0, 5.0, 6.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        let f = coordinates(&mesh, None).unwrap();
        assert_eq!(f.components(), &["X", "Y", "Z"]);
        assert_eq!(f.node_count(), 2);
        assert_eq!(f.value(a.id(), "X").unwrap(), 1.0);
        assert_eq!(f.value(a.id(), "Y").unwrap(), 2.0);
        assert_eq!(f.value(a.id(), "Z").unwrap(), 3.0);
        assert_eq!(f.value(b.id(), "Z").unwrap(), 6.0);
    }

    #[test]
    fn non_poi1_mesh_is_converted_and_deduplicated() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let d = Node::create_in(cfg.clone(), &[1.5, 1.0]).unwrap();

        // Two triangles sharing edge (b, c): 4 unique nodes.
        let mut tri = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let f = coordinates(&tri, None).unwrap();
        assert_eq!(f.components(), &["X", "Y"]);
        assert_eq!(f.node_count(), 4, "shared nodes must appear once");
        assert_eq!(f.value(c.id(), "X").unwrap(), 0.5);
        assert_eq!(f.value(c.id(), "Y").unwrap(), 1.0);
        assert_eq!(f.value(d.id(), "X").unwrap(), 1.5);
    }

    #[test]
    fn default_components_follow_dimension() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[7.0, 8.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = coordinates(&mesh, None).unwrap();
        assert_eq!(f.components(), &["X", "Y"]);
    }

    #[test]
    fn explicit_component_subset() {
        let cfg = insert(Configuration::new(3).unwrap());
        let a = Node::create_in(cfg.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = coordinates(&mesh, Some(vec!["X".into(), "Z".into()])).unwrap();
        assert_eq!(f.components(), &["X", "Z"]);
        assert_eq!(f.value(a.id(), "X").unwrap(), 1.0);
        assert_eq!(f.value(a.id(), "Z").unwrap(), 3.0);
    }

    #[test]
    fn rejects_unknown_component() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        assert!(coordinates(&mesh, Some(vec!["W".into()])).is_err());
    }

    #[test]
    fn rejects_axis_beyond_dimension() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        // "Z" needs dim ≥ 3 but the mesh is 2-D.
        assert!(coordinates(&mesh, Some(vec!["Z".into()])).is_err());
    }

    #[test]
    fn rejects_empty_mesh() {
        assert!(coordinates(&Mesh::empty(), None).is_err());
    }
}
