use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::store::Handle;

/// Map a coordinate-component name (`"X"`, `"Y"`, `"Z"`) to its axis index.
fn axis_index(name: &str) -> Option<usize> {
    match name {
        "X" => Some(0),
        "Y" => Some(1),
        "Z" => Some(2),
        _ => None,
    }
}

/// Build a [`NodeField`] carrying the position of every node of `mesh`
/// — one [`SubNodeField`] per submesh, supported on the distinct nodes of
/// its zone (interface nodes are stored once per zone, with identical
/// values by construction).
///
/// The field has one component per requested axis (`"X"`, `"Y"`, `"Z"`),
/// each holding that node's coordinate in the Coords's active
/// coordinate set. `components = None` requests all the axes the mesh's
/// `Coords` actually has: `["X"]` in 1-D, `["X", "Y"]` in 2-D,
/// `["X", "Y", "Z"]` in 3-D.
///
/// Errors if `mesh` has no submeshes, if a requested component is not one
/// of `"X"` / `"Y"` / `"Z"`, or if it names an axis the Coords does
/// not have (e.g. `"Z"` on a 2-D mesh).
pub fn positions(mesh: &Mesh, components: Option<Vec<String>>) -> Result<NodeField> {
    let coords = mesh.coords()?;
    let dim = coords.read().dim() as usize;

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
                    "positions: unknown component \"{name}\" (expected X, Y or Z)"
                ))
            })?;
            if a >= dim {
                return Err(PyrucastError::Message(format!(
                    "positions: component \"{name}\" needs dimension ≥ {}, \
                     but the Coords is {dim}-D",
                    a + 1
                )));
            }
            Ok(a)
        })
        .collect::<Result<_>>()?;

    let mut out = NodeField::default();
    for sm in mesh {
        let mut sub = SubNodeField::from_support(sm, components.clone())?;
        // Read this zone's coordinates under a single Coords lock.
        let nodes: Vec<NodeId> = sub.nodes().to_vec();
        let coords: Vec<Vec<f64>> = {
            let c = coords.read();
            nodes
                .iter()
                .map(|&nid| c.position(nid).map(|s| s.to_vec()))
                .collect::<Result<_>>()?
        };
        for (ni, coord) in coords.iter().enumerate() {
            for (ci, &axis) in axes.iter().enumerate() {
                sub.set(ni, ci, coord[axis])?;
            }
        }
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::field::Field;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::store::Handle;

    #[test]
    fn poi1_mesh_positions_xyz() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[4.0, 5.0, 6.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        let f = positions(&mesh, None).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(Field::components(&f).unwrap(), vec!["X", "Y", "Z"]);
        assert_eq!(f.node_count().unwrap(), 2);
        assert_eq!(f.value(a.id(), "X").unwrap(), 1.0);
        assert_eq!(f.value(a.id(), "Y").unwrap(), 2.0);
        assert_eq!(f.value(a.id(), "Z").unwrap(), 3.0);
        assert_eq!(f.value(b.id(), "Z").unwrap(), 6.0);
    }

    #[test]
    fn non_poi1_mesh_uses_distinct_nodes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.5, 1.0]).unwrap();

        // Two triangles sharing edge (b, c) in one submesh: 4 unique nodes.
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let f = positions(&tri, None).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f.node_count().unwrap(), 4, "shared nodes must appear once");
        assert_eq!(f.value(c.id(), "X").unwrap(), 0.5);
        assert_eq!(f.value(c.id(), "Y").unwrap(), 1.0);
        assert_eq!(f.value(d.id(), "X").unwrap(), 1.5);
    }

    #[test]
    fn one_sub_per_submesh_with_coherent_interface() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        for cell in [[n0.id(), n1.id(), n2.id()], [n1.id(), n3.id(), n2.id()]] {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&cell).unwrap();
            mesh.add_sub(Handle::new(sm)).unwrap();
        }

        let f = positions(&mesh, None).unwrap();
        assert_eq!(f.len(), 2, "one sub-field per submesh");
        assert_eq!(f.node_count().unwrap(), 4);
        // Interface nodes are duplicated across zones with equal values.
        f.check().unwrap();
        assert_eq!(f.value(n1.id(), "X").unwrap(), 1.0);
    }

    #[test]
    fn default_components_follow_dimension() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[7.0, 8.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = positions(&mesh, None).unwrap();
        assert_eq!(Field::components(&f).unwrap(), vec!["X", "Y"]);
    }

    #[test]
    fn explicit_component_subset() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = positions(&mesh, Some(vec!["X".into(), "Z".into()])).unwrap();
        assert_eq!(Field::components(&f).unwrap(), vec!["X", "Z"]);
        assert_eq!(f.value(a.id(), "X").unwrap(), 1.0);
        assert_eq!(f.value(a.id(), "Z").unwrap(), 3.0);
    }

    #[test]
    fn rejects_unknown_component() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        assert!(positions(&mesh, Some(vec!["W".into()])).is_err());
    }

    #[test]
    fn rejects_axis_beyond_dimension() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        // "Z" needs dim ≥ 3 but the mesh is 2-D.
        assert!(positions(&mesh, Some(vec!["Z".into()])).is_err());
    }

    #[test]
    fn rejects_empty_mesh() {
        assert!(positions(&Mesh::empty(), None).is_err());
    }
}
