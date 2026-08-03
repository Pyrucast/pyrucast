use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::field::Field;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, write};

/// Map a coordinate-component name (`"X"`, `"Y"`, `"Z"`) to its axis index.
fn axis_index(name: &str) -> Option<usize> {
    match name {
        "X" => Some(0),
        "Y" => Some(1),
        "Z" => Some(2),
        _ => None,
    }
}

/// Build a [`NodeField`] carrying the coordinates of every node of `mesh`
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
pub fn coordinates(mesh: &Mesh, components: Option<Vec<String>>) -> Result<NodeField> {
    let coords = mesh.coords()?;
    let dim = read(&coords)?.dim() as usize;

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
            let c = read(&coords)?;
            nodes
                .iter()
                .map(|&nid| c.coord(nid).map(|s| s.to_vec()))
                .collect::<Result<_>>()?
        };
        for (ni, coord) in coords.iter().enumerate() {
            for (ci, &axis) in axes.iter().enumerate() {
                sub.set(ni, ci, coord[axis])?;
            }
        }
        out.add_sub(insert(sub))?;
    }
    Ok(out)
}

/// Resolve the per-axis component names: one name per spatial axis, in axis
/// order. `None` falls back to `default[..dim]`. Errors if the count does
/// not match `dim` or if a name is absent from `field` (union of zones).
fn resolve_axis_components(
    field: &NodeField,
    components: Option<Vec<String>>,
    dim: usize,
    default: &[&str],
) -> Result<Vec<String>> {
    let comps = match components {
        Some(c) => c,
        None => default[..dim.min(default.len())]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    if comps.len() != dim {
        return Err(PyrucastError::Message(format!(
            "expected {dim} component name(s) (one per axis), got {}",
            comps.len()
        )));
    }
    let have = Field::components(field)?;
    for name in &comps {
        if !have.iter().any(|c| c == name) {
            return Err(PyrucastError::Message(format!(
                "component '{name}' not in field (has: [{}])",
                have.join(", ")
            )));
        }
    }
    Ok(comps)
}

/// Per distinct node of `field`, the value of each of `comps` — read
/// through the aggregate (first zone wins), so interface nodes appear
/// exactly once. Errors if a node lacks one of the components.
fn per_node_values(field: &NodeField, comps: &[String]) -> Result<Vec<(NodeId, Vec<f64>)>> {
    let nodes = field.node_ids()?;
    let mut out = Vec::with_capacity(nodes.len());
    for nid in nodes {
        let mut values = Vec::with_capacity(comps.len());
        for name in comps {
            values.push(field.value(nid, name)?);
        }
        out.push((nid, values));
    }
    Ok(out)
}

/// **Set** node coordinates from `field` (absolute): for every distinct
/// node `n` of the field, `coord[a] = field.value(n, components[a])` on
/// the active coordinate set. `components` lists one field-component name
/// per spatial axis, in axis order; `None` → `["X", "Y", "Z"][..dim]`
/// (symmetric with [`coordinates`]). In-place on the field's
/// `Coords`.
pub fn set_coordinates(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let coords = field.coords()?;
    let dim = read(&coords)?.dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["X", "Y", "Z"])?;
    let targets = per_node_values(field, &comps)?;
    let mut c = write(&coords)?;
    for (nid, coord) in &targets {
        c.set_coord(*nid, coord)?;
    }
    Ok(())
}

/// **Displace** nodes by `field` (incremental): for every distinct node
/// `n`, `coord[a] += field.value(n, components[a])` on the active
/// coordinate set — an interface node shared by several zones is displaced
/// exactly once. `components` lists one displacement-component name per
/// spatial axis, in axis order; `None` → `["ux", "uy", "uz"][..dim]`.
/// In-place on the field's `Coords`.
pub fn displace(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let coords = field.coords()?;
    let dim = read(&coords)?.dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["ux", "uy", "uz"])?;
    let increments = per_node_values(field, &comps)?;
    let mut c = write(&coords)?;
    for (nid, inc) in &increments {
        let mut coord = c.coord(*nid)?.to_vec();
        for (a, dv) in inc.iter().enumerate() {
            coord[a] += dv;
        }
        c.set_coord(*nid, &coord)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::store::insert;

    #[test]
    fn poi1_mesh_coordinates_xyz() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[4.0, 5.0, 6.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        let f = coordinates(&mesh, None).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.5, 1.0]).unwrap();

        // Two triangles sharing edge (b, c) in one submesh: 4 unique nodes.
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let f = coordinates(&tri, None).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f.node_count().unwrap(), 4, "shared nodes must appear once");
        assert_eq!(f.value(c.id(), "X").unwrap(), 0.5);
        assert_eq!(f.value(c.id(), "Y").unwrap(), 1.0);
        assert_eq!(f.value(d.id(), "X").unwrap(), 1.5);
    }

    #[test]
    fn one_sub_per_submesh_with_coherent_interface() {
        let coords = insert(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        for cell in [[n0.id(), n1.id(), n2.id()], [n1.id(), n3.id(), n2.id()]] {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&cell).unwrap();
            mesh.add_sub(insert(sm)).unwrap();
        }

        let f = coordinates(&mesh, None).unwrap();
        assert_eq!(f.len(), 2, "one sub-field per submesh");
        assert_eq!(f.node_count().unwrap(), 4);
        // Interface nodes are duplicated across zones with equal values.
        f.check().unwrap();
        assert_eq!(f.value(n1.id(), "X").unwrap(), 1.0);
    }

    #[test]
    fn default_components_follow_dimension() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[7.0, 8.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = coordinates(&mesh, None).unwrap();
        assert_eq!(Field::components(&f).unwrap(), vec!["X", "Y"]);
    }

    #[test]
    fn explicit_component_subset() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 2.0, 3.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();

        let f = coordinates(&mesh, Some(vec!["X".into(), "Z".into()])).unwrap();
        assert_eq!(Field::components(&f).unwrap(), vec!["X", "Z"]);
        assert_eq!(f.value(a.id(), "X").unwrap(), 1.0);
        assert_eq!(f.value(a.id(), "Z").unwrap(), 3.0);
    }

    #[test]
    fn rejects_unknown_component() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        assert!(coordinates(&mesh, Some(vec!["W".into()])).is_err());
    }

    #[test]
    fn rejects_axis_beyond_dimension() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        // "Z" needs dim ≥ 3 but the mesh is 2-D.
        assert!(coordinates(&mesh, Some(vec!["Z".into()])).is_err());
    }

    #[test]
    fn rejects_empty_mesh() {
        assert!(coordinates(&Mesh::empty(), None).is_err());
    }

    #[test]
    fn set_coordinates_writes_active_config() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        // Field of target positions (X/Y), round-tripped from the reader.
        let f = coordinates(&mesh, None).unwrap();
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set_value(a.id(), "X", 10.0).unwrap();
            s.set_value(a.id(), "Y", 20.0).unwrap();
        }

        set_coordinates(&f, None).unwrap();
        {
            let c = read(&coords).unwrap();
            assert_eq!(c.coord(a.id()).unwrap(), &[10.0, 20.0]);
            assert_eq!(c.coord(b.id()).unwrap(), &[1.0, 1.0]);
        }
    }

    #[test]
    fn displace_adds_to_active_config() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut d = SubNodeField::from_poi1(&support, vec!["ux".into(), "uy".into()]).unwrap();
        d.set_value(a.id(), "ux", 5.0).unwrap();
        d.set_value(a.id(), "uy", -1.0).unwrap();
        d.set_value(b.id(), "ux", 2.0).unwrap();

        displace(&NodeField::from_sub(d), None).unwrap();
        {
            let c = read(&coords).unwrap();
            assert_eq!(c.coord(a.id()).unwrap(), &[5.0, -1.0]);
            assert_eq!(c.coord(b.id()).unwrap(), &[3.0, 1.0]);
        }
    }

    #[test]
    fn displace_moves_interface_nodes_once() {
        // Two zones sharing node `s`: the increment must apply once, not twice.
        let coords = insert(Coords::new(1).unwrap());
        let s = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::empty();
        for _ in 0..2 {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[s.id()]).unwrap();
            mesh.add_sub(insert(sm)).unwrap();
        }
        let f = NodeField::new(&mesh, vec!["ux".into()]).unwrap();
        for i in 0..2 {
            write(&f.get(i).unwrap())
                .unwrap()
                .set_value(s.id(), "ux", 0.5)
                .unwrap();
        }

        displace(&f, None).unwrap();
        assert_eq!(read(&coords).unwrap().coord(s.id()).unwrap(), &[1.5]);
    }

    #[test]
    fn writers_reject_wrong_component_count() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let f = coordinates(&mesh, None).unwrap();
        // 2-D Coords needs 2 axis components, not 1.
        assert!(set_coordinates(&f, Some(vec!["X".into()])).is_err());
        // Unknown component name.
        assert!(set_coordinates(&f, Some(vec!["X".into(), "BOGUS".into()])).is_err());
    }
}
