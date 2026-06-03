use crate::error::{PyrucastError, Result};
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::NodeField;
use crate::store::{insert, with, with_mut};

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
/// otherwise it is first converted with [`crate::ops::mesher::to_poi1()`].
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
    let support = insert(SubMesh::poi1_from_node_ids(cfg.clone(), &nodes)?);
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

/// Resolve the per-axis component names: one name per spatial axis, in axis
/// order. `None` falls back to `default[..dim]`. Errors if the count does
/// not match `dim` or if a name is absent from `field`.
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
    let have = field.components();
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

/// **Set** node coordinates from `field` (absolute): for every node `n` of
/// the field, `coord[a] = field.value(n, components[a])` on the active
/// coordinate set. `components` lists one field-component name per spatial
/// axis, in axis order; `None` → `["X", "Y", "Z"][..dim]` (symmetric with
/// [`coordinates`]). In-place on the field's `Configuration`.
pub fn set_coordinates(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let cfg = field.configuration();
    let dim = with(&cfg, |c| c.dim())? as usize;
    let comps = resolve_axis_components(field, components, dim, &["X", "Y", "Z"])?;
    with_mut(&cfg, |c| -> Result<()> {
        for &nid in field.nodes() {
            let mut coord = Vec::with_capacity(dim);
            for name in &comps {
                coord.push(field.value(nid, name)?);
            }
            c.set_coord(nid, &coord)?;
        }
        Ok(())
    })?
}

/// **Displace** nodes by `field` (incremental): for every node `n`,
/// `coord[a] += field.value(n, components[a])` on the active coordinate
/// set. `components` lists one displacement-component name per spatial
/// axis, in axis order; `None` → `["ux", "uy", "uz"][..dim]`. In-place on
/// the field's `Configuration`.
pub fn displace(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let cfg = field.configuration();
    let dim = with(&cfg, |c| c.dim())? as usize;
    let comps = resolve_axis_components(field, components, dim, &["ux", "uy", "uz"])?;
    with_mut(&cfg, |c| -> Result<()> {
        for &nid in field.nodes() {
            let mut coord = c.coord(nid)?.to_vec();
            for (a, name) in comps.iter().enumerate() {
                coord[a] += field.value(nid, name)?;
            }
            c.set_coord(nid, &coord)?;
        }
        Ok(())
    })?
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

    #[test]
    fn set_coordinates_writes_active_set() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        // Field of target positions (X/Y), round-tripped from the reader.
        let mut f = coordinates(&mesh, None).unwrap();
        f.set_value(a.id(), "X", 10.0).unwrap();
        f.set_value(a.id(), "Y", 20.0).unwrap();

        set_coordinates(&f, None).unwrap();
        with(&cfg, |c| {
            assert_eq!(c.coord(a.id()).unwrap(), &[10.0, 20.0]);
            assert_eq!(c.coord(b.id()).unwrap(), &[1.0, 1.0]);
        })
        .unwrap();
    }

    #[test]
    fn displace_adds_to_active_set() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut d = NodeField::from_poi1(&support, vec!["ux".into(), "uy".into()]).unwrap();
        d.set_value(a.id(), "ux", 5.0).unwrap();
        d.set_value(a.id(), "uy", -1.0).unwrap();
        d.set_value(b.id(), "ux", 2.0).unwrap();

        displace(&d, None).unwrap();
        with(&cfg, |c| {
            assert_eq!(c.coord(a.id()).unwrap(), &[5.0, -1.0]);
            assert_eq!(c.coord(b.id()).unwrap(), &[3.0, 1.0]);
        })
        .unwrap();
    }

    #[test]
    fn writers_reject_wrong_component_count() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let f = coordinates(&mesh, None).unwrap();
        // 2-D Configuration needs 2 axis components, not 1.
        assert!(set_coordinates(&f, Some(vec!["X".into()])).is_err());
        // Unknown component name.
        assert!(set_coordinates(&f, Some(vec!["X".into(), "BOGUS".into()])).is_err());
    }
}
