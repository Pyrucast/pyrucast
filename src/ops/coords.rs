//! Operators that **write into the coordinates** — the only two that
//! do.
//!
//! They are the write side of
//! [`node_field::positions`](fn@crate::ops::node_field::positions), which
//! reads the geometry as a nodal field: [`set`](fn@set) puts absolute
//! positions back, [`displace`](fn@displace) adds an increment. Both act on
//! the [`Coords`](crate::coords::Coords) the field hangs from, in place.

use crate::atoms::NodeId;
use crate::containers::field::Field;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};

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
/// (symmetric with [`crate::ops::node_field::positions`](fn@crate::ops::node_field::positions)). In-place on the field's
/// `Coords`.
pub fn set(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let coords = field.coords()?;
    let dim = coords.read().dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["X", "Y", "Z"])?;
    let targets = per_node_values(field, &comps)?;
    let mut c = coords.write();
    for (nid, coord) in &targets {
        c.set_position(*nid, coord)?;
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
    let dim = coords.read().dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["ux", "uy", "uz"])?;
    let increments = per_node_values(field, &comps)?;
    let mut c = coords.write();
    for (nid, inc) in &increments {
        let mut coord = c.position(*nid)?.to_vec();
        for (a, dv) in inc.iter().enumerate() {
            coord[a] += dv;
        }
        c.set_position(*nid, &coord)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::node_field::positions;

    #[test]
    fn set_writes_active_config() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        // Field of target positions (X/Y), round-tripped from the reader.
        let f = positions(&mesh, None).unwrap();
        {
            let mut s = f.get(0).unwrap().write();
            s.set_value(a.id(), "X", 10.0).unwrap();
            s.set_value(a.id(), "Y", 20.0).unwrap();
        }

        set(&f, None).unwrap();
        {
            let c = coords.read();
            assert_eq!(c.position(a.id()).unwrap(), &[10.0, 20.0]);
            assert_eq!(c.position(b.id()).unwrap(), &[1.0, 1.0]);
        }
    }

    #[test]
    fn displace_adds_to_active_config() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut d = SubNodeField::from_poi1(&support, vec!["ux".into(), "uy".into()]).unwrap();
        d.set_value(a.id(), "ux", 5.0).unwrap();
        d.set_value(a.id(), "uy", -1.0).unwrap();
        d.set_value(b.id(), "ux", 2.0).unwrap();

        displace(&NodeField::from_sub(d), None).unwrap();
        {
            let c = coords.read();
            assert_eq!(c.position(a.id()).unwrap(), &[5.0, -1.0]);
            assert_eq!(c.position(b.id()).unwrap(), &[3.0, 1.0]);
        }
    }

    #[test]
    fn displace_moves_interface_nodes_once() {
        // Two zones sharing node `s`: the increment must apply once, not twice.
        let coords = Handle::new(Coords::new(1).unwrap());
        let s = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::empty();
        for _ in 0..2 {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[s.id()]).unwrap();
            mesh.add_sub(Handle::new(sm)).unwrap();
        }
        let f = NodeField::new(&mesh, vec!["ux".into()]).unwrap();
        for i in 0..2 {
            f.get(i)
                .unwrap()
                .write()
                .set_value(s.id(), "ux", 0.5)
                .unwrap();
        }

        displace(&f, None).unwrap();
        assert_eq!(coords.read().position(s.id()).unwrap(), &[1.5]);
    }

    #[test]
    fn writers_reject_wrong_component_count() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let f = positions(&mesh, None).unwrap();
        // 2-D Coords needs 2 axis components, not 1.
        assert!(set(&f, Some(vec!["X".into()])).is_err());
        // Unknown component name.
        assert!(set(&f, Some(vec!["X".into(), "BOGUS".into()])).is_err());
    }
}
