//! Rigid-body copies of a mesh: [`translate`] and [`rotate`].
//!
//! Both build a **brand-new mesh with its own fresh nodes** — the source
//! mesh and its nodes are left untouched. The result mirrors the input
//! submesh by submesh (same element types, same face colours, same
//! connectivity structure); only the node coordinates are transformed.
//! Nodes shared between cells of the source stay shared in the copy.

use crate::aggregate::Aggregate;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::{Mesh, Node, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};
use std::collections::HashMap;

/// Copy `mesh` into a fresh mesh, applying `f` to every node's coordinates.
///
/// Every distinct source node yields exactly one fresh node (so nodes shared
/// between cells stay shared), placed at `f(old_coord)`. The result keeps the
/// submesh order, element types and face colours of `mesh`; the source mesh
/// is left untouched.
fn map_coords(mesh: &Mesh, mut f: impl FnMut(&[f64]) -> Result<Vec<f64>>) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // One fresh node per distinct source node (first-seen order is irrelevant
    // here since the map is keyed by the original id).
    let mut fresh: HashMap<NodeId, NodeId> = HashMap::new();
    for sm in mesh {
        for &id in read(sm)?.connectivity() {
            if let std::collections::hash_map::Entry::Vacant(e) = fresh.entry(id) {
                let old = read(&coords)?.coord(id)?.to_vec();
                let new_coord = f(&old)?;
                let node = Node::create_in(coords.clone(), &new_coord)?;
                e.insert(node.id());
            }
        }
    }

    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = read(sm_handle)?;
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(color);
        let npc = et.nodes_per_cell();
        for chunk in conn.chunks(npc) {
            let mapped: Vec<NodeId> = chunk.iter().map(|id| fresh[id]).collect();
            new_sm.add_cell(&mapped)?;
        }
        result.add_sub(insert(new_sm))?;
    }
    Ok(result)
}

/// Translate `mesh` by `vector`, returning a fresh copy with its own nodes.
///
/// `vector` must match the mesh's coordinate dimension. The original mesh is
/// left untouched.
pub fn translate(mesh: &Mesh, vector: &[f64]) -> Result<Mesh> {
    let dim = read(&mesh.coords()?)?.dim() as usize;
    if vector.len() != dim {
        return Err(PyrucastError::Message(format!(
            "translate: vector has {} components but the mesh is {}-D",
            vector.len(),
            dim
        )));
    }
    map_coords(mesh, |c| {
        Ok(c.iter().zip(vector).map(|(&x, &d)| x + d).collect())
    })
}

/// Rotate `mesh` by `angle` (radians), returning a fresh copy with its own
/// nodes. The original mesh is left untouched.
///
/// - **2-D** (`center` has 2 components): rotation about the point `center`,
///   counterclockwise for a positive `angle`. `axis` is ignored.
/// - **3-D** (`center` has 3 components): rotation about the line through
///   `center` directed by `axis` (Rodrigues' formula), right-handed about
///   `axis`. `axis` is required and must be non-degenerate; it need not be
///   normalized.
pub fn rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Result<Mesh> {
    let dim = read(&mesh.coords()?)?.dim() as usize;
    if center.len() != dim {
        return Err(PyrucastError::Message(format!(
            "rotate: center has {} components but the mesh is {}-D",
            center.len(),
            dim
        )));
    }
    let (cos, sin) = (angle.cos(), angle.sin());
    match dim {
        2 => {
            let (cx, cy) = (center[0], center[1]);
            map_coords(mesh, |c| {
                let (x, y) = (c[0] - cx, c[1] - cy);
                Ok(vec![cx + cos * x - sin * y, cy + sin * x + cos * y])
            })
        }
        3 => {
            let axis = axis.ok_or_else(|| {
                PyrucastError::Message("rotate: a 3-D mesh needs a rotation axis".into())
            })?;
            if axis.len() != 3 {
                return Err(PyrucastError::Message(format!(
                    "rotate: axis has {} components but must be 3-D",
                    axis.len()
                )));
            }
            let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if norm == 0.0 {
                return Err(PyrucastError::Message(
                    "rotate: axis must be non-zero".into(),
                ));
            }
            let u = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
            let (cx, cy, cz) = (center[0], center[1], center[2]);
            map_coords(mesh, |c| {
                let v = [c[0] - cx, c[1] - cy, c[2] - cz];
                // Rodrigues: v' = v cosθ + (u×v) sinθ + u (u·v)(1-cosθ).
                let cross = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
                let k = dot * (1.0 - cos);
                Ok(vec![
                    cx + v[0] * cos + cross[0] * sin + u[0] * k,
                    cy + v[1] * cos + cross[1] * sin + u[1] * k,
                    cz + v[2] * cos + cross[2] * sin + u[2] * k,
                ])
            })
        }
        other => Err(PyrucastError::Message(format!(
            "rotate: only 2-D and 3-D meshes are supported (got {other}-D)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::ElementType;
    use crate::store::insert;
    use std::f64::consts::PI;

    fn tri2d(coords: &crate::store::Handle<Coords>, pts: [[f64; 2]; 3]) -> (Mesh, [Node; 3]) {
        let n0 = Node::create_in(coords.clone(), &pts[0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &pts[1]).unwrap();
        let n2 = Node::create_in(coords.clone(), &pts[2]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        (m, [n0, n1, n2])
    }

    #[test]
    fn translate_makes_fresh_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, src) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let out = translate(&m, &[10.0, 5.0]).unwrap();
        assert_eq!(out.element_types().unwrap(), vec![ElementType::TRI3]);
        // Fresh node, distinct id from the source.
        let n0 = out.node(0, 0, 0).unwrap();
        assert_ne!(n0.id(), src[0].id());
        assert_eq!(n0.coord().unwrap(), vec![10.0, 5.0]);
        assert_eq!(out.node(0, 0, 1).unwrap().coord().unwrap(), vec![11.0, 5.0]);
        // Source is untouched.
        assert_eq!(src[0].coord().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn translate_keeps_shared_nodes_shared() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        m.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let out = translate(&m, &[0.0, 0.0]).unwrap();
        // b appears in both cells; the copy must reuse one fresh node for it.
        assert_eq!(
            out.node(0, 0, 1).unwrap().id(),
            out.node(0, 1, 0).unwrap().id()
        );
    }

    #[test]
    fn rotate_2d_quarter_turn() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]);

        let out = rotate(&m, PI / 2.0, &[0.0, 0.0], None).unwrap();
        let n0 = out.node(0, 0, 0).unwrap().coord().unwrap();
        assert!((n0[0] - 0.0).abs() < 1e-12 && (n0[1] - 1.0).abs() < 1e-12);
        let n1 = out.node(0, 0, 1).unwrap().coord().unwrap();
        assert!((n1[0] - 0.0).abs() < 1e-12 && (n1[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rotate_3d_about_z() {
        let coords = insert(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 5.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[0.0, 1.0, 5.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 0.0, 5.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();

        let out = rotate(&m, PI / 2.0, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        let p0 = out.node(0, 0, 0).unwrap().coord().unwrap();
        // (1,0,5) rotated +90° about z → (0,1,5).
        assert!(
            (p0[0]).abs() < 1e-12 && (p0[1] - 1.0).abs() < 1e-12 && (p0[2] - 5.0).abs() < 1e-12
        );
    }

    #[test]
    fn rotate_3d_needs_axis() {
        let coords = insert(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[n0.id()]).unwrap();
        assert!(rotate(&m, 1.0, &[0.0, 0.0, 0.0], None).is_err());
    }

    #[test]
    fn translate_rejects_wrong_dim() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(translate(&m, &[1.0, 2.0, 3.0]).is_err());
    }
}
