//! Nearest-node query — the closest mesh node to a physical point.
//!
//! Where [`super::locate_points`] inverts the iso-parametric map to find the
//! *cell* a point sits inside, [`nearest_node`] answers the simpler, purely
//! nodal question: of all the nodes the mesh references, which one is closest
//! (in Euclidean distance) to the given coordinates? It is the natural way to
//! pick a node to pin a boundary condition or read a result at, when you know
//! roughly *where* it is but not its id.

use crate::atoms::{Node, NodeId};
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::store::read;
use std::collections::HashSet;

/// Return the mesh node closest (Euclidean distance) to `point`.
///
/// `point` must have the mesh `Coords` spatial dimension. Only nodes actually
/// referenced by a cell of the mesh are considered; ties are broken by the
/// smaller `NodeId` (so the result is independent of iteration order).
///
/// Returns an error if the mesh has no submeshes or references no nodes, or if
/// `point`'s length does not match the coordinate dimension.
pub fn nearest_node(mesh: &Mesh, point: &[f64]) -> Result<Node> {
    let coords_handle = mesh.coords()?;

    // Gather the unique node ids the mesh references, across all submeshes.
    let mut seen: HashSet<NodeId> = HashSet::new();
    for sm in mesh {
        let s = read(sm)?;
        for &nid in s.connectivity() {
            seen.insert(nid);
        }
    }

    let best = {
        let c = read(&coords_handle)?;
        if point.len() != c.dim() as usize {
            return Err(PyrucastError::Message(format!(
                "nearest_node: point has {} coordinates, mesh is {}-D",
                point.len(),
                c.dim()
            )));
        }
        let mut best: Option<(NodeId, f64)> = None;
        for &nid in &seen {
            let x = c.coord(nid)?;
            let d2: f64 = x.iter().zip(point).map(|(a, b)| (a - b) * (a - b)).sum();
            // Strictly-less keeps the first (smallest id) on a tie, but `seen`
            // is a set with no stable order, so compare ids explicitly.
            match best {
                Some((bid, bd2)) if bd2 < d2 || (bd2 == d2 && bid.0 <= nid.0) => {}
                _ => best = Some((nid, d2)),
            }
        }
        best
    };

    let (nid, _) = best
        .ok_or_else(|| PyrucastError::Message("nearest_node: mesh references no nodes".into()))?;
    Node::acquire(coords_handle, nid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::store::insert;

    #[test]
    fn nearest_on_grid() {
        let coords = insert(Coords::new(2).unwrap());
        let n00 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n10 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n11 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n00.id(), n10.id(), n11.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Closest to a point just past the far corner is n11.
        let found = nearest_node(&mesh, &[0.9, 0.9]).unwrap();
        assert_eq!(found.id(), n11.id());

        // Closest to the origin is n00.
        let found = nearest_node(&mesh, &[-0.2, 0.1]).unwrap();
        assert_eq!(found.id(), n00.id());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);
        assert!(nearest_node(&mesh, &[0.0, 0.0, 0.0]).is_err());
    }
}
