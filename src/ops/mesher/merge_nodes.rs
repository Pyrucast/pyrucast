use crate::aggregate::Aggregate;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::{Coords, Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};
use std::collections::HashMap;

/// Weld together nodes closer than `tol` (Euclidean distance), rewriting the
/// connectivity to refer to a single representative per cluster.
///
/// The result mirrors `mesh`: same submeshes, in the same order, each keeping
/// its element type and face colour, but with every reference to a welded-away
/// node redirected to its cluster representative. The representative is the
/// node with the **smallest [`NodeId`]** in the cluster, and it **keeps its own
/// coordinates** (no averaging — a deliberate choice, like the rest of the
/// pipeline, to avoid silently moving geometry). Welded-away nodes are left in
/// the shared [`Coords`]; once nothing references them they become collectable
/// by [`Coords::gc`](crate::containers::mesh::Coords::gc).
///
/// Cells that collapse — i.e. reference the same representative more than once
/// after welding (a SEG2 whose two ends merge, a TRI3 with two coincident
/// corners, …) — are **dropped**, since they are degenerate. POI1 cells, being
/// single nodes, never collapse and are always kept (de-duplicating colocated
/// points is [`consolidate`](fn@crate::ops::mesher::consolidate)'s job, not this
/// one's).
///
/// Every node referenced by the result is increfed afresh by the new
/// submeshes; `mesh` itself is left untouched. Errors if `tol` is negative or
/// if `mesh` has no submeshes (no Coords to attach to).
pub fn merge_nodes(mesh: &Mesh, tol: f64) -> Result<Mesh> {
    if tol < 0.0 {
        return Err(PyrucastError::Message(format!(
            "merge_nodes: tol must be ≥ 0, got {tol}"
        )));
    }
    let coords_handle = mesh.coords()?;

    // Map every referenced node to its cluster representative.
    let representative = build_representatives(mesh, &coords_handle, tol)?;

    // Rebuild every submesh with remapped connectivity, dropping degenerate
    // cells.
    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = read(sm_handle)?;
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let npc = et.nodes_per_cell();

        let mut new_sm = SubMesh::new(coords_handle.clone(), et);
        new_sm.set_face_color(color);

        for chunk in conn.chunks(npc) {
            let mapped: Vec<NodeId> = chunk
                .iter()
                .map(|n| representative.get(n).copied().unwrap_or(*n))
                .collect();
            if is_degenerate(&mapped) {
                continue;
            }
            new_sm.add_cell(&mapped)?;
        }

        result.add_sub(insert(new_sm))?;
    }

    Ok(result)
}

/// Assign each referenced node a representative id via a uniform spatial grid
/// of cell size `tol`. Two points within `tol` differ by at most `tol` on each
/// axis, i.e. at most one grid cell, so scanning the 3^dim neighbourhood finds
/// every candidate. Nodes are processed in ascending id order, so the
/// representative of a cluster is always its smallest id.
fn build_representatives(
    mesh: &Mesh,
    coords_handle: &crate::store::Handle<Coords>,
    tol: f64,
) -> Result<HashMap<NodeId, NodeId>> {
    // Unique referenced ids, ascending.
    let mut ids: Vec<NodeId> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for sm_handle in mesh {
            for &n in read(sm_handle)?.connectivity() {
                if seen.insert(n) {
                    ids.push(n);
                }
            }
        }
    }
    ids.sort_by_key(|n| n.0);

    let coords = read(coords_handle)?;
    let dim = coords.dim() as usize;
    // A positive cell size even when tol == 0 (then only exactly coincident
    // points share a cell, and the tol² test keeps only true duplicates).
    let cell = if tol > 0.0 { tol } else { 1.0 };
    let tol2 = tol * tol;

    let offsets = neighbour_offsets(dim);
    // Grid cell key → representatives already placed there.
    let mut grid: HashMap<Vec<i64>, Vec<NodeId>> = HashMap::new();
    let mut representative: HashMap<NodeId, NodeId> = HashMap::new();

    for &id in &ids {
        let p = coords.coord(id)?;
        let base: Vec<i64> = p.iter().map(|&x| (x / cell).floor() as i64).collect();

        let mut rep = None;
        'search: for off in &offsets {
            let key: Vec<i64> = base.iter().zip(off).map(|(b, o)| b + o).collect();
            if let Some(candidates) = grid.get(&key) {
                for &c in candidates {
                    if dist2(p, coords.coord(c)?) <= tol2 {
                        rep = Some(c);
                        break 'search;
                    }
                }
            }
        }

        match rep {
            Some(r) => {
                representative.insert(id, r);
            }
            None => {
                // New representative: it stands for itself and joins the grid.
                representative.insert(id, id);
                grid.entry(base).or_default().push(id);
            }
        }
    }

    Ok(representative)
}

/// Squared Euclidean distance between two coordinate slices of equal length.
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Whether a cell references the same node twice or more (after welding).
fn is_degenerate(nodes: &[NodeId]) -> bool {
    nodes
        .iter()
        .enumerate()
        .any(|(i, n)| nodes[i + 1..].contains(n))
}

/// All offsets in `{-1, 0, 1}^dim` (the 3^dim grid neighbourhood).
fn neighbour_offsets(dim: usize) -> Vec<Vec<i64>> {
    let mut offsets = vec![Vec::new()];
    for _ in 0..dim {
        let mut next = Vec::with_capacity(offsets.len() * 3);
        for prefix in &offsets {
            for d in [-1i64, 0, 1] {
                let mut v = prefix.clone();
                v.push(d);
                next.push(v);
            }
        }
        offsets = next;
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;

    fn coords2() -> crate::store::Handle<Coords> {
        insert(Coords::new(2).unwrap())
    }

    #[test]
    fn welds_two_coincident_corners() {
        let coords = coords2();
        // Two triangles sharing an edge, but the shared edge is described by
        // two pairs of *distinct* but nearly coincident nodes.
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0 + 1e-9, 0.0]).unwrap();
        let c2 = Node::create_in(coords.clone(), &[0.5, 1.0 - 1e-9]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.5, 1.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        mesh.add_cell(&[b2.id(), d.id(), c2.id()]).unwrap();

        let merged = merge_nodes(&mesh, 1e-6).unwrap();
        assert_eq!(merged.cell_count().unwrap(), 2, "both triangles survive");

        // b2 → b and c2 → c, so the second triangle now uses b and c.
        let tri1: Vec<_> = (0..3).map(|i| merged.node(0, 1, i).unwrap().id()).collect();
        assert_eq!(tri1, vec![b.id(), d.id(), c.id()]);
    }

    #[test]
    fn drops_degenerate_cell() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        // c sits right on top of b → the SEG2 (b, c) collapses.
        let c = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[b.id(), c.id()]).unwrap();

        let merged = merge_nodes(&mesh, 1e-6).unwrap();
        assert_eq!(
            merged.cell_count().unwrap(),
            1,
            "the (b,c) segment is dropped"
        );
    }

    #[test]
    fn representative_is_smallest_id_and_keeps_its_coords() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[5.0, 5.0]).unwrap();
        // c is near b but with a higher id → b stays, c is welded onto it.
        let c = Node::create_in(coords.clone(), &[5.0 + 1e-9, 5.0]).unwrap();

        // Both b and c are referenced; b has the smaller id so it stays.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[a.id(), c.id()]).unwrap();

        let merged = merge_nodes(&mesh, 1e-6).unwrap();
        let welded = merged.node(0, 1, 1).unwrap();
        assert_eq!(welded.id(), b.id());
        assert_eq!(read(&coords).unwrap().coord(b.id()).unwrap(), &[5.0, 5.0]);
    }

    #[test]
    fn tol_zero_welds_only_exact_duplicates() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let exact = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let near = Node::create_in(coords.clone(), &[1.0 + 1e-9, 0.0]).unwrap();

        // All four nodes are referenced. a, b are the smaller ids (reps);
        // exact sits exactly on a, near sits 1e-9 from b.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), near.id()]).unwrap();
        mesh.add_cell(&[b.id(), exact.id()]).unwrap();

        let merged = merge_nodes(&mesh, 0.0).unwrap();
        // exact → a (distance 0), but near stays distinct from b (distance 1e-9 > 0).
        assert_eq!(merged.node(0, 0, 1).unwrap().id(), near.id());
        assert_eq!(merged.node(0, 1, 1).unwrap().id(), a.id());
        assert_ne!(near.id(), b.id());
    }

    #[test]
    fn negative_tol_is_error() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        assert!(merge_nodes(&mesh, -1.0).is_err());
    }

    #[test]
    fn leaves_input_untouched_and_increfs_result() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();

        // before: a in SEG2 + Node = 2.
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
        let merged = merge_nodes(&mesh, 1e-6).unwrap();
        // +1 from the result submesh.
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 3);
        drop(merged);
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
    }
}
