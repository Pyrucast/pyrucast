use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, write};
use std::collections::HashMap;

/// Weld together nodes closer than `tol` (Euclidean distance), rewriting the
/// connectivity to refer to a single representative per cluster.
///
/// Each cluster is represented by the node with the **smallest [`NodeId`]**,
/// and that representative **keeps its own coordinates** (no averaging — a
/// deliberate choice, like the rest of the pipeline, to avoid silently moving
/// geometry). Welded-away nodes are left in the shared [`Coords`]; once
/// nothing references them they become collectable by
/// [`Coords::gc`](crate::coords::Coords::gc). Errors if `tol` is negative or
/// if `mesh` has no submeshes (no Coords to attach to).
///
/// # `in_place`
///
/// `false` — the **copying** weld. The result mirrors `mesh`: same submeshes,
/// in the same order, each keeping its element type and face colour, with
/// every reference to a welded-away node redirected. Every node referenced by
/// the result is increfed afresh by the new submeshes; `mesh` itself is left
/// untouched. Cells that **collapse** (referencing the same representative
/// twice: a SEG2 whose two ends merge, a TRI3 with two coincident corners, …)
/// are **dropped**, being degenerate. POI1 cells, single nodes, never collapse
/// and are always kept — de-duplicating colocated points is
/// [`consolidate`](fn@crate::ops::mesh::consolidate)'s job, not this one's.
///
/// `true` — the **in-place** weld: the connectivity of `mesh`'s own submeshes
/// is rewritten, through
/// [`SubMesh::remap_nodes`](crate::containers::mesh::SubMesh::remap_nodes),
/// and **the same mesh** comes back — an aggregate over the very same submesh
/// slots, whose insides have changed. Nothing is copied and nothing has to be
/// re-plumbed: the value handed back and the argument are one mesh seen twice.
///
/// The assumed, wanted side effect is what welding *several* meshes takes.
/// Since the aggregate operators share their submeshes rather than copying
/// them, `mesh_a | mesh_b` is a mesh over the same slots — so welding that
/// union in place reaches `mesh_a` and `mesh_b` themselves, which afterwards
/// really do share their interface nodes. The copying weld would leave both
/// originals apart.
///
/// It stays defensible because the **mesh structure is preserved**: same
/// submeshes, same element types, same number of cells in the same order —
/// only *which node* a cell refers to changes, so every index a caller holds
/// (cell numbers, and the element fields keyed on them) stays valid. Hence two
/// refusals, both checked over the whole mesh **before anything is written**,
/// so a rejected call leaves every submesh untouched:
///
/// - a cell that would **collapse** is an error here instead of being dropped
///   — dropping would change the cell count, which is exactly the invariant
///   in-place callers rely on. Lower `tol`, or weld by copy;
/// - a **sealed** submesh is an error: a finite-element space, field or matrix
///   has captured it and reads its node numbering.
///
/// # Tally
///
/// Every call prints one line on **stdout** once the weld is done — how many
/// nodes were welded away, how many cells dropped, at which tolerance. A weld
/// is a step you want to see in a build log: `tol` is a guess about the
/// geometry, and this line is what tells you it guessed right.
pub fn merge_nodes(mesh: &Mesh, tol: f64, in_place: bool) -> Result<Mesh> {
    if tol < 0.0 {
        return Err(PyrucastError::Message(format!(
            "merge_nodes: tol must be ≥ 0, got {tol}"
        )));
    }
    let coords_handle = mesh.coords()?;

    // Map every referenced node to its cluster representative.
    let representative = build_representatives(mesh, &coords_handle, tol)?;
    let welded = representative
        .iter()
        .filter(|(id, rep)| *id != *rep)
        .count();

    let (result, dropped) = if in_place {
        // An in-place weld never drops a cell — it refuses instead.
        (weld_in_place(mesh, &representative)?, 0)
    } else {
        weld_into_copy(mesh, &coords_handle, &representative)?
    };
    println!("{}", summary(welded, dropped, tol, in_place));
    Ok(result)
}

/// The tally line printed by every weld: what it changed, in one sentence.
fn summary(welded: usize, dropped: usize, tol: f64, in_place: bool) -> String {
    let how = if in_place { " (in place)" } else { "" };
    let cells = if in_place {
        // Saying "0 cells dropped" would suggest it could have been otherwise.
        String::from("cells untouched")
    } else {
        format!("{dropped} cell(s) dropped")
    };
    format!("merge_nodes{how}: {welded} node(s) welded, {cells}, tol = {tol}")
}

/// Rebuild every submesh with remapped connectivity, dropping degenerate
/// cells — the copying half of [`merge_nodes`]. Returns the new mesh and how
/// many cells were dropped.
fn weld_into_copy(
    mesh: &Mesh,
    coords_handle: &crate::store::Handle<Coords>,
    representative: &HashMap<NodeId, NodeId>,
) -> Result<(Mesh, usize)> {
    let mut result = Mesh::empty();
    let mut dropped = 0;
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
                dropped += 1;
                continue;
            }
            new_sm.add_cell(&mapped)?;
        }

        result.add_sub(insert(new_sm))?;
    }

    Ok((result, dropped))
}

/// Rename the nodes of `mesh`'s own submeshes — the in-place half of
/// [`merge_nodes`]. Refuses (before writing anything) a sealed submesh or a
/// cell that would collapse; see [`merge_nodes`] for why.
fn weld_in_place(mesh: &Mesh, representative: &HashMap<NodeId, NodeId>) -> Result<Mesh> {
    // Pre-flight over the whole mesh: an in-place run is all-or-nothing.
    for (si, sm_handle) in mesh.into_iter().enumerate() {
        let s = read(sm_handle)?;
        if s.is_sealed() {
            return Err(PyrucastError::Message(format!(
                "merge_nodes(in_place): submesh {si} is sealed — a finite-element \
                 space, field or matrix already reads its nodes; weld before \
                 building them, or weld by copy (in_place = false)"
            )));
        }
        let npc = s.element_type().nodes_per_cell();
        if npc == 0 {
            continue;
        }
        for (ci, chunk) in s.connectivity().chunks(npc).enumerate() {
            let mapped: Vec<NodeId> = chunk
                .iter()
                .map(|n| representative.get(n).copied().unwrap_or(*n))
                .collect();
            if is_degenerate(&mapped) {
                return Err(PyrucastError::Message(format!(
                    "merge_nodes(in_place): cell {ci} of submesh {si} ({}) would \
                     collapse — welding it away would change the cell count, which \
                     an in-place weld preserves; lower tol, or weld by copy \
                     (in_place = false), which drops degenerate cells",
                    s.element_type()
                )));
            }
        }
    }

    for sm_handle in mesh {
        write(sm_handle)?.remap_nodes(representative)?;
    }

    // The same mesh back: an aggregate over the very same submesh slots (the
    // handles are shared, not deep-copied), now welded.
    mesh.subset(0..mesh.len())
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
        let p = coords.position(id)?;
        let base: Vec<i64> = p.iter().map(|&x| (x / cell).floor() as i64).collect();

        let mut rep = None;
        'search: for off in &offsets {
            let key: Vec<i64> = base.iter().zip(off).map(|(b, o)| b + o).collect();
            if let Some(candidates) = grid.get(&key) {
                for &c in candidates {
                    if dist2(p, coords.position(c)?) <= tol2 {
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
    use crate::atoms::ElementType;
    use crate::atoms::Node;

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

        let merged = merge_nodes(&mesh, 1e-6, false).unwrap();
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

        let merged = merge_nodes(&mesh, 1e-6, false).unwrap();
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

        let merged = merge_nodes(&mesh, 1e-6, false).unwrap();
        let welded = merged.node(0, 1, 1).unwrap();
        assert_eq!(welded.id(), b.id());
        assert_eq!(
            read(&coords).unwrap().position(b.id()).unwrap(),
            &[5.0, 5.0]
        );
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

        let merged = merge_nodes(&mesh, 0.0, false).unwrap();
        // exact → a (distance 0), but near stays distinct from b (distance 1e-9 > 0).
        assert_eq!(merged.node(0, 0, 1).unwrap().id(), near.id());
        assert_eq!(merged.node(0, 1, 1).unwrap().id(), a.id());
        assert_ne!(near.id(), b.id());
    }

    #[test]
    fn in_place_welds_through_a_union_and_reaches_both_meshes() {
        // Two SEG2 pieces meshed apart, meeting at a duplicated node.
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0 + 1e-9, 0.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();

        let mut left = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        left.add_cell(&[a.id(), b.id()]).unwrap();
        let mut right = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        right.add_cell(&[b2.id(), d.id()]).unwrap();

        // The union shares the two submeshes, so welding it welds them.
        let both = left.union(&right).unwrap();
        let welded = merge_nodes(&both, 1e-6, true).unwrap();
        // The same mesh back: same submesh slots, welded insides.
        assert_eq!(welded.len(), both.len());
        assert_eq!(welded.get(0).unwrap().index(), both.get(0).unwrap().index());

        // b2 → b in `right` itself — the two pieces now share their node.
        assert_eq!(right.node(0, 0, 0).unwrap().id(), b.id());
        assert_eq!(left.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(left.cell_count().unwrap(), 1);
        assert_eq!(right.cell_count().unwrap(), 1);
    }

    #[test]
    fn in_place_moves_refcounts_and_leaves_positions_alone() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[5.0, 5.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[5.0 + 1e-9, 5.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[a.id(), b2.id()]).unwrap();

        assert_eq!(read(&coords).unwrap().refcount(b.id()), 2);
        merge_nodes(&mesh, 1e-6, true).unwrap();

        // b2's connectivity unit moved to b; b2 survives through its Node only.
        assert_eq!(read(&coords).unwrap().refcount(b.id()), 3);
        assert_eq!(read(&coords).unwrap().refcount(b2.id()), 1);
        // The representative keeps its own coordinates — no averaging.
        assert_eq!(
            read(&coords).unwrap().position(b.id()).unwrap(),
            &[5.0, 5.0]
        );
    }

    #[test]
    fn in_place_refuses_a_collapsing_cell_without_touching_anything() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        // c sits on top of b → the SEG2 (b, c) would collapse.
        let c = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[b.id(), c.id()]).unwrap();

        assert!(merge_nodes(&mesh, 1e-6, true).is_err());
        // Nothing was written: the mesh still holds both cells, on c.
        assert_eq!(mesh.cell_count().unwrap(), 2);
        assert_eq!(mesh.node(0, 1, 1).unwrap().id(), c.id());
        // The copying variant is the way out — it drops the degenerate cell.
        assert_eq!(
            merge_nodes(&mesh, 1e-6, false)
                .unwrap()
                .cell_count()
                .unwrap(),
            1
        );
    }

    #[test]
    fn in_place_refuses_a_sealed_submesh_without_touching_anything() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0 + 1e-9, 0.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();

        let mut sealed = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        sealed.add_cell(&[b2.id(), d.id()]).unwrap();
        crate::containers::mesh::seal(&sealed.get(0).unwrap()).unwrap();

        let mut open = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        open.add_cell(&[a.id(), b.id()]).unwrap();

        // The open submesh comes first, but the sealed one still vetoes the
        // whole run before any write.
        let both = open.union(&sealed).unwrap();
        assert!(merge_nodes(&both, 1e-6, true).is_err());
        assert_eq!(sealed.node(0, 0, 0).unwrap().id(), b2.id());
        assert_eq!(open.node(0, 0, 1).unwrap().id(), b.id());
    }

    #[test]
    fn tally_line_reports_both_welds() {
        // Copying: cells can be dropped, so they are counted.
        assert_eq!(
            summary(3, 1, 1e-6, false),
            "merge_nodes: 3 node(s) welded, 1 cell(s) dropped, tol = 0.000001"
        );
        // In place: no cell can be dropped — saying "0 dropped" would suggest
        // it could have been otherwise.
        assert_eq!(
            summary(3, 0, 1e-6, true),
            "merge_nodes (in place): 3 node(s) welded, cells untouched, tol = 0.000001"
        );
    }

    #[test]
    fn negative_tol_is_error() {
        let coords = coords2();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        assert!(merge_nodes(&mesh, -1.0, false).is_err());
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
        let merged = merge_nodes(&mesh, 1e-6, false).unwrap();
        // +1 from the result submesh.
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 3);
        drop(merged);
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
    }
}
