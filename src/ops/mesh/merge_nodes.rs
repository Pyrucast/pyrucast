use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::parallel::*;

/// Weld together nodes closer than `tol` (Euclidean distance), rewriting the
/// connectivity to refer to a single representative per cluster.
///
/// A cluster is a **connected component** of the « closer than `tol` »
/// relation: the pairing propagates from node to node, so on a chain a—b—c
/// where a and c are further apart than `tol`, all three still end up welded
/// together — which is also why a `tol` of the order of the element size
/// collapses a whole region rather than a seam. Each cluster is represented by
/// the node with the **smallest [`NodeId`]**, and that representative **keeps
/// its own coordinates** (no averaging — a deliberate choice, like the rest of
/// the pipeline, to avoid silently moving geometry). Welded-away nodes are
/// left in the shared [`Coords`]; once
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// // Deux lignes bout à bout mais aux nœuds **distincts** : la fusion les
/// // recoud, et le nuage passe de quatre nœuds à trois.
/// let a = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[1.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let b = mesh::line(&p(&[1.0, 0.0, 0.0]), &p(&[2.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let deux = a.union(&b)?;
/// // `to_poi1` dédoublonne **par zone** : sur deux zones il compte encore
/// // quatre nœuds. Consolider d'abord donne le vrai nuage.
/// assert_eq!(mesh::to_poi1(&mesh::consolidate(&deux)?)?.cell_count(), 4);
/// let cousu = mesh::merge_nodes(&deux, 1e-6, false)?;
/// assert_eq!(mesh::to_poi1(&mesh::consolidate(&cousu)?)?.cell_count(), 3);
/// // La structure est intacte : mêmes zones, mêmes mailles, dans le même
/// // ordre — seul **à quel nœud** une maille se réfère a changé.
/// assert_eq!(cousu.cell_count(), deux.cell_count());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn merge_nodes(mesh: &Mesh, tol: f64, in_place: bool) -> Result<Mesh> {
    if tol < 0.0 {
        return Err(PyrucastError::Message(format!(
            "merge_nodes: tol must be ≥ 0, got {tol}"
        )));
    }
    let coords_handle = mesh.coords()?;

    // Map every referenced node to its cluster representative.
    let (representative, welded) = build_representatives(mesh, &coords_handle, tol)?;

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
    coords_handle: &crate::handle::Handle<Coords>,
    representative: &[u32],
) -> Result<(Mesh, usize)> {
    let mut result = Mesh::empty();
    let mut dropped = 0;
    for sm_handle in mesh {
        // The rewritten connectivity is built whole, under the submesh guard
        // alone; the `Coords` lock is taken once, afterwards, by
        // `from_connectivity`.
        let (et, color, mapped, degenerate) = {
            let s = sm_handle.read();
            let et = s.element_type();
            let npc = et.nodes_per_cell();
            let conn = s.connectivity();
            let mapped: Vec<NodeId> = conn
                .par_iter()
                .with_min_len(MIN_PARALLEL_LEN)
                .map(|&n| image(representative, n))
                .collect();
            let degenerate: Vec<bool> = mapped
                .par_chunks(npc)
                .with_min_len(MIN_PARALLEL_LEN)
                .map(is_degenerate)
                .collect();
            (et, s.face_color(), mapped, degenerate)
        };

        let npc = et.nodes_per_cell();
        let dropped_here = degenerate.iter().filter(|d| **d).count();
        dropped += dropped_here;
        let conn = if dropped_here == 0 {
            // Nothing collapsed: the mapped connectivity is the answer.
            mapped
        } else {
            let mut kept = Vec::with_capacity(mapped.len() - dropped_here * npc);
            for (cell, &deg) in mapped.chunks(npc).zip(&degenerate) {
                if !deg {
                    kept.extend_from_slice(cell);
                }
            }
            kept
        };

        let mut new_sm = SubMesh::from_connectivity(coords_handle.clone(), et, conn)?;
        new_sm.set_face_color(color);
        result.add_sub(Handle::new(new_sm))?;
    }

    Ok((result, dropped))
}

/// Rename the nodes of `mesh`'s own submeshes — the in-place half of
/// [`merge_nodes`]. Refuses (before writing anything) a sealed submesh or a
/// cell that would collapse; see [`merge_nodes`] for why.
fn weld_in_place(mesh: &Mesh, representative: &[u32]) -> Result<Mesh> {
    // Pre-flight over the whole mesh: an in-place run is all-or-nothing.
    for (si, sm_handle) in mesh.into_iter().enumerate() {
        let s = sm_handle.read();
        if s.is_sealed() {
            return Err(PyrucastError::Message(format!(
                "merge_nodes(in_place): submesh {si} is sealed — a finite-element \
                 space, a matrix, or a node field supported on this POI1 cloud \
                 already reads its nodes; weld before building them, or weld by \
                 copy (in_place = false)"
            )));
        }
        let npc = s.element_type().nodes_per_cell();
        if npc == 0 {
            continue;
        }
        // `position_first` names the same cell a sequential scan would, so the
        // message does not depend on the thread count.
        let collapsing = s
            .connectivity()
            .par_chunks(npc)
            .with_min_len(MIN_PARALLEL_LEN)
            .position_first(|cell| collapses(representative, cell));
        if let Some(ci) = collapsing {
            return Err(PyrucastError::Message(format!(
                "merge_nodes(in_place): cell {ci} of submesh {si} ({}) would \
                 collapse — welding it away would change the cell count, which \
                 an in-place weld preserves; lower tol, or weld by copy \
                 (in_place = false), which drops degenerate cells",
                s.element_type()
            )));
        }
    }

    for sm_handle in mesh {
        sm_handle.write().remap_nodes_dense(representative)?;
    }

    // The same mesh back: an aggregate over the very same submesh slots (the
    // handles are shared, not deep-copied), now welded.
    mesh.subset(0..mesh.len())
}

/// Map every referenced node to its cluster representative — the smallest
/// `NodeId` of its connected component under « closer than `tol` » — and count
/// how many nodes were welded away.
///
/// The table is **dense**, indexed by `NodeId.0`: a node that stands for itself
/// maps to itself. Three passes, and not a single hash lookup — `NodeId` is
/// already an index into the `Coords`, and the grid buckets are a plain array:
///
/// 1. a uniform grid over the referenced nodes, in CSR form
///    ([`NodeGrid`]), whose cell is never smaller than `tol`;
/// 2. a **parallel** neighbour search emitting one edge per pair within `tol`,
///    each node visiting only the buckets its `tol`-ball actually touches —
///    one, in the ordinary case where `tol` is far below the element size;
/// 3. a union-find closing the components, always linking to the smaller root.
///
/// The result does not depend on the thread count: the edge set is the same
/// whatever the order it is collected in, and so is the smallest id of a
/// component.
fn build_representatives(
    mesh: &Mesh,
    coords_handle: &crate::handle::Handle<Coords>,
    tol: f64,
) -> Result<(Vec<u32>, usize)> {
    let capacity = coords_handle.read().capacity();

    // Referenced ids, ascending, through a dense bitmap: no hash set, and the
    // ascending order comes out of the scan rather than out of a sort.
    let mut referenced = vec![false; capacity];
    for sm_handle in mesh {
        for &n in sm_handle.read().connectivity() {
            match referenced.get_mut(n.0 as usize) {
                Some(slot) => *slot = true,
                // An id the `Coords` never handed out: refused here rather
                // than left to trip the dense table further down.
                None => {
                    return Err(PyrucastError::Message(format!(
                        "merge_nodes: node {} does not belong to this Coords",
                        n.0
                    )));
                }
            }
        }
    }
    let ids: Vec<NodeId> = (0..capacity as u32)
        .filter(|&i| referenced[i as usize])
        .map(NodeId)
        .collect();

    // Every node stands for itself until an edge says otherwise.
    let mut parent: Vec<u32> = (0..capacity as u32).collect();
    if ids.is_empty() {
        return Ok((parent, 0));
    }

    let guard = coords_handle.read();
    let coords: &Coords = &guard;
    // Paid once for the whole cloud, so the search reads positions unchecked.
    coords.ensure_all_alive(&ids)?;

    let grid = NodeGrid::build(&ids, coords, tol);
    let tol2 = tol * tol;
    // One edge per pair within tol, the smaller id second. Building the pairs
    // is where the time goes, and every node's search is independent.
    let edges: Vec<(u32, u32)> = ids
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .flat_map_iter(|&id| {
            let p = coords.position_alive(id);
            let mut local: Vec<(u32, u32)> = Vec::new();
            grid.for_each_candidate(p, tol, &mut |other: NodeId| {
                if other.0 < id.0 && dist2(p, coords.position_alive(other)) <= tol2 {
                    local.push((id.0, other.0));
                }
            });
            local
        })
        .collect();
    drop(guard);

    for &(a, b) in &edges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            // The smaller id takes the root: a cluster is represented by its
            // smallest node, whatever order the edges arrived in.
            let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[child as usize] = root;
        }
    }

    // Full compression, so the table is a direct lookup from here on.
    let mut welded = 0;
    for &id in &ids {
        let root = find(&mut parent, id.0);
        parent[id.0 as usize] = root;
        if root != id.0 {
            welded += 1;
        }
    }

    Ok((parent, welded))
}

/// Union-find root of `x`, with path halving.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let grand = parent[parent[x as usize] as usize];
        parent[x as usize] = grand;
        x = grand;
    }
    x
}

/// A uniform grid over the referenced nodes, in CSR form: the ids of bucket
/// `b` are `items[starts[b]..starts[b + 1]]`, ascending.
///
/// The cell is **never smaller than `tol`**, so the ball of radius `tol` around
/// a point spills over at most one bucket per axis — which is what lets a
/// search visit the buckets it really touches instead of a fixed 3^dim
/// neighbourhood.
struct NodeGrid {
    dim: usize,
    lo: Vec<f64>,
    /// 1 / cell size per axis (0 on an axis with no extent).
    inv: Vec<f64>,
    res: Vec<usize>,
    stride: Vec<usize>,
    starts: Vec<u32>,
    items: Vec<NodeId>,
}

impl NodeGrid {
    fn build(ids: &[NodeId], coords: &Coords, tol: f64) -> NodeGrid {
        let dim = coords.dim() as usize;
        let n = ids.len();

        let mut lo = vec![f64::INFINITY; dim];
        let mut hi = vec![f64::NEG_INFINITY; dim];
        for &id in ids {
            for (a, &x) in coords.position_alive(id).iter().enumerate() {
                lo[a] = lo[a].min(x);
                hi[a] = hi[a].max(x);
            }
        }

        // Only the axes with an extent are worth cutting: a plane sitting in
        // 3-D spends its buckets in the plane, not on the flat axis.
        let live = lo
            .iter()
            .zip(&hi)
            .filter(|(l, h)| *h - *l > 0.0)
            .count()
            .max(1);
        let target = (n as f64).powf(1.0 / live as f64).round().max(1.0);
        let mut res = vec![1usize; dim];
        let mut inv = vec![0.0f64; dim];
        for a in 0..dim {
            let extent = hi[a] - lo[a];
            if extent <= 0.0 {
                continue;
            }
            // ~n buckets in all, but never a cell below tol.
            let mut r = target;
            if tol > 0.0 {
                r = r.min((extent / tol).floor().max(1.0));
            }
            res[a] = r as usize;
            inv[a] = res[a] as f64 / extent;
        }
        let mut stride = vec![1usize; dim];
        for a in 1..dim {
            stride[a] = stride[a - 1] * res[a - 1];
        }
        let total = stride[dim - 1] * res[dim - 1];

        let mut grid = NodeGrid {
            dim,
            lo,
            inv,
            res,
            stride,
            starts: vec![0; total + 1],
            items: vec![NodeId(0); n],
        };

        // Counting sort: count, prefix-sum, scatter. Ids are visited in
        // ascending order, so each bucket comes out ascending too.
        for &id in ids {
            let b = grid.bucket(coords.position_alive(id));
            grid.starts[b + 1] += 1;
        }
        for b in 0..total {
            grid.starts[b + 1] += grid.starts[b];
        }
        let mut cursor = grid.starts.clone();
        for &id in ids {
            let b = grid.bucket(coords.position_alive(id));
            grid.items[cursor[b] as usize] = id;
            cursor[b] += 1;
        }
        grid
    }

    /// Index along `axis` of the bucket holding coordinate `x`, clamped to the
    /// grid (a point outside the bbox belongs to the border bucket).
    fn axis_index(&self, axis: usize, x: f64) -> usize {
        let i = ((x - self.lo[axis]) * self.inv[axis]).floor();
        i.clamp(0.0, (self.res[axis] - 1) as f64) as usize
    }

    /// Linear index of the bucket holding point `p`.
    fn bucket(&self, p: &[f64]) -> usize {
        (0..self.dim)
            .map(|a| self.axis_index(a, p[a]) * self.stride[a])
            .sum()
    }

    /// Call `f` on every node of every bucket the ball of radius `tol` around
    /// `p` touches — at most two rows per axis, and usually just the one the
    /// point itself lands in. Recursing over the axes keeps it dimension-
    /// agnostic without allocating a thing.
    fn for_each_candidate(&self, p: &[f64], tol: f64, f: &mut impl FnMut(NodeId)) {
        self.visit(0, 0, p, tol, f);
    }

    fn visit(&self, axis: usize, acc: usize, p: &[f64], tol: f64, f: &mut impl FnMut(NodeId)) {
        if axis == self.dim {
            for &id in &self.items[self.starts[acc] as usize..self.starts[acc + 1] as usize] {
                f(id);
            }
            return;
        }
        let first = self.axis_index(axis, p[axis] - tol);
        let last = self.axis_index(axis, p[axis] + tol);
        for i in first..=last {
            self.visit(axis + 1, acc + i * self.stride[axis], p, tol, f);
        }
    }
}

/// Squared Euclidean distance between two coordinate slices of equal length.
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The image of a node through the dense representative table.
///
/// The table covers the whole `Coords`, and every id it is asked about comes
/// out of a connectivity that [`build_representatives`] has already checked —
/// so the index is the caller's contract, honoured once per zone rather than
/// once per node.
fn image(table: &[u32], n: NodeId) -> NodeId {
    NodeId(table[n.0 as usize])
}

/// Whether a cell references the same node twice or more (after welding).
fn is_degenerate(nodes: &[NodeId]) -> bool {
    nodes
        .iter()
        .enumerate()
        .any(|(i, n)| nodes[i + 1..].contains(n))
}

/// Whether a cell **would** collapse once welded — read off the table, without
/// materialising the welded cell.
fn collapses(table: &[u32], cell: &[NodeId]) -> bool {
    cell.iter().enumerate().any(|(i, &n)| {
        let rep = image(table, n);
        cell[i + 1..].iter().any(|&m| image(table, m) == rep)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;

    fn coords2() -> crate::handle::Handle<Coords> {
        Handle::new(Coords::new(2).unwrap())
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
        assert_eq!(merged.cell_count(), 2, "both triangles survive");

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
        assert_eq!(merged.cell_count(), 1, "the (b,c) segment is dropped");
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
        assert_eq!(coords.read().position(b.id()).unwrap(), &[5.0, 5.0]);
    }

    #[test]
    fn a_chain_of_close_nodes_closes_into_one_cluster() {
        let coords = coords2();
        // a—b—c espacés de 0,9 : a touche b, b touche c, mais a et c sont à
        // 1,8, soit bien plus que la tolérance.
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.9, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.8, 0.0]).unwrap();
        let far = Node::create_in(coords.clone(), &[10.0, 0.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), far.id()]).unwrap();
        mesh.add_cell(&[b.id(), far.id()]).unwrap();
        mesh.add_cell(&[c.id(), far.id()]).unwrap();

        let merged = merge_nodes(&mesh, 1.0, false).unwrap();
        // La grappe est une composante connexe : la chaîne se referme, et les
        // trois nœuds pointent sur `a`, le plus petit identifiant.
        for cell in 0..3 {
            assert_eq!(merged.node(0, cell, 0).unwrap().id(), a.id());
        }
        // Le nœud lointain, lui, n'a pas bougé.
        assert_eq!(merged.node(0, 0, 1).unwrap().id(), far.id());
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
        assert!(welded.get(0).unwrap().same_object(&both.get(0).unwrap()));

        // b2 → b in `right` itself — the two pieces now share their node.
        assert_eq!(right.node(0, 0, 0).unwrap().id(), b.id());
        assert_eq!(left.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(left.cell_count(), 1);
        assert_eq!(right.cell_count(), 1);
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

        assert_eq!(coords.read().refcount(b.id()), 2);
        merge_nodes(&mesh, 1e-6, true).unwrap();

        // b2's connectivity unit moved to b; b2 survives through its Node only.
        assert_eq!(coords.read().refcount(b.id()), 3);
        assert_eq!(coords.read().refcount(b2.id()), 1);
        // The representative keeps its own coordinates — no averaging.
        assert_eq!(coords.read().position(b.id()).unwrap(), &[5.0, 5.0]);
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
        assert_eq!(mesh.cell_count(), 2);
        assert_eq!(mesh.node(0, 1, 1).unwrap().id(), c.id());
        // The copying variant is the way out — it drops the degenerate cell.
        assert_eq!(merge_nodes(&mesh, 1e-6, false).unwrap().cell_count(), 1);
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
        crate::containers::mesh::seal(&sealed.get(0).unwrap());

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
        assert_eq!(coords.read().refcount(a.id()), 2);
        let merged = merge_nodes(&mesh, 1e-6, false).unwrap();
        // +1 from the result submesh.
        assert_eq!(coords.read().refcount(a.id()), 3);
        drop(merged);
        assert_eq!(coords.read().refcount(a.id()), 2);
    }
}
