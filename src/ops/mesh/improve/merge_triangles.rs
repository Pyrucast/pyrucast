//! Removing the triangles from a quadrangle-dominant mesh — in pairs, because
//! there is no other way.
//!
//! ## Why pairs
//!
//! Counting the sides of every face gives `4Q + 3T = 2·E_interior + E_boundary`,
//! so `T ≡ E_boundary (mod 2)`: **the number of triangles has the parity of the
//! mesh's boundary edge count**. The boundary is the caller's and untouchable,
//! so that parity is fixed and no sequence of local operations can change it. A
//! mesh with an odd number of triangles keeps one, whatever is done to it.
//!
//! This is also why "collapse a triangle to a node" does not work: merging its
//! three corners into one costs every neighbouring quadrangle an edge, and each
//! becomes a triangle in turn. Nothing is gained.
//!
//! ## The two moves
//!
//! - **Merge.** Two triangles sharing an edge are one quadrangle. That is the
//!   simplest move that removes triangles, and it removes exactly two — but it
//!   is take-it-or-leave-it: there is only one quadrangle to be had, and if it
//!   comes out re-entrant the pair is stuck.
//! - **Regroup.** Take one or two neighbouring quadrangles in with them and cut
//!   the outline of the group in two. Around a node, a cell with `k` corners
//!   gives `k - 2` edges to that outline, so two triangles and `n`
//!   quadrangles give `2n + 2`: a **hexagon** for one quadrangle in a strip and
//!   for two round a node, and a hexagon splits into two quadrangles across any
//!   of its three main diagonals. Where the plain merge offered one candidate
//!   this offers three, and a pair the merge could not take is often taken
//!   here.
//!
//!   Three quadrangles would give an octagon, which wants three quadrangles and
//!   far more ways to cut it. The line is drawn at two: the return tapers and
//!   the search does not.
//! - **Walk.** A triangle and a neighbouring quadrangle share an edge, and
//!   their union is a **pentagon**, which splits into a quadrangle and a
//!   triangle in five ways. Choosing another one moves the triangle one cell
//!   across. The pentagon's outline never changes, so the mesh's boundary
//!   cannot: walking is safe against the contract by construction.
//!
//! So: merge what is already adjacent, then walk the rest together and merge
//! them too. Every move is refused unless both resulting cells are valid, which
//! is what keeps a stubborn pair from being bought at the price of a sliver.
//!
//! Quality straight after a merge is poor — the quadrangle inherits whatever
//! shape the two triangles had. Follow with
//! [`cleanup`](fn@super::cleanup::cleanup) and
//! [`regularize`](fn@super::regularize::regularize).

use super::Surface;
use crate::atoms::Point2;
use crate::containers::mesh::Mesh;
use crate::error::Result;
use crate::ops::mesh::paving::geom::{orient, quad_is_valid};
use std::collections::{HashMap, VecDeque};

/// How many cells a triangle may be walked before the pair is given up. A
/// triangle that has to cross half the mesh to find a partner is not worth the
/// damage the crossing would do.
const MAX_WALK: usize = 24;

/// A cell of the working mesh. Dead cells are left in place and dropped at the
/// end, so indices stay stable while the walk rewrites its way across.
#[derive(Clone, Copy)]
enum Cell {
    Quad([u32; 4]),
    Tri([u32; 3]),
    Dead,
}

impl Cell {
    fn nodes(&self) -> &[u32] {
        match self {
            Cell::Quad(q) => q,
            Cell::Tri(t) => t,
            Cell::Dead => &[],
        }
    }

    fn is_tri(&self) -> bool {
        matches!(self, Cell::Tri(_))
    }
}

/// Remove the triangles of `mesh`, two at a time.
///
/// The result is a fresh mesh over the caller's own nodes: the connectivity
/// changed, no node moved and none was added. When the mesh has an odd number
/// of triangles one necessarily remains — see the module documentation — and
/// when a pair cannot be brought together without producing an invalid cell it
/// is left alone rather than forced.
///
/// **Call it again after smoothing.** A merge refused because the union of two
/// triangles is not convex often becomes admissible once the nodes have moved,
/// so `merge_triangles` → `cleanup` → `regularize`, repeated, gets further than
/// one pass and keeps every intermediate mesh valid. On a circle paved by
/// `grid_surface` that takes 32 triangles to 20, then to 14, and there it
/// settles.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // Apparie les triangles en quadrangles quand la paire est de qualité.
/// // Un triangle isolé n'a personne avec qui s'apparier.
/// assert_eq!(mesh::merge_triangles(&maillage)?.cell_count()?, 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn merge_triangles(mesh: &Mesh) -> Result<Mesh> {
    let mut surf = Surface::read(mesh, "merge_triangles")?;
    let mut cells: Vec<Cell> = surf
        .quads
        .iter()
        .map(|q| Cell::Quad(*q))
        .chain(surf.tris.iter().map(|t| Cell::Tri(*t)))
        .collect();

    let mut progress = true;
    while progress {
        while merge_adjacent(&mut cells, &surf.pts) {}
        progress = regroup_with_a_quad(&mut cells, &surf.pts)
            || walk_a_pair_together(&mut cells, &surf.pts);
    }

    surf.quads.clear();
    surf.tris.clear();
    for c in &cells {
        match c {
            Cell::Quad(q) => surf.quads.push(*q),
            Cell::Tri(t) => surf.tris.push(*t),
            Cell::Dead => {}
        }
    }
    surf.to_mesh_same_nodes("merge_triangles")
}

/// Undirected edge to the cells carrying it.
fn edge_map(cells: &[Cell]) -> HashMap<(u32, u32), Vec<usize>> {
    let mut m: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        let n = c.nodes();
        for k in 0..n.len() {
            let (a, b) = (n[k], n[(k + 1) % n.len()]);
            m.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    m
}

/// Fuse one pair of triangles that already share an edge. Returns whether it
/// found one — the caller loops.
fn merge_adjacent(cells: &mut [Cell], pts: &[Point2]) -> bool {
    let edges = edge_map(cells);
    let mut keys: Vec<&(u32, u32)> = edges.keys().collect();
    keys.sort_unstable();
    for k in keys {
        let owners = &edges[k];
        if owners.len() != 2 || !owners.iter().all(|&i| cells[i].is_tri()) {
            continue;
        }
        let (i, j) = (owners[0], owners[1]);
        let Some(quad) = fuse(cells[i], cells[j], *k) else {
            continue;
        };
        // Never traded away, as everywhere else in the pavers: two triangles
        // whose union is not convex stay two triangles. Smoothing the mesh and
        // calling again is what unlocks them — the composition converges.
        if !quad_is_valid(quad.map(|v| pts[v as usize])) {
            continue;
        }
        cells[i] = Cell::Quad(quad);
        cells[j] = Cell::Dead;
        return true;
    }
    false
}

/// Take a neighbouring quadrangle in with a stuck pair of triangles and
/// re-cut the hexagon the three of them make. Returns whether one was taken.
///
/// The plain merge offers a single candidate quadrangle and no recourse when it
/// comes out re-entrant. Bringing a quadrangle in turns the problem into
/// cutting a hexagon in two, which has three answers instead of one — and,
/// being a re-cut of the same outline, it can no more change the mesh's
/// boundary than the walk can.
fn regroup_with_a_quad(cells: &mut [Cell], pts: &[Point2]) -> bool {
    let edges = edge_map(cells);
    let mut keys: Vec<&(u32, u32)> = edges.keys().collect();
    keys.sort_unstable();
    for k in &keys {
        let owners = &edges[*k];
        if owners.len() != 2 || !owners.iter().all(|&i| cells[i].is_tri()) {
            continue;
        }
        let (i, j) = (owners[0], owners[1]);
        // Every quadrangle sharing an edge with either triangle is a candidate
        // third cell. Deterministic order, so the answer does not depend on
        // how the hash map happened to lay out.
        let mut thirds: Vec<usize> = Vec::new();
        for e in &keys {
            let o = &edges[*e];
            if o.len() != 2 {
                continue;
            }
            for (a, b) in [(o[0], o[1]), (o[1], o[0])] {
                if (a == i || a == j) && !cells[b].is_tri() && !thirds.contains(&b) {
                    thirds.push(b);
                }
            }
        }
        // One quadrangle first, then two: a pair costs more to try and is
        // only ever needed when one will not do.
        let mut groups: Vec<Vec<usize>> = thirds.iter().map(|&t| vec![t]).collect();
        for a in 0..thirds.len() {
            for b in (a + 1)..thirds.len() {
                groups.push(vec![thirds[a], thirds[b]]);
            }
        }
        for group in groups {
            let mut patch = vec![cells[i], cells[j]];
            patch.extend(group.iter().map(|&t| cells[t]));
            let ring = match patch_ring(&patch) {
                Some(r) if r.len() == 6 => r,
                _ => continue,
            };
            for d in 0..3 {
                let a = [ring[d], ring[d + 1], ring[d + 2], ring[d + 3]];
                let b = [
                    ring[(d + 3) % 6],
                    ring[(d + 4) % 6],
                    ring[(d + 5) % 6],
                    ring[d],
                ];
                if !quad_is_valid(a.map(|v| pts[v as usize]))
                    || !quad_is_valid(b.map(|v| pts[v as usize]))
                {
                    continue;
                }
                cells[i] = Cell::Quad(a);
                cells[j] = Cell::Quad(b);
                for &t in &group {
                    cells[t] = Cell::Dead;
                }
                return true;
            }
        }
    }
    false
}

/// The outline of a set of cells, in order, or `None` when they do not make a
/// single simply-connected patch.
fn patch_ring(patch: &[Cell]) -> Option<Vec<u32>> {
    let mut count: HashMap<(u32, u32), usize> = HashMap::new();
    let mut directed: Vec<(u32, u32)> = Vec::new();
    for c in patch {
        let n = c.nodes();
        for k in 0..n.len() {
            let (u, v) = (n[k], n[(k + 1) % n.len()]);
            *count.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            directed.push((u, v));
        }
    }
    let mut next: HashMap<u32, u32> = HashMap::new();
    for (u, v) in directed {
        if count[&(u.min(v), u.max(v))] == 1 {
            // A vertex on the outline twice would make it pinched, and a
            // pinched outline is not a polygon to cut.
            if next.insert(u, v).is_some() {
                return None;
            }
        }
    }
    let start = *next.keys().min()?;
    let mut ring = vec![start];
    let mut cur = start;
    loop {
        cur = *next.get(&cur)?;
        if cur == start {
            return (ring.len() == next.len()).then_some(ring);
        }
        ring.push(cur);
        if ring.len() > next.len() {
            return None;
        }
    }
}

/// The quadrangle two triangles sharing `edge` make, wound like the first.
fn fuse(a: Cell, b: Cell, edge: (u32, u32)) -> Option<[u32; 4]> {
    let (Cell::Tri(t1), Cell::Tri(t2)) = (a, b) else {
        return None;
    };
    // In `t1`, the shared edge runs one way; the apex is the third corner.
    let s = (0..3).find(|&s| {
        let (u, v) = (t1[s], t1[(s + 1) % 3]);
        (u.min(v), u.max(v)) == edge
    })?;
    let (p, q, apex1) = (t1[s], t1[(s + 1) % 3], t1[(s + 2) % 3]);
    let apex2 = *t2.iter().find(|&&v| v != p && v != q)?;
    // Walking p → apex2 → q → apex1 goes round the union the way `t1` turns.
    Some([p, apex2, q, apex1])
}

/// Walk one triangle toward the nearest other and fuse them. Returns whether
/// anything moved.
fn walk_a_pair_together(cells: &mut [Cell], pts: &[Point2]) -> bool {
    let tris: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].is_tri()).collect();
    if tris.len() < 2 {
        return false;
    }
    for &start in &tris {
        let Some(path) = path_to_another_triangle(cells, start) else {
            continue;
        };
        if path.len() > MAX_WALK {
            continue;
        }
        if walk_along(cells, pts, start, &path) {
            return true;
        }
    }
    false
}

/// Shortest run of quadrangles from the triangle at `start` to another
/// triangle, in the dual graph. The last entry is that other triangle.
fn path_to_another_triangle(cells: &[Cell], start: usize) -> Option<Vec<usize>> {
    let edges = edge_map(cells);
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); cells.len()];
    let mut keys: Vec<&(u32, u32)> = edges.keys().collect();
    keys.sort_unstable();
    for k in keys {
        let o = &edges[k];
        if o.len() == 2 {
            neighbours[o[0]].push(o[1]);
            neighbours[o[1]].push(o[0]);
        }
    }
    let mut came: HashMap<usize, usize> = HashMap::new();
    let mut queue = VecDeque::from([start]);
    came.insert(start, start);
    while let Some(c) = queue.pop_front() {
        for &n in &neighbours[c] {
            if came.contains_key(&n) {
                continue;
            }
            came.insert(n, c);
            if cells[n].is_tri() {
                let mut path = vec![n];
                let mut cur = n;
                while cur != start {
                    cur = came[&cur];
                    if cur != start {
                        path.push(cur);
                    }
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(n);
        }
    }
    None
}

/// Re-split the pentagons along `path` so the triangle at `start` advances,
/// then fuse it with the triangle at the end. All or nothing.
fn walk_along(cells: &mut [Cell], pts: &[Point2], start: usize, path: &[usize]) -> bool {
    let backup: Vec<Cell> = cells.to_vec();
    let mut here = start;
    for (step, &next) in path.iter().enumerate() {
        let last = step + 1 == path.len();
        if last {
            // `next` is the other triangle: fuse rather than walk.
            let Some(edge) = shared_edge(cells[here], cells[next]) else {
                break;
            };
            let Some(quad) = fuse(cells[here], cells[next], edge) else {
                break;
            };
            if !quad_is_valid(quad.map(|v| pts[v as usize])) {
                break;
            }
            cells[here] = Cell::Quad(quad);
            cells[next] = Cell::Dead;
            return true;
        }
        match advance(cells[here], cells[next], pts) {
            Some((moved_tri, made_quad)) => {
                cells[here] = Cell::Quad(made_quad);
                cells[next] = Cell::Tri(moved_tri);
                here = next;
            }
            None => break,
        }
    }
    cells.copy_from_slice(&backup);
    false
}

/// The undirected edge two cells share, if they share exactly one.
fn shared_edge(a: Cell, b: Cell) -> Option<(u32, u32)> {
    let (na, nb) = (a.nodes(), b.nodes());
    for i in 0..na.len() {
        let (u, v) = (na[i], na[(i + 1) % na.len()]);
        let key = (u.min(v), u.max(v));
        for j in 0..nb.len() {
            let (x, y) = (nb[j], nb[(j + 1) % nb.len()]);
            if (x.min(y), x.max(y)) == key {
                return Some(key);
            }
        }
    }
    None
}

/// Re-split the pentagon a triangle and a quadrangle make, so the triangle
/// ends up elsewhere. Returns the moved triangle and the quadrangle left
/// behind.
///
/// The pentagon's outline is not touched, only the cut inside it, so this move
/// can never change the mesh's boundary — which is what makes walking a
/// triangle across the mesh safe against the contract by construction.
fn advance(tri: Cell, quad: Cell, pts: &[Point2]) -> Option<([u32; 3], [u32; 4])> {
    let (Cell::Tri(t), Cell::Quad(q)) = (tri, quad) else {
        return None;
    };
    let edge = shared_edge(tri, quad)?;
    // The triangle runs p → r along the shared edge, so the quadrangle runs
    // r → p; its two far corners close the pentagon.
    let s = (0..3).find(|&s| {
        let (u, v) = (t[s], t[(s + 1) % 3]);
        (u.min(v), u.max(v)) == edge
    })?;
    let (p, r, apex) = (t[s], t[(s + 1) % 3], t[(s + 2) % 3]);
    let k = (0..4).find(|&k| q[k] == r && q[(k + 1) % 4] == p)?;
    // Walking p → far → far → r → apex goes round the union once.
    let ring = [p, q[(k + 2) % 4], q[(k + 3) % 4], r, apex];

    // Five ways to cut an ear off a pentagon; ear `i` is the triangle
    // (i, i+1, i+2) and the quadrangle (i+2, i+3, i+4, i). The cut it came in
    // with is ear 3 — every other one moves the triangle.
    let mut best: Option<(f64, [u32; 3], [u32; 4])> = None;
    for ear in 0..5 {
        if ear == 3 {
            continue;
        }
        let tri_new = [ring[ear], ring[(ear + 1) % 5], ring[(ear + 2) % 5]];
        let quad_new = [
            ring[(ear + 2) % 5],
            ring[(ear + 3) % 5],
            ring[(ear + 4) % 5],
            ring[ear],
        ];
        let a = orient(
            pts[tri_new[0] as usize],
            pts[tri_new[1] as usize],
            pts[tri_new[2] as usize],
        );
        if a <= 0.0 || !quad_is_valid(quad_new.map(|v| pts[v as usize])) {
            continue;
        }
        // Prefer the fattest triangle: a sliver is both a poor cell and a poor
        // thing to have to walk any further.
        if best.is_none_or(|(b, _, _)| a > b) {
            best = Some((a, tri_new, quad_new));
        }
    }
    best.map(|(_, t, q)| (t, q))
}
