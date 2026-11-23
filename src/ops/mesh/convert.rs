//! Element-type conversion: re-tile a mesh into a different element type,
//! **keeping the geometry of every cell**.
//!
//! [`convert`] changes the element type of every submesh to `target`, one zone
//! at a time. The supported conversions:
//!
//! - **identity** — `target` equals the submesh type: the submesh is copied
//!   verbatim (same nodes, re-referenced);
//! - **`QUA4 → TRI3`** — each quad splits into two triangles along the
//!   diagonal `(0, 2)`: `(0, 1, 2)` and `(0, 2, 3)`;
//! - **`QUA8 → QUA9`** — each cell gains its centre: the point of the `QUA8`
//!   itself at the middle of its reference square, `−¼ Σ corners + ½ Σ
//!   mid-nodes`. That is the mean of the corners when the edges are straight,
//!   and a point of the curved cell when they are not: the `QUA9` keeps the
//!   geometry of the `QUA8` exactly;
//! - **`QUA9 → TRI6`** — the split of `QUA4 → TRI3`, the centre becoming the
//!   mid-node of the diagonal;
//! - **`QUA8 → TRI6`** — the centre of `QUA8 → QUA9`, then the split of
//!   `QUA9 → TRI6`;
//! - **`HEX8 → TET4`** — each hexahedron splits into six tetrahedra sharing
//!   the main diagonal `(0, 6)` (the Freudenthal/Kuhn subdivision), the same
//!   in every cell. Cutting a hexahedron cuts its faces; two neighbours cut
//!   their common face alike, so the result stays conforming.
//!
//! **`alternate`** applies to the splits of quadrangles (`QUA4 → TRI3`,
//! `QUA9 → TRI6`, `QUA8 → TRI6`). By default every quadrangle is cut along its
//! diagonal `(0, 2)`: an arbitrary mesh — the skin of a solid, say — has no
//! grid order to alternate along, and the volume mesher recovers such skins
//! best cut alike. With `alternate`, a quadrangle of odd rank in its zone is
//! cut along `(1, 3)` instead — `(0, 1, 3)` and `(1, 2, 3)` —, so that on a
//! grid laid out layer by layer, as the structured generators (`sweep`,
//! `transfinite`, `extrude`, `revolve`) lay theirs out, the diagonal
//! alternates from one cell to the next.
//!
//! Corners and mid-nodes are re-used; the only nodes created are the centres
//! of `QUA8 → QUA9` and `QUA8 → TRI6`. Promoting to a higher degree
//! (`TRI3 → TRI6`, …) creates mid-edge nodes shared between zones and is
//! [`to_quadratic`](fn@crate::ops::mesh::to_quadratic)'s job, not this
//! operator's.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// The two splits of one cell of `src` into cells of `target` — for a cell of
/// even rank, then for a cell of odd rank — as local node-index tuples. Read
/// off the element itself.
///
/// Returns an error for any unsupported `(src, target)` pair.
fn split_of(src: ElementType, target: ElementType) -> Result<[&'static [&'static [usize]]; 2]> {
    src.as_kind().split_into(target).ok_or_else(|| {
        PyrucastError::Message(format!(
            "convert: no {src} → {target} conversion (supported: identity, QUA4 → TRI3, \
             QUA8 → QUA9, QUA8 → TRI6, QUA9 → TRI6, HEX8 → TET4)"
        ))
    })
}

/// Convert every submesh of `mesh` to `target` (see the module docs): split
/// each cell into cells of the target type, or give a `QUA8` its centre.
/// `alternate` alternates the diagonal of the quadrangles from one cell of the
/// zone to the next, instead of cutting them all along `(0, 2)`.
///
/// The result mirrors `mesh` submesh by submesh (same order, same face
/// colours). Errors on any submesh whose type cannot be converted to `target`
/// (supported: identity, `QUA4 → TRI3`, `QUA8 → QUA9`, `QUA8 → TRI6`,
/// `QUA9 → TRI6`, `HEX8 → TET4`). `mesh` is left untouched.
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
/// // Two quadrangles side by side. Cut alike, the first triangle of the
/// // second one is its corners 0, 1, 2; alternated, it is its corners 0, 1, 3.
/// # let q = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
/// #          [2.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// # let n: Vec<Node> = q.iter().map(|x| p(x)).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[4].id(), n[5].id()])?;
/// # sm.add_cell(&[n[1].id(), n[2].id(), n[3].id(), n[4].id()])?;
/// let strip = Mesh::from_submesh(sm);
/// let alike = mesh::convert(&strip, ElementType::TRI3, false)?;
/// assert_eq!(alike.cell_count(), 4);
/// assert_eq!(alike.node(0, 2, 2)?.id(), n[3].id());
/// let alternated = mesh::convert(&strip, ElementType::TRI3, true)?;
/// assert_eq!(alternated.node(0, 2, 2)?.id(), n[4].id());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn convert(mesh: &Mesh, target: ElementType, alternate: bool) -> Result<Mesh> {
    let mut result = Mesh::empty();
    for sm_h in mesh {
        // Guard held for the zone's read pass; the `Coords` write of the
        // centres is on another object.
        let zone = convert_zone(&sm_h.read(), target, alternate)?;
        result.add_sub(Handle::new(zone))?;
    }
    Ok(result)
}

/// Convert one zone to `target`, alternating the diagonal of its quadrangles
/// or not (see the module docs), keeping its face colour.
pub(super) fn convert_zone(
    zone: &SubMesh,
    target: ElementType,
    alternate: bool,
) -> Result<SubMesh> {
    let (src, coords, conn) = (zone.element_type(), zone.coords(), zone.connectivity());
    let (out, centres) = if src == target {
        // Identity: the connectivity travels verbatim.
        (conn.to_vec(), 0..0)
    } else if src == ElementType::QUA8 && matches!(target, ElementType::QUA9 | ElementType::TRI6) {
        let (qua9, centres) = with_centres(conn, &coords)?;
        if target == ElementType::QUA9 {
            (qua9, centres)
        } else {
            (split(&qua9, ElementType::QUA9, target, alternate)?, centres)
        }
    } else {
        (split(conn, src, target, alternate)?, 0..0)
    };
    let mut sub = SubMesh::from_connectivity(coords.clone(), target, out)?;
    sub.set_face_color(zone.face_color());
    if !centres.is_empty() {
        // The zone owns the centres now; hand back the unit `add_nodes` gave.
        let owned: Vec<NodeId> = centres.map(NodeId).collect();
        coords.write().decref_all(&owned)?;
    }
    Ok(sub)
}

/// The cells of `conn`, of type `src`, each split into cells of `target`: the
/// split [`split_of`] gives for a cell of even rank, and, with `alternate`,
/// the one it gives for a cell of odd rank to the cells of odd rank.
fn split(
    conn: &[NodeId],
    src: ElementType,
    target: ElementType,
    alternate: bool,
) -> Result<Vec<NodeId>> {
    let [even, odd] = split_of(src, target)?;
    let splits = [even, if alternate { odd } else { even }];
    let npc = src.nodes_per_cell();
    Ok(conn
        .chunks(npc)
        .enumerate()
        .flat_map(|(rank, cell)| {
            splits[rank % 2]
                .iter()
                .flat_map(move |sub| sub.iter().map(|&i| cell[i]))
        })
        .collect())
}

/// The `QUA9` connectivity of the `QUA8` cells of `conn`: each cell followed
/// by its centre, the point of the `QUA8` at the middle of its reference
/// square, `x(0, 0) = −¼ Σ corners + ½ Σ mid-nodes`. Every centre is created
/// in one `Coords` write; the range of their ids comes back with the
/// connectivity, for the caller to hand back once the zone owns them.
fn with_centres(
    conn: &[NodeId],
    coords: &Handle<Coords>,
) -> Result<(Vec<NodeId>, std::ops::Range<u32>)> {
    let positions: Vec<f64> = {
        let c = coords.read();
        let dim = c.dim() as usize;
        let mut flat = Vec::with_capacity(conn.len() / 8 * dim);
        for cell in conn.chunks(8) {
            let mut p: [&[f64]; 8] = [&[]; 8];
            for (k, &id) in cell.iter().enumerate() {
                p[k] = c.position(id)?;
            }
            flat.extend((0..dim).map(|d| {
                -0.25 * (p[0][d] + p[1][d] + p[2][d] + p[3][d])
                    + 0.5 * (p[4][d] + p[5][d] + p[6][d] + p[7][d])
            }));
        }
        flat
    };
    let centres = coords.write().add_nodes(&positions)?;
    let qua9 = conn
        .chunks(8)
        .zip(centres.clone())
        .flat_map(|(cell, centre)| cell.iter().copied().chain(std::iter::once(NodeId(centre))))
        .collect();
    Ok((qua9, centres))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;

    #[test]
    fn qua4_to_tri3_splits_each_quad_in_two() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        m.add_cell(&n).unwrap();

        let tri = convert(&m, ElementType::TRI3, false).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count(), 2);
        // A cell of rank 0: (0,1,2) and (0,2,3), corners re-used, no new node.
        assert_eq!(tri.node(0, 0, 0).unwrap().id(), n[0]);
        assert_eq!(tri.node(0, 0, 1).unwrap().id(), n[1]);
        assert_eq!(tri.node(0, 0, 2).unwrap().id(), n[2]);
        assert_eq!(tri.node(0, 1, 0).unwrap().id(), n[0]);
        assert_eq!(tri.node(0, 1, 1).unwrap().id(), n[2]);
        assert_eq!(tri.node(0, 1, 2).unwrap().id(), n[3]);
    }

    /// The ids of the nodes the two triangles `first` and `first + 1` share —
    /// the diagonal that cut their quadrangle — sorted.
    fn diagonal(mesh: &Mesh, first: usize) -> Vec<NodeId> {
        let ids = |c: usize| -> Vec<NodeId> {
            (0..3).map(|k| mesh.node(0, c, k).unwrap().id()).collect()
        };
        let (a, b) = (ids(first), ids(first + 1));
        let mut common: Vec<NodeId> = a.into_iter().filter(|id| b.contains(id)).collect();
        common.sort();
        common
    }

    #[test]
    fn qua4_to_tri3_alternates_its_diagonals() {
        // Three quadrangles per layer, two layers, ordered layer by layer as a
        // structured generator lays them out.
        let coords = Handle::new(Coords::new(2).unwrap());
        let node = |i: usize, j: usize| -> NodeId {
            Node::create_in(coords.clone(), &[i as f64, j as f64])
                .unwrap()
                .id()
        };
        let grid: Vec<Vec<NodeId>> = (0..=2)
            .map(|j| (0..=3).map(|i| node(i, j)).collect())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        let mut quads = Vec::new();
        for j in 0..2 {
            for i in 0..3 {
                let q = [
                    grid[j][i],
                    grid[j][i + 1],
                    grid[j + 1][i + 1],
                    grid[j + 1][i],
                ];
                m.add_cell(&q).unwrap();
                quads.push(q);
            }
        }
        // Alternated: even rank, diagonal 0-2; odd rank, diagonal 1-3. Rank 3,
        // the first cell of the second layer, is odd: with three cells per
        // layer the pattern is a checkerboard. Not alternated: 0-2 everywhere.
        let alternated = convert(&m, ElementType::TRI3, true).unwrap();
        let alike = convert(&m, ElementType::TRI3, false).unwrap();
        assert_eq!(alternated.cell_count(), 12);
        for (rank, q) in quads.iter().enumerate() {
            let sorted = |mut v: Vec<NodeId>| {
                v.sort();
                v
            };
            let want = if rank % 2 == 0 {
                vec![q[0], q[2]]
            } else {
                vec![q[1], q[3]]
            };
            assert_eq!(diagonal(&alternated, 2 * rank), sorted(want), "rank {rank}");
            assert_eq!(
                diagonal(&alike, 2 * rank),
                sorted(vec![q[0], q[2]]),
                "rank {rank}"
            );
        }
    }

    #[test]
    fn qua8_to_qua9_keeps_the_cell_geometry() {
        // A cell of an arc of radius 7 (15°) extruded along y: its centre is
        // on the cylinder, half a layer up — the mean of its corners would be
        // 0.06 inside.
        let (r, h) = (7.0_f64, 2.0_f64);
        let coords = Handle::new(Coords::new(3).unwrap());
        let on_arc = |deg: f64, y: f64| {
            let a = deg.to_radians();
            Node::create_in(coords.clone(), &[r * a.cos(), y, r * a.sin()])
                .unwrap()
                .id()
        };
        let mid = |a: NodeId, b: NodeId| {
            let (pa, pb) = {
                let c = coords.read();
                (
                    c.position(a).unwrap().to_vec(),
                    c.position(b).unwrap().to_vec(),
                )
            };
            let p: Vec<f64> = pa.iter().zip(&pb).map(|(x, y)| 0.5 * (x + y)).collect();
            Node::create_in(coords.clone(), &p).unwrap().id()
        };
        let c = [
            on_arc(0.0, 0.0),
            on_arc(15.0, 0.0),
            on_arc(15.0, h),
            on_arc(0.0, h),
        ];
        let cell = [
            c[0],
            c[1],
            c[2],
            c[3],
            on_arc(7.5, 0.0),
            mid(c[1], c[2]),
            on_arc(7.5, h),
            mid(c[3], c[0]),
        ];
        let mut curved = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA8));
        curved.add_cell(&cell).unwrap();
        let qua9 = convert(&curved, ElementType::QUA9, false).unwrap();
        let x = qua9.node(0, 0, 8).unwrap().position().unwrap();
        assert!(
            (x[0].hypot(x[2]) - r).abs() < 1e-12,
            "radius {}",
            x[0].hypot(x[2])
        );
        assert!((x[1] - 0.5 * h).abs() < 1e-12);

        // Straight edges: the centre is the mean of the corners.
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap().id();
        let corners = [[0.0, 0.0], [2.0, 0.0], [1.5, 1.0], [0.5, 1.2]];
        let ids: Vec<NodeId> = corners.iter().map(|q| p(q[0], q[1])).collect();
        let mids: Vec<NodeId> = (0..4)
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                p(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            })
            .collect();
        let mut flat = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA8));
        flat.add_cell(&[ids, mids].concat()).unwrap();
        let x = convert(&flat, ElementType::QUA9, false)
            .unwrap()
            .node(0, 0, 8)
            .unwrap()
            .position()
            .unwrap();
        assert!(
            (x[0] - 1.0).abs() < 1e-12 && (x[1] - 0.55).abs() < 1e-12,
            "{x:?}"
        );
    }

    #[test]
    fn quadratic_quadrangles_split_through_their_centre() {
        // Two QUA8 side by side on a straight strip, ranks 0 and 1.
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap().id();
        let row0: Vec<NodeId> = (0..5).map(|i| p(0.5 * i as f64, 0.0)).collect();
        let row1: Vec<NodeId> = (0..5).map(|i| p(0.5 * i as f64, 0.5)).collect();
        let row2: Vec<NodeId> = (0..5).map(|i| p(0.5 * i as f64, 1.0)).collect();
        let cell = |i: usize| {
            [
                row0[i],
                row0[i + 2],
                row2[i + 2],
                row2[i],
                row0[i + 1],
                row1[i + 2],
                row2[i + 1],
                row1[i],
            ]
        };
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA8));
        let cells = [cell(0), cell(2)];
        for c in &cells {
            m.add_cell(c).unwrap();
        }
        let before = coords.read().node_count();

        // QUA8 → TRI6 creates one node per cell, the centre; alternated here,
        // so that both splits show.
        let tri6 = convert(&m, ElementType::TRI6, true).unwrap();
        assert_eq!(tri6.element_types().unwrap(), vec![ElementType::TRI6]);
        assert_eq!(tri6.cell_count(), 4);
        assert_eq!(coords.read().node_count(), before + 2);
        let ids = |t: usize| -> Vec<NodeId> {
            (0..6).map(|k| tri6.node(0, t, k).unwrap().id()).collect()
        };
        // Rank 0, diagonal 0-2: (0, 1, 2 | 4, 5, centre) and (0, 2, 3 | centre, 6, 7).
        let c0 = ids(0)[5];
        assert_eq!(
            ids(0),
            vec![
                cells[0][0],
                cells[0][1],
                cells[0][2],
                cells[0][4],
                cells[0][5],
                c0
            ]
        );
        assert_eq!(
            ids(1),
            vec![
                cells[0][0],
                cells[0][2],
                cells[0][3],
                c0,
                cells[0][6],
                cells[0][7]
            ]
        );
        // Rank 1, diagonal 1-3: (0, 1, 3 | 4, centre, 7) and (1, 2, 3 | 5, 6, centre).
        let c1 = ids(2)[4];
        assert_eq!(
            ids(2),
            vec![
                cells[1][0],
                cells[1][1],
                cells[1][3],
                cells[1][4],
                c1,
                cells[1][7]
            ]
        );
        assert_eq!(
            ids(3),
            vec![
                cells[1][1],
                cells[1][2],
                cells[1][3],
                cells[1][5],
                cells[1][6],
                c1
            ]
        );
        // On straight edges the centre is the middle of the diagonal.
        assert_eq!(
            tri6.node(0, 0, 5).unwrap().position().unwrap(),
            vec![0.5, 0.5]
        );

        // QUA9 → TRI6 creates nothing: the centre is already there. Not
        // alternated, the cell of rank 1 is cut along 0-2 too.
        let qua9 = convert(&m, ElementType::QUA9, false).unwrap();
        let before = coords.read().node_count();
        let again = convert(&qua9, ElementType::TRI6, false).unwrap();
        assert_eq!(coords.read().node_count(), before);
        assert_eq!(
            again.node(0, 1, 3).unwrap().id(),
            qua9.node(0, 0, 8).unwrap().id()
        );
        assert_eq!(
            again.node(0, 2, 2).unwrap().id(),
            qua9.node(0, 1, 2).unwrap().id()
        );
    }

    #[test]
    fn hex8_to_tet4_gives_six_tets_no_new_node() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<NodeId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        let tets = convert(&m, ElementType::TET4, false).unwrap();
        assert_eq!(tets.element_types().unwrap(), vec![ElementType::TET4]);
        assert_eq!(tets.cell_count(), 6);

        // No node created: the coords still hold exactly the 8 corners.
        let live: usize = {
            let c = coords.read();
            n.iter().filter(|&&id| c.refcount(id) > 0).count()
        };
        assert_eq!(live, 8);

        // The six tets tile the unit cube: their volumes sum to 1.
        let vol = |ids: [NodeId; 4]| -> f64 {
            let c = coords.read();
            let p: Vec<Vec<f64>> = ids
                .iter()
                .map(|&i| c.position(i).unwrap().to_vec())
                .collect();
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let e3 = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
            let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
                - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
                + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
            det.abs() / 6.0
        };
        let total: f64 = (0..6)
            .map(|k| {
                vol([
                    tets.node(0, k, 0).unwrap().id(),
                    tets.node(0, k, 1).unwrap().id(),
                    tets.node(0, k, 2).unwrap().id(),
                    tets.node(0, k, 3).unwrap().id(),
                ])
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-12, "six tets must tile the cube");
    }

    #[test]
    fn identity_copies_verbatim() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let out = convert(&m, ElementType::TRI3, false).unwrap();
        assert_eq!(out.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(out.cell_count(), 1);
        assert_eq!(out.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(out.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(out.node(0, 0, 2).unwrap().id(), c.id());
    }

    #[test]
    fn preserves_face_color() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        let color = crate::atoms::RgbColor::new(10, 20, 30);
        sm.set_face_color(color);
        let mut m = Mesh::from_submesh(sm);
        m.add_cell(&n).unwrap();

        let tri = convert(&m, ElementType::TRI3, false).unwrap();
        let out_color = tri.get(0).unwrap().read().face_color();
        assert_eq!(out_color, color);
    }

    #[test]
    fn rejects_unsupported_conversion() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // TRI3 → QUA4 is not a split.
        assert!(convert(&m, ElementType::QUA4, false).is_err());
    }
}
