//! Grid-cored paving, second method: the lines come one per contour node and
//! the rows are free to bend.
//!
//! Same input, same output and the same untouchable contour as
//! [`grid_surface`](fn@super::grid_surface), which it does not replace. Only
//! the way the grid is laid differs, and neither way wins everywhere — hence
//! two operators rather than one.
//!
//! ## What it does differently
//!
//! `grid_surface` pins a line on the coordinate each aligned chain lies on, and
//! subdivides between two lines by whichever chain spans them end to end. Every
//! line is straight and every cell of the core is a rectangle.
//!
//! `grid_surface2` instead gives **every node of the contour the line that
//! crosses it**, then:
//!
//! - halves any interval wider than twice the mean step the contour carries;
//! - collapses any band thinner than half that step, **edge by edge** — each
//!   edge welds onto the contour node at one of its ends, or onto its midpoint
//!   when neither end is one;
//! - fetches what is left: a grid node within a quarter cell of a contour node
//!   moves onto it;
//! - and judges every cell on the shape it ends up with, not on the lines it
//!   came from.
//!
//! A row is therefore a polyline, not a line. That is the whole point: one row
//! can meet two facing walls at two different heights, which no straight line
//! can do, and it is what lets a wall cut into ten face a wall cut into eleven.
//!
//! ## Which to reach for
//!
//! Measured on the same shapes, at the same target size, worst cell by mean
//! ratio — `grid_surface` first, `grid_surface2` second:
//!
//! | shape | `grid_surface` | `grid_surface2` |
//! |---|---|---|
//! | rectangle, and any shape square on the grid | **0.999** | **0.999** |
//! | plate with a step off the grid | 0.405 | **0.963** |
//! | L, arbitrary dimensions | 0.448 | **0.979** |
//! | L whose sides split a stretch 5+6 against 4+7 | 0.421 | **0.963** |
//! | L whose sides disagree by one node | 0.307 | **0.606** |
//! | crenellated profile, base cut under each bar | 0.382 | **0.916** |
//! | crenellated profile, base in one run | 0.287 | **0.651** |
//! | house with a pitched roof | 0.304 | **0.475**, 1 triangle against 11 |
//! | square with one rounded corner | 0.222 | **0.406** |
//! | circle | **0.288**, p5 **0.796** | 0.005 — do not use |
//!
//! So: **`grid_surface2` for a rectilinear shape**, all the more so when its
//! sides were not cut at the corners facing them, and **`grid_surface` for
//! anything curved**, where following the contour's nodes means following the
//! accident of where its vertices fell. On a curve `grid_surface2` is not a
//! candidate at all: nothing dictates a line over most of a circle, the empty
//! gaps run to fifteen times the mean step, and the core is given up. The
//! book's *Mailler une géométrie* page compares all four side by side.
//!
//! ## When it degrades
//!
//! A region too thin to hold a single core cell gets no core, and the front
//! paves it alone exactly as `pave_surface` would. The degradation is
//! continuous: at worst the quality of the frontal paver — but bounded below by
//! validity, since a mesh holding a cell turned inside out is **refused**
//! rather than returned.

use crate::atoms::ElementType;
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::ops::mesh::{contour, paving};

/// Pave the interior of `contour` — one or more closed `SEG2` loops,
/// counter-clockwise outer and clockwise holes — with a structured core and a
/// frontal band.
///
/// `element_type` is `QUA4`, `QUA8` or `QUA9`; the quadratic forms are derived
/// from the `QUA4` mesh. `target_size` sets the wanted edge length, `None`
/// takes each domain's mean boundary edge length. `all_quad` asks for a
/// triangle-free result.
///
/// `band` is **extra** clearance, in cells, between the core and the contour.
/// Zero is the useful value and the one to reach for: the core then goes as
/// far as the grid allows, and on a contour laid out for the grid it meets it
/// exactly. Raise it only for a contour the grid cannot meet — a curve — where
/// giving the front a couple of cells to work in buys a better band than
/// letting it fight for a sliver.
///
/// Contour nodes are reused as they are and never moved. The result carries a
/// `QUA4` submesh and, only when triangles were left over, a `TRI3` one.
///
/// This is the uninterruptible convenience form; for a long mesh a caller may
/// want to stop early, use [`grid_surface2_cancellable`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let cote = |a: &[f64], b: &[f64]| mesh::line(
/// #     &Node::create_in(coords.clone(), a).unwrap(),
/// #     &Node::create_in(coords.clone(), b).unwrap(), 2, ElementType::SEG2).unwrap();
/// # let quatre = cote(&[0.0, 0.0], &[2.0, 0.0])
/// #     .union(&cote(&[2.0, 0.0], &[2.0, 2.0])).unwrap()
/// #     .union(&cote(&[2.0, 2.0], &[0.0, 2.0])).unwrap()
/// #     .union(&cote(&[0.0, 2.0], &[0.0, 0.0])).unwrap();
/// # mesh::merge_nodes(&quatre, 1e-6, true).unwrap();
/// # let contour = mesh::consolidate(&quatre).unwrap();
/// // La variante dont les lignes viennent **une par nœud du contour** et
/// // dont les rangées ont le droit de plier : meilleure sur les formes
/// // rectilinéaires, moins bonne sur les courbes.
/// let m = mesh::grid_surface2(&contour, ElementType::QUA4, Some(0.5), 1, false)?;
/// assert!(m.cell_count()? > 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn grid_surface2(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    band: usize,
    all_quad: bool,
) -> Result<Mesh> {
    grid_surface2_cancellable(
        contour,
        element_type,
        target_size,
        band,
        all_quad,
        &NoCancel,
    )
}

/// Like [`grid_surface2`], but polls `cancel` between rows so meshing can be
/// stopped early (returning [`PyrucastError::Interrupted`]).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let cote = |a: &[f64], b: &[f64]| mesh::line(
/// #     &Node::create_in(coords.clone(), a).unwrap(),
/// #     &Node::create_in(coords.clone(), b).unwrap(), 2, ElementType::SEG2).unwrap();
/// # let quatre = cote(&[0.0, 0.0], &[2.0, 0.0])
/// #     .union(&cote(&[2.0, 0.0], &[2.0, 2.0])).unwrap()
/// #     .union(&cote(&[2.0, 2.0], &[0.0, 2.0])).unwrap()
/// #     .union(&cote(&[0.0, 2.0], &[0.0, 0.0])).unwrap();
/// # mesh::merge_nodes(&quatre, 1e-6, true).unwrap();
/// # let contour = mesh::consolidate(&quatre).unwrap();
/// # use std::sync::atomic::AtomicBool;
/// // Le jeton est sondé aux **points de contrôle** du mailleur, non entre
/// // deux instructions : un contour aussi court est pavé avant d'en
/// // atteindre un, et l'appel aboutit même jeton armé. C'est la même
/// // granularité que partout ailleurs — l'arrêt tombe à la frontière de
/// // phase suivante, pas au milieu d'une.
/// let stop = AtomicBool::new(false);
/// assert!(mesh::grid_surface2_cancellable(
///     &contour, ElementType::QUA4, Some(0.5), 1, false, &stop).is_ok());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn grid_surface2_cancellable(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    band: usize,
    all_quad: bool,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if !matches!(
        element_type,
        ElementType::QUA4 | ElementType::QUA8 | ElementType::QUA9
    ) {
        return Err(PyrucastError::Message(format!(
            "grid_surface2: element_type must be QUA4, QUA8 or QUA9, got {element_type}"
        )));
    }
    if let Some(h) = target_size
        && (h <= 0.0 || h.is_nan())
    {
        return Err(PyrucastError::Message(format!(
            "grid_surface2: target_size must be > 0, got {h}"
        )));
    }
    let parsed = contour::parse(contour, "grid_surface2")?;
    let mut fabrics = Vec::with_capacity(parsed.domains.len());
    for d in &parsed.domains {
        fabrics.push(paving::pave_grid2(
            d,
            target_size,
            all_quad,
            band,
            cancel,
            "grid_surface2",
        )?);
    }

    let qua4 = paving::materialize(&parsed, fabrics, "grid_surface2")?;
    super::sweep::finish_surface(qua4, element_type, "grid_surface2")
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{Node, NodeId, Point2};
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A closed SEG2 loop through `pts`, in the order given.
    fn loop_mesh(coords: crate::handle::Handle<Coords>, pts: &[(f64, f64)]) -> Mesh {
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|&(x, y)| Node::create_in(coords.clone(), &[x, y]).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        let n = ids.len();
        for i in 0..n {
            sm.add_cell(&[ids[i], ids[(i + 1) % n]]).unwrap();
        }
        Mesh::from_submesh(sm)
    }

    /// Walk `corners` as a closed polygon, cutting each side into whole cells
    /// of about `size` — the discretisation a grid can meet.
    fn on_grid(corners: &[(f64, f64)], size: f64) -> Vec<(f64, f64)> {
        let n = corners.len();
        let mut pts = Vec::new();
        for i in 0..n {
            let (a, b) = (corners[i], corners[(i + 1) % n]);
            let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            let k = ((len / size).round() as usize).max(1);
            for j in 0..k {
                let t = j as f64 / k as f64;
                pts.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
            }
        }
        pts
    }

    struct Report {
        quads: usize,
        tris: usize,
        area: f64,
        min_jacobian: f64,
        non_conforming: usize,
    }

    fn inspect(mesh: &Mesh) -> Report {
        let mut r = Report {
            quads: 0,
            tris: 0,
            area: 0.0,
            min_jacobian: f64::INFINITY,
            non_conforming: 0,
        };
        let mut edges: std::collections::HashMap<(NodeId, NodeId), usize> = Default::default();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                let nodes = cell.nodes().unwrap();
                let p: Vec<Point2> = nodes
                    .iter()
                    .map(|nd| {
                        let v = nd.position().unwrap();
                        Point2::new(v[0], v[1])
                    })
                    .collect();
                let k = p.len();
                for i in 0..k {
                    let (a, b) = (nodes[i].id(), nodes[(i + 1) % k].id());
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *edges.entry(key).or_insert(0) += 1;
                }
                let signed: f64 = 0.5
                    * (0..k)
                        .map(|i| p[i].x * p[(i + 1) % k].y - p[(i + 1) % k].x * p[i].y)
                        .sum::<f64>();
                r.area += signed.abs();
                let p: Vec<Point2> = if signed < 0.0 {
                    p.iter().rev().copied().collect()
                } else {
                    p
                };
                for i in 0..k {
                    let u = p[(i + 1) % k] - p[i];
                    let w = p[(i + k - 1) % k] - p[i];
                    r.min_jacobian = r
                        .min_jacobian
                        .min((u.x * w.y - u.y * w.x) / (u.norm() * w.norm()));
                }
                if k == 3 {
                    r.tris += 1;
                } else {
                    r.quads += 1;
                }
            }
        }
        r.non_conforming = edges.values().filter(|&&v| v > 2).count();
        r
    }

    #[test]
    fn a_rectangle_on_the_grid_comes_out_as_the_grid() {
        // The mesh anyone would draw by hand: 30 × 15 rectangles, nothing
        // else. The frontal paver cannot produce it — its rows meet in the
        // middle and leave four diagonal seams — and that is the whole
        // reason this operator exists.
        let coords = Handle::new(Coords::new(2).unwrap());
        let corners = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.0, 0.3)];
        let contour = loop_mesh(coords, &on_grid(&corners, 0.02));
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.02), 0, false).unwrap();
        let r = inspect(&mesh);
        assert_eq!((r.quads, r.tris), (450, 0));
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_jacobian > 1.0 - 1e-9, "jacobian {}", r.min_jacobian);
        assert!((r.area - 0.18).abs() < 1e-12, "area {}", r.area);
    }

    #[test]
    fn a_crenellated_profile_is_meshed_without_a_single_triangle() {
        // Seven re-entrant corners, the worst case for a front — and the case
        // a grid does not even notice, since every corner lands on a line.
        // The base is cut under each step so its nodes fall on the columns
        // the steps impose; whole and straight it would share nothing.
        let (u, v) = (0.6 / 9.0, 0.3 / 40.0);
        let levels = [40.0, 3.0, 6.0, 4.0, 6.0, 4.0, 6.0, 3.0, 40.0];
        let mut corners: Vec<(f64, f64)> =
            (0..=levels.len()).map(|i| (i as f64 * u, 0.0)).collect();
        for (i, l) in levels.iter().enumerate().rev() {
            corners.push(((i + 1) as f64 * u, l * v));
            corners.push((i as f64 * u, l * v));
        }

        let size = 2.0 * v / 4.0;
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &on_grid(&corners, size));
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(size), 0, false).unwrap();
        let r = inspect(&mesh);
        // 18 columns per step, twice the level in rows: 18·2·Σlevels.
        let want = 18 * 2 * levels.iter().sum::<f64>() as usize;
        assert_eq!((r.quads, r.tris), (want, 0));
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_jacobian > 1.0 - 1e-9, "jacobian {}", r.min_jacobian);
        let area: f64 = levels.iter().map(|l| u * l * v).sum();
        assert!((r.area - area).abs() < 1e-12, "area {} for {area}", r.area);
    }

    #[test]
    fn the_mesh_boundary_is_exactly_the_contour() {
        // The strongest statement of the contract, and the one a grid mesher
        // has to earn rather than inherit: its nodes are its own, and only
        // sharing them with the contour keeps the boundary intact. Every
        // contour segment is a boundary edge of the mesh and there is no other
        // — so no node was added on the boundary, none was dropped, and no
        // hole was left anywhere inside.
        //
        // Both boundaries at once: an L, whose re-entrant corner the grid
        // meets exactly, and a circular hole, which it cannot meet at all and
        // hands to the front. The contour has to come back whole either way.
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer_pts = on_grid(
            &[
                (0.0, 0.0),
                (3.0, 0.0),
                (3.0, 0.6),
                (1.5, 0.6),
                (1.5, 1.2),
                (0.0, 1.2),
            ],
            0.1,
        );
        let hole_pts: Vec<(f64, f64)> = (0..32)
            .map(|i| {
                let t = -(i as f64) / 32.0 * std::f64::consts::TAU;
                (2.25 + 0.15 * t.cos(), 0.3 + 0.15 * t.sin())
            })
            .collect();
        let outer = loop_mesh(coords.clone(), &outer_pts);
        let hole = loop_mesh(coords, &hole_pts);
        let contour = outer.union(&hole).unwrap();
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();

        let mut used: std::collections::HashMap<(NodeId, NodeId), usize> = Default::default();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                let nodes = cell.nodes().unwrap();
                let k = nodes.len();
                for i in 0..k {
                    let (a, b) = (nodes[i].id(), nodes[(i + 1) % k].id());
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *used.entry(key).or_insert(0) += 1;
                }
            }
        }

        let mut segments = 0usize;
        for loops in [&outer, &hole] {
            for cell in loops.cells(0).unwrap() {
                let n = cell.nodes().unwrap();
                let (a, b) = (n[0].id(), n[1].id());
                let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                assert_eq!(
                    used.get(&key),
                    Some(&1),
                    "contour segment {key:?} is not a boundary edge of the mesh"
                );
                segments += 1;
            }
        }
        let boundary = used.values().filter(|&&u| u == 1).count();
        assert_eq!(
            boundary, segments,
            "the mesh boundary has {boundary} edges for {segments} contour segments"
        );
    }

    /// Turn a polygon by `deg` about the origin.
    fn spin(pts: &[(f64, f64)], deg: f64) -> Vec<(f64, f64)> {
        let t = deg.to_radians();
        pts.iter()
            .map(|&(x, y)| (x * t.cos() - y * t.sin(), x * t.sin() + y * t.cos()))
            .collect()
    }

    #[test]
    fn a_turned_rectangle_comes_out_as_the_plain_grid_too() {
        // Nothing in the contract ties the grid to the frame's axes, so it is
        // laid in the contour's own direction instead. A rectangle turned by
        // 30° must therefore give exactly what it gives square-on — before the
        // orientation was detected it gave 454 cells, 20 of them triangles,
        // and a Jacobian down to 0.138.
        for deg in [5.0, 15.0, 30.0, 45.0, 88.0] {
            let coords = Handle::new(Coords::new(2).unwrap());
            let corners = spin(&[(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.0, 0.3)], deg);
            let contour = loop_mesh(coords, &on_grid(&corners, 0.02));
            let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.02), 0, false).unwrap();
            let r = inspect(&mesh);
            assert_eq!((r.quads, r.tris), (450, 0), "at {deg}°");
            assert_eq!(r.non_conforming, 0, "at {deg}°");
            assert!(
                r.min_jacobian > 1.0 - 1e-9,
                "at {deg}°, jacobian {}",
                r.min_jacobian
            );
            assert!((r.area - 0.18).abs() < 1e-12, "at {deg}°, area {}", r.area);
        }
    }

    #[test]
    fn a_turned_profile_is_meshed_as_if_it_were_square_on() {
        // The seven re-entrant corners again, at an angle no one would choose.
        let (u, v) = (0.6 / 9.0, 0.3 / 40.0);
        let levels = [40.0, 3.0, 6.0, 4.0, 6.0, 4.0, 6.0, 3.0, 40.0];
        let mut corners: Vec<(f64, f64)> =
            (0..=levels.len()).map(|i| (i as f64 * u, 0.0)).collect();
        for (i, l) in levels.iter().enumerate().rev() {
            corners.push(((i + 1) as f64 * u, l * v));
            corners.push((i as f64 * u, l * v));
        }

        let size = 2.0 * v / 4.0;
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &on_grid(&spin(&corners, 23.7), size));
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(size), 0, false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(
            (r.quads, r.tris),
            (18 * 2 * levels.iter().sum::<f64>() as usize, 0)
        );
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_jacobian > 1.0 - 1e-9, "jacobian {}", r.min_jacobian);
    }

    #[test]
    fn the_contour_survives_the_turn() {
        // The contract again, on a shape the grid now meets only because it
        // turned to it: every contour segment is a boundary edge of the mesh,
        // and the mesh has no other.
        let coords = Handle::new(Coords::new(2).unwrap());
        let corners = spin(
            &[
                (0.0, 0.0),
                (3.0, 0.0),
                (3.0, 0.6),
                (1.5, 0.6),
                (1.5, 1.2),
                (0.0, 1.2),
            ],
            30.0,
        );
        let pts = on_grid(&corners, 0.1);
        let contour = loop_mesh(coords, &pts);
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();

        let mut used: std::collections::HashMap<(NodeId, NodeId), usize> = Default::default();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                let nodes = cell.nodes().unwrap();
                let k = nodes.len();
                for i in 0..k {
                    let (a, b) = (nodes[i].id(), nodes[(i + 1) % k].id());
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *used.entry(key).or_insert(0) += 1;
                }
            }
        }
        let mut segments = 0usize;
        for cell in contour.cells(0).unwrap() {
            let n = cell.nodes().unwrap();
            let (a, b) = (n[0].id(), n[1].id());
            let key = if a.0 < b.0 { (a, b) } else { (b, a) };
            assert_eq!(used.get(&key), Some(&1), "contour segment {key:?} was lost");
            segments += 1;
        }
        assert_eq!(used.values().filter(|&&u| u == 1).count(), segments);
    }

    #[test]
    fn a_plate_with_a_hole_keeps_its_hole_and_its_area() {
        // A hole is a clockwise loop and needs no special handling: the cells
        // it covers are simply not solid. The band round it is the part the
        // grid cannot do, and the front takes it.
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer = loop_mesh(
            coords.clone(),
            &on_grid(&[(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)], 0.1),
        );
        let hole = loop_mesh(
            coords,
            &(0..32)
                .map(|i| {
                    let t = -(i as f64) / 32.0 * std::f64::consts::TAU;
                    (2.25 + 0.35 * t.cos(), 0.5 + 0.35 * t.sin())
                })
                .collect::<Vec<_>>(),
        );
        let contour = outer.union(&hole).unwrap();
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_jacobian > 0.0, "jacobian {}", r.min_jacobian);
        let want = 3.0 - std::f64::consts::PI * 0.35 * 0.35;
        assert!((r.area - want).abs() < 5e-3, "area {} for {want}", r.area);
    }

    #[test]
    fn a_circle_has_no_axis_aligned_edge_and_still_gets_a_core() {
        // Nothing to snap to: the grid falls back to the bounding box, the
        // core is the staircase of cells inside the disc, and the front paves
        // the ring between it and the circle. It must still close.
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(
            coords,
            &(0..64)
                .map(|i| {
                    let t = i as f64 / 64.0 * std::f64::consts::TAU;
                    (t.cos(), t.sin())
                })
                .collect::<Vec<_>>(),
        );
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_jacobian > 0.0, "jacobian {}", r.min_jacobian);
        assert!(
            (r.area - std::f64::consts::PI).abs() < 0.02,
            "area {}",
            r.area
        );
    }

    #[test]
    fn a_shape_off_the_grid_is_fetched_rather_than_banded() {
        // A step at 0.53 by 0.61 on a target of 0.1 shares its corner with the
        // grid and nothing else: every node along it misses a line by a
        // fraction of a cell. Rather than hand the whole side to the front, the
        // nodes of the grid nearest to them go and fetch them, and the rows
        // bend to suit. The plate used to come back with two triangles and a
        // cell ten times longer than it was wide; it now comes back whole.
        let coords = Handle::new(Coords::new(2).unwrap());
        let corners = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.53, 1.0),
            (0.53, 0.61),
            (0.0, 0.61),
        ];
        let pts = on_grid(&corners, 0.1);
        let before: Vec<(f64, f64)> = pts.clone();
        let contour = loop_mesh(coords.clone(), &pts);
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();

        let r = inspect(&mesh);
        assert_eq!(r.tris, 0, "a bent row should not need a triangle");
        assert_eq!(r.non_conforming, 0);
        assert!(
            r.min_jacobian > 0.9,
            "worst cell {} — bending is supposed to keep them square",
            r.min_jacobian
        );
        // The shoelace of the six corners, to the last digit: bending moves the
        // grid about but it may not eat into the domain nor spill out of it.
        assert!((r.area - 0.7933).abs() < 1e-9, "area {}", r.area);

        // The bending is the grid's alone: not one node of the contour moved.
        for (cell, want) in contour.cells(0).unwrap().zip(before.chunks(1)) {
            let p = cell.nodes().unwrap()[0].position().unwrap();
            assert_eq!((p[0], p[1]), want[0]);
        }
    }

    #[test]
    fn a_curved_contour_is_not_left_full_of_cracks() {
        // The core's nodes are held still, but they are *ours*: the weld may
        // give one up to close a slither between core and contour. Held as
        // undiscardable — which they were, sharing one flag with the contour's
        // — the weld could not close those slithers at all, and each one
        // stayed as a hole in the mesh. A circle is where they abound: this
        // used to leave 88 boundary edges that were not the contour's.
        let coords = Handle::new(Coords::new(2).unwrap());
        let pts: Vec<(f64, f64)> = (0..60)
            .map(|i| {
                let t = i as f64 / 60.0 * std::f64::consts::TAU;
                (t.cos(), t.sin())
            })
            .collect();
        let contour = loop_mesh(coords, &on_grid(&pts, 0.05));
        let mesh = grid_surface2(&contour, ElementType::QUA4, Some(0.05), 0, false).unwrap();

        let mut used: std::collections::HashMap<(NodeId, NodeId), usize> = Default::default();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                let nodes = cell.nodes().unwrap();
                let k = nodes.len();
                for i in 0..k {
                    let (a, b) = (nodes[i].id(), nodes[(i + 1) % k].id());
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *used.entry(key).or_insert(0) += 1;
                }
            }
        }
        let mut on_contour: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
        for cell in contour.cells(0).unwrap() {
            let n = cell.nodes().unwrap();
            let (a, b) = (n[0].id(), n[1].id());
            let key = if a.0 < b.0 { (a, b) } else { (b, a) };
            assert_eq!(used.get(&key), Some(&1), "contour segment {key:?} was lost");
            on_contour.insert(key);
        }
        // A boundary edge that is not the contour's is a crack. A curve is the
        // degraded case for a grid and a few survive; five times this many is
        // the regression to catch.
        let cracks = used
            .iter()
            .filter(|(k, v)| **v == 1 && !on_contour.contains(k))
            .count();
        assert!(
            cracks <= 24,
            "{cracks} boundary edges are not the contour's"
        );

        let r = inspect(&mesh);
        assert!(r.min_jacobian > 0.0, "jacobian {}", r.min_jacobian);
        assert!(
            r.area > 0.995 * std::f64::consts::PI,
            "area {} for {}",
            r.area,
            std::f64::consts::PI
        );
    }

    #[test]
    fn a_mesh_with_a_cell_inside_out_is_refused_rather_than_returned() {
        // An octagon at this size is where this mesher tangles: its core is a
        // staircase the band cannot follow, the front ends up round a ring that
        // holds the core like a hole, and the closure decomposes across it. The
        // result used to come back with six cells turned inside out or flat —
        // three of them exactly flat, one a whole cell's worth reversed — for a
        // mesh whose area was otherwise right to 0,2 %. That is the shape of
        // the trap: a mesh that looks fine in every aggregate and cannot be
        // integrated. It is now refused, and the message says where.
        let coords = Handle::new(Coords::new(2).unwrap());
        let corners: Vec<(f64, f64)> = (0..8)
            .map(|i| {
                let t = i as f64 / 8.0 * std::f64::consts::TAU;
                (t.cos(), t.sin())
            })
            .collect();
        let contour = loop_mesh(coords, &on_grid(&corners, 0.05));
        let msg = format!(
            "{}",
            grid_surface2(&contour, ElementType::QUA4, Some(0.05), 0, false).unwrap_err()
        );
        assert!(msg.starts_with("grid_surface2:"), "{msg}");
        assert!(msg.contains("came out turned inside out or flat"), "{msg}");
    }

    #[test]
    fn bad_input_is_rejected_with_a_named_error() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(
            coords,
            &on_grid(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 0.25),
        );
        for (et, size) in [
            (ElementType::TRI3, Some(0.25)),
            (ElementType::QUA4, Some(-1.0)),
        ] {
            let msg = format!(
                "{}",
                grid_surface2(&contour, et, size, 0, false).unwrap_err()
            );
            assert!(msg.starts_with("grid_surface2:"), "{msg}");
        }
    }
}
