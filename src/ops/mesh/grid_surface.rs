//! Grid-cored paving of a closed contour into quadrangles.
//!
//! Same input and same output as [`pave_surface`](fn@super::pave_surface), and
//! the same guarantee that the contour is untouchable — only the interior is
//! obtained differently. Where `pave_surface` walks a front inward until two of
//! its rows meet, `grid_surface` fills everything more than a few cells from
//! the boundary with a **tensor grid**, and lets the front pave only the strip
//! that is left.
//!
//! ## Why this exists
//!
//! The two families of quadrangle mesher fail at opposite ends. A front is
//! perfect where it starts, since its first row follows the contour exactly,
//! and doubtful where two of its rows meet, since there it must reconcile two
//! discretisations that have no reason to agree — that meeting line is what
//! carries the valence defects, the leftover triangles and the flattest cells.
//! A grid is the mirror image: every cell is a rectangle by construction, and
//! all of its difficulty is at the boundary. Taking the interior from the grid
//! and the boundary from the front leaves neither weakness in the result.
//!
//! On a rectilinear contour whose dimensions are multiples of the target size,
//! the outcome is the structured mesh drawn by hand: every cell a rectangle,
//! every Jacobian 1, and no triangle anywhere.
//!
//! ## The grid is laid in the contour's own direction
//!
//! Nothing in the contract ties the grid to the frame's axes: the orientation
//! is a free internal choice. So it is taken from the contour — the angle that
//! turns the greatest length of it into axis-aligned edges, sought over a
//! quarter turn, which is all a grid's four-fold symmetry leaves distinct.
//!
//! Without it the method would be a trick that only works on shapes someone
//! happened to draw square-on. A rectangle turned by 30° used to come out at
//! 454 cells, 20 of them triangles, with a Jacobian down to 0.138 — worse on
//! its worst cell than the frontal paver. It now comes out at 450 perfect
//! rectangles, exactly as if it had never been turned.
//!
//! A contour with no dominant direction — a circle, whose edge angles are
//! spread evenly — keeps the axes, since turning it would swap one arbitrary
//! orientation for another. And the axes win ties, so a shape already square
//! with the frame is never lost to a rounding-width improvement elsewhere.
//!
//! ## The core meets the contour rather than stopping short of it
//!
//! The grid's lines are taken from the contour, in two steps. First the
//! shape's own structure: consecutive contour edges running the same way are
//! grouped into a **chain**, and a chain at least a cell long pins a line at
//! the coordinate it lies on. It is the chain's length that is judged, never
//! its pieces' — a wall is a wall whether the caller cut it in three or in
//! thirty.
//!
//! Then the subdivision between two consecutive lines. Where a chain of the
//! contour spans that gap end to end, **its own nodes are the lines**: the
//! caller has already decided how many pieces the side takes, and recomputing
//! that decision here only means disagreeing with it sooner or later. A side
//! 5.5 cells long is five or six depending on which way the halfway mark
//! rounds, and one disagreement sends the whole side to the band. Where no
//! chain spans the gap, it is subdivided uniformly at about the target size, as
//! there is then nothing to agree with.
//!
//! A grid node that lands on a contour node **is** that node — the same vertex,
//! not a copy — so the core reaches the boundary instead of stopping a hair
//! short of it.
//!
//! What is left for the front is the contour and the core's boundary minus the
//! edges they share, since a segment walked once each way bounds nothing at
//! all. On a rectilinear domain laid out on the grid nothing is left, no front
//! runs, and the mesh is the grid. Elsewhere the leftovers chain into loops the
//! front paves as usual — with one difference: a loop belonging entirely to the
//! core is **frozen**. It stays live, so the front sees it, keeps clear of it
//! and seams onto it, but lays no row of its own. Two live fronts meet wherever
//! they happen to; one front *lands*, on an interface that was chosen.
//!
//! ## The contour has to be discretised for a grid
//!
//! This is the one thing asked of the caller, and it is a real constraint. A
//! grid can only meet a contour whose nodes fall on grid lines, and the grid's
//! lines are dictated by the shape's own features. Take a profile 0.6 wide made
//! of nine steps: the steps put lines every 0.0667, so cells of about 0.00375
//! give 18 columns per step, at 0.0037037. A base discretised as one straight
//! run of 160 segments of 0.00375 misses every one of them, by 1.2 % — enough
//! that not a single node is shared and the whole boundary falls to the front.
//! Cutting that base under each step instead, so each piece gets its own 18
//! segments, costs nothing and shares every node.
//!
//! The rule is therefore: **break every side at the shape's own corners.** How
//! many cells each piece then takes is the caller's business and no longer has
//! to be guessed at — the grid follows whatever cutting it is given, so a side
//! cut into five where the arithmetic would have said six is honoured, not
//! overruled. What the grid cannot do is honour two cuttings of the same span
//! at once, and a side running past a corner asks exactly that. Nothing checks
//! it, because a contour that does not satisfy it is not an error — it just
//! gets more band and less grid.
//!
//! ## When it degrades
//!
//! A region too thin to hold a single core cell gets no core, and the front
//! paves it alone exactly as `pave_surface` would. A contour off the grid gets
//! a wide band and the same treatment. The degradation is continuous: at worst
//! the quality of the frontal paver, never an error.

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
/// want to stop early, use [`grid_surface_cancellable`].
pub fn grid_surface(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    band: usize,
    all_quad: bool,
) -> Result<Mesh> {
    grid_surface_cancellable(
        contour,
        element_type,
        target_size,
        band,
        all_quad,
        &NoCancel,
    )
}

/// Like [`grid_surface`], but polls `cancel` between rows so meshing can be
/// stopped early (returning [`PyrucastError::Interrupted`]).
pub fn grid_surface_cancellable(
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
            "grid_surface: element_type must be QUA4, QUA8 or QUA9, got {element_type}"
        )));
    }
    if let Some(h) = target_size {
        if h <= 0.0 || h.is_nan() {
            return Err(PyrucastError::Message(format!(
                "grid_surface: target_size must be > 0, got {h}"
            )));
        }
    }
    let parsed = contour::parse(contour, "grid_surface")?;
    let mut fabrics = Vec::with_capacity(parsed.domains.len());
    for d in &parsed.domains {
        fabrics.push(paving::pave_grid(d, target_size, all_quad, band, cancel)?);
    }

    let qua4 = paving::materialize(&parsed, fabrics, "grid_surface")?;
    super::sweep::finish_surface(qua4, element_type, "grid_surface")
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{Node, NodeId, Point2};
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::store::insert;

    /// A closed SEG2 loop through `pts`, in the order given.
    fn loop_mesh(coords: crate::store::Handle<Coords>, pts: &[(f64, f64)]) -> Mesh {
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
        let coords = insert(Coords::new(2).unwrap());
        let corners = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.0, 0.3)];
        let contour = loop_mesh(coords, &on_grid(&corners, 0.02));
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.02), 0, false).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &on_grid(&corners, size));
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(size), 0, false).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
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
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();

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
            let coords = insert(Coords::new(2).unwrap());
            let corners = spin(&[(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.0, 0.3)], deg);
            let contour = loop_mesh(coords, &on_grid(&corners, 0.02));
            let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.02), 0, false).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &on_grid(&spin(&corners, 23.7), size));
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(size), 0, false).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
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
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();

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
        let coords = insert(Coords::new(2).unwrap());
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
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let contour = loop_mesh(
            coords,
            &(0..64)
                .map(|i| {
                    let t = i as f64 / 64.0 * std::f64::consts::TAU;
                    (t.cos(), t.sin())
                })
                .collect::<Vec<_>>(),
        );
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.1), 0, false).unwrap();
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
    fn a_curved_contour_is_not_left_full_of_cracks() {
        // The core's nodes are held still, but they are *ours*: the weld may
        // give one up to close a slither between core and contour. Held as
        // undiscardable — which they were, sharing one flag with the contour's
        // — the weld could not close those slithers at all, and each one
        // stayed as a hole in the mesh. A circle is where they abound: this
        // used to leave 88 boundary edges that were not the contour's.
        let coords = insert(Coords::new(2).unwrap());
        let pts: Vec<(f64, f64)> = (0..60)
            .map(|i| {
                let t = i as f64 / 60.0 * std::f64::consts::TAU;
                (t.cos(), t.sin())
            })
            .collect();
        let contour = loop_mesh(coords, &on_grid(&pts, 0.05));
        let mesh = grid_surface(&contour, ElementType::QUA4, Some(0.05), 0, false).unwrap();

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
            .filter(|(k, &v)| v == 1 && !on_contour.contains(k))
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
    fn bad_input_is_rejected_with_a_named_error() {
        let coords = insert(Coords::new(2).unwrap());
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
                grid_surface(&contour, et, size, 0, false).unwrap_err()
            );
            assert!(msg.starts_with("grid_surface:"), "{msg}");
        }
    }
}
