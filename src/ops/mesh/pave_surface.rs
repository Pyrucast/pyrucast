//! Frontal paving of a closed contour into quadrangles.
//!
//! Where [`triangulate_surface`](fn@super::triangulate_surface) triangulates by
//! constrained Delaunay and can only *recombine* triangles into quadrangles
//! afterwards, `pave_surface` lays quadrangles down directly, in rows walking
//! inward from the boundary. That is the whole point: the rows follow the
//! contour, which is where finite-element accuracy is usually decided, and the
//! result is quadrangular by construction rather than by luck of pairing.
//!
//! Pipeline per domain (one outer counter-clockwise loop and its clockwise
//! holes, parsed by `contour`):
//!
//! 1. Seed the advancing front with the domain's boundary loops.
//! 2. Lay a whole row of quadrangles along a loop (`paving::row`): each
//!    front node is given as many quadrangles as its interior angle asks for,
//!    at the local element size.
//! 3. Refuse and retreat if the row would produce a quadrangle that is not
//!    strictly convex, or edges that cross the front. Every such test runs on
//!    the exact predicates of `predicates`.
//! 4. Seam front nodes that have come within touching distance — which splits
//!    a loop where the domain is concave and joins two loops where a hole is
//!    being swallowed.
//! 5. Close small loops with quadrangles (`paving::close`).
//! 6. Smooth, under a validity guard that never moves a contour node.
//!
//! A 3-D contour is fitted to its best plane, paved there, and lifted back.
//!
//! ## The contour is untouchable
//!
//! Every node of the contour comes back in the mesh, at its own position, and
//! **no node is ever added on a boundary edge**. The boundary discretisation
//! is the caller's: it usually carries the boundary conditions, and a node
//! silently inserted in the middle of a segment would be a node nobody asked
//! for. Nothing in the paver may split a boundary edge, and the seam — the one
//! operation that discards a node — refuses to discard a contour one.
//!
//! The corollary is that a contour the paver cannot work with is reported, not
//! worked around. There are two such cases and both come back as an error
//! naming the problem:
//!
//! - `all_quad` on a loop with an **odd** number of segments. A polygon with
//!   an odd number of sides has no filling by quadrangles alone, paving
//!   provably cannot change that parity — a row preserves it, a seam removes
//!   two nodes — and evening the count out would mean adding a boundary node.
//! - a contour so coarse or so uneven for the requested size that the
//!   advancing front **folds onto itself**, leaving a region that cannot be
//!   filled. The error names where.
//!
//! Left to itself (`all_quad = false`) an odd loop simply costs one triangle,
//! returned in a separate `TRI3` submesh, along with the few cells a distorted
//! leftover polygon could not make square — the closure prefers a pair of
//! triangles to a cell with a negative Jacobian, and validity is never traded
//! away.

use crate::atoms::ElementType;
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::ops::mesh::{contour, paving};

/// Pave the interior of `contour` — one or more closed `SEG2` loops, counter-
/// clockwise outer and clockwise holes, per [`super::border()`]'s convention —
/// with `element_type` cells.
///
/// `element_type` is `QUA4`, `QUA8` or `QUA9`; the quadratic forms are derived
/// from the `QUA4` mesh. `target_size` sets the wanted edge length, `None`
/// takes each domain's mean boundary edge length. `all_quad` guarantees a
/// triangle-free result, at the cost of at most one extra node per boundary
/// loop of odd length.
///
/// Contour nodes are reused as they are and never moved. The result carries a
/// `QUA4` submesh and, only when triangles were left over, a `TRI3` one.
///
/// This is the uninterruptible convenience form; for a long mesh a caller may
/// want to stop early, use [`pave_surface_cancellable`].
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
/// // Le pavage frontal : des quadrangles posés en rangées qui suivent le
/// // contour, plutôt que des triangles appariés après coup.
/// let m = mesh::pave_surface(&contour, ElementType::QUA4, Some(0.5), false)?;
/// assert!(m.cell_count()? > 0);
/// // `all_quad` exige que rien ne reste triangulaire — ce qui n'est
/// // possible que si le contour a un nombre **pair** de segments.
/// assert!(mesh::pave_surface(&contour, ElementType::QUA4, Some(0.5), true).is_ok());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn pave_surface(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    all_quad: bool,
) -> Result<Mesh> {
    pave_surface_cancellable(contour, element_type, target_size, all_quad, &NoCancel)
}

/// Like [`pave_surface`], but polls `cancel` between rows so meshing can be
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
/// # use std::sync::atomic::{AtomicBool, Ordering};
/// // Le jeton est sondé aux points de contrôle du mailleur : armé d'avance,
/// // l'appel s'arrête au premier d'entre eux.
/// let stop = AtomicBool::new(true);
/// assert!(mesh::pave_surface_cancellable(
///     &contour, ElementType::QUA4, Some(0.5), false, &stop).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn pave_surface_cancellable(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    all_quad: bool,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if !matches!(
        element_type,
        ElementType::QUA4 | ElementType::QUA8 | ElementType::QUA9
    ) {
        return Err(PyrucastError::Message(format!(
            "pave_surface: element_type must be QUA4, QUA8 or QUA9, got {element_type}"
        )));
    }
    if let Some(h) = target_size {
        if h <= 0.0 || h.is_nan() {
            return Err(PyrucastError::Message(format!(
                "pave_surface: target_size must be > 0, got {h}"
            )));
        }
    }

    let parsed = contour::parse(contour, "pave_surface")?;
    let mut fabrics = Vec::with_capacity(parsed.domains.len());
    for d in &parsed.domains {
        fabrics.push(paving::pave(
            d,
            target_size,
            all_quad,
            cancel,
            "pave_surface",
        )?);
    }

    let qua4 = paving::materialize(&parsed, fabrics, "pave_surface")?;
    // The quad family only; the up-front validation above already rejected
    // anything else, so the error arm is unreachable here.
    super::sweep::finish_surface(qua4, element_type, "pave_surface")
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
    use std::collections::HashMap;
    use std::sync::atomic::AtomicBool;

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

    fn rect_loop(w: f64, h: f64, nx: usize, ny: usize) -> Vec<(f64, f64)> {
        let mut v = Vec::new();
        for i in 0..nx {
            v.push((w * i as f64 / nx as f64, 0.0));
        }
        for i in 0..ny {
            v.push((w, h * i as f64 / ny as f64));
        }
        for i in 0..nx {
            v.push((w - w * i as f64 / nx as f64, h));
        }
        for i in 0..ny {
            v.push((0.0, h - h * i as f64 / ny as f64));
        }
        v
    }

    fn circle_loop(cx: f64, cy: f64, r: f64, n: usize, clockwise: bool) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let mut t = i as f64 / n as f64 * std::f64::consts::TAU;
                if clockwise {
                    t = -t;
                }
                (cx + r * t.cos(), cy + r * t.sin())
            })
            .collect()
    }

    /// Cell counts, areas and the conformity of the produced mesh.
    struct Report {
        quads: usize,
        tris: usize,
        area: f64,
        min_quality: f64,
        non_conforming: usize,
    }

    fn inspect(mesh: &Mesh) -> Report {
        use crate::ops::mesh::paving::geom::{orient, quad_quality, tri_quality};
        let coords = mesh.coords().unwrap();
        let c = coords.read();
        let at = |id: NodeId| {
            let v = c.position(id).unwrap();
            Point2::new(v[0], v[1])
        };
        let mut r = Report {
            quads: 0,
            tris: 0,
            area: 0.0,
            min_quality: f64::INFINITY,
            non_conforming: 0,
        };
        let mut edges: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        let mut bump = |a: NodeId, b: NodeId| {
            let k = if a.0 < b.0 { (a, b) } else { (b, a) };
            *edges.entry(k).or_insert(0) += 1;
        };
        for sm in mesh {
            let s = sm.read();
            let npc = s.element_type().nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                for i in 0..npc {
                    bump(cell[i], cell[(i + 1) % npc]);
                }
                let p: Vec<Point2> = cell.iter().map(|&n| at(n)).collect();
                if npc == 4 {
                    r.quads += 1;
                    r.area += 0.5 * (orient(p[0], p[1], p[2]) + orient(p[0], p[2], p[3]));
                    r.min_quality = r.min_quality.min(quad_quality([p[0], p[1], p[2], p[3]]));
                } else {
                    r.tris += 1;
                    r.area += 0.5 * orient(p[0], p[1], p[2]);
                    r.min_quality = r.min_quality.min(tri_quality(p[0], p[1], p[2]));
                }
            }
        }
        r.non_conforming = edges.values().filter(|&&v| v > 2).count();
        r
    }

    #[test]
    fn a_square_is_paved_with_quadrangles_only() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &rect_loop(1.0, 1.0, 8, 8));
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.125), false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.tris, 0, "{} quads, {} tris", r.quads, r.tris);
        assert!((r.area - 1.0).abs() < 1e-9, "area {}", r.area);
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_quality > 0.0, "min quality {}", r.min_quality);
    }

    #[test]
    fn a_plate_with_a_hole_is_paved_and_covers_its_area() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer = loop_mesh(coords.clone(), &rect_loop(3.0, 1.0, 30, 10));
        let hole = loop_mesh(coords, &circle_loop(2.25, 0.5, 0.35, 32, true));
        let contour = outer.union(&hole).unwrap();
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.1), false).unwrap();
        let r = inspect(&mesh);
        // Seaming discards one of two vertices, so the contour has to be
        // checked here too — this is the geometry where the outer front and
        // the hole's front meet and get seamed together.
        {
            let all: std::collections::HashSet<NodeId> = mesh
                .into_iter()
                .flat_map(|sm| sm.read().connectivity().to_vec())
                .collect();
            for sub in [&outer[0], &hole[0]] {
                for n in sub.read().connectivity() {
                    assert!(all.contains(n), "contour node {n:?} was seamed away");
                }
            }
        }
        let expect = 3.0 - std::f64::consts::PI * 0.35 * 0.35;
        println!(
            "plate: {} quads, {} tris, area {:.6} (hole polygon {:.6}), q_min {:.3}",
            r.quads, r.tris, r.area, expect, r.min_quality
        );
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_quality > 0.0);
        // The hole is a 32-gon, so a little larger than the disc it samples.
        assert!((r.area - expect).abs() < 0.01, "area {}", r.area);
    }

    #[test]
    fn cancellable_stops_on_a_preset_flag() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &rect_loop(1.0, 1.0, 20, 20));
        let flag = AtomicBool::new(true);
        let err = pave_surface_cancellable(&contour, ElementType::QUA4, Some(0.02), false, &flag)
            .unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    fn a_concave_l_shape_is_paved_without_leaving_the_material() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let mut pts = Vec::new();
        let step = 0.1;
        for i in 0..20 {
            pts.push((i as f64 * step, 0.0));
        }
        for i in 0..10 {
            pts.push((2.0, i as f64 * step));
        }
        for i in 0..15 {
            pts.push((2.0 - i as f64 * step, 1.0));
        }
        for i in 0..10 {
            pts.push((0.5, 1.0 + i as f64 * step));
        }
        for i in 0..5 {
            pts.push((0.5 - i as f64 * step, 2.0));
        }
        for i in 0..20 {
            pts.push((0.0, 2.0 - i as f64 * step));
        }
        let contour = loop_mesh(coords, &pts);
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.1), false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_quality > 0.0, "min quality {}", r.min_quality);
        // 2×1 foot plus 0.5×1 leg. A hair of tolerance: where two parts of
        // the front graze each other the remnant between them is dropped
        // rather than filled with cells no solver could integrate.
        assert!((r.area - 2.5).abs() < 1e-3, "area {}", r.area);
    }

    #[test]
    fn narrow_bands_meeting_head_on_do_not_turn_the_front_inside_out() {
        // A crenellated profile: two full-height towers with a run of shallow
        // notches between them. Every band under a notch is only a handful of
        // cells deep, so the front coming up from the base and the one coming
        // down from the notch meet head-on all along it, leaving slithers
        // behind. A row laid on a slither used to reverse its loop, after
        // which every further row inflated it until it left the material
        // entirely — reported, much later and far away, as a fold.
        let coords = Handle::new(Coords::new(2).unwrap());
        let (u, v) = (0.6 / 9.0, 0.3 / 40.0);
        let levels = [40.0, 3.0, 6.0, 4.0, 6.0, 4.0, 6.0, 3.0, 40.0];
        let mut corners: Vec<(f64, f64)> = vec![(0.0, 0.0), (0.6, 0.0)];
        for (i, l) in levels.iter().enumerate().rev() {
            corners.push(((i + 1) as f64 * u, l * v));
            corners.push((i as f64 * u, l * v));
        }

        let size = 2.0 * v / 4.0; // four cells across the shortest step
        let mut pts = Vec::new();
        let n = corners.len();
        for i in 0..n {
            let (a, b) = (corners[i], corners[(i + 1) % n]);
            let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            let k = (len / size).round().max(4.0) as usize;
            for j in 0..k {
                let t = j as f64 / k as f64;
                pts.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
            }
        }

        let contour = loop_mesh(coords, &pts);
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(size), false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.non_conforming, 0);
        assert!(r.min_quality > 0.0, "min quality {}", r.min_quality);
        // 0.6 × 0.3 towers over a 0.6-wide base, minus the notches.
        let area: f64 = levels.iter().map(|l| u * l * v).sum();
        assert!((r.area - area).abs() < 1e-3, "area {} for {area}", r.area);
    }

    #[test]
    fn all_quad_on_an_odd_contour_is_an_error_rather_than_a_silent_triangle() {
        // 4 + 4 + 4 + 5 = 17 segments. A polygon with an odd number of sides
        // has no filling by quadrangles alone, and the contour is the
        // caller's: nothing here may add a node to it to even the count out.
        // So the honest answer is to say so.
        let coords = Handle::new(Coords::new(2).unwrap());
        let mut pts = rect_loop(1.0, 1.0, 4, 4);
        pts.push((0.0, 0.125));
        assert_eq!(pts.len() % 2, 1);
        let contour = loop_mesh(coords, &pts);

        let err = pave_surface(&contour, ElementType::QUA4, Some(0.25), true).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.starts_with("pave_surface:"), "{msg}");
        assert!(
            msg.contains("17 segments"),
            "the count should be named: {msg}"
        );
        assert!(msg.contains("odd"), "{msg}");

        // Left to itself the same contour meshes fine, paying the one
        // triangle that parity makes unavoidable.
        let r = inspect(&pave_surface(&contour, ElementType::QUA4, Some(0.25), false).unwrap());
        assert!(r.tris <= 1, "odd parity costs at most one: {}", r.tris);
        assert_eq!(r.non_conforming, 0);
        assert!((r.area - 1.0).abs() < 2e-2, "area {}", r.area);
    }

    #[test]
    fn the_contour_nodes_come_back_untouched() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let pts = rect_loop(1.0, 1.0, 6, 6);
        let contour = loop_mesh(coords.clone(), &pts);
        let before: Vec<(NodeId, Vec<f64>)> = {
            let c = coords.read();
            let s = contour[0].read();
            s.connectivity()
                .iter()
                .map(|&n| (n, c.position(n).unwrap().to_vec()))
                .collect()
        };
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.15), false).unwrap();
        let c = coords.read();
        for (id, coord) in &before {
            assert_eq!(
                &c.position(*id).unwrap().to_vec(),
                coord,
                "node {id:?} moved"
            );
        }
        // And they are still the ones the mesh is built on.
        let used: HashMap<NodeId, ()> = mesh
            .into_iter()
            .flat_map(|sm| sm.read().connectivity().to_vec())
            .map(|n| (n, ()))
            .collect();
        for (id, _) in &before {
            assert!(used.contains_key(id), "contour node {id:?} was abandoned");
        }
    }

    #[test]
    fn the_mesh_boundary_is_exactly_the_contour() {
        // The strongest statement of the contract: every contour segment is a
        // boundary edge of the mesh, and there is no other boundary edge —
        // so no node was added on the boundary, none was dropped, and no hole
        // was left anywhere inside.
        let coords = Handle::new(Coords::new(2).unwrap());
        let pts = rect_loop(3.0, 1.0, 30, 10);
        let contour = loop_mesh(coords, &pts);
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.1), false).unwrap();

        let mut used: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        for sm in &mesh {
            let s = sm.read();
            let npc = s.element_type().nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                for i in 0..npc {
                    let (a, b) = (cell[i], cell[(i + 1) % npc]);
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *used.entry(key).or_insert(0) += 1;
                }
            }
        }
        let contour_conn = contour[0].read();
        for seg in contour_conn.connectivity().chunks(2) {
            let key = if seg[0].0 < seg[1].0 {
                (seg[0], seg[1])
            } else {
                (seg[1], seg[0])
            };
            assert_eq!(
                used.get(&key),
                Some(&1),
                "contour segment {key:?} is not on the boundary"
            );
        }
        let boundary = used.values().filter(|&&u| u == 1).count();
        assert_eq!(
            boundary,
            pts.len(),
            "the mesh boundary has {boundary} edges for {} contour segments",
            pts.len()
        );
    }

    #[test]
    fn a_planar_contour_in_3d_is_paved_in_its_own_plane() {
        // The formation benchmark lives in the y = 0 plane of a 3-D Coords.
        let coords = Handle::new(Coords::new(3).unwrap());
        let ids: Vec<NodeId> = rect_loop(1.0, 1.0, 6, 6)
            .iter()
            .map(|&(x, z)| Node::create_in(coords.clone(), &[x, 0.0, z]).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        let n = ids.len();
        for i in 0..n {
            sm.add_cell(&[ids[i], ids[(i + 1) % n]]).unwrap();
        }
        let contour = Mesh::from_submesh(sm);
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.2), false).unwrap();
        let c = coords.read();
        for sm in &mesh {
            for &node in sm.read().connectivity() {
                assert!(
                    c.position(node).unwrap()[1].abs() < 1e-12,
                    "a node left the y = 0 plane"
                );
            }
        }
    }

    #[test]
    fn quadratic_forms_are_derived_from_the_quadrangles() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &rect_loop(1.0, 1.0, 4, 4));
        for (et, npc) in [(ElementType::QUA8, 8), (ElementType::QUA9, 9)] {
            let mesh = pave_surface(&contour, et, Some(0.25), true).unwrap();
            assert_eq!(mesh.element_types().unwrap(), vec![et]);
            assert_eq!(mesh[0].read().element_type().nodes_per_cell(), npc);
        }
    }

    #[test]
    fn bad_input_is_rejected_with_a_named_error() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords.clone(), &rect_loop(1.0, 1.0, 4, 4));
        for err in [
            pave_surface(&contour, ElementType::TRI3, None, false).unwrap_err(),
            pave_surface(&contour, ElementType::QUA4, Some(0.0), false).unwrap_err(),
            pave_surface(&contour, ElementType::QUA4, Some(-1.0), false).unwrap_err(),
        ] {
            assert!(
                format!("{err}").starts_with("pave_surface:"),
                "unhelpful error: {err}"
            );
        }
        // A contour of the wrong element type is caught by the shared parser,
        // which still names this operator.
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let d = Node::create_in(coords, &[0.0, 1.0]).unwrap();
        sm.add_cell(&[a.id(), b.id(), d.id()]).unwrap();
        let err =
            pave_surface(&Mesh::from_submesh(sm), ElementType::QUA4, None, false).unwrap_err();
        assert!(format!("{err}").starts_with("pave_surface:"), "{err}");
    }

    #[test]
    fn the_default_size_follows_the_contour_discretisation() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &rect_loop(1.0, 1.0, 10, 10));
        let mesh = pave_surface(&contour, ElementType::QUA4, None, false).unwrap();
        let r = inspect(&mesh);
        assert_eq!(r.non_conforming, 0);
        assert!((r.area - 1.0).abs() < 1e-9);
        // Boundary spacing 0.1 over a unit square: of the order of 100 cells.
        assert!(
            (40..400).contains(&(r.quads + r.tris)),
            "{} cells",
            r.quads + r.tris
        );
    }

    /// Throughput and shape, on the plate-with-a-hole benchmark. Not part of
    /// the normal run: `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "performance check, run explicitly with --ignored"]
    fn perf_and_quality_plate_with_hole_under_30s() {
        use crate::ops::mesh::paving::geom::{quad_quality, tri_quality};
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer = loop_mesh(coords.clone(), &rect_loop(0.30, 0.10, 60, 20));
        let hole = loop_mesh(coords, &circle_loop(0.225, 0.05, 0.035, 64, true));
        let contour = outer.union(&hole).unwrap();

        let t0 = std::time::Instant::now();
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.00029), false).unwrap();
        let dt = t0.elapsed();

        let c = mesh.coords().unwrap().read();
        let at = |id: NodeId| {
            let v = c.position(id).unwrap();
            Point2::new(v[0], v[1])
        };
        let (mut nq, mut nt) = (0usize, 0usize);
        let mut worst = f64::INFINITY;
        let mut under = 0usize;
        for sm in &mesh {
            let s = sm.read();
            let npc = s.element_type().nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                let p: Vec<Point2> = cell.iter().map(|&n| at(n)).collect();
                let q = if npc == 4 {
                    nq += 1;
                    quad_quality([p[0], p[1], p[2], p[3]])
                } else {
                    nt += 1;
                    tri_quality(p[0], p[1], p[2])
                };
                worst = worst.min(q);
                if q < 0.5 {
                    under += 1;
                }
            }
        }
        let total = nq + nt;
        let rate = total as f64 / dt.as_secs_f64();
        println!(
            "pave_surface: {total} cells ({nq} QUA4 + {nt} TRI3) in {:.2} s\n\
             \x20 rate {:.0} cells/s | quads {:.1} % | worst scaled Jacobian {:.3} | below 0.5: {:.2} %",
            dt.as_secs_f64(),
            rate,
            100.0 * nq as f64 / total as f64,
            worst,
            100.0 * under as f64 / total as f64,
        );
        assert!(dt.as_secs_f64() < 30.0, "took {:?}", dt);
        assert!(rate > 10_000.0, "only {rate:.0} cells/s");
        assert!(worst > 0.0, "an inverted cell got through: {worst}");
        assert!(
            nq as f64 / total as f64 > 0.95,
            "only {:.1} % quadrangles",
            100.0 * nq as f64 / total as f64
        );
    }

    /// A full quality read-out on the plate benchmark, for judging the mesh
    /// rather than merely asserting it is valid. Run explicitly:
    /// `cargo test --release -- --ignored quality_report --nocapture`.
    #[test]
    #[ignore = "reporting run, not an assertion"]
    fn quality_report_plate_with_hole() {
        use crate::ops::mesh::paving::geom::{quad_quality, tri_quality};
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer = loop_mesh(coords.clone(), &rect_loop(0.30, 0.10, 60, 20));
        let hole = loop_mesh(coords, &circle_loop(0.225, 0.05, 0.035, 64, true));
        let contour = outer.union(&hole).unwrap();
        let target = 0.0016;

        let t0 = std::time::Instant::now();
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(target), false).unwrap();
        let dt = t0.elapsed();

        let c = mesh.coords().unwrap().read();
        let at = |id: NodeId| {
            let v = c.position(id).unwrap();
            Point2::new(v[0], v[1])
        };

        let mut jac: Vec<f64> = Vec::new();
        let mut angles: Vec<f64> = Vec::new();
        let mut aspect: Vec<f64> = Vec::new();
        let mut edges: Vec<f64> = Vec::new();
        let mut valence: HashMap<NodeId, usize> = HashMap::new();
        let mut edge_use: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        let (mut nq, mut nt) = (0usize, 0usize);

        for sm in &mesh {
            let s = sm.read();
            let npc = s.element_type().nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                let p: Vec<Point2> = cell.iter().map(|&n| at(n)).collect();
                if npc == 4 {
                    nq += 1;
                    jac.push(quad_quality([p[0], p[1], p[2], p[3]]));
                } else {
                    nt += 1;
                    jac.push(tri_quality(p[0], p[1], p[2]));
                }
                let mut lo = f64::INFINITY;
                let mut hi: f64 = 0.0;
                for i in 0..npc {
                    let (prev, cur, next) = (p[(i + npc - 1) % npc], p[i], p[(i + 1) % npc]);
                    let (u, w) = (prev - cur, next - cur);
                    let ang = (u.dot(&w) / (u.norm() * w.norm())).clamp(-1.0, 1.0).acos();
                    angles.push(ang.to_degrees());
                    let l = w.norm();
                    lo = lo.min(l);
                    hi = hi.max(l);
                    edges.push(l);
                    *valence.entry(cell[i]).or_insert(0) += 1;
                    let (a, b) = (cell[i], cell[(i + 1) % npc]);
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *edge_use.entry(key).or_insert(0) += 1;
                }
                aspect.push(hi / lo);
            }
        }

        let pct = |v: &mut Vec<f64>, q: f64| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() - 1) as f64 * q) as usize]
        };
        let total = nq + nt;
        let boundary: usize = edge_use.values().filter(|&&u| u == 1).count();
        let interior_nodes: Vec<NodeId> = {
            let mut on_boundary: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for ((a, b), &u) in &edge_use {
                if u == 1 {
                    on_boundary.insert(*a);
                    on_boundary.insert(*b);
                }
            }
            valence
                .keys()
                .filter(|n| !on_boundary.contains(n))
                .copied()
                .collect()
        };
        let mut hist: HashMap<usize, usize> = HashMap::new();
        for n in &interior_nodes {
            *hist.entry(valence[n]).or_insert(0) += 1;
        }
        let irregular = interior_nodes.iter().filter(|n| valence[n] != 4).count();

        println!("\n┌─ pave_surface — plaque 30 × 10 cm, trou r = 3,5 cm, taille visée {target} m");
        println!("│");
        println!(
            "│  mailles          {total}  ({nq} QUA4 + {nt} TRI3, {:.2} % quadrangles)",
            100.0 * nq as f64 / total as f64
        );
        println!(
            "│  temps            {:.2} s  ({:.0} mailles/s)",
            dt.as_secs_f64(),
            total as f64 / dt.as_secs_f64()
        );
        println!(
            "│  nœuds            {}  (dont {} intérieurs)",
            valence.len(),
            interior_nodes.len()
        );
        println!("│  arêtes de bord   {boundary}");
        println!(
            "│  conformité       {}",
            if edge_use.values().all(|&u| u <= 2) {
                "OK — aucune arête à plus de 2 mailles"
            } else {
                "ROMPUE"
            }
        );
        println!("│");
        println!("│  Jacobien normalisé (1 = carré, ≤ 0 = inintégrable)");
        println!(
            "│    min {:.3}   p1 {:.3}   p5 {:.3}   médiane {:.3}   moyenne {:.3}",
            pct(&mut jac.clone(), 0.0),
            pct(&mut jac.clone(), 0.01),
            pct(&mut jac.clone(), 0.05),
            pct(&mut jac.clone(), 0.5),
            jac.iter().sum::<f64>() / jac.len() as f64
        );
        for seuil in [0.0, 0.2, 0.5, 0.7] {
            println!(
                "│    sous {seuil:.1} : {:.3} %  ({} mailles)",
                100.0 * jac.iter().filter(|&&j| j < seuil).count() as f64 / total as f64,
                jac.iter().filter(|&&j| j < seuil).count()
            );
        }
        println!("│");
        println!("│  Angles (degrés)");
        println!(
            "│    min {:.1}   p1 {:.1}   médiane {:.1}   p99 {:.1}   max {:.1}",
            pct(&mut angles.clone(), 0.0),
            pct(&mut angles.clone(), 0.01),
            pct(&mut angles.clone(), 0.5),
            pct(&mut angles.clone(), 0.99),
            pct(&mut angles.clone(), 1.0)
        );
        println!(
            "│    sous 30° : {:.2} %      au-dessus de 150° : {:.2} %",
            100.0 * angles.iter().filter(|&&a| a < 30.0).count() as f64 / angles.len() as f64,
            100.0 * angles.iter().filter(|&&a| a > 150.0).count() as f64 / angles.len() as f64
        );
        println!("│");
        println!("│  Élancement (arête la plus longue / la plus courte)");
        println!(
            "│    médiane {:.2}   p99 {:.2}   max {:.2}",
            pct(&mut aspect.clone(), 0.5),
            pct(&mut aspect.clone(), 0.99),
            pct(&mut aspect.clone(), 1.0)
        );
        println!("│");
        println!("│  Taille d'arête (visée {target})");
        println!(
            "│    p1 {:.5}   médiane {:.5}   p99 {:.5}",
            pct(&mut edges.clone(), 0.01),
            pct(&mut edges.clone(), 0.5),
            pct(&mut edges.clone(), 0.99)
        );
        println!("│");
        println!("│  Valence des nœuds intérieurs (4 = régulier)");
        let mut ks: Vec<usize> = hist.keys().copied().collect();
        ks.sort_unstable();
        for k in ks {
            println!(
                "│    {k} : {:>6}  ({:.2} %)",
                hist[&k],
                100.0 * hist[&k] as f64 / interior_nodes.len() as f64
            );
        }
        println!(
            "│    irréguliers : {irregular} sur {} ({:.2} %)",
            interior_nodes.len(),
            100.0 * irregular as f64 / interior_nodes.len() as f64
        );
        println!("└─");
    }
}
