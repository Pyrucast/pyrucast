//! Frontal paving of a closed contour into quadrangles.
//!
//! Where [`triangulate_surface`](super::triangulate_surface) triangulates by
//! constrained Delaunay and can only *recombine* triangles into quadrangles
//! afterwards, `pave_surface` lays quadrangles down directly, in rows walking
//! inward from the boundary. That is the whole point: the rows follow the
//! contour, which is where finite-element accuracy is usually decided, and the
//! result is quadrangular by construction rather than by luck of pairing.
//!
//! Pipeline per domain (one outer counter-clockwise loop and its clockwise
//! holes, parsed by [`super::contour`]):
//!
//! 1. Seed the advancing front with the domain's boundary loops.
//! 2. Lay a whole row of quadrangles along a loop
//!    ([`paving::row`](super::paving::row)): each front node is given as many
//!    quadrangles as its interior angle asks for, at the local element size.
//! 3. Refuse and retreat if the row would produce a quadrangle that is not
//!    strictly convex, or edges that cross the front. Every such test runs on
//!    the exact predicates of [`super::predicates`].
//! 4. Seam front nodes that have come within touching distance — which splits
//!    a loop where the domain is concave and joins two loops where a hole is
//!    being swallowed.
//! 5. Close small loops with quadrangles ([`paving::close`](super::paving::close)).
//! 6. Smooth, under a validity guard that never moves a contour node.
//!
//! A 3-D contour is fitted to its best plane, paved there, and lifted back.
//!
//! ## Triangles, and how to have none
//!
//! A polygon with an even number of sides can always be filled with
//! quadrangles alone; an odd one always leaves exactly one triangle. Paving
//! provably cannot change that parity — a row preserves it and a seam removes
//! two nodes — so the count is decided by the contour before meshing starts.
//! With `all_quad`, any boundary loop with an odd number of segments therefore
//! receives **one** extra node, on its longest segment, and the result is
//! guaranteed free of triangles. Without it, the leftover triangles come back
//! in a separate `TRI3` submesh.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{ElementType, Mesh, Node, NodeId, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::ops::mesher::{contour, paving};
use crate::store::insert;

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
        fabrics.push(paving::pave(d, target_size, all_quad, cancel)?);
    }

    let qua4 = materialize(&parsed, fabrics)?;
    match element_type {
        ElementType::QUA4 => Ok(qua4),
        ElementType::QUA8 => super::to_quadratic(&qua4),
        ElementType::QUA9 => super::sweep::qua8_to_qua9(&super::to_quadratic(&qua4)?),
        _ => unreachable!("element type already validated"),
    }
}

/// Turn the per-domain fabrics into a `Mesh` on the contour's own `Coords`.
fn materialize(parsed: &contour::Contour, fabrics: Vec<paving::Fabric>) -> Result<Mesh> {
    let coords = &parsed.coords;
    let mut quad_sub: Option<SubMesh> = None;
    let mut tri_sub: Option<SubMesh> = None;
    let mut kept: Vec<Node> = Vec::new();

    for fab in fabrics {
        let mut flat: Vec<NodeId> = fab.contour_ids.clone();
        for p in &fab.pts[fab.contour_ids.len()..] {
            let node = Node::create_in(coords.clone(), &parsed.frame.to_world(*p, parsed.dim))?;
            flat.push(node.id());
            kept.push(node);
        }
        if !fab.quads.is_empty() {
            let sub =
                quad_sub.get_or_insert_with(|| SubMesh::new(coords.clone(), ElementType::QUA4));
            for q in &fab.quads {
                sub.add_cell(&[
                    flat[q[0] as usize],
                    flat[q[1] as usize],
                    flat[q[2] as usize],
                    flat[q[3] as usize],
                ])?;
            }
        }
        if !fab.tris.is_empty() {
            let sub =
                tri_sub.get_or_insert_with(|| SubMesh::new(coords.clone(), ElementType::TRI3));
            for t in &fab.tris {
                sub.add_cell(&[
                    flat[t[0] as usize],
                    flat[t[1] as usize],
                    flat[t[2] as usize],
                ])?;
            }
        }
    }

    let mut mesh = Mesh::empty();
    if let Some(q) = quad_sub {
        if q.cell_count() > 0 {
            mesh.add_sub(insert(q))?;
        }
    }
    if let Some(t) = tri_sub {
        if t.cell_count() > 0 {
            mesh.add_sub(insert(t))?;
        }
    }
    drop(kept);
    if mesh.is_empty() {
        return Err(PyrucastError::Message(
            "pave_surface: produced no cell".into(),
        ));
    }
    Ok(mesh)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, Point2};
    use crate::store::{insert, read};
    use std::collections::HashMap;
    use std::sync::atomic::AtomicBool;

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
        use crate::ops::mesher::paving::geom::{orient, quad_quality, tri_quality};
        let coords = mesh.coords().unwrap();
        let c = read(&coords).unwrap();
        let at = |id: NodeId| {
            let v = c.coord(id).unwrap();
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
            let s = read(sm).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
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
        let coords = insert(Coords::new(2).unwrap());
        let outer = loop_mesh(coords.clone(), &rect_loop(3.0, 1.0, 30, 10));
        let hole = loop_mesh(coords, &circle_loop(2.25, 0.5, 0.35, 32, true));
        let contour = outer.union(&hole).unwrap();
        let mesh = pave_surface(&contour, ElementType::QUA4, Some(0.1), false).unwrap();
        let r = inspect(&mesh);
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
        let coords = insert(Coords::new(2).unwrap());
        let contour = loop_mesh(coords, &rect_loop(1.0, 1.0, 20, 20));
        let flag = AtomicBool::new(true);
        let err = pave_surface_cancellable(&contour, ElementType::QUA4, Some(0.02), false, &flag)
            .unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }
}
