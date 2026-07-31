//! Hexahedron-dominant volume meshing: a boundary layer grown inward from a
//! closed surface, closed onto a tetrahedral core.
//!
//! Where [`triangulate_volume`](super::triangulate_volume) fills a solid with
//! tetrahedra throughout, `pave_volume` puts **hexahedra where they matter** —
//! in the layer against the boundary, where stress and flux gradients are
//! steepest and where an element's shape decides the accuracy — and leaves the
//! interior, where the field is smooth, to tetrahedra.
//!
//! ```text
//! skin (QUA4 / TRI3)  ──►  offset inward  ──►  HEX8 / PENTA6   (boundary layer)
//!                                                    │
//!                       inner square faces ──► PYRA5 │          (the junction)
//!                                                    ▼
//!                              a void bounded by triangles only
//!                                                    │
//!                                       triangulate_volume  ──►  TET4
//! ```
//!
//! ## Why the pyramids are not optional
//!
//! The layer's inner faces are **squares**, and a tetrahedron has none. Split
//! each square into two triangles and the hexahedron on the other side still
//! sees one square face, with a node hanging in the middle of nothing: the
//! mesh is no longer conforming, and no solver can assemble across that face.
//! A pyramid is the one element that presents a square on one side and
//! triangles on the other, so it is exactly what the junction needs — see
//! [`PYRA5`](crate::containers::mesh::ElementType::PYRA5), whose shape
//! functions reduce to `QUA4`'s on the base and stay linear along the edges to
//! the apex, which is what makes the continuity hold.
//!
//! Once every square face is capped, what is left of the void is bounded by
//! triangles alone, and the existing tetrahedral mesher takes it from there —
//! in **strict** mode, so it reuses those triangles verbatim and the two parts
//! of the mesh meet node for node.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId, Point3, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::ops::mesher::plaster::shell::{Facet, Shell};
use crate::store::{insert, read, Handle};
use std::collections::HashMap;

/// How deep a capping pyramid's apex sits below its base, as a fraction of the
/// base's mean edge.
///
/// Shallower than it looks like it should be, and deliberately. A pyramid half
/// its base wide is the better-shaped cell, but it juts far enough into the
/// void that neighbouring pyramids start cutting through one another and the
/// core mesher is handed a surface with no well-defined inside. A quarter
/// keeps the void close to the shape the layer left, which is what the core
/// mesher can actually fill.
const APEX_DEPTH: f64 = 0.25;

/// Two apexes closer than this many mean edges are the same point, and get the
/// same node.
const APEX_WELD: f64 = 1e-7;

/// Mesh the inside of a closed `QUA4`/`TRI3` envelope with a hexahedral
/// boundary layer over a tetrahedral core.
///
/// `layers` boundary layers are grown inward, each `thickness` deep;
/// `thickness` defaults to the envelope's mean edge length, which gives
/// roughly cube-shaped cells. `core_size` is the target element size for the
/// tetrahedral core, and follows [`triangulate_volume`](super::triangulate_volume)'s
/// convention.
///
/// The envelope's normals must point **out of the material**, exactly as for
/// [`triangulate_volume`](super::triangulate_volume); its nodes are reused as
/// they are. The result carries a `HEX8` submesh (from the quadrangular
/// facets), a `PENTA6` one (from the triangular facets), a `PYRA5` one (the
/// junction) and a `TET4` one (the core), each present only if non-empty.
///
/// This is the uninterruptible convenience form; for a long mesh a caller may
/// want to stop early, use [`pave_volume_cancellable`].
pub fn pave_volume(
    envelope: &Mesh,
    layers: usize,
    thickness: Option<f64>,
    core_size: Option<f64>,
) -> Result<Mesh> {
    pave_volume_cancellable(envelope, layers, thickness, core_size, &NoCancel)
}

/// Like [`pave_volume`], but polls `cancel` while meshing the core so a long
/// run can be stopped early (returning [`PyrucastError::Interrupted`]).
pub fn pave_volume_cancellable(
    envelope: &Mesh,
    layers: usize,
    thickness: Option<f64>,
    core_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if layers == 0 {
        return Err(PyrucastError::Message(
            "pave_volume: layers must be at least 1; for an all-tetrahedron mesh \
             use triangulate_volume"
                .into(),
        ));
    }
    if let Some(t) = thickness {
        if t <= 0.0 || t.is_nan() {
            return Err(PyrucastError::Message(format!(
                "pave_volume: thickness must be > 0, got {t}"
            )));
        }
    }
    let coords = envelope.coords()?;
    let mut shell = Shell::extract(envelope, "pave_volume")?;
    if shell.volume() <= 0.0 {
        return Err(PyrucastError::Message(format!(
            "pave_volume: the envelope encloses a signed volume of {:.3e}, so its normals \
             point into the material rather than out of it. Run `invert` on it.",
            shell.volume()
        )));
    }
    let step = thickness.unwrap_or_else(|| shell.mean_edge());

    let mut kept: Vec<Node> = Vec::new();
    let mut hexes: Vec<[NodeId; 8]> = Vec::new();
    let mut prisms: Vec<[NodeId; 6]> = Vec::new();

    // ── The boundary layers ───────────────────────────────────────────────
    for layer in 0..layers {
        cancel.check()?;
        let offsets = shell.inward_offsets(step);
        let (inner, nodes) = shell.offset_by(&offsets, &coords)?;
        kept.extend(nodes);
        for f in &shell.facets {
            match f {
                // The facet is the cell's outer face and the offset copy its
                // inner one. `HEX8` reads bottom-then-top with the bottom
                // wound counter-clockwise seen from the top, which is what the
                // inner copy is: it carries the facet's own winding, and the
                // facet faces outward.
                Facet::Quad(q) => hexes.push([
                    inner.nodes[q[0] as usize],
                    inner.nodes[q[1] as usize],
                    inner.nodes[q[2] as usize],
                    inner.nodes[q[3] as usize],
                    shell.nodes[q[0] as usize],
                    shell.nodes[q[1] as usize],
                    shell.nodes[q[2] as usize],
                    shell.nodes[q[3] as usize],
                ]),
                Facet::Tri(t) => prisms.push([
                    inner.nodes[t[0] as usize],
                    inner.nodes[t[1] as usize],
                    inner.nodes[t[2] as usize],
                    shell.nodes[t[0] as usize],
                    shell.nodes[t[1] as usize],
                    shell.nodes[t[2] as usize],
                ]),
            }
        }
        if inner.volume() <= 0.0 {
            return Err(PyrucastError::Message(format!(
                "pave_volume: layer {} turns the envelope inside out — a thickness of {step} \
                 is more than the solid can take there. Ask for a thinner layer, fewer of \
                 them, or refine the envelope.",
                layer + 1
            )));
        }
        shell = inner;
    }

    // ── The junction, and the void it leaves ──────────────────────────────
    //
    // Two apexes can land on the same point, and not by accident: where three
    // faces meet at a convex corner, the corner cell of each pushes its apex
    // toward the same place, and at a depth of half the base's edge they
    // coincide exactly. Two pyramids sharing an apex node is perfectly sound —
    // it is two more cells round one vertex. Two *nodes* at the same position
    // is not, so the apexes are looked up by position before being created.
    let tol = shell.mean_edge() * APEX_WELD;
    let mut apexes: HashMap<[i64; 3], NodeId> = HashMap::new();
    let mut pyramids: Vec<[NodeId; 5]> = Vec::new();
    let mut void: Vec<[NodeId; 3]> = Vec::new();
    for f in &shell.facets {
        match f {
            Facet::Quad(q) => {
                let base: Vec<Point3> = q.iter().map(|&i| shell.points[i as usize]).collect();
                let centre = Point3::from(
                    base.iter()
                        .fold(Point3::origin(), |a, p| a + p.coords)
                        .coords
                        / 4.0,
                );
                let edge = (0..4)
                    .map(|i| (base[(i + 1) % 4] - base[i]).norm())
                    .sum::<f64>()
                    / 4.0;
                let normal = quad_normal(&base);
                // The shell's normals point away from the void, so the apex
                // goes the other way.
                let apex = centre - normal * (edge * APEX_DEPTH);
                let key = [
                    (apex.x / tol).round() as i64,
                    (apex.y / tol).round() as i64,
                    (apex.z / tol).round() as i64,
                ];
                let apex_id = match apexes.get(&key) {
                    Some(&id) => id,
                    None => {
                        let a = Node::create_in(coords.clone(), &[apex.x, apex.y, apex.z])?;
                        let id = a.id();
                        kept.push(a);
                        apexes.insert(key, id);
                        id
                    }
                };

                // Seen from the apex the base runs the other way round, so it
                // is reversed for `PYRA5`'s "base counter-clockwise from the
                // apex" convention.
                let b: [NodeId; 4] = [
                    shell.nodes[q[0] as usize],
                    shell.nodes[q[3] as usize],
                    shell.nodes[q[2] as usize],
                    shell.nodes[q[1] as usize],
                ];
                pyramids.push([b[0], b[1], b[2], b[3], apex_id]);
                // The void's boundary at a pyramid is the pyramid's own
                // triangles, taken the other way round: the void's outward
                // normal points *into* the pyramid.
                for i in 0..4 {
                    void.push([b[(i + 1) % 4], b[i], apex_id]);
                }
            }
            // A triangular facet already faces the void with the right
            // orientation: the shell's normal points out of the void.
            Facet::Tri(t) => void.push([
                shell.nodes[t[0] as usize],
                shell.nodes[t[1] as usize],
                shell.nodes[t[2] as usize],
            ]),
        }
    }

    // Two pyramids that share an apex also share the triangle along their
    // common base edge, once from each side. That face is *between* them, not
    // on the void, so the pair cancels out.
    let mut times: HashMap<[u32; 3], usize> = HashMap::new();
    for t in &void {
        let mut k = [t[0].0, t[1].0, t[2].0];
        k.sort_unstable();
        *times.entry(k).or_insert(0) += 1;
    }
    void.retain(|t| {
        let mut k = [t[0].0, t[1].0, t[2].0];
        k.sort_unstable();
        times[&k] == 1
    });

    // ── The core ──────────────────────────────────────────────────────────
    let core = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for t in &void {
            sm.add_cell(t)?;
        }
        let env = Mesh::from_submesh(sm);
        // Strict: the core has to reuse these triangles verbatim, or it would
        // not meet the pyramids node for node.
        super::triangulate_volume_cancellable(&env, core_size, false, cancel).map_err(|e| {
            PyrucastError::Message(format!(
                "pave_volume: the layer went in cleanly but its inner surface could not be \
                 filled with tetrahedra ({e}). The void left by the layer is too awkward \
                 for the core mesher: try a thinner layer or a finer envelope."
            ))
        })?
    };

    materialize(&coords, hexes, prisms, pyramids, core)
}

/// Outward unit normal of a quadrangular facet, by Newell's method so a
/// slightly non-planar quadrangle still gets a sensible answer.
fn quad_normal(p: &[Point3]) -> crate::containers::mesh::Vector3 {
    let mut n = crate::containers::mesh::Vector3::zeros();
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    let len = n.norm();
    if len == 0.0 {
        n
    } else {
        n / len
    }
}

fn materialize(
    coords: &Handle<Coords>,
    hexes: Vec<[NodeId; 8]>,
    prisms: Vec<[NodeId; 6]>,
    pyramids: Vec<[NodeId; 5]>,
    core: Mesh,
) -> Result<Mesh> {
    let mut mesh = Mesh::empty();
    let mut push = |et: ElementType, cells: &[&[NodeId]]| -> Result<()> {
        if cells.is_empty() {
            return Ok(());
        }
        let mut sm = SubMesh::new(coords.clone(), et);
        for c in cells {
            sm.add_cell(c)?;
        }
        mesh.add_sub(insert(sm))
    };
    let h: Vec<&[NodeId]> = hexes.iter().map(|c| c.as_slice()).collect();
    let p: Vec<&[NodeId]> = prisms.iter().map(|c| c.as_slice()).collect();
    let y: Vec<&[NodeId]> = pyramids.iter().map(|c| c.as_slice()).collect();
    push(ElementType::HEX8, &h)?;
    push(ElementType::PENTA6, &p)?;
    push(ElementType::PYRA5, &y)?;
    for sub in &core {
        mesh.add_sub(sub.clone())?;
    }
    if mesh.is_empty() {
        return Err(PyrucastError::Message(
            "pave_volume: produced no cell".into(),
        ));
    }
    let _ = read(coords)?;
    Ok(mesh)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use std::collections::HashMap;

    /// The skin of a `nx × ny × nz` box of hexahedra: a closed QUA4 shell with
    /// outward normals, built with the operators rather than by hand.
    fn box_skin(nx: usize, ny: usize, nz: usize) -> Mesh {
        let coords = insert(Coords::new(3).unwrap());
        let corner = |x: f64, z: f64| Node::create_in(coords.clone(), &[x, 0.0, z]).unwrap();
        let (a, b) = (corner(0.0, 0.0), corner(1.0, 0.0));
        let (c, d) = (corner(1.0, 1.0), corner(0.0, 1.0));
        let line = |p: &Node, q: &Node, n: usize| {
            crate::ops::mesher::line(p, q, n, ElementType::SEG2).unwrap()
        };
        // Wound so the face's normal is +y, the direction it is extruded in:
        // otherwise every hexahedron comes out inside-out and so does the skin.
        let mut ring = line(&a, &d, nz);
        for m in [line(&d, &c, nx), line(&c, &b, nz), line(&b, &a, nx)] {
            ring = ring.union(&m).unwrap();
        }
        let contour = crate::ops::mesher::consolidate(&ring).unwrap();
        let face = super::super::pave_surface(&contour, ElementType::QUA4, None, true).unwrap();
        let solid = crate::ops::mesher::extrude(&face, &[0.0, 1.0, 0.0], ny).unwrap();
        crate::ops::mesher::skin(&solid, None).unwrap()
    }

    /// Cells per element type.
    fn kinds(mesh: &Mesh) -> HashMap<ElementType, usize> {
        mesh.element_types()
            .unwrap()
            .into_iter()
            .zip(mesh.cell_counts().unwrap())
            .collect()
    }

    /// Every facet of every cell, counted. A conforming closed mesh uses each
    /// interior facet twice and each boundary facet once.
    fn facet_use(mesh: &Mesh) -> HashMap<Vec<NodeId>, usize> {
        let mut count = HashMap::new();
        for sm in mesh {
            let s = read(sm).unwrap();
            let et = s.element_type();
            let npc = et.nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                for f in crate::ops::mesher::orient::facets_of(et) {
                    let mut key: Vec<NodeId> = f.iter().map(|&i| cell[i]).collect();
                    key.sort_unstable_by_key(|n| n.0);
                    *count.entry(key).or_insert(0) += 1;
                }
            }
        }
        count
    }

    #[test]
    fn a_box_gets_a_hexahedral_layer_over_a_tetrahedral_core() {
        let skin = box_skin(3, 3, 3);
        let mesh = pave_volume(&skin, 1, Some(0.15), Some(0.4)).unwrap();
        let k = kinds(&mesh);
        // Six faces of 3 × 3 quadrangles, one hexahedron and one pyramid each.
        assert_eq!(k.get(&ElementType::HEX8), Some(&54), "{k:?}");
        assert_eq!(k.get(&ElementType::PYRA5), Some(&54), "{k:?}");
        assert!(k.get(&ElementType::TET4).is_some_and(|&n| n > 0), "{k:?}");
        assert!(!k.contains_key(&ElementType::PENTA6), "{k:?}");
    }

    #[test]
    fn the_result_is_conforming_and_its_boundary_is_the_envelope() {
        let skin = box_skin(3, 3, 3);
        let mesh = pave_volume(&skin, 1, Some(0.15), Some(0.4)).unwrap();
        let used = facet_use(&mesh);
        assert!(
            used.values().all(|&n| n <= 2),
            "a facet is shared by more than two cells"
        );
        // The boundary facets are exactly the envelope's own.
        let boundary: usize = used.values().filter(|&&n| n == 1).count();
        assert_eq!(
            boundary, 54,
            "the mesh boundary should be the 54 skin quads"
        );
    }

    #[test]
    fn every_cell_has_a_positive_volume() {
        let skin = box_skin(3, 3, 3);
        let mesh = pave_volume(&skin, 1, Some(0.15), Some(0.4)).unwrap();
        let coords = mesh.coords().unwrap();
        for sub in &mesh {
            let space = crate::containers::finite_element_space::SubFiniteElementSpace::new(
                sub.clone(),
                crate::containers::finite_element_space::Interpolation::Lagrange1,
                crate::containers::finite_element_space::QuadratureRule::Gauss,
            )
            .unwrap();
            for c in 0..read(sub).unwrap().cell_count() {
                for g in 0..space.gauss_count() {
                    let det = space.det_jacobian(c, g).unwrap();
                    assert!(
                        det > 0.0,
                        "cell {c} of {} has |J| = {det}",
                        space.element_type().unwrap()
                    );
                }
            }
        }
        let _ = coords;
    }

    #[test]
    fn several_layers_stack_inward() {
        let skin = box_skin(3, 3, 3);
        let mesh = pave_volume(&skin, 2, Some(0.08), Some(0.4)).unwrap();
        let k = kinds(&mesh);
        assert_eq!(k.get(&ElementType::HEX8), Some(&108), "{k:?}");
        assert_eq!(k.get(&ElementType::PYRA5), Some(&54), "{k:?}");
    }

    #[test]
    fn an_envelope_turned_inside_out_is_named_as_such() {
        let skin = crate::ops::mesher::invert(&box_skin(2, 2, 2)).unwrap();
        let err = pave_volume(&skin, 1, Some(0.15), Some(0.4)).unwrap_err();
        assert!(
            format!("{err}").contains("normals point into the material"),
            "{err}"
        );
    }

    #[test]
    fn too_thick_a_layer_is_refused_with_a_reason() {
        let skin = box_skin(2, 2, 2);
        let err = pave_volume(&skin, 1, Some(2.0), Some(0.4)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.starts_with("pave_volume:"), "{msg}");
    }
}
