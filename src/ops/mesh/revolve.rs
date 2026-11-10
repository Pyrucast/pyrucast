use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Revolve a mesh by `n_layers` layers over a total `angle` (radians) — the
/// rotational companion of [`extrude`](fn@super::extrude).
///
/// Each node is swept along the circle it describes about the rotation centre
/// (2-D) or axis (3-D), and consecutive angular positions are linked by one
/// layer of cells: SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8, and in quadratic
/// SEG3 → QUA8, TRI6 → PENTA15, QUA8 → HEX20 — the mid-node of an edge along
/// the sweep sits at the half angle, on the arc. Other element types produce
/// an error.
///
/// - **2-D** (`center` has 2 components): revolution about the point
///   `center`, counterclockwise for a positive `angle`; `axis` is ignored.
/// - **3-D** (`center` has 3 components): revolution about the line through
///   `center` directed by `axis` (right-handed); `axis` is required and need
///   not be normalized.
///
/// `|angle|` may not exceed a full turn; a full turn **closes** the ring —
/// the last node layer is the first one again, so there is no seam and no
/// duplicated node.
///
/// A **line** may touch the axis by one corner: that node stays put, and the
/// cells touching it close into triangles — SEG2 → TRI3, SEG3 → TRI6 —, a
/// submesh of their own after the quadrangles of their zone. A line from the
/// axis to the rim thus revolves into a disc sector: a fan of triangles at the
/// centre, a ring of quadrangles around it, a single node at the centre. A
/// surface cell may not touch the axis (its prism would collapse), nor may a
/// line lie along it.
///
/// Nodes shared between cells of the source stay shared in the result. Source
/// nodes are re-used (refcount incremented); the other layers are created.
///
/// Node ordering:
/// - QUA4: `bot[0], bot[1], top[1], top[0]`
/// - PENTA6: `bot[0..3], top[0..3]`
/// - HEX8: `bot[0..4], top[0..4]`
/// - QUA8, PENTA15, HEX20: the corners as above, then the mid-nodes in the
///   edge order of the linear cell — the bottom face's, the top face's, then
///   those of the edges along the sweep.
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
/// // Un demi-tour d'un segment autour de l'axe z : quatre QUA4.
/// let l = mesh::line(&p(&[1.0, 0.0, 0.0]), &p(&[1.0, 0.0, 1.0]), 1, ElementType::SEG2)?;
/// let s = mesh::revolve(&l, std::f64::consts::PI, 4, &[0.0, 0.0, 0.0],
///                       Some(&[0.0, 0.0, 1.0]))?;
/// assert_eq!(s.cell_count(), 4);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn revolve(
    mesh: &Mesh,
    angle: f64,
    n_layers: usize,
    center: &[f64],
    axis: Option<&[f64]>,
) -> Result<Mesh> {
    crate::ops::mesh::sweep_kernel::revolve(mesh, angle, n_layers, center, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::mesh::line::line;
    use std::f64::consts::{PI, TAU};

    /// Number of distinct nodes used by the first submesh of `mesh`.
    fn distinct_nodes(mesh: &Mesh) -> usize {
        mesh.get(0).unwrap().read().node_index().len()
    }

    /// A radial SEG2 line, from (1, 0) to (2, 0), in `n` segments.
    fn radial_seg2_2d(n: usize) -> Mesh {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords, &[2.0, 0.0]).unwrap();
        line(&a, &b, n, ElementType::SEG2).unwrap()
    }

    /// A TRI3 face in the plane y = 0, offset from the z axis.
    fn tri3_off_axis() -> Mesh {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[2.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[1.0, 0.0, 1.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        tri
    }

    #[test]
    fn revolve_seg2_to_qua4_2d() {
        let seg = radial_seg2_2d(2);
        let ring = revolve(&seg, PI / 2.0, 3, &[0.0, 0.0], None).unwrap();
        assert_eq!(ring.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(ring.cell_count(), 6);

        // Layer 0 re-uses the source nodes, at the source position.
        assert_eq!(
            ring.node(0, 0, 0).unwrap().position().unwrap(),
            vec![1.0, 0.0]
        );
        // The far side of the first layer sits 30° round, at radius 1.
        let p = ring.node(0, 0, 3).unwrap().position().unwrap();
        assert!((p[0] - (PI / 6.0).cos()).abs() < 1e-12);
        assert!((p[1] - (PI / 6.0).sin()).abs() < 1e-12);
    }

    #[test]
    fn revolve_shared_nodes_stay_shared() {
        let seg = radial_seg2_2d(2);
        let ring = revolve(&seg, PI / 2.0, 1, &[0.0, 0.0], None).unwrap();
        // Node shared by the two source segments: shared on both layers.
        let base_cell0 = ring.node(0, 0, 1).unwrap();
        let base_cell1 = ring.node(0, 1, 0).unwrap();
        assert_eq!(base_cell0.id(), base_cell1.id());
        let top_cell0 = ring.node(0, 0, 2).unwrap();
        let top_cell1 = ring.node(0, 1, 3).unwrap();
        assert_eq!(top_cell0.id(), top_cell1.id());
    }

    #[test]
    fn full_turn_closes_the_ring() {
        let seg = radial_seg2_2d(1);
        let ring = revolve(&seg, TAU, 4, &[0.0, 0.0], None).unwrap();
        assert_eq!(ring.cell_count(), 4);

        // The last layer is the first one again: 4 angular positions × 2
        // radial nodes, not 5.
        assert_eq!(distinct_nodes(&ring), 8);
        let first = ring.node(0, 0, 0).unwrap();
        let last = ring.node(0, 3, 3).unwrap();
        assert_eq!(first.id(), last.id());
    }

    #[test]
    fn quadratic_revolve_puts_its_mid_nodes_on_the_arcs() {
        // A radial SEG3 from radius 1 to 2, a quarter turn in 2 layers: the
        // mid-nodes along the sweep (local 5 and 7 of the QUA8) are on the
        // circles at the half angle, not on the chords.
        use crate::ops::mesh::to_quadratic;
        let seg3 = to_quadratic(&radial_seg2_2d(1)).unwrap();
        let ring = revolve(&seg3, PI / 2.0, 2, &[0.0, 0.0], None).unwrap();
        assert_eq!(ring.element_types().unwrap(), vec![ElementType::QUA8]);
        let polar = |c: usize, k: usize| {
            let x = ring.node(0, c, k).unwrap().position().unwrap();
            ((x[0] * x[0] + x[1] * x[1]).sqrt(), x[1].atan2(x[0]))
        };
        for (cell, angle) in [(0, PI / 8.0), (1, 3.0 * PI / 8.0)] {
            let (r_outer, a_outer) = polar(cell, 5);
            let (r_inner, a_inner) = polar(cell, 7);
            assert!((r_outer - 2.0).abs() < 1e-12 && (a_outer - angle).abs() < 1e-12);
            assert!((r_inner - 1.0).abs() < 1e-12 && (a_inner - angle).abs() < 1e-12);
        }
    }

    #[test]
    fn quadratic_full_turn_closes_the_ring() {
        // 4 layers round a SEG3 of one cell: 4 positions × 3 nodes on the
        // whole levels, 4 × 2 corners on the half levels — 20, no duplicate.
        use crate::ops::mesh::to_quadratic;
        let seg3 = to_quadratic(&radial_seg2_2d(1)).unwrap();
        let ring = revolve(&seg3, TAU, 4, &[0.0, 0.0], None).unwrap();
        assert_eq!(ring.cell_count(), 4);
        assert_eq!(distinct_nodes(&ring), 20);
        assert_eq!(
            ring.node(0, 3, 3).unwrap().id(),
            ring.node(0, 0, 0).unwrap().id()
        );
    }

    #[test]
    fn revolve_tri3_to_penta6_3d() {
        let tri = tri3_off_axis();
        let wedge = revolve(&tri, PI / 6.0, 2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        assert_eq!(wedge.element_types().unwrap(), vec![ElementType::PENTA6]);
        assert_eq!(wedge.cell_count(), 2);

        // Bottom face of the first layer = the source triangle.
        assert_eq!(
            wedge.node(0, 0, 0).unwrap().position().unwrap(),
            vec![1.0, 0.0, 0.0]
        );
        // Its top face sits 15° round the z axis, same radius and height.
        let p = wedge.node(0, 0, 4).unwrap().position().unwrap();
        assert!((p[0] - 2.0 * (PI / 12.0).cos()).abs() < 1e-12);
        assert!((p[1] - 2.0 * (PI / 12.0).sin()).abs() < 1e-12);
        assert!(p[2].abs() < 1e-12);
    }

    #[test]
    fn revolve_qua4_to_hex8_3d() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[2.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0, 0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 0.0, 1.0]).unwrap();
        let mut qua = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        qua.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()]).unwrap();

        let tube = revolve(&qua, TAU, 8, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        assert_eq!(tube.element_types().unwrap(), vec![ElementType::HEX8]);
        assert_eq!(tube.cell_count(), 8);
        // Closed ring: 8 angular positions × 4 section nodes.
        assert_eq!(distinct_nodes(&tube), 32);
    }

    #[test]
    fn revolve_keeps_cells_positively_oriented() {
        // Sweeping counterclockwise about +z from a face whose normal points
        // that way must give prisms with a positive Jacobian.
        use crate::containers::finite_element_space::{
            Interpolation, QuadratureRule, SubFiniteElementSpace,
        };

        let tri = tri3_off_axis();
        let wedge = revolve(&tri, PI / 4.0, 2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        for sub in &wedge {
            let space = SubFiniteElementSpace::new(
                sub.clone(),
                Interpolation::Lagrange1,
                QuadratureRule::Gauss,
            )
            .unwrap();
            for c in 0..sub.read().cell_count() {
                for g in 0..space.gauss_count() {
                    let det = space.det_jacobian(c, g).unwrap();
                    assert!(det > 0.0, "cell {c} has |J| = {det}");
                }
            }
        }
    }

    /// Signed area of a planar polygon (shoelace), counterclockwise positive.
    fn polygon_area(points: &[[f64; 2]]) -> f64 {
        let n = points.len();
        (0..n)
            .map(|i| {
                let (a, b) = (points[i], points[(i + 1) % n]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
            / 2.0
    }

    /// Every cell of `mesh` as its corner polygon, 2-D.
    fn cell_polygons(mesh: &Mesh) -> Vec<(ElementType, Vec<[f64; 2]>)> {
        let coords = mesh.coords().unwrap();
        let c = coords.read();
        let mut out = Vec::new();
        for sm in mesh {
            let sm = sm.read();
            let et = sm.element_type();
            let (npc, corners) = (et.nodes_per_cell(), et.as_kind().corner_count());
            for cell in sm.connectivity().chunks(npc) {
                let poly = cell[..corners]
                    .iter()
                    .map(|&id| {
                        let p = c.position(id).unwrap();
                        [p[0], p[1]]
                    })
                    .collect();
                out.push((et, poly));
            }
        }
        out
    }

    /// A line of 2 SEG2 from the axis, turned by 90° in 7 sectors: 7
    /// quadrangles around 7 triangles, one node at the centre, every cell
    /// counterclockwise, and the area of the inscribed polygon.
    #[test]
    fn a_line_from_the_axis_revolves_into_a_fan_and_a_ring() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let centre = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let rim = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let seg = line(&centre, &rim, 2, ElementType::SEG2).unwrap();
        let sector = revolve(&seg, PI / 2.0, 7, &[0.0, 0.0], None).unwrap();

        assert_eq!(sector.len(), 2);
        let (ring, fan) = (sector.get(0).unwrap(), sector.get(1).unwrap());
        assert_eq!(ring.read().element_type(), ElementType::QUA4);
        assert_eq!(fan.read().element_type(), ElementType::TRI3);
        assert_eq!((ring.read().cell_count(), fan.read().cell_count()), (7, 7));
        let mut all: Vec<_> = ring.read().connectivity().to_vec();
        all.extend_from_slice(fan.read().connectivity());
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 1 + 2 * 8);
        // The centre node is the source's, shared by every triangle.
        for cell in fan.read().connectivity().chunks(3) {
            assert!(cell.contains(&centre.id()));
        }
        let polygons = cell_polygons(&sector);
        assert!(polygons.iter().all(|(_, p)| polygon_area(p) > 0.0));
        let area: f64 = polygons.iter().map(|(_, p)| polygon_area(p)).sum();
        let expected = 7.0 * 0.5 * 4.0 * (PI / 14.0).sin();
        assert!((area - expected).abs() < 1e-12, "{area} ≠ {expected}");
    }

    /// A full turn closes the fan too: no node twice, the centre once.
    #[test]
    fn a_full_turn_from_the_axis_doubles_no_node() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let centre = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let rim = Node::create_in(coords, &[1.0, 0.0]).unwrap();
        // From the rim to the centre: the axis is the second node.
        let seg = line(&rim, &centre, 1, ElementType::SEG2).unwrap();
        let disc = revolve(&seg, TAU, 4, &[0.0, 0.0], None).unwrap();
        assert_eq!(disc.len(), 1);
        assert_eq!(
            disc.get(0).unwrap().read().element_type(),
            ElementType::TRI3
        );
        assert_eq!(distinct_nodes(&disc), 1 + 4);
        // A line running towards the centre sweeps clockwise cells, as its
        // quadrangles would: every triangle turns the same way.
        let polygons = cell_polygons(&disc);
        assert!(polygons.iter().all(|(_, p)| polygon_area(p) < 0.0));
    }

    /// In quadratic, the fan is TRI6: the mid-node of the outer edge sits on
    /// the arc, at the half angle.
    #[test]
    fn a_seg3_from_the_axis_gives_tri6_with_mid_nodes_on_the_arc() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let centre = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let rim = Node::create_in(coords.clone(), &[3.0, 0.0]).unwrap();
        let seg = line(&centre, &rim, 1, ElementType::SEG3).unwrap();
        let fan = revolve(&seg, PI / 2.0, 3, &[0.0, 0.0], None).unwrap();
        assert_eq!(fan.len(), 1);
        let fan = fan.get(0).unwrap();
        assert_eq!(fan.read().element_type(), ElementType::TRI6);
        let c = coords.read();
        for (k, cell) in fan.read().connectivity().chunks(6).enumerate() {
            // [centre, rim at k, rim at k + 1, mid of the first radius,
            //  mid of the arc, mid of the second radius]
            assert_eq!(cell[0], centre.id());
            let arc = c.position(cell[4]).unwrap();
            assert!((arc[0].hypot(arc[1]) - 3.0).abs() < 1e-12);
            let half = (k as f64 + 0.5) * PI / 6.0;
            assert!((arc[1].atan2(arc[0]) - half).abs() < 1e-12);
            for mid in [cell[3], cell[5]] {
                let m = c.position(mid).unwrap();
                assert!((m[0].hypot(m[1]) - 1.5).abs() < 1e-12);
            }
        }
    }

    /// A surface touching the axis, or a line lying along it, is refused.
    #[test]
    fn what_would_collapse_on_the_axis_is_refused() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[1.0, 0.0, 1.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        let err = revolve(&tri, PI, 2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap_err();
        assert!(
            format!("{err}").contains("only a line"),
            "unexpected: {err}"
        );

        let n3 = Node::create_in(coords.clone(), &[0.0, 0.0, 2.0]).unwrap();
        // One segment along the axis, one off it.
        let mut lines = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        lines.add_cell(&[n0.id(), n3.id()]).unwrap();
        lines.add_cell(&[n0.id(), n1.id()]).unwrap();
        let err = revolve(&lines, PI, 2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap_err();
        assert!(
            format!("{err}").contains("sweep nothing"),
            "unexpected: {err}"
        );
    }

    /// The fan and its ring extrude into prisms and hexahedra, sharing nodes.
    #[test]
    fn the_sector_extrudes_into_prisms_and_hexahedra() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let centre = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let rim = Node::create_in(coords, &[0.0, 0.0, -1.0]).unwrap();
        let seg = line(&centre, &rim, 2, ElementType::SEG2).unwrap();
        let sector = revolve(&seg, PI / 2.0, 7, &[0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0])).unwrap();
        let solid = crate::ops::mesh::extrude(&sector, &[0.0, 0.2, 0.0], 2).unwrap();
        let types: Vec<ElementType> = solid.into_iter().map(|z| z.read().element_type()).collect();
        assert_eq!(types, [ElementType::HEX8, ElementType::PENTA6]);
        let mut all: Vec<_> = Vec::new();
        for z in &solid {
            all.extend_from_slice(z.read().connectivity());
        }
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 17 * 3);
    }

    #[test]
    fn revolve_rejects_bad_arguments() {
        let seg = radial_seg2_2d(1);
        // Zero layers, zero angle, more than a full turn.
        assert!(revolve(&seg, PI, 0, &[0.0, 0.0], None).is_err());
        assert!(revolve(&seg, 0.0, 2, &[0.0, 0.0], None).is_err());
        assert!(revolve(&seg, 1.01 * TAU, 2, &[0.0, 0.0], None).is_err());
        // Centre of the wrong dimension.
        assert!(revolve(&seg, PI, 2, &[0.0, 0.0, 0.0], None).is_err());

        // 3-D without an axis.
        let tri = tri3_off_axis();
        assert!(revolve(&tri, PI, 2, &[0.0, 0.0, 0.0], None).is_err());
        assert!(revolve(&tri, PI, 2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 0.0])).is_err());
    }

    #[test]
    fn revolve_rejects_a_surface_in_2d() {
        // A TRI3 would sweep a PENTA6, which 2-D coordinates cannot hold.
        let coords = Handle::new(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        assert!(revolve(&tri, PI / 2.0, 2, &[0.0, 0.0], None).is_err());
    }

    #[test]
    fn revolve_rejects_unsupported_element_type() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut pts = Mesh::from_submesh(SubMesh::new(coords, ElementType::POI1));
        pts.add_cell(&[a.id()]).unwrap();
        assert!(revolve(&pts, PI / 2.0, 1, &[0.0, 0.0], None).is_err());
    }
}
