use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Revolve a mesh by `n_layers` layers over a total `angle` (radians) — the
/// rotational companion of [`extrude`](fn@super::extrude).
///
/// Each node is swept along the circle it describes about the rotation centre
/// (2-D) or axis (3-D), and consecutive angular positions are linked by one
/// layer of cells: SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8. Other element
/// types produce an error.
///
/// - **2-D** (`center` has 2 components): revolution about the point
///   `center`, counterclockwise for a positive `angle`; `axis` is ignored.
/// - **3-D** (`center` has 3 components): revolution about the line through
///   `center` directed by `axis` (right-handed); `axis` is required and need
///   not be normalized.
///
/// `|angle|` may not exceed a full turn; a full turn **closes** the ring —
/// the last node layer is the first one again, so there is no seam and no
/// duplicated node. No node may lie on the axis (it would collapse the cells
/// touching it).
///
/// Nodes shared between cells of the source stay shared in the result. Source
/// nodes are re-used (refcount incremented); the other layers are created.
///
/// Node ordering:
/// - QUA4: `bot[0], bot[1], top[1], top[0]`
/// - PENTA6: `bot[0..3], top[0..3]`
/// - HEX8: `bot[0..4], top[0..4]`
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

    #[test]
    fn revolve_rejects_nodes_on_the_axis() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords, &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(revolve(&seg, PI, 2, &[0.0, 0.0], None).is_err());
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
