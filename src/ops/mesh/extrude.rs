use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Extrude a mesh by `n_layers` layers along `direction`.
///
/// `direction` is the **total** displacement vector; each intermediate
/// layer is placed at an evenly spaced fraction. Supported element types:
/// SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8, and their quadratic siblings
/// SEG3 → QUA8, TRI6 → PENTA15, QUA8 → HEX20: the source's mid-nodes are
/// copied on every layer, and each edge along the extrusion gets its
/// mid-node halfway. Other types produce an error.
///
/// Nodes shared between cells in the source mesh remain shared in the
/// extruded mesh. Source nodes are re-used (refcount incremented);
/// intermediate layer nodes are newly created.
///
/// Node ordering:
/// - QUA4: `bot[0], bot[1], top[1], top[0]`
/// - PENTA6: `bot[0..3], top[0..3]`
/// - HEX8: `bot[0..4], top[0..4]`
/// - QUA8, PENTA15, HEX20: the corners as above, then the mid-nodes in the
///   edge order of the linear cell — the bottom face's, the top face's, then
///   those of the edges along the extrusion.
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
/// // An extruded SEG2 becomes a QUA4: one layer, one cell.
/// let l = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[1.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let s = mesh::extrude(&l, &[0.0, 1.0, 0.0], 3)?;
/// assert_eq!(s.cell_count(), 3);
/// assert_eq!(s.element_types()?, vec![ElementType::QUA4]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Result<Mesh> {
    crate::ops::mesh::sweep_kernel::extrude(mesh, direction, n_layers)
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

    #[test]
    fn extrude_seg2_to_qua4() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let seg = line(&a, &b, 2, ElementType::SEG2).unwrap();

        let qua = extrude(&seg, &[0.0, 3.0], 3).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(qua.cell_count(), 6);

        let n = qua.node(0, 0, 0).unwrap();
        assert_eq!(n.position().unwrap(), vec![0.0, 0.0]);
        let n = qua.node(0, 0, 3).unwrap();
        assert!((n.position().unwrap()[0]).abs() < 1e-12);
        assert!((n.position().unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn extrude_seg2_shared_nodes_stay_shared() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let seg = line(&a, &b, 2, ElementType::SEG2).unwrap();

        let qua = extrude(&seg, &[0.0, 1.0], 1).unwrap();
        let mid_cell0 = qua.node(0, 0, 1).unwrap();
        let mid_cell1 = qua.node(0, 1, 0).unwrap();
        assert_eq!(mid_cell0.id(), mid_cell1.id());
    }

    #[test]
    fn extrude_qua4_to_hex8() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[1.0, 1.0, 0.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let mut qua_mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        qua_mesh
            .add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()])
            .unwrap();

        let hex = extrude(&qua_mesh, &[0.0, 0.0, 2.0], 1).unwrap();
        assert_eq!(hex.element_types().unwrap(), vec![ElementType::HEX8]);
        assert_eq!(hex.cell_count(), 1);

        assert_eq!(hex.node(0, 0, 0).unwrap().id(), n0.id());
        assert_eq!(hex.node(0, 0, 1).unwrap().id(), n1.id());
        assert_eq!(hex.node(0, 0, 2).unwrap().id(), n2.id());
        assert_eq!(hex.node(0, 0, 3).unwrap().id(), n3.id());
        let top0 = hex.node(0, 0, 4).unwrap();
        assert_eq!(top0.position().unwrap(), vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn extrude_tri3_to_penta6() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();

        let penta = extrude(&tri, &[0.0, 0.0, 2.0], 2).unwrap();
        assert_eq!(penta.element_types().unwrap(), vec![ElementType::PENTA6]);
        assert_eq!(penta.cell_count(), 2);

        // First layer: bottom triangle reuses the source nodes.
        assert_eq!(penta.node(0, 0, 0).unwrap().id(), n0.id());
        assert_eq!(penta.node(0, 0, 1).unwrap().id(), n1.id());
        assert_eq!(penta.node(0, 0, 2).unwrap().id(), n2.id());
        // Top of first layer sits one step (z = 1) above the base.
        let top0 = penta.node(0, 0, 3).unwrap();
        assert_eq!(top0.position().unwrap(), vec![0.0, 0.0, 1.0]);
    }

    /// Node positions of every cell of every zone, in local order.
    fn cell_positions(mesh: &Mesh) -> Vec<Vec<Vec<f64>>> {
        let mut out = Vec::new();
        for (z, sm) in mesh.into_iter().enumerate() {
            let (n_cells, npc) = {
                let s = sm.read();
                (s.cell_count(), s.element_type().nodes_per_cell())
            };
            for c in 0..n_cells {
                out.push(
                    (0..npc)
                        .map(|k| mesh.node(z, c, k).unwrap().position().unwrap())
                        .collect(),
                );
            }
        }
        out
    }

    #[test]
    fn quadratic_extrusion_matches_the_promoted_linear_one() {
        // On straight geometry, extruding the quadratic source and promoting
        // the extruded linear mesh give the same cells, node by node — SEG3 →
        // QUA8, TRI6 → PENTA15 and QUA8 → HEX20 alike.
        use crate::ops::mesh::to_quadratic;
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
        let n: Vec<Node> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.9, 0.0],
            [0.1, 1.0, 0.0],
        ]
        .iter()
        .map(|x| p(x))
        .collect();
        let seg = line(&n[0], &n[1], 2, ElementType::SEG2).unwrap();
        let mut qua = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        qua.add_cell(&[n[0].id(), n[1].id(), n[2].id(), n[3].id()])
            .unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();

        for (linear, expected) in [
            (seg, ElementType::QUA8),
            (tri, ElementType::PENTA15),
            (qua, ElementType::HEX20),
        ] {
            let direction = if expected == ElementType::QUA8 {
                [0.0, 1.5, 0.0]
            } else {
                [0.0, 0.0, 1.5]
            };
            let native = extrude(&to_quadratic(&linear).unwrap(), &direction, 3).unwrap();
            let promoted = to_quadratic(&extrude(&linear, &direction, 3).unwrap()).unwrap();
            assert_eq!(native.element_types().unwrap(), vec![expected]);
            let (a, b) = (cell_positions(&native), cell_positions(&promoted));
            assert_eq!(a.len(), b.len());
            for (ca, cb) in a.iter().zip(b.iter()) {
                for (x, y) in ca.iter().zip(cb.iter()) {
                    for d in 0..3 {
                        assert!((x[d] - y[d]).abs() < 1e-12, "{expected}: {x:?} vs {y:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn extrude_refuses_linear_and_quadratic_zones_together() {
        use crate::ops::mesh::to_quadratic;
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
        let seg2 = line(&p(&[0.0, 0.0]), &p(&[1.0, 0.0]), 1, ElementType::SEG2).unwrap();
        let seg3 =
            to_quadratic(&line(&p(&[2.0, 0.0]), &p(&[3.0, 0.0]), 1, ElementType::SEG2).unwrap())
                .unwrap();
        let mixed = seg2.union(&seg3).unwrap();
        let err = extrude(&mixed, &[0.0, 1.0], 1).unwrap_err();
        assert!(err.to_string().contains("linear and quadratic"));
    }

    #[test]
    fn extrude_rejects_zero_layers() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(extrude(&seg, &[0.0, 1.0], 0).is_err());
    }

    #[test]
    fn extrude_rejects_wrong_direction_dim() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(extrude(&seg, &[0.0, 1.0, 0.0], 1).is_err());
    }

    #[test]
    fn extrude_rejects_unsupported_element_type() {
        // POI1 has no reference frame to extrude into a higher-dimension cell.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut pts = Mesh::from_submesh(SubMesh::new(coords, ElementType::POI1));
        pts.add_cell(&[a.id()]).unwrap();
        assert!(extrude(&pts, &[0.0, 1.0], 1).is_err());
    }
}
