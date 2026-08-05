use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};

/// Sweep two SEG2 meshes into a mesh of `element_type`, building `n_layers`
/// layers. `element_type` is one of `QUA4` (default), `TRI3`, `QUA8`,
/// `QUA9`, `TRI6`.
///
/// Both meshes must be single-submesh SEG2 meshes with the same number of
/// elements, attached to the same `Coords`. `n_layers` must be ≥ 1.
///
/// Column `j` of `mesh_a` is linearly interpolated with column `j` of
/// `mesh_b` to produce the intermediate layers. Endpoint nodes from both
/// meshes are re-used (refcount incremented); intermediate nodes are
/// created at evenly spaced positions.
///
/// A `QUA4` mesh is always built first; `TRI3`/`QUA8`/`QUA9`/`TRI6` are then
/// derived from it (diagonal split for triangles, [`super::to_quadratic`]
/// for the quadratic siblings, and a fresh center node per cell for `QUA9`).
///
/// QUA4 node order per element (counterclockwise, `mesh_a` side first):
/// `[k][j]`, `[k][j+1]`, `[k+1][j+1]`, `[k+1][j]`.
pub fn sweep(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    n_layers: usize,
    element_type: ElementType,
) -> Result<Mesh> {
    let qua4 = crate::ops::mesh::sweep_kernel::qua4_between(mesh_a, mesh_b, n_layers)?;
    finish_surface(qua4, element_type, "sweep")
}

/// Deliver a `QUA4` grid as the surface type the caller asked for.
///
/// The three structured surface meshers — [`sweep`],
/// [`transfinite`](super::transfinite), [`pave_surface`](super::pave_surface) —
/// all build a quadrangle grid first and then owe the same five conversions.
/// `op` only names the operator in the error message.
pub(super) fn finish_surface(qua4: Mesh, element_type: ElementType, op: &str) -> Result<Mesh> {
    match element_type {
        ElementType::QUA4 => Ok(qua4),
        ElementType::QUA8 => super::to_quadratic(&qua4),
        ElementType::QUA9 => qua8_to_qua9(&super::to_quadratic(&qua4)?),
        ElementType::TRI3 => qua4_to_tri3(&qua4),
        ElementType::TRI6 => super::to_quadratic(&qua4_to_tri3(&qua4)?),
        other => Err(PyrucastError::Message(format!(
            "{op}: unsupported element type {other} (expected QUA4, TRI3, QUA8, QUA9, TRI6)"
        ))),
    }
}

/// Split each QUA4 cell into two TRI3 cells along the `(0, 2)` diagonal:
/// `(0, 1, 2)` and `(0, 2, 3)`. Nodes are re-used, no new node is created.
///
/// Shared with [`super::transfinite`], which derives `TRI3`/`TRI6` from a
/// QUA4 grid the same way [`sweep`] does.
pub(super) fn qua4_to_tri3(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();
    for sm_h in mesh {
        let (color, conn) = {
            let s = read(sm_h)?;
            (s.face_color(), s.connectivity().to_vec())
        };
        let mut new_sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        new_sm.set_face_color(color);
        for cell in conn.chunks(4) {
            new_sm.add_cell(&[cell[0], cell[1], cell[2]])?;
            new_sm.add_cell(&[cell[0], cell[2], cell[3]])?;
        }
        result.add_sub(insert(new_sm))?;
    }
    Ok(result)
}

/// Promote each QUA8 cell to QUA9 by adding a fresh center node (node 8, the
/// mean of the 4 corners) — [`super::to_quadratic`] only produces the
/// serendipity QUA8, which has no center node to re-use.
///
/// Shared with [`super::transfinite`] (see [`qua4_to_tri3`]).
pub(super) fn qua8_to_qua9(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();
    for sm_h in mesh {
        let (color, conn) = {
            let s = read(sm_h)?;
            (s.face_color(), s.connectivity().to_vec())
        };
        let mut new_sm = SubMesh::new(coords.clone(), ElementType::QUA9);
        new_sm.set_face_color(color);
        for cell in conn.chunks(8) {
            let center: Vec<f64> = {
                let c = read(&coords)?;
                let corners: Vec<Vec<f64>> = cell[..4]
                    .iter()
                    .map(|&id| -> Result<Vec<f64>> { Ok(c.position(id)?.to_vec()) })
                    .collect::<Result<_>>()?;
                let dim = corners[0].len();
                (0..dim)
                    .map(|d| corners.iter().map(|p| p[d]).sum::<f64>() / 4.0)
                    .collect()
            };
            let center_node = Node::create_in(coords.clone(), &center)?;
            let mut nodes: Vec<NodeId> = cell.to_vec();
            nodes.push(center_node.id());
            new_sm.add_cell(&nodes)?;
        }
        result.add_sub(insert(new_sm))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::ops::mesh::line::line;
    use crate::store::insert;

    #[test]
    fn sweep_basic() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a2, 2, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b2, 2, ElementType::SEG2).unwrap();

        let qua = sweep(&mesh_a, &mesh_b, 2, ElementType::QUA4).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(qua.cell_count().unwrap(), 4);

        let n00 = qua.node(0, 0, 0).unwrap();
        assert_eq!(n00.position().unwrap(), vec![0.0, 0.0]);
        let n01 = qua.node(0, 0, 1).unwrap();
        assert!((n01.position().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n01.position().unwrap()[1]).abs() < 1e-12);
        let n02 = qua.node(0, 0, 2).unwrap();
        assert!((n02.position().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n02.position().unwrap()[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sweep_one_layer_reuses_endpoints() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA4).unwrap();
        assert_eq!(qua.cell_count().unwrap(), 1);

        assert_eq!(qua.node(0, 0, 0).unwrap().id(), a0.id());
        assert_eq!(qua.node(0, 0, 1).unwrap().id(), a1.id());
        assert_eq!(qua.node(0, 0, 2).unwrap().id(), b1.id());
        assert_eq!(qua.node(0, 0, 3).unwrap().id(), b0.id());
    }

    #[test]
    fn sweep_rejects_zero_layers() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let m = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(sweep(&m, &m, 0, ElementType::QUA4).is_err());
    }

    #[test]
    fn sweep_rejects_mismatched_elem_counts() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let m1 = line(&a, &b, 1, ElementType::SEG2).unwrap();
        let m2 = line(&a, &c, 2, ElementType::SEG2).unwrap();
        assert!(sweep(&m1, &m2, 1, ElementType::QUA4).is_err());
    }

    #[test]
    fn sweep_rejects_non_seg2() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();

        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut tri_mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri_mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        assert!(sweep(&tri_mesh, &seg, 1, ElementType::QUA4).is_err());
        assert!(sweep(&seg, &tri_mesh, 1, ElementType::QUA4).is_err());
    }

    #[test]
    fn sweep_rejects_unsupported_element_type() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(sweep(&seg, &seg, 1, ElementType::HEX8).is_err());
    }

    #[test]
    fn sweep_tri3_splits_each_quad_in_two() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a2, 2, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b2, 2, ElementType::SEG2).unwrap();

        let tri = sweep(&mesh_a, &mesh_b, 2, ElementType::TRI3).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        // 4 QUA4 cells → 8 TRI3 cells.
        assert_eq!(tri.cell_count().unwrap(), 8);
    }

    #[test]
    fn sweep_qua8_promotes_to_quadratic() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua8 = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA8).unwrap();
        assert_eq!(qua8.element_types().unwrap(), vec![ElementType::QUA8]);
        assert_eq!(qua8.cell_count().unwrap(), 1);
    }

    #[test]
    fn sweep_qua9_has_center_node() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[2.0, 2.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua9 = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA9).unwrap();
        assert_eq!(qua9.element_types().unwrap(), vec![ElementType::QUA9]);
        assert_eq!(qua9.cell_count().unwrap(), 1);

        let center = qua9.node(0, 0, 8).unwrap();
        assert_eq!(center.position().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn sweep_tri6_splits_then_promotes() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let tri6 = sweep(&mesh_a, &mesh_b, 1, ElementType::TRI6).unwrap();
        assert_eq!(tri6.element_types().unwrap(), vec![ElementType::TRI6]);
        assert_eq!(tri6.cell_count().unwrap(), 2);
    }
}
