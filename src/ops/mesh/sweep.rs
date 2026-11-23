use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::mesh::convert::convert_zone;
use crate::ops::mesh::sweep_kernel::{surface_between, swept_types, unsweepable, volume_between};

/// Sweep between two meshes into `n_layers` layers of cells of
/// `element_type`: a surface between two lines, a volume between two
/// surfaces.
///
/// Both meshes must be single-submesh meshes of the same type, with the same
/// number of cells, attached to the same `Coords`; `n_layers` must be ≥ 1.
/// The nodes of both meshes are re-used (refcount incremented); the layers
/// in between are interpolated linearly and created.
///
/// - **Two lines**: node `j` of `mesh_a` is linked to node `j` of `mesh_b`.
///   SEG2 lines give `QUA4`, `TRI3`, `QUA8`, `QUA9` or `TRI6`; SEG3 lines
///   give `QUA8`, `QUA9` or `TRI6`, their own mid-nodes re-used, so that a
///   strip between two arcs follows them. A `QUA4` or `QUA8` strip is built
///   first ([`surface_between`]), then delivered by [`finish_surface`]:
///   triangles alternate their diagonal from one cell to the next. **Two
///   closed loops** (each ending on its own first node) give a closed ring:
///   the seam is shared, not doubled.
/// - **Two surfaces**: cell `i` of `mesh_a` faces cell `i` of `mesh_b`, node
///   by node ([`volume_between`]). `TRI3`, `QUA4`, `TRI6` and `QUA8` faces
///   give `PENTA6`, `HEX8`, `PENTA15` and `HEX20`, the one `element_type`
///   they allow.
///
/// `element_type` is checked against the type of `mesh_a` before any node is
/// created ([`check_element_type`]).
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
/// // Two **conforming** lines stitched by QUA4: the mesh is
/// // not an extrusion, the two borders being allowed to differ.
/// let a = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[1.0, 0.0, 0.0]), 2, ElementType::SEG2)?;
/// let b = mesh::line(&p(&[0.0, 1.0, 0.0]), &p(&[2.0, 1.0, 0.0]), 2, ElementType::SEG2)?;
/// let s = mesh::sweep(&a, &b, 1, ElementType::QUA4)?;
/// assert_eq!(s.cell_count(), 2);
///
/// // Two surfaces stitched by hexahedra: the volume counterpart.
/// # let q = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// # let n: Vec<Node> = q.iter().map(|x| p(x)).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id(), n[3].id()])?;
/// # let bottom = Mesh::from_submesh(sm);
/// let top = mesh::translate(&bottom, &[0.0, 0.0, 1.0])?;
/// let v = mesh::sweep(&bottom, &top, 2, ElementType::HEX8)?;
/// assert_eq!(v.cell_count(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn sweep(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    n_layers: usize,
    element_type: ElementType,
) -> Result<Mesh> {
    let source = mesh_a.unit()?.read().element_type();
    check_element_type(&[source], element_type, "sweep")?;
    if source.topological_dim() == 2 {
        volume_between(mesh_a, mesh_b, n_layers)
    } else {
        finish_surface(
            surface_between(mesh_a, mesh_b, n_layers)?,
            element_type,
            "sweep",
        )
    }
}

/// Check, before any node is created, that every type in `sources` can give
/// `element_type` when swept ([`swept_types`]): the error names the source
/// type, the type asked and the types it gives. `op` names the operator.
pub(super) fn check_element_type(
    sources: &[ElementType],
    element_type: ElementType,
    op: &str,
) -> Result<()> {
    for &source in sources {
        let types = swept_types(source);
        if types.is_empty() {
            return Err(unsweepable(source, op));
        }
        if !types.contains(&element_type) {
            let names: Vec<String> = types.iter().map(ElementType::to_string).collect();
            let gives = match names.split_last() {
                Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
                _ => names.concat(),
            };
            return Err(PyrucastError::Message(format!(
                "{op}: a {source} zone cannot give {element_type} (a {source} zone gives {gives})"
            )));
        }
    }
    Ok(())
}

/// Deliver a `QUA4` or `QUA8` base surface as the surface type the caller
/// asked for, by the three steps the type calls for: cut the quadrangles
/// into triangles ([`convert_zone`], the diagonal alternating from one cell
/// to the next), promote to degree 2 ([`super::to_quadratic`], which shares
/// the mid-nodes between zones), give the quadrangles their centre
/// ([`convert_zone`]).
///
/// Every node a step adds sits on the geometry already built. From a `QUA4`
/// base the triangles are cut before being promoted, so the mid-node of a
/// diagonal is the middle of its chord and each triangle stays straight;
/// from a `QUA8` base the centre comes first, on the curved cell, and the
/// triangles take it as the mid-node of their diagonal. Zones of triangles —
/// the few triangles a paver leaves among its quadrangles — pass
/// through each step, promoted with the rest.
///
/// The structured meshers ([`sweep`], [`transfinite`](super::transfinite))
/// and the pavers ([`pave_surface`](super::pave_surface) and the grid meshers, which
/// ask only for quadrangles) deliver this way. `op` only names the operator
/// in the error message.
pub(super) fn finish_surface(base: Mesh, element_type: ElementType, op: &str) -> Result<Mesh> {
    use ElementType::*;
    let quadratic = base.element_types()?.contains(&QUA8);
    match (quadratic, element_type) {
        (false, QUA4) | (true, QUA8) => Ok(base),
        (false, TRI3) => quadrangles_to(&base, TRI3),
        (false, QUA8) => super::to_quadratic(&base),
        (false, QUA9) => quadrangles_to(&super::to_quadratic(&base)?, QUA9),
        (false, TRI6) => super::to_quadratic(&quadrangles_to(&base, TRI3)?),
        (true, QUA9) | (true, TRI6) => quadrangles_to(&base, element_type),
        (_, other) => Err(PyrucastError::Message(format!(
            "{op}: a {} surface cannot give {other} (expected {})",
            if quadratic { QUA8 } else { QUA4 },
            if quadratic {
                "QUA8, QUA9 or TRI6"
            } else {
                "QUA4, TRI3, QUA8, QUA9 or TRI6"
            }
        ))),
    }
}

/// `mesh` with its quadrangle zones converted to `target` by [`convert_zone`],
/// the diagonal of the triangles alternating from one cell to the next; the
/// other zones pass through as they are.
fn quadrangles_to(mesh: &Mesh, target: ElementType) -> Result<Mesh> {
    let mut result = Mesh::empty();
    for zone in mesh {
        let s = zone.read();
        if matches!(
            s.element_type(),
            ElementType::QUA4 | ElementType::QUA8 | ElementType::QUA9
        ) {
            result.add_sub(Handle::new(convert_zone(&s, target, true)?))?;
        } else {
            result.add_sub(zone.clone())?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::{Node, NodeId};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::mesh::line::line;

    /// A single-TRI3 mesh on `coords` from three coordinates, with its three
    /// corner nodes.
    fn tri(coords: &Handle<Coords>, pts: [[f64; 3]; 3]) -> (Mesh, [Node; 3]) {
        let n = pts.map(|p| Node::create_in(coords.clone(), &p).unwrap());
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        (m, n)
    }

    #[test]
    fn sweep_basic() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a2, 2, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b2, 2, ElementType::SEG2).unwrap();

        let qua = sweep(&mesh_a, &mesh_b, 2, ElementType::QUA4).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(qua.cell_count(), 4);

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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA4).unwrap();
        assert_eq!(qua.cell_count(), 1);

        assert_eq!(qua.node(0, 0, 0).unwrap().id(), a0.id());
        assert_eq!(qua.node(0, 0, 1).unwrap().id(), a1.id());
        assert_eq!(qua.node(0, 0, 2).unwrap().id(), b1.id());
        assert_eq!(qua.node(0, 0, 3).unwrap().id(), b0.id());
    }

    #[test]
    fn sweep_rejects_zero_layers() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let m = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(sweep(&m, &m, 0, ElementType::QUA4).is_err());
        // Between two surfaces as well.
        let coords = Handle::new(Coords::new(3).unwrap());
        let (t, _) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(sweep(&t, &t, 0, ElementType::PENTA6).is_err());
    }

    #[test]
    fn sweep_rejects_mismatched_elem_counts() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let m1 = line(&a, &b, 1, ElementType::SEG2).unwrap();
        let m2 = line(&a, &c, 2, ElementType::SEG2).unwrap();
        assert!(sweep(&m1, &m2, 1, ElementType::QUA4).is_err());
    }

    #[test]
    fn sweep_rejects_non_seg2() {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert!(sweep(&seg, &seg, 1, ElementType::HEX8).is_err());
    }

    #[test]
    fn sweep_tri3_splits_each_quad_in_two() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a2, 2, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b2, 2, ElementType::SEG2).unwrap();

        let tri = sweep(&mesh_a, &mesh_b, 2, ElementType::TRI3).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        // 4 QUA4 cells → 8 TRI3 cells.
        assert_eq!(tri.cell_count(), 8);
    }

    #[test]
    fn sweep_qua8_promotes_to_quadratic() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua8 = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA8).unwrap();
        assert_eq!(qua8.element_types().unwrap(), vec![ElementType::QUA8]);
        assert_eq!(qua8.cell_count(), 1);
    }

    #[test]
    fn sweep_qua9_has_center_node() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[2.0, 2.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let qua9 = sweep(&mesh_a, &mesh_b, 1, ElementType::QUA9).unwrap();
        assert_eq!(qua9.element_types().unwrap(), vec![ElementType::QUA9]);
        assert_eq!(qua9.cell_count(), 1);

        let center = qua9.node(0, 0, 8).unwrap();
        assert_eq!(center.position().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn qua9_keeps_the_triangles_of_a_quad_dominant_mesh() {
        // A quadrangle beside a triangle, as a paver leaves them: the QUA8
        // zone gains its centres, the TRI6 one passes through as it is.
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut quads = SubMesh::new(coords.clone(), ElementType::QUA4);
        quads
            .add_cell(&[n[0].id(), n[1].id(), n[2].id(), n[3].id()])
            .unwrap();
        let mut tris = SubMesh::new(coords.clone(), ElementType::TRI3);
        tris.add_cell(&[n[1].id(), n[4].id(), n[2].id()]).unwrap();
        let mut mixed = Mesh::empty();
        mixed.add_sub(Handle::new(quads)).unwrap();
        mixed.add_sub(Handle::new(tris)).unwrap();

        let qua9 = finish_surface(mixed, ElementType::QUA9, "test").unwrap();
        assert_eq!(
            qua9.element_types().unwrap(),
            vec![ElementType::QUA9, ElementType::TRI6]
        );
        assert_eq!(qua9.cell_count(), 2);
        assert_eq!(
            qua9.node(0, 0, 8).unwrap().position().unwrap(),
            vec![0.5, 0.5]
        );
    }

    /// A closed square loop of four SEG2, corners at `±h`.
    fn square_loop(coords: &Handle<Coords>, h: f64) -> Mesh {
        let n: Vec<Node> = [[-h, -h], [h, -h], [h, h], [-h, h]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        for i in 0..4 {
            sm.add_cell(&[n[i].id(), n[(i + 1) % 4].id()]).unwrap();
        }
        Mesh::from_submesh(sm)
    }

    /// The distinct node ids a one-zone mesh references.
    fn distinct_nodes(mesh: &Mesh) -> std::collections::HashSet<NodeId> {
        mesh.get(0)
            .unwrap()
            .read()
            .connectivity()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn sweep_two_closed_loops_close_the_ring() {
        // 4 outer + 4 inner + 2 intermediate rings of 4: 16 nodes. A doubled
        // seam would add one node per intermediate ring, 18 in all.
        let coords = Handle::new(Coords::new(2).unwrap());
        let outer = square_loop(&coords, 2.0);
        let inner = square_loop(&coords, 1.0);
        let ring = sweep(&outer, &inner, 3, ElementType::QUA4).unwrap();
        assert_eq!(ring.cell_count(), 12);
        assert_eq!(distinct_nodes(&ring).len(), 16);
        // Last column of every layer = first column: cell 3 of a layer ends
        // where cell 0 starts.
        for k in 0..3 {
            assert_eq!(
                ring.node(0, 4 * k + 3, 1).unwrap().id(),
                ring.node(0, 4 * k, 0).unwrap().id()
            );
        }
    }

    #[test]
    fn sweep_refuses_a_closed_loop_onto_an_open_line() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let closed = square_loop(&coords, 2.0);
        let a = Node::create_in(coords.clone(), &[0.0, 5.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[4.0, 5.0]).unwrap();
        let open = line(&a, &b, 4, ElementType::SEG2).unwrap();
        let err = sweep(&closed, &open, 2, ElementType::QUA4).unwrap_err();
        assert!(err
            .to_string()
            .contains("mesh_a is a closed loop but mesh_b is open"));
    }

    /// Two parallel straight lines, `n` cells each, of the given degree.
    fn two_lines(coords: &Handle<Coords>, n: usize, et: ElementType) -> (Mesh, Mesh) {
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap();
        (
            line(&p(0.0, 0.0), &p(2.0, 0.5), n, et).unwrap(),
            line(&p(0.0, 1.0), &p(2.0, 1.5), n, et).unwrap(),
        )
    }

    /// Node positions of every cell, in local order.
    fn cell_positions(mesh: &Mesh) -> Vec<Vec<Vec<f64>>> {
        let npc = mesh.get(0).unwrap().read().element_type().nodes_per_cell();
        (0..mesh.cell_count())
            .map(|c| {
                (0..npc)
                    .map(|k| mesh.node(0, c, k).unwrap().position().unwrap())
                    .collect()
            })
            .collect()
    }

    fn assert_same_cells(a: &Mesh, b: &Mesh) {
        let (pa, pb) = (cell_positions(a), cell_positions(b));
        assert_eq!(pa.len(), pb.len());
        for (c, (ca, cb)) in pa.iter().zip(pb.iter()).enumerate() {
            assert_eq!(ca.len(), cb.len());
            for (k, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                for d in 0..x.len() {
                    assert!(
                        (x[d] - y[d]).abs() < 1e-12,
                        "cell {c}, node {k}: {x:?} vs {y:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn sweep_seg3_lines_match_the_promoted_strip_on_straight_lines() {
        // Between straight lines, the native quadratic strip and the QUA4
        // strip promoted by `to_quadratic` are the same mesh, node by node.
        let coords = Handle::new(Coords::new(2).unwrap());
        let (a2, b2) = two_lines(&coords, 3, ElementType::SEG2);
        let (a3, b3) = two_lines(&coords, 3, ElementType::SEG3);
        for et in [ElementType::QUA8, ElementType::QUA9] {
            let promoted = sweep(&a2, &b2, 2, et).unwrap();
            let native = sweep(&a3, &b3, 2, et).unwrap();
            assert_eq!(native.element_types().unwrap(), vec![et]);
            assert_same_cells(&native, &promoted);
        }
    }

    #[test]
    fn sweep_seg3_between_arcs_follows_them() {
        // A quarter ring between radii 1 and 2: the mid-nodes of the first
        // and last rows are those of the arcs, on the circles.
        use crate::ops::mesh::arc::arc;
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap();
        let o = p(0.0, 0.0);
        let inner = arc(&p(1.0, 0.0), &o, &p(0.0, 1.0), 4, ElementType::SEG3).unwrap();
        let outer = arc(&p(2.0, 0.0), &o, &p(0.0, 2.0), 4, ElementType::SEG3).unwrap();
        let ring = sweep(&inner, &outer, 2, ElementType::QUA8).unwrap();
        assert_eq!(ring.cell_count(), 8);
        let radius = |c: usize, k: usize| {
            let x = ring.node(0, c, k).unwrap().position().unwrap();
            (x[0] * x[0] + x[1] * x[1]).sqrt()
        };
        for j in 0..4 {
            assert!((radius(j, 4) - 1.0).abs() < 1e-12); // first row, on the inner arc
            assert!((radius(4 + j, 6) - 2.0).abs() < 1e-12); // last row, on the outer arc
        }
    }

    #[test]
    fn sweep_seg3_refuses_linear_targets_and_mixed_degrees() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let (a2, _) = two_lines(&coords, 2, ElementType::SEG2);
        let (a3, b3) = two_lines(&coords, 2, ElementType::SEG3);
        // No demotion, and the refusal comes before any node is created.
        let before = coords.read().node_count();
        assert!(sweep(&a3, &b3, 1, ElementType::QUA4).is_err());
        assert!(sweep(&a3, &b3, 1, ElementType::TRI3).is_err());
        assert_eq!(coords.read().node_count(), before);
        assert!(sweep(&a3, &a2, 1, ElementType::QUA8).is_err());
        assert!(sweep(&a2, &a3, 1, ElementType::QUA8).is_err());
        // Triangles, quadratic: the QUA9 cut through its centre.
        let tri6 = sweep(&a3, &b3, 1, ElementType::TRI6).unwrap();
        assert_eq!(tri6.element_types().unwrap(), vec![ElementType::TRI6]);
        assert_eq!(tri6.cell_count(), 4);
    }

    #[test]
    fn sweep_two_closed_seg3_loops_close_the_ring() {
        // Two concentric SEG3 circles of 6 cells, 2 layers: the seam is
        // shared on every level, mid-levels included.
        use crate::ops::mesh::circle::circle;
        let coords = Handle::new(Coords::new(2).unwrap());
        let o = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let inner = circle(&o, &[0.0, 0.0, 1.0], 1.0, 6, ElementType::SEG3).unwrap();
        let outer = circle(&o, &[0.0, 0.0, 1.0], 2.0, 6, ElementType::SEG3).unwrap();
        let ring = sweep(&inner, &outer, 2, ElementType::QUA8).unwrap();
        // Three corner levels of 6 corners + 6 mid-nodes, two mid-levels of 6
        // nodes: 36 + 12 = 48. A doubled seam would add one per level.
        assert_eq!(distinct_nodes(&ring).len(), 48);
    }

    #[test]
    fn sweep_tri6_splits_then_promotes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line(&a0, &a1, 1, ElementType::SEG2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line(&b0, &b1, 1, ElementType::SEG2).unwrap();

        let tri6 = sweep(&mesh_a, &mesh_b, 1, ElementType::TRI6).unwrap();
        assert_eq!(tri6.element_types().unwrap(), vec![ElementType::TRI6]);
        assert_eq!(tri6.cell_count(), 2);
    }

    #[test]
    fn sweep_of_two_triangles_gives_prisms() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let (a, na) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let (b, nb) = tri(&coords, [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]);

        let solid = sweep(&a, &b, 2, ElementType::PENTA6).unwrap();
        assert_eq!(solid.element_types().unwrap(), vec![ElementType::PENTA6]);
        assert_eq!(solid.cell_count(), 2);

        // Bottom of the first prism reuses mesh_a nodes.
        assert_eq!(solid.node(0, 0, 0).unwrap().id(), na[0].id());
        assert_eq!(solid.node(0, 0, 1).unwrap().id(), na[1].id());
        assert_eq!(solid.node(0, 0, 2).unwrap().id(), na[2].id());
        // Top of the last prism reuses mesh_b nodes.
        assert_eq!(solid.node(0, 1, 3).unwrap().id(), nb[0].id());
        assert_eq!(solid.node(0, 1, 4).unwrap().id(), nb[1].id());
        assert_eq!(solid.node(0, 1, 5).unwrap().id(), nb[2].id());
        // Interpolated interior layer sits at z = 1.
        let mid = solid.node(0, 0, 3).unwrap();
        assert!((mid.position().unwrap()[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sweep_of_surfaces_keeps_shared_nodes_shared() {
        // Two triangles sharing an edge → two prisms sharing a quad face.
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap().id();
        let (a0, a1, a2, a3) = (
            p(&[0.0, 0.0, 0.0]),
            p(&[1.0, 0.0, 0.0]),
            p(&[0.0, 1.0, 0.0]),
            p(&[1.0, 1.0, 0.0]),
        );
        let mut a = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        a.add_cell(&[a0, a1, a2]).unwrap();
        a.add_cell(&[a1, a3, a2]).unwrap();
        let (b0, b1, b2, b3) = (
            p(&[0.0, 0.0, 1.0]),
            p(&[1.0, 0.0, 1.0]),
            p(&[0.0, 1.0, 1.0]),
            p(&[1.0, 1.0, 1.0]),
        );
        let mut b = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        b.add_cell(&[b0, b1, b2]).unwrap();
        b.add_cell(&[b1, b3, b2]).unwrap();

        let solid = sweep(&a, &b, 3, ElementType::PENTA6).unwrap();
        assert_eq!(solid.cell_count(), 6);
        // Node a1 is shared by both source triangles; both prisms of the
        // first layer must reference the same node for it.
        assert_eq!(
            solid.node(0, 0, 1).unwrap().id(), // cell 0, local 1 = a1
            solid.node(0, 1, 0).unwrap().id(), // cell 1, local 0 = a1
        );
    }

    #[test]
    fn sweep_rejects_surfaces_of_different_types() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let (a, _) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap().id();
        let q = [
            p(&[0.0, 0.0, 1.0]),
            p(&[1.0, 0.0, 1.0]),
            p(&[1.0, 1.0, 1.0]),
            p(&[0.0, 1.0, 1.0]),
        ];
        let mut quad = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        quad.add_cell(&q).unwrap();
        assert!(sweep(&a, &quad, 1, ElementType::PENTA6).is_err());
        // A surface gives its one volume, nothing else.
        assert!(sweep(&a, &a, 1, ElementType::HEX8).is_err());
    }

    #[test]
    fn sweep_of_two_surfaces_matches_extrude() {
        // A face and its translated copy, swept, give the cells the face
        // extruded gives: the same patterns, the same half levels.
        use crate::ops::mesh::{extrude, to_quadratic, translate};
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap().id();
        let n = [
            p(&[0.0, 0.0, 0.0]),
            p(&[1.0, 0.0, 0.0]),
            p(&[2.0, 0.2, 0.0]),
            p(&[0.1, 1.0, 0.0]),
            p(&[1.1, 1.1, 0.0]),
            p(&[2.1, 1.3, 0.0]),
        ];
        // Counter-clockwise seen from +z: the extrusion along +z keeps the
        // cells as they are.
        let mut quads = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        quads.add_cell(&[n[0], n[1], n[4], n[3]]).unwrap();
        quads.add_cell(&[n[1], n[2], n[5], n[4]]).unwrap();
        let mut tris = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tris.add_cell(&[n[0], n[1], n[4]]).unwrap();
        tris.add_cell(&[n[0], n[4], n[3]]).unwrap();
        let d = [0.0, 0.0, 1.5];
        for linear in [tris, quads] {
            let quadratic = to_quadratic(&linear).unwrap();
            for face in [&linear, &quadratic] {
                let source = face.unit().unwrap().read().element_type();
                let solid = swept_types(source)[0];
                let swept = sweep(face, &translate(face, &d).unwrap(), 3, solid).unwrap();
                let extruded = extrude(face, &d, 3).unwrap();
                assert_eq!(swept.element_types().unwrap(), vec![solid]);
                assert_same_cells(&swept, &extruded);
            }
        }
    }
}
