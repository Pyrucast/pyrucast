//! Element-type conversion: re-tile a mesh into a different element type
//! **without moving or adding geometry** on the existing corners.
//!
//! [`convert`] changes the element type of every submesh to `target`,
//! splitting each cell into cells of the target type. Only conversions that
//! keep the same corner nodes (no new node, no displacement) are supported:
//!
//! - **identity** — `target` equals the submesh type: the submesh is copied
//!   verbatim (same nodes, re-referenced);
//! - **`QUA4 → TRI3`** — each quad splits into two triangles along the
//!   `(0, 2)` diagonal: `(0, 1, 2)` and `(0, 2, 3)`;
//! - **`HEX8 → TET4`** — each hexahedron splits into six tetrahedra sharing
//!   the main diagonal `(0, 6)` (the Freudenthal/Kuhn subdivision), a
//!   space-filling split that stays conforming across shared faces.
//!
//! Promoting to a quadratic type (`TRI3 → TRI6`, …) creates mid-edge nodes
//! and is [`to_quadratic`](fn@crate::ops::mesh::to_quadratic)'s job, not
//! this operator's.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};

/// The six tetrahedra of the Freudenthal/Kuhn subdivision of a HEX8, each as
/// four local corner indices. All six share the main diagonal `(0, 6)`; the
/// split is consistent across cells (every hexahedron is cut the same way),
/// so faces shared by two hexahedra are cut along the same diagonal and the
/// result stays conforming. Local HEX8 order: bottom face `0-1-2-3` CCW, top
/// face `4-5-6-7` CCW (node `k+4` above node `k`).
const HEX8_TO_TET4: [[usize; 4]; 6] = [
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
];

/// The cells of the `target` split of one cell of `src`, as local
/// corner-index tuples — or `None` when `src == target` (handled as a copy).
///
/// Returns an error for any unsupported `(src, target)` pair.
fn split_of(src: ElementType, target: ElementType) -> Result<&'static [&'static [usize]]> {
    match (src, target) {
        (ElementType::QUA4, ElementType::TRI3) => Ok(&[&[0, 1, 2], &[0, 2, 3]]),
        (ElementType::HEX8, ElementType::TET4) => Ok(&[
            &HEX8_TO_TET4[0],
            &HEX8_TO_TET4[1],
            &HEX8_TO_TET4[2],
            &HEX8_TO_TET4[3],
            &HEX8_TO_TET4[4],
            &HEX8_TO_TET4[5],
        ]),
        (s, t) => Err(PyrucastError::Message(format!(
            "convert: no {s} → {t} conversion (supported: identity, QUA4 → TRI3, HEX8 → TET4)"
        ))),
    }
}

/// Convert every submesh of `mesh` to `target`, splitting each cell into cells
/// of the target type. Corner nodes are re-used (re-referenced), no node is
/// created or moved. Submeshes already of type `target` are copied verbatim.
///
/// The result mirrors `mesh` submesh by submesh (same order, same face
/// colours). Errors on any submesh whose type cannot be converted to `target`
/// (supported: identity, `QUA4 → TRI3`, `HEX8 → TET4`). `mesh` is left
/// untouched.
pub fn convert(mesh: &Mesh, target: ElementType) -> Result<Mesh> {
    let coords = mesh.coords()?;

    let mut result = Mesh::empty();
    for sm_h in mesh {
        let (src, color, conn) = {
            let s = read(sm_h)?;
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let mut new_sm = SubMesh::new(coords.clone(), target);
        new_sm.set_face_color(color);

        if src == target {
            // Identity: copy the connectivity verbatim.
            let npc = src.nodes_per_cell();
            for cell in conn.chunks(npc) {
                new_sm.add_cell(cell)?;
            }
        } else {
            let splits = split_of(src, target)?;
            let npc = src.nodes_per_cell();
            for cell in conn.chunks(npc) {
                for sub in splits {
                    let nodes: Vec<NodeId> = sub.iter().map(|&i| cell[i]).collect();
                    new_sm.add_cell(&nodes)?;
                }
            }
        }
        result.add_sub(insert(new_sm))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::store::insert;

    #[test]
    fn qua4_to_tri3_splits_each_quad_in_two() {
        let coords = insert(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        m.add_cell(&n).unwrap();

        let tri = convert(&m, ElementType::TRI3).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count().unwrap(), 2);
        // (0,1,2) and (0,2,3): corners re-used, no new node.
        assert_eq!(tri.node(0, 0, 0).unwrap().id(), n[0]);
        assert_eq!(tri.node(0, 0, 1).unwrap().id(), n[1]);
        assert_eq!(tri.node(0, 0, 2).unwrap().id(), n[2]);
        assert_eq!(tri.node(0, 1, 0).unwrap().id(), n[0]);
        assert_eq!(tri.node(0, 1, 1).unwrap().id(), n[2]);
        assert_eq!(tri.node(0, 1, 2).unwrap().id(), n[3]);
    }

    #[test]
    fn hex8_to_tet4_gives_six_tets_no_new_node() {
        let coords = insert(Coords::new(3).unwrap());
        let n: Vec<NodeId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        let tets = convert(&m, ElementType::TET4).unwrap();
        assert_eq!(tets.element_types().unwrap(), vec![ElementType::TET4]);
        assert_eq!(tets.cell_count().unwrap(), 6);

        // No node created: the coords still hold exactly the 8 corners.
        let live: usize = {
            let c = read(&coords).unwrap();
            n.iter().filter(|&&id| c.refcount(id) > 0).count()
        };
        assert_eq!(live, 8);

        // The six tets tile the unit cube: their volumes sum to 1.
        let vol = |ids: [NodeId; 4]| -> f64 {
            let c = read(&coords).unwrap();
            let p: Vec<Vec<f64>> = ids.iter().map(|&i| c.coord(i).unwrap().to_vec()).collect();
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let e3 = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
            let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
                - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
                + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
            det.abs() / 6.0
        };
        let total: f64 = (0..6)
            .map(|k| {
                vol([
                    tets.node(0, k, 0).unwrap().id(),
                    tets.node(0, k, 1).unwrap().id(),
                    tets.node(0, k, 2).unwrap().id(),
                    tets.node(0, k, 3).unwrap().id(),
                ])
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-12, "six tets must tile the cube");
    }

    #[test]
    fn identity_copies_verbatim() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let out = convert(&m, ElementType::TRI3).unwrap();
        assert_eq!(out.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(out.cell_count().unwrap(), 1);
        assert_eq!(out.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(out.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(out.node(0, 0, 2).unwrap().id(), c.id());
    }

    #[test]
    fn preserves_face_color() {
        let coords = insert(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        let color = crate::atoms::RgbColor::new(10, 20, 30);
        sm.set_face_color(color);
        let mut m = Mesh::from_submesh(sm);
        m.add_cell(&n).unwrap();

        let tri = convert(&m, ElementType::TRI3).unwrap();
        let out_color = read(&tri.get(0).unwrap()).unwrap().face_color();
        assert_eq!(out_color, color);
    }

    #[test]
    fn rejects_unsupported_conversion() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // TRI3 → QUA4 is not a corner-preserving split.
        assert!(convert(&m, ElementType::QUA4).is_err());
    }
}
