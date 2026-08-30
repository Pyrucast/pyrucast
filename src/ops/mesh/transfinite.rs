use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// Build a structured surface bounded by four `SEG2` sides, by (discrete)
/// transfinite interpolation — the Coons-patch generalization of
/// [`super::sweep()`] from two lines to four.
///
/// `side1`/`side3` and `side2`/`side4` are the two pairs of **opposite**
/// sides; each pair must have the same element count. The four sides must
/// form a closed contour, traversed consistently: the last node of
/// `side1` must equal the first node of `side2`, the last of `side2` the
/// first of `side3`, and so on around `side3 → side4 → side1`.
///
/// A `QUA4` mesh is always built first; `TRI3`/`QUA8`/`QUA9`/`TRI6` are
/// then derived from it, exactly as in [`super::sweep()`]. Every boundary
/// node (all four sides, corners included) is re-used (refcount
/// incremented); interior nodes are newly created by bilinear blending of
/// the four boundary curves, corrected against the four corners (discrete
/// Coons patch — exact on the boundary).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::ops::{element_field, matrix, mesh, scatter};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// // Quatre côtés **chaînés bout à bout** — la fin de l'un est le début du
/// // suivant — et opposés deux à deux de même découpage : la grille les
/// // interpole.
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// # let (a, b, c, d) = (p(&[0.0, 0.0]), p(&[2.0, 0.0]), p(&[2.0, 2.0]), p(&[0.0, 2.0]));
/// let bas = mesh::line(&a, &b, 2, ElementType::SEG2)?;
/// let droite = mesh::line(&b, &c, 3, ElementType::SEG2)?;
/// let haut = mesh::line(&c, &d, 2, ElementType::SEG2)?;
/// let gauche = mesh::line(&d, &a, 3, ElementType::SEG2)?;
/// let m = mesh::transfinite(&bas, &droite, &haut, &gauche, ElementType::QUA4)?;
/// assert_eq!(m.cell_count()?, 2 * 3);
/// // Deux côtés opposés de découpages différents : refusé, en le disant.
/// let trop = mesh::line(&c, &d, 5, ElementType::SEG2)?;
/// assert!(mesh::transfinite(&bas, &droite, &trop, &gauche, ElementType::QUA4).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn transfinite(
    side1: &Mesh,
    side2: &Mesh,
    side3: &Mesh,
    side4: &Mesh,
    element_type: ElementType,
) -> Result<Mesh> {
    let qua4 = transfinite_qua4(side1, side2, side3, side4)?;
    super::sweep::finish_surface(qua4, element_type, "transfinite")
}

fn transfinite_qua4(side1: &Mesh, side2: &Mesh, side3: &Mesh, side4: &Mesh) -> Result<Mesh> {
    let (coords, side1_ids, side1_coords) = side_columns(side1, "side1")?;
    let (coords2, side2_ids, side2_coords) = side_columns(side2, "side2")?;
    let (coords3, side3_ids, side3_coords) = side_columns(side3, "side3")?;
    let (coords4, side4_ids, side4_coords) = side_columns(side4, "side4")?;
    for (label, c) in [
        ("side2", &coords2),
        ("side3", &coords3),
        ("side4", &coords4),
    ] {
        if !coords.same_object(c) {
            return Err(PyrucastError::Message(format!(
                "transfinite: side1 and {label} belong to different Coords"
            )));
        }
    }

    let n1 = side1_ids.len() - 1;
    let n2 = side2_ids.len() - 1;
    if n1 == 0 || n2 == 0 {
        return Err(PyrucastError::Message(
            "transfinite: every side must have at least 1 element".into(),
        ));
    }
    if side3_ids.len() - 1 != n1 {
        return Err(PyrucastError::Message(format!(
            "transfinite: side1 has {n1} elements but side3 (opposite) has {}",
            side3_ids.len() - 1
        )));
    }
    if side4_ids.len() - 1 != n2 {
        return Err(PyrucastError::Message(format!(
            "transfinite: side2 has {n2} elements but side4 (opposite) has {}",
            side4_ids.len() - 1
        )));
    }

    if side1_ids[n1] != side2_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side1 must equal first node of side2".into(),
        ));
    }
    if side2_ids[n2] != side3_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side2 must equal first node of side3".into(),
        ));
    }
    if side3_ids[n1] != side4_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side3 must equal first node of side4".into(),
        ));
    }
    if side4_ids[n2] != side1_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side4 must equal first node of side1".into(),
        ));
    }

    // Corners, read off side1/side3 (u ranges over side1/side3, v over side2/side4).
    let p00 = &side1_coords[0];
    let p10 = &side1_coords[n1];
    let p01 = &side3_coords[n1];
    let p11 = &side3_coords[0];

    // Only the **interior** of the patch is new — its border is the four
    // sides' own nodes. The interior is laid out flat, row by row, and created
    // in one locked pass over the `Coords`.
    let mut flat: Vec<f64> = Vec::new();
    for i in 1..n1 {
        for j in 1..n2 {
            let u = i as f64 / n1 as f64;
            let v = j as f64 / n2 as f64;
            let c1 = &side1_coords[i];
            let c3 = &side3_coords[n1 - i];
            let c4 = &side4_coords[n2 - j];
            let c2 = &side2_coords[j];
            flat.extend((0..c1.len()).map(|d| {
                (1.0 - v) * c1[d] + v * c3[d] + (1.0 - u) * c4[d] + u * c2[d]
                    - ((1.0 - u) * (1.0 - v) * p00[d]
                        + u * (1.0 - v) * p10[d]
                        + (1.0 - u) * v * p01[d]
                        + u * v * p11[d])
            }));
        }
    }
    let first = coords.write().add_nodes(&flat)?.start;
    let inner_rows = n1.saturating_sub(1);
    let inner_cols = n2.saturating_sub(1);
    // Node at grid position (i, j) — the border tests come first, and in this
    // order, so a corner belongs to the side that owns it.
    let at = |i: usize, j: usize| -> NodeId {
        if j == 0 {
            side1_ids[i]
        } else if j == n2 {
            side3_ids[n1 - i]
        } else if i == 0 {
            side4_ids[n2 - j]
        } else if i == n1 {
            side2_ids[j]
        } else {
            NodeId(first + ((i - 1) * inner_cols + (j - 1)) as u32)
        }
    };

    let mut conn: Vec<NodeId> = Vec::with_capacity(n1 * n2 * 4);
    for i in 0..n1 {
        for j in 0..n2 {
            conn.extend_from_slice(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    let sm = SubMesh::from_connectivity(coords.clone(), ElementType::QUA4, conn)?;
    // The patch owns the fresh nodes now; hand back the unit `add_nodes`
    // handed us.
    let owned: Vec<NodeId> = (first..first + (inner_rows * inner_cols) as u32)
        .map(NodeId)
        .collect();
    coords.write().decref_all(&owned)?;
    Ok(Mesh::from_submesh(sm))
}

/// Ordered node ids and coordinates of a single-submesh `SEG2` line mesh,
/// column `j` = first node of elem 0 (j=0), or second node of elem j-1
/// (j≥1) — the same convention as [`super::sweep_kernel::qua4_between`].
type SideColumns = (Handle<Coords>, Vec<NodeId>, Vec<Vec<f64>>);

fn side_columns(mesh: &Mesh, label: &str) -> Result<SideColumns> {
    if mesh.len() != 1 {
        return Err(PyrucastError::Message(format!(
            "transfinite: {label} must have exactly one submesh"
        )));
    }
    let sm = mesh.get(0)?;
    let (coords, et, n_elems, conn) = {
        let s = sm.read();
        (
            s.coords(),
            s.element_type(),
            s.cell_count(),
            s.connectivity().to_vec(),
        )
    };
    if et != ElementType::SEG2 {
        return Err(PyrucastError::Message(format!(
            "transfinite: {label} must be a SEG2 mesh"
        )));
    }
    let ids: Vec<NodeId> = std::iter::once(conn[0])
        .chain((1..=n_elems).map(|j| conn[2 * j - 1]))
        .collect();
    let coord_vals: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    Ok((coords, ids, coord_vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::mesh::line::line;

    /// Builds the unit-square contour, corners shared between adjacent
    /// sides, `n1` elements on side1/side3 and `n2` on side2/side4.
    fn unit_square(n1: usize, n2: usize) -> (Mesh, Mesh, Mesh, Mesh) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let p0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let p1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let p2 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let p3 = Node::create_in(coords, &[0.0, 1.0]).unwrap();

        let side1 = line(&p0, &p1, n1, ElementType::SEG2).unwrap();
        let side2 = line(&p1, &p2, n2, ElementType::SEG2).unwrap();
        let side3 = line(&p2, &p3, n1, ElementType::SEG2).unwrap();
        let side4 = line(&p3, &p0, n2, ElementType::SEG2).unwrap();
        (side1, side2, side3, side4)
    }

    #[test]
    fn transfinite_unit_square_qua4_grid() {
        let (side1, side2, side3, side4) = unit_square(4, 3);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::QUA4).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(mesh.cell_count().unwrap(), 12);

        // Interior node (i=2, j=1) of a 4x3 grid (n1=4, n2=3) on the unit
        // square: exact bilinear position (0.5, 1/3), since all four sides
        // are straight. It's corner 0 of cell (i=2, j=1) = linear index
        // i*n2+j = 7.
        let n = mesh.node(0, 2 * 3 + 1, 0).unwrap();
        let c = n.position().unwrap();
        assert!((c[0] - 0.5).abs() < 1e-12, "x={}", c[0]);
        assert!((c[1] - 1.0 / 3.0).abs() < 1e-12, "y={}", c[1]);
    }

    #[test]
    fn transfinite_one_by_one_reuses_all_four_corners() {
        let (side1, side2, side3, side4) = unit_square(1, 1);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::QUA4).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);

        let p0 = side1.node(0, 0, 0).unwrap().id();
        let p1 = side1.node(0, 0, 1).unwrap().id();
        let p2 = side2.node(0, 0, 1).unwrap().id();
        let p3 = side3.node(0, 0, 1).unwrap().id();

        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), p0);
        assert_eq!(mesh.node(0, 0, 1).unwrap().id(), p1);
        assert_eq!(mesh.node(0, 0, 2).unwrap().id(), p2);
        assert_eq!(mesh.node(0, 0, 3).unwrap().id(), p3);
    }

    /// The node id at column `i` of a `SEG2` line mesh with `n` elements
    /// (node 0 = elem 0 corner 0, node k = elem k-1 corner 1), mirroring
    /// `side_columns`' own convention.
    fn column_id(mesh: &Mesh, i: usize) -> NodeId {
        if i == 0 {
            mesh.node(0, 0, 0).unwrap().id()
        } else {
            mesh.node(0, i - 1, 1).unwrap().id()
        }
    }

    #[test]
    fn transfinite_boundary_nodes_are_reused() {
        let (n1, n2) = (3, 2);
        let (side1, side2, side3, side4) = unit_square(n1, n2);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::QUA4).unwrap();

        for i in 0..n1 {
            // Bottom row (j=0): reuses side1 in order.
            assert_eq!(mesh.node(0, i * n2, 0).unwrap().id(), column_id(&side1, i));
            // Top row (j=n2): reuses side3, reversed.
            assert_eq!(
                mesh.node(0, i * n2 + (n2 - 1), 3).unwrap().id(),
                column_id(&side3, n1 - i)
            );
        }
        for j in 0..n2 {
            // Left column (i=0): reuses side4, reversed.
            assert_eq!(mesh.node(0, j, 0).unwrap().id(), column_id(&side4, n2 - j));
            // Right column (i=n1): reuses side2, in order.
            assert_eq!(
                mesh.node(0, (n1 - 1) * n2 + j, 1).unwrap().id(),
                column_id(&side2, j)
            );
        }
    }

    #[test]
    fn transfinite_tri3_splits_each_quad() {
        let (side1, side2, side3, side4) = unit_square(2, 2);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::TRI3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(mesh.cell_count().unwrap(), 8);
    }

    #[test]
    fn transfinite_qua8_promotes_to_quadratic() {
        let (side1, side2, side3, side4) = unit_square(2, 2);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::QUA8).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::QUA8]);
        assert_eq!(mesh.cell_count().unwrap(), 4);
    }

    #[test]
    fn transfinite_rejects_mismatched_opposite_side_counts() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let p0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let p1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let p2 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let p3 = Node::create_in(coords, &[0.0, 1.0]).unwrap();

        let side1 = line(&p0, &p1, 3, ElementType::SEG2).unwrap();
        let side2 = line(&p1, &p2, 2, ElementType::SEG2).unwrap();
        let side3 = line(&p2, &p3, 4, ElementType::SEG2).unwrap(); // ≠ side1
        let side4 = line(&p3, &p0, 2, ElementType::SEG2).unwrap();

        assert!(transfinite(&side1, &side2, &side3, &side4, ElementType::QUA4).is_err());
    }

    #[test]
    fn transfinite_rejects_open_contour() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let p0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let p1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let p2 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let p3 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        // Disconnected corner: side4 does not close back onto side1's start.
        let p3_bis = Node::create_in(coords, &[0.0, 1.0]).unwrap();

        let side1 = line(&p0, &p1, 2, ElementType::SEG2).unwrap();
        let side2 = line(&p1, &p2, 2, ElementType::SEG2).unwrap();
        let side3 = line(&p2, &p3, 2, ElementType::SEG2).unwrap();
        let side4 = line(&p3_bis, &p0, 2, ElementType::SEG2).unwrap();

        assert!(transfinite(&side1, &side2, &side3, &side4, ElementType::QUA4).is_err());
    }

    #[test]
    fn transfinite_rejects_unsupported_element_type() {
        let (side1, side2, side3, side4) = unit_square(1, 1);
        assert!(transfinite(&side1, &side2, &side3, &side4, ElementType::HEX8).is_err());
    }

    #[test]
    fn transfinite_rejects_non_seg2_side() {
        let (side1, side2, side3, side4) = unit_square(1, 1);
        let coords = side1.coords().unwrap();
        let a = Node::create_in(coords.clone(), &[5.0, 5.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[6.0, 5.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[5.5, 6.0]).unwrap();
        let mut tri_mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri_mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        assert!(transfinite(&tri_mesh, &side2, &side3, &side4, ElementType::QUA4).is_err());
    }
}
