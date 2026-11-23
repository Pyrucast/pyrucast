use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::mesh::sweep::{check_element_type, finish_surface};
use crate::ops::mesh::sweep_kernel::line_row;

/// Build a structured surface bounded by four sides — four `SEG2` lines or
/// four `SEG3` lines — by (discrete) transfinite interpolation, the
/// generalization of [`super::sweep()`] from two lines to four.
///
/// `side1`/`side3` and `side2`/`side4` are the two pairs of **opposite**
/// sides; each pair must have the same element count. The four sides must
/// form a closed contour, traversed consistently: the last node of
/// `side1` must equal the first node of `side2`, the last of `side2` the
/// first of `side3`, and so on around `side3 → side4 → side1`.
///
/// Every node of the four sides (corners included, and the mid-nodes of
/// `SEG3` sides) is re-used (refcount incremented). The interior corners are
/// newly created by the discrete Coons patch: the blend of the two opposite
/// sides in each direction, corrected by the four corners, which reproduces
/// the sides exactly on the border whatever their shape. It reads the corners
/// of the sides only.
///
/// - **`SEG2` sides** give `QUA4`, `TRI3`, `QUA8`, `QUA9` or `TRI6`: a `QUA4`
///   grid is built first, then delivered as by [`super::sweep()`].
/// - **`SEG3` sides** give `QUA8`, `QUA9` or `TRI6`: a `QUA8` grid is built
///   first, its border mid-nodes those of the sides, so that a patch
///   bounded by arcs follows them; an interior mid-node sits at the middle of
///   the chord between its two corners. The grid is then delivered as by
///   [`super::sweep()`]: the centre of a `QUA9` cell, or the mid-node of the
///   diagonal of its triangles, is the point of the `QUA8` at its middle.
///
/// Four sides of different degrees, or an `element_type` the sides do not
/// give, are refused before any node is created.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
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
/// # let modele = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// // Four sides **chained end to end** — the end of one is the start of the
/// // next — and opposite in pairs with the same split: the grid
/// // interpolates them.
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// # let (a, b, c, d) = (p(&[0.0, 0.0]), p(&[2.0, 0.0]), p(&[2.0, 2.0]), p(&[0.0, 2.0]));
/// let bottom = mesh::line(&a, &b, 2, ElementType::SEG2)?;
/// let right = mesh::line(&b, &c, 3, ElementType::SEG2)?;
/// let top = mesh::line(&c, &d, 2, ElementType::SEG2)?;
/// let left = mesh::line(&d, &a, 3, ElementType::SEG2)?;
/// let m = mesh::transfinite(&bottom, &right, &top, &left, ElementType::QUA4)?;
/// assert_eq!(m.cell_count(), 2 * 3);
/// // Two opposite sides split differently: refused, saying so.
/// let too_many = mesh::line(&c, &d, 5, ElementType::SEG2)?;
/// assert!(mesh::transfinite(&bottom, &right, &too_many, &left, ElementType::QUA4).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn transfinite(
    side1: &Mesh,
    side2: &Mesh,
    side3: &Mesh,
    side4: &Mesh,
    element_type: ElementType,
) -> Result<Mesh> {
    // The type check needs a line side; a side that is not one line is
    // refused by `side_columns`, in its own words.
    if let [line @ (ElementType::SEG2 | ElementType::SEG3)] = side1.element_types()?[..] {
        check_element_type(&[line], element_type, "transfinite")?;
    }
    let base = transfinite_surface(side1, side2, side3, side4)?;
    finish_surface(base, element_type, "transfinite")
}

/// The base surface of [`transfinite`]: a `QUA4` grid between four `SEG2`
/// sides, a `QUA8` grid between four `SEG3` sides.
///
/// The grid is read at the fine level: with `SEG3` sides, the corners sit at
/// the even indices of the sides' rows and the mid-nodes at the odd ones;
/// with `SEG2` sides every index is a corner.
fn transfinite_surface(side1: &Mesh, side2: &Mesh, side3: &Mesh, side4: &Mesh) -> Result<Mesh> {
    let (coords, et1, side1_ids, side1_coords) = side_columns(side1, "side1")?;
    let (coords2, et2, side2_ids, side2_coords) = side_columns(side2, "side2")?;
    let (coords3, et3, side3_ids, side3_coords) = side_columns(side3, "side3")?;
    let (coords4, et4, side4_ids, side4_coords) = side_columns(side4, "side4")?;
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
    for (label, et) in [("side2", et2), ("side3", et3), ("side4", et4)] {
        if et != et1 {
            return Err(PyrucastError::Message(format!(
                "transfinite: side1 is {et1} but {label} is {et} — the four sides must be of \
                 the same degree"
            )));
        }
    }
    let quadratic = et1 == ElementType::SEG3;
    // The distance between two corners in a side's row.
    let step = if quadratic { 2 } else { 1 };
    let (last1, last2) = (side1_ids.len() - 1, side2_ids.len() - 1);
    let (n1, n2) = (last1 / step, last2 / step);
    if n1 == 0 || n2 == 0 {
        return Err(PyrucastError::Message(
            "transfinite: every side must have at least 1 element".into(),
        ));
    }
    if side3_ids.len() - 1 != last1 {
        return Err(PyrucastError::Message(format!(
            "transfinite: side1 has {n1} elements but side3 (opposite) has {}",
            (side3_ids.len() - 1) / step
        )));
    }
    if side4_ids.len() - 1 != last2 {
        return Err(PyrucastError::Message(format!(
            "transfinite: side2 has {n2} elements but side4 (opposite) has {}",
            (side4_ids.len() - 1) / step
        )));
    }

    if side1_ids[last1] != side2_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side1 must equal first node of side2".into(),
        ));
    }
    if side2_ids[last2] != side3_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side2 must equal first node of side3".into(),
        ));
    }
    if side3_ids[last1] != side4_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side3 must equal first node of side4".into(),
        ));
    }
    if side4_ids[last2] != side1_ids[0] {
        return Err(PyrucastError::Message(
            "transfinite: last node of side4 must equal first node of side1".into(),
        ));
    }

    // Corners, read off side1/side3 (u ranges over side1/side3, v over side2/side4).
    let p00 = &side1_coords[0];
    let p10 = &side1_coords[last1];
    let p01 = &side3_coords[last1];
    let p11 = &side3_coords[0];
    let dim = p00.len();

    // The position of the grid point (i, j): on the border the side's own
    // node, inside the Coons patch. At a corner index it reads the corners of
    // the sides only.
    let corner = |i: usize, j: usize| -> [f64; 3] {
        let mut x = [0.0; 3];
        let border = if j == 0 {
            Some(&side1_coords[i])
        } else if j == last2 {
            Some(&side3_coords[last1 - i])
        } else if i == 0 {
            Some(&side4_coords[last2 - j])
        } else if i == last1 {
            Some(&side2_coords[j])
        } else {
            None
        };
        match border {
            Some(p) => x[..dim].copy_from_slice(p),
            None => {
                let u = i as f64 / last1 as f64;
                let v = j as f64 / last2 as f64;
                let c1 = &side1_coords[i];
                let c3 = &side3_coords[last1 - i];
                let c4 = &side4_coords[last2 - j];
                let c2 = &side2_coords[j];
                for d in 0..dim {
                    x[d] = (1.0 - v) * c1[d] + v * c3[d] + (1.0 - u) * c4[d] + u * c2[d]
                        - ((1.0 - u) * (1.0 - v) * p00[d]
                            + u * (1.0 - v) * p10[d]
                            + (1.0 - u) * v * p01[d]
                            + u * v * p11[d]);
                }
            }
        }
        x
    };
    let midpoint = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        let mut m = [0.0; 3];
        for d in 0..dim {
            m[d] = 0.5 * (a[d] + b[d]);
        }
        m
    };

    // The node of every grid point, decided once: the border from the sides,
    // the interior laid out flat and created in one locked pass over the
    // `Coords`. The corners take the Coons patch, the mid-nodes the middle of
    // the chord between their two corners; a QUA8 has no node where two odd
    // indices cross (the middle of a cell).
    let width = last2 + 1;
    let mut grid: Vec<NodeId> = vec![NodeId(u32::MAX); (last1 + 1) * width];
    for i in 0..=last1 {
        grid[i * width] = side1_ids[i];
        grid[i * width + last2] = side3_ids[last1 - i];
    }
    for j in 0..=last2 {
        grid[j] = side4_ids[last2 - j];
        grid[last1 * width + j] = side2_ids[j];
    }
    let mut flat: Vec<f64> = Vec::new();
    let mut interior: Vec<usize> = Vec::new();
    for i in 1..last1 {
        for j in 1..last2 {
            let x = match (i % step, j % step) {
                (0, 0) => corner(i, j),
                (1, 0) => midpoint(corner(i - 1, j), corner(i + 1, j)),
                (0, 1) => midpoint(corner(i, j - 1), corner(i, j + 1)),
                _ => continue,
            };
            flat.extend_from_slice(&x[..dim]);
            // The rank among the created nodes, turned into an id below.
            grid[i * width + j] = NodeId(interior.len() as u32);
            interior.push(i * width + j);
        }
    }
    let first = coords.write().add_nodes(&flat)?.start;
    for &k in &interior {
        grid[k].0 += first;
    }
    let at = |i: usize, j: usize| grid[i * width + j];

    let element_type = if quadratic {
        ElementType::QUA8
    } else {
        ElementType::QUA4
    };
    let mut conn: Vec<NodeId> = Vec::with_capacity(n1 * n2 * element_type.nodes_per_cell());
    for i in 0..n1 {
        for j in 0..n2 {
            let (i0, i1, j0, j1) = (step * i, step * (i + 1), step * j, step * (j + 1));
            conn.extend_from_slice(&[at(i0, j0), at(i1, j0), at(i1, j1), at(i0, j1)]);
            if quadratic {
                // The mid-nodes of the edges 0-1, 1-2, 2-3, 3-0.
                let (im, jm) = (i0 + 1, j0 + 1);
                conn.extend_from_slice(&[at(im, j0), at(i1, jm), at(im, j1), at(i0, jm)]);
            }
        }
    }
    let sm = SubMesh::from_connectivity(coords.clone(), element_type, conn)?;
    // The patch owns the fresh nodes now; hand back the unit `add_nodes`
    // handed us.
    let owned: Vec<NodeId> = (first..first + interior.len() as u32).map(NodeId).collect();
    coords.write().decref_all(&owned)?;
    Ok(Mesh::from_submesh(sm))
}

/// The node row of a single-submesh `SEG2` or `SEG3` side, read in order
/// along it ([`line_row`]: corners, and for a `SEG3` the mid-node of each
/// cell between its corners), with its element type and the positions of
/// the row's nodes — the same convention as
/// [`super::sweep_kernel::surface_between`].
type SideColumns = (Handle<Coords>, ElementType, Vec<NodeId>, Vec<Vec<f64>>);

fn side_columns(mesh: &Mesh, label: &str) -> Result<SideColumns> {
    if mesh.len() != 1 {
        return Err(PyrucastError::Message(format!(
            "transfinite: {label} must have exactly one submesh"
        )));
    }
    let sm = mesh.get(0)?;
    // Guard held while the connectivity is read; `Coords` is a distinct
    // object, locked after it as everywhere else.
    let s = sm.read();
    let (coords, et, n_elems, conn) = (
        s.coords(),
        s.element_type(),
        s.cell_count(),
        s.connectivity(),
    );
    if !matches!(et, ElementType::SEG2 | ElementType::SEG3) {
        return Err(PyrucastError::Message(format!(
            "transfinite: {label} must be a SEG2 or SEG3 line, got {et}"
        )));
    }
    let ids = line_row(conn, et, n_elems);
    let coord_vals: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    Ok((coords, et, ids, coord_vals))
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
        assert_eq!(mesh.cell_count(), 12);

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
        assert_eq!(mesh.cell_count(), 1);

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
        assert_eq!(mesh.cell_count(), 8);
    }

    #[test]
    fn transfinite_qua8_promotes_to_quadratic() {
        let (side1, side2, side3, side4) = unit_square(2, 2);
        let mesh = transfinite(&side1, &side2, &side3, &side4, ElementType::QUA8).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::QUA8]);
        assert_eq!(mesh.cell_count(), 4);
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
    fn transfinite_rejects_a_side_that_is_not_a_line() {
        let (side1, side2, side3, side4) = unit_square(1, 1);
        let coords = side1.coords().unwrap();
        let a = Node::create_in(coords.clone(), &[5.0, 5.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[6.0, 5.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[5.5, 6.0]).unwrap();
        let mut tri_mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri_mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        assert!(transfinite(&tri_mesh, &side2, &side3, &side4, ElementType::QUA4).is_err());
    }

    /// Node positions of every cell of the single zone, in local order.
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

    /// The four sides of the rectangle [0, 3] × [0, 2], of type `et`, with 3
    /// cells on side1/side3 and 2 on side2/side4.
    fn rectangle(coords: &Handle<Coords>, et: ElementType) -> [Mesh; 4] {
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap();
        let (a, b, c, d) = (p(0.0, 0.0), p(3.0, 0.0), p(3.0, 2.0), p(0.0, 2.0));
        [
            line(&a, &b, 3, et).unwrap(),
            line(&b, &c, 2, et).unwrap(),
            line(&c, &d, 3, et).unwrap(),
            line(&d, &a, 2, et).unwrap(),
        ]
    }

    #[test]
    fn transfinite_seg3_sides_match_the_promoted_grid_on_straight_sides() {
        // Straight SEG3 sides, or SEG2 sides promoted: the same cells, node
        // by node — Coons corners, chord mid-nodes, centres at the mean of the
        // corners of these rectangular cells.
        let coords = Handle::new(Coords::new(2).unwrap());
        let [a2, b2, c2, d2] = rectangle(&coords, ElementType::SEG2);
        let [a3, b3, c3, d3] = rectangle(&coords, ElementType::SEG3);
        for et in [ElementType::QUA8, ElementType::QUA9, ElementType::TRI6] {
            let promoted = transfinite(&a2, &b2, &c2, &d2, et).unwrap();
            let native = transfinite(&a3, &b3, &c3, &d3, et).unwrap();
            assert_eq!(native.element_types().unwrap(), vec![et]);
            let (x, y) = (cell_positions(&native), cell_positions(&promoted));
            assert_eq!(x.len(), y.len(), "{et}");
            for (cx, cy) in x.iter().zip(&y) {
                for (px, py) in cx.iter().zip(cy) {
                    assert!(
                        (px[0] - py[0]).abs() < 1e-12 && (px[1] - py[1]).abs() < 1e-12,
                        "{et}: {px:?} vs {py:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn transfinite_seg3_between_arcs_keeps_their_mid_nodes() {
        // A quarter ring between radii 1 and 2, two SEG3 cells per side.
        use crate::ops::mesh::arc::arc;
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]).unwrap();
        let o = p(0.0, 0.0);
        let (a, b, c, d) = (p(1.0, 0.0), p(2.0, 0.0), p(0.0, 2.0), p(0.0, 1.0));
        let side1 = line(&a, &b, 2, ElementType::SEG3).unwrap();
        let outer = arc(&b, &o, &c, 2, ElementType::SEG3).unwrap();
        let side3 = line(&c, &d, 2, ElementType::SEG3).unwrap();
        let inner = arc(&d, &o, &a, 2, ElementType::SEG3).unwrap();
        let qua8 = transfinite(&side1, &outer, &side3, &inner, ElementType::QUA8).unwrap();
        assert_eq!(qua8.element_types().unwrap(), vec![ElementType::QUA8]);
        assert_eq!(qua8.cell_count(), 4);
        let at = |m: &Mesh, c: usize, k: usize| m.node(0, c, k).unwrap();
        // Cell 0 (next to (1, 0)): its edge 3-0 lies on the inner arc, and its
        // mid-node is the arc's own, at 22.5°, radius 1.
        let arc_mid = inner.node(0, 1, 2).unwrap();
        assert_eq!(at(&qua8, 0, 7).id(), arc_mid.id());
        let x = arc_mid.position().unwrap();
        assert!((x[0].hypot(x[1]) - 1.0).abs() < 1e-12);
        // Its edge 1-2 is inside: the mid-node is the middle of the chord
        // between (1.5, 0) and the Coons corner (1.06066, 1.06066).
        let x = at(&qua8, 0, 5).position().unwrap();
        assert!(
            (x[0] - 1.28033).abs() < 1e-5 && (x[1] - 0.53033).abs() < 1e-5,
            "{x:?}"
        );
        // The centre of the QUA9: half the inner arc's sagitta off the mean of
        // the corners.
        let qua9 = transfinite(&side1, &outer, &side3, &inner, ElementType::QUA9).unwrap();
        let x = at(&qua9, 0, 8).position().unwrap();
        assert!(
            (x[0] - 1.10210).abs() < 1e-5 && (x[1] - 0.45651).abs() < 1e-5,
            "{x:?}"
        );
    }

    #[test]
    fn transfinite_seg3_refuses_linear_targets_and_mixed_degrees() {
        // Refused before any node is created: no demotion of SEG3 sides, and
        // four sides of one degree.
        let coords = Handle::new(Coords::new(2).unwrap());
        let [a, b, c, d] = rectangle(&coords, ElementType::SEG3);
        let [_, b2, _, _] = rectangle(&coords, ElementType::SEG2);
        let before = coords.read().node_count();
        assert!(transfinite(&a, &b, &c, &d, ElementType::QUA4).is_err());
        assert!(transfinite(&a, &b, &c, &d, ElementType::TRI3).is_err());
        let err = transfinite(&a, &b2, &c, &d, ElementType::QUA8).unwrap_err();
        assert!(err.to_string().contains("same degree"), "{err}");
        assert_eq!(coords.read().node_count(), before);
    }
}
