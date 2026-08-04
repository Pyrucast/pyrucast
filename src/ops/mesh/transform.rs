//! Rigid copies of a mesh: [`translate`], [`rotate`] and the symmetries
//! [`symmetry_point`], [`symmetry_line`], [`symmetry_plane`] (Cast3m `SYME`).
//!
//! All build a **brand-new mesh with its own fresh nodes** — the source
//! mesh and its nodes are left untouched. The result mirrors the input
//! submesh by submesh (same element types, same face colours, same
//! connectivity structure); only the node coordinates are transformed.
//! Nodes shared between cells of the source stay shared in the copy.
//!
//! # Orientation
//!
//! A symmetry can be **orientation-reversing** — its linear part has a
//! negative determinant, which would turn every cell inside out (negative
//! Jacobian, inward normals on a skin). Those operators therefore also apply
//! [`ElementType::reversal_permutation`](crate::atoms::ElementType::reversal_permutation)
//! to each cell, exactly as [`invert`](fn@super::invert) does, so the copy
//! comes out with the same handedness as the source and is directly usable
//! for a computation. Which cases reverse depends on the dimension:
//!
//! | operator | 2-D | 3-D |
//! |---|---|---|
//! | [`symmetry_point`] | direct (half-turn) | reversed |
//! | [`symmetry_line`] | reversed | direct (half-turn about the line) |
//! | [`symmetry_plane`] | reversed | reversed |
//!
//! Call [`invert`](fn@super::invert) on the result to get the raw mirrored
//! connectivity back.

use crate::aggregate::Aggregate;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};
use std::collections::HashMap;

/// Copy `mesh` into a fresh mesh, applying `f` to every node's coordinates.
///
/// Every distinct source node yields exactly one fresh node (so nodes shared
/// between cells stay shared), placed at `f(old_coord)`. The result keeps the
/// submesh order, element types and face colours of `mesh`; the source mesh
/// is left untouched.
///
/// `reverse` additionally flips each cell's node order (the same permutation
/// as [`invert`](fn@super::invert)), which is what an orientation-reversing
/// `f` needs to keep the copy's Jacobians positive.
fn map_coords(
    mesh: &Mesh,
    reverse: bool,
    mut f: impl FnMut(&[f64]) -> Result<Vec<f64>>,
) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // One fresh node per distinct source node (first-seen order is irrelevant
    // here since the map is keyed by the original id).
    let mut fresh: HashMap<NodeId, NodeId> = HashMap::new();
    for sm in mesh {
        for &id in read(sm)?.connectivity() {
            if let std::collections::hash_map::Entry::Vacant(e) = fresh.entry(id) {
                let old = read(&coords)?.position(id)?.to_vec();
                let new_coord = f(&old)?;
                let node = Node::create_in(coords.clone(), &new_coord)?;
                e.insert(node.id());
            }
        }
    }

    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = read(sm_handle)?;
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(color);
        let npc = et.nodes_per_cell();
        let perm = et.reversal_permutation();
        for chunk in conn.chunks(npc) {
            let mapped: Vec<NodeId> = if reverse {
                perm.iter().map(|&i| fresh[&chunk[i]]).collect()
            } else {
                chunk.iter().map(|id| fresh[id]).collect()
            };
            new_sm.add_cell(&mapped)?;
        }
        result.add_sub(insert(new_sm))?;
    }
    Ok(result)
}

/// Translate `mesh` by `vector`, returning a fresh copy with its own nodes.
///
/// `vector` must match the mesh's coordinate dimension. The original mesh is
/// left untouched.
pub fn translate(mesh: &Mesh, vector: &[f64]) -> Result<Mesh> {
    let dim = read(&mesh.coords()?)?.dim() as usize;
    if vector.len() != dim {
        return Err(PyrucastError::Message(format!(
            "translate: vector has {} components but the mesh is {}-D",
            vector.len(),
            dim
        )));
    }
    map_coords(mesh, false, |c| {
        Ok(c.iter().zip(vector).map(|(&x, &d)| x + d).collect())
    })
}

/// Rotate `mesh` by `angle` (radians), returning a fresh copy with its own
/// nodes. The original mesh is left untouched.
///
/// - **2-D** (`center` has 2 components): rotation about the point `center`,
///   counterclockwise for a positive `angle`. `axis` is ignored.
/// - **3-D** (`center` has 3 components): rotation about the line through
///   `center` directed by `axis` (Rodrigues' formula), right-handed about
///   `axis`. `axis` is required and must be non-degenerate; it need not be
///   normalized.
pub fn rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Result<Mesh> {
    let dim = read(&mesh.coords()?)?.dim() as usize;
    if center.len() != dim {
        return Err(PyrucastError::Message(format!(
            "rotate: center has {} components but the mesh is {}-D",
            center.len(),
            dim
        )));
    }
    let (cos, sin) = (angle.cos(), angle.sin());
    match dim {
        2 => {
            let (cx, cy) = (center[0], center[1]);
            map_coords(mesh, false, |c| {
                let (x, y) = (c[0] - cx, c[1] - cy);
                Ok(vec![cx + cos * x - sin * y, cy + sin * x + cos * y])
            })
        }
        3 => {
            let axis = axis.ok_or_else(|| {
                PyrucastError::Message("rotate: a 3-D mesh needs a rotation axis".into())
            })?;
            if axis.len() != 3 {
                return Err(PyrucastError::Message(format!(
                    "rotate: axis has {} components but must be 3-D",
                    axis.len()
                )));
            }
            let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if norm == 0.0 {
                return Err(PyrucastError::Message(
                    "rotate: axis must be non-zero".into(),
                ));
            }
            let u = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
            let (cx, cy, cz) = (center[0], center[1], center[2]);
            map_coords(mesh, false, |c| {
                let v = [c[0] - cx, c[1] - cy, c[2] - cz];
                // Rodrigues: v' = v cosθ + (u×v) sinθ + u (u·v)(1-cosθ).
                let cross = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
                let k = dot * (1.0 - cos);
                Ok(vec![
                    cx + v[0] * cos + cross[0] * sin + u[0] * k,
                    cy + v[1] * cos + cross[1] * sin + u[1] * k,
                    cz + v[2] * cos + cross[2] * sin + u[2] * k,
                ])
            })
        }
        other => Err(PyrucastError::Message(format!(
            "rotate: only 2-D and 3-D meshes are supported (got {other}-D)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Symmetries (Cast3m `SYME`)
// ---------------------------------------------------------------------------

/// The coordinate dimension of `mesh`, checking that `point` (named `name`
/// in the error message) matches it.
fn dim_of(mesh: &Mesh, point: &[f64], name: &str, op: &str) -> Result<usize> {
    let dim = read(&mesh.coords()?)?.dim() as usize;
    if point.len() != dim {
        return Err(PyrucastError::Message(format!(
            "{op}: {name} has {} components but the mesh is {}-D",
            point.len(),
            dim
        )));
    }
    Ok(dim)
}

/// `v` normalized, erroring out if it is the zero vector.
fn unit(v: &[f64], name: &str, op: &str) -> Result<Vec<f64>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err(PyrucastError::Message(format!(
            "{op}: {name} must be non-zero"
        )));
    }
    Ok(v.iter().map(|x| x / norm).collect())
}

/// Mirror `mesh` through the point `center`, returning a fresh copy with its
/// own nodes (Cast3m `SYME … POINT`). The original mesh is left untouched.
///
/// Every node goes to `2·center − x`, i.e. `center` is the midpoint of each
/// node and its image. This is the half-turn about `center` in 2-D and the
/// central inversion in 3-D; `center` must match the mesh dimension.
///
/// Orientation-reversing in 3-D (and in 1-D), so the cells are re-ordered
/// there — see the [module documentation](self).
pub fn symmetry_point(mesh: &Mesh, center: &[f64]) -> Result<Mesh> {
    let dim = dim_of(mesh, center, "center", "symmetry_point")?;
    // x ↦ 2c − x flips every axis: reversing iff the dimension is odd.
    map_coords(mesh, dim % 2 == 1, |c| {
        Ok(c.iter().zip(center).map(|(&x, &g)| 2.0 * g - x).collect())
    })
}

/// Mirror `mesh` through the (infinite) line running through `a` and `b`,
/// returning a fresh copy with its own nodes (Cast3m `SYME … DROIT`). The
/// original mesh is left untouched.
///
/// Each node is reflected across the line: the component along the line is
/// kept, the perpendicular one is negated. In **2-D** this is the mirror
/// image about the line; in **3-D** it is the half-turn about it (a rotation
/// of π, so *not* the mirror image through a plane — use
/// [`symmetry_plane`] for that).
///
/// Orientation-reversing in 2-D only — see the [module documentation](self).
///
/// Errors if `a` or `b` do not have the coordinate dimension, if they
/// coincide (no direction), or if the mesh is 1-D (where the line is the
/// whole space).
pub fn symmetry_line(mesh: &Mesh, a: &[f64], b: &[f64]) -> Result<Mesh> {
    let dim = dim_of(mesh, a, "a", "symmetry_line")?;
    dim_of(mesh, b, "b", "symmetry_line")?;
    if dim < 2 {
        return Err(PyrucastError::Message(format!(
            "symmetry_line: needs a 2-D or 3-D mesh (got {dim}-D)"
        )));
    }
    let dir: Vec<f64> = b.iter().zip(a).map(|(&bi, &ai)| bi - ai).collect();
    let d = unit(&dir, "the direction a → b", "symmetry_line")?;
    // The line keeps one direction and flips the dim−1 others.
    map_coords(mesh, dim % 2 == 0, |c| {
        let v: Vec<f64> = c.iter().zip(a).map(|(&x, &ai)| x - ai).collect();
        let along: f64 = v.iter().zip(&d).map(|(&vi, &di)| vi * di).sum();
        // x' = a + 2 (v·d̂) d̂ − v.
        Ok((0..v.len())
            .map(|i| a[i] + 2.0 * along * d[i] - v[i])
            .collect())
    })
}

/// Mirror `mesh` through the plane running through `origin` with normal
/// `normal`, returning a fresh copy with its own nodes (Cast3m
/// `SYME … PLAN`). The original mesh is left untouched.
///
/// Each node is reflected across the plane: `x ↦ x − 2((x − origin)·n̂) n̂`.
/// `normal` need not be normalized, and its sign is irrelevant. In **2-D**
/// the "plane" is the line through `origin` perpendicular to `normal` — the
/// same map as [`symmetry_line`], with the line given by a normal instead of
/// two points.
///
/// Always orientation-reversing, so the cells are re-ordered — see the
/// [module documentation](self).
///
/// Errors if `origin` or `normal` do not have the coordinate dimension, or
/// if `normal` is the zero vector.
pub fn symmetry_plane(mesh: &Mesh, origin: &[f64], normal: &[f64]) -> Result<Mesh> {
    dim_of(mesh, origin, "origin", "symmetry_plane")?;
    dim_of(mesh, normal, "normal", "symmetry_plane")?;
    let n = unit(normal, "normal", "symmetry_plane")?;
    // A single flipped direction, whatever the dimension.
    map_coords(mesh, true, |c| {
        let gap: f64 = c
            .iter()
            .zip(origin)
            .zip(&n)
            .map(|((&x, &o), &ni)| (x - o) * ni)
            .sum();
        Ok(c.iter()
            .zip(&n)
            .map(|(&x, &ni)| x - 2.0 * gap * ni)
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::coords::Coords;
    use crate::store::insert;
    use std::f64::consts::PI;

    fn tri2d(coords: &crate::store::Handle<Coords>, pts: [[f64; 2]; 3]) -> (Mesh, [Node; 3]) {
        let n0 = Node::create_in(coords.clone(), &pts[0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &pts[1]).unwrap();
        let n2 = Node::create_in(coords.clone(), &pts[2]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        (m, [n0, n1, n2])
    }

    #[test]
    fn translate_makes_fresh_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, src) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let out = translate(&m, &[10.0, 5.0]).unwrap();
        assert_eq!(out.element_types().unwrap(), vec![ElementType::TRI3]);
        // Fresh node, distinct id from the source.
        let n0 = out.node(0, 0, 0).unwrap();
        assert_ne!(n0.id(), src[0].id());
        assert_eq!(n0.position().unwrap(), vec![10.0, 5.0]);
        assert_eq!(
            out.node(0, 0, 1).unwrap().position().unwrap(),
            vec![11.0, 5.0]
        );
        // Source is untouched.
        assert_eq!(src[0].position().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn translate_keeps_shared_nodes_shared() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        m.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let out = translate(&m, &[0.0, 0.0]).unwrap();
        // b appears in both cells; the copy must reuse one fresh node for it.
        assert_eq!(
            out.node(0, 0, 1).unwrap().id(),
            out.node(0, 1, 0).unwrap().id()
        );
    }

    #[test]
    fn rotate_2d_quarter_turn() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]);

        let out = rotate(&m, PI / 2.0, &[0.0, 0.0], None).unwrap();
        let n0 = out.node(0, 0, 0).unwrap().position().unwrap();
        assert!((n0[0] - 0.0).abs() < 1e-12 && (n0[1] - 1.0).abs() < 1e-12);
        let n1 = out.node(0, 0, 1).unwrap().position().unwrap();
        assert!((n1[0] - 0.0).abs() < 1e-12 && (n1[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rotate_3d_about_z() {
        let coords = insert(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 5.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[0.0, 1.0, 5.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 0.0, 5.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();

        let out = rotate(&m, PI / 2.0, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        let p0 = out.node(0, 0, 0).unwrap().position().unwrap();
        // (1,0,5) rotated +90° about z → (0,1,5).
        assert!(
            (p0[0]).abs() < 1e-12 && (p0[1] - 1.0).abs() < 1e-12 && (p0[2] - 5.0).abs() < 1e-12
        );
    }

    #[test]
    fn rotate_3d_needs_axis() {
        let coords = insert(Coords::new(3).unwrap());
        let n0 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[n0.id()]).unwrap();
        assert!(rotate(&m, 1.0, &[0.0, 0.0, 0.0], None).is_err());
    }

    /// Signed area of a 2-D triangular cell, positive when counterclockwise.
    fn signed_area(mesh: &Mesh) -> f64 {
        let p: Vec<Vec<f64>> = (0..3)
            .map(|i| mesh.node(0, 0, i).unwrap().position().unwrap())
            .collect();
        0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
            - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
    }

    /// Signed volume of a 3-D tetrahedral cell, positive for the reference
    /// (direct) node ordering.
    fn signed_volume(mesh: &Mesh) -> f64 {
        let p: Vec<Vec<f64>> = (0..4)
            .map(|i| mesh.node(0, 0, i).unwrap().position().unwrap())
            .collect();
        let e: Vec<Vec<f64>> = (1..4)
            .map(|k| (0..3).map(|j| p[k][j] - p[0][j]).collect())
            .collect();
        (e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
            - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
            + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]))
            / 6.0
    }

    fn tet(coords: &crate::store::Handle<Coords>, pts: [[f64; 3]; 4]) -> Mesh {
        let ids: Vec<_> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
        m.add_cell(&ids).unwrap();
        m
    }

    #[test]
    fn symmetry_point_2d_is_a_half_turn() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, src) = tri2d(&coords, [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]);

        let out = symmetry_point(&m, &[0.0, 0.0]).unwrap();
        assert_eq!(out.node(0, 0, 0).unwrap().position().unwrap(), [-1.0, 0.0]);
        assert_eq!(out.node(0, 0, 1).unwrap().position().unwrap(), [-2.0, 0.0]);
        assert_eq!(out.node(0, 0, 2).unwrap().position().unwrap(), [-1.0, -1.0]);
        // Direct in 2-D: same winding as the source, and fresh nodes.
        assert!(signed_area(&out) * signed_area(&m) > 0.0);
        assert_ne!(out.node(0, 0, 0).unwrap().id(), src[0].id());
    }

    #[test]
    fn symmetry_point_3d_keeps_the_jacobian_positive() {
        let coords = insert(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        );
        assert!(signed_volume(&m) > 0.0);

        let out = symmetry_point(&m, &[0.0, 0.0, 0.0]).unwrap();
        // Reversing in 3-D: the cell is re-ordered, so the volume stays
        // positive while every node is at −x.
        assert!(signed_volume(&out) > 0.0);
        let mut got: Vec<Vec<f64>> = (0..4)
            .map(|i| out.node(0, 0, i).unwrap().position().unwrap())
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![0.0, 0.0, -1.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn symmetry_line_2d_mirrors_and_reverses() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 1.0], [1.0, 1.0], [0.0, 3.0]]);

        // Mirror about the x axis: y ↦ −y.
        let out = symmetry_line(&m, &[0.0, 0.0], &[1.0, 0.0]).unwrap();
        // TRI3 reversal is [0, 2, 1]: slot 1 of the copy is the image of the
        // source's node 2.
        assert_eq!(out.node(0, 0, 0).unwrap().position().unwrap(), [0.0, -1.0]);
        assert_eq!(out.node(0, 0, 1).unwrap().position().unwrap(), [0.0, -3.0]);
        assert_eq!(out.node(0, 0, 2).unwrap().position().unwrap(), [1.0, -1.0]);
        // Reversing in 2-D: the winding is restored.
        assert!(signed_area(&out) * signed_area(&m) > 0.0);
    }

    #[test]
    fn symmetry_line_3d_is_a_half_turn() {
        let coords = insert(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [1.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
            ],
        );

        // About the z axis: the same map as a rotation of π, node for node.
        let out = symmetry_line(&m, &[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]).unwrap();
        let turned = rotate(&m, PI, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0])).unwrap();
        for i in 0..4 {
            let a = out.node(0, 0, i).unwrap().position().unwrap();
            let b = turned.node(0, 0, i).unwrap().position().unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
        assert!(signed_volume(&out) > 0.0);
    }

    #[test]
    fn symmetry_plane_3d_mirrors_and_reverses() {
        let coords = insert(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
        );

        // Mirror through z = 0, normal given unnormalized on purpose.
        let out = symmetry_plane(&m, &[0.0, 0.0, 0.0], &[0.0, 0.0, 3.0]).unwrap();
        assert!(signed_volume(&out) > 0.0);
        let mut got: Vec<Vec<f64>> = (0..4)
            .map(|i| out.node(0, 0, i).unwrap().position().unwrap())
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![
                vec![0.0, 0.0, -2.0],
                vec![0.0, 0.0, -1.0],
                vec![0.0, 1.0, -1.0],
                vec![1.0, 0.0, -1.0],
            ]
        );
    }

    #[test]
    fn symmetry_plane_2d_matches_symmetry_line() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 1.0], [1.0, 1.0], [0.0, 3.0]]);

        // The line y = x, once by two points, once by its normal.
        let by_line = symmetry_line(&m, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        let by_plane = symmetry_plane(&m, &[0.0, 0.0], &[1.0, -1.0]).unwrap();
        for i in 0..3 {
            let a = by_line.node(0, 0, i).unwrap().position().unwrap();
            let b = by_plane.node(0, 0, i).unwrap().position().unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }

    #[test]
    fn symmetry_keeps_shared_nodes_shared() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        m.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let out = symmetry_plane(&m, &[0.0, 0.0], &[0.0, 1.0]).unwrap();
        // b sits in slot 2 of the first cell and slot 0 of the second once
        // both are reversed; it must still be one single fresh node.
        assert_eq!(
            out.node(0, 0, 2).unwrap().id(),
            out.node(0, 1, 0).unwrap().id()
        );
    }

    #[test]
    fn symmetry_rejects_bad_geometry() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(symmetry_point(&m, &[0.0, 0.0, 0.0]).is_err());
        assert!(symmetry_line(&m, &[0.0, 0.0], &[0.0, 0.0]).is_err());
        assert!(symmetry_plane(&m, &[0.0, 0.0], &[0.0, 0.0]).is_err());
        assert!(symmetry_plane(&m, &[0.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn translate_rejects_wrong_dim() {
        let coords = insert(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(translate(&m, &[1.0, 2.0, 3.0]).is_err());
    }
}
