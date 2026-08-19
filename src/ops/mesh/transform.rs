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
//! | [`symmetry_plane`] | — (3-D only) | reversed |
//!
//! Call [`invert`](fn@super::invert) on the result to get the raw mirrored
//! connectivity back.

use crate::aggregate::Aggregate;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
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
        for &id in sm.read().connectivity() {
            if let std::collections::hash_map::Entry::Vacant(e) = fresh.entry(id) {
                let old = coords.read().position(id)?.to_vec();
                let new_coord = f(&old)?;
                let node = Node::create_in(coords.clone(), &new_coord)?;
                e.insert(node.id());
            }
        }
    }

    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = sm_handle.read();
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
        result.add_sub(Handle::new(new_sm))?;
    }
    Ok(result)
}

/// Translate `mesh` by `vector`, returning a fresh copy with its own nodes.
///
/// `vector` must match the mesh's coordinate dimension. The original mesh is
/// left untouched.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let barre = Mesh::from_submesh(sm);
/// # let ou = |m: &Mesh, i| m.node(0, 0, i).unwrap().position().unwrap();
/// // Des nœuds **neufs**, aux positions décalées : l'original ne bouge pas.
/// let deplacee = mesh::translate(&barre, &[0.0, 2.0, 0.0])?;
/// assert_eq!(ou(&deplacee, 0), vec![0.0, 2.0, 0.0]);
/// assert_eq!(ou(&barre, 0), vec![0.0, 0.0, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn translate(mesh: &Mesh, vector: &[f64]) -> Result<Mesh> {
    let dim = mesh.coords()?.read().dim() as usize;
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let barre = Mesh::from_submesh(sm);
/// # let ou = |m: &Mesh, i| m.node(0, 0, i).unwrap().position().unwrap();
/// // Un quart de tour autour de z : (1, 0, 0) devient (0, 1, 0).
/// let tournee = mesh::rotate(
///     &barre, std::f64::consts::FRAC_PI_2, &[0.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0]))?;
/// let p = ou(&tournee, 1);
/// assert!(p[0].abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Result<Mesh> {
    let dim = mesh.coords()?.read().dim() as usize;
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
    let dim = mesh.coords()?.read().dim() as usize;
    if point.len() != dim {
        return Err(PyrucastError::Message(format!(
            "{op}: {name} has {} components but the mesh is {}-D",
            point.len(),
            dim
        )));
    }
    Ok(dim)
}

/// Euclidean norm of `v`.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `v` normalized, erroring out if it is the zero vector.
fn unit(v: &[f64], name: &str, op: &str) -> Result<Vec<f64>> {
    let n = norm(v);
    if n == 0.0 {
        return Err(PyrucastError::Message(format!(
            "{op}: {name} must be non-zero"
        )));
    }
    Ok(v.iter().map(|x| x / n).collect())
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let barre = Mesh::from_submesh(sm);
/// # let ou = |m: &Mesh, i| m.node(0, 0, i).unwrap().position().unwrap();
/// // Chaque point passe de l'autre côté du centre. La transformation est de
/// // déterminant −1, donc la connectivité est **réinversée** pour garder une
/// // orientation directe : l'image de (1, 0, 0) est en position locale 0.
/// let image = mesh::symmetry_point(&barre, &[0.0, 0.0, 0.0])?;
/// assert_eq!(ou(&image, 0), vec![-1.0, 0.0, 0.0]);
/// assert_eq!(ou(&image, 1), vec![0.0, 0.0, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let barre = Mesh::from_submesh(sm);
/// # let ou = |m: &Mesh, i| m.node(0, 0, i).unwrap().position().unwrap();
/// // Symétrie **axiale** autour de l'axe des x : un point sur cet axe ne
/// // bouge pas.
/// # let hors_axe = Node::create_in(coords.clone(), &[0.5, 1.0, 0.0])?;
/// # let mut s2 = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # s2.add_cell(&[n[0].id(), hors_axe.id()])?;
/// # let coude = Mesh::from_submesh(s2);
/// let image = mesh::symmetry_line(&coude, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0])?;
/// assert_eq!(ou(&image, 0), vec![0.0, 0.0, 0.0]);
/// assert_eq!(ou(&image, 1), vec![0.5, -1.0, 0.0]);
/// // En 3-D une symétrie axiale est une **rotation** d'un demi-tour : son
/// // déterminant vaut +1, donc la connectivité est laissée telle quelle —
/// // contrairement à `symmetry_point` et `symmetry_plane`.
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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

/// Mirror `mesh` through the plane running through the three points `a`, `b`
/// and `c`, returning a fresh copy with its own nodes (Cast3m
/// `SYME … PLAN`). The original mesh is left untouched.
///
/// Each node is reflected across the plane: `x ↦ x − 2((x − a)·n̂) n̂`, where
/// `n̂` is the unit normal of the plane. The three points play symmetric
/// roles — only the plane they span matters, not their order (a permutation
/// flips `n̂`, which the formula is insensitive to).
///
/// **3-D only**: three points define a plane in space. In 2-D the mirror is
/// [`symmetry_line`], which takes the two points of the line.
///
/// Always orientation-reversing, so the cells are re-ordered — see the
/// [module documentation](self).
///
/// Errors if the mesh is not 3-D, if `a`, `b` or `c` is not a 3-D point, or
/// if the three points are aligned (they then span no plane).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let barre = Mesh::from_submesh(sm);
/// # let ou = |m: &Mesh, i| m.node(0, 0, i).unwrap().position().unwrap();
/// // Le plan est donné par **trois points**, non par une normale.
/// # let hors_plan = Node::create_in(coords.clone(), &[0.5, 0.0, 1.0])?;
/// # let mut s2 = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # s2.add_cell(&[n[0].id(), hors_plan.id()])?;
/// # let oblique = Mesh::from_submesh(s2);
/// // Le plan z = 0 : seule la troisième coordonnée change de signe. Comme
/// // pour `symmetry_point`, la connectivité est réinversée — l'image du
/// // second nœud arrive en position locale 0.
/// let image = mesh::symmetry_plane(
///     &oblique, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?;
/// assert_eq!(ou(&image, 0), vec![0.5, 0.0, -1.0]);
/// assert_eq!(ou(&image, 1), vec![0.0, 0.0, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn symmetry_plane(mesh: &Mesh, a: &[f64], b: &[f64], c: &[f64]) -> Result<Mesh> {
    let dim = dim_of(mesh, a, "a", "symmetry_plane")?;
    dim_of(mesh, b, "b", "symmetry_plane")?;
    dim_of(mesh, c, "c", "symmetry_plane")?;
    if dim != 3 {
        return Err(PyrucastError::Message(format!(
            "symmetry_plane: three points span a plane in 3-D only (the mesh \
             is {dim}-D — use symmetry_line for the mirror about a line)"
        )));
    }
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    // ‖ab × ac‖ = ‖ab‖‖ac‖ sin θ: comparing it to the product of the lengths
    // tests the *angle*, so the check does not depend on the mesh's scale.
    let norms = norm(&ab) * norm(&ac);
    if norm(&cross) <= 1e-12 * norms {
        return Err(PyrucastError::Message(
            "symmetry_plane: a, b and c are aligned (or coincide) — they span \
             no plane"
                .into(),
        ));
    }
    let n = unit(&cross, "the plane normal", "symmetry_plane")?;
    // A single flipped direction: always orientation-reversing.
    map_coords(mesh, true, |x| {
        let gap: f64 = x
            .iter()
            .zip(a)
            .zip(&n)
            .map(|((&xi, &ai), &ni)| (xi - ai) * ni)
            .sum();
        Ok(x.iter()
            .zip(&n)
            .map(|(&xi, &ni)| xi - 2.0 * gap * ni)
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use std::f64::consts::PI;

    fn tri2d(coords: &crate::handle::Handle<Coords>, pts: [[f64; 2]; 3]) -> (Mesh, [Node; 3]) {
        let n0 = Node::create_in(coords.clone(), &pts[0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &pts[1]).unwrap();
        let n2 = Node::create_in(coords.clone(), &pts[2]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        (m, [n0, n1, n2])
    }

    #[test]
    fn translate_makes_fresh_nodes() {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]);

        let out = rotate(&m, PI / 2.0, &[0.0, 0.0], None).unwrap();
        let n0 = out.node(0, 0, 0).unwrap().position().unwrap();
        assert!((n0[0] - 0.0).abs() < 1e-12 && (n0[1] - 1.0).abs() < 1e-12);
        let n1 = out.node(0, 0, 1).unwrap().position().unwrap();
        assert!((n1[0] - 0.0).abs() < 1e-12 && (n1[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rotate_3d_about_z() {
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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

    fn tet(coords: &crate::handle::Handle<Coords>, pts: [[f64; 3]; 4]) -> Mesh {
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
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
        );

        // Mirror through z = 0, given by three of its points.
        let out = symmetry_plane(&m, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 4.0, 0.0]).unwrap();
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
    fn symmetry_plane_ignores_the_order_of_its_three_points() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
        );

        // Same plane, points given in the other cyclic order: swapping two of
        // them flips the normal, which the reflection does not see.
        let p = [[1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [1.0, 2.0, 0.0]];
        let out = symmetry_plane(&m, &p[0], &p[1], &p[2]).unwrap();
        let swapped = symmetry_plane(&m, &p[0], &p[2], &p[1]).unwrap();
        for i in 0..4 {
            let a = out.node(0, 0, i).unwrap().position().unwrap();
            let b = swapped.node(0, 0, i).unwrap().position().unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }

    #[test]
    fn symmetry_keeps_shared_nodes_shared() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        m.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let out = symmetry_line(&m, &[0.0, 0.0], &[1.0, 0.0]).unwrap();
        // b sits in slot 2 of the first cell and slot 0 of the second once
        // both are reversed; it must still be one single fresh node.
        assert_eq!(
            out.node(0, 0, 2).unwrap().id(),
            out.node(0, 1, 0).unwrap().id()
        );
    }

    #[test]
    fn symmetry_rejects_bad_geometry() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let (flat, _) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(symmetry_point(&flat, &[0.0, 0.0, 0.0]).is_err());
        assert!(symmetry_line(&flat, &[0.0, 0.0], &[0.0, 0.0]).is_err());
        // A plane needs three points in space: no 2-D mesh, no aligned points.
        assert!(symmetry_plane(&flat, &[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]).is_err());

        let coords = Handle::new(Coords::new(3).unwrap());
        let m = tet(
            &coords,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        );
        assert!(symmetry_plane(&m, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0]).is_err());
        // Aligned (and, in the second case, coincident) points.
        assert!(symmetry_plane(&m, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], &[3.0, 3.0, 3.0]).is_err());
        assert!(symmetry_plane(&m, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[0.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn translate_rejects_wrong_dim() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let (m, _) = tri2d(&coords, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!(translate(&m, &[1.0, 2.0, 3.0]).is_err());
    }
}
