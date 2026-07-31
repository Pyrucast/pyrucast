//! The closed surface a volume mesh is grown from, and how to push it inward.
//!
//! A shell is a watertight set of `QUA4` and `TRI3` facets whose normals point
//! **out of the material** — the same convention as
//! [`triangulate_volume`](crate::ops::mesher::triangulate_volume), so a skin
//! taken from one mesher feeds straight into the other.
//!
//! ## Offsetting is not "move each node along its normal"
//!
//! Averaging the incident facet normals gives a direction, and that direction
//! is not enough. Move a cube's corner by `t` along the averaged normal
//! `(1,1,1)/√3` and each of the three faces ends up only `t/√3` away — the
//! layer comes out thinnest exactly where the geometry turns, which is where
//! it can least afford to be. Worse, at a tetrahedron's corner the averaged
//! normal is *tangent* to one of the incident faces, so moving along it offsets
//! that face by nothing at all. No amount of rescaling repairs that: the
//! direction itself is wrong.
//!
//! What the offset node actually has to be is the point where the incident
//! facets, each pushed inward by `t`, meet. Writing `n_j` for the outward
//! normals, that is
//!
//! ```text
//! d · n_j = -t   for every incident facet j,
//! ```
//!
//! three equations for three unknowns at a corner, more than three on a smooth
//! patch and fewer on an edge. Solving it in the least-squares sense — through
//! the normal equations `(NᵀN) d = -t Nᵀ1` — covers all three cases at once,
//! and returns the exact intersection whenever one exists. On the cube corner
//! it gives `d = -t(1,1,1)`; on the tetrahedron's, the direction no average
//! could have found.

use crate::containers::mesh::{Coords, ElementType, Mesh, NodeId, Point3, Vector3};
use crate::error::{PyrucastError, Result};
use crate::store::{read, Handle};
use std::collections::HashMap;

/// One facet of the shell, as indices into [`Shell::nodes`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Facet {
    Tri([u32; 3]),
    Quad([u32; 4]),
}

impl Facet {
    pub fn corners(&self) -> &[u32] {
        match self {
            Facet::Tri(t) => t,
            Facet::Quad(q) => q,
        }
    }
}

/// A watertight, consistently oriented surface of `TRI3` and `QUA4` facets.
#[derive(Debug)]
pub struct Shell {
    /// Store identity of each node, in the order the shell indexes them.
    pub nodes: Vec<NodeId>,
    pub points: Vec<Point3>,
    pub facets: Vec<Facet>,
}

impl Shell {
    /// Read `mesh` as a shell, rejecting anything that is not a closed,
    /// consistently oriented surface of triangles and quadrangles.
    pub fn extract(mesh: &Mesh, op: &str) -> Result<Shell> {
        let coords = mesh.coords()?;
        if read(&coords)?.dim() != 3 {
            return Err(PyrucastError::Message(format!(
                "{op}: the envelope must live in a 3-D Coords"
            )));
        }
        let c = read(&coords)?;
        let mut index: HashMap<NodeId, u32> = HashMap::new();
        let mut nodes = Vec::new();
        let mut points = Vec::new();
        let mut facets = Vec::new();
        let local = |id: NodeId,
                     nodes: &mut Vec<NodeId>,
                     points: &mut Vec<Point3>,
                     index: &mut HashMap<NodeId, u32>|
         -> Result<u32> {
            if let Some(&i) = index.get(&id) {
                return Ok(i);
            }
            let p = c.coord(id)?;
            let i = nodes.len() as u32;
            nodes.push(id);
            points.push(Point3::new(p[0], p[1], p[2]));
            index.insert(id, i);
            Ok(i)
        };

        for sm in mesh {
            let s = read(sm)?;
            let et = s.element_type();
            if !matches!(et, ElementType::TRI3 | ElementType::QUA4) {
                return Err(PyrucastError::Message(format!(
                    "{op}: the envelope must be made of TRI3 and QUA4 facets, got {et}"
                )));
            }
            let npc = et.nodes_per_cell();
            for cell in s.connectivity().chunks(npc) {
                let mut ids = [0u32; 4];
                for (k, &n) in cell.iter().enumerate() {
                    ids[k] = local(n, &mut nodes, &mut points, &mut index)?;
                }
                facets.push(if npc == 3 {
                    Facet::Tri([ids[0], ids[1], ids[2]])
                } else {
                    Facet::Quad(ids)
                });
            }
        }
        if facets.is_empty() {
            return Err(PyrucastError::Message(format!(
                "{op}: the envelope is empty"
            )));
        }

        let shell = Shell {
            nodes,
            points,
            facets,
        };
        shell.check_closed(op)?;
        Ok(shell)
    }

    /// Every edge must be used exactly twice, once in each direction — which
    /// is watertightness and orientation consistency in a single test.
    fn check_closed(&self, op: &str) -> Result<()> {
        let mut seen: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &self.facets {
            let c = f.corners();
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                *seen.entry(key).or_insert(0) += dir;
            }
        }
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        for f in &self.facets {
            let c = f.corners();
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        for (key, n) in &counts {
            if *n != 2 {
                return Err(PyrucastError::Message(format!(
                    "{op}: the envelope is not closed — the edge between nodes {:?} and {:?} \
                     belongs to {n} facet(s), not 2",
                    self.nodes[key.0 as usize], self.nodes[key.1 as usize]
                )));
            }
        }
        for (key, balance) in &seen {
            if *balance != 0 {
                return Err(PyrucastError::Message(format!(
                    "{op}: the envelope's facets disagree on orientation across the edge \
                     between nodes {:?} and {:?}; run `orient` on it first",
                    self.nodes[key.0 as usize], self.nodes[key.1 as usize]
                )));
            }
        }
        Ok(())
    }

    /// Area-weighted outward normal of a facet, and its area.
    fn facet_normal(&self, f: &Facet) -> (Vector3, f64) {
        let c = f.corners();
        // Newell's method: right for a quadrangle that is not quite planar.
        let mut n = Vector3::zeros();
        for i in 0..c.len() {
            let a = self.points[c[i] as usize];
            let b = self.points[c[(i + 1) % c.len()] as usize];
            n.x += (a.y - b.y) * (a.z + b.z);
            n.y += (a.z - b.z) * (a.x + b.x);
            n.z += (a.x - b.x) * (a.y + b.y);
        }
        let area = n.norm() * 0.5;
        if area == 0.0 {
            (Vector3::zeros(), 0.0)
        } else {
            (n / n.norm(), area)
        }
    }

    /// Mean edge length over the whole shell — the natural default thickness.
    pub fn mean_edge(&self) -> f64 {
        let (mut total, mut n) = (0.0, 0usize);
        for f in &self.facets {
            let c = f.corners();
            for i in 0..c.len() {
                total += (self.points[c[(i + 1) % c.len()] as usize] - self.points[c[i] as usize])
                    .norm();
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            total / n as f64
        }
    }

    /// Signed volume enclosed by the shell, positive when the normals point
    /// outward. Used to catch an envelope handed over inside-out.
    pub fn volume(&self) -> f64 {
        let mut v = 0.0;
        for f in &self.facets {
            let c = f.corners();
            for i in 1..c.len() - 1 {
                let (a, b, d) = (
                    self.points[c[0] as usize],
                    self.points[c[i] as usize],
                    self.points[c[i + 1] as usize],
                );
                v += a.coords.dot(&b.coords.cross(&d.coords));
            }
        }
        v / 6.0
    }

    /// Inward offset of every node, placing it where its incident facets meet
    /// once each has been pushed in by `thickness`.
    ///
    /// See the module docs: this is a small least-squares solve per node, not
    /// a step along an averaged normal, because the averaged normal can be
    /// tangent to an incident facet and offset it by nothing.
    pub fn inward_offsets(&self, thickness: f64) -> Vec<Vector3> {
        use nalgebra::Matrix3;
        let normals: Vec<(Vector3, f64)> =
            self.facets.iter().map(|f| self.facet_normal(f)).collect();
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); self.points.len()];
        for (fi, f) in self.facets.iter().enumerate() {
            for &v in f.corners() {
                incident[v as usize].push(fi);
            }
        }
        (0..self.points.len())
            .map(|v| {
                // Normal equations of `d · n_j = -t`, weighted by facet area so
                // a sliver does not outvote the faces that matter.
                let mut ata = Matrix3::zeros();
                let mut atb = Vector3::zeros();
                let mut mean = Vector3::zeros();
                for &fi in &incident[v] {
                    let (n, area) = normals[fi];
                    ata += n * n.transpose() * area;
                    atb -= n * (thickness * area);
                    mean += n * area;
                }
                if mean.norm() == 0.0 {
                    return Vector3::zeros();
                }
                // A node on a flat patch leaves the system rank-deficient in
                // the tangent directions; nudging the diagonal picks the
                // shortest solution there, which is the one along the normal.
                let trace = ata.trace().max(f64::MIN_POSITIVE);
                ata += Matrix3::identity() * (trace * TANGENT_REGULARISATION);
                let d = ata
                    .try_inverse()
                    .map(|inv| inv * atb)
                    .unwrap_or(-mean.normalize() * thickness);
                // A near-degenerate corner can send the intersection far away;
                // it is capped rather than allowed to turn the layer inside out.
                let cap = thickness * MAX_OFFSET_RATIO;
                if d.norm() > cap {
                    d.normalize() * cap
                } else {
                    d
                }
            })
            .collect()
    }

    /// The shell obtained by moving every node by `offset`, keeping the
    /// facets' connectivity and orientation.
    pub fn offset_by(
        &self,
        offset: &[Vector3],
        coords: &Handle<Coords>,
    ) -> Result<(Shell, Vec<crate::containers::mesh::Node>)> {
        use crate::containers::mesh::Node;
        let mut nodes = Vec::with_capacity(self.points.len());
        let mut points = Vec::with_capacity(self.points.len());
        let mut kept = Vec::with_capacity(self.points.len());
        for (i, p) in self.points.iter().enumerate() {
            let q = p + offset[i];
            let node = Node::create_in(coords.clone(), &[q.x, q.y, q.z])?;
            nodes.push(node.id());
            kept.push(node);
            points.push(q);
        }
        Ok((
            Shell {
                nodes,
                points,
                facets: self.facets.clone(),
            },
            kept,
        ))
    }
}

/// Diagonal nudge that makes the normal equations solvable on a flat patch,
/// where the tangent directions are unconstrained.
const TANGENT_REGULARISATION: f64 = 1e-12;

/// Largest offset, in multiples of the thickness, allowed at a sharp corner
/// where the offset planes meet a long way off.
const MAX_OFFSET_RATIO: f64 = 6.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Node, SubMesh};
    use crate::store::insert;

    /// The six faces of an axis-aligned box, as QUA4 with outward normals.
    fn box_shell(lo: [f64; 3], hi: [f64; 3]) -> Mesh {
        let coords = insert(Coords::new(3).unwrap());
        let corner = |i: usize| {
            let p = [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ];
            Node::create_in(coords.clone(), &p).unwrap().id()
        };
        let n: Vec<NodeId> = (0..8).map(corner).collect();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        // Outward-oriented, in the (x−, x+, y−, y+, z−, z+) order.
        for f in [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ] {
            sm.add_cell(&[n[f[0]], n[f[1]], n[f[2]], n[f[3]]]).unwrap();
        }
        Mesh::from_submesh(sm)
    }

    #[test]
    fn a_box_is_a_closed_shell_of_the_right_volume() {
        let m = box_shell([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let s = Shell::extract(&m, "test").unwrap();
        assert_eq!(s.facets.len(), 6);
        assert_eq!(s.points.len(), 8);
        assert!((s.volume() - 24.0).abs() < 1e-12, "volume {}", s.volume());
    }

    #[test]
    fn an_open_or_inconsistent_envelope_is_named_as_such() {
        // Drop one face: five quadrangles no longer close.
        let m = box_shell([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let coords = m.coords().unwrap();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        let s = read(&m[0]).unwrap();
        for cell in s.connectivity().chunks(4).take(5) {
            sm.add_cell(cell).unwrap();
        }
        let err = Shell::extract(&Mesh::from_submesh(sm), "test").unwrap_err();
        assert!(format!("{err}").contains("not closed"), "{err}");
    }

    #[test]
    fn the_offset_moves_every_face_by_the_thickness_asked_for() {
        // The point of the correction: at a box corner the averaged normal is
        // the diagonal, and a plain step along it would leave each face short
        // by a factor √3.
        let m = box_shell([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let s = Shell::extract(&m, "test").unwrap();
        let off = s.inward_offsets(0.1);
        for (i, o) in off.iter().enumerate() {
            let p = s.points[i];
            let q = p + o;
            for k in 0..3 {
                let moved = (q[k] - p[k]).abs();
                assert!(
                    (moved - 0.1).abs() < 1e-9,
                    "node {i} moved {moved} along axis {k}, wanted 0.1"
                );
            }
            // And inward: the offset box is strictly inside the unit cube.
            for k in 0..3 {
                assert!(q[k] > 0.0 - 1e-12 && q[k] < 1.0 + 1e-12);
            }
        }
    }

    #[test]
    fn the_offset_of_a_slanted_shell_keeps_each_facet_at_the_full_distance() {
        // A tetrahedron: the corner normals lean much further than a box's.
        let coords = insert(Coords::new(3).unwrap());
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let n: Vec<NodeId> = p
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        for f in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]] {
            sm.add_cell(&[n[f[0]], n[f[1]], n[f[2]]]).unwrap();
        }
        let s = Shell::extract(&Mesh::from_submesh(sm), "test").unwrap();
        assert!((s.volume() - 1.0 / 6.0).abs() < 1e-12, "{}", s.volume());
        let t = 0.05;
        let off = s.inward_offsets(t);
        // Every facet must have moved inward by at least `t`.
        for (fi, f) in s.facets.iter().enumerate() {
            let (normal, _) = s.facet_normal(f);
            for &v in f.corners() {
                let moved = -off[v as usize].dot(&normal);
                assert!(
                    moved >= t - 1e-12,
                    "facet {fi} moved {moved} at node {v}, wanted at least {t}"
                );
            }
        }
    }
}
