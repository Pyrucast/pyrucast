//! The closed surface a volume mesh is grown from, and how to push it inward.
//!
//! A shell is a watertight set of `QUA4` and `TRI3` facets whose normals point
//! **out of the material** — the same convention as
//! [`triangulate_volume`](fn@crate::ops::mesher::triangulate_volume), so a skin
//! taken from one mesher feeds straight into the other.
//!
//! Everything that happens *to* the front once it starts moving — the local
//! step, the seams, the smoothing — lives in [`super::front`]; this module
//! only reads the surface in and vouches for it.
//!
use crate::containers::mesh::{ElementType, Mesh, NodeId, Point3};
use crate::error::{PyrucastError, Result};
use crate::store::read;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, Node, SubMesh};
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
    fn a_box_reads_as_a_closed_shell() {
        let m = box_shell([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let s = Shell::extract(&m, "test").unwrap();
        assert_eq!(s.facets.len(), 6);
        assert_eq!(s.points.len(), 8, "the eight corners, each read once");
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
}
