//! Boundary-extraction operator: the skin of a volume mesh, split by face.
//!
//! [`skin`] is the 3-D companion of [`crate::ops::mesher::border()`]: where
//! `border` returns the boundary *loops* of a surface mesh, `skin` returns the
//! boundary *surface* of a volume mesh (TET4 / PENTA6 / HEX8 cells), grouped
//! into the flat faces of the solid — one submesh per face.
//!
//! A *boundary* facet is a volume-element facet (a TET4 face, a HEX8 face, …)
//! used by exactly one cell; interior facets are shared by two cells and
//! cancel. The boundary facets are then grouped into flat faces: two adjacent
//! facets (sharing an edge) belong to the same face when they are nearly
//! coplanar — their outward normals differ by at most a threshold angle. A
//! cube thus yields six submeshes, a prism five (two caps and three sides).

use crate::aggregate::Aggregate;
use crate::containers::mesh::{Coords, ElementType, Mesh, NodeId, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};
use std::collections::HashMap;

/// Default coplanarity threshold: two boundary facets whose outward normals
/// differ by at most this angle are treated as part of the same flat face.
const DEFAULT_ANGLE_DEG: f64 = 1.0;

/// Faces of a TET4 — each oriented outwards (CCW seen from outside).
const TET4_FACES: [&[usize]; 4] = [&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]];

/// Faces of a HEX8 — bottom / top / four lateral, outward-oriented, in the
/// `[bot[0..4], top[0..4]]` convention used by [`crate::ops::mesher::extrude`].
const HEX8_FACES: [&[usize]; 6] = [
    &[0, 3, 2, 1], // bottom (normal opposed to extrusion direction)
    &[4, 5, 6, 7], // top
    &[0, 1, 5, 4],
    &[1, 2, 6, 5],
    &[2, 3, 7, 6],
    &[3, 0, 4, 7],
];

/// Faces of a PENTA6 prism — two triangular caps then three quadrilateral
/// sides, outward-oriented, in the `[bot[0..3], top[0..3]]` convention.
const PENTA6_FACES: [&[usize]; 5] = [
    &[0, 2, 1],    // bottom triangle (normal opposed to extrusion direction)
    &[3, 4, 5],    // top triangle
    &[0, 1, 4, 3], // side
    &[1, 2, 5, 4], // side
    &[2, 0, 3, 5], // side
];

/// Undirected edge key: the two node ids sorted, so an edge and its reverse
/// share the same key (adjacency is orientation-agnostic).
fn edge_key(u: NodeId, v: NodeId) -> (NodeId, NodeId) {
    if u.0 <= v.0 {
        (u, v)
    } else {
        (v, u)
    }
}

/// Local facets of a volume element type, as slices of local node indices,
/// each oriented outwards. `None` for non-volume types.
fn element_facets(et: ElementType) -> Option<&'static [&'static [usize]]> {
    match et {
        ElementType::TET4 => Some(&TET4_FACES),
        ElementType::HEX8 => Some(&HEX8_FACES),
        ElementType::PENTA6 => Some(&PENTA6_FACES),
        _ => None,
    }
}

/// A boundary facet: its node ids in outward order and its unit normal.
struct Facet {
    nodes: Vec<NodeId>,
    normal: [f64; 3],
}

/// Newell's method: robust unit normal of a (possibly non-planar) polygon.
/// Returns `[0.0; 3]` for a degenerate facet.
fn facet_normal(c: &Coords, nodes: &[NodeId]) -> Result<[f64; 3]> {
    let mut n = [0.0f64; 3];
    let k = nodes.len();
    for i in 0..k {
        let p = c.coord(nodes[i])?;
        let q = c.coord(nodes[(i + 1) % k])?;
        n[0] += (p[1] - q[1]) * (p[2] + q[2]);
        n[1] += (p[2] - q[2]) * (p[0] + q[0]);
        n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        for x in &mut n {
            *x /= len;
        }
    }
    Ok(n)
}

/// Extract the boundary surface of a **volume** mesh (TET4 / PENTA6 / HEX8
/// cells) as one TRI3 / QUA4 submesh per **flat face** of the solid.
///
/// Every element facet is taken with its outward orientation; a facet used by
/// exactly one cell is a *boundary* facet (interior facets appear twice and
/// cancel). Boundary facets from all volume submeshes are pooled together,
/// then grouped into flat faces: adjacent facets (sharing an edge) join the
/// same face while their outward normals stay within `angle_deg` degrees of
/// each other (default 1°). Each group becomes one submesh — one TRI3 and/or
/// one QUA4 submesh per flat face (a face made of both triangles and quads,
/// e.g. at a TET4/HEX8 interface, yields both).
///
/// A cube yields six submeshes, a prism five (two triangular caps, three
/// quadrilateral sides). The original nodes are reused (and re-referenced).
///
/// POI1 submeshes are ignored. Errors if the mesh has no volume cells, if it
/// carries cells that are neither POI1, TET4, PENTA6 nor HEX8, or if the
/// coordinate space is not 3-D.
pub fn skin(mesh: &Mesh, angle_deg: Option<f64>) -> Result<Mesh> {
    let angle = angle_deg.unwrap_or(DEFAULT_ANGLE_DEG);
    let cos_tol = angle.to_radians().cos();

    let coords = mesh.coords()?;
    {
        let c = read(&coords)?;
        if c.dim() != 3 {
            return Err(PyrucastError::Message(format!(
                "skin: only 3-D volume meshes are supported, got dim={}",
                c.dim()
            )));
        }
    }

    // 1. Count every element facet across all volume submeshes, keyed by its
    //    sorted node-id set so a facet and its neighbour's copy collide. A
    //    boundary facet is one whose key occurs exactly once.
    let mut counts: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut any_volume = false;
    let facet_key = |nodes: &[NodeId]| -> Vec<u32> {
        let mut k: Vec<u32> = nodes.iter().map(|n| n.0).collect();
        k.sort_unstable();
        k
    };
    for sm in mesh {
        let (et, conn) = {
            let s = read(sm)?;
            (s.element_type(), s.connectivity().to_vec())
        };
        if et == ElementType::POI1 {
            continue;
        }
        let facets = element_facets(et).ok_or_else(|| {
            PyrucastError::Message(format!(
                "skin: only volume meshes (TET4/PENTA6/HEX8) are supported, got {}",
                et
            ))
        })?;
        any_volume = true;
        let npc = et.nodes_per_cell();
        for cell in conn.chunks(npc) {
            for f in facets {
                let nodes: Vec<NodeId> = f.iter().map(|&li| cell[li]).collect();
                *counts.entry(facet_key(&nodes)).or_insert(0) += 1;
            }
        }
    }
    if !any_volume {
        return Err(PyrucastError::Message(
            "skin: mesh has no volume cells (TET4/PENTA6/HEX8)".into(),
        ));
    }

    // 2. Collect the boundary facets (outward-oriented) and their normals, in
    //    a deterministic order so the grouping is reproducible.
    let mut facets: Vec<Facet> = Vec::new();
    {
        let c = read(&coords)?;
        for sm in mesh {
            let (et, conn) = {
                let s = read(sm)?;
                (s.element_type(), s.connectivity().to_vec())
            };
            if et == ElementType::POI1 {
                continue;
            }
            let local = element_facets(et).unwrap();
            let npc = et.nodes_per_cell();
            for cell in conn.chunks(npc) {
                for f in local {
                    let nodes: Vec<NodeId> = f.iter().map(|&li| cell[li]).collect();
                    if counts.get(&facet_key(&nodes)) == Some(&1) {
                        let normal = facet_normal(&c, &nodes)?;
                        facets.push(Facet { nodes, normal });
                    }
                }
            }
        }
    }

    // 3. Build facet adjacency via shared undirected edges, then flood-fill
    //    into flat faces, crossing an edge only between near-coplanar facets.
    let mut edge_owners: HashMap<(NodeId, NodeId), Vec<usize>> = HashMap::new();
    for (fi, facet) in facets.iter().enumerate() {
        let k = facet.nodes.len();
        for i in 0..k {
            let key = edge_key(facet.nodes[i], facet.nodes[(i + 1) % k]);
            edge_owners.entry(key).or_default().push(fi);
        }
    }

    let coplanar = |a: usize, b: usize| -> bool {
        let na = facets[a].normal;
        let nb = facets[b].normal;
        let dot = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
        dot >= cos_tol
    };

    let mut group_of = vec![usize::MAX; facets.len()];
    let mut n_groups = 0usize;
    for seed in 0..facets.len() {
        if group_of[seed] != usize::MAX {
            continue;
        }
        let g = n_groups;
        n_groups += 1;
        group_of[seed] = g;
        let mut stack = vec![seed];
        while let Some(fi) = stack.pop() {
            let k = facets[fi].nodes.len();
            for i in 0..k {
                let key = edge_key(facets[fi].nodes[i], facets[fi].nodes[(i + 1) % k]);
                for &nb in &edge_owners[&key] {
                    if group_of[nb] == usize::MAX && coplanar(fi, nb) {
                        group_of[nb] = g;
                        stack.push(nb);
                    }
                }
            }
        }
    }

    // 4. Materialise: one TRI3 and/or one QUA4 submesh per group, in group
    //    order. Facets keep their outward orientation; nodes are reused.
    let mut result = Mesh::empty();
    for g in 0..n_groups {
        let mut tris: Vec<&Facet> = Vec::new();
        let mut quads: Vec<&Facet> = Vec::new();
        for (fi, facet) in facets.iter().enumerate() {
            if group_of[fi] != g {
                continue;
            }
            match facet.nodes.len() {
                3 => tris.push(facet),
                4 => quads.push(facet),
                other => {
                    return Err(PyrucastError::Message(format!(
                        "skin: unexpected facet with {} nodes",
                        other
                    )));
                }
            }
        }
        emit_submesh(&mut result, &coords, ElementType::TRI3, &tris)?;
        emit_submesh(&mut result, &coords, ElementType::QUA4, &quads)?;
    }
    Ok(result)
}

/// Append a submesh of the given surface type from `facets` (skipped if empty).
fn emit_submesh(
    result: &mut Mesh,
    coords: &Handle<Coords>,
    et: ElementType,
    facets: &[&Facet],
) -> Result<()> {
    if facets.is_empty() {
        return Ok(());
    }
    let mut sub = SubMesh::new(coords.clone(), et);
    for facet in facets {
        sub.add_cell(&facet.nodes)?;
    }
    result.add_sub(insert(sub))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    /// Eight corner nodes of the unit cube, indexed as HEX8 expects:
    /// bottom CCW (z=0) then top CCW (z=1).
    fn cube_nodes(coords: &Handle<Coords>) -> Vec<NodeId> {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        pts.iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect()
    }

    #[test]
    fn single_hex_gives_six_quad_faces() {
        let coords = insert(Coords::new(3).unwrap());
        let n = cube_nodes(&coords);
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 6, "six flat faces of the cube");
        assert_eq!(
            sk.element_types().unwrap(),
            vec![ElementType::QUA4; 6],
            "each face is one QUA4"
        );
        assert_eq!(sk.cell_counts().unwrap(), vec![1; 6]);
    }

    #[test]
    fn single_tet_gives_four_tri_faces() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
        m.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();

        let sk = skin(&m, None).unwrap();
        // A tet's four faces meet at sharp dihedral angles → four groups.
        assert_eq!(sk.len(), 4);
        assert_eq!(sk.element_types().unwrap(), vec![ElementType::TRI3; 4]);
    }

    #[test]
    fn cube_of_two_hexes_still_gives_six_faces() {
        // Split the unit cube in two along z = 0.5. The interior shared face
        // must cancel, and the two coplanar halves of each lateral side must
        // merge into a single flat face.
        let coords = insert(Coords::new(3).unwrap());
        let node =
            |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]).unwrap().id();
        // z = 0, 0.5, 1 layers, each a CCW square.
        let layer = |z: f64| {
            [
                node(0.0, 0.0, z),
                node(1.0, 0.0, z),
                node(1.0, 1.0, z),
                node(0.0, 1.0, z),
            ]
        };
        let l0 = layer(0.0);
        let l1 = layer(0.5);
        let l2 = layer(1.0);
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&[l0[0], l0[1], l0[2], l0[3], l1[0], l1[1], l1[2], l1[3]])
            .unwrap();
        m.add_cell(&[l1[0], l1[1], l1[2], l1[3], l2[0], l2[1], l2[2], l2[3]])
            .unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 6, "still six flat faces after the split");
        // The four lateral faces have two quads each; caps have one.
        let mut counts = sk.cell_counts().unwrap();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn prism_gives_two_tri_caps_and_three_quad_sides() {
        let coords = insert(Coords::new(3).unwrap());
        let node =
            |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]).unwrap().id();
        let n = [
            node(0.0, 0.0, 0.0),
            node(1.0, 0.0, 0.0),
            node(0.0, 1.0, 0.0),
            node(0.0, 0.0, 1.0),
            node(1.0, 0.0, 1.0),
            node(0.0, 1.0, 1.0),
        ];
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::PENTA6));
        m.add_cell(&n).unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 5, "two caps + three sides");
        let n_tri = sk
            .element_types()
            .unwrap()
            .iter()
            .filter(|&&t| t == ElementType::TRI3)
            .count();
        let n_quad = sk
            .element_types()
            .unwrap()
            .iter()
            .filter(|&&t| t == ElementType::QUA4)
            .count();
        assert_eq!(
            (n_tri, n_quad),
            (2, 3),
            "two triangular caps, three quad sides"
        );
    }

    #[test]
    fn large_angle_merges_all_faces_of_a_cube() {
        let coords = insert(Coords::new(3).unwrap());
        let n = cube_nodes(&coords);
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        // A 180° threshold makes every adjacent facet "coplanar": one group.
        let sk = skin(&m, Some(180.0)).unwrap();
        assert_eq!(sk.len(), 1);
        assert_eq!(sk.cell_counts().unwrap(), vec![6]);
    }

    #[test]
    fn reuses_nodes_and_increfs() {
        let coords = insert(Coords::new(3).unwrap());
        let n = cube_nodes(&coords);
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        // corner 0 is used by 3 of the cube's 6 faces.
        let before = read(&coords).unwrap().refcount(n[0]);
        let sk = skin(&m, None).unwrap();
        assert_eq!(read(&coords).unwrap().refcount(n[0]), before + 3);
        drop(sk);
        assert_eq!(read(&coords).unwrap().refcount(n[0]), before);
    }

    #[test]
    fn surface_cells_are_rejected() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(skin(&m, None).is_err());
    }

    #[test]
    fn no_volume_cells_is_error() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        assert!(skin(&m, None).is_err());
    }
}
