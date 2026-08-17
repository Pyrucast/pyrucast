//! Boundary-extraction operator: the skin of a volume mesh, split by face.
//!
//! [`skin()`] is the 3-D companion of [`crate::ops::mesh::border()`]: where
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
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::collections::HashMap;

/// Default coplanarity threshold: two boundary facets whose outward normals
/// differ by at most this angle are treated as part of the same flat face.
const DEFAULT_ANGLE_DEG: f64 = 1.0;

/// Undirected edge key: the two node ids sorted, so an edge and its reverse
/// share the same key (adjacency is orientation-agnostic).
fn edge_key(u: NodeId, v: NodeId) -> (NodeId, NodeId) {
    if u.0 <= v.0 {
        (u, v)
    } else {
        (v, u)
    }
}

/// Local facets of a **volume** element type, read from the element itself.
/// `None` for point, line and surface types, which have no skin.
fn element_facets(et: ElementType) -> Option<&'static [crate::atoms::Facet]> {
    (et.topological_dim() == 3).then(|| et.as_kind().facets())
}

/// A boundary facet of the volume mesh.
struct BoundaryFacet {
    /// The facet seen as an element of its own: `TRI3`/`QUA4` off a linear
    /// cell, `TRI6`/`QUA8`/`QUA9` off a quadratic one.
    element_type: ElementType,
    /// Global node ids in `element_type`'s local order — corners first — so the
    /// facet can be emitted verbatim as a cell.
    nodes: Vec<NodeId>,
    /// Outward unit normal.
    normal: [f64; 3],
}

impl BoundaryFacet {
    /// The facet's corner ids. Adjacency, keying and the normal all read these:
    /// they are what two neighbouring cells agree on, and the only nodes that
    /// describe the facet's polygon.
    fn corners(&self) -> &[NodeId] {
        &self.nodes[..self.element_type.as_kind().corner_count()]
    }
}

/// Newell's method: robust unit normal of a (possibly non-planar) polygon.
/// Returns `[0.0; 3]` for a degenerate facet. `nodes` must be the facet's
/// **corners**, in boundary order.
fn facet_normal(c: &Coords, nodes: &[NodeId]) -> Result<[f64; 3]> {
    let mut n = [0.0f64; 3];
    let k = nodes.len();
    for i in 0..k {
        let p = c.position(nodes[i])?;
        let q = c.position(nodes[(i + 1) % k])?;
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

/// Extract the boundary surface of a **volume** mesh as one surface submesh
/// per **flat face** of the solid and per facet type.
///
/// Works on every volume type — `TET4`, `PYRA5`, `PENTA6`, `HEX8` and their
/// quadratic counterparts `TET10`, `PENTA15`, `HEX20`, `HEX27` — because each
/// element declares its own facets
/// ([`ElementKind::facets`](crate::atoms::ElementKind::facets)). **A facet is
/// emitted in its own type**: a `HEX8` yields `QUA4` faces, a `TET10` yields
/// `TRI6` faces, a `HEX27` yields `QUA9` faces, so the skin of a quadratic
/// mesh is itself quadratic and keeps its mid-side nodes.
///
/// Every element facet is taken with its outward orientation; a facet used by
/// exactly one cell is a *boundary* facet (interior facets appear twice and
/// cancel). Sharing is decided on the facet's **corners**, so cells of
/// different degrees still cancel correctly. Boundary facets from all volume
/// submeshes are pooled together, then grouped into flat faces: adjacent
/// facets (sharing an edge) join the same face while their outward normals
/// stay within `angle_deg` degrees of each other (default 1°). Each group
/// becomes one submesh per facet type — a face made of both triangles and
/// quads, e.g. at a `PYRA5`/`HEX8` interface, yields both.
///
/// A cube yields six submeshes, a prism five (two triangular caps, three
/// quadrilateral sides). The original nodes are reused (and re-referenced).
///
/// POI1 submeshes are ignored. Errors if the mesh has no volume cells, if it
/// carries cells of a lower topological dimension, or if the coordinate space
/// is not 3-D.
pub fn skin(mesh: &Mesh, angle_deg: Option<f64>) -> Result<Mesh> {
    let angle = angle_deg.unwrap_or(DEFAULT_ANGLE_DEG);
    let cos_tol = angle.to_radians().cos();

    let coords = mesh.coords()?;
    {
        let c = coords.read();
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
            let s = sm.read();
            (s.element_type(), s.connectivity().to_vec())
        };
        if et == ElementType::POI1 {
            continue;
        }
        let facets = element_facets(et).ok_or_else(|| {
            PyrucastError::Message(format!(
                "skin: only volume meshes are supported, got {} (topological dim {})",
                et,
                et.topological_dim()
            ))
        })?;
        any_volume = true;
        let npc = et.nodes_per_cell();
        for cell in conn.chunks(npc) {
            for f in facets {
                // Keyed on corners: a facet shared by a linear and a quadratic
                // cell, or by two cells of any degree, yields the same key.
                let corners: Vec<NodeId> = f.corners().iter().map(|&li| cell[li]).collect();
                *counts.entry(facet_key(&corners)).or_insert(0) += 1;
            }
        }
    }
    if !any_volume {
        return Err(PyrucastError::Message(
            "skin: mesh has no volume cells".into(),
        ));
    }

    // 2. Collect the boundary facets (outward-oriented) and their normals, in
    //    a deterministic order so the grouping is reproducible.
    let mut facets: Vec<BoundaryFacet> = Vec::new();
    {
        let c = coords.read();
        for sm in mesh {
            let (et, conn) = {
                let s = sm.read();
                (s.element_type(), s.connectivity().to_vec())
            };
            if et == ElementType::POI1 {
                continue;
            }
            let local = element_facets(et).unwrap();
            let npc = et.nodes_per_cell();
            for cell in conn.chunks(npc) {
                for f in local {
                    let nodes: Vec<NodeId> = f.nodes.iter().map(|&li| cell[li]).collect();
                    let n_corners = f.element_type.as_kind().corner_count();
                    if counts.get(&facet_key(&nodes[..n_corners])) == Some(&1) {
                        let normal = facet_normal(&c, &nodes[..n_corners])?;
                        facets.push(BoundaryFacet {
                            element_type: f.element_type,
                            nodes,
                            normal,
                        });
                    }
                }
            }
        }
    }

    // 3. Build facet adjacency via shared undirected edges, then flood-fill
    //    into flat faces, crossing an edge only between near-coplanar facets.
    let mut edge_owners: HashMap<(NodeId, NodeId), Vec<usize>> = HashMap::new();
    for (fi, facet) in facets.iter().enumerate() {
        let corners = facet.corners();
        let k = corners.len();
        for i in 0..k {
            let key = edge_key(corners[i], corners[(i + 1) % k]);
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
            let corners = facets[fi].corners();
            let k = corners.len();
            for i in 0..k {
                let key = edge_key(corners[i], corners[(i + 1) % k]);
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
        // One submesh per (group, facet type), the types in the order the
        // parent elements declare them — deterministic, and stable whatever
        // the facets' discovery order.
        let mut types: Vec<ElementType> = Vec::new();
        for (fi, facet) in facets.iter().enumerate() {
            if group_of[fi] == g && !types.contains(&facet.element_type) {
                types.push(facet.element_type);
            }
        }
        types.sort_unstable_by_key(|et| et.name());
        for et in types {
            let cells: Vec<&BoundaryFacet> = facets
                .iter()
                .enumerate()
                .filter(|(fi, f)| group_of[*fi] == g && f.element_type == et)
                .map(|(_, f)| f)
                .collect();
            emit_submesh(&mut result, &coords, et, &cells)?;
        }
    }
    Ok(result)
}

/// Append a submesh of the given surface type from `facets` (skipped if empty).
fn emit_submesh(
    result: &mut Mesh,
    coords: &Handle<Coords>,
    et: ElementType,
    facets: &[&BoundaryFacet],
) -> Result<()> {
    if facets.is_empty() {
        return Ok(());
    }
    let mut sub = SubMesh::new(coords.clone(), et);
    for facet in facets {
        sub.add_cell(&facet.nodes)?;
    }
    result.add_sub(Handle::new(sub))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::handle::Handle;

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
        let coords = Handle::new(Coords::new(3).unwrap());
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

    /// Regression: `skin` used to carry its own facet table listing only
    /// TET4/HEX8/PENTA6, so a pyramid — the very element `pave_volume`
    /// produces to join a hex layer to a tet core — was rejected outright.
    /// Reading the element's own facets covers it.
    #[test]
    fn pyramid_gives_a_quad_base_and_four_tri_sides() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let n: Vec<NodeId> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::PYRA5));
        m.add_cell(&n).unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 5, "the base and the four slanted sides");
        let types = sk.element_types().unwrap();
        assert_eq!(
            types.iter().filter(|&&t| t == ElementType::QUA4).count(),
            1,
            "the square base"
        );
        assert_eq!(
            types.iter().filter(|&&t| t == ElementType::TRI3).count(),
            4,
            "the four triangles"
        );
    }

    /// Regression: a quadratic volume produced no skin at all. Its faces are
    /// now emitted in their own quadratic type, mid-side nodes included, so
    /// the skin of a TET10 is a TRI6 surface rather than a TRI3 one.
    #[test]
    fn quadratic_tet_gives_four_tri6_faces_carrying_their_mid_nodes() {
        let coords = Handle::new(Coords::new(3).unwrap());
        // Corners of the unit tet, then the six edge midpoints in TET10 order:
        // (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let n: Vec<NodeId> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET10));
        m.add_cell(&n).unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 4, "four faces at sharp dihedral angles");
        assert_eq!(sk.element_types().unwrap(), vec![ElementType::TRI6; 4]);

        // Every emitted node is one of the ten, and each face carries three
        // mid-side nodes — i.e. the skin really is quadratic.
        let c = coords.read();
        for sub in &sk {
            let s = sub.read();
            for cell in s.connectivity().chunks(6) {
                for (i, &node) in cell.iter().enumerate() {
                    let p = c.position(node).unwrap();
                    let on_a_half = p.iter().any(|v| (*v - 0.5).abs() < 1e-12);
                    assert_eq!(
                        i >= 3,
                        on_a_half,
                        "slot {i} should{} be a mid-side node",
                        if i >= 3 { "" } else { " not" }
                    );
                }
            }
        }
    }

    /// A quadratic hex has nine-node faces: the centre node must travel too.
    #[test]
    fn hex27_gives_qua9_faces() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let k = ElementType::HEX27.as_kind();
        let n: Vec<NodeId> = k
            .ref_nodes()
            .iter()
            .map(|xi| Node::create_in(coords.clone(), xi).unwrap().id())
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX27));
        m.add_cell(&n).unwrap();

        let sk = skin(&m, None).unwrap();
        assert_eq!(sk.len(), 6);
        assert_eq!(sk.element_types().unwrap(), vec![ElementType::QUA9; 6]);
        assert_eq!(sk.cell_counts().unwrap(), vec![1; 6]);
    }

    #[test]
    fn single_tet_gives_four_tri_faces() {
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let coords = Handle::new(Coords::new(3).unwrap());
        let n = cube_nodes(&coords);
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        m.add_cell(&n).unwrap();

        // corner 0 is used by 3 of the cube's 6 faces.
        let before = coords.read().refcount(n[0]);
        let sk = skin(&m, None).unwrap();
        assert_eq!(coords.read().refcount(n[0]), before + 3);
        drop(sk);
        assert_eq!(coords.read().refcount(n[0]), before);
    }

    #[test]
    fn surface_cells_are_rejected() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(skin(&m, None).is_err());
    }

    #[test]
    fn no_volume_cells_is_error() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        assert!(skin(&m, None).is_err());
    }
}
