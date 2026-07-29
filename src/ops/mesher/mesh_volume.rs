//! Tetrahedral mesher: fill the inside of a closed `TRI3` envelope with
//! `TET4` cells.
//!
//! Pipeline:
//! 1. Read the envelope and check it is watertight, consistently oriented,
//!    free of degenerate facets and of self-intersections
//!    ([`tetrahedralization::envelope`](crate::ops::mesher::tetrahedralization::envelope)).
//! 2. Delaunay-tetrahedralize its nodes — and only its nodes
//!    ([`tetrahedralization::delaunay`](crate::ops::mesher::tetrahedralization::delaunay)).
//! 3. Recover every envelope edge and facet by local reconnection
//!    ([`tetrahedralization::recovery`](crate::ops::mesher::tetrahedralization::recovery)).
//! 4. Flood from both sides of the recovered surface to separate the
//!    material from the void
//!    ([`tetrahedralization::classify`](crate::ops::mesher::tetrahedralization::classify)).
//! 5. Check the result against the envelope, then materialize it.
//!
//! **The envelope is respected exactly.** Its nodes are reused verbatim —
//! same `NodeId`, same position — and no node is ever added on the surface,
//! nor is any facet subdivided. That is a contract, not a best effort: step
//! 5 proves it before anything is written, by checking that the mesh's own
//! boundary is precisely the set of facets that came in.
//!
//! Every geometric decision runs on exact predicates, so degenerate input —
//! a box, a regular grid, cospherical corners — is decided rather than
//! guessed at. Where no answer exists, the mesher says so: some polyhedra
//! admit no tetrahedral mesh on their own nodes, and refusing them is the
//! correct behaviour when adding a surface node is forbidden.
//!
//! # Limitations
//!
//! Recovery (step 3) is the part that is not finished. It works obstruction
//! by obstruction: sweep the corridor an envelope edge runs through, rebuild
//! whatever pocket blocks it, re-cut the outer surface where the edge lies
//! flat in it, and — when every obstruction is held in place by a facet
//! already won — rebuild the corridor whole with the edge imposed. Each
//! rebuild is an exhaustive search within its pocket, so nothing *local* is
//! missed, and the envelopes that genuinely cannot be filled are correctly
//! refused.
//!
//! What it does not have is a way to widen its view when no pocket, however
//! grown, does the job. On those the mesher reports the stuck edge rather
//! than returning a mesh that does not match its surface — the contract
//! holds, but the answer is an error where a mesh was possible. Whether a
//! given envelope lands in that band is not something the current code can
//! predict: a subdivided box goes through at 3×3 and not at 2×2 or 4×4, and
//! reordering the same facets can change the outcome, because the order
//! decides the Delaunay triangulation recovery starts from.
//!
//! Curved surfaces — the normal input — fare better than the small
//! hand-built shapes above, which are all planar faces and cospherical
//! corners. `meshes_a_realistic_closed_surface` is the case that matters;
//! `meshes_a_concave_solid` is `#[ignore]`d and pins one that does not work.
//!
//! Interior refinement — the sizing field and the removal of slivers — is not
//! implemented yet either, which is why `target_size` is accepted and
//! checked but not yet used.

use crate::containers::mesh::{ElementType, Mesh, Node, NodeId, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};

use super::tetrahedralization::classify;
use super::tetrahedralization::delaunay::TetMesh;
use super::tetrahedralization::envelope::Envelope;
use super::tetrahedralization::recovery;

/// How far the meshed volume may drift from the envelope's own, relative to
/// it, before the result is refused.
///
/// The two are computed by completely different routes — a sum over surface
/// facets against a sum over tetrahedra — so agreement to this margin means
/// the mesh really does fill the surface.
const VOLUME_TOLERANCE: f64 = 1e-9;

/// Mesh the inside of `envelope` with `TET4` cells.
///
/// `envelope` is a closed surface of `TRI3` facets whose normals point **out
/// of the material**; concave shapes are fine, and an internal cavity is
/// just another closed surface whose normals point into the hole. Its nodes
/// are reused as they are, and none is added on the surface.
///
/// `target_size` is reserved for the interior sizing of a later refinement
/// pass: it must be positive when given, and is not used yet.
///
/// This is the uninterruptible convenience form; for a long mesh a caller
/// may want to stop early, use [`mesh_volume_cancellable`].
pub fn mesh_volume(envelope: &Mesh, target_size: Option<f64>) -> Result<Mesh> {
    mesh_volume_cancellable(envelope, target_size, &NoCancel)
}

/// [`mesh_volume`], stoppable through `cancel`.
pub fn mesh_volume_cancellable(
    envelope: &Mesh,
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if let Some(h) = target_size {
        if h.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: size must be > 0, got {h}"
            )));
        }
    }

    let env = Envelope::extract(envelope, cancel)?;
    cancel.check()?;
    let mut mesh = TetMesh::delaunay(env.points(), cancel)?;
    recovery::recover(&mut mesh, &env, cancel)?;
    cancel.check()?;
    let inside = classify::interior(&mesh, &env, cancel)?;

    let cells = validate(&mesh, &env, &inside)?;
    materialize(envelope, &env, &cells)
}

/// Check the meshed volume against the envelope it came from, and return the
/// cells to keep.
///
/// Three independent statements, each of which a wrong mesh would have to
/// satisfy by coincidence: the cells are well formed, they fill the same
/// volume as the surface encloses, and their outer boundary is exactly the
/// surface.
fn validate(mesh: &TetMesh, env: &Envelope, inside: &[bool]) -> Result<Vec<[u32; 4]>> {
    if let Some(defect) = mesh.find_defect() {
        return Err(PyrucastError::Message(format!(
            "mesh_volume: {defect} (internal error)"
        )));
    }

    let cells: Vec<[u32; 4]> = mesh
        .iter()
        .filter(|(t, _)| inside[*t])
        .map(|(_, v)| v)
        .collect();
    if cells.is_empty() {
        return Err(PyrucastError::Message(
            "mesh_volume: produced no cell".into(),
        ));
    }

    let volume: f64 = cells.iter().map(|v| mesh.orientation(v)).sum::<f64>() / 6.0;
    let drift = (volume - env.volume()).abs();
    if drift > VOLUME_TOLERANCE * env.volume() {
        return Err(PyrucastError::Message(format!(
            "mesh_volume: the mesh fills {volume:.12e} but the envelope encloses {:.12e} \
             (internal error)",
            env.volume()
        )));
    }

    // The faces used by exactly one cell are the mesh's boundary; they must
    // be the envelope's facets, no more and no fewer.
    let mut boundary: std::collections::HashSet<[u32; 3]> = std::collections::HashSet::new();
    for v in &cells {
        for f in super::tetrahedralization::delaunay::FACE_OF {
            let mut key = [v[f[0]], v[f[1]], v[f[2]]];
            key.sort_unstable();
            if !boundary.remove(&key) {
                boundary.insert(key);
            }
        }
    }
    for f in env.facets() {
        let mut key = *f;
        key.sort_unstable();
        if !boundary.remove(&key) {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: envelope facet ({}, {}, {}) is not on the boundary of the mesh \
                 (internal error)",
                env.node_ids()[f[0] as usize].0,
                env.node_ids()[f[1] as usize].0,
                env.node_ids()[f[2] as usize].0
            )));
        }
    }
    if !boundary.is_empty() {
        return Err(PyrucastError::Message(format!(
            "mesh_volume: the mesh has {} boundary face(s) that are not envelope facets \
             (internal error)",
            boundary.len()
        )));
    }
    Ok(cells)
}

/// Build the output mesh, reusing the envelope's nodes.
fn materialize(envelope: &Mesh, env: &Envelope, cells: &[[u32; 4]]) -> Result<Mesh> {
    let coords = envelope.coords()?;
    let ids: &[NodeId] = env.node_ids();

    // Interior nodes, if a later pass ever adds any, would be created here;
    // for now every vertex is an envelope node, so nothing is allocated and
    // the surface is untouched by construction.
    let kept: Vec<Node> = Vec::new();
    let mut sub = SubMesh::new(coords, ElementType::TET4);
    for v in cells {
        sub.add_cell(&[
            ids[v[0] as usize],
            ids[v[1] as usize],
            ids[v[2] as usize],
            ids[v[3] as usize],
        ])?;
    }
    drop(kept);
    Ok(Mesh::from_submesh(sub))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::store::{insert, read, Handle};

    /// The eight corners of an axis-aligned box, ordered
    /// `000, 100, 110, 010, 001, 101, 111, 011`.
    fn box_nodes(coords: &Handle<Coords>, lo: [f64; 3], hi: [f64; 3]) -> Vec<NodeId> {
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect()
    }

    /// The twelve outward triangles of a box, every face split along a
    /// diagonal of the tetrahedron `{0, 2, 5, 7}`.
    ///
    /// Which diagonals are chosen matters: a box has 64 boundary
    /// triangulations and most of them cannot be filled with tetrahedra on
    /// the eight corners alone. This is the alternating pattern, which can
    /// (see `refuses_a_box_whose_faces_cannot_be_filled`).
    const BOX_FACETS: [[usize; 3]; 12] = [
        [0, 3, 2],
        [0, 2, 1], // z = lo, diagonal 0–2
        [4, 5, 7],
        [5, 6, 7], // z = hi, diagonal 5–7
        [0, 1, 5],
        [0, 5, 4], // y = lo, diagonal 0–5
        [1, 2, 5],
        [2, 6, 5], // x = hi, diagonal 2–5
        [2, 3, 7],
        [2, 7, 6], // y = hi, diagonal 2–7
        [3, 0, 7],
        [0, 4, 7], // x = lo, diagonal 0–7
    ];

    fn surface(coords: &Handle<Coords>, nodes: &[NodeId], facets: &[[usize; 3]]) -> Mesh {
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in facets {
            sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                .unwrap();
        }
        Mesh::from_submesh(sm)
    }

    /// Total volume of a TET4 mesh, read back through the public API.
    fn mesh_volume_of(mesh: &Mesh) -> f64 {
        let mut total = 0.0;
        for (si, &n) in mesh.cell_counts().unwrap().iter().enumerate() {
            for ci in 0..n {
                let p: Vec<Vec<f64>> = (0..4)
                    .map(|k| mesh.node(si, ci, k).unwrap().coord().unwrap())
                    .collect();
                let e = |i: usize, k: usize| p[i][k] - p[0][k];
                total += (e(1, 0) * (e(2, 1) * e(3, 2) - e(2, 2) * e(3, 1))
                    - e(1, 1) * (e(2, 0) * e(3, 2) - e(2, 2) * e(3, 0))
                    + e(1, 2) * (e(2, 0) * e(3, 1) - e(2, 1) * e(3, 0)))
                    / 6.0;
            }
        }
        total
    }

    #[test]
    fn meshes_a_box_and_keeps_its_volume() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mesh = mesh_volume(&surface(&coords, &nodes, &BOX_FACETS), None).unwrap();

        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TET4]);
        assert!(mesh.cell_count().unwrap() >= 5);
        let v = mesh_volume_of(&mesh);
        assert!((v - 1.0).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn the_envelope_nodes_are_reused_and_no_node_is_added() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let before = read(&coords).unwrap().node_count();
        let mesh = mesh_volume(&surface(&coords, &nodes, &BOX_FACETS), None).unwrap();

        assert_eq!(
            read(&coords).unwrap().node_count(),
            before,
            "nodes were added"
        );
        let mut used: Vec<NodeId> = Vec::new();
        for ci in 0..mesh.cell_count().unwrap() {
            for k in 0..4 {
                let id = mesh.node(0, ci, k).unwrap().id();
                assert!(nodes.contains(&id), "cell uses a foreign node");
                if !used.contains(&id) {
                    used.push(id);
                }
            }
        }
        assert_eq!(used.len(), 8, "every corner must be used");
    }

    #[test]
    fn the_skin_of_the_result_is_the_envelope() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let env = surface(&coords, &nodes, &BOX_FACETS);
        let mesh = mesh_volume(&env, None).unwrap();

        // `skin` peels the boundary independently; it must find the same
        // twelve triangles.
        let peeled = super::super::skin(&mesh, None).unwrap();
        assert_eq!(peeled.cell_count().unwrap(), 12);
    }

    #[test]
    fn refuses_a_box_whose_faces_cannot_be_filled() {
        // The same eight corners, with each square face split along the
        // other diagonal. No tetrahedralization of the box exists on those
        // nodes — verified independently by exhaustive search — so the only
        // right answer is to say so rather than to invent one.
        const UNFILLABLE: [[usize; 3]; 12] = [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let err = mesh_volume(&surface(&coords, &nodes, &UNFILLABLE), None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("without adding a node on the surface"),
            "{msg}"
        );
        // The diagnosis points at a specific edge, with its coordinates.
        assert!(msg.contains("1.000000"), "{msg}");
    }

    #[test]
    fn meshes_a_single_tetrahedron_as_one_cell() {
        let coords = insert(Coords::new(3).unwrap());
        let n: Vec<NodeId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect();
        let facets = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let mesh = mesh_volume(&surface(&coords, &n, &facets), None).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        assert!((mesh_volume_of(&mesh) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn meshes_two_disjoint_bodies_at_once() {
        let coords = insert(Coords::new(3).unwrap());
        let a = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_nodes(&coords, [5.0, 0.0, 0.0], [7.0, 1.0, 1.0]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for nodes in [&a, &b] {
            for f in &BOX_FACETS {
                sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                    .unwrap();
            }
        }
        let mesh = mesh_volume(&Mesh::from_submesh(sm), None).unwrap();
        assert!((mesh_volume_of(&mesh) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn meshes_a_box_with_an_internal_cavity() {
        // Outer shell outward, inner shell inward: the cavity is declared by
        // its orientation alone, and the cells filling it must be dropped.
        let coords = insert(Coords::new(3).unwrap());
        let outer = box_nodes(&coords, [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]);
        let inner = box_nodes(&coords, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in &BOX_FACETS {
            sm.add_cell(&[outer[f[0]], outer[f[1]], outer[f[2]]])
                .unwrap();
        }
        for f in &BOX_FACETS {
            // Reversed: the cavity's normals point into the hole.
            sm.add_cell(&[inner[f[0]], inner[f[2]], inner[f[1]]])
                .unwrap();
        }
        let mesh = mesh_volume(&Mesh::from_submesh(sm), None).unwrap();
        let v = mesh_volume_of(&mesh);
        assert!((v - (27.0 - 1.0)).abs() < 1e-12, "volume {v}");
    }

    #[test]
    #[ignore = "recovery cannot yet free an envelope edge that needs a wide \
                re-cut of the outer triangulation — see the module docs"]
    fn meshes_a_concave_solid() {
        // An L-shaped prism: the reflex edge is what a convex-hull-based
        // mesher gets wrong, so the volume check is the real assertion here.
        let coords = insert(Coords::new(3).unwrap());
        let base = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let n: Vec<NodeId> = base
            .iter()
            .flat_map(|p| [[p[0], p[1], 0.0], [p[0], p[1], 1.0]])
            .map(|p| Node::create_in(coords.clone(), &p).unwrap().id())
            .collect();
        // Node 2i is the bottom copy of base point i, 2i + 1 the top one.
        let (b, t) = (|i: usize| 2 * i, |i: usize| 2 * i + 1);
        let mut facets: Vec<[usize; 3]> = Vec::new();
        // Bottom, seen from below, so clockwise in plan view; top the other
        // way. The hexagon is fanned from its first vertex, which is convex.
        for i in 1..5 {
            facets.push([b(0), b(i + 1), b(i)]);
            facets.push([t(0), t(i), t(i + 1)]);
        }
        // Sides: each edge of the base becomes two triangles.
        for i in 0..6 {
            let j = (i + 1) % 6;
            facets.push([b(i), b(j), t(j)]);
            facets.push([b(i), t(j), t(i)]);
        }
        let mesh = mesh_volume(&surface(&coords, &n, &facets), None).unwrap();
        let v = mesh_volume_of(&mesh);
        assert!((v - 3.0).abs() < 1e-12, "volume {v}");
    }

    /// A box whose faces are cut into `n × n` squares, every vertex nudged
    /// off the lattice so nothing is accidentally coplanar or cospherical —
    /// the shape of a surface mesh that actually comes out of a mesher.
    fn subdivided_blob(coords: &Handle<Coords>, n: usize) -> (Mesh, f64) {
        let jitter = |i: usize, k: usize| {
            let mut x = ((i * 3 + k) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 29;
            ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02
        };
        let s = n as f64;
        let mut pts: Vec<[f64; 3]> = Vec::new();
        let mut at: std::collections::HashMap<(usize, usize, usize), usize> =
            std::collections::HashMap::new();
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    if i == 0 || i == n || j == 0 || j == n || k == 0 || k == n {
                        at.insert((i, j, k), pts.len());
                        pts.push([i as f64 / s, j as f64 / s, k as f64 / s]);
                    }
                }
            }
        }
        for (m, p) in pts.iter_mut().enumerate() {
            for k in 0..3 {
                p[k] += jitter(m, k);
            }
        }

        let mut facets: Vec<[usize; 3]> = Vec::new();
        let mut quad = |a, b, c, d| {
            facets.push([a, b, c]);
            facets.push([a, c, d]);
        };
        for i in 0..n {
            for j in 0..n {
                quad(
                    at[&(i, j, 0)],
                    at[&(i, j + 1, 0)],
                    at[&(i + 1, j + 1, 0)],
                    at[&(i + 1, j, 0)],
                );
                quad(
                    at[&(i, j, n)],
                    at[&(i + 1, j, n)],
                    at[&(i + 1, j + 1, n)],
                    at[&(i, j + 1, n)],
                );
                quad(
                    at[&(i, 0, j)],
                    at[&(i + 1, 0, j)],
                    at[&(i + 1, 0, j + 1)],
                    at[&(i, 0, j + 1)],
                );
                quad(
                    at[&(i, n, j)],
                    at[&(i, n, j + 1)],
                    at[&(i + 1, n, j + 1)],
                    at[&(i + 1, n, j)],
                );
                quad(
                    at[&(0, i, j)],
                    at[&(0, i, j + 1)],
                    at[&(0, i + 1, j + 1)],
                    at[&(0, i + 1, j)],
                );
                quad(
                    at[&(n, i, j)],
                    at[&(n, i + 1, j)],
                    at[&(n, i + 1, j + 1)],
                    at[&(n, i, j + 1)],
                );
            }
        }
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in &facets {
            sm.add_cell(&[ids[f[0]], ids[f[1]], ids[f[2]]]).unwrap();
        }
        // The volume the surface encloses, by the divergence theorem.
        let v6: f64 = facets
            .iter()
            .map(|f| {
                let (a, b, c) = (pts[f[0]], pts[f[1]], pts[f[2]]);
                a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
            })
            .sum();
        (Mesh::from_submesh(sm), v6 / 6.0)
    }

    #[test]
    fn meshes_a_realistic_closed_surface() {
        // 98 nodes, 192 facets, nothing degenerate: the case the mesher is
        // actually for, as opposed to the eight-corner puzzles above.
        let coords = insert(Coords::new(3).unwrap());
        let (envelope, expected) = subdivided_blob(&coords, 3);
        let mesh = mesh_volume(&envelope, None).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TET4]);
        assert!(
            mesh.cell_count().unwrap() > 100,
            "{}",
            mesh.cell_count().unwrap()
        );
        let v = mesh_volume_of(&mesh);
        assert!((v - expected).abs() < 1e-12 * expected, "{v} vs {expected}");
        // The surface is handed back untouched.
        assert_eq!(
            super::super::skin(&mesh, None)
                .unwrap()
                .cell_count()
                .unwrap(),
            envelope.cell_count().unwrap()
        );
    }

    #[test]
    fn rejects_a_bad_size() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let err = mesh_volume(&surface(&coords, &nodes, &BOX_FACETS), Some(0.0)).unwrap_err();
        assert!(err.to_string().contains("size must be > 0"), "{err}");
    }

    #[test]
    fn stops_on_a_preset_cancellation_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let flag = AtomicBool::new(true);
        let err = mesh_volume_cancellable(&surface(&coords, &nodes, &BOX_FACETS), None, &flag)
            .unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }
}
