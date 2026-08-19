//! Tetrahedral mesher: fill the inside of a closed `TRI3` envelope with
//! `TET4` cells.
//!
//! Pipeline:
//! 1. Read the envelope and check it is watertight, consistently oriented,
//!    free of degenerate facets and of self-intersections
//!    (`tetrahedralization::envelope`).
//! 2. Delaunay-tetrahedralize its nodes — and only its nodes
//!    (`tetrahedralization::delaunay`).
//! 3. Recover every envelope edge and facet by local reconnection
//!    (`tetrahedralization::recovery`).
//! 4. Flood from both sides of the recovered surface to separate the
//!    material from the void
//!    (`tetrahedralization::classify`).
//! 5. Check the result against the envelope, then materialize it.
//!
//! **The envelope is respected exactly.** Its nodes are reused verbatim —
//! same `NodeId`, same position — and no node is ever added on the surface,
//! nor is any facet subdivided. That is a contract, not a best effort: step
//! 5 proves it before anything is written, by checking that the mesh's own
//! boundary is precisely the set of facets that came in.
//!
//! # Letting the envelope be cut finer
//!
//! `allow_surface_nodes` trades the second half of that contract for
//! robustness, and it is the difference between a mesher that works on real
//! input and one that does not.
//!
//! What stays: the **shape**. Every node the mesher adds sits on the edge or
//! the facet it divides, so the surface is the same surface — only its
//! triangulation is finer. What goes: the **discretisation**. The skin of the
//! result no longer matches the surface mesh that was passed in, which
//! matters if two solids are meant to share a conforming interface, and not
//! at all if the envelope was only ever a way to describe a form. A warning
//! on stderr says how many nodes were added, because the difference is easy
//! to overlook — and the result **names them**, in a second submesh of `POI1`
//! alongside the cells, so a caller can act on it rather than read stderr.
//! That submesh appears only when there is something to name.
//!
//! It is also why it works. Recovery is hard precisely because a segment
//! that cannot be fitted has, otherwise, no way out; subdivide it and the
//! problem splits into two easier ones, and a segment closely enough
//! surrounded by its own subdivisions is recovered by the Delaunay
//! triangulation unaided.
//!
//! What is added is then handed back wherever the mesh will part with it
//! (`tetrahedralization::simplify`).
//! Asking *afterwards* is a far kinder question than asking recovery to
//! manage without: there is already a valid mesh, and each point only needs
//! its own neighbourhood rearranged. What comes back is the **collateral**
//! subdivision — the cuts that other cuts made necessary — while the ones
//! that were essential stay, as they must: removing them would put the mesh
//! back in the situation that had no answer.
//!
//! On the plate of `formation/maillage_test.py`, 3308 facets, this is 15
//! nodes added and 12 given back, leaving **3**, in under a second, where
//! the strict mode cannot finish at all. On a small plate of flat panels
//! nothing comes back, because every cut there was essential.
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
//! A pocket has two fillers. The exhaustive search above is complete — it alone
//! can prove a small pocket has *no* filling — but it is exponential and runs
//! out of road at half a dozen cells. The other one does not search: it
//! **computes** the Delaunay triangulation of the pocket's own vertices and
//! asks whether that contains every face of the pocket's surface, for the
//! price of one triangulation whatever the pocket's size. When it does not, it
//! names the face it stumbled on, and swallowing the cell beyond that face
//! makes the obstruction impossible to repeat. That is what settles the
//! **flat quadrilaterals of the surface** an extrusion's side wall is made
//! of — the facet goes in as a wall with both its faces, cutting the pocket
//! in two, and each half is filled from its own side.
//!
//! What it does **not** do is edges. A Delaunay triangulation is what it is:
//! one cannot *ask* it for an edge, and a missing envelope edge is missing
//! precisely because it is not Delaunay. So a blocked edge remains the
//! business of the flips, and where they do not suffice, of
//! `allow_surface_nodes`.
//!
//! # Quality
//!
//! A valid mesh is not the same as a usable one. Everything above only
//! guarantees that the cells fill the envelope with positive volumes; it
//! says nothing about their *shape*, and a tetrahedralization of a surface's
//! own nodes is always full of flat ones. Two further passes turn it into a
//! mesh one can compute on:
//!
//! - `tetrahedralization::refine` puts nodes
//!   **inside** the solid, at the circumcentres of the cells whose
//!   radius-edge ratio or size says they are wrong;
//! - `tetrahedralization::smooth` then chases
//!   the **slivers** refinement cannot reach, by reconnecting cells, taking
//!   out the edges a sliver is flat across, and relaxing interior nodes.
//!
//! What that is worth, measured on the plate of `formation/maillage_test.py`
//! (the median dihedral angle sits at 47° throughout, so what moves is the
//! tail):
//!
//! | cells | under 10° | under 1° | flattest / average |
//! |---|---|---|---|
//! | 28 000 | 0.59 % → 0.00 % | 0 → 0 | 5.5·10⁻² → 1.5·10⁻¹ |
//! | 402 000 | 0.94 % → 0.11 % | 0.022 % → 0.000 % | 7.3·10⁻⁵ → 3.6·10⁻³ |
//! | 977 000 | 0.91 % → 0.18 % | 0.028 % → 0.001 % | 4.0·10⁻³ → 7.1·10⁻⁴ |
//!
//! The second column is the whole point: a cell under 1° has a nearly
//! singular element matrix, and it does not take many to sink a computation.
//! The passes cost about 1.35× the meshing time, at every size.
//!
//! **What can still defeat all three** is a sliver whose four corners are all
//! nodes of the envelope. No node can be inserted to break it up and none of
//! its own can be moved, since the caller's nodes are fixed by contract. It
//! has become rare — no envelope in the test suite produces one — but it
//! remains possible, and cutting the surface there, when
//! `allow_surface_nodes` allows it, does not always work either: the four
//! corners can simply be in the wrong places.
//!
//! Rather than hand back a mesh with a singular cell in it, the mesher
//! **refuses**, and says where: the one thing that always clears such a cell
//! is a different surface discretisation, and only the caller can supply
//! that. The line is drawn at a shape of 10⁻⁴, several orders below anything
//! a sound mesh produces (see `DEGENERATE_SHAPE`), so a merely thin cell is
//! never mistaken for a flat one.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, Node, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};

use super::tetrahedralization::classify;
use super::tetrahedralization::delaunay::TetMesh;
use super::tetrahedralization::envelope::Envelope;
use super::tetrahedralization::recovery;
use super::tetrahedralization::refine;
use super::tetrahedralization::simplify;
use super::tetrahedralization::smooth;

/// How far the meshed volume may drift from the envelope's own, relative to
/// it, before the result is refused.
///
/// The two are computed by completely different routes — a sum over surface
/// facets against a sum over tetrahedra — so agreement to this margin means
/// the mesh really does fill the surface.
const VOLUME_TOLERANCE: f64 = 1e-9;

/// Rounds of envelope subdivision before the mesher gives up.
///
/// Each round cuts the pieces that would not fit and starts again. The
/// theory says this terminates — a segment closely enough surrounded by its
/// own subdivisions is recovered by the Delaunay triangulation unaided — so
/// the cap is a guard against pathological input, not the normal exit.
const MAX_SUBDIVISIONS: usize = 12;

/// Shape below which a cell is called unusable.
///
/// The measure is `η = 12 (3V)^(2/3) / Σℓ²`: 1 for a regular tetrahedron, 0
/// for a flat one, and independent of how big the mesh is. Sound cells sit
/// well above this even in the tail — the worst on the plate of
/// `formation/maillage_test.py` is 4·10⁻² — while a genuinely flat one lands
/// several orders below, so the line between "thin" and "degenerate" is not
/// a matter of taste.
const DEGENERATE_SHAPE: f64 = 1e-4;

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
/// may want to stop early, use [`triangulate_volume_cancellable`].
pub fn triangulate_volume(
    envelope: &Mesh,
    target_size: Option<f64>,
    allow_surface_nodes: bool,
) -> Result<Mesh> {
    triangulate_volume_cancellable(envelope, target_size, allow_surface_nodes, &NoCancel)
}

/// [`triangulate_volume`], stoppable through `cancel`.
pub fn triangulate_volume_cancellable(
    envelope: &Mesh,
    target_size: Option<f64>,
    allow_surface_nodes: bool,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if let Some(h) = target_size {
        if h.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(PyrucastError::Message(format!(
                "triangulate_volume: size must be > 0, got {h}"
            )));
        }
    }

    let mut env = Envelope::extract(envelope, cancel)?;
    let mut unusable: Option<PyrucastError> = None;
    for _ in 0..MAX_SUBDIVISIONS {
        cancel.check()?;
        let mut mesh = TetMesh::delaunay(env.points(), cancel)?;
        // A caller that allows subdividing has a cheaper way out than the
        // exhaustive pocket rebuilds, so recovery is told not to fight.
        let effort = if allow_surface_nodes {
            recovery::Effort::QUICK
        } else {
            recovery::Effort::THOROUGH
        };
        let stuck = recovery::recover(&mut mesh, &env, effort, cancel)?;

        if stuck.is_empty() {
            cancel.check()?;
            // The subdivision points were the price of getting a mesh, not
            // something wanted for itself: now that there is one, try to
            // give them back.
            reclaim(&mut mesh, &mut env, cancel)?;

            // Only now is there a solid to fill: the envelope is in place,
            // so the inside can be told from the outside and nodes can be
            // put where the mesh needs them.
            let walls = classify::walls_of(&env);
            let mut inside = classify::interior_within(&mesh, &env, &walls, cancel)?;
            let target = target_size.unwrap_or_else(|| env.mean_edge_length());
            refine::refine(&mut mesh, &mut inside, &walls, target, cancel)?;
            let inside = classify::interior_within(&mesh, &env, &walls, cancel)?;

            // Refinement cannot remove a sliver — no node insertion can —
            // so the mesh is improved rather than subdivided. Only the
            // interior nodes may move; the caller's own stay where they are.
            let movable: Vec<bool> = (0..mesh.points().len())
                .map(|i| i >= env.points().len())
                .collect();
            let protect = recovery::Protected::new(env.facets());
            smooth::smooth(&mut mesh, &inside, &movable, &walls, &protect, cancel)?;
            drop(protect);
            let inside = classify::interior_within(&mesh, &env, &walls, cancel)?;

            // A sliver whose four corners are all the caller's own nodes is
            // beyond both passes: nothing can be inserted to break it up and
            // none of its corners may move. The only cure is a finer surface
            // there, which is the caller's to allow.
            let stubborn = smooth::stubborn(&mesh, &inside, env.points().len(), DEGENERATE_SHAPE);
            if !stubborn.is_empty() {
                let complaint = degenerate(&mesh, &stubborn);
                let Some(cuts) = facets_of(&stubborn, &walls, &env) else {
                    return Err(complaint);
                };
                if !allow_surface_nodes {
                    return Err(complaint);
                }
                // Cutting the surface there may or may not free the cell;
                // remember the complaint in case it never does.
                unusable = Some(complaint);
                env.subdivide(&cuts)?;
                continue;
            }

            let cells = validate(&mesh, &env, &inside)?;
            let added = warn_if_subdivided(&env, &cells);
            return materialize(envelope, &env, mesh.points(), &cells, &added);
        }
        if !allow_surface_nodes {
            return Err(recovery::describe(&mesh, &stuck[0]));
        }
        // Cut the envelope finer where it would not go in, and start over.
        // Rebuilding from scratch rather than patching keeps the
        // triangulation a true Delaunay one at the start of every attempt,
        // which is what the recovery reasons about.
        env.subdivide(&stuck)?;
    }
    // Whichever wall the loop ran into, say which one.
    Err(unusable.unwrap_or_else(|| {
        PyrucastError::Message(format!(
            "triangulate_volume: the envelope still would not fit after {MAX_SUBDIVISIONS} rounds of \
             subdividing it — the surface is likely self-touching or extremely thin somewhere"
        ))
    }))
}

/// The envelope facets carried by cells that cannot be improved.
///
/// Cutting those facets is what gives the next attempt somewhere to put a
/// node. `None` when a cell has no facet of its own to cut, which leaves
/// nothing to try.
fn facets_of(
    stubborn: &[([u32; 4], f64)],
    walls: &std::collections::HashSet<[u32; 3]>,
    env: &Envelope,
) -> Option<Vec<recovery::Stuck>> {
    let mut cuts: Vec<recovery::Stuck> = Vec::new();
    for (v, _) in stubborn {
        for f in super::tetrahedralization::delaunay::FACE_OF {
            let face = [v[f[0]], v[f[1]], v[f[2]]];
            let mut k = face;
            k.sort_unstable();
            if !walls.contains(&k) {
                continue;
            }
            // Report it the way the envelope stores it, so subdividing finds
            // the facet it is meant to cut.
            if let Some(&g) = env.facets().iter().find(|g| {
                let mut s = **g;
                s.sort_unstable();
                s == k
            }) {
                cuts.push(recovery::Stuck::Facet(g));
            }
        }
    }
    cuts.sort_unstable_by_key(|s| match s {
        recovery::Stuck::Facet(f) => *f,
        recovery::Stuck::Edge(a, b) => [*a, *b, 0],
    });
    cuts.dedup();
    (!cuts.is_empty()).then_some(cuts)
}

/// The report handed back when a cell cannot be made usable.
fn degenerate(mesh: &TetMesh, stubborn: &[([u32; 4], f64)]) -> PyrucastError {
    let (v, quality) = stubborn
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("the list is not empty");
    let centre: Vec<String> = (0..3)
        .map(|k| {
            let mid: f64 = v.iter().map(|&i| mesh.points()[i as usize][k]).sum::<f64>() / 4.0;
            format!("{mid:.6}")
        })
        .collect();
    PyrucastError::Message(format!(
        "triangulate_volume: the mesh has {} flat cell(s) that cannot be improved — the worst, around \
         ({}), has a shape of {quality:.2e} where 1 is a regular tetrahedron. Its four corners \
         are all nodes of the envelope, so no node can be added to break it up and none of its \
         own may be moved. Refine the surface mesh around that point, or pass \
         allow_surface_nodes to let the mesher cut the envelope there itself.",
        stubborn.len(),
        centre.join(", ")
    ))
}

/// Take back every subdivision point the mesh will part with.
///
/// Asking now is a much kinder question than the one recovery faced. Recovery
/// had to find, in a triangulation that did not suit it, some arrangement
/// holding the envelope; here there already *is* a valid mesh, and each point
/// only needs its own neighbourhood rearranged without it. A local repair on
/// a sound structure succeeds far more often, and far faster, than a search
/// forward from an unsuitable one.
///
/// Whatever will not come out simply stays. The caller allowed the envelope
/// to be cut, so a few surviving points are a shortfall, not a failure.
fn reclaim(mesh: &mut TetMesh, env: &mut Envelope, cancel: &dyn Cancel) -> Result<()> {
    for (m, origin) in env.added_points() {
        cancel.check()?;
        let Some((dropped, merged)) = env.merge_around(m, origin) else {
            continue;
        };
        // The index has to be rebuilt each time: a successful removal changes
        // the envelope's facets, and the next attempt must see them.
        let taken = {
            let protect = recovery::Protected::new(env.facets());
            simplify::remove_vertex(mesh, m, &merged, &protect)?
        };
        if taken {
            env.unsplit(&dropped, &merged);
        }
    }
    Ok(())
}

/// Tell the caller, on stderr, when the envelope had to be cut finer, and
/// return which of its points are the new ones.
///
/// Allowing it is a deliberate choice, but the consequence is easy to
/// overlook: the shape is untouched, yet the skin of the result no longer
/// matches the surface mesh that was handed in, so two solids meshed this
/// way no longer share a conforming interface. Saying so out loud costs
/// nothing and saves a puzzling afternoon.
///
/// A message on stderr is however a poor thing to act on. The points come
/// back so they can be handed over as part of the mesh, where a caller can
/// look at them, plot them, or hand them to whatever needs to know the
/// interface moved. That changes the shape of the result — a submesh appears
/// that was not there otherwise — so the warning says so outright rather than
/// leaving it to be discovered by a loop that reads a marker as a cell.
fn warn_if_subdivided(env: &Envelope, cells: &[[u32; 4]]) -> Vec<u32> {
    let given = env.given_node_count() as u32;
    // Only the points that sit *on* the envelope count here; anything past
    // its own list is an interior node, which was never in question.
    let on_surface = env.points().len() as u32;
    let mut kept: Vec<u32> = cells
        .iter()
        .flatten()
        .copied()
        .filter(|&i| i >= given && i < on_surface)
        .collect();
    kept.sort_unstable();
    kept.dedup();
    let added = kept.len();
    if added == 0 {
        return kept;
    }
    eprintln!(
        "triangulate_volume: warning — the envelope would not fit as given, so {added} node(s) were \
         added on it. Its shape is unchanged (each new node lies on the edge or facet it \
         divides), but the skin of the result no longer matches the surface mesh passed in. \
         The result therefore carries a SECOND SUBMESH, of POI1, naming those {added} node(s): \
         its element types are [TET4, POI1] rather than [TET4] alone, so anything that walks \
         the submeshes or counts cells must take the TET4 one and leave the markers out."
    );
    kept
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
            "triangulate_volume: {defect} (internal error)"
        )));
    }

    let cells: Vec<[u32; 4]> = mesh
        .iter()
        .filter(|(t, _)| inside[*t])
        .map(|(_, v)| v)
        .collect();
    if cells.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_volume: produced no cell".into(),
        ));
    }

    let volume: f64 = cells.iter().map(|v| mesh.orientation(v)).sum::<f64>() / 6.0;
    let drift = (volume - env.volume()).abs();
    if drift > VOLUME_TOLERANCE * env.volume() {
        return Err(PyrucastError::Message(format!(
            "triangulate_volume: the mesh fills {volume:.12e} but the envelope encloses {:.12e} \
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
                "triangulate_volume: an envelope facet at {:?} is not on the boundary of the mesh \
                 (internal error)",
                env.points()[f[0] as usize]
            )));
        }
    }
    if !boundary.is_empty() {
        return Err(PyrucastError::Message(format!(
            "triangulate_volume: the mesh has {} boundary face(s) that are not envelope facets \
             (internal error)",
            boundary.len()
        )));
    }
    Ok(cells)
}

/// Build the output mesh, reusing the envelope's nodes.
///
/// `added` are the points that had to be put on the envelope, which come back
/// as a second submesh of `POI1` alongside the cells — the one part of the
/// result the caller did not ask for, named rather than left to be
/// rediscovered by comparing skins.
fn materialize(
    envelope: &Mesh,
    env: &Envelope,
    points: &[[f64; 3]],
    cells: &[[u32; 4]],
    added: &[u32],
) -> Result<Mesh> {
    let coords = envelope.coords()?;

    // The caller's own nodes come back untouched. Past them come the points
    // put on the envelope to make it fit, and past those the interior nodes
    // refinement placed; both need a node of their own, and only where a
    // cell actually uses them. The `Node` handles are held until the cells
    // hold them, so the store cannot collect a point between its creation
    // and its first use.
    let given = env.given_node_count();
    let mut ids: Vec<NodeId> = env.node_ids().to_vec();
    ids.resize(points.len(), NodeId(u32::MAX));
    let mut kept: Vec<Node> = Vec::new();
    for v in cells {
        for &i in v {
            if (i as usize) < given || ids[i as usize] != NodeId(u32::MAX) {
                continue;
            }
            let node = Node::create_in(coords.clone(), &points[i as usize])?;
            ids[i as usize] = node.id();
            kept.push(node);
        }
    }

    let mut sub = SubMesh::new(coords.clone(), ElementType::TET4);
    for v in cells {
        sub.add_cell(&[
            ids[v[0] as usize],
            ids[v[1] as usize],
            ids[v[2] as usize],
            ids[v[3] as usize],
        ])?;
    }
    let mut out = Mesh::from_submesh(sub);

    if !added.is_empty() {
        // Every one of these is used by a cell — that is how it was
        // collected — so it has a node by now.
        let marks: Vec<NodeId> = added.iter().map(|&i| ids[i as usize]).collect();
        out = out.union(&Mesh::from_submesh(SubMesh::poi1_from_node_ids(
            coords, &marks,
        )?))?;
    }
    drop(kept);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::handle::Handle;

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

    /// How many cells the `TET4` submesh holds.
    ///
    /// The result may carry a second submesh of `POI1` naming the nodes that
    /// had to be added on the envelope, which is not made of cells to measure.
    fn tet_count(mesh: &Mesh) -> usize {
        mesh.cell_counts().unwrap()[0]
    }

    /// Total volume of a TET4 mesh, read back through the public API.
    fn volume_of(mesh: &Mesh) -> f64 {
        let mut total = 0.0;
        for (si, &n) in mesh.cell_counts().unwrap().iter().enumerate() {
            if mesh.element_types().unwrap()[si] != ElementType::TET4 {
                continue;
            }
            for ci in 0..n {
                let p: Vec<Vec<f64>> = (0..4)
                    .map(|k| mesh.node(si, ci, k).unwrap().position().unwrap())
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
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mesh = triangulate_volume(&surface(&coords, &nodes, &BOX_FACETS), None, false).unwrap();

        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TET4]);
        assert!(mesh.cell_count().unwrap() >= 5);
        let v = volume_of(&mesh);
        assert!((v - 1.0).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn the_envelope_nodes_are_reused_and_none_is_added_on_it() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let envelope = surface(&coords, &nodes, &BOX_FACETS);
        let mesh = triangulate_volume(&envelope, None, false).unwrap();

        // Every corner the caller gave is still there, with its own identity.
        let mut used: Vec<NodeId> = Vec::new();
        for ci in 0..mesh.cell_count().unwrap() {
            for k in 0..4 {
                let id = mesh.node(0, ci, k).unwrap().id();
                if !used.contains(&id) {
                    used.push(id);
                }
            }
        }
        assert!(
            nodes.iter().all(|n| used.contains(n)),
            "an envelope node went missing"
        );

        // Refinement adds nodes, but strictly *inside*: peeling the result
        // gives back the twelve facets that came in, no more.
        let peeled = super::super::skin(&mesh, None).unwrap();
        assert_eq!(peeled.cell_count().unwrap(), 12);
    }

    #[test]
    fn the_skin_of_the_result_is_the_envelope() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let env = surface(&coords, &nodes, &BOX_FACETS);
        let mesh = triangulate_volume(&env, None, false).unwrap();

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
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let err =
            triangulate_volume(&surface(&coords, &nodes, &UNFILLABLE), None, false).unwrap_err();
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
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let mesh = triangulate_volume(&surface(&coords, &n, &facets), None, false).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        assert!((volume_of(&mesh) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn meshes_two_disjoint_bodies_at_once() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let a = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_nodes(&coords, [5.0, 0.0, 0.0], [7.0, 1.0, 1.0]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for nodes in [&a, &b] {
            for f in &BOX_FACETS {
                sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                    .unwrap();
            }
        }
        let mesh = triangulate_volume(&Mesh::from_submesh(sm), None, false).unwrap();
        assert!((volume_of(&mesh) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn meshes_a_box_with_an_internal_cavity() {
        // Outer shell outward, inner shell inward: the cavity is declared by
        // its orientation alone, and the cells filling it must be dropped.
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let mesh = triangulate_volume(&Mesh::from_submesh(sm), None, false).unwrap();
        let v = volume_of(&mesh);
        assert!((v - (27.0 - 1.0)).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn meshes_a_concave_solid() {
        // An L-shaped prism: the reflex edge is what a convex-hull-based
        // mesher gets wrong, so the volume check is the real assertion here.
        let coords = Handle::new(Coords::new(3).unwrap());
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
        let mesh = triangulate_volume(&surface(&coords, &n, &facets), None, true).unwrap();
        let v = volume_of(&mesh);
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

    /// The envelope of a plate meshed by `triangulate_surface`, extruded and
    /// peeled — the pipeline `formation/maillage_test.py` uses.
    ///
    /// `n` segments per side of a unit square, `size` the target edge length.
    /// Note the `invert`: `extrude` does not check that its direction lies on
    /// the same side as the source surface's normal, so `skin` of the result
    /// comes back with its normals pointing into the material.
    fn extruded_plate(coords: &Handle<Coords>, n: usize, size: f64) -> Mesh {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for i in 0..n {
            pts.push([i as f64 / n as f64, 0.0, 0.0]);
        }
        for i in 0..n {
            pts.push([1.0, 0.0, i as f64 / n as f64]);
        }
        for i in 0..n {
            pts.push([1.0 - i as f64 / n as f64, 0.0, 1.0]);
        }
        for i in 0..n {
            pts.push([0.0, 0.0, 1.0 - i as f64 / n as f64]);
        }
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        for i in 0..ids.len() {
            sm.add_cell(&[ids[i], ids[(i + 1) % ids.len()]]).unwrap();
        }
        let contour = Mesh::from_submesh(sm);

        let plate =
            super::super::triangulate_surface(&contour, ElementType::TRI3, Some(size)).unwrap();
        let solid = super::super::extrude(&plate, &[0.0, 0.4, 0.0], 1).unwrap();
        let skin = super::super::skin(&solid, None).unwrap();
        let skin = super::super::convert(&skin, ElementType::TRI3).unwrap();
        super::super::invert(&skin).unwrap()
    }

    #[test]
    fn meshes_a_realistic_closed_surface() {
        // A plate meshed, extruded and peeled, at a size that gives a few
        // thousand cells: the shape of input the mesher is actually for, as
        // opposed to the eight-corner puzzles above.
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 8, 0.15);
        let mesh = triangulate_volume(&envelope, None, true).unwrap();

        // Nodes had to be added on the envelope here, so the result carries a
        // second submesh naming them.
        assert_eq!(
            mesh.element_types().unwrap(),
            vec![ElementType::TET4, ElementType::POI1]
        );
        assert!(tet_count(&mesh) > 2000, "{}", tet_count(&mesh));
        let v = volume_of(&mesh);
        assert!((v - 0.4).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn the_nodes_it_adds_on_the_envelope_come_back_as_a_poi1_submesh() {
        // A warning on stderr is not something a script can act on. When the
        // envelope had to be cut, the points that did the cutting are handed
        // back as part of the result, so the caller can see exactly where its
        // surface stopped matching.
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 5, 0.25);
        let given: Vec<Vec<f64>> = {
            let c = coords.read();
            c.iter_live()
                .map(|id| c.position(id).unwrap().to_vec())
                .collect()
        };

        let mesh = triangulate_volume(&envelope, None, true).unwrap();
        let types = mesh.element_types().unwrap();
        assert_eq!(types, vec![ElementType::TET4, ElementType::POI1]);

        // One node per cell, none of them the caller's own, and every one of
        // them used by the volume — a marker for a point that is not in the
        // mesh would be worse than no marker at all.
        let marks = mesh.cell_counts().unwrap()[1];
        assert!(marks > 0, "the submesh is there, so it must name something");
        for ci in 0..marks {
            let p = mesh.node(1, ci, 0).unwrap().position().unwrap();
            assert!(
                !given.iter().any(|q| q == &p),
                "{p:?} was the caller's node all along"
            );
            let id = mesh.node(1, ci, 0).unwrap().id();
            let used = (0..tet_count(&mesh))
                .any(|t| (0..4).any(|k| mesh.node(0, t, k).unwrap().id() == id));
            assert!(used, "{p:?} is named but no cell uses it");
        }
    }

    #[test]
    fn a_mesh_that_needed_no_extra_node_has_no_poi1_submesh() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mesh = triangulate_volume(&surface(&coords, &nodes, &BOX_FACETS), None, true).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TET4]);
    }

    #[test]
    fn meshes_the_envelope_of_an_extruded_plate() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 2, 0.5);
        let mesh = triangulate_volume(&envelope, None, false).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TET4]);
        let v = volume_of(&mesh);
        assert!((v - 0.4).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn meshes_a_finer_extruded_plate_without_touching_its_skin() {
        // The side-wall quadrilaterals of an extrusion used to defeat
        // recovery outright, and the envelope had to be cut finer to get a
        // mesh at all. Retriangulating the pocket around a stuck facet
        // settles them, so the strict contract holds here: the envelope
        // comes back exactly as it went in.
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 3, 0.35);

        let mesh = triangulate_volume(&envelope, None, false).unwrap();
        let v = volume_of(&mesh);
        assert!((v - 0.4).abs() < 1e-12, "volume {v}");

        // Nodes were added — refinement fills the inside — but none of them
        // on the envelope, which is what the strict contract is about.
        let peeled = super::super::skin(&mesh, None).unwrap();
        assert_eq!(
            peeled.cell_count().unwrap(),
            envelope.cell_count().unwrap(),
            "the skin is the envelope, facet for facet"
        );
    }

    #[test]
    fn every_node_it_puts_on_the_skin_lies_on_the_envelope() {
        // The promise of the permissive mode is that the *shape* survives: a
        // subdivision point sits on the edge it divides, so the surface is
        // the same surface, cut finer. Interior nodes are another matter and
        // are not in question here, so the check is made on the skin of the
        // result — the only place a node could have moved the boundary.
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 5, 0.25);
        let given: Vec<Vec<f64>> = {
            let c = coords.read();
            c.iter_live()
                .map(|id| c.position(id).unwrap().to_vec())
                .collect()
        };

        let mesh = triangulate_volume(&envelope, None, true).unwrap();
        assert!((volume_of(&mesh) - 0.4).abs() < 1e-12);

        let dist = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };
        let peeled = super::super::skin(&mesh, None).unwrap();
        let mut checked = 0;
        for (si, &n) in peeled.cell_counts().unwrap().iter().enumerate() {
            for ci in 0..n {
                for k in 0..3 {
                    let p = peeled.node(si, ci, k).unwrap().position().unwrap();
                    if given.iter().any(|q| q == &p) {
                        continue; // one of the caller's own nodes
                    }
                    // Anything else on the skin has to sit on a segment
                    // between two of them.
                    let on_a_segment = given.iter().enumerate().any(|(i, a)| {
                        given[i + 1..].iter().any(|b| {
                            let len = dist(a, b);
                            len > 0.0 && (dist(a, &p) + dist(&p, b) - len).abs() < 1e-12 * len
                        })
                    });
                    assert!(on_a_segment, "skin node {p:?} is off the envelope");
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "nothing was added, so nothing was checked");
    }

    /// The smallest dihedral angle of a cell, in degrees, and its volume.
    ///
    /// The angle across an edge is read by flattening the two opposite
    /// corners into the plane perpendicular to it.
    fn shape_of(p: &[Vec<f64>; 4]) -> (f64, f64) {
        const EDGES: [(usize, usize, usize, usize); 6] = [
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (0, 3, 1, 2),
            (1, 2, 0, 3),
            (1, 3, 0, 2),
            (2, 3, 0, 1),
        ];
        let sub = |a: &[f64], b: &[f64]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let volume = dot(
            sub(&p[1], &p[0]),
            cross(sub(&p[2], &p[0]), sub(&p[3], &p[0])),
        )
        .abs()
            / 6.0;

        let mut worst = f64::INFINITY;
        for (a, b, c, d) in EDGES {
            let e = sub(&p[b], &p[a]);
            let len = dot(e, e).sqrt();
            let unit = [e[0] / len, e[1] / len, e[2] / len];
            let flatten = |x: &[f64]| {
                let w = sub(x, &p[a]);
                let along = dot(w, unit);
                [
                    w[0] - along * unit[0],
                    w[1] - along * unit[1],
                    w[2] - along * unit[2],
                ]
            };
            let (u, w) = (flatten(&p[c]), flatten(&p[d]));
            let cos = (dot(u, w) / (dot(u, u).sqrt() * dot(w, w).sqrt())).clamp(-1.0, 1.0);
            worst = worst.min(cos.acos().to_degrees());
        }
        (worst, volume)
    }

    #[test]
    fn the_cells_are_shaped_well_enough_to_compute_with() {
        // Valid is not the same as usable. Without interior nodes the mesh
        // carries slivers — cells of four nearly coplanar corners, whose
        // element matrix is close to singular — and a handful of them is
        // enough to sink a computation. This is what refinement and the
        // sliver pass are for, so this is what has to be measured.
        let coords = Handle::new(Coords::new(3).unwrap());
        let envelope = extruded_plate(&coords, 6, 0.2);
        let mesh = triangulate_volume(&envelope, None, true).unwrap();

        let n = tet_count(&mesh);
        let mut angles = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        for ci in 0..n {
            let p: [Vec<f64>; 4] =
                std::array::from_fn(|k| mesh.node(0, ci, k).unwrap().position().unwrap());
            let (angle, volume) = shape_of(&p);
            angles.push(angle);
            volumes.push(volume);
        }

        // The bulk has to be well shaped: a regular tetrahedron measures
        // 70.5°, so a median near 45° is a healthy mesh.
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = angles[n / 2];
        assert!(median > 30.0, "median dihedral angle is only {median:.1}°");

        // And the tail has to stay a tail.
        let pinched = angles.iter().filter(|&&a| a < 10.0).count();
        assert!(
            100 * pinched < 6 * n,
            "{pinched} of {n} cells are under 10°"
        );

        // Nothing may be *exactly* flat. A few slivers pinned to the surface
        // do survive — their four corners are the caller's own nodes, so
        // neither an insertion nor a move can reach them — but a cell of no
        // volume at all would mean something upstream went wrong.
        let mean: f64 = volumes.iter().sum::<f64>() / n as f64;
        let flattest = volumes.iter().cloned().fold(f64::INFINITY, f64::min) / mean;
        assert!(
            flattest > 1e-12,
            "flattest cell is {flattest:.2e} of the mean"
        );
    }

    #[test]
    fn a_flat_cell_nothing_can_reach_is_named_rather_than_returned() {
        // No envelope in this suite produces one any more — taking a bad
        // cell's own edges out reaches what used to be hopeless — so the
        // refusal is exercised on its own. What matters about it is not that
        // it happens but what it says: the caller can only act on *where* and
        // *what to do*.
        let coords = Handle::new(Coords::new(3).unwrap());
        let (envelope, _) = subdivided_blob(&coords, 3);
        let env = Envelope::extract(&envelope, &NoCancel).unwrap();
        let mesh = TetMesh::delaunay(env.points(), &NoCancel).unwrap();

        // A healthy mesh has nothing to report, which is what makes the
        // report worth reading.
        let inside = vec![true; mesh.slot_count()];
        let (_, v) = mesh.iter().next().expect("a non-empty triangulation");
        let msg = degenerate(&mesh, &[(v, 1e-9)]).to_string();
        assert!(msg.contains("flat cell"), "{msg}");
        assert!(msg.contains("Refine the surface mesh"), "{msg}");
        assert!(msg.contains("around ("), "{msg}");

        // And the floor is calibrated, not nominal: the cells of a sound
        // triangulation are nowhere near it.
        let flagged = smooth::stubborn(&mesh, &inside, env.points().len(), DEGENERATE_SHAPE);
        assert!(
            flagged.is_empty(),
            "{} sound cell(s) were called degenerate",
            flagged.len()
        );
    }

    #[test]
    fn the_blob_that_used_to_carry_a_hopeless_cell_now_meshes_strictly() {
        // It used to be refused: one cell of it was flat and beyond every
        // move the sliver pass had. Taking the cell's own edges out reaches
        // it, so the envelope goes through as given.
        let coords = Handle::new(Coords::new(3).unwrap());
        let (envelope, volume) = subdivided_blob(&coords, 3);
        let mesh = triangulate_volume(&envelope, None, false).unwrap();
        let v = volume_of(&mesh);
        assert!((v - volume).abs() < 1e-9 * volume, "volume {v} vs {volume}");
    }

    #[test]
    fn rejects_a_bad_size() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let err = triangulate_volume(&surface(&coords, &nodes, &BOX_FACETS), Some(0.0), false)
            .unwrap_err();
        assert!(err.to_string().contains("size must be > 0"), "{err}");
    }

    #[test]
    fn stops_on_a_preset_cancellation_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let flag = AtomicBool::new(true);
        let err = triangulate_volume_cancellable(
            &surface(&coords, &nodes, &BOX_FACETS),
            None,
            false,
            &flag,
        )
        .unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }
}
