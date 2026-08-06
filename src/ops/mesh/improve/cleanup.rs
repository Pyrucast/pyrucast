//! Topological cleanup of an existing surface mesh: changing who is next to
//! whom, and never where a node sits.

use super::Surface;
use crate::containers::mesh::Mesh;
use crate::error::Result;
use crate::ops::mesh::paving::cleanup as pass;

/// Fix the connectivity of `mesh`: remove its doublets and switch the
/// diagonals that lower the valence error. No node moves.
///
/// Two defects are worth naming, and neither is geometry:
///
/// - a **doublet** — an interior node with only two quadrangles around it,
///   which therefore share *two* edges. The node sits in a wedge and both
///   cells stay pinched however they are smoothed; merging the two into one
///   removes node and wedge together.
/// - a node with the **wrong valence**. An interior node wants four cells: with
///   three the corners average 120°, with five 72°, and no smoothing will
///   change that, because the angles around a node sum to 2π whatever the
///   positions.
///
/// The only move is the diagonal switch: two quadrangles sharing an edge form
/// a hexagon, which splits across any of its three diagonals, and switching
/// moves one unit of valence from the two nodes on the old diagonal to the two
/// on the new. It changes no node and no boundary, so it cannot make the mesh
/// non-conforming, and it is applied only when it strictly lowers the total
/// valence error and leaves both cells convex.
///
/// Triangles are read for incidence and never touched — removing them is
/// [`merge_triangles`](fn@super::merge_triangles::merge_triangles)'s job, and
/// it is a different problem: their number has the parity of the boundary's
/// edge count, so they only ever go in pairs.
///
/// The result is a fresh mesh over the caller's own nodes: the connectivity
/// changed, the geometry did not.
pub fn cleanup(mesh: &Mesh) -> Result<Mesh> {
    let mut surf = Surface::read(mesh, "cleanup")?;
    pass::run(&surf.pts, &surf.movable, &mut surf.quads, &surf.tris);
    surf.to_mesh_same_nodes("cleanup")
}
