//! Taking a subdivision point back out of the mesh.
//!
//! When the envelope had to be cut finer to be fitted, the points that did
//! the cutting are not wanted for their own sake — they were the price of
//! getting a mesh at all. Once there *is* a mesh, each of them can be tried
//! for removal, and the question is a far kinder one than the one recovery
//! faced: not "find me an arrangement that has this facet", but "rearrange
//! this neighbourhood without that vertex", starting from something already
//! valid and correct.
//!
//! Removing a vertex is the same operation as any other rebuild — empty its
//! **star**, the cells that touch it, and fill the hole again — with one
//! detail doing all the work: [`fill`] only ever uses vertices that appear on
//! the region's boundary. Take the vertex off that boundary and it cannot
//! come back, without anyone having to forbid it.
//!
//! For a point in the middle of an envelope edge, taking it off the boundary
//! means putting the two halves of each cut facet back together:
//!
//! ```text
//!        w                       w
//!       /|\                     / \
//!      / | \        →          /   \        the pair (u,m,w), (m,v,w)
//!     u--m--v                 u-----v       becomes (u,v,w) again
//! ```
//!
//! The star of a vertex is usually too big for an exhaustive fill, so it is
//! made to shrink first: every interior edge running to the vertex that can
//! be taken out removes a cell from the star, and a star of a few cells is
//! well within reach. When it will not shrink far enough, the point stays —
//! a mesh with a few more nodes than asked for, which is what the caller
//! agreed to when it allowed the subdivision.
//!
//! Widening the star instead of shrinking it — swallowing the cell beyond
//! whichever face the filler stumbled on — is the natural other move, and it
//! is deliberately not made here. A wider pocket rebuilt under permission to
//! re-cut the star's surface reshapes the mesh's outer skin over cells that
//! have nothing to do with the vertex being removed; measured, it lost more
//! subdivision points than it reclaimed. Widening belongs where the pocket's
//! surface is held fixed, which is boundary recovery.

use crate::error::Result;

use super::delaunay::{Boundary, TetMesh};
use super::fill::{delaunay_fill, fill, Constraints, DEFAULT_BUDGET, EXHAUSTIVE_LIMIT};
use super::flips::{relevant, remove_edge};
use super::recovery::Protected;

/// Cells a star may hold for the rebuild to be attempted.
///
/// Well past what the exhaustive filler can take: beyond that size the
/// Delaunay filler carries the attempt, and it does not care how big the
/// star is.
const STAR_LIMIT: usize = 40;

/// Attempts at shrinking one star before the vertex is left alone.
const SHRINK_BUDGET: usize = 64;

/// Largest pocket an edge removal may grow to while thinning a star.
const THINNING_REGION: usize = 8;

/// Try to take vertex `m` out of `mesh`, putting `merged` where the faces
/// around it used to be.
///
/// `merged` are the whole pieces that take the place of the halves around `m`;
/// `protect` is the rest of the envelope, which the rebuild must leave alone.
///
/// Returns whether the vertex is gone. A refusal leaves the mesh exactly as
/// it was.
pub fn remove_vertex(
    mesh: &mut TetMesh,
    m: u32,
    merged: &[[u32; 3]],
    protect: &Protected<'_>,
) -> Result<bool> {
    for _ in 0..SHRINK_BUDGET {
        let star = mesh.tets_around_vertex(m);
        if star.is_empty() {
            return Ok(false);
        }
        if star.len() <= STAR_LIMIT && rebuild_star(mesh, m, &star, merged, protect)? {
            return Ok(true);
        }
        if !thin_star(mesh, m, &star, protect)? {
            return Ok(false);
        }
    }
    Ok(false)
}

/// Empty the star and fill it again, this time without `m`.
///
/// Where the merged faces go depends on where the halves were. On the outer
/// surface of the mesh they are part of the star's own boundary, and the
/// merged face takes their place there. Inside the mesh — which is where an
/// envelope facet of a concave solid sits while the volume is being built —
/// they are buried in the star, and the merged face is asked for instead.
fn rebuild_star(
    mesh: &mut TetMesh,
    m: u32,
    star: &[u32],
    merged: &[[u32; 3]],
    protect: &Protected<'_>,
) -> Result<bool> {
    let mut boundary = mesh.region_boundary(star);
    let surfaced: Vec<[u32; 3]> = boundary
        .iter()
        .filter(|f| f.contains(&m))
        .copied()
        .collect();
    boundary.retain(|f| !f.contains(&m));

    // The rest of the envelope inside the star must survive — but not the
    // halves around `m`, which are precisely what is being undone. Requiring
    // those would make the vertex impossible to remove, since every one of
    // them holds it.
    let mut required: Vec<[u32; 3]> = relevant(mesh, star, protect)
        .into_iter()
        .filter(|f| !f.contains(&m))
        .collect();
    let mut mode = Boundary::Preserved;
    for w in merged {
        let halves: Vec<&[u32; 3]> = surfaced
            .iter()
            .filter(|h| w.iter().filter(|x| h.contains(x)).count() == 2)
            .collect();
        if halves.is_empty() {
            required.push(*w);
        } else {
            boundary.push(facing_like(w, &halves));
            // The star's own surface changes shape, by the hair between the
            // midpoint and the segment it was meant to sit on.
            mode = Boundary::MayRecutHull;
        }
    }

    // The Delaunay filler first: it costs one triangulation whatever the
    // pocket's size, and is what makes a star of twenty cells worth
    // attempting at all.
    let want = Constraints {
        with_faces: &required,
        ..Default::default()
    };
    // The exhaustive filler is kept for the small pockets it can actually
    // settle: it is the only one able to prove there is no filling at all,
    // and it is not limited to Delaunay ones.
    let Ok(cells) = delaunay_fill(mesh.points(), &boundary, want).or_else(|missing| {
        (boundary.len() <= EXHAUSTIVE_LIMIT)
            .then(|| fill(mesh.points(), &boundary, want, DEFAULT_BUDGET))
            .flatten()
            .ok_or(missing)
    }) else {
        return Ok(false);
    };
    if cells.iter().any(|c| c.contains(&m)) {
        return Ok(false); // `m` is off the boundary, so this cannot happen
    }

    // A refusal costs nothing: the swap decides whether the cells tile the
    // star before it touches the mesh.
    match mesh.replace_region_with(star, &cells, "a vertex removal", mode) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Take one interior edge running to `m` out, so its star loses a cell.
///
/// Only edges away from the surface are eligible: removing one that lies on
/// the envelope would change the surface, which is exactly what this whole
/// exercise is trying to undo.
fn thin_star(mesh: &mut TetMesh, m: u32, star: &[u32], protect: &Protected<'_>) -> Result<bool> {
    let mut neighbours: Vec<u32> = star
        .iter()
        .filter_map(|&t| mesh.tet(t as usize))
        .flatten()
        .filter(|&x| x != m)
        .collect();
    neighbours.sort_unstable();
    neighbours.dedup();

    for x in neighbours {
        if !protect.on_edge(m, x).is_empty() {
            continue; // an edge of the envelope: not ours to remove
        }
        if remove_edge(mesh, m, x, protect, THINNING_REGION)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `f`, wound the same way as the halves it replaces.
///
/// The merged face covers the same ground as they did, so it must face the
/// same way; a directed edge shared with one of them settles it.
fn facing_like(f: &[u32; 3], like: &[&[u32; 3]]) -> [u32; 3] {
    for half in like {
        // A directed edge shared with the half means the same winding.
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            for j in 0..3 {
                if half[j] == a && half[(j + 1) % 3] == b {
                    return *f;
                }
                if half[j] == b && half[(j + 1) % 3] == a {
                    return [f[0], f[2], f[1]];
                }
            }
        }
    }
    *f
}
