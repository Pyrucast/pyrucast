//! Putting nodes inside the solid, so the mesh becomes usable.
//!
//! Up to this point the mesh uses the envelope's nodes and nothing else. That
//! is enough to be *valid* — every cell has a positive volume and the
//! boundary is exactly the surface handed in — and nowhere near enough to be
//! *good*. A tetrahedralization of a surface's own nodes always carries
//! cells whose four corners are nearly coplanar, and a handful of those is
//! enough to make a finite-element matrix singular. No care taken over the
//! boundary recovery removes them, because they are not its doing: there is
//! simply nowhere else for the nodes to be.
//!
//! The cure is Delaunay refinement. Repeatedly, take the worst-shaped cell
//! and put a node at the centre of its circumsphere:
//!
//! ```text
//!     a badly shaped cell        its circumcentre is, by construction,
//!     ────────────────────       further from every one of its corners
//!      ____________              than they are from each other
//!      \__________/       →      so the cells replacing it cannot be
//!        a sliver                as thin as the one that is gone
//! ```
//!
//! Two numbers decide what "worst" means:
//!
//! - the **radius-edge ratio** `ρ = R / ℓ`, the circumradius over the
//!   shortest edge. A regular tetrahedron has `ρ = √6/4 ≈ 0.61`; a needle or
//!   a wedge has a large one. Splitting every cell above a threshold `B`
//!   terminates for any `B > 2`, which is why 2 is where the threshold sits.
//! - the **size**, `R` against the target edge length. This is what a caller
//!   asking for a mesh of a given fineness is asking for.
//!
//! What refinement provably cannot fix is the **sliver**: four corners close
//! to one plane and spread evenly round a circle. Its shortest edge is
//! respectable and its circumradius is small, so `ρ` says nothing is wrong,
//! while its volume is almost zero. Slivers are the business of the pass
//! that follows this one.
//!
//! Every insertion stays inside the solid: the cavity that opens around the
//! new node stops at the envelope, so the surface is never touched and the
//! nodes the caller supplied are never moved.

use std::collections::HashSet;

use crate::error::Result;
use crate::interrupt::Cancel;

use super::delaunay::TetMesh;

/// Radius-edge ratio above which a cell is split.
///
/// Delaunay refinement is only guaranteed to terminate above 2; sitting just
/// over it takes as much as the guarantee allows.
const RADIUS_EDGE_MAX: f64 = 2.05;

/// A regular tetrahedron's circumradius, as a multiple of its edge — the
/// conversion between "target edge length" and "target circumradius".
const REGULAR_RADIUS: f64 = 0.612_372_435_695_794_5; // √6 / 4

/// Nodes inserted per starting cell before refinement is called off.
///
/// Termination is a theorem, but a theorem about exact arithmetic and a
/// well-shaped input; the cap keeps a pathological envelope from running
/// away, and is generous enough that a sound one never reaches it.
const BUDGET_PER_CELL: usize = 12;

/// Fill the inside with nodes until no cell is badly shaped or too large.
///
/// `inside` says which slots hold material, and is kept up to date as cells
/// come and go. `walls` are the envelope's facets, which the refinement
/// never crosses and never touches. `target` is the desired edge length.
///
/// Returns how many nodes were inserted.
pub fn refine(
    mesh: &mut TetMesh,
    inside: &mut Vec<bool>,
    walls: &HashSet<[u32; 3]>,
    target: f64,
    cancel: &dyn Cancel,
) -> Result<usize> {
    let ceiling = target * REGULAR_RADIUS;
    let budget = BUDGET_PER_CELL * mesh.len().max(1);
    let mut inserted = 0;
    let mut refused: HashSet<[u32; 4]> = HashSet::new();

    // A queue, not a sweep. Splitting a cell only reshapes its own
    // neighbourhood, so re-examining the whole mesh after every insertion
    // spends a full pass to learn almost nothing — and turns a linear job
    // quadratic. Only the cells just created can be newly bad, so only they
    // go back in.
    let mut queue: Vec<u32> = (0..mesh.slot_count() as u32)
        .filter(|&t| inside.get(t as usize).copied().unwrap_or(false))
        .collect();
    let mut head = 0usize;

    while head < queue.len() && inserted < budget {
        cancel.check()?;
        let t = queue[head] as usize;
        head += 1;

        // The slot may have been recycled since it was queued, so everything
        // is checked again here rather than trusted.
        let Some(v) = mesh.tet(t) else { continue };
        if !inside.get(t).copied().unwrap_or(false) || refused.contains(&key(&v)) {
            continue;
        }
        if !is_bad(mesh, &v, ceiling) {
            continue;
        }

        let Some(centre) = circumcentre(mesh, &v) else {
            refused.insert(key(&v));
            continue;
        };
        // A node landing on top of an existing one buys nothing and costs
        // conditioning, so it is refused rather than placed.
        if too_close(mesh, &v, &centre, target) {
            refused.insert(key(&v));
            continue;
        }

        // A node inside a boundary facet's own sphere would make the cells
        // against that facet worse, not better, and asking for it again and
        // again is how refinement runs away instead of stopping. The textbook
        // answer is to split the facet instead; the envelope is not ours to
        // split, so the cell is simply left as it is.
        let clear = |f: &[[f64; 3]; 3]| !encroaches(&centre, f);
        match mesh.insert_within(centre, inside, walls, &clear)? {
            Some(created) => {
                inside.resize(mesh.slot_count(), false);
                for c in &created {
                    inside[*c as usize] = true;
                }
                queue.extend(created);
                inserted += 1;
            }
            // Outside the solid, or the cavity would not close: nothing to
            // be done about this cell here.
            None => {
                refused.insert(key(&v));
            }
        }

        // The queue only ever grows; drop what is behind now and then, so a
        // long run does not hold every slot it has ever looked at.
        if head > 1 << 16 && head * 2 > queue.len() {
            queue.drain(..head);
            head = 0;
        }
    }
    Ok(inserted)
}

/// Whether a cell is too badly shaped, or simply too big.
fn is_bad(mesh: &TetMesh, v: &[u32; 4], ceiling: f64) -> bool {
    let Some(r) = circumradius(mesh, v) else {
        return false;
    };
    let shortest = shortest_edge(mesh, v);
    if shortest <= 0.0 {
        return false;
    }
    r / shortest > RADIUS_EDGE_MAX || (ceiling > 0.0 && r > ceiling)
}

fn key(v: &[u32; 4]) -> [u32; 4] {
    let mut k = *v;
    k.sort_unstable();
    k
}

fn shortest_edge(mesh: &TetMesh, v: &[u32; 4]) -> f64 {
    let p = mesh.points();
    let mut best = f64::INFINITY;
    for a in 0..4 {
        for b in a + 1..4 {
            best = best.min(distance(&p[v[a] as usize], &p[v[b] as usize]));
        }
    }
    best
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Whether the candidate sits on top of a corner of the cell it would split,
/// or so close to one that it would only add noise.
///
/// The floor is taken against the target as well as against the cell,
/// because a floor that shrinks with the cells it measures never stops
/// anything: each split makes the next one look acceptable.
fn too_close(mesh: &TetMesh, v: &[u32; 4], c: &[f64; 3], target: f64) -> bool {
    let floor = (0.25 * shortest_edge(mesh, v)).max(0.2 * target);
    v.iter()
        .any(|&i| distance(&mesh.points()[i as usize], c) < floor)
}

/// Whether `p` lies inside the sphere through the corners of facet `f`.
///
/// This is the classical encroachment test. A point inside that sphere is
/// nearer the facet than the facet's own corners are to each other, and
/// filling there produces cells flatter than the ones it was meant to fix.
fn encroaches(p: &[f64; 3], f: &[[f64; 3]; 3]) -> bool {
    let (a, b, c) = (f[0], f[1], f[2]);
    // The centre lies in the facet's plane, equidistant from its corners.
    let (u, v) = (
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
    );
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if nn == 0.0 {
        return false;
    }
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let cross = |x: [f64; 3], y: [f64; 3]| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let (p1, p2) = (cross(n, u), cross(v, n));
    let centre = [
        a[0] + (vv * p1[0] + uu * p2[0]) / (2.0 * nn),
        a[1] + (vv * p1[1] + uu * p2[1]) / (2.0 * nn),
        a[2] + (vv * p1[2] + uu * p2[2]) / (2.0 * nn),
    ];
    distance(&centre, p) < distance(&centre, &a)
}

fn circumradius(mesh: &TetMesh, v: &[u32; 4]) -> Option<f64> {
    let c = circumcentre(mesh, v)?;
    Some(distance(&mesh.points()[v[0] as usize], &c))
}

/// Centre of the sphere through the four corners.
///
/// It solves `2 (pᵢ − p₀) · x = |pᵢ|² − |p₀|²` for `i = 1, 2, 3`: each row
/// says the centre is equidistant from `p₀` and `pᵢ`.
fn circumcentre(mesh: &TetMesh, v: &[u32; 4]) -> Option<[f64; 3]> {
    let p = mesh.points();
    let (a, rest) = (&p[v[0] as usize], [v[1], v[2], v[3]]);
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for (i, &j) in rest.iter().enumerate() {
        let q = &p[j as usize];
        for k in 0..3 {
            m[i][k] = 2.0 * (q[k] - a[k]);
            rhs[i] += q[k] * q[k] - a[k] * a[k];
        }
    }
    let base = det3(&m);
    if base == 0.0 || !base.is_finite() {
        return None;
    }
    let mut c = [0.0f64; 3];
    for col in 0..3 {
        let mut mc = m;
        for row in 0..3 {
            mc[row][col] = rhs[row];
        }
        c[col] = det3(&mc) / base;
    }
    c.iter().all(|x| x.is_finite()).then_some(c)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
