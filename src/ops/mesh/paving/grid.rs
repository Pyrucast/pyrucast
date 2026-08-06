//! The structured core: a tensor grid laid inside the domain, away from its
//! boundary.
//!
//! This is the half of [`grid_surface`](fn@crate::ops::mesh::grid_surface) that
//! the advancing front cannot do. A front is perfect where it starts — its
//! first row follows the contour exactly — and doubtful where two of its rows
//! meet, because there it has to reconcile two discretisations that have no
//! reason to agree. A grid is the mirror image: every cell it lays is a
//! rectangle by construction, and all its difficulty is at the boundary. Taking
//! the interior from the grid and the boundary from the front gives a mesh with
//! neither weakness.
//!
//! ## The grid is laid in the contour's own direction
//!
//! The orientation is a free internal choice, so [`preferred_angle`] takes it
//! from the contour and [`build`] works in that frame throughout, turning back
//! only at the two points that touch the [`Fabric`]. Without it the whole
//! method would only pay off on shapes someone happened to draw square-on.
//!
//! ## The grid is snapped to the contour, not to the bounding box
//!
//! Subdividing the bounding box uniformly is the obvious thing and it is
//! wrong. A step at `x = 0.0667` on a domain 0.6 wide asked for cells of
//! 0.00375 would fall between columns 17 and 18 of a uniform grid, and every
//! cell along that step would be cut — the core would stop short of the whole
//! feature and the band would have to fill it with rows.
//!
//! So the grid lines are chosen from the contour itself: every axis-aligned
//! edge long enough to matter contributes its coordinate as a line, and the
//! gaps between consecutive lines are subdivided uniformly at about the target
//! size. On a rectilinear contour every corner then lands on a grid node and
//! the core covers the domain up to the band. On a contour with no direction
//! at all — a circle — nothing is contributed and the result degrades
//! gracefully to the uniform grid over the bounding box.
//!
//! ## What comes out
//!
//! [`build`] writes the core's quadrangles straight into the [`Fabric`] and
//! returns the loops of what is left to fill — the contour and the core's
//! boundary minus the edges they share, since a segment walked once each way
//! bounds nothing. Each loop is flagged with whether it is entirely the core's,
//! which is what the caller needs to decide whether it may be frozen. On a
//! rectilinear domain on the grid, no loop comes back at all.

use super::Fabric;
use crate::atoms::Point2;
use crate::ops::mesh::contour::{point_in_polygon, Domain};
use std::collections::{HashMap, HashSet};

/// An axis-aligned edge shorter than this many target sizes does not get to
/// place a grid line: it is a corner cut or a discretisation artefact, not a
/// feature of the shape.
const FEATURE_EDGE: f64 = 0.999;

/// Two snap lines closer than this many target sizes are the same line. Below
/// it a column of cells could not hold a sane aspect ratio anyway.
const SNAP_TOLERANCE: f64 = 0.25;

/// Beyond this many lines per axis the contour is not rectilinear in any
/// useful sense — it is a curve whose every edge is axis-aligned by accident.
/// Snapping then costs more than it buys, and the uniform grid is used.
const MAX_SNAP_LINES: usize = 4096;

/// Below this share of the perimeter running one way, the contour has no
/// dominant direction at all — a circle — and turning the grid would only
/// trade one arbitrary orientation for another.
const DOMINANT_SHARE: f64 = 0.2;

/// A candidate has to beat the axes by this share of the perimeter to be
/// preferred to them. A shape already square with the frame must never lose
/// that to rounding.
const TURN_MARGIN: f64 = 0.02;

/// Half-width of the window, in radians, inside which an edge counts as
/// aligned. About a degree: past that an edge of a few cells' length no longer
/// pins a grid line under [`Grid::over`]'s own test.
const ALIGN_WINDOW: f64 = 0.017;

/// The frame the grid is laid in: the contour's own preferred direction.
///
/// Nothing in the contract ties the grid to the frame's axes — the orientation
/// is a free internal choice, and taking it from the contour gives a shape
/// turned by 30° exactly what a square-on shape gets. See [`preferred_angle`].
#[derive(Clone, Copy)]
pub struct Frame2 {
    cos: f64,
    sin: f64,
}

impl Frame2 {
    /// The frame turned by `angle`. Zero is the identity and is short-circuited
    /// rather than multiplied through by one, so a contour already square with
    /// the axes comes out bit for bit as it went in.
    fn at(angle: f64) -> Frame2 {
        if angle == 0.0 {
            Frame2 { cos: 1.0, sin: 0.0 }
        } else {
            Frame2 {
                cos: angle.cos(),
                sin: angle.sin(),
            }
        }
    }

    fn identity(self) -> bool {
        self.sin == 0.0 && self.cos == 1.0
    }

    /// A point of the caller's plane, in the grid's frame.
    fn to_grid(self, p: Point2) -> Point2 {
        if self.identity() {
            return p;
        }
        Point2::new(
            self.cos * p.x + self.sin * p.y,
            -self.sin * p.x + self.cos * p.y,
        )
    }

    /// A point of the grid's frame, back in the caller's plane.
    fn to_local(self, p: Point2) -> Point2 {
        if self.identity() {
            return p;
        }
        Point2::new(
            self.cos * p.x - self.sin * p.y,
            self.sin * p.x + self.cos * p.y,
        )
    }
}

/// The direction the contour mostly runs in, in `[0, π/2)`.
///
/// A grid has a four-fold symmetry, so a quarter turn covers every distinct
/// orientation. Among those, the useful one is the angle that turns the
/// greatest **length** of contour into axis-aligned edges — because an
/// axis-aligned edge is exactly what pins a grid line in [`Grid::over`], and a
/// pinned line is what lets a grid node land on a contour node. The objective
/// is therefore the mesher's own criterion rather than a stand-in for it.
///
/// Two guards, and both earn their place:
///
/// - a contour with no dominant direction — a circle, whose edge angles are
///   spread evenly — gets `0.0`, since turning it would swap one arbitrary
///   orientation for another and cost the caller reproducibility;
/// - the axes win ties. A shape already square with the frame is the case this
///   whole function must not break, and it would be a poor trade to lose it to
///   a rounding-width improvement somewhere else.
pub fn preferred_angle(domain: &Domain) -> f64 {
    let quarter = std::f64::consts::FRAC_PI_2;
    let mut edges: Vec<(f64, f64)> = Vec::new();
    let mut perimeter = 0.0;
    for l in std::iter::once(&domain.outer).chain(&domain.holes) {
        let n = l.pts.len();
        for i in 0..n {
            let d = l.pts[(i + 1) % n] - l.pts[i];
            let len = d.norm();
            if len <= 0.0 {
                continue;
            }
            perimeter += len;
            edges.push((d.y.atan2(d.x).rem_euclid(quarter), len));
        }
    }
    if edges.is_empty() {
        return 0.0;
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // The aligned length at a candidate is the weight of the edges whose angle
    // falls within the window around it — a sliding window on a circle of
    // period π/2, so the ends wrap into each other.
    let aligned_at = |theta: f64| -> f64 {
        edges
            .iter()
            .filter(|(a, _)| {
                let d = (a - theta).rem_euclid(quarter);
                d <= ALIGN_WINDOW || d >= quarter - ALIGN_WINDOW
            })
            .map(|(_, w)| w)
            .sum()
    };

    let axes = aligned_at(0.0);
    let mut best = (axes, 0.0);
    for &(a, _) in &edges {
        let w = aligned_at(a);
        // Strictly better, so the lowest angle wins a tie and the answer does
        // not depend on the order the contour was built in.
        if w > best.0 {
            best = (w, a);
        }
    }
    if best.0 < DOMINANT_SHARE * perimeter || best.0 <= axes + TURN_MARGIN * perimeter {
        return 0.0;
    }
    best.1
}

/// A tensor grid: the coordinates of its lines along each axis.
pub struct Grid {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl Grid {
    /// Lay a grid over `domain` with cells of about `target`, its lines snapped
    /// to the contour's axis-aligned edges.
    pub fn over(domain: &Domain, target: f64) -> Grid {
        let pts = &domain.outer.pts;
        let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in pts {
            x0 = x0.min(p.x);
            x1 = x1.max(p.x);
            y0 = y0.min(p.y);
            y1 = y1.max(p.y);
        }

        // A vertical edge pins an x line, a horizontal one pins a y line. Both
        // ends of the domain's extent are lines whatever happens.
        let (mut sx, mut sy) = (vec![x0, x1], vec![y0, y1]);
        for l in std::iter::once(&domain.outer).chain(&domain.holes) {
            let n = l.pts.len();
            for i in 0..n {
                let (a, b) = (l.pts[i], l.pts[(i + 1) % n]);
                let (dx, dy) = ((b.x - a.x).abs(), (b.y - a.y).abs());
                if dx <= SNAP_TOLERANCE * target * 0.5 && dy >= FEATURE_EDGE * target {
                    sx.push(0.5 * (a.x + b.x));
                } else if dy <= SNAP_TOLERANCE * target * 0.5 && dx >= FEATURE_EDGE * target {
                    sy.push(0.5 * (a.y + b.y));
                }
            }
        }

        Grid {
            xs: fill(cluster(sx, SNAP_TOLERANCE * target), target),
            ys: fill(cluster(sy, SNAP_TOLERANCE * target), target),
        }
    }

    pub fn nx(&self) -> usize {
        self.xs.len() - 1
    }

    pub fn ny(&self) -> usize {
        self.ys.len() - 1
    }

    fn node(&self, i: usize, j: usize) -> Point2 {
        Point2::new(self.xs[i], self.ys[j])
    }

    fn centre(&self, i: usize, j: usize) -> Point2 {
        Point2::new(
            0.5 * (self.xs[i] + self.xs[i + 1]),
            0.5 * (self.ys[j] + self.ys[j + 1]),
        )
    }
}

/// Sort, then merge coordinates closer than `tol` into their mean.
fn cluster(mut v: Vec<f64>, tol: f64) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out: Vec<f64> = Vec::with_capacity(v.len());
    let (mut sum, mut count) = (0.0, 0usize);
    for x in v {
        if count > 0 && x - sum / count as f64 > tol {
            out.push(sum / count as f64);
            (sum, count) = (0.0, 0);
        }
        sum += x;
        count += 1;
    }
    if count > 0 {
        out.push(sum / count as f64);
    }
    out
}

/// Subdivide each gap between consecutive lines uniformly, as close to
/// `target` as a whole number of cells allows.
fn fill(lines: Vec<f64>, target: f64) -> Vec<f64> {
    if lines.len() < 2 {
        return lines;
    }
    if lines.len() > MAX_SNAP_LINES {
        // Not a rectilinear shape: keep the extent, drop the rest.
        let (a, b) = (lines[0], lines[lines.len() - 1]);
        let k = (((b - a) / target).round() as usize).max(1);
        return (0..=k).map(|i| a + (b - a) * i as f64 / k as f64).collect();
    }
    let mut out = vec![lines[0]];
    for w in lines.windows(2) {
        let (a, b) = (w[0], w[1]);
        let k = (((b - a) / target).round() as usize).max(1);
        for i in 1..=k {
            out.push(a + (b - a) * i as f64 / k as f64);
        }
    }
    out
}

/// The structured core, once written into the fabric.
pub struct Core {
    /// How many cells the core holds.
    pub cells: usize,
    /// The loops bounding what the front still has to fill, material on the
    /// left, each flagged `true` when every one of its edges came from the
    /// core — the loops that may be frozen.
    pub band: Vec<(Vec<u32>, bool)>,
}

/// Fill `domain` with grid cells and write them into `fab`.
///
/// `contour_loops` are the contour's vertices, already in `fab`, in front
/// order. A grid node landing on one of them **is** it: the core then meets
/// the contour instead of stopping a hair short of it, and there is simply no
/// band to pave there. That sharing is what makes a rectilinear domain come
/// out as the structured mesh drawn by hand.
///
/// `band` is extra clearance, in cells, for a contour the grid cannot meet —
/// a curve, or a rectilinear shape off the grid. It buys the front room to
/// work, at the cost of the cells it takes back from the core.
pub fn build(
    fab: &mut Fabric,
    domain: &Domain,
    contour_loops: &[Vec<u32>],
    target: f64,
    band: usize,
) -> Core {
    // ── Turn the domain into the grid's frame ─────────────────────────────
    // Everything below then works on a contour that is as square with the axes
    // as it can be made, which is the only thing the grid ever wanted. The
    // frame comes back out at the two points that touch the fabric, and
    // nowhere else — see `known` and the emission below.
    let frame = Frame2::at(preferred_angle(domain));
    let turned = Domain {
        outer: turn(&domain.outer, frame),
        holes: domain.holes.iter().map(|h| turn(h, frame)).collect(),
    };
    let domain = &turned;

    let grid = Grid::over(domain, target);
    let (nx, ny) = (grid.nx(), grid.ny());
    if nx == 0 || ny == 0 {
        return Core {
            cells: 0,
            band: contour_loops.iter().map(|l| (l.clone(), false)).collect(),
        };
    }

    // ── Which cells are entirely in the material ──────────────────────────
    // A cell is solid when the boundary misses it and its centre is inside.
    // Marking every cell a boundary edge's bounding box touches is
    // conservative — it can only shrink the core — and on a contour
    // discretised at about the target size it over-marks almost nothing.
    let mut cut = vec![false; nx * ny];
    for l in std::iter::once(&domain.outer).chain(&domain.holes) {
        let n = l.pts.len();
        for i in 0..n {
            let (a, b) = (l.pts[i], l.pts[(i + 1) % n]);
            for (ci, cj) in cells_touching(&grid, a, b) {
                cut[cj * nx + ci] = true;
            }
        }
    }
    let mut solid = vec![false; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            if cut[j * nx + i] {
                continue;
            }
            let c = grid.centre(i, j);
            solid[j * nx + i] = point_in_polygon(c, &domain.outer.pts)
                && !domain.holes.iter().any(|h| point_in_polygon(c, &h.pts));
        }
    }

    // ── Pull back only where the core misses the contour ──────────────────
    let known = ContourNodes::build(fab, contour_loops, target, frame);
    let mut keep = clear_of_unmet(&grid, &solid, nx, ny, &known, contour_loops, band);
    tidy(&mut keep, nx, ny);

    // ── Emit the cells, sharing the contour's nodes where they coincide ───
    // Only the nodes cells actually use are added, and each exactly once.
    let mut vert: HashMap<(usize, usize), u32> = HashMap::new();
    let mut cells = 0usize;
    for j in 0..ny {
        for i in 0..nx {
            if !keep[j * nx + i] {
                continue;
            }
            let mut q = [0u32; 4];
            for (k, (di, dj)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
                q[k] = *vert.entry((i + di, j + dj)).or_insert_with(|| {
                    let p = grid.node(i + di, j + dj);
                    // The contour's own node when there is one, else a fresh
                    // one — put back in the caller's plane, since the frame is
                    // the grid's business and nobody else's.
                    known
                        .at(p)
                        .unwrap_or_else(|| fab.add(frame.to_local(p), false))
                });
            }
            fab.push_quad(q);
            cells += 1;
        }
    }

    let core_loops = boundary_loops(&keep, nx, ny, &vert);
    Core {
        cells,
        band: band_loops(contour_loops, &core_loops),
    }
}

/// One loop, expressed in the grid's frame. The node ids are the caller's and
/// travel unchanged: only the geometry turns.
fn turn(l: &crate::ops::mesh::contour::Loop2D, frame: Frame2) -> crate::ops::mesh::contour::Loop2D {
    crate::ops::mesh::contour::Loop2D {
        node_ids: l.node_ids.clone(),
        pts: l.pts.iter().map(|&p| frame.to_grid(p)).collect(),
    }
}

/// The contour's vertices, indexed so a grid node can ask whether it is one.
struct ContourNodes {
    grid: super::proximity::PointGrid,
    pts: Vec<Point2>,
    ids: Vec<u32>,
    tol: f64,
}

impl ContourNodes {
    fn build(fab: &Fabric, loops: &[Vec<u32>], target: f64, frame: Frame2) -> ContourNodes {
        let ids: Vec<u32> = loops.iter().flatten().copied().collect();
        // The fabric holds the caller's plane; the grid asks its questions in
        // its own frame, so the contour is indexed there.
        let pts: Vec<Point2> = ids
            .iter()
            .map(|&v| frame.to_grid(fab.pts[v as usize]))
            .collect();
        ContourNodes {
            // A grid line snapped to a contour edge lands on that edge to
            // rounding; anything further off is a different node, and taking
            // it for the same one would tear the mesh.
            tol: SNAP_TOLERANCE * target * 0.01,
            grid: super::proximity::PointGrid::build(&pts, target),
            pts,
            ids,
        }
    }

    /// The contour vertex at `p`, if there is one.
    fn at(&self, p: Point2) -> Option<u32> {
        let k = self.grid.nearest_index_within(&self.pts, p, self.tol)?;
        Some(self.ids[k])
    }
}

/// What the front still has to fill: the contour and the core's boundary,
/// minus the edges they share.
///
/// Both sets are wound with the material on their left — the contour by
/// convention, the core's boundary because [`boundary_loops`] emits it
/// clockwise around the core. Where the core reaches the contour the same
/// segment appears in both, once each way; those two cancel, and the region
/// between them, which is nothing at all, goes with them. What survives chains
/// into the loops the front works on — and on a rectilinear domain on the
/// grid, nothing survives and there is no front to run.
fn band_loops(contour: &[Vec<u32>], core: &[Vec<u32>]) -> Vec<(Vec<u32>, bool)> {
    let mut edges: HashMap<(u32, u32), bool> = HashMap::new();
    for (l, from_core) in contour
        .iter()
        .map(|l| (l, false))
        .chain(core.iter().map(|l| (l, true)))
    {
        let n = l.len();
        for i in 0..n {
            let (a, b) = (l[i], l[(i + 1) % n]);
            if edges.remove(&(b, a)).is_none() {
                edges.insert((a, b), from_core);
            }
        }
    }

    // Chain what survived. Away from a point where the core touches the
    // contour, every vertex carries exactly one outgoing edge and the walk is
    // forced; sorting first keeps it deterministic where a vertex carries
    // several.
    let mut out_of: HashMap<u32, Vec<(u32, bool)>> = HashMap::new();
    let mut keys: Vec<(u32, u32)> = edges.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        out_of.entry(k.0).or_default().push((k.1, edges[&k]));
    }

    let mut out = Vec::new();
    let mut starts: Vec<u32> = out_of.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        while out_of.get(&s).is_some_and(|v| !v.is_empty()) {
            let mut ring = vec![s];
            let mut all_core = true;
            let mut cur = s;
            while let Some((nxt, from_core)) = out_of.get_mut(&cur).and_then(|v| v.pop()) {
                all_core &= from_core;
                if nxt == s {
                    break;
                }
                ring.push(nxt);
                cur = nxt;
            }
            if ring.len() >= 3 {
                out.push((ring, all_core));
            }
        }
    }
    out
}

/// The cells whose **interior** the segment `a → b` enters.
///
/// The open interior, and that is the whole point: a boundary edge lying
/// exactly along a grid line cuts neither of the two cells it separates, it
/// just runs between them. Testing bounding boxes instead would mark both, the
/// outermost ring of cells would be lost on every axis-aligned shape, and the
/// core would stop one cell short of a contour it could have met exactly.
fn cells_touching(grid: &Grid, a: Point2, b: Point2) -> Vec<(usize, usize)> {
    let i0 = column(&grid.xs, a.x.min(b.x));
    let i1 = column(&grid.xs, a.x.max(b.x));
    let j0 = column(&grid.ys, a.y.min(b.y));
    let j1 = column(&grid.ys, a.y.max(b.y));
    let mut out = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let eps = 1e-9 * (grid.xs[i + 1] - grid.xs[i]).min(grid.ys[j + 1] - grid.ys[j]);
            if enters(
                a,
                b,
                grid.xs[i] + eps,
                grid.xs[i + 1] - eps,
                grid.ys[j] + eps,
                grid.ys[j + 1] - eps,
            ) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Does the segment `a → b` meet the box `[x0, x1] × [y0, y1]`?
/// Liang–Barsky, kept to the clipping test itself.
fn enters(a: Point2, b: Point2, x0: f64, x1: f64, y0: f64, y1: f64) -> bool {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-dx, a.x - x0),
        (dx, x1 - a.x),
        (-dy, a.y - y0),
        (dy, y1 - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return false; // parallel to this side and outside it
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// The index of the cell column containing `v`, clamped to the grid.
fn column(lines: &[f64], v: f64) -> usize {
    match lines.binary_search_by(|x| x.partial_cmp(&v).unwrap()) {
        Ok(i) => i.min(lines.len() - 2),
        Err(0) => 0,
        Err(i) => (i - 1).min(lines.len() - 2),
    }
}

/// Drop the core cells standing against a boundary the grid does not reach.
///
/// The core stops where the cells stop being solid, and that edge is of two
/// kinds. Where it lies **on the contour** — both ends of the face are contour
/// nodes and the face is a contour segment — the core has met the boundary
/// exactly, shares its nodes with it, and there is nothing between the two:
/// that edge is finished, and eroding it would throw away a perfect row of
/// cells to hand the front work it has no reason to do.
///
/// Everywhere else the core merely stops *near* the boundary, at anything from
/// a whole cell down to nothing, and the front has to bridge the gap. A gap of
/// nothing is the one thing it cannot do — that is a sliver, and a sliver is
/// what turns a front inside out. So the cell behind such a face goes, which
/// buys the front a full cell to work in, and `band` more go with it when the
/// caller wants the interface pushed further from the boundary still.
///
/// This is why a shape that is rectilinear on one side and round on the other
/// gets the best of both: the straight sides keep their grid right up to the
/// contour, and only the round part pays for a band.
fn clear_of_unmet(
    grid: &Grid,
    solid: &[bool],
    nx: usize,
    ny: usize,
    known: &ContourNodes,
    contour_loops: &[Vec<u32>],
    band: usize,
) -> Vec<bool> {
    let mut on_contour: HashSet<(u32, u32)> = HashSet::new();
    for l in contour_loops {
        let n = l.len();
        for i in 0..n {
            let (a, b) = (l[i], l[(i + 1) % n]);
            on_contour.insert((a.min(b), a.max(b)));
        }
    }
    let met = |a: Point2, b: Point2| match (known.at(a), known.at(b)) {
        (Some(u), Some(v)) => on_contour.contains(&(u.min(v), u.max(v))),
        _ => false,
    };

    let at = |i: i64, j: i64| -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < nx
            && (j as usize) < ny
            && solid[j as usize * nx + i as usize]
    };
    let mut unmet = vec![false; nx * ny];
    for j in 0..ny as i64 {
        for i in 0..nx as i64 {
            if !at(i, j) {
                continue;
            }
            let (u, v) = (i as usize, j as usize);
            // The four faces, each with the two grid nodes bounding it.
            let faces = [
                ((1, 0), (u + 1, v), (u + 1, v + 1)),
                ((-1, 0), (u, v + 1), (u, v)),
                ((0, 1), (u + 1, v + 1), (u, v + 1)),
                ((0, -1), (u, v), (u + 1, v)),
            ];
            for ((di, dj), a, b) in faces {
                if at(i + di, j + dj) {
                    continue;
                }
                if !met(grid.node(a.0, a.1), grid.node(b.0, b.1)) {
                    unmet[v * nx + u] = true;
                    break;
                }
            }
            // A corner is a face's worth of trouble too: two cells meeting
            // only diagonally leave the band pinched to a point between them.
            if !unmet[v * nx + u] {
                for (di, dj) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    if !at(i + di, j + dj) && at(i + di, j) && at(i, j + dj) {
                        unmet[v * nx + u] = true;
                        break;
                    }
                }
            }
        }
    }

    let mut out = solid.to_vec();
    for j in 0..ny as i64 {
        for i in 0..nx as i64 {
            let r = band as i64;
            let near = (-r..=r).any(|dj| {
                (-r..=r).any(|di| {
                    let (ii, jj) = (i + di, j + dj);
                    ii >= 0
                        && jj >= 0
                        && (ii as usize) < nx
                        && (jj as usize) < ny
                        && unmet[jj as usize * nx + ii as usize]
                })
            });
            if near {
                out[j as usize * nx + i as usize] = false;
            }
        }
    }
    out
}

/// Drop the cells that would make the core's boundary unusable: spurs, which
/// hang by one edge, and diagonal pinches, where two cells meet at a corner
/// only and the boundary would pass through that node twice.
fn tidy(keep: &mut [bool], nx: usize, ny: usize) {
    loop {
        let mut changed = false;
        let at = |k: &[bool], i: i64, j: i64| -> bool {
            i >= 0
                && j >= 0
                && (i as usize) < nx
                && (j as usize) < ny
                && k[j as usize * nx + i as usize]
        };
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                if !at(keep, i, j) {
                    continue;
                }
                let n = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .into_iter()
                    .filter(|&(di, dj)| at(keep, i + di, j + dj))
                    .count();
                if n < 2 {
                    keep[j as usize * nx + i as usize] = false;
                    changed = true;
                }
            }
        }
        for j in 0..ny as i64 - 1 {
            for i in 0..nx as i64 - 1 {
                // The four cells around the node (i+1, j+1): a pinch is the
                // two diagonals kept and the other two dropped.
                let (a, b) = (at(keep, i, j), at(keep, i + 1, j + 1));
                let (c, d) = (at(keep, i + 1, j), at(keep, i, j + 1));
                if a && b && !c && !d {
                    keep[j as usize * nx + i as usize] = false;
                    changed = true;
                } else if c && d && !a && !b {
                    keep[j as usize * nx + i as usize + 1] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            return;
        }
    }
}

/// Chain the core's boundary edges into closed loops, wound clockwise.
///
/// Every edge with a kept cell on one side and nothing on the other is a
/// boundary edge. Emitting each of them with the core on its **right** makes
/// the walk clockwise around the core, which is what a hole is.
fn boundary_loops(
    keep: &[bool],
    nx: usize,
    ny: usize,
    vert: &HashMap<(usize, usize), u32>,
) -> Vec<Vec<u32>> {
    let at = |i: i64, j: i64| -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < nx
            && (j as usize) < ny
            && keep[j as usize * nx + i as usize]
    };
    // next[a] = b for the directed boundary edge a → b, in node indices.
    let mut next: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for j in 0..ny as i64 {
        for i in 0..nx as i64 {
            if !at(i, j) {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            // Counter-clockwise around the cell would put the core on the
            // left; each edge is emitted the other way round.
            if !at(i as i64, j as i64 - 1) {
                next.insert((i + 1, j), (i, j)); // bottom, right to left
            }
            if !at(i as i64 + 1, j as i64) {
                next.insert((i + 1, j + 1), (i + 1, j)); // right, top to bottom
            }
            if !at(i as i64, j as i64 + 1) {
                next.insert((i, j + 1), (i + 1, j + 1)); // top, left to right
            }
            if !at(i as i64 - 1, j as i64) {
                next.insert((i, j), (i, j + 1)); // left, bottom to top
            }
        }
    }

    let mut out = Vec::new();
    let mut seen: HashMap<(usize, usize), bool> = HashMap::new();
    let starts: Vec<(usize, usize)> = {
        let mut k: Vec<(usize, usize)> = next.keys().copied().collect();
        k.sort_unstable();
        k
    };
    for s in starts {
        if seen.contains_key(&s) {
            continue;
        }
        let mut ring = Vec::new();
        let mut cur = s;
        loop {
            seen.insert(cur, true);
            ring.push(vert[&cur]);
            let Some(&nxt) = next.get(&cur) else { break };
            cur = nxt;
            if cur == s || seen.contains_key(&cur) {
                break;
            }
        }
        if ring.len() >= 4 {
            out.push(ring);
        }
    }
    out
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::NodeId;
    use crate::ops::mesh::contour::Loop2D;

    fn loop2d(pts: &[(f64, f64)]) -> Loop2D {
        Loop2D {
            node_ids: (0..pts.len() as u32).map(NodeId).collect(),
            pts: pts.iter().map(|&(x, y)| Point2::new(x, y)).collect(),
        }
    }

    fn spin(pts: &[(f64, f64)], deg: f64) -> Vec<(f64, f64)> {
        let t = deg.to_radians();
        pts.iter()
            .map(|&(x, y)| (x * t.cos() - y * t.sin(), x * t.sin() + y * t.cos()))
            .collect()
    }

    fn only(outer: Loop2D) -> Domain {
        Domain {
            outer,
            holes: Vec::new(),
        }
    }

    #[test]
    fn a_square_on_shape_is_left_exactly_where_it_is() {
        // The case the whole function must not break: zero, and *exactly*
        // zero, so the frame short-circuits to the identity and a contour
        // already on the grid comes back bit for bit.
        let d = only(loop2d(&[(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)]));
        assert_eq!(preferred_angle(&d), 0.0);
    }

    #[test]
    fn a_turned_shape_gives_back_its_own_angle() {
        for deg in [5.0, 23.7, 30.0, 60.0, 88.0] {
            let d = only(loop2d(&spin(
                &[(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)],
                deg,
            )));
            // Modulo the quarter turn the grid is symmetric under.
            let want = deg.to_radians().rem_euclid(std::f64::consts::FRAC_PI_2);
            let got = preferred_angle(&d);
            // Circular distance on the quarter turn.
            let d = (got - want).rem_euclid(std::f64::consts::FRAC_PI_2);
            let err = d.min(std::f64::consts::FRAC_PI_2 - d);
            assert!(err < 1e-9, "{deg}°: got {got}, want {want}");
        }
    }

    #[test]
    fn a_circle_has_no_direction_and_keeps_the_axes() {
        // Angles spread evenly: no window holds a fifth of the perimeter, so
        // turning would only swap one arbitrary orientation for another.
        let pts: Vec<(f64, f64)> = (0..64)
            .map(|i| {
                let t = i as f64 / 64.0 * std::f64::consts::TAU;
                (t.cos(), t.sin())
            })
            .collect();
        assert_eq!(preferred_angle(&only(loop2d(&pts))), 0.0);
    }

    #[test]
    fn the_axes_win_a_tie_against_a_marginal_rival() {
        // A square on the axes with one short chamfer at 30°. The chamfer must
        // not be allowed to turn the whole grid for the sake of its own length.
        let d = only(loop2d(&[
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (0.1, 1.0),
            (0.0, 0.94),
        ]));
        assert_eq!(preferred_angle(&d), 0.0);
    }

    #[test]
    fn the_frame_is_a_round_trip() {
        let f = Frame2::at(0.4);
        let p = Point2::new(1.25, -3.5);
        let back = f.to_local(f.to_grid(p));
        assert!((back - p).norm() < 1e-12, "{back:?} for {p:?}");
        // And the identity really is the identity, not a multiplication by one.
        let id = Frame2::at(0.0);
        assert_eq!(id.to_grid(p), p);
        assert_eq!(id.to_local(p), p);
    }
}
