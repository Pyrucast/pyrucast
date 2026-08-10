//! The structured core, second method: a grid whose lines come one per contour
//! node, and whose rows are free to bend.
//!
//! The sibling of [`grid`](super::grid), and the difference is where the lines
//! come from. There, a chain of the contour pins the line it lies on and the
//! space between two lines is subdivided by whichever chain spans it. Here,
//! **every node asks for the line that crosses it**, bands too thin to be cells
//! are collapsed edge by edge, and a row that has been collapsed or fetched is
//! a polyline rather than a straight line.
//!
//! Which is better depends on the shape, and both are kept for that reason.
//! See [`grid_surface2`](fn@crate::ops::mesh::grid_surface2) for the measured
//! comparison.
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
//! So the grid lines are chosen from the contour itself, **one per node**.
//! Consecutive edges running the same way are grouped into a **chain**; every
//! node of a chain asks for the line that crosses it, and a chain a cell long
//! or more also asks for the line it lies on. Candidates closer together than
//! half a cell then merge at their mean, so two facing walls that disagree
//! meet in the middle rather than one of them winning outright. Only the gaps
//! the contour left empty are subdivided, and there uniformly.
//!
//! Taking the lines one per node is what keeps the caller's own subdivision. It
//! is also what avoids nailing a line onto a feature and subdividing either
//! side of it: that reads as respect for the shape, and is in fact how a wall
//! ends up forced into a different number of rows than the wall facing it — 5+6
//! on one side against 4+7 on the other, and a row with nowhere to go.
//!
//! What is left off by a fraction of a cell is fetched: the grid node nearest
//! to a contour node goes and stands on it, and the rows bend at their ends to
//! suit. The contour never moves — it is the grid that gives way, which is the
//! only order that leaves the caller's mesh alone. **Cells are then judged on
//! where their corners are**, bent and all; judging them on the straight grid
//! is what used to condemn a whole row for a ledge sitting a hundredth of a
//! cell off a line.
//!
//! A contour with no direction at all — a circle — gets none of this. Its
//! chords lie near an axis by accident, so following their nodes would trade a
//! regular grid for the happenstance of where its vertices fell: it gets the
//! uniform grid over its extent, and no fetching.
//!
//! ## What comes out
//!
//! [`build`] writes the core's quadrangles straight into the [`Fabric`] and
//! returns the loops of what is left to fill — the contour and the core's
//! boundary minus the edges they share, since a segment walked once each way
//! bounds nothing. Each loop is flagged with whether it is entirely the core's,
//! which is what the caller needs to decide whether it may be frozen. On a
//! rectilinear domain on the grid, no loop comes back at all.

use super::grid::{band_loops, boundary_loops, column, preferred_angle, tidy, turn, Core, Frame2};
use super::Fabric;
use crate::atoms::Point2;
use crate::ops::mesh::contour::{point_in_polygon, Domain};
use std::collections::{HashMap, HashSet};

/// How far an edge may stray, per unit it runs, and still count as aligned —
/// a quarter of a cell over a cell's length, so about 14°.
///
/// It is a reach, not a precision: the line an edge places has to stay close
/// enough to the edge to be worth placing, and a quarter cell is that. Measured
/// on a circle, whose chords are the only edges that ever sit near the
/// boundary of the window: at this width its meshing loses 24 triangles out of
/// 24 and gains none, where a window three times narrower keeps them all and
/// one half again wider brings 74.
///
/// Not to be confused with [`ALIGN_WINDOW`], which is far tighter and answers a
/// different question — *which way does the shape run*, not *what does this
/// edge ask for*.
const ALIGN_SLOPE: f64 = 0.25;

/// How far, in target sizes, a grid node may be pulled off its lines to land on
/// a contour node.
///
/// Nothing says a grid line has to be straight. Where one passes a contour node
/// it just misses, the node of the grid nearest to it goes and fetches it, and
/// the two cells hanging off that node take up the slack — the line runs
/// straight through the interior and bends at its ends. **The contour does not
/// move**; it is the grid that gives way, which is the only order that keeps
/// the caller's mesh untouched.
///
/// A quarter, and that is what makes it safe rather than merely small: the
/// collapse leaves no band thinner than half the mean interval, so a corner can
/// never reach halfway across its own cell and no cell can be turned inside
/// out. The quality guard in [`Anchors::build`] is there for what is ugly, not
/// for what is impossible.
const ANCHOR_REACH: f64 = 0.25;

/// A cell whose worst corner falls below this after bending gives its anchor
/// back. Well under what the band would have offered in its place, since a cell
/// that stays is only worth keeping if it beats the alternative.
const ANCHOR_FLOOR: f64 = 0.3;

/// A tensor grid: the coordinates of its lines along each axis.
pub struct Grid {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    /// Where a node has been put by a collapse, when that is not where its two
    /// lines cross. Collapsing a band welds each of its edges separately, so
    /// the surviving row is a polyline, not a straight line.
    pub moved: HashMap<(usize, usize), Point2>,
    /// The mean interval the contour itself carries, per axis: the average step
    /// between two consecutive nodes of the chains that dictate that axis.
    ///
    /// Measured on the contour and not on the line list, and that is what makes
    /// it usable. Taking the extent over the number of lines would halve the
    /// mean the moment two facing walls interleave their nodes — the very case
    /// the collapse exists to settle — and the floor would drop below the gap
    /// it was meant to close.
    mean: (f64, f64),
}

impl Grid {
    /// Lay a grid over `domain`, its lines taken from the contour's own nodes.
    ///
    /// Steps 3.1 to 3.3 of the method, in order: one line per node of every
    /// chain running the other way, then a line down the middle of any interval
    /// wider than twice the mean. The mean is the box divided by the number of
    /// intervals the nodes made, so the scale comes from the contour itself and
    /// not from the target size.
    ///
    /// Splitting repeats until nothing is over twice the mean — one pass would
    /// leave an interval of five means at two and a half.
    pub fn over(domain: &Domain, _target: f64) -> Grid {
        let pts = &domain.outer.pts;
        let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in pts {
            x0 = x0.min(p.x);
            x1 = x1.max(p.x);
            y0 = y0.min(p.y);
            y1 = y1.max(p.y);
        }

        // Both ends of the domain's extent are lines whatever happens.
        let (mut sx, mut sy) = (vec![x0, x1], vec![y0, y1]);
        let (mut step_x, mut step_y) = ((0.0, 0usize), (0.0, 0usize));
        for l in std::iter::once(&domain.outer).chain(&domain.holes) {
            for run in runs(l, ALIGN_SLOPE) {
                // A horizontal chain is crossed by vertical lines, so its nodes
                // — which `along` holds in x — are the x lines. And the other
                // way round.
                let (lines, step) = if run.horizontal {
                    (&mut sx, &mut step_x)
                } else {
                    (&mut sy, &mut step_y)
                };
                lines.extend_from_slice(&run.along);
                step.0 += run.along[run.along.len() - 1] - run.along[0];
                step.1 += run.along.len() - 1;
            }
        }
        let mean_of = |(total, count): (f64, usize), extent: f64| match count {
            0 => extent,
            n => total / n as f64,
        };
        let mean = (mean_of(step_x, x1 - x0), mean_of(step_y, y1 - y0));

        let mut grid = Grid {
            xs: split_or_leave(tidy_lines(sx), mean.0),
            ys: split_or_leave(tidy_lines(sy), mean.1),
            moved: HashMap::new(),
            mean,
        };
        // Steps 3.4 and 3.5.
        let pts: Vec<Point2> = std::iter::once(&domain.outer)
            .chain(&domain.holes)
            .flat_map(|l| l.pts.iter().copied())
            .collect();
        let cell = ((x1 - x0) / grid.nx() as f64).max((y1 - y0) / grid.ny() as f64);
        let near = super::proximity::PointGrid::build(&pts, cell.max(f64::MIN_POSITIVE));
        grid.collapse_thin_bands(&pts, &near);
        grid
    }

    /// Steps 3.4 and 3.5: drop every band thinner than half the mean interval.
    ///
    /// The band is not deleted line-first but **edge by edge**: each of its edges
    /// is collapsed onto the contour node at one of its ends, or onto its midpoint
    /// when neither end is one. So the surviving row is a polyline that follows the
    /// contour wherever the contour had something to say, and runs straight
    /// elsewhere — which is what lets two facing walls, cut differently, both be
    /// met by the same row.
    fn collapse_thin_bands(&mut self, pts: &[Point2], near: &super::proximity::PointGrid) {
        let grid = self;
        for axis in [false, true] {
            let lines = if axis { &grid.xs } else { &grid.ys };
            if lines.len() < 3 {
                continue;
            }
            let mean = if axis { grid.mean.0 } else { grid.mean.1 };
            let (floor, tol) = (0.5 * mean, 1e-9 * mean);

            // Which lines survive, and which line each dropped one is welded to.
            let (mut keep, mut weld) = (vec![true; lines.len()], vec![0usize; lines.len()]);
            let mut anchor = 0usize;
            for k in 1..lines.len() {
                weld[k] = k;
                if lines[k] - lines[anchor] < floor {
                    // The far end of the extent may not be dropped, so it is the
                    // one before it that goes.
                    if k == lines.len() - 1 {
                        keep[anchor] = anchor != 0;
                        weld[anchor] = k;
                        anchor = k;
                    } else {
                        keep[k] = false;
                        weld[k] = anchor;
                    }
                } else {
                    anchor = k;
                }
            }

            // A line welded onto one that was itself welded away has to follow
            // it the rest of the way, or it would be sent to an index that no
            // longer exists.
            for k in 0..lines.len() {
                let mut to = weld[k];
                while !keep[to] {
                    to = weld[to];
                }
                weld[k] = to;
            }

            let survivors: Vec<usize> = (0..lines.len()).filter(|&k| keep[k]).collect();
            if survivors.len() == lines.len() {
                continue;
            }
            let index: HashMap<usize, usize> = survivors
                .iter()
                .enumerate()
                .map(|(new, &old)| (old, new))
                .collect();

            // Where each weld lands, column by column, before the lines are
            // renumbered: the contour node at one end if there is one, the midpoint
            // otherwise.
            let across = if axis { grid.ys.len() } else { grid.xs.len() };
            let mut placed: Vec<((usize, usize), Point2)> = Vec::new();
            for (k, &to) in weld.iter().enumerate() {
                if k == to {
                    continue;
                }
                for c in 0..across {
                    // The two ends of the edge, where they actually are — the
                    // pass on the other axis may already have bent them, and
                    // looking them up on their nominal lines would miss the
                    // contour node they were bent onto.
                    let end = |line: usize| {
                        let ij = if axis { (line, c) } else { (c, line) };
                        grid.node(ij.0, ij.1)
                    };
                    let (a, b) = (end(k), end(to));
                    let at = |p: Point2| near.nearest_index_within(pts, p, tol).map(|_| p);
                    let p = at(b)
                        .or_else(|| at(a))
                        .unwrap_or_else(|| Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y)));
                    let ij = if axis {
                        (index[&to], c)
                    } else {
                        (c, index[&to])
                    };
                    placed.push((ij, p));
                }
            }

            let surviving: Vec<f64> = survivors.iter().map(|&k| lines[k]).collect();
            // The lines that survive keep their index order, so the positions
            // already recorded on the other axis have to follow the renumbering.
            let old = std::mem::take(&mut grid.moved);
            grid.moved = old
                .into_iter()
                .filter_map(|((i, j), p)| {
                    let k = if axis { i } else { j };
                    index.get(&k).map(|&n| {
                        let ij = if axis { (n, j) } else { (i, n) };
                        (ij, p)
                    })
                })
                .collect();
            if axis {
                grid.xs = surviving;
            } else {
                grid.ys = surviving;
            }
            grid.moved.extend(placed);
        }
    }

    pub fn nx(&self) -> usize {
        self.xs.len() - 1
    }

    pub fn ny(&self) -> usize {
        self.ys.len() - 1
    }

    fn node(&self, i: usize, j: usize) -> Point2 {
        self.moved
            .get(&(i, j))
            .copied()
            .unwrap_or_else(|| Point2::new(self.xs[i], self.ys[j]))
    }
}

/// One maximal chain of consecutive contour edges sharing an alignment.
struct Run {
    horizontal: bool,
    /// Its own nodes, along the axis it runs in, ascending. These are the
    /// coordinates of the lines that **cross** the chain, and taking them as
    /// they are is what makes the grid meet the contour instead of guessing at
    /// where the caller put its nodes.
    along: Vec<f64>,
}

/// Group `l`'s edges into maximal aligned chains, `slope` being how far an edge
/// may stray per unit it runs and still count as aligned.
///
/// A wall cut into four segments is one wall, and it is the wall's own length
/// that says whether the shape has a feature there — not its pieces'. Asking
/// the question piece by piece made the answer depend on the caller's
/// discretisation and on nothing else: the same wall placed a grid line when
/// its pieces came out at 1.125 target sizes and placed none at 0.975.
fn runs(l: &crate::ops::mesh::contour::Loop2D, slope: f64) -> Vec<Run> {
    let n = l.pts.len();
    if n < 2 {
        return Vec::new();
    }
    let class = |i: usize| -> Option<bool> {
        let (a, b) = (l.pts[i], l.pts[(i + 1) % n]);
        let (dx, dy) = ((b.x - a.x).abs(), (b.y - a.y).abs());
        if dy <= slope * dx && dx > 0.0 {
            Some(true)
        } else if dx <= slope * dy && dy > 0.0 {
            Some(false)
        } else {
            None
        }
    };

    // Start where the class changes, so a chain straddling the loop's seam is
    // not cut in two by the arbitrary place the loop happens to open at.
    let start = (0..n)
        .find(|&i| class(i) != class((i + n - 1) % n))
        .unwrap_or(0);

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let Some(horizontal) = class((start + i) % n) else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j < n && class((start + j) % n) == Some(horizontal) {
            j += 1;
        }
        let mut along = Vec::with_capacity(j - i + 1);
        for t in i..=j {
            let p = l.pts[(start + t) % n];
            along.push(if horizontal { p.x } else { p.y });
        }
        if along[0] > along[along.len() - 1] {
            along.reverse();
        }
        out.push(Run { horizontal, along });
        i = j;
    }
    out
}

/// Sort the candidate lines and drop the exact duplicates — a corner is a node
/// of both the chains meeting there and proposes its line twice.
fn tidy_lines(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let span = v[v.len() - 1] - v[0];
    v.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * span.max(1.0));
    v
}

/// Step 3.2: cut an interval the contour left wide, or give it up.
///
/// `mean` is the step the contour itself carries along this axis.
///
/// Between two and three means, the gap is cut into whole cells of about the
/// mean. **Beyond three, nothing is added**: a gap that wide is one the contour
/// says nothing at all about, and filling it with rows only creates cells that
/// exist to be eroded by whatever oblique boundary crosses them — the front
/// then gets its work broken into slivers instead of one clean region.
/// [`classify`] drops those cells from the core, and the front takes the region
/// whole.
///
/// Measured on a house whose roof leaves 0.40 of empty height above its walls
/// at a mean of 0.05: filling that gap cost five triangles where giving it up
/// costs one, and the fifth percentile goes from 0.644 to 0.676.
fn split_or_leave(lines: Vec<f64>, mean: f64) -> Vec<f64> {
    if lines.len() < 2 {
        return lines;
    }
    let mut out = vec![lines[0]];
    for w in lines.windows(2) {
        let (a, b) = (w[0], w[1]);
        let span = b - a;
        if span > 2.0 * mean && span <= 3.0 * mean {
            let k = ((span / mean).round() as usize).max(2);
            for i in 1..k {
                out.push(a + span * i as f64 / k as f64);
            }
        }
        out.push(b);
    }
    out
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

    // ── Send the grid to fetch the contour where it nearly touches it ─────
    // First of all, and that order is the whole point. A cell is judged on
    // where its corners **are**, and bending is what puts them there: a ledge
    // sitting a hundredth of a cell off a line runs through the interior of a
    // whole row of the straight grid and condemns it, when all it needed was
    // for that row to come and meet it.
    let known = ContourNodes::build(fab, contour_loops, frame);

    // ── Step 5: the grid goes and fetches the contour nodes it passes ─────
    let mut anchors = Anchors::build(&grid, &known, target);

    // ── Which cells are entirely in the material ──────────────────────────
    // A cell is solid when the boundary misses it and its centroid is inside.
    // Both questions are asked of the bent cell.
    //
    // The two settle each other: dropping an anchor moves corners back onto
    // the lines, which can let the boundary through a cell that had been
    // spared, so the classification is worth asking again. Each pass only ever
    // hands anchors back, so the loop closes — and on a contour the grid can
    // meet it does not go round twice.
    let mut solid = classify(&grid, &anchors, domain, nx, ny);
    while anchors.settle(&grid, &solid, nx, ny) {
        solid = classify(&grid, &anchors, domain, nx, ny);
    }

    // ── Pull back only where the core misses the contour ──────────────────
    let mut keep = clear_of_unmet(&solid, nx, ny, &anchors, contour_loops, band);
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
                    // The contour's own node when this one went to fetch it,
                    // else a fresh one on the lines — put back in the caller's
                    // plane, since the frame is the grid's business and nobody
                    // else's.
                    anchors.id(i + di, j + dj).unwrap_or_else(|| {
                        fab.add(frame.to_local(grid.node(i + di, j + dj)), false, false)
                    })
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

/// The contour's vertices, in the grid's frame, alongside their fabric ids.
struct ContourNodes {
    pts: Vec<Point2>,
    ids: Vec<u32>,
}

impl ContourNodes {
    fn build(fab: &Fabric, loops: &[Vec<u32>], frame: Frame2) -> ContourNodes {
        let ids: Vec<u32> = loops.iter().flatten().copied().collect();
        // The fabric holds the caller's plane; the grid asks its questions in
        // its own frame, so the contour is indexed there.
        let pts: Vec<Point2> = ids
            .iter()
            .map(|&v| frame.to_grid(fab.pts[v as usize]))
            .collect();
        ContourNodes { pts, ids }
    }
}

/// Which contour node each grid node has gone to fetch.
///
/// A grid line has no obligation to stay straight. Where one passes a contour
/// node it just misses, the nearest node of the grid goes and gets it, and the
/// two cells hanging off that node absorb the offset: the line runs straight
/// through the interior and bends at its ends. That is what pays for a contour
/// the grid cannot meet exactly — and it costs the caller nothing, because
/// **the contour never moves**. Only the grid gives way.
///
/// The pairing is one to one in both directions. Two grid nodes fetching the
/// same contour node would collapse onto each other and take their cell with
/// them; one grid node serving two contour nodes cannot be in two places.
struct Anchors {
    to: HashMap<(usize, usize), (u32, Point2)>,
}

impl Anchors {
    fn build(grid: &Grid, known: &ContourNodes, target: f64) -> Anchors {
        let reach = ANCHOR_REACH * target;
        // One candidate per contour node: the grid node nearest to it. A second
        // choice would be a longer reach for a node whose first choice already
        // has a closer claimant, which is not a bargain worth the bookkeeping.
        let mut want: Vec<(f64, usize, (usize, usize))> = Vec::new();
        for (k, p) in known.pts.iter().enumerate() {
            let ij = (nearest_line(&grid.xs, p.x), nearest_line(&grid.ys, p.y));
            let d = (grid.node(ij.0, ij.1) - *p).norm();
            if d <= reach {
                want.push((d, k, ij));
            }
        }
        // Nearest first — so a node that already sits exactly on the grid is
        // served before anything else can take its place — and ties settled by
        // contour node, so the result does not depend on the order the loops
        // happened to be walked in.
        want.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

        let mut to: HashMap<(usize, usize), (u32, Point2)> = HashMap::new();
        let mut taken = vec![false; known.pts.len()];
        for (_, k, ij) in want {
            if taken[k] || to.contains_key(&ij) {
                continue;
            }
            taken[k] = true;
            to.insert(ij, (known.ids[k], known.pts[k]));
        }

        Anchors { to }
    }

    /// Give back the anchors of every solid cell they leave worse than the band
    /// would have been, and say whether any were handed back — the caller has
    /// to reclassify if so, since a corner returning to its lines can let the
    /// boundary back into a cell.
    fn settle(&mut self, grid: &Grid, solid: &[bool], nx: usize, ny: usize) -> bool {
        let mut giving_back: Vec<(usize, usize)> = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if !solid[j * nx + i] {
                    continue;
                }
                let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
                if !corners.iter().any(|c| self.to.contains_key(c)) {
                    continue;
                }
                if super::geom::quad_quality(self.cell(grid, i, j)) < ANCHOR_FLOOR {
                    giving_back.extend(corners.iter().filter(|c| self.to.contains_key(c)));
                }
            }
        }
        for c in &giving_back {
            self.to.remove(c);
        }
        !giving_back.is_empty()
    }

    /// The contour node this grid node fetched, if it fetched one.
    fn id(&self, i: usize, j: usize) -> Option<u32> {
        self.to.get(&(i, j)).map(|&(v, _)| v)
    }

    /// Where this grid node actually is: on the contour node it fetched, or on
    /// its lines when it fetched none.
    fn point(&self, grid: &Grid, i: usize, j: usize) -> Point2 {
        self.to
            .get(&(i, j))
            .map_or_else(|| grid.node(i, j), |&(_, p)| p)
    }

    /// Cell `(i, j)` as it actually is, counter-clockwise.
    fn cell(&self, grid: &Grid, i: usize, j: usize) -> [Point2; 4] {
        [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(di, dj)| self.point(grid, i + di, j + dj))
    }
}

/// The index of the line of `lines` nearest to `v`.
fn nearest_line(lines: &[f64], v: f64) -> usize {
    let k = lines.partition_point(|&x| x < v);
    if k == 0 {
        return 0;
    }
    if k == lines.len() || v - lines[k - 1] <= lines[k] - v {
        k - 1
    } else {
        k
    }
}

/// The cells whose **interior** the segment `a → b` enters.
///
/// The open interior, and that is the whole point: a boundary edge lying
/// exactly along a grid line cuts neither of the two cells it separates, it
/// just runs between them. Testing bounding boxes instead would mark both, the
/// outermost ring of cells would be lost on every axis-aligned shape, and the
/// core would stop one cell short of a contour it could have met exactly.
fn cells_touching(
    grid: &Grid,
    anchors: &Anchors,
    nx: usize,
    ny: usize,
    a: Point2,
    b: Point2,
) -> Vec<(usize, usize)> {
    // The index window comes from the straight grid, so it is widened by one
    // cell each way: a corner may have been fetched up to a quarter cell off
    // its lines, and the segment that fetched it lies that much outside the
    // window its own coordinates would give.
    let span = |lines: &[f64], lo: f64, hi: f64, n: usize| {
        (
            column(lines, lo).saturating_sub(1),
            (column(lines, hi) + 1).min(n - 1),
        )
    };
    let (i0, i1) = span(&grid.xs, a.x.min(b.x), a.x.max(b.x), nx);
    let (j0, j1) = span(&grid.ys, a.y.min(b.y), a.y.max(b.y), ny);
    let mut out = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            if enters(a, b, anchors.cell(grid, i, j)) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Does the segment `a → b` meet the **open** interior of the convex,
/// counter-clockwise quadrangle `q`?
///
/// Liang–Barsky against the four sides rather than against a box, because a
/// cell whose corners have been fetched is no longer a rectangle. Each side is
/// pushed inward by a rounding, which is what keeps a segment lying *along* a
/// side from counting as entering — the property the whole core depends on,
/// since a contour edge that runs between two cells cuts neither.
fn enters(a: Point2, b: Point2, q: [Point2; 4]) -> bool {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for k in 0..4 {
        let (p0, p1) = (q[k], q[(k + 1) % 4]);
        let e = p1 - p0;
        // Inward normal of a counter-clockwise side, unnormalised; the epsilon
        // is scaled by its length so it stays a true distance.
        let n = Point2::new(-e.y, e.x);
        let len = n.coords.norm();
        if len == 0.0 {
            continue;
        }
        let num = n.x * (a.x - p0.x) + n.y * (a.y - p0.y) - 1e-9 * len;
        let den = n.x * d.x + n.y * d.y;
        if den == 0.0 {
            if num < 0.0 {
                return false; // parallel to this side and outside it
            }
        } else {
            let r = -num / den;
            if den > 0.0 {
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

/// Mark the cells the boundary misses and whose centroid is in the material —
/// both judged on the cell as it actually is, corners fetched and all.
fn classify(grid: &Grid, anchors: &Anchors, domain: &Domain, nx: usize, ny: usize) -> Vec<bool> {
    let mut cut = vec![false; nx * ny];
    for l in std::iter::once(&domain.outer).chain(&domain.holes) {
        let n = l.pts.len();
        for i in 0..n {
            let (a, b) = (l.pts[i], l.pts[(i + 1) % n]);
            for (ci, cj) in cells_touching(grid, anchors, nx, ny, a, b) {
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
            // A cell `split_or_leave` gave up on: the contour said nothing
            // about that stretch, so the front gets the region whole rather
            // than a stack of rows that only exist to be eroded.
            let (w, h) = (grid.xs[i + 1] - grid.xs[i], grid.ys[j + 1] - grid.ys[j]);
            if w > 2.0 * grid.mean.0 || h > 2.0 * grid.mean.1 {
                continue;
            }
            let q = anchors.cell(grid, i, j);
            let c = Point2::new(
                0.25 * (q[0].x + q[1].x + q[2].x + q[3].x),
                0.25 * (q[0].y + q[1].y + q[2].y + q[3].y),
            );
            solid[j * nx + i] = point_in_polygon(c, &domain.outer.pts)
                && !domain.holes.iter().any(|h| point_in_polygon(c, &h.pts));
        }
    }
    solid
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
    solid: &[bool],
    nx: usize,
    ny: usize,
    anchors: &Anchors,
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
    // A face lies on the contour when both its grid nodes went and fetched a
    // contour node and the two they fetched are neighbours on it. Asked of the
    // anchors rather than of the geometry, so a face that only reaches the
    // contour by bending counts exactly as much as one that was already there.
    let met =
        |a: (usize, usize), b: (usize, usize)| match (anchors.id(a.0, a.1), anchors.id(b.0, b.1)) {
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
                if !met(a, b) {
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

    fn only(outer: Loop2D) -> Domain {
        Domain {
            outer,
            holes: Vec::new(),
        }
    }

    /// A closed loop through `corners`, each side cut into a whole number of
    /// pieces of about `h` — the discretisation a caller hands in.
    fn cut(corners: &[(f64, f64)], h: f64) -> Loop2D {
        let n = corners.len();
        let mut pts = Vec::new();
        for i in 0..n {
            let (a, b) = (corners[i], corners[(i + 1) % n]);
            let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            let k = ((len / h).round() as usize).max(1);
            for j in 0..k {
                let t = j as f64 / k as f64;
                pts.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
            }
        }
        loop2d(&pts)
    }

    /// Every position the grid actually puts a node at, lines and welds alike.
    fn nodes(g: &Grid) -> Vec<Point2> {
        (0..g.xs.len())
            .flat_map(|i| (0..g.ys.len()).map(move |j| (i, j)))
            .map(|(i, j)| g.node(i, j))
            .collect()
    }

    #[test]
    fn a_wall_is_reached_whatever_its_pieces_measure() {
        // Whether the grid reaches a wall must depend on the wall, not on the
        // caller's arithmetic. It is asked of the node **positions** and not of
        // the line list: the wall's column is welded onto its neighbour, so no
        // line carries x = 0.53, and yet every node along the wall sits on it.
        let h = 0.1;
        for tenths in 2..9u32 {
            let step = tenths as f64 * 0.1 + 0.01;
            let d = only(cut(
                &[
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.53, 1.0),
                    (0.53, step),
                    (0.0, step),
                ],
                h,
            ));
            let g = Grid::over(&d, h);
            let on_it = nodes(&g)
                .iter()
                .filter(|p| (p.x - 0.53).abs() < 1e-9)
                .count();
            assert!(
                on_it >= 2,
                "the wall at x=0.53 was reached by {on_it} nodes, step at y={step}"
            );
        }
    }

    #[test]
    fn one_row_reaches_both_of_two_walls_that_disagree() {
        // An L 1.03 by 0.98 at a target of 0.1: the right wall spans y in
        // [0, 0.47] and cuts it in five, at a pitch of 0.094; the left spans
        // [0, 0.98] and cuts it in ten, at 0.098. No straight row can be at
        // both heights at once — so the rows are not straight. Each is welded
        // column by column onto whichever contour node its end meets, and both
        // walls come out on the grid.
        let d = only(cut(
            &[
                (0.0, 0.0),
                (1.03, 0.0),
                (1.03, 0.47),
                (0.41, 0.47),
                (0.41, 0.98),
                (0.0, 0.98),
            ],
            0.1,
        ));
        let g = Grid::over(&d, 0.1);
        let all = nodes(&g);
        let sits = |x: f64, y: f64| {
            all.iter()
                .any(|p| (p.x - x).abs() < 1e-9 && (p.y - y).abs() < 1e-9)
        };
        for k in 1..10 {
            let y = 0.98 * k as f64 / 10.0;
            assert!(sits(0.0, y), "left wall node at y={y} is off the grid");
        }
        for k in 1..5 {
            let y = 0.47 * k as f64 / 5.0;
            assert!(sits(1.03, y), "right wall node at y={y} is off the grid");
        }
    }

    #[test]
    fn the_lines_follow_the_contour_rather_than_recompute_it() {
        // A rectangle 0.55 tall at a target of 0.1 is five cells or six
        // depending on which way the halfway mark is rounded, and the caller
        // has already chosen: it cut its sides into five. The grid has to make
        // the same choice, and the only way to be sure of that is to read it
        // off the contour rather than round again.
        let mut pts = Vec::new();
        for i in 0..10 {
            pts.push((i as f64 * 0.1, 0.0));
        }
        for i in 0..5 {
            pts.push((1.0, i as f64 * 0.11));
        }
        for i in 0..10 {
            pts.push((1.0 - i as f64 * 0.1, 0.55));
        }
        for i in 0..5 {
            pts.push((0.0, 0.55 - i as f64 * 0.11));
        }
        let g = Grid::over(&only(loop2d(&pts)), 0.1);
        let want = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55];
        assert_eq!(g.ys.len(), want.len(), "ys = {:?}", g.ys);
        for (got, w) in g.ys.iter().zip(want) {
            assert!((got - w).abs() < 1e-9, "ys = {:?}", g.ys);
        }
    }
}
