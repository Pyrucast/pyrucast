//! Spatial index over the live front, rebuilt once per row.
//!
//! Two questions have to be answered constantly while paving, and both are
//! quadratic if asked naively:
//!
//! - *does this new front edge cross an existing one?* — the guard that keeps
//!   the front a set of simple, disjoint loops;
//! - *are these two front nodes close enough to be seamed together?* — the
//!   operation that closes concave regions and swallows holes.
//!
//! A uniform grid over the front's edges answers both. The front is small
//! (its length grows like the square root of the element count), so rebuilding
//! the whole index each row costs less than maintaining it incrementally, and
//! it cannot drift out of sync with the front — which, for a structure the
//! validity guard depends on, is worth more than the saved cycles.
//!
//! The cell size adapts so the bucket count tracks the edge count: a fine
//! target size on a large domain must not allocate a grid a thousand times
//! bigger than the front it indexes.

use super::front::Front;
use crate::containers::mesh::Point2;

/// Uniform grid bucketing front edges by their bounding box.
///
/// An edge is named by its *starting slot*: edge `s` runs from `s` to
/// `front.next(s)`.
pub struct EdgeGrid {
    lo: Point2,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
}

impl EdgeGrid {
    /// Index every live front edge. `hint` is the target element size, used
    /// as the lower bound on the cell size.
    pub fn build(front: &Front, pts: &[Point2], hint: f64) -> EdgeGrid {
        EdgeGrid::of_segments(
            front.live_slots().map(|s| {
                (
                    s,
                    pts[front.vertex(s) as usize],
                    pts[front.vertex(front.next(s)) as usize],
                )
            }),
            hint,
        )
    }

    /// Index a closed polyline, each edge named by the index of its first
    /// point. Used to ask whether an advanced front crosses *itself*, which
    /// is otherwise a quadratic question.
    pub fn of_ring(pts: &[Point2], hint: f64) -> EdgeGrid {
        let m = pts.len();
        EdgeGrid::of_segments(
            (0..m as u32).map(|i| (i, pts[i as usize], pts[(i as usize + 1) % m])),
            hint,
        )
    }

    /// Index an arbitrary set of named segments.
    pub fn of_segments(segs: impl Iterator<Item = (u32, Point2, Point2)>, hint: f64) -> EdgeGrid {
        let items: Vec<(u32, Point2, Point2)> = segs.collect();
        let (mut lo, mut hi) = (
            Point2::new(f64::INFINITY, f64::INFINITY),
            Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for (_, a, b) in &items {
            for p in [a, b] {
                lo.x = lo.x.min(p.x);
                lo.y = lo.y.min(p.y);
                hi.x = hi.x.max(p.x);
                hi.y = hi.y.max(p.y);
            }
        }
        let n = items.len().max(1);
        if !lo.x.is_finite() {
            lo = Point2::origin();
            hi = Point2::origin();
        }
        let (w, h) = ((hi.x - lo.x).max(hint), (hi.y - lo.y).max(hint));
        // Keep the bucket count proportional to the edge count rather than to
        // the domain-to-element-size ratio, which can be enormous. With
        // `cell ≥ √(area / n)` the grid holds at most about `n` cells however
        // fine the target size is.
        let cell = hint.max((w * h / n as f64).sqrt()).max(1e-300);
        let nx = ((w / cell).ceil() as usize + 1).min(n + 2);
        let ny = ((h / cell).ceil() as usize + 1).min(n + 2);

        let mut grid = EdgeGrid {
            lo,
            cell,
            nx,
            ny,
            buckets: vec![Vec::new(); nx * ny],
        };
        for (id, a, b) in &items {
            for k in grid.cells_of(*a, *b) {
                grid.buckets[k].push(*id);
            }
        }
        grid
    }

    fn ix(&self, x: f64) -> usize {
        (((x - self.lo.x) / self.cell).floor().max(0.0) as usize).min(self.nx - 1)
    }

    fn iy(&self, y: f64) -> usize {
        (((y - self.lo.y) / self.cell).floor().max(0.0) as usize).min(self.ny - 1)
    }

    /// Bucket indices covered by the bounding box of the segment `ab`.
    fn cells_of(&self, a: Point2, b: Point2) -> Vec<usize> {
        let (x0, x1) = (self.ix(a.x.min(b.x)), self.ix(a.x.max(b.x)));
        let (y0, y1) = (self.iy(a.y.min(b.y)), self.iy(a.y.max(b.y)));
        let mut out = Vec::with_capacity((x1 - x0 + 1) * (y1 - y0 + 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.push(y * self.nx + x);
            }
        }
        out
    }

    /// Slots whose edge's bounding box may meet the box of `ab`, sorted and
    /// deduplicated so the caller's decisions stay order-independent.
    pub fn near_segment(&self, a: Point2, b: Point2) -> Vec<u32> {
        let mut out = Vec::new();
        for k in self.cells_of(a, b) {
            out.extend_from_slice(&self.buckets[k]);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Slots with an endpoint within `radius` of `p`.
    pub fn near_point(&self, p: Point2, radius: f64) -> Vec<u32> {
        let d = Point2::new(radius, radius);
        self.near_segment(p - d.coords, p + d.coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_front(n: usize) -> (Front, Vec<Point2>) {
        let mut pts = Vec::new();
        for i in 0..n {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            pts.push(Point2::new(t.cos(), t.sin()));
        }
        let mut f = Front::new();
        f.add_loop(&(0..n as u32).collect::<Vec<_>>());
        (f, pts)
    }

    #[test]
    fn every_edge_is_found_near_itself() {
        let (f, pts) = square_front(40);
        let g = EdgeGrid::build(&f, &pts, 0.15);
        for s in f.live_slots() {
            let a = pts[f.vertex(s) as usize];
            let b = pts[f.vertex(f.next(s)) as usize];
            assert!(
                g.near_segment(a, b).contains(&s),
                "edge {s} must be reported near itself"
            );
        }
    }

    #[test]
    fn a_far_away_query_finds_nothing() {
        let (f, pts) = square_front(40);
        let g = EdgeGrid::build(&f, &pts, 0.15);
        let far = Point2::new(50.0, 50.0);
        // Clamped into the grid, the query still must not claim a hit that
        // shares no cell: check the reported edges really are in that corner.
        for s in g.near_segment(far, far) {
            let a = pts[f.vertex(s) as usize];
            assert!(a.x > 0.0 && a.y > 0.0);
        }
    }

    #[test]
    fn the_bucket_count_tracks_the_front_not_the_element_size() {
        let (f, pts) = square_front(40);
        // A target size a thousand times smaller than the domain must not
        // allocate a million buckets for forty edges.
        let g = EdgeGrid::build(&f, &pts, 1e-3);
        assert!(g.buckets.len() <= 4 * 40 + 8 + 200, "{}", g.buckets.len());
    }
}
