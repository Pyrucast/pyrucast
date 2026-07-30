//! One row of quadrangles, laid along a whole front loop at once.
//!
//! ## How many quadrangles meet a front node
//!
//! Take a front node `v` with interior angle `θ`. After the row, `v` is an
//! interior vertex, and the neighbours it acquires form a path from `v`'s front
//! predecessor to its front successor. If `k` quadrangles end up around `v`,
//! that path has `k + 1` vertices, so exactly `k - 1` of them are new. Wanting
//! every corner near a right angle fixes `k`:
//!
//! | interior angle `θ` | `k` | name | new nodes |
//! |---|---|---|---|
//! | `θ < 3π/4` | 1 | end | 0 |
//! | `3π/4 ≤ θ < 5π/4` | 2 | side | 1 |
//! | `5π/4 ≤ θ < 7π/4` | 3 | corner | 2 |
//! | `7π/4 ≤ θ` | 4 | reversal | 3 |
//!
//! which is just `k = round(θ / (π/2))`, clamped to `1..=4`. The classical
//! paving table is not a set of tuned thresholds but a consequence of asking
//! for right angles.
//!
//! ## Where the nodes go
//!
//! The material sector at `v` spans `θ`, from the direction of the successor
//! round to the direction of the predecessor. Splitting it into `k` equal
//! sectors gives `k - 1` interior rays; the new nodes `x₁ … x_{k-1}` sit on
//! them at the local size `d`, numbered from the successor side. `x_j` is
//! obtained by rotating the unit successor direction counter-clockwise by
//! `j·θ/k`, which stays correct past `θ = π` where summing unit edge vectors
//! would flip to the outside.
//!
//! Two adjacent quadrangles account for the sector's two outer slices; the
//! `k - 2` slices in between need one more node each — `w_j`, on the bisector
//! between `x_j` and `x_{j+1}` at `d / cos(θ/2k)`, the distance that makes the
//! slice square. The wedge is then closed by the quadrangle `(v, x_j, w_j,
//! x_{j+1})`.
//!
//! ## Ends share a node
//!
//! An end node produces nothing and merges its two incident quadrangles into
//! one, `(v₋, v, v₊, Y)`. That single `Y` is the node its two neighbours would
//! each have produced, so the two are **identified**. Chains of such
//! identifications are resolved by union-find, which is also what lets a small
//! loop collapse to a single centre node without any special case.
//!
//! Where `Y` goes is not free. Writing `a = v₋ - v`, `b = v₊ - v` and
//! `Y(t) = v + t·(a + b)`, the merged quadrangle's four orientations work out
//! to be positive exactly when **`t > 1/2`** — the corner at `Y` flips sign at
//! `(1-t)² - t²`, and the other three are positive for every `t > 0`. Three of
//! its corners are existing vertices, so this is a constraint on shape, not on
//! scale: a reflex result stays reflex however far the row retreats, which is
//! why an end node placed by any other rule can deadlock the paver outright.
//!
//! `t = 1` is the parallelogram point, and it is the right answer whenever the
//! row is advancing by about the length of the corner's own edges: on a right
//! angle with edges `L` it puts `Y` at `L√2`, making the merged quadrangle an
//! exact square. What it must not do is stay far out when the row has retreated
//! to a fraction of the edge length — that drags the neighbours' nodes with it
//! and wrecks their quadrangles instead. So `t` scales with the ratio of the
//! advance to the corner's characteristic edge length `|a + b| / √2`, clamped
//! into `[0.6, 1]`: comfortably clear of the `1/2` that convexity forbids, and
//! exactly `1` in the well-proportioned case.
//!
//! Checked on a square boundary of 5 nodes a side: 20 slots, 4 ends → 16
//! quadrangles and a 12-slot loop; then 8 and a 4-slot loop; then closure.
//! `16 + 8 + 1 = 25 = 5 × 5`.

use super::front::Front;
use super::geom::{interior_angle, quad_is_valid, rot};
use crate::containers::mesh::Point2;

/// Interior-angle span of one quadrangle corner: the row aims for right
/// angles, and `k = round(θ / QUARTER_TURN)` follows from that.
const QUARTER_TURN: f64 = std::f64::consts::FRAC_PI_2;

/// How far a wedge node may sit beyond the ring of `x` nodes.
const WEDGE_REACH_MAX: f64 = 2.0;

/// Interior angle past which a neighbour is too reflex to give up its node to
/// an end. Deliberately well above `π`, the angle of an ordinary straight
/// front node.
const REFLEX_NEIGHBOUR: f64 = 1.15 * std::f64::consts::PI;

/// How far along the diagonal an end node may be pulled back toward its own
/// corner. Convexity of the merged quadrangle fails at exactly `1/2`, so this
/// keeps a margin above it.
const END_REACH_MIN: f64 = 0.6;

/// A front edge longer than this many times the wanted size asks for one more
/// quadrangle at its ends, which is how the front refines itself where it
/// expands — around a hole, typically.
const REFINE_RATIO: f64 = 1.45;

// There is deliberately no symmetric coarsening rule. Demoting a node to an
// end merges its two quadrangles into `(v₋, v, v₊, Y)`, three of whose corners
// are existing vertices — so that quadrangle is convex only if the corner at
// `v` already is, and no amount of retreating can rescue it otherwise. The
// angle test below is therefore the *only* thing allowed to create an end. A
// front that has grown too dense is thinned by seaming instead, which removes
// two slots and cannot invert anything.

/// A corner of a planned quadrangle: either a vertex that already exists, or
/// one of this row's new nodes, still identified by its local index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Corner {
    Old(u32),
    New(u32),
}

/// A row, planned but not yet committed to the mesh.
#[derive(Debug)]
pub struct RowPlan {
    /// Positions of this row's new nodes, indexed by local id.
    pub pts: Vec<Point2>,
    /// New nodes forming the advanced front, in walking order.
    pub chain: Vec<u32>,
    /// The quadrangles the row lays down.
    pub quads: Vec<[Corner; 4]>,
    /// For each quadrangle, the loop slot it was generated from — so a caller
    /// that has to retreat knows *where* to retreat.
    pub owners: Vec<usize>,
}

/// Minimal union-find over the row's new nodes, used to identify the nodes
/// that an end node makes its two neighbours share.
struct Union(Vec<u32>);

impl Union {
    fn new(n: usize) -> Union {
        Union((0..n as u32).collect())
    }
    fn find(&mut self, mut i: u32) -> u32 {
        while self.0[i as usize] != i {
            let g = self.0[self.0[i as usize] as usize];
            self.0[i as usize] = g;
            i = g;
        }
        i
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Lowest index wins, so the result never depends on call order.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi as usize] = lo;
        }
    }
}

/// Number of quadrangles wanted at a node of interior angle `theta`.
fn quads_at(theta: f64) -> usize {
    ((theta / QUARTER_TURN).round() as i64).clamp(1, 4) as usize
}

/// Plan one row over the loop containing `rep`.
///
/// `size(i)` is how far the `i`-th slot should advance — already scaled by any
/// retreat the caller is applying — while `spacing(i)` is the element size
/// actually wanted there. The two are deliberately separate: a retreat is a
/// geometric concession, not a change of target, and letting it drive the
/// refinement rule below would make every refused row *add* nodes and the
/// front grow without bound.
///
/// Fails with the list of slots to blame when the row would lay down a
/// quadrangle that is not strictly convex — that is, one a finite-element code
/// could not integrate. The caller retreats *there* and tries again, rather
/// than shrinking the whole row for one bad corner.
pub fn plan(
    front: &Front,
    pts: &[Point2],
    rep: u32,
    size: &dyn Fn(usize) -> f64,
    spacing: &dyn Fn(usize) -> f64,
    all_quad: bool,
) -> Result<RowPlan, Vec<usize>> {
    let slots = front.loop_slots(rep);
    let n = slots.len();
    if n < 3 {
        return Err(Vec::new());
    }
    let v: Vec<u32> = slots.iter().map(|&s| front.vertex(s)).collect();
    let p: Vec<Point2> = v.iter().map(|&i| pts[i as usize]).collect();

    // ── Classification ────────────────────────────────────────────────────
    let theta: Vec<f64> = (0..n)
        .map(|i| interior_angle(p[(i + n - 1) % n], p[i], p[(i + 1) % n]))
        .collect();
    let mut k: Vec<usize> = theta.iter().map(|&t| quads_at(t)).collect();
    // Size control. The angle alone says nothing about how long the front's
    // edges have become: a front expanding round a hole stretches, and would
    // otherwise carry ever coarser elements inward. One extra quadrangle at
    // such a node refines it locally.
    for i in 0..n {
        let want = spacing(i);
        if size(i) <= 0.0 || want <= 0.0 || size(i).is_nan() || want.is_nan() {
            return Err(vec![i]);
        }
        let span = 0.5 * ((p[(i + 1) % n] - p[i]).norm() + (p[i] - p[(i + n - 1) % n]).norm());
        if span > REFINE_RATIO * want && k[i] < 4 {
            k[i] += 1;
        }
    }
    // An end does not produce a node of its own: it commandeers one from each
    // neighbour and pins it at the parallelogram point. A markedly reflex
    // neighbour has none to spare — its own sector needs the node it has,
    // pointing somewhere else entirely — and pinning it there gives a
    // quadrangle no retreat can straighten. Note the threshold is well past a
    // straight front: `θ = π` is the commonest front node there is, and
    // barring ends next to it would disqualify every right-angled corner.
    for i in 0..n {
        if k[i] == 1
            && (theta[(i + n - 1) % n] > REFLEX_NEIGHBOUR || theta[(i + 1) % n] > REFLEX_NEIGHBOUR)
        {
            k[i] = 2;
        }
    }
    // Two adjacent ends would each want to consume the same neighbour's node;
    // promote the later one to a plain side so every end stands alone.
    for i in 0..n {
        if k[i] == 1 && k[(i + n - 1) % n] == 1 {
            k[i] = 2;
        }
    }
    if n > 1 && k[0] == 1 && k[n - 1] == 1 {
        k[0] = 2;
    }
    if k.iter().all(|&x| x == 1) {
        return Err((0..n).collect());
    }

    // A row normally preserves the loop's parity, which is what the
    // all-quadrangle guarantee rests on. The exception is a chain of ends
    // whose identifications run together, merging more than two nodes at once
    // and losing a node the count was relying on. Demoting one end to a plain
    // side restores it; each demotion removes an end, so this settles.
    let mut k = k;
    loop {
        let built = assemble(front, pts, rep, &v, &p, &theta, &k, size);
        let Ok(plan) = built else { return built };
        if !all_quad || plan.chain.len() % 2 == n % 2 || plan.chain.len() < 3 {
            return Ok(plan);
        }
        match (0..n).find(|&i| k[i] == 1) {
            Some(i) => k[i] = 2,
            None => return Ok(plan),
        }
    }
}

/// Build the row for a settled classification.
#[allow(clippy::too_many_arguments)]
fn assemble(
    front: &Front,
    pts: &[Point2],
    rep: u32,
    v: &[u32],
    p: &[Point2],
    theta: &[f64],
    k: &[usize],
    size: &dyn Fn(usize) -> f64,
) -> Result<RowPlan, Vec<usize>> {
    let _ = (front, rep);
    let n = v.len();
    // ── New nodes, slot by slot ───────────────────────────────────────────
    // `chain_of[i]` walks slot `i`'s new nodes from the predecessor side to
    // the successor side: x_{k-1}, w_{k-2}, …, w_1, x_1.
    let mut pos: Vec<Point2> = Vec::new();
    let mut chain_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        if k[i] == 1 {
            continue;
        }
        let (ki, th, d) = (k[i], theta[i], size(i));
        let a = p[(i + 1) % n] - p[i];
        let na = a.norm();
        if na == 0.0 {
            return Err(vec![i]);
        }
        let dir = a / na;
        let ring: Vec<Point2> = (1..ki)
            .map(|j| p[i] + rot(dir, th * j as f64 / ki as f64) * d)
            .collect();
        let reach = (1.0 / (th / (2.0 * ki as f64)).cos())
            .abs()
            .min(WEDGE_REACH_MAX);
        let wedge: Vec<Point2> = (1..ki.saturating_sub(1))
            .map(|j| p[i] + rot(dir, th * (j as f64 + 0.5) / ki as f64) * (d * reach))
            .collect();

        // Walking order is the reverse of the numbering, interleaved.
        let mut chain = Vec::with_capacity(2 * ki - 3);
        for j in (1..ki).rev() {
            chain.push(push(&mut pos, ring[j - 1]));
            if j > 1 {
                chain.push(push(&mut pos, wedge[j - 2]));
            }
        }
        chain_of[i] = chain;
    }

    // ── Ends identify their two neighbours' shared node ───────────────────
    let mut uf = Union::new(pos.len());
    let mut forced: Vec<Option<Point2>> = vec![None; pos.len()];
    for i in 0..n {
        if k[i] != 1 {
            continue;
        }
        let before = &chain_of[(i + n - 1) % n];
        let after = &chain_of[(i + 1) % n];
        let (Some(&last), Some(&first)) = (before.last(), after.first()) else {
            return Err(vec![i]);
        };
        uf.union(last, first);
        let diag = (p[(i + n - 1) % n] - p[i]) + (p[(i + 1) % n] - p[i]);
        let reach = diag.norm();
        let t = if reach > 0.0 {
            (size(i) * std::f64::consts::SQRT_2 / reach).clamp(END_REACH_MIN, 1.0)
        } else {
            1.0
        };
        let y = p[i] + diag * t;
        forced[last as usize] = Some(y);
        forced[first as usize] = Some(y);
    }

    // A class containing a forced node sits where the end that forced it
    // wants it; anything else sits at the mean of what was merged into it.
    // Two ends can force the same class from opposite sides, in which case
    // they meet halfway.
    let mut forced_class: Vec<Option<(Point2, usize)>> = vec![None; pos.len()];
    let mut sum = vec![(0.0f64, 0.0f64, 0usize); pos.len()];
    for i in 0..pos.len() {
        let r = uf.find(i as u32) as usize;
        if let Some(y) = forced[i] {
            let slot = &mut forced_class[r];
            *slot = Some(match *slot {
                None => (y, 1),
                Some((acc, c)) => (Point2::from(acc.coords + y.coords), c + 1),
            });
        }
        sum[r].0 += pos[i].x;
        sum[r].1 += pos[i].y;
        sum[r].2 += 1;
    }
    let place = |r: usize| match forced_class[r] {
        Some((acc, c)) => Point2::from(acc.coords / c as f64),
        None => {
            let (sx, sy, c) = sum[r];
            Point2::new(sx / c as f64, sy / c as f64)
        }
    };

    let mut remap = vec![u32::MAX; pos.len()];
    let mut out_pts: Vec<Point2> = Vec::new();
    for i in 0..pos.len() {
        let r = uf.find(i as u32) as usize;
        if remap[r] == u32::MAX {
            remap[r] = push(&mut out_pts, place(r));
        }
        remap[i] = remap[r];
    }
    let at = |i: u32| remap[i as usize];

    // ── The advanced front ────────────────────────────────────────────────
    let mut chain: Vec<u32> = Vec::new();
    for i in 0..n {
        for &c in &chain_of[i] {
            let c = at(c);
            if chain.last() != Some(&c) {
                chain.push(c);
            }
        }
    }
    while chain.len() > 1 && chain.first() == chain.last() {
        chain.pop();
    }

    // ── Quadrangles ───────────────────────────────────────────────────────
    let mut quads: Vec<[Corner; 4]> = Vec::new();
    let mut owners: Vec<usize> = Vec::new();
    for i in 0..n {
        if k[i] == 1 {
            // The merged quadrangle of an end node.
            let Some(&y) = chain_of[(i + 1) % n].first() else {
                return Err(vec![i]);
            };
            let y = at(y);
            owners.push(i);
            quads.push([
                Corner::Old(v[(i + n - 1) % n]),
                Corner::Old(v[i]),
                Corner::Old(v[(i + 1) % n]),
                Corner::New(y),
            ]);
            continue;
        }
        // The wedges strung between this slot's own new nodes.
        let c = &chain_of[i];
        let mut t = 0;
        while t + 2 < c.len() {
            owners.push(i);
            quads.push([
                Corner::Old(v[i]),
                Corner::New(at(c[t + 2])),
                Corner::New(at(c[t + 1])),
                Corner::New(at(c[t])),
            ]);
            t += 2;
        }
        // The quadrangle over the front edge to the successor, unless that
        // edge belongs to an end node's merged quadrangle.
        let j = (i + 1) % n;
        if k[j] >= 2 {
            let (Some(&head), Some(&tail)) = (chain_of[j].first(), c.last()) else {
                return Err(vec![i]);
            };
            owners.push(i);
            quads.push([
                Corner::Old(v[i]),
                Corner::Old(v[j]),
                Corner::New(at(head)),
                Corner::New(at(tail)),
            ]);
        }
    }

    let plan = RowPlan {
        pts: out_pts,
        chain,
        quads,
        owners,
    };
    let where_ = |c: Corner| match c {
        Corner::Old(i) => pts[i as usize],
        Corner::New(i) => plan.pts[i as usize],
    };
    let mut blame: Vec<usize> = plan
        .quads
        .iter()
        .zip(&plan.owners)
        .filter(|(q, _)| !quad_is_valid([where_(q[0]), where_(q[1]), where_(q[2]), where_(q[3])]))
        .map(|(_, &o)| o)
        .collect();
    if !blame.is_empty() {
        blame.sort_unstable();
        blame.dedup();
        return Err(blame);
    }
    Ok(plan)
}

fn push(v: &mut Vec<Point2>, p: Point2) -> u32 {
    v.push(p);
    (v.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A closed loop of `n` points per side of the unit square, walked
    /// counter-clockwise.
    fn square_loop(per_side: usize) -> (Front, Vec<Point2>) {
        let mut pts = Vec::new();
        let s = per_side as f64;
        for i in 0..per_side {
            pts.push(Point2::new(i as f64 / s, 0.0));
        }
        for i in 0..per_side {
            pts.push(Point2::new(1.0, i as f64 / s));
        }
        for i in 0..per_side {
            pts.push(Point2::new(1.0 - i as f64 / s, 1.0));
        }
        for i in 0..per_side {
            pts.push(Point2::new(0.0, 1.0 - i as f64 / s));
        }
        let mut f = Front::new();
        f.add_loop(&(0..pts.len() as u32).collect::<Vec<_>>());
        (f, pts)
    }

    #[test]
    fn the_classification_is_round_theta_over_a_right_angle() {
        assert_eq!(quads_at(0.3), 1);
        assert_eq!(quads_at(PI / 2.0), 1); // 90° sits on the 1|2 boundary
        assert_eq!(quads_at(PI), 2);
        assert_eq!(quads_at(3.0 * PI / 2.0), 3);
        assert_eq!(quads_at(1.9 * PI), 4);
        assert_eq!(quads_at(10.0), 4); // clamped
    }

    #[test]
    fn a_square_row_makes_one_quad_per_edge_less_the_corners() {
        let (f, pts) = square_loop(5);
        let rep = f.live_slots().next().unwrap();
        let plan = plan(&f, &pts, rep, &|_| 0.2, &|_| 0.2, false).unwrap();
        // 20 slots, 4 corners classified as ends.
        assert_eq!(plan.quads.len(), 16);
        assert_eq!(plan.chain.len(), 12);
    }

    #[test]
    fn the_row_tiles_the_strip_without_gap_or_overlap() {
        let (f, pts) = square_loop(5);
        let rep = f.live_slots().next().unwrap();
        let plan = plan(&f, &pts, rep, &|_| 0.2, &|_| 0.2, false).unwrap();
        let at = |c: Corner| match c {
            Corner::Old(i) => pts[i as usize],
            Corner::New(i) => plan.pts[i as usize],
        };
        // The strip between the old loop and the new one has area 1 - (area
        // enclosed by the chain); the quadrangles must add up to exactly that.
        let laid: f64 = plan
            .quads
            .iter()
            .map(|q| {
                let (a, b, c, d) = (at(q[0]), at(q[1]), at(q[2]), at(q[3]));
                0.5 * ((b - a).x * (c - a).y - (b - a).y * (c - a).x)
                    + 0.5 * ((c - a).x * (d - a).y - (c - a).y * (d - a).x)
            })
            .sum();
        let inner: Vec<Point2> = plan.chain.iter().map(|&i| plan.pts[i as usize]).collect();
        let inner_area = crate::ops::mesher::triangulation::signed_area(&inner);
        assert!(
            (laid + inner_area - 1.0).abs() < 1e-9,
            "laid {laid} + inner {inner_area} should tile the unit square"
        );
    }

    #[test]
    fn every_planned_quad_is_convex_and_counter_clockwise() {
        let (f, pts) = square_loop(6);
        let rep = f.live_slots().next().unwrap();
        let plan = plan(&f, &pts, rep, &|_| 0.15, &|_| 0.15, false).unwrap();
        let at = |c: Corner| match c {
            Corner::Old(i) => pts[i as usize],
            Corner::New(i) => plan.pts[i as usize],
        };
        for q in &plan.quads {
            assert!(quad_is_valid([at(q[0]), at(q[1]), at(q[2]), at(q[3])]));
        }
    }

    #[test]
    fn a_row_that_cannot_stay_convex_is_refused_with_the_slots_to_blame() {
        // An L-shaped loop: the reflex corner's node leans back over the
        // front, and past a certain advance the quadrangle beside it turns
        // reflex. `plan` refuses it and names the slots, so the caller can
        // retreat there instead of shrinking the whole row.
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 0.5),
            Point2::new(0.5, 0.5),
            Point2::new(0.5, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let mut f = Front::new();
        let rep = f.add_loop(&(0..6).collect::<Vec<_>>());
        assert!(plan(&f, &pts, rep, &|_| 0.12, &|_| 0.12, false).is_ok());
        let blame = plan(&f, &pts, rep, &|_| 1.4, &|_| 1.4, false).unwrap_err();
        assert!(!blame.is_empty(), "a refusal must say where");
        assert!(blame.iter().all(|&i| i < 6));
    }

    #[test]
    fn plan_judges_quadrangles_and_leaves_collisions_to_the_caller() {
        // Advancing a unit square by five units puts every new node far
        // outside the domain, yet each individual quadrangle is a long,
        // perfectly convex strip. `plan` therefore accepts it: whether the
        // advanced front crosses itself is a global question, answered by the
        // driver's `chain_is_free`, and keeping the two apart is what makes
        // each of them testable.
        let (f, pts) = square_loop(5);
        let rep = f.live_slots().next().unwrap();
        assert!(plan(&f, &pts, rep, &|_| 5.0, &|_| 5.0, false).is_ok());
    }

    #[test]
    fn a_row_cannot_change_the_loop_parity() {
        // `k = 1` contributes nothing and merges two nodes into one; every
        // other `k` contributes the odd count `2k - 3`. The two effects
        // cancel, so the advanced loop always has the parity of the old one.
        // Parity is therefore a property of the contour, and only a seam or
        // an edge split can change it — which is what the all-quadrangle
        // guarantee hangs on.
        for per_side in 3..9 {
            let (f, pts) = square_loop(per_side);
            let rep = f.live_slots().next().unwrap();
            let n = f.loop_len(rep);
            let h = 0.6 / per_side as f64;
            if let Ok(plan) = plan(&f, &pts, rep, &|_| h, &|_| h, false) {
                assert_eq!(
                    plan.chain.len() % 2,
                    n % 2,
                    "{per_side} per side: {} vs {n}",
                    plan.chain.len()
                );
            }
        }
    }

    #[test]
    fn a_loop_that_is_all_ends_and_sides_collapses_onto_one_node() {
        // Eight slots round a square: four 90° ends alternating with four
        // 180° sides. Every side's node is shared with both its neighbours,
        // so the union-find funnels them all onto a single centre and the
        // four merged quadrangles tile the square outright. This is the one
        // case where the parity argument above does not apply — and it does
        // not need to, since the loop is finished.
        // The advance is asked to match the front's own edge length, so
        // neither the refining nor the coarsening rule fires and the pure
        // angle classification is what is being exercised.
        let (f, pts) = square_loop(2);
        let rep = f.live_slots().next().unwrap();
        let plan = plan(&f, &pts, rep, &|_| 0.5, &|_| 0.5, false).unwrap();
        assert_eq!(plan.pts.len(), 1);
        assert_eq!(plan.chain.len(), 1);
        assert_eq!(plan.quads.len(), 4);
        let c = plan.pts[0];
        assert!(
            (c.x - 0.5).abs() < 1e-9 && (c.y - 0.5).abs() < 1e-9,
            "{c:?}"
        );
    }
}
