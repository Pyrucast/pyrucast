//! Camera math: 3D → 2D orthographic projection and bounding box helpers.
//!
//! The camera is described by a [`View`]. The projector
//! builds three orthonormal axes (`right`, `up`, `forward`) from the yaw
//! and pitch angles, then projects every world point onto the screen
//! plane:
//!
//! - `screen_x = (p - target) · right`
//! - `screen_y = (p - target) · up`
//! - `depth   = (p - target) · (-forward)`  (positive = away from viewer)
//!
//! Used by every backend (PNG, SVG, interactive window) — the trait
//! [`Drawable`](crate::viz::Drawable) gets the same `Projector` regardless.

use crate::viz::View;

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy)]
pub struct Bbox3 {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Bbox3 {
    /// Empty bbox (min > max on every axis). Use [`extend`](Bbox3::extend)
    /// to grow it.
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    /// Grow the bbox to include the point `p`.
    pub fn extend(&mut self, p: [f64; 3]) {
        for k in 0..3 {
            if p[k] < self.min[k] {
                self.min[k] = p[k];
            }
            if p[k] > self.max[k] {
                self.max[k] = p[k];
            }
        }
    }

    /// True when no point has been added.
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0]
    }

    /// Geometric centre, or `[0, 0, 0]` when empty.
    pub fn center(&self) -> [f64; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Length of the diagonal, or `0` when empty.
    pub fn diagonal(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ─── Projector ──────────────────────────────────────────────────────────────

/// Orthographic 3D → 2D projection.
///
/// `right`, `up` and `forward` form a right-handed orthonormal frame
/// centred on `target`.
#[derive(Debug, Clone, Copy)]
pub struct Projector {
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub forward: [f64; 3],
    pub target: [f64; 3],
}

impl Projector {
    /// Build a projector from a [`View`] and a fallback target (used when
    /// the view does not specify one — typically the bbox centre).
    pub fn new(view: &View, default_target: [f64; 3]) -> Self {
        let yaw = view.yaw.to_radians();
        let pitch = view.pitch.to_radians();

        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();

        // Camera position direction (unit sphere around target).
        let pos = [cp * cy, cp * sy, sp];
        let forward = [-pos[0], -pos[1], -pos[2]];

        // Right = forward × world_up, with a fallback when forward ‖ world_up
        // (i.e. straight-down or straight-up views).
        let world_up: [f64; 3] = [0.0, 0.0, 1.0];
        let mut right = cross(forward, world_up);
        if length(right) < 1e-12 {
            right = [1.0, 0.0, 0.0];
        }
        let right = normalize(right);
        let up = normalize(cross(right, forward));

        Self {
            right,
            up,
            forward,
            target: view.target.unwrap_or(default_target),
        }
    }

    /// Project a world point. Returns `[screen_x, screen_y, depth]` where
    /// `depth` increases with distance from the viewer (i.e. it is the
    /// component along `forward`, which points from camera into the scene).
    pub fn project(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [
            p[0] - self.target[0],
            p[1] - self.target[1],
            p[2] - self.target[2],
        ];
        let sx = dot(d, self.right);
        let sy = dot(d, self.up);
        let depth = dot(d, self.forward);
        [sx, sy, depth]
    }
}

// ─── Vector helpers (kept local to avoid an extra dep) ──────────────────────

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = length(a);
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn bbox_extend_and_center() {
        let mut b = Bbox3::empty();
        assert!(b.is_empty());
        b.extend([0.0, 0.0, 0.0]);
        b.extend([2.0, 4.0, 6.0]);
        assert_eq!(b.center(), [1.0, 2.0, 3.0]);
        assert!(approx(b.diagonal(), (4.0_f64 + 16.0 + 36.0).sqrt()));
    }

    #[test]
    fn front_view_projects_y_to_x_and_z_to_y() {
        let view = View::front();
        let proj = Projector::new(&view, [0.0; 3]);
        let p = proj.project([5.0, 3.0, 2.0]);
        // yaw=0, pitch=0 → camera at +X, right ≈ +Y, up ≈ +Z.
        assert!(approx(p[0], 3.0)); // y
        assert!(approx(p[1], 2.0)); // z
    }

    #[test]
    fn top_view_projects_x_y() {
        let view = View::top();
        let proj = Projector::new(&view, [0.0; 3]);
        let p = proj.project([5.0, 3.0, 2.0]);
        // Looking straight down: screen X = x, screen Y = y.
        assert!(approx(p[0], 5.0));
        assert!(approx(p[1], 3.0));
    }

    #[test]
    fn projector_is_orthonormal() {
        let view = View::iso();
        let proj = Projector::new(&view, [0.0; 3]);
        assert!(approx(dot(proj.right, proj.up), 0.0));
        assert!(approx(dot(proj.right, proj.forward), 0.0));
        assert!(approx(dot(proj.up, proj.forward), 0.0));
        assert!(approx(length(proj.right), 1.0));
        assert!(approx(length(proj.up), 1.0));
        assert!(approx(length(proj.forward), 1.0));
    }

    #[test]
    fn depth_increases_away_from_camera() {
        let proj = Projector::new(&View::front(), [0.0; 3]);
        // Camera at +X. Larger x = closer (smaller depth);
        // smaller x = farther (larger depth).
        let near = proj.project([5.0, 0.0, 0.0]);
        let far = proj.project([-5.0, 0.0, 0.0]);
        assert!(far[2] > near[2]);
    }

    #[test]
    fn target_override_recentres() {
        let view = View {
            target: Some([10.0, 20.0, 30.0]),
            ..View::front()
        };
        let proj = Projector::new(&view, [0.0; 3]);
        let p = proj.project([10.0, 25.0, 32.0]);
        // y, z relative to target.
        assert!(approx(p[0], 5.0));
        assert!(approx(p[1], 2.0));
    }
}
