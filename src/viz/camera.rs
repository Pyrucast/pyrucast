//! Camera math: 3-D → 2-D orthographic projection and bounding-box helpers.
//!
//! The camera is described by a [`View`] (yaw, pitch, scale, target).
//! [`Projector`] builds three orthonormal axes (`right`, `up`, `forward`)
//! from the yaw and pitch angles and projects every world point onto the
//! screen plane:
//!
//! - `screen_x = (p - target) · right`
//! - `screen_y = (p - target) · up`
//! - `depth   = (p - target) · forward`  (positive = away from viewer)
//!
//! Used by every backend (PNG, SVG, interactive window) — the trait
//! [`Drawable`](crate::viz::Drawable) gets the same `Projector`
//! regardless. All vector / point arithmetic goes through
//! [`nalgebra`].

use crate::triangulation::{Point3, Vector3};
use crate::viz::View;

/// Axis-aligned bounding box in 3-D.
#[derive(Debug, Clone, Copy)]
pub struct Bbox3 {
    pub min: Point3,
    pub max: Point3,
}

impl Bbox3 {
    /// Empty bbox (min > max on every axis). Use [`extend`](Bbox3::extend)
    /// to grow it.
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Grow the bbox to include the point `p`.
    pub fn extend(&mut self, p: Point3) {
        self.min = Point3::from(self.min.coords.zip_map(&p.coords, f64::min));
        self.max = Point3::from(self.max.coords.zip_map(&p.coords, f64::max));
    }

    /// True when no point has been added.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    /// Geometric centre, or the origin when empty.
    pub fn center(&self) -> Point3 {
        if self.is_empty() {
            return Point3::origin();
        }
        Point3::from((self.min.coords + self.max.coords) * 0.5)
    }

    /// Length of the diagonal, or `0` when empty.
    pub fn diagonal(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        (self.max - self.min).norm()
    }
}

// ─── Projector ──────────────────────────────────────────────────────────────

/// Orthographic 3-D → 2-D projection.
///
/// `right`, `up` and `forward` form a right-handed orthonormal frame
/// centred on `target`.
#[derive(Debug, Clone, Copy)]
pub struct Projector {
    pub right: Vector3,
    pub up: Vector3,
    pub forward: Vector3,
    pub target: Point3,
}

impl Projector {
    /// Build a projector from a [`View`] and a fallback target (used when
    /// the view does not specify one — typically the bbox centre).
    pub fn new(view: &View, default_target: Point3) -> Self {
        let yaw = view.yaw.to_radians();
        let pitch = view.pitch.to_radians();

        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();

        // Camera position direction (unit sphere around target).
        let pos = Vector3::new(cp * cy, cp * sy, sp);
        let forward = -pos;

        // Right = forward × world_up, with a fallback when forward ‖ world_up
        // (i.e. straight-down or straight-up views).
        let world_up = Vector3::z();
        let mut right = forward.cross(&world_up);
        if right.norm() < 1e-12 {
            right = Vector3::x();
        }
        let right = right.normalize();
        let up = right.cross(&forward).normalize();

        Self {
            right,
            up,
            forward,
            target: view.target.unwrap_or(default_target),
        }
    }

    /// Project a world point. Returns `(screen_x, screen_y, depth)` where
    /// `depth` increases with distance from the viewer (i.e. it is the
    /// component along `forward`, which points from camera into the scene).
    pub fn project(&self, p: Point3) -> Vector3 {
        let d = p - self.target;
        Vector3::new(d.dot(&self.right), d.dot(&self.up), d.dot(&self.forward))
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn p3(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }

    #[test]
    fn bbox_extend_and_center() {
        let mut b = Bbox3::empty();
        assert!(b.is_empty());
        b.extend(p3(0.0, 0.0, 0.0));
        b.extend(p3(2.0, 4.0, 6.0));
        assert_eq!(b.center(), p3(1.0, 2.0, 3.0));
        assert!(approx(b.diagonal(), (4.0_f64 + 16.0 + 36.0).sqrt()));
    }

    #[test]
    fn front_view_projects_y_to_x_and_z_to_y() {
        let view = View::front();
        let proj = Projector::new(&view, Point3::origin());
        let p = proj.project(p3(5.0, 3.0, 2.0));
        // yaw=0, pitch=0 → camera at +X, right ≈ +Y, up ≈ +Z.
        assert!(approx(p.x, 3.0));
        assert!(approx(p.y, 2.0));
    }

    #[test]
    fn top_view_projects_x_y() {
        let view = View::top();
        let proj = Projector::new(&view, Point3::origin());
        let p = proj.project(p3(5.0, 3.0, 2.0));
        assert!(approx(p.x, 5.0));
        assert!(approx(p.y, 3.0));
    }

    #[test]
    fn projector_is_orthonormal() {
        let view = View::iso();
        let proj = Projector::new(&view, Point3::origin());
        assert!(approx(proj.right.dot(&proj.up), 0.0));
        assert!(approx(proj.right.dot(&proj.forward), 0.0));
        assert!(approx(proj.up.dot(&proj.forward), 0.0));
        assert!(approx(proj.right.norm(), 1.0));
        assert!(approx(proj.up.norm(), 1.0));
        assert!(approx(proj.forward.norm(), 1.0));
    }

    #[test]
    fn depth_increases_away_from_camera() {
        let proj = Projector::new(&View::front(), Point3::origin());
        // Camera at +X. Larger x = closer (smaller depth);
        // smaller x = farther (larger depth).
        let near = proj.project(p3(5.0, 0.0, 0.0));
        let far = proj.project(p3(-5.0, 0.0, 0.0));
        assert!(far.z > near.z);
    }

    #[test]
    fn target_override_recentres() {
        let view = View {
            target: Some(p3(10.0, 20.0, 30.0)),
            ..View::front()
        };
        let proj = Projector::new(&view, Point3::origin());
        let p = proj.project(p3(10.0, 25.0, 32.0));
        assert!(approx(p.x, 5.0));
        assert!(approx(p.y, 2.0));
    }
}
