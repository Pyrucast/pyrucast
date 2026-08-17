//! Node selection by geometric region — the Cast3m `POIN … PLAN / DROIT /
//! CYLI / SPHE / …` family.
//!
//! Every operator here answers the same question with a different region:
//! *which nodes of this mesh sit in (or on) that shape?* They all share one
//! shape, so they compose and read alike:
//!
//! ```text
//! points_<in|on|below>_<shape>(mesh, …geometry…, tol) -> Mesh (POI1)
//! ```
//!
//! The result mirrors `mesh` **submesh by submesh** — same order, one POI1
//! submesh per input submesh, **possibly empty** — so a selection keeps the
//! zoning of its source and the caller can tell *which* zone a node came
//! from. Use [`consolidate`](fn@super::consolidate) to fuse the zones into a
//! single point cloud, or index the result to work zone by zone.
//!
//! The one query that does *not* return a mesh is the nearest node, which by
//! construction picks exactly one node. It is therefore not an operator at all
//! but a **method** of the mesh —
//! [`Mesh::nearest_node`](crate::containers::mesh::Mesh::nearest_node) — and it
//! returns a [`Node`](crate::atoms::Node).
//!
//! # Tolerance
//!
//! Every operator takes `tol`, the geometric precision of the test, measured
//! as a **distance to the region's surface**. `tol = None` asks for the
//! default: `1e-6 ×` the diagonal of the mesh bounding box — the scale-free
//! choice, which is what makes `points_on_plane` usable at all on nodes whose
//! coordinates come out of a mesher rather than out of exact arithmetic.
//!
//! The two families read as:
//!
//! - `points_in_*` — inside the closed region, **grown** by `tol`;
//! - `points_on_*` — within `tol` of the region's surface, on either side.
//!
//! # Coordinates
//!
//! Nodes are tested in the coordinates they are **stored** in. Under an
//! axisymmetric [`CoordinateFrame`](crate::coords::CoordinateFrame)
//! that is the meridian half-plane `(r, z)`, not the 3-D body of revolution:
//! a "sphere" there is a circle in the `(r, z)` plane, not a sphere in space.

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::Handle;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public operators — spheres
// ---------------------------------------------------------------------------

/// Nodes **inside** the sphere of centre `center` and radius `radius`
/// (Cast3m `POIN … DEDANS` on a sphere).
///
/// Keeps the nodes at a distance `≤ radius + tol` from `center` — the closed
/// ball, grown by the tolerance. In 2-D the "sphere" is a disc, in 1-D a
/// segment: `center` simply has to match the mesh dimension.
///
/// See the [module documentation](self) for the shared result layout (one
/// POI1 submesh per input submesh) and the meaning of `tol` (`None` ⇒
/// `1e-6 ×` the bounding-box diagonal).
///
/// Errors if `center`'s length is not the coordinate dimension, if `radius`
/// is negative, or if `tol` is negative.
pub fn points_in_sphere(
    mesh: &Mesh,
    center: &[f64],
    radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_in_sphere")?;
    ctx.check_point(center, "center")?;
    ctx.check_radius(radius, "radius")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| distance(x, center) <= radius + tol)
}

/// Nodes **on** the sphere of centre `center` and radius `radius`
/// (Cast3m `POIN … SPHE`).
///
/// Keeps the nodes whose distance to `center` is within `tol` of `radius` —
/// a spherical shell of thickness `2 tol`, so nodes just inside and just
/// outside both make it. In 2-D this selects a circle.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `center`'s length is not the coordinate dimension, if `radius`
/// is negative, or if `tol` is negative.
pub fn points_on_sphere(
    mesh: &Mesh,
    center: &[f64],
    radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_sphere")?;
    ctx.check_point(center, "center")?;
    ctx.check_radius(radius, "radius")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| (distance(x, center) - radius).abs() <= tol)
}

// ---------------------------------------------------------------------------
// Public operators — planes
// ---------------------------------------------------------------------------

/// Nodes **on** the plane through `origin` with normal `normal`
/// (Cast3m `POIN … PLAN`).
///
/// Keeps the nodes whose signed distance to the plane is at most `tol` in
/// absolute value — a slab of thickness `2 tol` centred on the plane. This
/// is *the* way to grab a boundary face of a box mesh to pin a boundary
/// condition on. `normal` need not be normalized. In 2-D the "plane" is a
/// line, in 3-D a plane: `origin` and `normal` just have to match the mesh
/// dimension.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `origin` or `normal` do not have the coordinate dimension, if
/// `normal` is the zero vector, or if `tol` is negative.
pub fn points_on_plane(
    mesh: &Mesh,
    origin: &[f64],
    normal: &[f64],
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_plane")?;
    ctx.check_point(origin, "origin")?;
    let n = ctx.unit_vector(normal, "normal")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| signed_gap(x, origin, &n).abs() <= tol)
}

/// Nodes **below** the plane through `origin` with normal `normal` — the
/// half-space the normal points *away* from.
///
/// Keeps the nodes whose signed distance `(x − origin) · n̂` is `≤ tol`, so
/// the plane itself is included (and, like everywhere in this module, a `tol`
/// band beyond it). There is no separate `points_above_plane`: flip the
/// normal, `points_below_plane(mesh, o, [-nx, -ny, -nz], tol)` is the other
/// half-space.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `origin` or `normal` do not have the coordinate dimension, if
/// `normal` is the zero vector, or if `tol` is negative.
pub fn points_below_plane(
    mesh: &Mesh,
    origin: &[f64],
    normal: &[f64],
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_below_plane")?;
    ctx.check_point(origin, "origin")?;
    let n = ctx.unit_vector(normal, "normal")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| signed_gap(x, origin, &n) <= tol)
}

// ---------------------------------------------------------------------------
// Public operators — line
// ---------------------------------------------------------------------------

/// Nodes **on** the (infinite) line through `a` and `b`
/// (Cast3m `POIN … DROIT`).
///
/// Keeps the nodes at a distance `≤ tol` from the line — a cylinder of radius
/// `tol` about it, unbounded in both directions. For a *bounded* selection
/// use [`points_in_cylinder`] with a small radius: it clips the axial extent
/// to the `a → b` segment.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `a` or `b` do not have the coordinate dimension, if they
/// coincide (no direction), or if `tol` is negative.
pub fn points_on_line(mesh: &Mesh, a: &[f64], b: &[f64], tol: Option<f64>) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_line")?;
    ctx.check_point(a, "a")?;
    ctx.check_point(b, "b")?;
    let axis = ctx.axis(a, b, "a", "b")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| axis.radius_at(x).1 <= tol)
}

// ---------------------------------------------------------------------------
// Public operators — cylinders
// ---------------------------------------------------------------------------

/// Nodes **inside** the finite cylinder of axis `base → top` and radius
/// `radius` (Cast3m `POIN … DEDANS` on a cylinder).
///
/// The cylinder is **capped**: a node is kept when it is within `radius + tol`
/// of the axis *and* its axial coordinate lies between the two end sections
/// (again with a `tol` margin). Use [`points_on_line`] instead for an
/// unbounded selection about the same axis.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `base` or `top` do not have the coordinate dimension, if they
/// coincide (zero-length axis), if `radius` is negative, or if `tol` is
/// negative.
pub fn points_in_cylinder(
    mesh: &Mesh,
    base: &[f64],
    top: &[f64],
    radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_in_cylinder")?;
    ctx.check_point(base, "base")?;
    ctx.check_point(top, "top")?;
    ctx.check_radius(radius, "radius")?;
    let axis = ctx.axis(base, top, "base", "top")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| {
        let (t, rho) = axis.radius_at(x);
        axis.spans(t, tol) && rho <= radius + tol
    })
}

/// Nodes **on** the lateral surface of the finite cylinder of axis
/// `base → top` and radius `radius` (Cast3m `POIN … CYLI`).
///
/// Keeps the nodes within `tol` of the *tube* — distance to the axis within
/// `tol` of `radius` — whose axial coordinate lies between the two end
/// sections. The end **discs are not part of the selection**: they are two
/// flat faces, and [`points_on_plane`] cuts those. This is the usual way to
/// grab the bore of a tube or the outside of a shaft.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `base` or `top` do not have the coordinate dimension, if they
/// coincide (zero-length axis), if `radius` is negative, or if `tol` is
/// negative.
pub fn points_on_cylinder(
    mesh: &Mesh,
    base: &[f64],
    top: &[f64],
    radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_cylinder")?;
    ctx.check_point(base, "base")?;
    ctx.check_point(top, "top")?;
    ctx.check_radius(radius, "radius")?;
    let axis = ctx.axis(base, top, "base", "top")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| {
        let (t, rho) = axis.radius_at(x);
        axis.spans(t, tol) && (rho - radius).abs() <= tol
    })
}

// ---------------------------------------------------------------------------
// Public operators — cones
// ---------------------------------------------------------------------------

/// Nodes **inside** the cone of axis `base → top`, radius `base_radius` at
/// `base` and `top_radius` at `top` (Cast3m `POIN … DEDANS` on a cone).
///
/// The shape is a **truncated cone** (a frustum), which covers the two
/// degenerate cases people actually mean by "cone": `top_radius = 0` gives a
/// true cone whose apex is `top`, and `top_radius = base_radius` gives a
/// cylinder. It is capped like [`points_in_cylinder`]: a node is kept when
/// its distance to the axis is at most the local radius (grown by `tol`) and
/// its axial coordinate lies between the two end sections.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `base` or `top` do not have the coordinate dimension, if they
/// coincide, if either radius is negative, or if `tol` is negative.
pub fn points_in_cone(
    mesh: &Mesh,
    base: &[f64],
    top: &[f64],
    base_radius: f64,
    top_radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_in_cone")?;
    ctx.check_point(base, "base")?;
    ctx.check_point(top, "top")?;
    ctx.check_radius(base_radius, "base_radius")?;
    ctx.check_radius(top_radius, "top_radius")?;
    let axis = ctx.axis(base, top, "base", "top")?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| {
        let (t, rho) = axis.radius_at(x);
        axis.spans(t, tol) && rho <= axis.taper(t, base_radius, top_radius) + tol
    })
}

/// Nodes **on** the lateral surface of the cone of axis `base → top`, radius
/// `base_radius` at `base` and `top_radius` at `top` (Cast3m `POIN … CONE`).
///
/// Keeps the nodes within `tol` of the slanted surface — the distance is the
/// **perpendicular** one, not the radial one, so the band stays `tol` wide
/// however steep the cone is — and whose axial coordinate lies between the
/// two end sections. As for [`points_on_cylinder`] the end discs are not part
/// of the selection.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if `base` or `top` do not have the coordinate dimension, if they
/// coincide, if either radius is negative, or if `tol` is negative.
pub fn points_on_cone(
    mesh: &Mesh,
    base: &[f64],
    top: &[f64],
    base_radius: f64,
    top_radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_cone")?;
    ctx.check_point(base, "base")?;
    ctx.check_point(top, "top")?;
    ctx.check_radius(base_radius, "base_radius")?;
    ctx.check_radius(top_radius, "top_radius")?;
    let axis = ctx.axis(base, top, "base", "top")?;
    let tol = ctx.tol;
    // The surface is slanted: a radial gap of `g` is a perpendicular distance
    // of `g / √(1 + k²)`, with `k` the radius growth per unit length.
    let k = (top_radius - base_radius) / axis.length;
    let band = tol * (1.0 + k * k).sqrt();
    ctx.select(mesh, |x| {
        let (t, rho) = axis.radius_at(x);
        axis.spans(t, tol) && (rho - axis.taper(t, base_radius, top_radius)).abs() <= band
    })
}

// ---------------------------------------------------------------------------
// Public operators — tori
// ---------------------------------------------------------------------------

/// Nodes **inside** the torus of centre `center`, axis `axis`, major radius
/// `major_radius` and minor radius `minor_radius`.
///
/// The torus has a **circular section**: it is the set of points at a
/// distance `≤ minor_radius` from the *directrix*, the circle of radius
/// `major_radius` drawn around `center` in the plane normal to `axis`. This
/// operator keeps the nodes inside that tube, grown by `tol` — the natural
/// way to select the material around a fillet or a rounded groove. `axis`
/// need not be normalized.
///
/// **3-D only**: a torus needs an out-of-plane axis to be a torus at all.
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if the mesh is not 3-D, if `center` or `axis` do not have three
/// components, if `axis` is the zero vector, if either radius is negative, or
/// if `tol` is negative.
pub fn points_in_torus(
    mesh: &Mesh,
    center: &[f64],
    axis: &[f64],
    major_radius: f64,
    minor_radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_in_torus")?;
    let torus = ctx.torus(center, axis, major_radius, minor_radius)?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| torus.gap(x) <= minor_radius + tol)
}

/// Nodes **on** the torus of centre `center`, axis `axis`, major radius
/// `major_radius` and minor radius `minor_radius`.
///
/// Keeps the nodes within `tol` of the tube's surface — those at a distance
/// from the directrix circle (see [`points_in_torus`]) within `tol` of
/// `minor_radius`. The torus being closed, there is no cap to worry about
/// here, unlike the cylinder and the cone.
///
/// **3-D only**, same as [`points_in_torus`].
///
/// See the [module documentation](self) for the shared result layout and the
/// meaning of `tol`.
///
/// Errors if the mesh is not 3-D, if `center` or `axis` do not have three
/// components, if `axis` is the zero vector, if either radius is negative, or
/// if `tol` is negative.
pub fn points_on_torus(
    mesh: &Mesh,
    center: &[f64],
    axis: &[f64],
    major_radius: f64,
    minor_radius: f64,
    tol: Option<f64>,
) -> Result<Mesh> {
    let ctx = Context::new(mesh, tol, "points_on_torus")?;
    let torus = ctx.torus(center, axis, major_radius, minor_radius)?;
    let tol = ctx.tol;
    ctx.select(mesh, |x| (torus.gap(x) - minor_radius).abs() <= tol)
}

// ---------------------------------------------------------------------------
// Shared engine
// ---------------------------------------------------------------------------

/// What every operator of this module needs before it can test a node: the
/// mesh dimension (to validate the geometry arguments), the resolved
/// tolerance, and the operator name (to prefix its error messages).
struct Context {
    dim: usize,
    tol: f64,
    op: &'static str,
}

impl Context {
    /// Read the dimension off the mesh and resolve `tol` (`None` ⇒ default).
    ///
    /// An empty mesh has no `Coords` and therefore no dimension; `dim` is left
    /// at 0 and every check passes trivially, since [`Self::select`] has
    /// nothing to select from anyway.
    fn new(mesh: &Mesh, tol: Option<f64>, op: &'static str) -> Result<Self> {
        if let Some(t) = tol {
            if t < 0.0 || t.is_nan() {
                return Err(PyrucastError::Message(format!(
                    "{op}: tol must be ≥ 0, got {t}"
                )));
            }
        }
        if mesh.is_empty() {
            return Ok(Self {
                dim: 0,
                tol: tol.unwrap_or(0.0),
                op,
            });
        }
        let dim = mesh.coords()?.read().dim() as usize;
        let tol = match tol {
            Some(t) => t,
            None => default_tol(mesh)?,
        };
        Ok(Self { dim, tol, op })
    }

    fn check_point(&self, p: &[f64], name: &str) -> Result<()> {
        if self.dim != 0 && p.len() != self.dim {
            return Err(PyrucastError::Message(format!(
                "{}: {name} has {} coordinates, mesh is {}-D",
                self.op,
                p.len(),
                self.dim
            )));
        }
        Ok(())
    }

    fn check_radius(&self, r: f64, name: &str) -> Result<()> {
        if r < 0.0 || r.is_nan() {
            return Err(PyrucastError::Message(format!(
                "{}: {name} must be ≥ 0, got {r}",
                self.op
            )));
        }
        Ok(())
    }

    /// Validate a direction vector and return it normalized.
    fn unit_vector(&self, v: &[f64], name: &str) -> Result<Vec<f64>> {
        self.check_point(v, name)?;
        let n = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        if n == 0.0 {
            return Err(PyrucastError::Message(format!(
                "{}: {name} must not be the zero vector",
                self.op
            )));
        }
        Ok(v.iter().map(|c| c / n).collect())
    }

    /// Validate a two-point axis and return it in the form the predicates use.
    fn axis(&self, a: &[f64], b: &[f64], a_name: &str, b_name: &str) -> Result<Axis> {
        let delta: Vec<f64> = b.iter().zip(a).map(|(bi, ai)| bi - ai).collect();
        let length = delta.iter().map(|c| c * c).sum::<f64>().sqrt();
        if length == 0.0 {
            return Err(PyrucastError::Message(format!(
                "{}: {a_name} and {b_name} must not coincide (zero-length axis)",
                self.op
            )));
        }
        Ok(Axis {
            origin: a.to_vec(),
            direction: delta.iter().map(|c| c / length).collect(),
            length,
        })
    }

    /// Validate the torus arguments (3-D only) and return the tested shape.
    fn torus(&self, center: &[f64], axis: &[f64], major: f64, minor: f64) -> Result<Torus> {
        if self.dim != 0 && self.dim != 3 {
            return Err(PyrucastError::Message(format!(
                "{}: a torus needs a 3-D mesh, this one is {}-D",
                self.op, self.dim
            )));
        }
        self.check_point(center, "center")?;
        self.check_radius(major, "major_radius")?;
        self.check_radius(minor, "minor_radius")?;
        Ok(Torus {
            center: center.to_vec(),
            axis: self.unit_vector(axis, "axis")?,
            major,
        })
    }

    /// Keep the nodes of every submesh that satisfy `keep`, as POI1.
    ///
    /// One output submesh per input submesh, in the same order, holding the
    /// selected nodes **de-duplicated in order of first appearance** — the
    /// ordering [`SubMesh::to_poi1`] uses, so a full selection reproduces
    /// [`to_poi1`](fn@super::to_poi1) node for node. A submesh with no
    /// selected node yields an empty POI1 submesh rather than disappearing:
    /// the caller can index the result alongside the source.
    fn select(&self, mesh: &Mesh, keep: impl Fn(&[f64]) -> bool) -> Result<Mesh> {
        let mut out = Mesh::empty();
        for sm in mesh {
            // Snapshot the node list, then release the submesh guard: building
            // the POI1 increfs in `Coords`, which takes a write lock, and no
            // long-lived read guard may straddle that.
            let (coords, nodes) = {
                let s = sm.read();
                let conn = s.connectivity();
                let mut known: HashSet<NodeId> = HashSet::with_capacity(conn.len());
                let unique: Vec<NodeId> = conn
                    .iter()
                    .copied()
                    .filter(|&nid| known.insert(nid))
                    .collect();
                (s.coords(), unique)
            };
            let kept: Vec<NodeId> = {
                let c = coords.read();
                let mut kept = Vec::new();
                for nid in nodes {
                    if keep(c.position(nid)?) {
                        kept.push(nid);
                    }
                }
                kept
            };
            out.add_sub(Handle::new(SubMesh::poi1_from_node_ids(coords, &kept)?))?;
        }
        Ok(out)
    }
}

/// An axis given by two points: the predicates need the origin, the unit
/// direction and the length (to clip the axial extent and to taper a cone).
struct Axis {
    origin: Vec<f64>,
    direction: Vec<f64>,
    length: f64,
}

impl Axis {
    /// Axial coordinate of `x` (0 at the origin, `length` at the far end) and
    /// its distance to the axis.
    fn radius_at(&self, x: &[f64]) -> (f64, f64) {
        let t: f64 = x
            .iter()
            .zip(&self.origin)
            .zip(&self.direction)
            .map(|((xi, oi), di)| (xi - oi) * di)
            .sum();
        let rho2: f64 = x
            .iter()
            .zip(&self.origin)
            .zip(&self.direction)
            .map(|((xi, oi), di)| {
                let perp = (xi - oi) - t * di;
                perp * perp
            })
            .sum();
        (t, rho2.sqrt())
    }

    /// Whether the axial coordinate `t` falls between the two end sections,
    /// with a `tol` margin at each end.
    fn spans(&self, t: f64, tol: f64) -> bool {
        t >= -tol && t <= self.length + tol
    }

    /// Radius of a cone at axial coordinate `t`, clamped to the end sections
    /// so the `tol` margins above see a flat extension rather than a radius
    /// running past the apex.
    fn taper(&self, t: f64, base_radius: f64, top_radius: f64) -> f64 {
        let s = (t / self.length).clamp(0.0, 1.0);
        base_radius + (top_radius - base_radius) * s
    }
}

/// A torus by its centre, its (unit) axis and its major radius — the minor
/// radius is what the predicates compare [`Torus::gap`] against, so it stays
/// outside.
struct Torus {
    center: Vec<f64>,
    axis: Vec<f64>,
    major: f64,
}

impl Torus {
    /// Distance from `x` to the directrix circle — the tube's centre line.
    fn gap(&self, x: &[f64]) -> f64 {
        let z: f64 = x
            .iter()
            .zip(&self.center)
            .zip(&self.axis)
            .map(|((xi, ci), ai)| (xi - ci) * ai)
            .sum();
        let rho2: f64 = x
            .iter()
            .zip(&self.center)
            .zip(&self.axis)
            .map(|((xi, ci), ai)| {
                let radial = (xi - ci) - z * ai;
                radial * radial
            })
            .sum();
        let in_plane = rho2.sqrt() - self.major;
        (in_plane * in_plane + z * z).sqrt()
    }
}

/// Euclidean distance between two points of the same dimension.
fn distance(x: &[f64], p: &[f64]) -> f64 {
    x.iter()
        .zip(p)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Signed distance from `x` to the plane `(origin, unit normal)`.
fn signed_gap(x: &[f64], origin: &[f64], normal: &[f64]) -> f64 {
    x.iter()
        .zip(origin)
        .zip(normal)
        .map(|((xi, oi), ni)| (xi - oi) * ni)
        .sum()
}

/// Default geometric precision: `1e-6 ×` the diagonal of the mesh bounding
/// box.
///
/// Tying the tolerance to the model's own size is what makes the operators
/// scale-free — the same call works on a millimetre bracket and on a
/// kilometre dam. A mesh whose nodes are all coincident has no scale to speak
/// of and gets an exact test (`0`).
fn default_tol(mesh: &Mesh) -> Result<f64> {
    let coords = mesh.coords()?;
    let c = coords.read();
    let dim = c.dim() as usize;
    let mut lo = vec![f64::INFINITY; dim];
    let mut hi = vec![f64::NEG_INFINITY; dim];
    for sm in mesh {
        let s = sm.read();
        for &nid in s.connectivity() {
            for (k, &v) in c.position(nid)?.iter().enumerate() {
                lo[k] = lo[k].min(v);
                hi[k] = hi[k].max(v);
            }
        }
    }
    let diagonal = lo
        .iter()
        .zip(&hi)
        .map(|(l, h)| {
            let d = h - l;
            if d.is_finite() {
                d * d
            } else {
                0.0 // no node at all: nothing to scale to.
            }
        })
        .sum::<f64>()
        .sqrt();
    Ok(1e-6 * diagonal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::coords::Coords;
    use crate::store::Handle;

    /// A 3×3 grid of POI1 nodes over [0, 2]², one node per integer pair, as a
    /// single-submesh mesh. Returns the mesh and the node ids row by row
    /// (`y` slowest).
    fn grid2d() -> (Mesh, Vec<NodeId>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        let mut ids = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let n = Node::create_in(coords.clone(), &[i as f64, j as f64]).unwrap();
                mesh.add_cell(&[n.id()]).unwrap();
                ids.push(n.id());
            }
        }
        (mesh, ids)
    }

    /// The node ids of a single-submesh POI1 selection, in order.
    fn selected(mesh: &Mesh) -> Vec<NodeId> {
        let s = mesh.items()[0].read();
        s.connectivity().to_vec()
    }

    #[test]
    fn sphere_selects_disc_then_circle() {
        let (mesh, n) = grid2d();

        // Disc of radius 1 about the centre (1, 1): the centre and its four
        // edge neighbours; the corners are at √2.
        let inside = points_in_sphere(&mesh, &[1.0, 1.0], 1.0, None).unwrap();
        assert_eq!(selected(&inside), vec![n[1], n[3], n[4], n[5], n[7]]);

        // The circle itself drops the centre.
        let on = points_on_sphere(&mesh, &[1.0, 1.0], 1.0, None).unwrap();
        assert_eq!(selected(&on), vec![n[1], n[3], n[5], n[7]]);

        // Radius √2 on the circle picks the four corners instead.
        let corners = points_on_sphere(&mesh, &[1.0, 1.0], 2f64.sqrt(), None).unwrap();
        assert_eq!(selected(&corners), vec![n[0], n[2], n[6], n[8]]);
    }

    #[test]
    fn plane_cuts_a_row_and_a_half_space() {
        let (mesh, n) = grid2d();

        // The line y = 1, normal +y: the middle row.
        let on = points_on_plane(&mesh, &[0.0, 1.0], &[0.0, 1.0], None).unwrap();
        assert_eq!(selected(&on), vec![n[3], n[4], n[5]]);

        // Below it: the middle row and the one under it. The normal need not
        // be normalized.
        let below = points_below_plane(&mesh, &[0.0, 1.0], &[0.0, 7.0], None).unwrap();
        assert_eq!(selected(&below), vec![n[0], n[1], n[2], n[3], n[4], n[5]]);

        // Flipping the normal gives the complementary half-space.
        let above = points_below_plane(&mesh, &[0.0, 1.0], &[0.0, -1.0], None).unwrap();
        assert_eq!(selected(&above), vec![n[3], n[4], n[5], n[6], n[7], n[8]]);
    }

    #[test]
    fn line_is_unbounded_where_the_cylinder_is_capped() {
        let (mesh, n) = grid2d();

        // The diagonal through (0, 0) and (1, 1) — the segment given is short,
        // but the line is infinite, so (2, 2) is on it too.
        let on = points_on_line(&mesh, &[0.0, 0.0], &[1.0, 1.0], None).unwrap();
        assert_eq!(selected(&on), vec![n[0], n[4], n[8]]);

        // The same axis as a thin cylinder stops at the end section.
        let capped = points_in_cylinder(&mesh, &[0.0, 0.0], &[1.0, 1.0], 1e-9, None).unwrap();
        assert_eq!(selected(&capped), vec![n[0], n[4]]);
    }

    #[test]
    fn cylinder_lateral_surface_excludes_the_caps() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        // On the tube (r = 1, z within [0, 2]), on the axis at mid-height,
        // on the bottom cap disc, and past the top.
        let pts = [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 3.0],
        ];
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|p| {
                let n = Node::create_in(coords.clone(), p).unwrap();
                mesh.add_cell(&[n.id()]).unwrap();
                n.id()
            })
            .collect();

        let base = [0.0, 0.0, 0.0];
        let top = [0.0, 0.0, 2.0];

        // Lateral surface only: the cap point is inside the disc, not on the
        // tube, and the fourth point is beyond the end section.
        let on = points_on_cylinder(&mesh, &base, &top, 1.0, None).unwrap();
        assert_eq!(selected(&on), vec![ids[0]]);

        // The solid holds the first three.
        let inside = points_in_cylinder(&mesh, &base, &top, 1.0, None).unwrap();
        assert_eq!(selected(&inside), vec![ids[0], ids[1], ids[2]]);
    }

    #[test]
    fn cone_tapers_between_its_end_radii() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        // Cone from radius 2 at z = 0 to an apex at z = 2: the local radius is
        // 1 at mid-height. On the surface, inside, and outside.
        let pts = [
            [1.0, 0.0, 1.0], // exactly on the slant
            [0.4, 0.0, 1.0], // well inside
            [1.6, 0.0, 1.0], // outside
        ];
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|p| {
                let n = Node::create_in(coords.clone(), p).unwrap();
                mesh.add_cell(&[n.id()]).unwrap();
                n.id()
            })
            .collect();

        let base = [0.0, 0.0, 0.0];
        let apex = [0.0, 0.0, 2.0];

        let on = points_on_cone(&mesh, &base, &apex, 2.0, 0.0, None).unwrap();
        assert_eq!(selected(&on), vec![ids[0]]);

        let inside = points_in_cone(&mesh, &base, &apex, 2.0, 0.0, None).unwrap();
        assert_eq!(selected(&inside), vec![ids[0], ids[1]]);

        // Equal radii degenerate to a cylinder: all three are within r = 2.
        let cyl = points_in_cone(&mesh, &base, &apex, 2.0, 2.0, None).unwrap();
        assert_eq!(selected(&cyl), vec![ids[0], ids[1], ids[2]]);
    }

    #[test]
    fn torus_is_a_tube_around_its_directrix() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        // Directrix: the circle of radius 2 in z = 0; minor radius 0.5.
        let pts = [
            [2.5, 0.0, 0.0], // on the tube, outer equator
            [2.0, 0.0, 0.0], // on the directrix, inside the tube
            [0.0, 0.0, 0.0], // the hole's centre, outside
            [0.0, 2.0, 0.5], // on the tube, a quarter turn away, top
        ];
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|p| {
                let n = Node::create_in(coords.clone(), p).unwrap();
                mesh.add_cell(&[n.id()]).unwrap();
                n.id()
            })
            .collect();

        let center = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];

        let on = points_on_torus(&mesh, &center, &axis, 2.0, 0.5, None).unwrap();
        assert_eq!(selected(&on), vec![ids[0], ids[3]]);

        let inside = points_in_torus(&mesh, &center, &axis, 2.0, 0.5, None).unwrap();
        assert_eq!(selected(&inside), vec![ids[0], ids[1], ids[3]]);
    }

    #[test]
    fn result_mirrors_the_submeshes_including_the_empty_ones() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 5.0]).unwrap();

        // Submesh 0 sits on y = 0, submesh 1 sits high above it.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let far = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[c.id()]).unwrap();
            Handle::new(sm)
        };
        mesh.add_sub(far).unwrap();

        let sel = points_on_plane(&mesh, &[0.0, 0.0], &[0.0, 1.0], None).unwrap();
        assert_eq!(sel.len(), 2, "one POI1 submesh per input submesh");
        assert_eq!(
            sel.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::POI1]
        );
        // The second zone selects nothing, and stays as an empty submesh.
        assert_eq!(sel.cell_counts().unwrap(), vec![2, 0]);
    }

    #[test]
    fn nodes_are_de_duplicated_in_order_of_first_appearance() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();

        // Two segments sharing node b: b appears twice in the connectivity.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[b.id(), c.id()]).unwrap();

        let sel = points_on_plane(&mesh, &[0.0, 0.0], &[0.0, 1.0], None).unwrap();
        assert_eq!(selected(&sel), vec![a.id(), b.id(), c.id()]);
    }

    #[test]
    fn explicit_tolerance_widens_the_band() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.01]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        // The default precision (1e-6 × diagonal) leaves the second node out.
        let tight = points_on_plane(&mesh, &[0.0, 0.0], &[0.0, 1.0], None).unwrap();
        assert_eq!(selected(&tight), vec![a.id()]);

        // A tolerance wider than its offset takes it in.
        let loose = points_on_plane(&mesh, &[0.0, 0.0], &[0.0, 1.0], Some(0.02)).unwrap();
        assert_eq!(selected(&loose), vec![a.id(), b.id()]);
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let (mesh, _) = grid2d();

        // Wrong dimension for the geometry argument.
        assert!(points_in_sphere(&mesh, &[0.0, 0.0, 0.0], 1.0, None).is_err());
        // Negative radius and negative tolerance.
        assert!(points_in_sphere(&mesh, &[0.0, 0.0], -1.0, None).is_err());
        assert!(points_in_sphere(&mesh, &[0.0, 0.0], 1.0, Some(-1e-9)).is_err());
        // Degenerate direction and axis.
        assert!(points_on_plane(&mesh, &[0.0, 0.0], &[0.0, 0.0], None).is_err());
        assert!(points_on_line(&mesh, &[1.0, 1.0], &[1.0, 1.0], None).is_err());
        // A torus needs a 3-D mesh.
        assert!(points_in_torus(&mesh, &[0.0, 0.0], &[0.0, 1.0], 1.0, 0.1, None).is_err());
    }

    #[test]
    fn empty_mesh_selects_nothing() {
        let sel = points_in_sphere(&Mesh::empty(), &[0.0, 0.0], 1.0, None).unwrap();
        assert_eq!(sel.len(), 0);
    }
}
