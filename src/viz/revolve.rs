//! Revolution of an axisymmetric plot into the 3-D body it describes.
//!
//! An [axisymmetric](crate::containers::mesh::Coords::axisymmetric) mesh is the
//! meridian half-plane `(r, z)` of a body of revolution: drawn as it stands, it
//! shows a flat 2-D section. A [`Revolve`] set on the
//! [`View`](crate::viz::View) sweeps that section around the axis `r = 0` and
//! draws the solid instead — the section keeps being the computed object, only
//! the picture changes.
//!
//! The sweep is applied to the **rendering primitives**, just before projection
//! (see [`render_primitives`](crate::viz::mesh_draw::render_primitives)), so
//! every plot inherits it identically: plain mesh, wireframe, field colouring
//! (flat or interpolated) and evolution frames.
//!
//! What each primitive becomes:
//!
//! - a **face** sweeps into a ring of matter. Only what can be seen is emitted:
//!   the lateral band swept by each *boundary* edge of the section (an edge
//!   shared by two cells stays buried inside the matter), plus a copy of the
//!   whole section at both ends when the sweep is partial;
//! - a **wire** — the element outline of the interpolated rendering — follows
//!   the same rule as a stroke-only band, so the element grid stays drawn on
//!   the swept surface;
//! - **segments** and **points** are repeated at every angular station, and the
//!   circles their endpoints describe are added: that is exactly the wireframe
//!   (resp. the node cloud) of the swept mesh.
//!
//! World placement: the meridian point `(r, z)` maps to
//! `(r·cos θ, r·sin θ, z)`, so the axis of revolution is the world Z axis — the
//! one the camera's `yaw` turns around.

use std::collections::HashMap;

use crate::containers::mesh::{Point3, RgbColor};
use crate::error::{PyrucastError, Result};
#[cfg(any(test, feature = "viz-interactive"))]
use crate::viz::camera::Bbox3;
use crate::viz::mesh_draw::{primitives_bbox, Primitive};

// ─── User-facing descriptor ─────────────────────────────────────────────────

/// How an axisymmetric meridian plot is swept into a 3-D body.
///
/// Held by [`View::revolve`](crate::viz::View::revolve) — `None` keeps the flat
/// `(r, z)` section, `Some` draws the body of revolution. The swept `angle` is
/// in degrees; a partial sweep (`angle < 360`) cuts the body open and shows the
/// meridian section at both ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Revolve {
    angle: f64,
    sectors: usize,
}

impl Revolve {
    /// Angular size of one sector when the count is derived from the angle.
    const DEGREES_PER_SECTOR: f64 = 10.0;
    /// Fewest sectors a sweep is discretized into, however small the angle.
    const MIN_SECTORS: usize = 3;

    /// Full turn (360°), the default sweep.
    pub fn full() -> Self {
        Self {
            angle: 360.0,
            sectors: 36,
        }
    }

    /// Sweep of `angle` degrees, discretized into one sector per 10° (at
    /// least [`MIN_SECTORS`](Self::MIN_SECTORS)).
    pub fn new(angle: f64) -> Result<Self> {
        let sectors = if angle.is_finite() && angle > 0.0 {
            ((angle / Self::DEGREES_PER_SECTOR).round() as usize).max(Self::MIN_SECTORS)
        } else {
            Self::MIN_SECTORS
        };
        Self::with_sectors(angle, sectors)
    }

    /// Sweep of `angle` degrees discretized into `sectors` angular steps —
    /// raise it for a smoother silhouette, lower it for a lighter picture.
    pub fn with_sectors(angle: f64, sectors: usize) -> Result<Self> {
        if !(angle.is_finite() && angle > 0.0 && angle <= 360.0) {
            return Err(PyrucastError::Message(format!(
                "revolve: the swept angle must lie in ]0, 360] degrees, got {angle}"
            )));
        }
        if sectors == 0 {
            return Err(PyrucastError::Message(
                "revolve: the sweep needs at least one sector".into(),
            ));
        }
        Ok(Self { angle, sectors })
    }

    /// Swept angle, in degrees.
    pub fn angle(&self) -> f64 {
        self.angle
    }

    /// Number of angular sectors the sweep is discretized into.
    pub fn sectors(&self) -> usize {
        self.sectors
    }

    /// Whether the sweep closes on itself (no end section to draw).
    fn is_full(&self) -> bool {
        self.angle >= 360.0 - 1e-9
    }

    /// The `(cos θ, sin θ)` stations of the sweep: `sectors + 1` values from
    /// `θ = 0` to `θ = angle`.
    fn stations(&self) -> Vec<(f64, f64)> {
        let step = self.angle.to_radians() / self.sectors as f64;
        (0..=self.sectors)
            .map(|k| {
                let theta = k as f64 * step;
                (theta.cos(), theta.sin())
            })
            .collect()
    }
}

impl Default for Revolve {
    fn default() -> Self {
        Self::full()
    }
}

// ─── Geometry ───────────────────────────────────────────────────────────────

/// Image of a meridian point `(r = x, z = y)` at the angular station
/// `(cos θ, sin θ)`.
fn rotate(p: Point3, (cos, sin): (f64, f64)) -> Point3 {
    Point3::new(p.x * cos, p.x * sin, p.y)
}

/// Bounding box of the body swept from the meridian box `bb`.
///
/// Conservative for a partial sweep (the full ring is used), which only costs a
/// little framing margin — the picture itself is sized from the primitives.
/// Only the interactive window needs it: a file export frames itself from the
/// primitives it just swept.
#[cfg(any(test, feature = "viz-interactive"))]
pub(crate) fn revolved_bbox(bb: &Bbox3) -> Bbox3 {
    if bb.is_empty() {
        return *bb;
    }
    let r = bb.max.x.abs().max(bb.min.x.abs());
    Bbox3 {
        min: Point3::new(-r, -r, bb.min.y),
        max: Point3::new(r, r, bb.max.y),
    }
}

// ─── Boundary of the meridian section ───────────────────────────────────────

/// One edge of the meridian section, ready to be swept into a lateral band.
struct Band {
    a: Point3,
    b: Point3,
    color: RgbColor,
    outline: bool,
}

/// Vertex identity key, quantized to `inv_tol⁻¹`: the sub-vertices of the
/// interpolated rendering are recomputed element by element, so two cells
/// sharing an edge may land a few ulps apart.
fn vertex_key(p: Point3, inv_tol: f64) -> [i64; 3] {
    [
        (p.x * inv_tol).round() as i64,
        (p.y * inv_tol).round() as i64,
        (p.z * inv_tol).round() as i64,
    ]
}

/// Edges belonging to a **single** polygon — the boundary of the meridian
/// section. An edge shared by two polygons lies inside the matter: swept, it
/// would be hidden by the ring itself, so it is dropped.
///
/// Insertion order is preserved (the hash map only counts), so the emitted
/// picture does not depend on the hashing.
fn boundary_edges<'a, I>(polygons: I, inv_tol: f64) -> Vec<Band>
where
    I: Iterator<Item = (&'a [Point3], RgbColor, bool)>,
{
    let mut bands: Vec<Band> = Vec::new();
    let mut count: Vec<u32> = Vec::new();
    let mut seen: HashMap<[[i64; 3]; 2], usize> = HashMap::new();
    for (verts, color, outline) in polygons {
        let n = verts.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            let (ka, kb) = (vertex_key(a, inv_tol), vertex_key(b, inv_tol));
            let key = if ka <= kb { [ka, kb] } else { [kb, ka] };
            match seen.get(&key) {
                Some(&idx) => count[idx] += 1,
                None => {
                    seen.insert(key, bands.len());
                    bands.push(Band {
                        a,
                        b,
                        color,
                        outline,
                    });
                    count.push(1);
                }
            }
        }
    }
    bands
        .into_iter()
        .zip(count)
        .filter(|(_, c)| *c == 1)
        .map(|(band, _)| band)
        .collect()
}

/// Sweep one boundary edge into the lateral band it describes, one polygon per
/// angular sector. An edge touching the axis sweeps into triangles; an edge
/// lying **on** the axis sweeps into nothing.
fn sweep_band(
    out: &mut Vec<Primitive>,
    band: &Band,
    stations: &[(f64, f64)],
    axis_eps: f64,
    wire: bool,
) {
    let a_on_axis = band.a.x.abs() <= axis_eps;
    let b_on_axis = band.b.x.abs() <= axis_eps;
    if a_on_axis && b_on_axis {
        return;
    }
    for w in stations.windows(2) {
        let (s0, s1) = (w[0], w[1]);
        let verts = if a_on_axis {
            vec![rotate(band.a, s0), rotate(band.b, s0), rotate(band.b, s1)]
        } else if b_on_axis {
            vec![rotate(band.a, s0), rotate(band.b, s0), rotate(band.a, s1)]
        } else {
            vec![
                rotate(band.a, s0),
                rotate(band.b, s0),
                rotate(band.b, s1),
                rotate(band.a, s1),
            ]
        };
        out.push(if wire {
            Primitive::Wire { verts }
        } else {
            Primitive::Face {
                verts,
                color: band.color,
                outline: band.outline,
            }
        });
    }
}

/// The circles described by the endpoints of the segment primitives, as one
/// chord per sector — the circumferential edges of the swept mesh, without
/// which a revolved wireframe would read as a handful of loose combs.
fn sweep_rings(
    out: &mut Vec<Primitive>,
    prims: &[Primitive],
    stations: &[(f64, f64)],
    inv_tol: f64,
    axis_eps: f64,
) {
    let mut seen: HashMap<[i64; 3], ()> = HashMap::new();
    for prim in prims {
        let Primitive::Segment { a, b, color } = prim else {
            continue;
        };
        for p in [a, b] {
            if p.x.abs() <= axis_eps {
                continue; // a node on the axis stays a node
            }
            if seen.insert(vertex_key(*p, inv_tol), ()).is_some() {
                continue;
            }
            for w in stations.windows(2) {
                out.push(Primitive::Segment {
                    a: rotate(*p, w[0]),
                    b: rotate(*p, w[1]),
                    color: *color,
                });
            }
        }
    }
}

// ─── Entry point ────────────────────────────────────────────────────────────

/// Sweep a meridian primitive list into the primitives of the body of
/// revolution. See the module documentation for what becomes of each kind.
pub(crate) fn revolve_primitives(prims: &[Primitive], rev: Revolve) -> Vec<Primitive> {
    if prims.is_empty() {
        return Vec::new();
    }
    let extent = primitives_bbox(prims).diagonal().max(f64::MIN_POSITIVE);
    let inv_tol = 1.0 / (1e-7 * extent);
    let axis_eps = 1e-9 * extent;

    let stations = rev.stations();
    // On a full turn the closing station repeats the first one, so a copy of
    // the section there would be drawn twice.
    let copies = if rev.is_full() {
        &stations[..stations.len() - 1]
    } else {
        &stations[..]
    };

    let mut out: Vec<Primitive> = Vec::new();

    // Faces → the skin of the swept solid.
    let faces = boundary_edges(
        prims.iter().filter_map(|p| match p {
            Primitive::Face {
                verts,
                color,
                outline,
            } => Some((verts.as_slice(), *color, *outline)),
            _ => None,
        }),
        inv_tol,
    );
    for band in &faces {
        sweep_band(&mut out, band, &stations, axis_eps, false);
    }

    // Wires (element outlines) → the element grid on that skin.
    let wires = boundary_edges(
        prims.iter().filter_map(|p| match p {
            Primitive::Wire { verts } => Some((verts.as_slice(), RgbColor::new(0, 0, 0), false)),
            _ => None,
        }),
        inv_tol,
    );
    for band in &wires {
        sweep_band(&mut out, band, &stations, axis_eps, true);
    }

    // End sections of a partial sweep: the meridian faces themselves, at both
    // ends of the swept angle.
    if !rev.is_full() {
        for &station in [stations[0], stations[stations.len() - 1]].iter() {
            for prim in prims {
                match prim {
                    Primitive::Face {
                        verts,
                        color,
                        outline,
                    } => out.push(Primitive::Face {
                        verts: verts.iter().map(|&v| rotate(v, station)).collect(),
                        color: *color,
                        outline: *outline,
                    }),
                    Primitive::Wire { verts } => out.push(Primitive::Wire {
                        verts: verts.iter().map(|&v| rotate(v, station)).collect(),
                    }),
                    _ => {}
                }
            }
        }
    }

    // Points and segments: the nodes and the meridian edges of the swept mesh.
    for &station in copies {
        for prim in prims {
            match prim {
                Primitive::Point { p, color } => out.push(Primitive::Point {
                    p: rotate(*p, station),
                    color: *color,
                }),
                Primitive::Segment { a, b, color } => out.push(Primitive::Segment {
                    a: rotate(*a, station),
                    b: rotate(*b, station),
                    color: *color,
                }),
                _ => {}
            }
        }
    }
    sweep_rings(&mut out, prims, &stations, inv_tol, axis_eps);

    out
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn face(verts: Vec<Point3>) -> Primitive {
        Primitive::Face {
            verts,
            color: RgbColor::new(1, 2, 3),
            outline: true,
        }
    }

    /// The unit square `1 ≤ r ≤ 2`, `0 ≤ z ≤ 1`, as two side-by-side cells
    /// sharing the edge `r = 1.5` (which is interior to the section).
    fn two_cells() -> Vec<Primitive> {
        let p = |r: f64, z: f64| Point3::new(r, z, 0.0);
        vec![
            face(vec![p(1.0, 0.0), p(1.5, 0.0), p(1.5, 1.0), p(1.0, 1.0)]),
            face(vec![p(1.5, 0.0), p(2.0, 0.0), p(2.0, 1.0), p(1.5, 1.0)]),
        ]
    }

    fn faces_of(prims: &[Primitive]) -> Vec<&Vec<Point3>> {
        prims
            .iter()
            .filter_map(|p| match p {
                Primitive::Face { verts, .. } => Some(verts),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn angle_must_be_a_positive_turn_at_most() {
        assert!(Revolve::with_sectors(0.0, 4).is_err());
        assert!(Revolve::with_sectors(-90.0, 4).is_err());
        assert!(Revolve::with_sectors(361.0, 4).is_err());
        assert!(Revolve::with_sectors(f64::NAN, 4).is_err());
        assert!(Revolve::with_sectors(360.0, 0).is_err());
        assert!(Revolve::with_sectors(90.0, 4).is_ok());
    }

    #[test]
    fn sector_count_follows_the_angle() {
        assert_eq!(Revolve::new(360.0).unwrap().sectors(), 36);
        assert_eq!(Revolve::new(90.0).unwrap().sectors(), 9);
        // Never below the floor, however thin the slice.
        assert_eq!(Revolve::new(1.0).unwrap().sectors(), Revolve::MIN_SECTORS);
        assert_eq!(Revolve::full().sectors(), 36);
    }

    #[test]
    fn interior_edges_do_not_sweep() {
        let rev = Revolve::with_sectors(360.0, 4).unwrap();
        let out = revolve_primitives(&two_cells(), rev);
        // Boundary of the section: 6 edges (2 tops, 2 bottoms, 2 sides) — the
        // shared edge at r = 1.5 is interior. 4 sectors each.
        assert_eq!(faces_of(&out).len(), 6 * 4);
    }

    #[test]
    fn a_partial_sweep_adds_both_end_sections() {
        let rev = Revolve::with_sectors(90.0, 4).unwrap();
        let out = revolve_primitives(&two_cells(), rev);
        // Same 24 lateral bands, plus a copy of both cells at each end.
        assert_eq!(faces_of(&out).len(), 6 * 4 + 2 * 2);
    }

    #[test]
    fn the_swept_body_spans_the_full_radius() {
        let rev = Revolve::with_sectors(360.0, 36).unwrap();
        let out = revolve_primitives(&two_cells(), rev);
        let bb = primitives_bbox(&out);
        // r goes up to 2 in every direction of the plane, z stays in [0, 1].
        assert!((bb.max.x - 2.0).abs() < 1e-9, "{:?}", bb);
        assert!((bb.min.x + 2.0).abs() < 1e-9, "{:?}", bb);
        assert!((bb.max.y - 2.0).abs() < 1e-9, "{:?}", bb);
        assert!((bb.min.y + 2.0).abs() < 1e-9, "{:?}", bb);
        assert!((bb.max.z - 1.0).abs() < 1e-12, "{:?}", bb);
        assert!(bb.min.z.abs() < 1e-12, "{:?}", bb);
    }

    #[test]
    fn an_edge_touching_the_axis_sweeps_into_triangles() {
        let p = |r: f64, z: f64| Point3::new(r, z, 0.0);
        // A triangle with a single vertex on the axis: the two edges reaching
        // r = 0 sweep into cones, the third one into a band of quadrangles.
        let prims = vec![face(vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0)])];
        let rev = Revolve::with_sectors(360.0, 4).unwrap();
        let out = revolve_primitives(&prims, rev);
        let verts = faces_of(&out);
        assert_eq!(verts.len(), 3 * 4);
        assert_eq!(verts.iter().filter(|v| v.len() == 3).count(), 2 * 4);
        assert_eq!(verts.iter().filter(|v| v.len() == 4).count(), 4);
    }

    #[test]
    fn an_edge_on_the_axis_sweeps_into_nothing() {
        let p = |r: f64, z: f64| Point3::new(r, z, 0.0);
        let prims = vec![face(vec![p(0.0, 0.0), p(0.0, 1.0), p(1.0, 1.0)])];
        let rev = Revolve::with_sectors(360.0, 4).unwrap();
        let out = revolve_primitives(&prims, rev);
        // Only the two edges off the axis sweep.
        assert_eq!(faces_of(&out).len(), 2 * 4);
    }

    #[test]
    fn segments_are_repeated_and_ringed() {
        let color = RgbColor::new(9, 9, 9);
        let prims = vec![Primitive::Segment {
            a: Point3::new(1.0, 0.0, 0.0),
            b: Point3::new(2.0, 0.0, 0.0),
            color,
        }];
        let rev = Revolve::with_sectors(360.0, 4).unwrap();
        let out = revolve_primitives(&prims, rev);
        // 4 copies of the meridian segment + one 4-chord ring per endpoint.
        assert_eq!(out.len(), 4 + 2 * 4);
    }

    #[test]
    fn points_are_repeated_once_per_station() {
        let prims = vec![Primitive::Point {
            p: Point3::new(1.0, 0.0, 0.0),
            color: RgbColor::new(0, 0, 0),
        }];
        // A full turn must not draw the closing station twice.
        let full = revolve_primitives(&prims, Revolve::with_sectors(360.0, 6).unwrap());
        assert_eq!(full.len(), 6);
        // A partial sweep keeps both ends.
        let part = revolve_primitives(&prims, Revolve::with_sectors(180.0, 6).unwrap());
        assert_eq!(part.len(), 7);
    }

    #[test]
    fn revolved_bbox_covers_the_ring() {
        let bb = Bbox3 {
            min: Point3::new(1.0, -2.0, 0.0),
            max: Point3::new(3.0, 5.0, 0.0),
        };
        let rb = revolved_bbox(&bb);
        assert_eq!(rb.min, Point3::new(-3.0, -3.0, -2.0));
        assert_eq!(rb.max, Point3::new(3.0, 3.0, 5.0));
        assert!(revolved_bbox(&Bbox3::empty()).is_empty());
    }
}
