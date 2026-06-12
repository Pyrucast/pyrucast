//! Field-aware colouring for mesh visualization.
//!
//! Given a [`crate::containers::node_field::NodeField`] sampled at the
//! nodes (consumed as a lock-free zone snapshot, first zone defining a
//! `(node, component)` pair wins), paint each mesh cell with the colour
//! the field value maps to under a standard colormap.
//!
//! Per-cell value = arithmetic mean of the field at the cell's nodes,
//! restricted to nodes that are in the field's support. Nodes absent
//! from the field do not contribute; a cell with no node in the
//! support gets value `0.0`.

use crate::aggregate::Aggregate;
use crate::containers::mesh::RgbColor;
use crate::containers::mesh::NodeId;
use crate::error::{PyrucastError, Result};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::FieldView;
use crate::store::read;
use crate::viz::camera::Bbox3;
use crate::viz::drawable::Drawable;
use crate::viz::mesh_draw::{render_primitives, submesh_primitives_with_colors, Primitive};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

/// A selectable colour scale for field plots.
///
/// Each variant maps a normalized position `t ∈ [0, 1]` to an RGB
/// colour. `Viridis` is the default: perceptually uniform and readable
/// in greyscale / for colour-vision deficiencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colormap {
    /// Perceptually uniform purple → blue → green → yellow.
    #[default]
    Viridis,
    /// Legacy blue → green → red ("jet-lite").
    Jet,
    /// Diverging blue → white → red; the midpoint of the scale is white,
    /// best for signed data centred on zero.
    CoolWarm,
    /// Thermal black → red → yellow → white.
    Hot,
    /// Greyscale black → white.
    Gray,
}

// Anchor tables: `(t, r, g, b)` control points, linearly interpolated by
// [`interp`]. `t` must be ascending, start at 0.0 and end at 1.0.
#[rustfmt::skip]
const VIRIDIS: &[(f64, u8, u8, u8)] = &[
    (0.0,  68,   1,  84), (0.1,  72,  35, 116), (0.2,  64,  67, 135),
    (0.3,  52,  94, 141), (0.4,  41, 120, 142), (0.5,  32, 144, 140),
    (0.6,  34, 167, 132), (0.7,  68, 190, 112), (0.8, 121, 209,  81),
    (0.9, 189, 222,  38), (1.0, 253, 231,  37),
];
#[rustfmt::skip]
const COOLWARM: &[(f64, u8, u8, u8)] = &[
    (0.0, 59, 76, 192), (0.5, 242, 242, 242), (1.0, 180, 4, 38),
];
#[rustfmt::skip]
const HOT: &[(f64, u8, u8, u8)] = &[
    (0.0, 0, 0, 0), (0.365, 255, 0, 0), (0.746, 255, 255, 0), (1.0, 255, 255, 255),
];

/// Piecewise-linear interpolation of an ascending anchor table at `t`.
fn interp(anchors: &[(f64, u8, u8, u8)], t: f64) -> RgbColor {
    let t = t.clamp(0.0, 1.0);
    for w in anchors.windows(2) {
        let (t0, r0, g0, b0) = w[0];
        let (t1, r1, g1, b1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let lerp = |a: u8, b: u8| (a as f64 + f * (b as f64 - a as f64)).round() as u8;
            return RgbColor::new(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1));
        }
    }
    let last = anchors[anchors.len() - 1];
    RgbColor::new(last.1, last.2, last.3)
}

impl Colormap {
    /// Parse a user-facing name (case-insensitive, common aliases
    /// accepted). Returns `None` for an unknown name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "viridis" => Some(Self::Viridis),
            "jet" | "jet-lite" | "jetlite" => Some(Self::Jet),
            "coolwarm" | "cool-warm" | "cool_warm" | "rdbu" => Some(Self::CoolWarm),
            "hot" => Some(Self::Hot),
            "gray" | "grey" | "grayscale" | "greyscale" => Some(Self::Gray),
            _ => None,
        }
    }

    /// Canonical names accepted by [`Colormap::from_name`], for help and
    /// error messages.
    pub fn names() -> &'static [&'static str] {
        &["viridis", "jet", "coolwarm", "hot", "gray"]
    }

    /// Sample the colormap at `t ∈ [0, 1]` (clamped outside the range).
    pub fn sample(self, t: f64) -> RgbColor {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => interp(VIRIDIS, t),
            Colormap::CoolWarm => interp(COOLWARM, t),
            Colormap::Hot => interp(HOT, t),
            Colormap::Gray => {
                let v = (t * 255.0).round() as u8;
                RgbColor::new(v, v, v)
            }
            Colormap::Jet => {
                let (r, g, b) = if t < 0.5 {
                    (0.0, 2.0 * t, 1.0 - 2.0 * t)
                } else {
                    (2.0 * t - 1.0, 2.0 - 2.0 * t, 0.0)
                };
                let to_u8 = |x: f64| (x * 255.0).round().clamp(0.0, 255.0) as u8;
                RgbColor::new(to_u8(r), to_u8(g), to_u8(b))
            }
        }
    }
}

/// Map `value` to a colour under `cmap`.
///
/// `value` is normalized via `t = (value - vmin) / (vmax - vmin)` and
/// clamped to `[0, 1]`. When `vmax ≤ vmin` (degenerate range) the
/// midpoint of the gradient (`t = 0.5`) is returned.
pub fn colormap(cmap: Colormap, value: f64, vmin: f64, vmax: f64) -> RgbColor {
    let t = if vmax > vmin {
        ((value - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    cmap.sample(t)
}

/// Mean of the field's `component` over the supplied node ids.
///
/// Nodes not in the field's support are simply ignored (they do not
/// contribute to the mean nor to the denominator). If **no** node is
/// in the support, returns `0.0`.
pub(crate) fn nodes_mean(
    field: &FieldView,
    node_ids: &[NodeId],
    component: &str,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &nid in node_ids {
        if let Some(v) = field.value_opt(nid, component) {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Per-cell field value for a `SubMesh`.
///
/// Length of the returned vector = number of cells in the submesh.
pub(crate) fn submesh_cell_values(
    sm: &SubMesh,
    field: &FieldView,
    component: &str,
) -> Result<Vec<f64>> {
    let conn = sm.connectivity();
    let npc = sm.element_type().nodes_per_cell();
    if npc == 0 || conn.is_empty() {
        return Ok(Vec::new());
    }
    let n_cells = conn.len() / npc;
    let mut out = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let ids = &conn[i * npc..(i + 1) * npc];
        out.push(nodes_mean(field, ids, component));
    }
    Ok(out)
}

/// Min / max of an arbitrary number of submesh-level value vectors.
/// Empty input → `(0.0, 1.0)` so the colormap stays sensible.
pub(crate) fn value_range<'a, I: IntoIterator<Item = &'a [f64]>>(values: I) -> (f64, f64) {
    let mut iter = values.into_iter().flat_map(|s| s.iter().copied());
    let Some(first) = iter.next() else {
        return (0.0, 1.0);
    };
    let mut mn = first;
    let mut mx = first;
    for v in iter {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

/// Build per-cell colours for one submesh from its per-cell values.
fn colors_from_values(values: &[f64], cmap: Colormap, vmin: f64, vmax: f64) -> Vec<RgbColor> {
    values.iter().map(|&v| colormap(cmap, v, vmin, vmax)).collect()
}

/// Compute the per-cell values of every submesh of a mesh, and the
/// global `(min, max)` range over all submeshes.
fn mesh_cell_values(
    mesh: &Mesh,
    field: &FieldView,
    component: &str,
) -> Result<(Vec<Vec<f64>>, f64, f64)> {
    let n_sub = mesh.len();
    let mut per_sub: Vec<Vec<f64>> = Vec::with_capacity(n_sub);
    for i in 0..n_sub {
        let sm = mesh.get(i)?;
        let vals = submesh_cell_values(&*read(&sm)?, field, component)?;
        per_sub.push(vals);
    }
    let slices: Vec<&[f64]> = per_sub.iter().map(|v| v.as_slice()).collect();
    let (vmin, vmax) = value_range(slices);
    Ok((per_sub, vmin, vmax))
}

/// Resolve the component name an end user wants to display.
///
/// `requested` `None` → first component of the field (its declared
/// primary component); `Some(name)` → check it exists.
pub(crate) fn resolve_component<'a>(
    field: &'a FieldView,
    requested: Option<&'a str>,
) -> Result<&'a str> {
    match requested {
        Some(name) => {
            if !field.components().iter().any(|c| c == name) {
                return Err(PyrucastError::Message(format!(
                    "field has no component named \"{}\" (available: {:?})",
                    name,
                    field.components()
                )));
            }
            Ok(name)
        }
        None => field.components().first().map(|s| s.as_str()).ok_or_else(|| {
            PyrucastError::Message("field has no components".into())
        }),
    }
}

// ─── Drawable wrappers ─────────────────────────────────────────────────────

/// `Drawable` over a [`Mesh`] coloured by a node field's component
/// (zone snapshot of a [`crate::containers::node_field::NodeField`]).
pub(crate) struct MeshFieldView<'a> {
    pub(crate) mesh: &'a Mesh,
    pub(crate) field: &'a FieldView,
    pub component: &'a str,
    /// Caller override for the colour-scale bounds; defaults to the
    /// data's own range.
    pub scale: crate::viz::ColorScale,
}

impl<'a> Drawable for MeshFieldView<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        self.mesh.bbox()
    }

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let (per_sub, dmin, dmax) =
            mesh_cell_values(self.mesh, self.field, self.component)?;
        let (vmin, vmax) = self.scale.resolve(dmin, dmax);
        let cmap = self.scale.cmap;
        let mut all_prims: Vec<Primitive> = Vec::new();
        for (i, values) in per_sub.iter().enumerate() {
            let sm = self.mesh.get(i)?;
            let colors = colors_from_values(values, cmap, vmin, vmax);
            let prims = submesh_primitives_with_colors(&*read(&sm)?, &colors)?;
            all_prims.extend(prims);
        }
        render_primitives(area, view, &all_prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        super::overlay::draw_colorbar(area, cmap, vmin, vmax)?;
        Ok(())
    }
}

/// `Drawable` over a single [`SubMesh`] coloured by a node field's
/// component (zone snapshot of a
/// [`crate::containers::node_field::NodeField`]).
pub(crate) struct SubMeshFieldView<'a> {
    pub(crate) submesh: &'a SubMesh,
    pub(crate) field: &'a FieldView,
    pub component: &'a str,
    /// Caller override for the colour-scale bounds; defaults to the
    /// data's own range.
    pub scale: crate::viz::ColorScale,
}

impl<'a> Drawable for SubMeshFieldView<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        self.submesh.bbox()
    }

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let values = submesh_cell_values(self.submesh, self.field, self.component)?;
        let (dmin, dmax) = value_range([values.as_slice()]);
        let (vmin, vmax) = self.scale.resolve(dmin, dmax);
        let cmap = self.scale.cmap;
        let colors = colors_from_values(&values, cmap, vmin, vmax);
        let prims = submesh_primitives_with_colors(self.submesh, &colors)?;
        render_primitives(area, view, &prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        super::overlay::draw_colorbar(area, cmap, vmin, vmax)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::SubMesh as RawSubMesh;
    use crate::containers::mesh::Node;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::store::insert;

    #[test]
    fn colormap_endpoints() {
        let c0 = colormap(Colormap::Jet, 0.0, 0.0, 1.0);
        assert_eq!(c0, RgbColor::new(0, 0, 255));
        let c1 = colormap(Colormap::Jet, 1.0, 0.0, 1.0);
        assert_eq!(c1, RgbColor::new(255, 0, 0));
        let c_mid = colormap(Colormap::Jet, 0.5, 0.0, 1.0);
        assert_eq!(c_mid, RgbColor::new(0, 255, 0));
    }

    #[test]
    fn colormap_clamps_out_of_range() {
        let lo = colormap(Colormap::Jet, -99.0, 0.0, 1.0);
        let hi = colormap(Colormap::Jet, 99.0, 0.0, 1.0);
        assert_eq!(lo, RgbColor::new(0, 0, 255));
        assert_eq!(hi, RgbColor::new(255, 0, 0));
    }

    #[test]
    fn colormap_degenerate_range_returns_midpoint() {
        // Jet midpoint is green; the degenerate range falls back to t=0.5.
        assert_eq!(colormap(Colormap::Jet, 7.0, 7.0, 7.0), RgbColor::new(0, 255, 0));
    }

    #[test]
    fn colormap_endpoints_for_each_scale() {
        // Anchor tables / formulas must hit their declared endpoints.
        assert_eq!(Colormap::Viridis.sample(0.0), RgbColor::new(68, 1, 84));
        assert_eq!(Colormap::Viridis.sample(1.0), RgbColor::new(253, 231, 37));
        assert_eq!(Colormap::CoolWarm.sample(0.0), RgbColor::new(59, 76, 192));
        assert_eq!(Colormap::CoolWarm.sample(0.5), RgbColor::new(242, 242, 242));
        assert_eq!(Colormap::CoolWarm.sample(1.0), RgbColor::new(180, 4, 38));
        assert_eq!(Colormap::Hot.sample(0.0), RgbColor::new(0, 0, 0));
        assert_eq!(Colormap::Hot.sample(1.0), RgbColor::new(255, 255, 255));
        assert_eq!(Colormap::Gray.sample(0.0), RgbColor::new(0, 0, 0));
        assert_eq!(Colormap::Gray.sample(1.0), RgbColor::new(255, 255, 255));
        assert_eq!(Colormap::Gray.sample(0.5), RgbColor::new(128, 128, 128));
    }

    #[test]
    fn colormap_from_name_and_default() {
        assert_eq!(Colormap::from_name("Viridis"), Some(Colormap::Viridis));
        assert_eq!(Colormap::from_name("  JET "), Some(Colormap::Jet));
        assert_eq!(Colormap::from_name("coolwarm"), Some(Colormap::CoolWarm));
        assert_eq!(Colormap::from_name("grey"), Some(Colormap::Gray));
        assert_eq!(Colormap::from_name("nope"), None);
        assert_eq!(Colormap::default(), Colormap::Viridis);
    }

    #[test]
    fn submesh_cell_values_averages_node_values_per_cell() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = RawSubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let sm_h = insert(sm);

        // Build a POI1 SubNodeField with T values [1, 2, 3] on a, b, c.
        let mut poi1 = RawSubMesh::new(cfg.clone(), ElementType::POI1);
        for n in [&a, &b, &c] {
            poi1.add_cell(&[n.id()]).unwrap();
        }
        let poi1_h = insert(poi1);
        let mut nf = SubNodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 1.0).unwrap();
        nf.set_value(b.id(), "T", 2.0).unwrap();
        nf.set_value(c.id(), "T", 3.0).unwrap();
        let view = NodeField::from_sub(nf).view().unwrap();

        let values = submesh_cell_values(&*read(&sm_h).unwrap(), &view, "T").unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 2.0).abs() < 1e-12); // (1 + 2 + 3) / 3
    }

    #[test]
    fn nodes_mean_ignores_nodes_outside_field_support() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        // Field defined only on `a`.
        let mut poi1 = RawSubMesh::new(cfg.clone(), ElementType::POI1);
        poi1.add_cell(&[a.id()]).unwrap();
        let poi1_h = insert(poi1);
        let mut nf = SubNodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 4.0).unwrap();
        let view = NodeField::from_sub(nf).view().unwrap();
        // Two-node "cell": only `a` is in the field; mean = 4.0.
        let mean = nodes_mean(&view, &[a.id(), b.id()], "T");
        assert!((mean - 4.0).abs() < 1e-12);
    }

    #[test]
    fn value_range_handles_empty_and_constant() {
        assert_eq!(value_range(std::iter::empty()), (0.0, 1.0));
        let v = vec![3.0, 3.0, 3.0];
        let (mn, mx) = value_range([v.as_slice()]);
        assert_eq!((mn, mx), (3.0, 3.0));
    }
}
