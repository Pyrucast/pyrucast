//! Field-aware colouring for mesh visualization.
//!
//! Paints each mesh cell with the colour its field value maps to under a
//! standard colormap. Two field kinds are accepted uniformly through
//! `FieldData`:
//!
//! - **node fields** ([`crate::containers::node_field::NodeField`]) —
//!   values live at the nodes; the per-cell nodal values are read
//!   directly (first zone defining a `(node, component)` pair wins);
//! - **element fields** ([`crate::containers::element_field::ElementField`])
//!   — values live at the Gauss points; the per-cell nodal values come
//!   from a least-squares fit of the Lagrange interpolant to the cell's
//!   Gauss values, **local to that cell**. No averaging ever happens
//!   across neighbouring elements: inter-element discontinuities are
//!   physical and must stay visible. With fewer Gauss points than nodes
//!   (e.g. a single point) the fit degenerates to the Gauss mean —
//!   a constant colour over the element.
//!
//! Flat rendering (one colour per cell) uses the arithmetic mean of
//! those per-cell nodal values.

use crate::aggregate::Aggregate;
use crate::atoms::RgbColor;
use crate::containers::element_field::{ElementFieldView, SubElementField};
use crate::containers::field::SubField;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::NodeFieldView;
use crate::error::{PyrucastError, Result};
use crate::store::{read, Handle};
use crate::viz::camera::Bbox3;
use crate::viz::drawable::Drawable;
use crate::viz::mesh_draw::{render_primitives, submesh_primitives_with_colors, Primitive};
use crate::viz::View;
use nalgebra as na;
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

// ─── Uniform field source ───────────────────────────────────────────────────

/// The two field kinds the viz layer colours by, behind one interface.
/// Holds the zero-copy views (owned read guards on every zone).
pub(crate) enum FieldData {
    Node(NodeFieldView),
    Element(ElementFieldView),
}

impl FieldData {
    /// Union of the zones' component names, first-seen order.
    pub(crate) fn components(&self) -> &[String] {
        match self {
            FieldData::Node(v) => v.components(),
            FieldData::Element(v) => v.components(),
        }
    }

    /// Drawing context specialised for one submesh: resolves the
    /// Element zone (and its fit operator) once per submesh.
    pub(crate) fn for_submesh(&self, sm: &Handle<SubMesh>) -> Result<SubmeshFieldCtx<'_>> {
        match self {
            FieldData::Node(v) => Ok(SubmeshFieldCtx::Node(v)),
            FieldData::Element(v) => {
                let zone = v.zone_for_submesh(sm)?.ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "no ElementField zone lives on submesh #{} — the field's \
                         FE space does not cover this (sub)mesh",
                        sm.index()
                    ))
                })?;
                let fit = FitOperator::for_zone(zone)?;
                Ok(SubmeshFieldCtx::Element { zone, fit })
            }
        }
    }
}

/// Field context specialised for one submesh (see [`FieldData::for_submesh`]).
pub(crate) enum SubmeshFieldCtx<'a> {
    Node(&'a NodeFieldView),
    Element {
        zone: &'a SubElementField,
        fit: FitOperator,
    },
}

impl SubmeshFieldCtx<'_> {
    /// Nodal values of `component` **for drawing cell `cell`** of `sm`
    /// (length = nodes-per-cell, in connectivity order).
    ///
    /// - Node field: direct lookup; a node missing from the support takes
    ///   the mean of the present ones (all missing → zeros), so the flat
    ///   per-cell mean matches the historical behaviour exactly.
    /// - Element field: least-squares fit local to the cell (see
    ///   [`FitOperator`]) — values are private to this cell's rendering.
    pub(crate) fn cell_node_values(
        &self,
        sm: &SubMesh,
        cell: usize,
        component: &str,
    ) -> Result<Vec<f64>> {
        match self {
            SubmeshFieldCtx::Node(view) => {
                let npc = sm.element_type().nodes_per_cell();
                let conn = sm.connectivity();
                let ids = &conn[cell * npc..(cell + 1) * npc];
                let raw: Vec<Option<f64>> = ids
                    .iter()
                    .map(|&nid| view.value_opt(nid, component))
                    .collect();
                let present: Vec<f64> = raw.iter().filter_map(|v| *v).collect();
                let fill = if present.is_empty() {
                    0.0
                } else {
                    present.iter().sum::<f64>() / present.len() as f64
                };
                Ok(raw.into_iter().map(|v| v.unwrap_or(fill)).collect())
            }
            SubmeshFieldCtx::Element { zone, fit } => {
                let ci = zone.component_index(component).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "ElementField zone has no component named \"{component}\""
                    ))
                })?;
                let mut gauss = Vec::with_capacity(zone.gauss_count());
                for g in 0..zone.gauss_count() {
                    gauss.push(zone.get(cell, g, ci)?);
                }
                Ok(fit.nodal_values(&gauss))
            }
        }
    }
}

/// Per-zone least-squares operator mapping a cell's `n_g` Gauss values to
/// `npc` nodal values **for drawing that cell only** — never averaged with
/// neighbouring cells, so inter-element discontinuities stay visible.
///
/// Solves `min Σ_g w_g (Σ_i N_i(ξ_g)·v_i − f_g)²`: with `A[g,i] = N_i(ξ_g)`
/// and `W = diag(w_g)`, the normal equations are `(AᵀWA) v = AᵀW f`. The
/// `npc×npc` factorization is shared by every cell of the zone (`A` only
/// depends on the FE subspace).
pub(crate) struct FitOperator {
    /// `Some((lu(AᵀWA), AᵀW))`, or `None` when under-determined
    /// (`n_g < npc`, e.g. one Gauss point) / singular → every nodal value
    /// falls back to the mean of the Gauss values (constant per cell).
    op: Option<(na::LU<f64, na::Dyn, na::Dyn>, na::DMatrix<f64>)>,
    npc: usize,
}

impl FitOperator {
    fn for_zone(zone: &SubElementField) -> Result<Self> {
        let fespace = zone.support();
        let s = read(&fespace)?;
        let npc = s.nodes_per_cell()?;
        let n_g = s.gauss_count();
        if n_g < npc {
            return Ok(Self { op: None, npc });
        }
        let mut a = na::DMatrix::zeros(n_g, npc);
        let mut atw = na::DMatrix::zeros(npc, n_g);
        for g in 0..n_g {
            let row = s.n_at_g(g)?;
            let w = s.gauss_weight(g)?;
            for i in 0..npc {
                a[(g, i)] = row[i];
                atw[(i, g)] = row[i] * w;
            }
        }
        let k = &atw * &a;
        let lu = k.lu();
        if !lu.is_invertible() {
            return Ok(Self { op: None, npc });
        }
        Ok(Self {
            op: Some((lu, atw)),
            npc,
        })
    }

    /// Nodal values for one cell from its Gauss values.
    fn nodal_values(&self, gauss_values: &[f64]) -> Vec<f64> {
        let mean = if gauss_values.is_empty() {
            0.0
        } else {
            gauss_values.iter().sum::<f64>() / gauss_values.len() as f64
        };
        if let Some((lu, atw)) = &self.op {
            let f = na::DVector::from_column_slice(gauss_values);
            let b = atw * f;
            if let Some(v) = lu.solve(&b) {
                return v.iter().copied().collect();
            }
        }
        vec![mean; self.npc]
    }
}

/// Per-cell (flat) field value for a `SubMesh`: the mean of the cell's
/// nodal values. Length of the returned vector = number of cells.
pub(crate) fn submesh_cell_values(
    ctx: &SubmeshFieldCtx<'_>,
    sm: &SubMesh,
    component: &str,
) -> Result<Vec<f64>> {
    let conn = sm.connectivity();
    let npc = sm.element_type().nodes_per_cell();
    if npc == 0 || conn.is_empty() {
        return Ok(Vec::new());
    }
    let n_cells = conn.len() / npc;
    let mut out = Vec::with_capacity(n_cells);
    for cell in 0..n_cells {
        let nodal = ctx.cell_node_values(sm, cell, component)?;
        out.push(nodal.iter().sum::<f64>() / nodal.len() as f64);
    }
    Ok(out)
}

/// Per-cell nodal values of `component` for every cell of `sm`
/// (smooth-rendering prepass; also yields the colour range).
pub(crate) fn submesh_nodal_values(
    ctx: &SubmeshFieldCtx<'_>,
    sm: &SubMesh,
    component: &str,
) -> Result<Vec<Vec<f64>>> {
    let npc = sm.element_type().nodes_per_cell();
    let conn = sm.connectivity();
    if npc == 0 || conn.is_empty() {
        return Ok(Vec::new());
    }
    (0..conn.len() / npc)
        .map(|cell| ctx.cell_node_values(sm, cell, component))
        .collect()
}

/// Interpolated (smooth) primitives of one submesh: each cell is split
/// into level-`n` sub-triangles whose geometry and value follow the
/// shape functions of the element — see [`crate::viz::subdivide`]. The
/// element boundary is drawn as a wire on top; the sub-faces carry no
/// outline. Values are the **per-element** nodal values of `nodal`
/// (one `Vec` per cell), so inter-element discontinuities remain.
pub(crate) fn submesh_primitives_smooth(
    sm: &SubMesh,
    nodal: &[Vec<f64>],
    cmap: Colormap,
    vmin: f64,
    vmax: f64,
    n: usize,
) -> Result<Vec<Primitive>> {
    use crate::atoms::Point3;
    use crate::containers::finite_element_space::Interpolation;
    use crate::viz::mesh_draw::pad3;
    use crate::viz::subdivide::{subdivide, CellSubdivision};

    let et = sm.element_type();
    let npc = et.nodes_per_cell();
    let conn = sm.connectivity();
    let n_cells = conn.len() / npc.max(1);
    if n_cells == 0 {
        return Ok(Vec::new());
    }
    let sub = match et {
        crate::atoms::ElementType::POI1 => CellSubdivision::Points,
        _ => subdivide(et, Interpolation::Lagrange1, n)?,
    };

    // Volume cells: keep only boundary faces. The `Faces` of the
    // subdivision are in the same order as the element's face table
    // (TET4_FACES / HEX8_FACES), so the keep-set indexes them directly.
    let keep = crate::viz::mesh_draw::boundary_faces(et, conn);

    // All node coordinates of the submesh, padded to 3-D.
    let coords = sm.coords();
    let coords: Vec<Point3> = {
        let c = read(&coords)?;
        conn.iter()
            .map(|&nid| c.coord(nid).map(pad3))
            .collect::<Result<_>>()?
    };

    let mut out = Vec::new();
    for cell in 0..n_cells {
        let xs = &coords[cell * npc..(cell + 1) * npc];
        let vs = &nodal[cell];
        let at = |w: &[f64]| -> (Point3, f64) {
            let mut p = Point3::new(0.0, 0.0, 0.0);
            let mut v = 0.0;
            for i in 0..npc {
                p += xs[i].coords * w[i];
                v += vs[i] * w[i];
            }
            (p, v)
        };
        match &sub {
            CellSubdivision::Points => {
                out.push(Primitive::Point {
                    p: xs[0],
                    color: colormap(cmap, vs[0], vmin, vmax),
                });
            }
            CellSubdivision::Segments { weights, segments } => {
                let pv: Vec<(Point3, f64)> = weights.iter().map(|w| at(w)).collect();
                for seg in segments {
                    let (a, va) = pv[seg[0]];
                    let (b, vb) = pv[seg[1]];
                    out.push(Primitive::Segment {
                        a,
                        b,
                        color: colormap(cmap, 0.5 * (va + vb), vmin, vmax),
                    });
                }
            }
            CellSubdivision::Faces(faces) => {
                for (fi, face) in faces.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(cell, fi))) {
                        continue;
                    }
                    let pv: Vec<(Point3, f64)> = face.weights.iter().map(|w| at(w)).collect();
                    for tri in &face.triangles {
                        let value = (pv[tri[0]].1 + pv[tri[1]].1 + pv[tri[2]].1) / 3.0;
                        out.push(Primitive::Face {
                            verts: vec![pv[tri[0]].0, pv[tri[1]].0, pv[tri[2]].0],
                            color: colormap(cmap, value, vmin, vmax),
                            outline: false,
                        });
                    }
                    out.push(Primitive::Wire {
                        verts: face.outline.iter().map(|w| at(w).0).collect(),
                    });
                }
            }
        }
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
    values
        .iter()
        .map(|&v| colormap(cmap, v, vmin, vmax))
        .collect()
}

/// Compute the per-cell values of every submesh of a mesh, and the
/// global `(min, max)` range over all submeshes.
fn mesh_cell_values(
    mesh: &Mesh,
    field: &FieldData,
    component: &str,
) -> Result<(Vec<Vec<f64>>, f64, f64)> {
    let n_sub = mesh.len();
    let mut per_sub: Vec<Vec<f64>> = Vec::with_capacity(n_sub);
    for i in 0..n_sub {
        let sm = mesh.get(i)?;
        let ctx = field.for_submesh(&sm)?;
        let vals = submesh_cell_values(&ctx, &*read(&sm)?, component)?;
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
    field: &'a FieldData,
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
        None => field
            .components()
            .first()
            .map(|s| s.as_str())
            .ok_or_else(|| PyrucastError::Message("field has no components".into())),
    }
}

// ─── Drawable wrappers ─────────────────────────────────────────────────────

/// `Drawable` over a [`Mesh`] coloured by a field component
/// (node field or element field, see [`FieldData`]).
pub(crate) struct MeshFieldView<'a> {
    pub(crate) mesh: &'a Mesh,
    pub(crate) field: &'a FieldData,
    pub component: &'a str,
    /// Caller override for the colour-scale bounds; defaults to the
    /// data's own range.
    pub scale: crate::viz::ColorScale,
    /// Subdivision level of the interpolated rendering; `0` = one flat
    /// colour per cell.
    pub smooth: usize,
}

impl<'a> Drawable for MeshFieldView<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        self.mesh.bbox()
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let cmap = self.scale.cmap;
        let mut all_prims: Vec<Primitive> = Vec::new();
        let (vmin, vmax);
        if self.smooth == 0 {
            let (per_sub, dmin, dmax) = mesh_cell_values(self.mesh, self.field, self.component)?;
            (vmin, vmax) = self.scale.resolve(dmin, dmax);
            for (i, values) in per_sub.iter().enumerate() {
                let sm = self.mesh.get(i)?;
                let colors = colors_from_values(values, cmap, vmin, vmax);
                let prims = submesh_primitives_with_colors(&*read(&sm)?, &colors)?;
                all_prims.extend(prims);
            }
        } else {
            // Prepass: per-element nodal values of every submesh + range.
            let mut per_sub: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.mesh.len());
            for i in 0..self.mesh.len() {
                let sm = self.mesh.get(i)?;
                let ctx = self.field.for_submesh(&sm)?;
                per_sub.push(submesh_nodal_values(&ctx, &*read(&sm)?, self.component)?);
            }
            let flat: Vec<f64> = per_sub
                .iter()
                .flat_map(|cells| cells.iter().flatten().copied())
                .collect();
            let (dmin, dmax) = value_range([flat.as_slice()]);
            (vmin, vmax) = self.scale.resolve(dmin, dmax);
            for (i, nodal) in per_sub.iter().enumerate() {
                let sm = self.mesh.get(i)?;
                all_prims.extend(submesh_primitives_smooth(
                    &*read(&sm)?,
                    nodal,
                    cmap,
                    vmin,
                    vmax,
                    self.smooth,
                )?);
            }
        }
        render_primitives(area, view, &all_prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        super::overlay::draw_colorbar(area, cmap, vmin, vmax)?;
        Ok(())
    }

    fn is_axisymmetric(&self) -> bool {
        self.mesh.is_axisymmetric()
    }
}

/// `Drawable` over a single [`SubMesh`] (by handle, so element-field
/// zones can be matched by identity) coloured by a field component.
pub(crate) struct SubMeshFieldView<'a> {
    pub(crate) submesh: &'a Handle<SubMesh>,
    pub(crate) field: &'a FieldData,
    pub component: &'a str,
    /// Caller override for the colour-scale bounds; defaults to the
    /// data's own range.
    pub scale: crate::viz::ColorScale,
    /// Subdivision level of the interpolated rendering; `0` = one flat
    /// colour per cell.
    pub smooth: usize,
}

impl<'a> Drawable for SubMeshFieldView<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        read(self.submesh)?.bbox()
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let ctx = self.field.for_submesh(self.submesh)?;
        let sm = read(self.submesh)?;
        let cmap = self.scale.cmap;
        let (prims, vmin, vmax);
        if self.smooth == 0 {
            let values = submesh_cell_values(&ctx, &sm, self.component)?;
            let (dmin, dmax) = value_range([values.as_slice()]);
            (vmin, vmax) = self.scale.resolve(dmin, dmax);
            let colors = colors_from_values(&values, cmap, vmin, vmax);
            prims = submesh_primitives_with_colors(&sm, &colors)?;
        } else {
            let nodal = submesh_nodal_values(&ctx, &sm, self.component)?;
            let flat: Vec<f64> = nodal.iter().flatten().copied().collect();
            let (dmin, dmax) = value_range([flat.as_slice()]);
            (vmin, vmax) = self.scale.resolve(dmin, dmax);
            prims = submesh_primitives_smooth(&sm, &nodal, cmap, vmin, vmax, self.smooth)?;
        }
        render_primitives(area, view, &prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        super::overlay::draw_colorbar(area, cmap, vmin, vmax)?;
        Ok(())
    }

    fn is_axisymmetric(&self) -> bool {
        read(self.submesh)
            .map(|sm| sm.is_axisymmetric())
            .unwrap_or(false)
    }
}

/// `Drawable` over the support nodes of a [`NodeField`], as a coloured
/// point cloud — the honest standalone rendering of a node field (its
/// POI1 support carries no connectivity; for surfaces, plot a mesh with
/// `field=`).
pub(crate) struct NodeFieldPointsView<'a> {
    /// Distinct support nodes with their (padded 3-D) position and value.
    pub(crate) points: Vec<(crate::atoms::Point3, f64)>,
    pub component: &'a str,
    pub scale: crate::viz::ColorScale,
    /// Whether the support coordinates are the meridian plane of a body of
    /// revolution — the cloud can then be swept like any other plot.
    pub axisymmetric: bool,
}

impl<'a> Drawable for NodeFieldPointsView<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        let mut bb = Bbox3::empty();
        for (p, _) in &self.points {
            bb.extend(*p);
        }
        Ok(bb)
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let values: Vec<f64> = self.points.iter().map(|(_, v)| *v).collect();
        let (dmin, dmax) = value_range([values.as_slice()]);
        let (vmin, vmax) = self.scale.resolve(dmin, dmax);
        let cmap = self.scale.cmap;
        let prims: Vec<Primitive> = self
            .points
            .iter()
            .map(|&(p, v)| Primitive::Point {
                p,
                color: colormap(cmap, v, vmin, vmax),
            })
            .collect();
        render_primitives(area, view, &prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        super::overlay::draw_colorbar(area, cmap, vmin, vmax)?;
        Ok(())
    }

    fn is_axisymmetric(&self) -> bool {
        self.axisymmetric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::field::Field;
    use crate::containers::mesh::SubMesh as RawSubMesh;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::coords::Coords;
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
        assert_eq!(
            colormap(Colormap::Jet, 7.0, 7.0, 7.0),
            RgbColor::new(0, 255, 0)
        );
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = RawSubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let sm_h = insert(sm);

        // Build a POI1 SubNodeField with T values [1, 2, 3] on a, b, c.
        let mut poi1 = RawSubMesh::new(coords.clone(), ElementType::POI1);
        for n in [&a, &b, &c] {
            poi1.add_cell(&[n.id()]).unwrap();
        }
        let poi1_h = insert(poi1);
        let mut nf = SubNodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 1.0).unwrap();
        nf.set_value(b.id(), "T", 2.0).unwrap();
        nf.set_value(c.id(), "T", 3.0).unwrap();
        let view = NodeField::from_sub(nf).view().unwrap();

        let data = FieldData::Node(view);
        let ctx = data.for_submesh(&sm_h).unwrap();
        let values = submesh_cell_values(&ctx, &read(&sm_h).unwrap(), "T").unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 2.0).abs() < 1e-12); // (1 + 2 + 3) / 3
    }

    #[test]
    fn cell_values_ignore_nodes_outside_field_support() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        // Field defined only on `a`.
        let mut poi1 = RawSubMesh::new(coords.clone(), ElementType::POI1);
        poi1.add_cell(&[a.id()]).unwrap();
        let poi1_h = insert(poi1);
        let mut nf = SubNodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 4.0).unwrap();
        let view = NodeField::from_sub(nf).view().unwrap();
        // SEG2 cell [a, b]: only `a` is in the field; missing node takes
        // the mean of the present ones → flat value 4.0.
        let mut seg = RawSubMesh::new(coords.clone(), ElementType::SEG2);
        seg.add_cell(&[a.id(), b.id()]).unwrap();
        let seg_h = insert(seg);
        let data = FieldData::Node(view);
        let ctx = data.for_submesh(&seg_h).unwrap();
        let values = submesh_cell_values(&ctx, &read(&seg_h).unwrap(), "T").unwrap();
        assert!((values[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn value_range_handles_empty_and_constant() {
        assert_eq!(value_range(std::iter::empty()), (0.0, 1.0));
        let v = vec![3.0, 3.0, 3.0];
        let (mn, mx) = value_range([v.as_slice()]);
        assert_eq!((mn, mx), (3.0, 3.0));
    }

    // ─── ElementField (Gauss values → per-element nodal values) ─────────

    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;

    /// Two TRI3 sharing the edge (b, c).
    fn two_tri_mesh_and_fespace() -> (Mesh, FiniteElementSpace) {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut sm = RawSubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.add_cell(&[b.id(), d.id(), c.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        (mesh, fes)
    }

    /// TRI3 (3 Gauss points = 3 nodes): Gauss values sampled from a
    /// nodal interpolant must be fitted back to those nodal values
    /// exactly.
    #[test]
    fn element_fit_reproduces_interpolated_nodal_values() {
        let (mesh, fes) = two_tri_mesh_and_fespace();
        let ef = ElementField::new(&fes, vec!["q".into()]).unwrap();
        let target = [1.0, 2.0, 4.0]; // nodal values of cell 0
        {
            let fespace = fes.get(0).unwrap();
            let s = read(&fespace).unwrap();
            let mut zone = crate::store::write(&ef.get(0).unwrap()).unwrap();
            for g in 0..s.gauss_count() {
                let n = s.n_at_g(g).unwrap();
                let f: f64 = (0..3).map(|i| n[i] * target[i]).sum();
                zone.set(0, g, 0, f).unwrap();
            }
        }
        let data = FieldData::Element(ef.view().unwrap());
        let sm = mesh.get(0).unwrap();
        let ctx = data.for_submesh(&sm).unwrap();
        let nodal = ctx.cell_node_values(&read(&sm).unwrap(), 0, "q").unwrap();
        for (v, t) in nodal.iter().zip(target) {
            assert!((v - t).abs() < 1e-10, "fit {nodal:?} ≠ target {target:?}");
        }
    }

    /// Under-determined fit (fewer Gauss points than nodes) falls back
    /// to the Gauss mean — constant per element.
    #[test]
    fn element_fit_underdetermined_falls_back_to_gauss_mean() {
        let fit = FitOperator { op: None, npc: 3 };
        let nodal = fit.nodal_values(&[2.0, 4.0]);
        assert_eq!(nodal, vec![3.0, 3.0, 3.0]);
    }

    /// Two neighbouring elements with different Gauss values keep their
    /// own nodal values on the shared edge — the discontinuity is
    /// preserved (no cross-element averaging).
    #[test]
    fn element_values_stay_discontinuous_across_elements() {
        let (mesh, fes) = two_tri_mesh_and_fespace();
        let ef = ElementField::new(&fes, vec!["q".into()]).unwrap();
        {
            let mut zone = crate::store::write(&ef.get(0).unwrap()).unwrap();
            zone.set_cell_uniform(0, "q", 1.0).unwrap();
            zone.set_cell_uniform(1, "q", 5.0).unwrap();
        }
        let data = FieldData::Element(ef.view().unwrap());
        let sm = mesh.get(0).unwrap();
        let ctx = data.for_submesh(&sm).unwrap();
        let guard = read(&sm).unwrap();
        let n0 = ctx.cell_node_values(&guard, 0, "q").unwrap();
        let n1 = ctx.cell_node_values(&guard, 1, "q").unwrap();
        // Constant Gauss values fit to the same constant nodal values;
        // the shared edge nodes (b, c) carry 1.0 on one side and 5.0 on
        // the other — per-element values, never averaged.
        for v in &n0 {
            assert!((v - 1.0).abs() < 1e-10);
        }
        for v in &n1 {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    /// A submesh not covered by any zone of the ElementField errors
    /// explicitly.
    #[test]
    fn element_field_missing_zone_errors() {
        let (mesh, fes) = two_tri_mesh_and_fespace();
        let ef = ElementField::new(&fes, vec!["q".into()]).unwrap();
        // A second, unrelated submesh.
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut poi1 = RawSubMesh::new(coords, ElementType::POI1);
        poi1.add_cell(&[a.id()]).unwrap();
        let other = insert(poi1);
        let data = FieldData::Element(ef.view().unwrap());
        assert!(data.for_submesh(&mesh.get(0).unwrap()).is_ok());
        assert!(data.for_submesh(&other).is_err());
    }
}
