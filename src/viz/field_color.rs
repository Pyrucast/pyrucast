//! Field-aware colouring for mesh visualization.
//!
//! Given a [`crate::node_field::NodeField`] sampled at the nodes, paint
//! each mesh cell with the colour the field value maps to under a
//! standard "jet-lite" colormap (blue → green → red, no perceptual
//! ordering pretensions — good enough for early-stage debugging).
//!
//! Per-cell value = arithmetic mean of the field at the cell's nodes,
//! restricted to nodes that are in the field's support. Nodes absent
//! from the field do not contribute; a cell with no node in the
//! support gets value `0.0`.

use crate::color::RgbColor;
use crate::configuration::NodeId;
use crate::error::{PyrucastError, Result};
use crate::mesh::{Mesh, SubMesh};
use crate::node_field::NodeField;
use crate::store::with;
use crate::viz::camera::Bbox3;
use crate::viz::drawable::Drawable;
use crate::viz::mesh_draw::{render_primitives, submesh_primitives_with_colors, Primitive};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

/// "Jet-lite" colormap, `t ∈ [0, 1]`:
/// `t = 0` → pure blue, `t = 0.5` → pure green, `t = 1` → pure red.
///
/// `value` is mapped via `t = (value - vmin) / (vmax - vmin)` and
/// clamped to `[0, 1]`. When `vmax ≤ vmin` (degenerate range) the
/// function returns the middle of the gradient (green).
pub fn colormap(value: f64, vmin: f64, vmax: f64) -> RgbColor {
    let t = if vmax > vmin {
        ((value - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let (r, g, b) = if t < 0.5 {
        (0.0, 2.0 * t, 1.0 - 2.0 * t)
    } else {
        (2.0 * t - 1.0, 2.0 - 2.0 * t, 0.0)
    };
    let to_u8 = |x: f64| -> u8 {
        let v = (x * 255.0).round();
        if v < 0.0 {
            0
        } else if v > 255.0 {
            255
        } else {
            v as u8
        }
    };
    RgbColor::new(to_u8(r), to_u8(g), to_u8(b))
}

/// Mean of the field's `component` over the supplied node ids.
///
/// Nodes not in the field's support are simply ignored (they do not
/// contribute to the mean nor to the denominator). If **no** node is
/// in the support, returns `0.0`.
pub(crate) fn nodes_mean(
    field: &NodeField,
    node_ids: &[NodeId],
    component: &str,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &nid in node_ids {
        if let Ok(v) = field.value(nid, component) {
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
    field: &NodeField,
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
fn colors_from_values(values: &[f64], vmin: f64, vmax: f64) -> Vec<RgbColor> {
    values.iter().map(|&v| colormap(v, vmin, vmax)).collect()
}

/// Compute the per-cell values of every submesh of a mesh, and the
/// global `(min, max)` range over all submeshes.
fn mesh_cell_values(
    mesh: &Mesh,
    field: &NodeField,
    component: &str,
) -> Result<(Vec<Vec<f64>>, f64, f64)> {
    let n_sub = mesh.submesh_count();
    let mut per_sub: Vec<Vec<f64>> = Vec::with_capacity(n_sub);
    for i in 0..n_sub {
        let sm = mesh.submesh(i)?;
        let vals = with(&sm, |s| submesh_cell_values(s, field, component))??;
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
pub fn resolve_component<'a>(
    field: &'a NodeField,
    requested: Option<&'a str>,
) -> Result<&'a str> {
    match requested {
        Some(name) => {
            if field.component_index(name).is_none() {
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

/// `Drawable` over a [`Mesh`] coloured by a [`NodeField`]'s component.
pub struct MeshFieldView<'a> {
    pub mesh: &'a Mesh,
    pub field: &'a NodeField,
    pub component: &'a str,
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
        let (per_sub, vmin, vmax) =
            mesh_cell_values(self.mesh, self.field, self.component)?;
        let mut all_prims: Vec<Primitive> = Vec::new();
        for (i, values) in per_sub.iter().enumerate() {
            let sm = self.mesh.submesh(i)?;
            let colors = colors_from_values(values, vmin, vmax);
            let prims = with(&sm, |s| submesh_primitives_with_colors(s, &colors))??;
            all_prims.extend(prims);
        }
        render_primitives(area, view, &all_prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        Ok(())
    }
}

/// `Drawable` over a single [`SubMesh`] coloured by a [`NodeField`]'s
/// component.
pub struct SubMeshFieldView<'a> {
    pub submesh: &'a SubMesh,
    pub field: &'a NodeField,
    pub component: &'a str,
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
        let (vmin, vmax) = value_range([values.as_slice()]);
        let colors = colors_from_values(&values, vmin, vmax);
        let prims = submesh_primitives_with_colors(self.submesh, &colors)?;
        render_primitives(area, view, &prims)?;
        super::overlay::draw_field_overlay(area, self.component, vmin, vmax)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Configuration;
    use crate::element_type::ElementType;
    use crate::mesh::SubMesh as RawSubMesh;
    use crate::node::Node;
    use crate::store::insert;

    #[test]
    fn colormap_endpoints() {
        let c0 = colormap(0.0, 0.0, 1.0);
        assert_eq!(c0, RgbColor::new(0, 0, 255));
        let c1 = colormap(1.0, 0.0, 1.0);
        assert_eq!(c1, RgbColor::new(255, 0, 0));
        let c_mid = colormap(0.5, 0.0, 1.0);
        assert_eq!(c_mid, RgbColor::new(0, 255, 0));
    }

    #[test]
    fn colormap_clamps_out_of_range() {
        let lo = colormap(-99.0, 0.0, 1.0);
        let hi = colormap(99.0, 0.0, 1.0);
        assert_eq!(lo, RgbColor::new(0, 0, 255));
        assert_eq!(hi, RgbColor::new(255, 0, 0));
    }

    #[test]
    fn colormap_degenerate_range_returns_green() {
        assert_eq!(colormap(7.0, 7.0, 7.0), RgbColor::new(0, 255, 0));
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

        // Build a POI1 NodeField with T values [1, 2, 3] on a, b, c.
        let mut poi1 = RawSubMesh::new(cfg.clone(), ElementType::POI1);
        for n in [&a, &b, &c] {
            poi1.add_cell(&[n.id()]).unwrap();
        }
        let poi1_h = insert(poi1);
        let mut nf = NodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 1.0).unwrap();
        nf.set_value(b.id(), "T", 2.0).unwrap();
        nf.set_value(c.id(), "T", 3.0).unwrap();

        let values = crate::store::with(&sm_h, |s| {
            submesh_cell_values(s, &nf, "T")
        })
        .unwrap()
        .unwrap();
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
        let mut nf = NodeField::from_poi1(&poi1_h, vec!["T".into()]).unwrap();
        nf.set_value(a.id(), "T", 4.0).unwrap();
        // Two-node "cell": only `a` is in the field; mean = 4.0.
        let mean = nodes_mean(&nf, &[a.id(), b.id()], "T");
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
