//! Orientation gizmo: a small red/green/blue X/Y/Z triad drawn as an
//! overlay so the viewer knows where world axes point under the current
//! camera.
//!
//! Painted last, in a fixed-size sub-area pinned to the bottom-left
//! corner of the drawing area, independent of the visualized object's
//! own bounding box / scale.

use crate::containers::mesh::{Point3, Vector3};
use crate::error::Result;
use crate::viz::camera::Projector;
use crate::viz::drawable::pl_err;
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

const GIZMO_SIZE_PX: u32 = 90;
const GIZMO_MARGIN_PX: i32 = 12;
const X_COLOR: RGBColor = RGBColor(220, 60, 60);
const Y_COLOR: RGBColor = RGBColor(60, 170, 60);
const Z_COLOR: RGBColor = RGBColor(60, 90, 220);

/// Draw the orientation gizmo on top of `area`. The gizmo only depends
/// on `view`'s yaw and pitch — `scale` and `target` are ignored so it
/// stays the same visual size regardless of the data being shown.
pub(crate) fn draw_gizmo<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    view: &View,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (_, h) = area.dim_in_pixel();
    let sub = area.clone().shrink(
        (
            GIZMO_MARGIN_PX,
            h as i32 - GIZMO_MARGIN_PX - GIZMO_SIZE_PX as i32,
        ),
        (GIZMO_SIZE_PX, GIZMO_SIZE_PX),
    );

    // Fixed [-1.3, 1.3] range so unit-length axes fit with room for the
    // letter label at the tip.
    let span = 1.3f64;
    let mut chart = ChartBuilder::on(&sub)
        .margin(0)
        .build_cartesian_2d(-span..span, -span..span)
        .map_err(pl_err)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_axes()
        .draw()
        .map_err(pl_err)?;

    // Project the three world axes through the same camera the main
    // scene uses. `target` is forced to origin so the projection is
    // direction-only.
    let view_for_gizmo = View {
        target: Some(Point3::origin()),
        ..*view
    };
    let proj = Projector::new(&view_for_gizmo, Point3::origin());
    let project = |dir: Vector3| -> (f64, f64) {
        let p = proj.project(Point3::from(dir));
        (p.x, p.y)
    };

    let axes: [((f64, f64), RGBColor, &str); 3] = [
        (project(Vector3::x()), X_COLOR, "X"),
        (project(Vector3::y()), Y_COLOR, "Y"),
        (project(Vector3::z()), Z_COLOR, "Z"),
    ];

    let origin = (0.0_f64, 0.0_f64);
    for (tip, color, label) in &axes {
        let line_style = ShapeStyle::from(color).stroke_width(2);
        chart
            .draw_series(LineSeries::new(vec![origin, *tip], line_style))
            .map_err(pl_err)?;

        // Label placed slightly past the tip along the axis direction.
        let label_pos = (tip.0 * 1.18, tip.1 * 1.18);
        let text_style = TextStyle::from(("sans-serif", 14).into_font()).color(color);
        chart
            .draw_series(std::iter::once(Text::new(
                label.to_string(),
                label_pos,
                text_style,
            )))
            .map_err(pl_err)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::SVGBackend;

    #[test]
    fn gizmo_renders_three_colored_segments_in_svg() {
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (200, 200));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            draw_gizmo(&area, &View::iso()).unwrap();
            area.present().unwrap();
        }
        let lower = buf.to_ascii_lowercase();
        for hex in ["dc3c3c", "3caa3c", "3c5adc"] {
            assert!(lower.contains(hex), "missing axis colour {hex} in SVG");
        }
        for label in ["x", "y", "z"] {
            // plotters writes <text ...>\n  X  \n</text>; match laxly.
            let needle = format!(">\n{label}\n</text>");
            assert!(
                lower.contains(&needle),
                "missing label {label} in SVG; got {buf}"
            );
        }
    }
}
