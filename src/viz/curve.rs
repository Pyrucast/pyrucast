//! X-Y curve rendering (scalar evolutions).
//!
//! A `CurvePlot` is a [`Drawable`] that draws one or more `(x, y)` line
//! series on a `plotters` Cartesian chart. It ignores the 3-D `View` (a curve
//! has no camera) — the dispatch in `super::render_curve` turns the gizmo
//! off so the 2-D chart fills the whole area.

use plotters::coord::Shift;
use plotters::prelude::*;

use crate::error::Result;
use crate::viz::camera::Bbox3;
use crate::viz::drawable::{pl_err, Drawable};
use crate::viz::View;

/// Distinct stroke colours cycled across series (one per zone).
const PALETTE: [RGBColor; 6] = [
    RGBColor(31, 119, 180),
    RGBColor(214, 39, 40),
    RGBColor(44, 160, 44),
    RGBColor(148, 103, 189),
    RGBColor(255, 127, 14),
    RGBColor(23, 190, 207),
];

/// A set of labelled `(x, y)` series drawn as connected lines with markers.
pub(crate) struct CurvePlot {
    pub series: Vec<(String, Vec<(f64, f64)>)>,
    pub x_label: String,
    pub y_label: String,
    pub title: String,
}

impl CurvePlot {
    /// `(xmin, xmax, ymin, ymax)` over every point, or `None` if no point.
    fn data_range(&self) -> Option<(f64, f64, f64, f64)> {
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for (_, pts) in &self.series {
            for &(x, y) in pts {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
        }
        if xmin.is_finite() {
            Some((xmin, xmax, ymin, ymax))
        } else {
            None
        }
    }
}

impl Drawable for CurvePlot {
    fn bbox(&self) -> Result<Bbox3> {
        // No 3-D extent: the curve ignores the camera.
        Ok(Bbox3::empty())
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, _view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let Some((xmin, mut xmax, mut ymin, mut ymax)) = self.data_range() else {
            return Ok(()); // nothing to draw
        };
        // Guard against degenerate ranges (single point / flat curve).
        if (xmax - xmin).abs() < f64::EPSILON {
            xmax = xmin + 1.0;
        }
        let pad = if (ymax - ymin).abs() < 1e-12 {
            1.0
        } else {
            (ymax - ymin) * 0.05
        };
        ymin -= pad;
        ymax += pad;

        let mut chart = ChartBuilder::on(area)
            .caption(&self.title, ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(42)
            .y_label_area_size(58)
            .build_cartesian_2d(xmin..xmax, ymin..ymax)
            .map_err(pl_err)?;

        chart
            .configure_mesh()
            .x_desc(&self.x_label)
            .y_desc(&self.y_label)
            .draw()
            .map_err(pl_err)?;

        let multi = self.series.len() > 1;
        for (i, (label, pts)) in self.series.iter().enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            chart
                .draw_series(LineSeries::new(pts.iter().copied(), color.stroke_width(2)))
                .map_err(pl_err)?
                .label(label.clone())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(2))
                });
            // Markers at the tabulated samples.
            chart
                .draw_series(pts.iter().map(|&p| Circle::new(p, 3, color.filled())))
                .map_err(pl_err)?;
        }

        if multi {
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.85))
                .border_style(BLACK)
                .draw()
                .map_err(pl_err)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_axis_labels_and_legend_in_svg() {
        let plot = CurvePlot {
            series: vec![
                ("zone 0".into(), vec![(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)]),
                ("zone 1".into(), vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]),
            ],
            x_label: "time".into(),
            y_label: "T".into(),
            title: "evolution".into(),
        };
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (640, 480));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            plot.draw_on(&area, &View::default()).unwrap();
            area.present().unwrap();
        }
        assert!(buf.contains("time"));
        assert!(buf.contains("evolution"));
        assert!(buf.contains("zone 0"));
    }

    #[test]
    fn empty_series_is_noop() {
        let plot = CurvePlot {
            series: vec![],
            x_label: "x".into(),
            y_label: "y".into(),
            title: String::new(),
        };
        let mut buf = String::new();
        let backend = SVGBackend::with_string(&mut buf, (320, 240));
        let area = backend.into_drawing_area();
        // No point → draw_on returns Ok without panicking.
        plot.draw_on(&area, &View::default()).unwrap();
    }
}
