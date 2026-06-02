//! Field-coloured rendering overlay — the "current component" button.
//!
//! Drawn on top of the rendered scene whenever a `NodeField` is supplied
//! (file export AND interactive window). In the interactive window the
//! same rectangle doubles as a **clickable button** that cycles through
//! the field's components — click detection uses [`button_rect`] so the
//! geometry stays in one place.

use crate::error::Result;
use crate::viz::drawable::pl_err;
use plotters::coord::Shift;
use plotters::prelude::*;

const BUTTON_MARGIN: i32 = 10;
const BUTTON_WIDTH: i32 = 260;
const BUTTON_HEIGHT: i32 = 26;
const BUTTON_FILL: RGBColor = RGBColor(245, 245, 245);
const BUTTON_BORDER: RGBColor = RGBColor(110, 110, 110);
const BUTTON_TEXT: RGBColor = RGBColor(20, 20, 20);

// ─── Colorbar geometry (right edge of the canvas) ───────────────────────────
const BAR_WIDTH: i32 = 16;
/// Space reserved to the right of the bar for tick marks + value labels.
const BAR_RIGHT_MARGIN: i32 = 58;
const BAR_TOP_MARGIN: i32 = 48;
const BAR_BOTTOM_MARGIN: i32 = 36;
const BAR_TICKS: usize = 5;

/// Pixel-space rectangle `(x, y, width, height)` of the button, in the
/// drawing area's local coordinate system (top-left origin). Centred at
/// the top of the area.
pub(crate) fn button_rect(area_width: u32) -> (i32, i32, i32, i32) {
    let x = (area_width as i32 - BUTTON_WIDTH) / 2;
    let y = BUTTON_MARGIN;
    (x.max(BUTTON_MARGIN), y, BUTTON_WIDTH, BUTTON_HEIGHT)
}

/// Is the pixel `(px, py)` inside the button on a canvas of size `(w, h)`?
pub(crate) fn click_hits_button(px: f64, py: f64, area_width: u32) -> bool {
    let (bx, by, bw, bh) = button_rect(area_width);
    let inside_x = px >= bx as f64 && px <= (bx + bw) as f64;
    let inside_y = py >= by as f64 && py <= (by + bh) as f64;
    inside_x && inside_y
}

/// Draw the labelled button (component name + value range) onto `area`.
pub(crate) fn draw_field_overlay<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    component: &str,
    vmin: f64,
    vmax: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (w, _) = area.dim_in_pixel();
    let (bx, by, bw, bh) = button_rect(w);

    // Filled background.
    let fill = ShapeStyle::from(&BUTTON_FILL).filled();
    area.draw(&Rectangle::new(
        [(bx, by), (bx + bw - 1, by + bh - 1)],
        fill,
    ))
    .map_err(pl_err)?;
    // Border (un-filled rectangle = stroke).
    let border = ShapeStyle::from(&BUTTON_BORDER);
    area.draw(&Rectangle::new(
        [(bx, by), (bx + bw - 1, by + bh - 1)],
        border,
    ))
    .map_err(pl_err)?;

    // Label centred vertically — plotters draws text with its top-left at
    // `pos`, so we shift down by ~6 px to feel centred for a 14-pt font.
    let label = format!("[{}]  min={:.3}  max={:.3}", component, vmin, vmax);
    let text_style = TextStyle::from(("sans-serif", 13).into_font()).color(&BUTTON_TEXT);
    area.draw_text(&label, &text_style, (bx + 10, by + 5))
        .map_err(pl_err)?;
    Ok(())
}

/// Draw a vertical colorbar on the right edge of `area`: the `cmap`
/// gradient (same one painting the cells) annotated with evenly spaced
/// value ticks. `vmin` maps to the bottom, `vmax` to the top.
///
/// No-op on a canvas too small to host the bar with its labels.
pub(crate) fn draw_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    cmap: crate::viz::Colormap,
    vmin: f64,
    vmax: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    let bar_x = w - BAR_RIGHT_MARGIN - BAR_WIDTH;
    let bar_top = BAR_TOP_MARGIN;
    let bar_bottom = h - BAR_BOTTOM_MARGIN;
    let bar_h = bar_bottom - bar_top;
    if bar_x <= 0 || bar_h <= 1 {
        return Ok(());
    }

    // Gradient: one filled row per pixel. `colormap` with the same
    // `cmap` / `(vmin, vmax)` as the cells keeps the bar and the mesh
    // consistent, including the degenerate (vmax ≤ vmin) → midpoint case.
    for py in 0..bar_h {
        let t = 1.0 - py as f64 / (bar_h - 1) as f64; // 1 at top, 0 at bottom
        let value = vmin + t * (vmax - vmin);
        let c = crate::viz::field_color::colormap(cmap, value, vmin, vmax);
        let style = ShapeStyle::from(&RGBColor(c.r, c.g, c.b)).filled();
        let y = bar_top + py;
        // 1-px-tall row: a zero-height rectangle (y..y) fills nothing,
        // so span y..y+1.
        area.draw(&Rectangle::new(
            [(bar_x, y), (bar_x + BAR_WIDTH - 1, y + 1)],
            style,
        ))
        .map_err(pl_err)?;
    }
    // Border around the bar.
    area.draw(&Rectangle::new(
        [(bar_x, bar_top), (bar_x + BAR_WIDTH - 1, bar_bottom - 1)],
        ShapeStyle::from(&BUTTON_BORDER),
    ))
    .map_err(pl_err)?;

    // Ticks + value labels, bottom (vmin) → top (vmax).
    let text_style = TextStyle::from(("sans-serif", 12).into_font()).color(&BUTTON_TEXT);
    let tick_style = ShapeStyle::from(&BUTTON_BORDER);
    for i in 0..BAR_TICKS {
        let frac = i as f64 / (BAR_TICKS - 1) as f64;
        let value = vmin + frac * (vmax - vmin);
        let y = bar_bottom - 1 - (frac * (bar_h - 1) as f64).round() as i32;
        area.draw(&PathElement::new(
            vec![(bar_x + BAR_WIDTH, y), (bar_x + BAR_WIDTH + 4, y)],
            tick_style,
        ))
        .map_err(pl_err)?;
        area.draw_text(
            &format!("{value:.3}"),
            &text_style,
            (bar_x + BAR_WIDTH + 8, y - 7),
        )
        .map_err(pl_err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_rect_centred() {
        let (x, _y, w, _h) = button_rect(800);
        // Centred horizontally on an 800-px canvas.
        assert_eq!(x, (800 - BUTTON_WIDTH) / 2);
        assert_eq!(w, BUTTON_WIDTH);
    }

    #[test]
    fn click_hits_button_inside_and_outside() {
        let (bx, by, bw, bh) = button_rect(800);
        let cx = (bx + bw / 2) as f64;
        let cy = (by + bh / 2) as f64;
        assert!(click_hits_button(cx, cy, 800));
        // Far away: not inside.
        assert!(!click_hits_button(0.0, 0.0, 800));
        assert!(!click_hits_button(799.0, 599.0, 800));
    }

    #[test]
    fn draw_overlay_renders_label_in_svg() {
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (400, 100));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            draw_field_overlay(&area, "T", 0.0, 1.5).unwrap();
            area.present().unwrap();
        }
        // The label text should appear in the SVG output.
        assert!(buf.contains("[T]"));
        assert!(buf.contains("min=0.000"));
        assert!(buf.contains("max=1.500"));
    }

    #[test]
    fn draw_colorbar_renders_value_ticks() {
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (400, 300));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            draw_colorbar(&area, crate::viz::Colormap::Viridis, 0.0, 2.0).unwrap();
            area.present().unwrap();
        }
        // 5 evenly spaced ticks over [0, 2]: endpoints and midpoint.
        assert!(buf.contains("0.000"));
        assert!(buf.contains("1.000"));
        assert!(buf.contains("2.000"));
    }

    #[test]
    fn draw_colorbar_skips_tiny_canvas() {
        // Too small to host the bar + labels — must be a no-op, not panic.
        let mut buf = String::new();
        {
            let backend = SVGBackend::with_string(&mut buf, (20, 20));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).unwrap();
            draw_colorbar(&area, crate::viz::Colormap::Viridis, 0.0, 1.0).unwrap();
            area.present().unwrap();
        }
    }
}
