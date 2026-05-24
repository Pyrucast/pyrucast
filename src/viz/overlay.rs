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
}
