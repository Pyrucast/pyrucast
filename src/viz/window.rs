//! Interactive window backend (feature `viz-interactive`).
//!
//! Uses `winit` for windowing and event loop, and `softbuffer` to present
//! a CPU-rendered framebuffer. The mesh is re-drawn with `plotters` on
//! each frame, identical to the offscreen PNG path — the only difference
//! is the output target.
//!
//! Mouse mapping:
//!
//! - left-button drag → updates `yaw` (horizontal) and `pitch` (vertical),
//! - right-button drag → pans (translates the camera target in the screen
//!   plane),
//! - mouse wheel → multiplies `scale` (zoom).
//!
//! Keyboard: `A` toggles the orientation gizmo, `Tab` cycles the field
//! component, ← / → step the evolution frames, and `R` — on an axisymmetric
//! plot only — sweeps the meridian section into its body of revolution (same
//! as clicking the top-left button).
//!
//! `winit::EventLoop::new()` may be called at most once per process on
//! most platforms. To stay usable from long-lived Python interpreters,
//! the event loop is cached in a thread-local and re-driven via
//! `run_app_on_demand`, so repeated `plot()` calls reuse the same loop.

use std::cell::{Cell, RefCell};
use std::num::NonZeroU32;
use std::rc::Rc;

use plotters::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::containers::field::Field as _;
use crate::error::{PyrucastError, Result};
use crate::viz::camera::Bbox3;
use crate::viz::drawable::Drawable;
use crate::viz::overlay;
use crate::viz::View;

const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;

/// Object-safe interface implemented by Drawables that expose a
/// field-component button in the interactive window. Allows the app to
/// detect button clicks and the `Tab` keyboard shortcut without
/// knowing the concrete Drawable type.
pub(crate) trait FieldButton {
    /// Cycle to the next component.
    fn cycle(&self);
}

/// Object-safe interface for a Drawable that exposes a **frame slider** (an
/// evolution of fields). The App drives it from the slider drag and the
/// ← / → keys without knowing the concrete Drawable type.
pub(crate) trait FrameControl {
    /// Number of tabulated frames.
    fn frame_count(&self) -> usize;
    /// Currently displayed frame index.
    fn current(&self) -> usize;
    /// Select frame `k` (clamped to the valid range).
    fn set_frame(&self, k: usize);
}

struct App<'a, D: Drawable> {
    object: &'a D,
    /// Optional field-cycle handler — `Some` when the Drawable also
    /// implements [`FieldButton`] (the field-aware mesh / submesh
    /// rendering path).
    field_button: Option<&'a dyn FieldButton>,
    /// Optional frame-slider handler — `Some` for an evolution of fields.
    frame_control: Option<&'a dyn FrameControl>,
    /// Whether a slider drag is in progress (suppresses camera rotation).
    sliding: bool,
    /// Cached so we don't recompute the bbox each frame.
    target: crate::atoms::Point3,
    /// Scene bounding box — kept so a pan drag can convert pixel deltas into a
    /// world-space shift of `target` at the current zoom.
    bbox: Bbox3,
    yaw: f64,
    pitch: f64,
    scale: f64,
    show_axes: bool,
    /// Current revolution of an axisymmetric plot; `None` = flat section.
    revolve: Option<crate::viz::Revolve>,
    /// Sweep asked for by the caller, restored when the toggle comes back on.
    revolve_pref: crate::viz::Revolve,
    /// Whether the object may be swept at all (axisymmetric geometry). The
    /// toggle — button and `R` key — is inert otherwise.
    can_revolve: bool,
    /// Bounding box of the flat section, kept to switch `bbox` back and forth
    /// as the sweep is toggled.
    section_bbox: Bbox3,
    /// Custom OS window title (`None` → the default "pyrucast").
    title: Option<String>,

    width: u32,
    height: u32,
    /// RGB buffer fed to plotters (length = `width * height * 3`).
    pixel_buf: Vec<u8>,

    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,

    dragging: bool,
    /// Whether a right-button pan drag is in progress (translates `target`).
    panning: bool,
    last_mouse: Option<(f64, f64)>,
    /// Live cursor position — kept up to date by [`WindowEvent::CursorMoved`]
    /// so [`WindowEvent::MouseInput`] can test whether the click landed on
    /// the field-component button.
    cursor: Option<(f64, f64)>,
}

impl<'a, D: Drawable> App<'a, D> {
    fn new(object: &'a D, view: View, bbox: Bbox3, title: Option<&str>) -> Self {
        Self::new_with_button(object, view, bbox, None, title)
    }

    fn new_with_button(
        object: &'a D,
        view: View,
        bbox: Bbox3,
        field_button: Option<&'a dyn FieldButton>,
        title: Option<&str>,
    ) -> Self {
        let can_revolve = object.is_axisymmetric();
        let revolve = view.revolve.filter(|_| can_revolve);
        let section_bbox = bbox;
        let bbox = match revolve {
            Some(_) => crate::viz::revolve::revolved_bbox(&section_bbox),
            None => section_bbox,
        };
        let target = view.target.unwrap_or_else(|| bbox.center());
        let w = INIT_WIDTH;
        let h = INIT_HEIGHT;
        Self {
            object,
            field_button,
            frame_control: None,
            sliding: false,
            target,
            bbox,
            yaw: view.yaw,
            pitch: view.pitch,
            scale: view.scale,
            show_axes: view.show_axes,
            revolve,
            revolve_pref: revolve.unwrap_or_default(),
            can_revolve,
            section_bbox,
            title: title.map(str::to_string),
            width: w,
            height: h,
            pixel_buf: vec![255; (w * h * 3) as usize],
            window: None,
            surface: None,
            dragging: false,
            panning: false,
            last_mouse: None,
            cursor: None,
        }
    }

    fn current_view(&self) -> View {
        View {
            yaw: self.yaw,
            pitch: self.pitch,
            scale: self.scale,
            target: Some(self.target),
            show_axes: self.show_axes,
            revolve: self.revolve,
        }
    }

    /// Switch between the flat meridian section and its body of revolution.
    /// The swept body is centred on the axis, not on the section, so the
    /// camera re-targets — otherwise the object would jump out of frame.
    fn toggle_revolve(&mut self) {
        if !self.can_revolve {
            return;
        }
        self.revolve = match self.revolve {
            Some(rev) => {
                self.revolve_pref = rev;
                None
            }
            None => Some(self.revolve_pref),
        };
        self.bbox = match self.revolve {
            Some(_) => crate::viz::revolve::revolved_bbox(&self.section_bbox),
            None => self.section_bbox,
        };
        self.target = self.bbox.center();
    }

    /// Translate the camera target in the screen plane by a pixel drag
    /// `(dx, dy)`. Moving the cursor right/down drags the scene the same way
    /// (grab-and-pull), so the target shifts opposite to the cursor along the
    /// projector's `right` / `up` axes, scaled to world units at the current
    /// zoom. Screen Y grows downward, hence the `+dy` on `up`.
    fn pan(&mut self, dx: f64, dy: f64) {
        let view = self.current_view();
        let proj = crate::viz::camera::Projector::new(&view, self.target);
        let wpp = crate::viz::camera::world_per_pixel(&view, &self.bbox, self.width, self.height);
        let shift = proj.right * (-dx * wpp) + proj.up * (dy * wpp);
        self.target += shift;
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.width = w;
        self.height = h;
        self.pixel_buf = vec![255; (w * h * 3) as usize];
        if let Some(surface) = self.surface.as_mut() {
            if let (Some(nw), Some(nh)) = (NonZeroU32::new(w), NonZeroU32::new(h)) {
                let _ = surface.resize(nw, nh);
            }
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn draw(&mut self) {
        let w = self.width;
        let h = self.height;
        if w == 0 || h == 0 {
            return;
        }
        // Read the camera out before the buffer is borrowed mutably.
        let view = self.current_view();
        let (show_axes, can_revolve, revolve) = (self.show_axes, self.can_revolve, self.revolve);
        // Re-create a fresh frame in the RGB buffer.
        self.pixel_buf.fill(255);
        {
            let backend = BitMapBackend::with_buffer(&mut self.pixel_buf, (w, h));
            let area = backend.into_drawing_area();
            if area.fill(&WHITE).is_ok() {
                let _ = self.object.draw_on(&area, &view);
                if show_axes {
                    let _ = crate::viz::axes::draw_gizmo(&area, &view);
                }
                let _ = overlay::draw_view_readout(&area, &view);
                if can_revolve {
                    let _ = overlay::draw_revolve_button(&area, revolve);
                }
                let _ = area.present();
            }
        }

        let surface = match self.surface.as_mut() {
            Some(s) => s,
            None => return,
        };
        let mut buf = match surface.buffer_mut() {
            Ok(b) => b,
            Err(_) => return,
        };
        let n = (w * h) as usize;
        for i in 0..n {
            let r = self.pixel_buf[3 * i] as u32;
            let g = self.pixel_buf[3 * i + 1] as u32;
            let b = self.pixel_buf[3 * i + 2] as u32;
            buf[i] = (r << 16) | (g << 8) | b;
        }
        let _ = buf.present();
    }
}

impl<'a, D: Drawable> ApplicationHandler for App<'a, D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title(self.title.as_deref().unwrap_or("pyrucast"))
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(_) => {
                event_loop.exit();
                return;
            }
        };
        let context = match softbuffer::Context::new(window.clone()) {
            Ok(c) => c,
            Err(_) => {
                event_loop.exit();
                return;
            }
        };
        let surface = match softbuffer::Surface::new(&context, window.clone()) {
            Ok(s) => s,
            Err(_) => {
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        self.window = Some(window);
        self.surface = Some(surface);
        self.resize(size.width.max(1), size.height.max(1));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Drop the surface and window *before* asking the loop to
                // exit. On some compositors (X11/Wayland), keeping them
                // alive past exit() causes the loop to spin on pending
                // expose/redraw events and never return.
                self.surface = None;
                self.window = None;
                event_loop.exit();
            }
            WindowEvent::Resized(size) => self.resize(size.width, size.height),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    // If the press landed on the revolution toggle, switch
                    // between section and body instead of starting a drag.
                    if let (true, Some((cx, cy))) = (self.can_revolve, self.cursor) {
                        if overlay::click_hits_revolve_button(cx, cy) {
                            self.toggle_revolve();
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    }
                    // If the press landed on the field-component button,
                    // cycle the field instead of starting a drag.
                    if let (Some(btn), Some((cx, cy))) = (self.field_button, self.cursor) {
                        if overlay::click_hits_button(cx, cy, self.width) {
                            btn.cycle();
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    }
                    // If the press landed on the frame slider, grab it
                    // (and start a slider drag) instead of rotating.
                    if let (Some(fc), Some((cx, cy))) = (self.frame_control, self.cursor) {
                        if let Some(k) = overlay::slider_frame_at(
                            cx,
                            cy,
                            self.width,
                            self.height,
                            fc.frame_count(),
                        ) {
                            fc.set_frame(k);
                            self.sliding = true;
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    }
                    self.dragging = true;
                } else {
                    self.dragging = false;
                    self.sliding = false;
                    self.last_mouse = None;
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                // Right-drag pans (translates the camera target in the screen
                // plane), complementing left-drag rotate and wheel zoom.
                if state == ElementState::Pressed {
                    self.panning = true;
                } else {
                    self.panning = false;
                    self.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                self.cursor = Some((x, y));
                if self.sliding {
                    if let Some(fc) = self.frame_control {
                        if let Some(k) = overlay::slider_frame_at(
                            x,
                            y,
                            self.width,
                            self.height,
                            fc.frame_count(),
                        ) {
                            if k != fc.current() {
                                fc.set_frame(k);
                                if let Some(w) = &self.window {
                                    w.request_redraw();
                                }
                            }
                        }
                    }
                    self.last_mouse = Some((x, y));
                    return;
                }
                if self.panning {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = x - lx;
                        let dy = y - ly;
                        self.pan(dx, dy);
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                    self.last_mouse = Some((x, y));
                    return;
                }
                if self.dragging {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = x - lx;
                        let dy = y - ly;
                        self.yaw -= dx * 0.4;
                        self.pitch = (self.pitch + dy * 0.4).clamp(-89.9, 89.9);
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
                self.last_mouse = Some((x, y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let factor = match delta {
                    MouseScrollDelta::LineDelta(_, y) => 1.15f64.powf(y as f64),
                    MouseScrollDelta::PixelDelta(p) => 1.001f64.powf(p.y),
                };
                self.scale = (self.scale * factor).clamp(0.01, 1000.0);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key,
                        repeat: false,
                        ..
                    },
                ..
            } => match &logical_key {
                Key::Character(s) if s.eq_ignore_ascii_case("a") => {
                    self.show_axes = !self.show_axes;
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
                Key::Character(s) if s.eq_ignore_ascii_case("r") => {
                    self.toggle_revolve();
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
                Key::Named(NamedKey::Tab) => {
                    if let Some(btn) = self.field_button {
                        btn.cycle();
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
                Key::Named(NamedKey::ArrowRight) => {
                    if let Some(fc) = self.frame_control {
                        let cur = fc.current();
                        if cur + 1 < fc.frame_count() {
                            fc.set_frame(cur + 1);
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                        }
                    }
                }
                Key::Named(NamedKey::ArrowLeft) => {
                    if let Some(fc) = self.frame_control {
                        let cur = fc.current();
                        if cur > 0 {
                            fc.set_frame(cur - 1);
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }
}

// ─── Field-coloured rendering: interactive entry points ───────────────────

/// Interactive Drawable that paints a mesh (or submesh) coloured by the
/// **currently selected** component of a field (node or element field,
/// see [`crate::viz::field_color::FieldData`]). Implements both
/// [`Drawable`] (for the App's render loop) and [`FieldButton`] (so the
/// App can cycle through components on click / Tab).
struct FieldDrawable<'a> {
    source: GeomSource<'a>,
    field: &'a crate::viz::field_color::FieldData,
    components: Vec<String>,
    /// Caller override for the colorbar bounds.
    scale: crate::viz::ColorScale,
    /// Subdivision level of the interpolated rendering (0 = flat).
    smooth: usize,
    /// Index into `components`. `Cell` because the App only has `&self`
    /// access on draw, but mutates this from the event-handling path.
    selected: Cell<usize>,
}

enum GeomSource<'a> {
    Mesh(&'a crate::containers::mesh::Mesh),
    SubMesh(&'a crate::handle::Handle<crate::containers::mesh::SubMesh>),
}

impl<'a> FieldDrawable<'a> {
    fn new(
        source: GeomSource<'a>,
        field: &'a crate::viz::field_color::FieldData,
        initial_component: &str,
        scale: crate::viz::ColorScale,
        smooth: usize,
    ) -> Self {
        let components: Vec<String> = field.components().to_vec();
        let selected = components
            .iter()
            .position(|c| c == initial_component)
            .unwrap_or(0);
        Self {
            source,
            field,
            components,
            scale,
            smooth,
            selected: Cell::new(selected),
        }
    }

    fn current_component(&self) -> &str {
        &self.components[self.selected.get()]
    }
}

impl<'a> Drawable for FieldDrawable<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        match &self.source {
            GeomSource::Mesh(m) => m.bbox(),
            GeomSource::SubMesh(sm) => sm.read().bbox(),
        }
    }

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &plotters::drawing::DrawingArea<DB, plotters::coord::Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let component = self.current_component();
        match &self.source {
            GeomSource::Mesh(m) => crate::viz::field_color::MeshFieldView {
                mesh: m,
                field: self.field,
                component,
                scale: self.scale,
                smooth: self.smooth,
            }
            .draw_on(area, view),
            GeomSource::SubMesh(sm) => crate::viz::field_color::SubMeshFieldView {
                submesh: sm,
                field: self.field,
                component,
                scale: self.scale,
                smooth: self.smooth,
            }
            .draw_on(area, view),
        }
    }

    fn is_axisymmetric(&self) -> bool {
        match &self.source {
            GeomSource::Mesh(m) => m.is_axisymmetric(),
            GeomSource::SubMesh(sm) => sm.read().is_axisymmetric(),
        }
    }
}

impl<'a> FieldButton for FieldDrawable<'a> {
    fn cycle(&self) {
        let n = self.components.len();
        if n == 0 {
            return;
        }
        let next = (self.selected.get() + 1) % n;
        self.selected.set(next);
    }
}

/// Run the interactive viewer on a `Mesh` coloured by a node-field
/// component (with a button that cycles through components).
pub(crate) fn run_interactive_mesh_field(
    mesh: &crate::containers::mesh::Mesh,
    field: &crate::viz::field_color::FieldData,
    initial_component: &str,
    scale: crate::viz::ColorScale,
    smooth: usize,
    view: View,
    title: Option<&str>,
) -> Result<()> {
    let drawable = FieldDrawable::new(
        GeomSource::Mesh(mesh),
        field,
        initial_component,
        scale,
        smooth,
    );
    crate::viz::check_revolve(&drawable, &view)?;
    let bbox = drawable.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(EventLoop::new().map_err(|e| PyrucastError::Message(format!("winit: {e}")))?);
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new_with_button(
            &drawable,
            view,
            bbox,
            Some(&drawable as &dyn FieldButton),
            title,
        );
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        Ok(())
    })
}

/// Run the interactive viewer on a `SubMesh` coloured by a node-field
/// component (same UX as [`run_interactive_mesh_field`]).
pub(crate) fn run_interactive_submesh_field(
    submesh: &crate::handle::Handle<crate::containers::mesh::SubMesh>,
    field: &crate::viz::field_color::FieldData,
    initial_component: &str,
    scale: crate::viz::ColorScale,
    smooth: usize,
    view: View,
    title: Option<&str>,
) -> Result<()> {
    let drawable = FieldDrawable::new(
        GeomSource::SubMesh(submesh),
        field,
        initial_component,
        scale,
        smooth,
    );
    crate::viz::check_revolve(&drawable, &view)?;
    let bbox = drawable.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(EventLoop::new().map_err(|e| PyrucastError::Message(format!("winit: {e}")))?);
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new_with_button(
            &drawable,
            view,
            bbox,
            Some(&drawable as &dyn FieldButton),
            title,
        );
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        Ok(())
    })
}

thread_local! {
    /// `EventLoop::new()` can only be called once per process on most
    /// platforms; we lazily build one and reuse it across `plot()` calls
    /// via `run_app_on_demand`. `EventLoop` is `!Send`, so a `thread_local!`
    /// `RefCell` is the right container — Python always drives us from the
    /// main thread.
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

/// Run the interactive viewer on `object`. Returns when the user closes
/// the window. Cancels with an error if winit fails to start.
pub(crate) fn run_interactive<D: Drawable>(
    object: &D,
    view: View,
    title: Option<&str>,
) -> Result<()> {
    let bbox = object.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(EventLoop::new().map_err(|e| PyrucastError::Message(format!("winit: {e}")))?);
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new(object, view, bbox, title);
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        // Touch the field so the compiler keeps it (the loop returns; we
        // just want to make sure the View round-trip is reachable).
        let _ = app.current_view();
        Ok(())
    })
}

// ─── Evolution of fields: interactive frame slider ──────────────────────────

/// Interactive Drawable for an **evolution of fields**: it paints the
/// currently selected **tabulated frame** (coloured by the selected
/// component) and draws the frame slider. Implements [`Drawable`],
/// [`FieldButton`] (component cycling) and [`FrameControl`] (frame slider /
/// arrow keys).
struct EvolutionFrames<'a> {
    /// Surface geometry; `None` ⇒ node frames as a point cloud.
    mesh: Option<&'a crate::containers::mesh::Mesh>,
    frames: &'a [crate::viz::FrameField],
    abscissas: &'a [f64],
    /// Label for the slider's abscissa value (the abscissa type, or a default).
    abscissa_label: String,
    components: Vec<String>,
    scale: crate::viz::ColorScale,
    smooth: usize,
    selected_frame: Cell<usize>,
    selected_comp: Cell<usize>,
}

impl<'a> EvolutionFrames<'a> {
    fn new(
        mesh: Option<&'a crate::containers::mesh::Mesh>,
        frames: &'a [crate::viz::FrameField],
        abscissas: &'a [f64],
        abscissa_label: &str,
        initial_component: &str,
        scale: crate::viz::ColorScale,
        smooth: usize,
    ) -> Result<Self> {
        let components = match &frames[0] {
            crate::viz::FrameField::Node(f) => crate::viz::field_color::FieldData::Node(f.view()?)
                .components()
                .to_vec(),
            crate::viz::FrameField::Element(f) => {
                crate::viz::field_color::FieldData::Element(f.view()?)
                    .components()
                    .to_vec()
            }
        };
        let selected_comp = components
            .iter()
            .position(|c| c == initial_component)
            .unwrap_or(0);
        Ok(Self {
            mesh,
            frames,
            abscissas,
            abscissa_label: abscissa_label.to_string(),
            components,
            scale,
            smooth,
            selected_frame: Cell::new(0),
            selected_comp: Cell::new(selected_comp),
        })
    }

    fn current_component(&self) -> &str {
        &self.components[self.selected_comp.get()]
    }
}

impl<'a> Drawable for EvolutionFrames<'a> {
    fn bbox(&self) -> Result<Bbox3> {
        match self.mesh {
            Some(m) => m.bbox(),
            None => match &self.frames[0] {
                crate::viz::FrameField::Node(f) => crate::viz::node_field_bbox(f),
                crate::viz::FrameField::Element(_) => Ok(Bbox3::empty()),
            },
        }
    }

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &plotters::drawing::DrawingArea<DB, plotters::coord::Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let k = self.selected_frame.get();
        let comp = self.current_component();
        match (self.mesh, &self.frames[k]) {
            (Some(m), crate::viz::FrameField::Node(f)) => {
                let data = crate::viz::field_color::FieldData::Node(f.view()?);
                crate::viz::field_color::MeshFieldView {
                    mesh: m,
                    field: &data,
                    component: comp,
                    scale: self.scale,
                    smooth: self.smooth,
                }
                .draw_on(area, view)?;
            }
            (Some(m), crate::viz::FrameField::Element(f)) => {
                let data = crate::viz::field_color::FieldData::Element(f.view()?);
                crate::viz::field_color::MeshFieldView {
                    mesh: m,
                    field: &data,
                    component: comp,
                    scale: self.scale,
                    smooth: self.smooth,
                }
                .draw_on(area, view)?;
            }
            (None, crate::viz::FrameField::Node(f)) => {
                let points = crate::viz::node_field_points(f, comp)?;
                crate::viz::field_color::NodeFieldPointsView {
                    points,
                    component: comp,
                    scale: self.scale,
                    axisymmetric: crate::viz::node_field_is_axisymmetric(f),
                }
                .draw_on(area, view)?;
            }
            (None, crate::viz::FrameField::Element(_)) => {
                return Err(PyrucastError::Message(
                    "evolution plot: element-field frames require a mesh".into(),
                ));
            }
        }
        overlay::draw_slider(
            area,
            k,
            self.frames.len(),
            &self.abscissa_label,
            self.abscissas[k],
        )?;
        Ok(())
    }

    fn is_axisymmetric(&self) -> bool {
        match self.mesh {
            Some(m) => m.is_axisymmetric(),
            None => match &self.frames[0] {
                crate::viz::FrameField::Node(f) => crate::viz::node_field_is_axisymmetric(f),
                crate::viz::FrameField::Element(_) => false,
            },
        }
    }
}

impl<'a> FieldButton for EvolutionFrames<'a> {
    fn cycle(&self) {
        let n = self.components.len();
        if n == 0 {
            return;
        }
        self.selected_comp.set((self.selected_comp.get() + 1) % n);
    }
}

impl<'a> FrameControl for EvolutionFrames<'a> {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }
    fn current(&self) -> usize {
        self.selected_frame.get()
    }
    fn set_frame(&self, k: usize) {
        let n = self.frames.len();
        if n > 0 {
            self.selected_frame.set(k.min(n - 1));
        }
    }
}

/// Run the interactive viewer on an evolution of fields: a frame slider (drag
/// or ← / →) picks the tabulated value, the field button / Tab cycles the
/// component.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_interactive_evolution(
    mesh: Option<&crate::containers::mesh::Mesh>,
    frames: &[crate::viz::FrameField],
    abscissas: &[f64],
    abscissa_label: &str,
    initial_component: Option<&str>,
    scale: crate::viz::ColorScale,
    smooth: usize,
    view: View,
    title: Option<&str>,
) -> Result<()> {
    if frames.is_empty() {
        return Err(PyrucastError::Message(
            "evolution plot: no tabulated frame".into(),
        ));
    }
    if mesh.is_none() && matches!(frames[0], crate::viz::FrameField::Element(_)) {
        return Err(PyrucastError::Message(
            "evolution plot: element-field frames require a mesh".into(),
        ));
    }
    let drawable = EvolutionFrames::new(
        mesh,
        frames,
        abscissas,
        abscissa_label,
        initial_component.unwrap_or(""),
        scale,
        smooth,
    )?;
    crate::viz::check_revolve(&drawable, &view)?;
    let bbox = drawable.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(EventLoop::new().map_err(|e| PyrucastError::Message(format!("winit: {e}")))?);
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new_with_button(
            &drawable,
            view,
            bbox,
            Some(&drawable as &dyn FieldButton),
            title,
        );
        app.frame_control = Some(&drawable as &dyn FrameControl);
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        Ok(())
    })
}
