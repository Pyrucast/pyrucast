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
//! - mouse wheel → multiplies `scale` (zoom).
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

struct App<'a, D: Drawable> {
    object: &'a D,
    /// Optional field-cycle handler — `Some` when the Drawable also
    /// implements [`FieldButton`] (the field-aware mesh / submesh
    /// rendering path).
    field_button: Option<&'a dyn FieldButton>,
    /// Cached so we don't recompute the bbox each frame.
    target: crate::mesh::point::Point3,
    yaw: f64,
    pitch: f64,
    scale: f64,
    show_axes: bool,

    width: u32,
    height: u32,
    /// RGB buffer fed to plotters (length = `width * height * 3`).
    pixel_buf: Vec<u8>,

    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,

    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    /// Live cursor position — kept up to date by [`WindowEvent::CursorMoved`]
    /// so [`WindowEvent::MouseInput`] can test whether the click landed on
    /// the field-component button.
    cursor: Option<(f64, f64)>,
}

impl<'a, D: Drawable> App<'a, D> {
    fn new(object: &'a D, view: View, bbox: Bbox3) -> Self {
        Self::new_with_button(object, view, bbox, None)
    }

    fn new_with_button(
        object: &'a D,
        view: View,
        bbox: Bbox3,
        field_button: Option<&'a dyn FieldButton>,
    ) -> Self {
        let target = view.target.unwrap_or_else(|| bbox.center());
        let w = INIT_WIDTH;
        let h = INIT_HEIGHT;
        Self {
            object,
            field_button,
            target,
            yaw: view.yaw,
            pitch: view.pitch,
            scale: view.scale,
            show_axes: view.show_axes,
            width: w,
            height: h,
            pixel_buf: vec![255; (w * h * 3) as usize],
            window: None,
            surface: None,
            dragging: false,
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
        }
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
        // Re-create a fresh frame in the RGB buffer.
        self.pixel_buf.fill(255);
        {
            let backend = BitMapBackend::with_buffer(&mut self.pixel_buf, (w, h));
            let area = backend.into_drawing_area();
            if area.fill(&WHITE).is_ok() {
                let view = View {
                    yaw: self.yaw,
                    pitch: self.pitch,
                    scale: self.scale,
                    target: Some(self.target),
                    show_axes: self.show_axes,
                };
                let _ = self.object.draw_on(&area, &view);
                if self.show_axes {
                    let _ = crate::viz::axes::draw_gizmo(&area, &view);
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
            .with_title("pyrucast")
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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
                    self.dragging = true;
                } else {
                    self.dragging = false;
                    self.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                self.cursor = Some((x, y));
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
                Key::Named(NamedKey::Tab) => {
                    if let Some(btn) = self.field_button {
                        btn.cycle();
                        if let Some(w) = &self.window {
                            w.request_redraw();
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
/// **currently selected** component of a NodeField. Implements both
/// [`Drawable`] (for the App's render loop) and [`FieldButton`] (so the
/// App can cycle through components on click / Tab).
struct FieldDrawable<'a> {
    source: FieldSource<'a>,
    field: &'a crate::containers::node_field::NodeField,
    components: Vec<String>,
    /// Index into `components`. `Cell` because the App only has `&self`
    /// access on draw, but mutates this from the event-handling path.
    selected: Cell<usize>,
}

enum FieldSource<'a> {
    Mesh(&'a crate::mesh::Mesh),
    SubMesh(&'a crate::mesh::SubMesh),
}

impl<'a> FieldDrawable<'a> {
    fn new(
        source: FieldSource<'a>,
        field: &'a crate::containers::node_field::NodeField,
        initial_component: &str,
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
            FieldSource::Mesh(m) => m.bbox(),
            FieldSource::SubMesh(sm) => sm.bbox(),
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
            FieldSource::Mesh(m) => crate::viz::field_color::MeshFieldView {
                mesh: m,
                field: self.field,
                component,
            }
            .draw_on(area, view),
            FieldSource::SubMesh(sm) => crate::viz::field_color::SubMeshFieldView {
                submesh: sm,
                field: self.field,
                component,
            }
            .draw_on(area, view),
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

/// Run the interactive viewer on a `Mesh` coloured by a `NodeField`
/// component (with a button that cycles through components).
pub(crate) fn run_interactive_mesh_field(
    mesh: &crate::mesh::Mesh,
    field: &crate::containers::node_field::NodeField,
    initial_component: &str,
    view: View,
) -> Result<()> {
    let drawable = FieldDrawable::new(FieldSource::Mesh(mesh), field, initial_component);
    let bbox = drawable.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                EventLoop::new()
                    .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?,
            );
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new_with_button(
            &drawable,
            view,
            bbox,
            Some(&drawable as &dyn FieldButton),
        );
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        Ok(())
    })
}

/// Run the interactive viewer on a `SubMesh` coloured by a `NodeField`
/// component (same UX as [`run_interactive_mesh_field`]).
pub(crate) fn run_interactive_submesh_field(
    submesh: &crate::mesh::SubMesh,
    field: &crate::containers::node_field::NodeField,
    initial_component: &str,
    view: View,
) -> Result<()> {
    let drawable = FieldDrawable::new(FieldSource::SubMesh(submesh), field, initial_component);
    let bbox = drawable.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                EventLoop::new()
                    .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?,
            );
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new_with_button(
            &drawable,
            view,
            bbox,
            Some(&drawable as &dyn FieldButton),
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
pub(crate) fn run_interactive<D: Drawable>(object: &D, view: View) -> Result<()> {
    let bbox = object.bbox()?;
    EVENT_LOOP.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                EventLoop::new()
                    .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?,
            );
        }
        let event_loop = slot.as_mut().expect("just initialised");
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app = App::new(object, view, bbox);
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
        // Touch the field so the compiler keeps it (the loop returns; we
        // just want to make sure the View round-trip is reachable).
        let _ = app.current_view();
        Ok(())
    })
}
