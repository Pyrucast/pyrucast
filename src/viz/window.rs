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
//! The whole loop is kept small (a single `App` struct + one event match);
//! adapting it for additional inputs is straightforward.

use std::num::NonZeroU32;
use std::rc::Rc;

use plotters::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::error::{PyrucastError, Result};
use crate::viz::camera::Bbox3;
use crate::viz::drawable::Drawable;
use crate::viz::View;

const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;

struct App<'a, D: Drawable> {
    object: &'a D,
    /// Cached so we don't recompute the bbox each frame.
    target: crate::triangulation::Point3,
    yaw: f64,
    pitch: f64,
    scale: f64,

    width: u32,
    height: u32,
    /// RGB buffer fed to plotters (length = `width * height * 3`).
    pixel_buf: Vec<u8>,

    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,

    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

impl<'a, D: Drawable> App<'a, D> {
    fn new(object: &'a D, view: View, bbox: Bbox3) -> Self {
        let target = view.target.unwrap_or_else(|| bbox.center());
        let w = INIT_WIDTH;
        let h = INIT_HEIGHT;
        Self {
            object,
            target,
            yaw: view.yaw,
            pitch: view.pitch,
            scale: view.scale,
            width: w,
            height: h,
            pixel_buf: vec![255; (w * h * 3) as usize],
            window: None,
            surface: None,
            dragging: false,
            last_mouse: None,
        }
    }

    fn current_view(&self) -> View {
        View {
            yaw: self.yaw,
            pitch: self.pitch,
            scale: self.scale,
            target: Some(self.target),
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
                };
                let _ = self.object.draw_on(&area, &view);
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size.width, size.height),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
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
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }
}

/// Run the interactive viewer on `object`. Returns when the user closes
/// the window. Cancels with an error if winit fails to start.
pub(crate) fn run_interactive<D: Drawable>(object: &D, view: View) -> Result<()> {
    let bbox = object.bbox()?;
    let event_loop = EventLoop::new()
        .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
    let mut app = App::new(object, view, bbox);
    event_loop
        .run_app(&mut app)
        .map_err(|e| PyrucastError::Message(format!("winit: {e}")))?;
    // Touch the field so the compiler keeps it (the loop returns; we just
    // want to make sure the View round-trip is reachable).
    let _ = app.current_view();
    Ok(())
}
