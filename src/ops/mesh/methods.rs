//! Delegation methods — the "subject" face of this module's operators.
//!
//! A free function meeting the three conditions of `CONVENTIONS.md`
//! (§ "The verb also exposed as a method") is **also** exposed as a method of
//! its subject, to allow chaining. These methods contain **no** logic: they
//! call the free function, receiver included. The free function stays the
//! canonical form and carries the documentation.
//!
//! The `impl`s live here rather than in `containers/`: a container must not
//! depend on an operator, and Rust allows an inherent `impl` in any module of
//! the crate that defines the type.

use crate::atoms::Band;
use crate::atoms::ElementType;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;

impl Mesh {
    /// See [`mesh::barycenter`](fn@crate::ops::mesh::barycenter).
    pub fn barycenter(&self) -> Result<Mesh> {
        crate::ops::mesh::barycenter(self)
    }

    /// See [`mesh::border`](fn@crate::ops::mesh::border).
    pub fn border(&self, angle_deg: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::border(self, angle_deg)
    }

    /// See [`mesh::chain`](fn@crate::ops::mesh::chain).
    pub fn chain(&self) -> Result<Mesh> {
        crate::ops::mesh::chain(self)
    }

    /// See [`mesh::consolidate`](fn@crate::ops::mesh::consolidate).
    pub fn consolidate(&self) -> Result<Mesh> {
        crate::ops::mesh::consolidate(self)
    }

    /// See [`mesh::convert`](fn@crate::ops::mesh::convert).
    pub fn convert(&self, target: ElementType) -> Result<Mesh> {
        crate::ops::mesh::convert(self, target)
    }

    /// See [`mesh::copy`](fn@crate::ops::mesh::copy).
    pub fn copy(&self, new_nodes: bool) -> Result<Mesh> {
        crate::ops::mesh::copy(self, new_nodes)
    }

    /// See [`mesh::elements_on`](fn@crate::ops::mesh::elements_on).
    pub fn elements_on(&self, points: &Mesh, strict: bool) -> Result<Mesh> {
        crate::ops::mesh::elements_on(self, points, strict)
    }

    /// See [`mesh::extrude`](fn@crate::ops::mesh::extrude).
    pub fn extrude(&self, direction: &[f64], n_layers: usize) -> Result<Mesh> {
        crate::ops::mesh::extrude(self, direction, n_layers)
    }

    /// See [`mesh::invert`](fn@crate::ops::mesh::invert).
    pub fn invert(&self) -> Result<Mesh> {
        crate::ops::mesh::invert(self)
    }

    /// See [`mesh::merge_nodes`](fn@crate::ops::mesh::merge_nodes).
    pub fn merge_nodes(&self, tol: f64, in_place: bool) -> Result<Mesh> {
        crate::ops::mesh::merge_nodes(self, tol, in_place)
    }

    /// See [`mesh::orient`](fn@crate::ops::mesh::orient).
    pub fn orient(&self) -> Result<Mesh> {
        crate::ops::mesh::orient(self)
    }

    /// See [`mesh::pave_surface`](fn@crate::ops::mesh::pave_surface).
    pub fn pave_surface(
        &self,
        element_type: ElementType,
        target_size: Option<f64>,
        prefer_quads: bool,
        relax: crate::ops::mesh::FrontRelax,
    ) -> Result<Mesh> {
        crate::ops::mesh::pave_surface(self, element_type, target_size, prefer_quads, relax)
    }

    /// See [`mesh::pave_volume`](fn@crate::ops::mesh::pave_volume).
    pub fn pave_volume(
        &self,
        layers: usize,
        thickness: Option<f64>,
        core_size: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::pave_volume(self, layers, thickness, core_size)
    }

    /// See [`mesh::points_below_plane`](fn@crate::ops::mesh::points_below_plane).
    pub fn points_below_plane(
        &self,
        origin: &[f64],
        normal: &[f64],
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_below_plane(self, origin, normal, tol)
    }

    /// See [`mesh::points_in_cone`](fn@crate::ops::mesh::points_in_cone).
    pub fn points_in_cone(
        &self,
        base: &[f64],
        top: &[f64],
        base_radius: f64,
        top_radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_in_cone(self, base, top, base_radius, top_radius, tol)
    }

    /// See [`mesh::points_in_cylinder`](fn@crate::ops::mesh::points_in_cylinder).
    pub fn points_in_cylinder(
        &self,
        base: &[f64],
        top: &[f64],
        radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_in_cylinder(self, base, top, radius, tol)
    }

    /// See [`mesh::points_in_sphere`](fn@crate::ops::mesh::points_in_sphere).
    pub fn points_in_sphere(&self, center: &[f64], radius: f64, tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_in_sphere(self, center, radius, tol)
    }

    /// See [`mesh::points_in_torus`](fn@crate::ops::mesh::points_in_torus).
    pub fn points_in_torus(
        &self,
        center: &[f64],
        axis: &[f64],
        major_radius: f64,
        minor_radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_in_torus(self, center, axis, major_radius, minor_radius, tol)
    }

    /// See [`mesh::points_on_cone`](fn@crate::ops::mesh::points_on_cone).
    pub fn points_on_cone(
        &self,
        base: &[f64],
        top: &[f64],
        base_radius: f64,
        top_radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_cone(self, base, top, base_radius, top_radius, tol)
    }

    /// See [`mesh::points_on_cylinder`](fn@crate::ops::mesh::points_on_cylinder).
    pub fn points_on_cylinder(
        &self,
        base: &[f64],
        top: &[f64],
        radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_cylinder(self, base, top, radius, tol)
    }

    /// See [`mesh::points_on_line`](fn@crate::ops::mesh::points_on_line).
    pub fn points_on_line(&self, a: &[f64], b: &[f64], tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_on_line(self, a, b, tol)
    }

    /// See [`mesh::points_on_plane`](fn@crate::ops::mesh::points_on_plane).
    pub fn points_on_plane(
        &self,
        origin: &[f64],
        normal: &[f64],
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_plane(self, origin, normal, tol)
    }

    /// See [`mesh::points_on_sphere`](fn@crate::ops::mesh::points_on_sphere).
    pub fn points_on_sphere(&self, center: &[f64], radius: f64, tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_on_sphere(self, center, radius, tol)
    }

    /// See [`mesh::points_on_torus`](fn@crate::ops::mesh::points_on_torus).
    pub fn points_on_torus(
        &self,
        center: &[f64],
        axis: &[f64],
        major_radius: f64,
        minor_radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_torus(self, center, axis, major_radius, minor_radius, tol)
    }

    /// See [`mesh::revolve`](fn@crate::ops::mesh::revolve).
    pub fn revolve(
        &self,
        angle: f64,
        n_layers: usize,
        center: &[f64],
        axis: Option<&[f64]>,
    ) -> Result<Mesh> {
        crate::ops::mesh::revolve(self, angle, n_layers, center, axis)
    }

    /// See [`mesh::rotate`](fn@crate::ops::mesh::rotate).
    pub fn rotate(&self, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Result<Mesh> {
        crate::ops::mesh::rotate(self, angle, center, axis)
    }

    /// See [`mesh::skin`](fn@crate::ops::mesh::skin).
    pub fn skin(&self, angle_deg: f64) -> Result<Mesh> {
        crate::ops::mesh::skin(self, angle_deg)
    }

    /// See [`mesh::sweep`](fn@crate::ops::mesh::sweep).
    pub fn sweep(&self, mesh_b: &Mesh, n_layers: usize, element_type: ElementType) -> Result<Mesh> {
        crate::ops::mesh::sweep(self, mesh_b, n_layers, element_type)
    }

    /// See [`mesh::sweep_solid`](fn@crate::ops::mesh::sweep_solid).
    pub fn sweep_solid(&self, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
        crate::ops::mesh::sweep_solid(self, mesh_b, n_layers)
    }

    /// See [`mesh::symmetry_line`](fn@crate::ops::mesh::symmetry_line).
    pub fn symmetry_line(&self, a: &[f64], b: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_line(self, a, b)
    }

    /// See [`mesh::symmetry_plane`](fn@crate::ops::mesh::symmetry_plane).
    pub fn symmetry_plane(&self, a: &[f64], b: &[f64], c: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_plane(self, a, b, c)
    }

    /// See [`mesh::symmetry_point`](fn@crate::ops::mesh::symmetry_point).
    pub fn symmetry_point(&self, center: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_point(self, center)
    }

    /// See [`mesh::to_poi1`](fn@crate::ops::mesh::to_poi1).
    pub fn to_poi1(&self) -> Result<Mesh> {
        crate::ops::mesh::to_poi1(self)
    }

    /// See [`mesh::to_quadratic`](fn@crate::ops::mesh::to_quadratic).
    pub fn to_quadratic(&self) -> Result<Mesh> {
        crate::ops::mesh::to_quadratic(self)
    }

    /// See [`mesh::transfinite`](fn@crate::ops::mesh::transfinite).
    pub fn transfinite(
        &self,
        side2: &Mesh,
        side3: &Mesh,
        side4: &Mesh,
        element_type: ElementType,
    ) -> Result<Mesh> {
        crate::ops::mesh::transfinite(self, side2, side3, side4, element_type)
    }

    /// See [`mesh::translate`](fn@crate::ops::mesh::translate).
    pub fn translate(&self, vector: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::translate(self, vector)
    }

    /// See [`mesh::triangulate_surface`](fn@crate::ops::mesh::triangulate_surface).
    pub fn triangulate_surface(
        &self,
        element_type: ElementType,
        target_size: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::triangulate_surface(self, element_type, target_size)
    }

    /// See [`mesh::triangulate_volume`](fn@crate::ops::mesh::triangulate_volume).
    pub fn triangulate_volume(
        &self,
        target_size: Option<f64>,
        allow_surface_nodes: bool,
    ) -> Result<Mesh> {
        crate::ops::mesh::triangulate_volume(self, target_size, allow_surface_nodes)
    }
}

// `select` starts from a field but returns a `Mesh`: the free function
// therefore lives in `ops::mesh`, and the `impl` follows it. The name loses
// its qualifier — the subject's type already says whether it is nodes or cells.
impl NodeField {
    /// See [`mesh::select_nodes`](fn@crate::ops::mesh::select_nodes).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_nodes(self, band, components)
    }
}

impl ElementField {
    /// See [`mesh::select_cells`](fn@crate::ops::mesh::select_cells).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_cells(self, band, components)
    }
}

impl SubNodeField {
    /// See [`mesh::select_sub_nodes`](fn@crate::ops::mesh::select_sub_nodes).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_sub_nodes(self, band, components)
    }
}

impl SubElementField {
    /// See [`mesh::select_sub_cells`](fn@crate::ops::mesh::select_sub_cells).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_sub_cells(self, band, components)
    }
}
