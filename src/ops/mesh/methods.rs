//! Méthodes de délégation — la face « sujet » des opérateurs de ce module.
//!
//! Une fonction libre qui remplit les trois conditions de `CONVENTIONS.md`
//! (§ « Le verbe exposé aussi en méthode ») est **aussi** exposée comme
//! méthode de son sujet, pour permettre le chaînage. Ces méthodes ne
//! contiennent **aucune** logique : elles appellent la fonction libre,
//! receveur compris. La fonction libre reste la forme canonique et porte la
//! documentation.
//!
//! Les `impl` vivent ici plutôt que dans `containers/` : un conteneur ne doit
//! pas dépendre d'un opérateur, et Rust autorise un `impl` inhérent dans
//! n'importe quel module de la crate de définition.

use crate::atoms::Band;
use crate::atoms::ElementType;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;

impl Mesh {
    /// Voir [`mesh::barycenter`](fn@crate::ops::mesh::barycenter).
    pub fn barycenter(&self) -> Result<Mesh> {
        crate::ops::mesh::barycenter(self)
    }

    /// Voir [`mesh::border`](fn@crate::ops::mesh::border).
    pub fn border(&self, angle_deg: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::border(self, angle_deg)
    }

    /// Voir [`mesh::chain`](fn@crate::ops::mesh::chain).
    pub fn chain(&self) -> Result<Mesh> {
        crate::ops::mesh::chain(self)
    }

    /// Voir [`mesh::consolidate`](fn@crate::ops::mesh::consolidate).
    pub fn consolidate(&self) -> Result<Mesh> {
        crate::ops::mesh::consolidate(self)
    }

    /// Voir [`mesh::convert`](fn@crate::ops::mesh::convert).
    pub fn convert(&self, target: ElementType) -> Result<Mesh> {
        crate::ops::mesh::convert(self, target)
    }

    /// Voir [`mesh::elements_on`](fn@crate::ops::mesh::elements_on).
    pub fn elements_on(&self, points: &Mesh, strict: bool) -> Result<Mesh> {
        crate::ops::mesh::elements_on(self, points, strict)
    }

    /// Voir [`mesh::extrude`](fn@crate::ops::mesh::extrude).
    pub fn extrude(&self, direction: &[f64], n_layers: usize) -> Result<Mesh> {
        crate::ops::mesh::extrude(self, direction, n_layers)
    }

    /// Voir [`mesh::invert`](fn@crate::ops::mesh::invert).
    pub fn invert(&self) -> Result<Mesh> {
        crate::ops::mesh::invert(self)
    }

    /// Voir [`mesh::merge_nodes`](fn@crate::ops::mesh::merge_nodes).
    pub fn merge_nodes(&self, tol: f64, in_place: bool) -> Result<Mesh> {
        crate::ops::mesh::merge_nodes(self, tol, in_place)
    }

    /// Voir [`mesh::orient`](fn@crate::ops::mesh::orient).
    pub fn orient(&self) -> Result<Mesh> {
        crate::ops::mesh::orient(self)
    }

    /// Voir [`mesh::pave_surface`](fn@crate::ops::mesh::pave_surface).
    pub fn pave_surface(
        &self,
        element_type: ElementType,
        target_size: Option<f64>,
        all_quad: bool,
        relax: crate::ops::mesh::FrontRelax,
    ) -> Result<Mesh> {
        crate::ops::mesh::pave_surface(self, element_type, target_size, all_quad, relax)
    }

    /// Voir [`mesh::pave_volume`](fn@crate::ops::mesh::pave_volume).
    pub fn pave_volume(
        &self,
        layers: usize,
        thickness: Option<f64>,
        core_size: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::pave_volume(self, layers, thickness, core_size)
    }

    /// Voir [`mesh::points_below_plane`](fn@crate::ops::mesh::points_below_plane).
    pub fn points_below_plane(
        &self,
        origin: &[f64],
        normal: &[f64],
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_below_plane(self, origin, normal, tol)
    }

    /// Voir [`mesh::points_in_cone`](fn@crate::ops::mesh::points_in_cone).
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

    /// Voir [`mesh::points_in_cylinder`](fn@crate::ops::mesh::points_in_cylinder).
    pub fn points_in_cylinder(
        &self,
        base: &[f64],
        top: &[f64],
        radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_in_cylinder(self, base, top, radius, tol)
    }

    /// Voir [`mesh::points_in_sphere`](fn@crate::ops::mesh::points_in_sphere).
    pub fn points_in_sphere(&self, center: &[f64], radius: f64, tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_in_sphere(self, center, radius, tol)
    }

    /// Voir [`mesh::points_in_torus`](fn@crate::ops::mesh::points_in_torus).
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

    /// Voir [`mesh::points_on_cone`](fn@crate::ops::mesh::points_on_cone).
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

    /// Voir [`mesh::points_on_cylinder`](fn@crate::ops::mesh::points_on_cylinder).
    pub fn points_on_cylinder(
        &self,
        base: &[f64],
        top: &[f64],
        radius: f64,
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_cylinder(self, base, top, radius, tol)
    }

    /// Voir [`mesh::points_on_line`](fn@crate::ops::mesh::points_on_line).
    pub fn points_on_line(&self, a: &[f64], b: &[f64], tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_on_line(self, a, b, tol)
    }

    /// Voir [`mesh::points_on_plane`](fn@crate::ops::mesh::points_on_plane).
    pub fn points_on_plane(
        &self,
        origin: &[f64],
        normal: &[f64],
        tol: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::points_on_plane(self, origin, normal, tol)
    }

    /// Voir [`mesh::points_on_sphere`](fn@crate::ops::mesh::points_on_sphere).
    pub fn points_on_sphere(&self, center: &[f64], radius: f64, tol: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::points_on_sphere(self, center, radius, tol)
    }

    /// Voir [`mesh::points_on_torus`](fn@crate::ops::mesh::points_on_torus).
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

    /// Voir [`mesh::revolve`](fn@crate::ops::mesh::revolve).
    pub fn revolve(
        &self,
        angle: f64,
        n_layers: usize,
        center: &[f64],
        axis: Option<&[f64]>,
    ) -> Result<Mesh> {
        crate::ops::mesh::revolve(self, angle, n_layers, center, axis)
    }

    /// Voir [`mesh::rotate`](fn@crate::ops::mesh::rotate).
    pub fn rotate(&self, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Result<Mesh> {
        crate::ops::mesh::rotate(self, angle, center, axis)
    }

    /// Voir [`mesh::skin`](fn@crate::ops::mesh::skin).
    pub fn skin(&self, angle_deg: Option<f64>) -> Result<Mesh> {
        crate::ops::mesh::skin(self, angle_deg)
    }

    /// Voir [`mesh::sweep`](fn@crate::ops::mesh::sweep).
    pub fn sweep(&self, mesh_b: &Mesh, n_layers: usize, element_type: ElementType) -> Result<Mesh> {
        crate::ops::mesh::sweep(self, mesh_b, n_layers, element_type)
    }

    /// Voir [`mesh::sweep_solid`](fn@crate::ops::mesh::sweep_solid).
    pub fn sweep_solid(&self, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
        crate::ops::mesh::sweep_solid(self, mesh_b, n_layers)
    }

    /// Voir [`mesh::symmetry_line`](fn@crate::ops::mesh::symmetry_line).
    pub fn symmetry_line(&self, a: &[f64], b: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_line(self, a, b)
    }

    /// Voir [`mesh::symmetry_plane`](fn@crate::ops::mesh::symmetry_plane).
    pub fn symmetry_plane(&self, a: &[f64], b: &[f64], c: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_plane(self, a, b, c)
    }

    /// Voir [`mesh::symmetry_point`](fn@crate::ops::mesh::symmetry_point).
    pub fn symmetry_point(&self, center: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::symmetry_point(self, center)
    }

    /// Voir [`mesh::to_poi1`](fn@crate::ops::mesh::to_poi1).
    pub fn to_poi1(&self) -> Result<Mesh> {
        crate::ops::mesh::to_poi1(self)
    }

    /// Voir [`mesh::to_quadratic`](fn@crate::ops::mesh::to_quadratic).
    pub fn to_quadratic(&self) -> Result<Mesh> {
        crate::ops::mesh::to_quadratic(self)
    }

    /// Voir [`mesh::transfinite`](fn@crate::ops::mesh::transfinite).
    pub fn transfinite(
        &self,
        side2: &Mesh,
        side3: &Mesh,
        side4: &Mesh,
        element_type: ElementType,
    ) -> Result<Mesh> {
        crate::ops::mesh::transfinite(self, side2, side3, side4, element_type)
    }

    /// Voir [`mesh::translate`](fn@crate::ops::mesh::translate).
    pub fn translate(&self, vector: &[f64]) -> Result<Mesh> {
        crate::ops::mesh::translate(self, vector)
    }

    /// Voir [`mesh::triangulate_surface`](fn@crate::ops::mesh::triangulate_surface).
    pub fn triangulate_surface(
        &self,
        element_type: ElementType,
        target_size: Option<f64>,
    ) -> Result<Mesh> {
        crate::ops::mesh::triangulate_surface(self, element_type, target_size)
    }

    /// Voir [`mesh::triangulate_volume`](fn@crate::ops::mesh::triangulate_volume).
    pub fn triangulate_volume(
        &self,
        target_size: Option<f64>,
        allow_surface_nodes: bool,
    ) -> Result<Mesh> {
        crate::ops::mesh::triangulate_volume(self, target_size, allow_surface_nodes)
    }
}

// `select` part d'un champ mais rend un `Mesh` : la fonction libre vit donc
// dans `ops::mesh`, et l'`impl` la suit. Le nom perd son qualificatif — le
// type du sujet dit déjà s'il s'agit de nœuds ou de mailles.
impl NodeField {
    /// Voir [`mesh::select_nodes`](fn@crate::ops::mesh::select_nodes).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_nodes(self, band, components)
    }
}

impl ElementField {
    /// Voir [`mesh::select_cells`](fn@crate::ops::mesh::select_cells).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_cells(self, band, components)
    }
}

impl SubNodeField {
    /// Voir [`mesh::select_sub_nodes`](fn@crate::ops::mesh::select_sub_nodes).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_sub_nodes(self, band, components)
    }
}

impl SubElementField {
    /// Voir [`mesh::select_sub_cells`](fn@crate::ops::mesh::select_sub_cells).
    pub fn select(&self, band: &Band, components: Option<Vec<String>>) -> Result<Mesh> {
        crate::ops::mesh::select_sub_cells(self, band, components)
    }
}
