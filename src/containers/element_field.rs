//! ElementField — multi-component values per `(cell, Gauss point)` on a
//! [`crate::containers::finite_element_space::FiniteElementSpace`].
//!
//! Hierarchy mirroring [`crate::containers::finite_element_space`]:
//!
//! - [`SubElementField`] — multi-component values per `(cell, Gauss point)`
//!   on a single [`crate::containers::finite_element_space::SubFiniteElementSpace`]. Where
//!   [`crate::containers::node_field::SubNodeField`] stores values **at nodes**, a
//!   `SubElementField` stores them **at the Gauss points of every cell**
//!   of a finite-element subspace.
//! - [`ElementField`] — aggregate of `SubElementField`, one per subspace
//!   of a [`crate::containers::finite_element_space::FiniteElementSpace`], in the same order.
//!
//! Typical uses:
//!
//! - **material properties** (Young's modulus, density, conductivity, …)
//!   evaluated at the Gauss points where the integrals are computed;
//! - **state / internal variables** (plastic strain, damage, hardening, …)
//!   that need to be remembered cell-by-cell, point-by-point;
//! - **derived quantities** (stresses, strains, fluxes, …) extracted from
//!   a solution for post-treatment.
//!
//! # Snapshot of the FE space layout
//!
//! On construction every `SubElementField` captures three dimensions of
//! its host `SubFiniteElementSpace`:
//!
//! - `cell_count`  — number of cells (`SubMesh::cell_count` at that moment);
//! - `gauss_count` — number of Gauss points per cell;
//! - `component_count` — chosen by the caller.
//!
//! The internal buffer is sized accordingly and **never reallocated**. The
//! mesh topology underlying the FE space is expected to stay frozen for
//! the lifetime of the field (per the contract documented on
//! [`crate::containers::finite_element_space::FiniteElementSpace`]). The Gauss-point coordinates
//! and weights are kept as reference data on the `SubFiniteElementSpace` itself and
//! may be re-read on demand; only the user data lives here.
//!
//! # Layout
//!
//! Values are stored flat, row-major, in the order **cell → gauss →
//! component** so that contiguous reads of a single cell or a single
//! Gauss point are cache-friendly:
//!
//! ```text
//! values[cell_idx * gauss_count * component_count
//!        + g * component_count
//!        + c]
//! ```
//!
//! # Example
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::containers::field::SubField;
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::element_field::ElementField;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, read, write};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//!
//! // Linear elasticity 2D — two material properties (E, nu) on every subspace.
//! let mat = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
//! assert_eq!(mat.len(), 1);
//!
//! let sub0 = mat.get(0).unwrap();
//! {
//!     let mut s = write(&sub0).unwrap();
//!     s.set_uniform("E", 210e9).unwrap();
//!     s.set_uniform("nu", 0.3).unwrap();
//! }
//!
//! let s = read(&sub0).unwrap();
//! assert_eq!(s.cell_count(), 1);
//! assert_eq!(s.gauss_count(), 3);   // TRI3 Hammer
//! assert_eq!(s.component_count(), 2);
//! assert_eq!(s.value(0, 0, "E").unwrap(), 210e9);
//! ```

use crate::aggregate::Aggregate;
use crate::containers::field::SubField;
use crate::error::{PyrucastError, Result};
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ─── SubElementField ───────────────────────────────────────────────────────

/// Multi-component values per `(cell, Gauss point)` on a single
/// [`SubFiniteElementSpace`].
///
/// Layout: flat row-major in the order *cell → gauss → component*
/// (see the module-level documentation).
#[derive(Serialize, Deserialize)]
pub struct SubElementField {
    fespace: Handle<SubFiniteElementSpace>,
    components: Vec<String>,
    /// Dimensions captured at construction; the buffer is never resized.
    n_cells: usize,
    n_gauss: usize,
    /// Flat row-major buffer of length `n_cells * n_gauss * components.len()`.
    values: Vec<f64>,
}

impl SubElementField {
    /// Build a sub-field on the given FE subspace with the supplied
    /// component names. Every value is initialized to `0.0`.
    ///
    /// # Errors
    ///
    /// - `components` is empty;
    /// - `components` contains a duplicate name.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, components: Vec<String>) -> Result<Self> {
        crate::containers::field::check_components("SubElementField", &components)?;
        let (n_cells, n_gauss) = {
            let s = read(&fespace)?;
            (s.cell_count()?, s.gauss_count())
        };
        let n_comp = components.len();
        let values = vec![0.0; n_cells * n_gauss * n_comp];
        Ok(Self {
            fespace,
            components,
            n_cells,
            n_gauss,
            values,
        })
    }

    /// Convenience: build a sub-field with a uniform value per component.
    ///
    /// `values_per_component` must have the same length as `components`.
    pub fn from_uniform_per_component(
        fespace: Handle<SubFiniteElementSpace>,
        components: Vec<String>,
        values_per_component: &[f64],
    ) -> Result<Self> {
        if values_per_component.len() != components.len() {
            return Err(PyrucastError::Message(format!(
                "from_uniform_per_component: {} values supplied for {} components",
                values_per_component.len(),
                components.len()
            )));
        }
        let names = components.clone();
        let mut field = Self::new(fespace, components)?;
        for (name, &v) in names.iter().zip(values_per_component) {
            field.set_uniform(name, v)?;
        }
        Ok(field)
    }

    // ── Structural accessors ────────────────────────────────────────────────

    /// Handle to the underlying FE subspace (internal clone).
    pub fn fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// Number of cells captured at construction.
    pub fn cell_count(&self) -> usize {
        self.n_cells
    }

    /// Number of Gauss points per cell.
    pub fn gauss_count(&self) -> usize {
        self.n_gauss
    }

    // `components`, `component_count`, `component_index`, … come from
    // the [`crate::containers::field::SubField`] trait.

    // ── Value access by indices ─────────────────────────────────────────────

    /// Read the value at `(cell, gauss, component_index)`.
    pub fn get(&self, cell: usize, gauss: usize, comp: usize) -> Result<f64> {
        let idx = self.linear_index(cell, gauss, comp)?;
        Ok(self.values[idx])
    }

    /// Write the value at `(cell, gauss, component_index)`.
    pub fn set(&mut self, cell: usize, gauss: usize, comp: usize, value: f64) -> Result<()> {
        let idx = self.linear_index(cell, gauss, comp)?;
        self.values[idx] = value;
        Ok(())
    }

    /// All component values at `(cell, gauss)`, in component order
    /// (length = `component_count`).
    pub fn point_values(&self, cell: usize, gauss: usize) -> Result<&[f64]> {
        self.check_cell(cell)?;
        self.check_gauss(gauss)?;
        let n_comp = self.components.len();
        let start = (cell * self.n_gauss + gauss) * n_comp;
        Ok(&self.values[start..start + n_comp])
    }

    // ── Value access by component name ──────────────────────────────────────

    /// Read by `(cell, gauss, component name)`.
    pub fn value(&self, cell: usize, gauss: usize, component: &str) -> Result<f64> {
        let c = self.component_index_or_err(component)?;
        self.get(cell, gauss, c)
    }

    /// Write by `(cell, gauss, component name)`.
    pub fn set_value(
        &mut self,
        cell: usize,
        gauss: usize,
        component: &str,
        value: f64,
    ) -> Result<()> {
        let c = self.component_index_or_err(component)?;
        self.set(cell, gauss, c, value)
    }

    // ── Bulk fillers ────────────────────────────────────────────────────────

    // `set_uniform` (constant-per-domain material properties) comes from
    // the [`crate::containers::field::SubField`] trait.

    /// Set every Gauss point of a given cell to the same value for one
    /// component (cell-piecewise-constant material).
    pub fn set_cell_uniform(&mut self, cell: usize, component: &str, value: f64) -> Result<()> {
        self.check_cell(cell)?;
        let c = self.component_index_or_err(component)?;
        let n_comp = self.components.len();
        for g in 0..self.n_gauss {
            self.values[(cell * self.n_gauss + g) * n_comp + c] = value;
        }
        Ok(())
    }

    // Scalar per-component operations (`add_to_component`, …) come from
    // the [`crate::containers::field::SubField`] trait.

    // ── Internals ───────────────────────────────────────────────────────────

    fn linear_index(&self, cell: usize, gauss: usize, comp: usize) -> Result<usize> {
        self.check_cell(cell)?;
        self.check_gauss(gauss)?;
        self.check_comp(comp)?;
        let n_comp = self.components.len();
        Ok((cell * self.n_gauss + gauss) * n_comp + comp)
    }

    fn check_cell(&self, cell: usize) -> Result<()> {
        if cell >= self.n_cells {
            return Err(PyrucastError::Message(format!(
                "SubElementField: cell index {} ≥ cell_count {}",
                cell, self.n_cells
            )));
        }
        Ok(())
    }

    fn check_gauss(&self, gauss: usize) -> Result<()> {
        if gauss >= self.n_gauss {
            return Err(PyrucastError::Message(format!(
                "SubElementField: gauss index {} ≥ gauss_count {}",
                gauss, self.n_gauss
            )));
        }
        Ok(())
    }

    fn check_comp(&self, comp: usize) -> Result<()> {
        if comp >= self.components.len() {
            return Err(PyrucastError::Message(format!(
                "SubElementField: component index {} ≥ component_count {}",
                comp,
                self.components.len()
            )));
        }
        Ok(())
    }

}

impl crate::containers::field::SubField for SubElementField {
    fn components(&self) -> &[String] {
        &self.components
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
    fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}


// ─── Clone ─────────────────────────────────────────────────────────────────

impl Clone for SubElementField {
    fn clone(&self) -> Self {
        Self {
            fespace: self.fespace.clone(),
            components: self.components.clone(),
            n_cells: self.n_cells,
            n_gauss: self.n_gauss,
            values: self.values.clone(),
        }
    }
}

// ─── Debug / Display ───────────────────────────────────────────────────────

impl fmt::Debug for SubElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bounded structure only — the per-(cell, gauss) values live in `dump()`.
        f.debug_struct("SubElementField")
            .field("fespace", &self.fespace)
            .field("cell_count", &self.n_cells)
            .field("gauss_count", &self.n_gauss)
            .field("components", &self.components)
            .finish()
    }
}

impl fmt::Display for SubElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubElementField: {} cell(s) × {} gauss × {} component(s) [{}]",
            self.n_cells,
            self.n_gauss,
            self.components.len(),
            self.components.join(", ")
        )
    }
}

impl crate::dump::Dump for SubElementField {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let ncomp = self.components.len();
        let mut headers = vec!["cell".to_string(), "gauss".to_string()];
        headers.extend(self.components.iter().cloned());
        let mut rows = Vec::with_capacity(self.n_cells * self.n_gauss);
        for cell in 0..self.n_cells {
            for g in 0..self.n_gauss {
                let base = (cell * self.n_gauss + g) * ncomp;
                let mut row = vec![cell.to_string(), g.to_string()];
                for c in 0..ncomp {
                    row.push(fmt_float(self.values[base + c], opts.precision));
                }
                rows.push(row);
            }
        }
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── Operators field OP f64 ────────────────────────────────────────────────
//
// Consuming versions (mutate self in place); reference versions clone first.

impl Add<f64> for SubElementField {
    type Output = SubElementField;
    fn add(mut self, rhs: f64) -> SubElementField {
        for v in &mut self.values {
            *v += rhs;
        }
        self
    }
}

impl Add<f64> for &SubElementField {
    type Output = SubElementField;
    fn add(self, rhs: f64) -> SubElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v += rhs;
        }
        out
    }
}

impl Sub<f64> for SubElementField {
    type Output = SubElementField;
    fn sub(mut self, rhs: f64) -> SubElementField {
        for v in &mut self.values {
            *v -= rhs;
        }
        self
    }
}

impl Sub<f64> for &SubElementField {
    type Output = SubElementField;
    fn sub(self, rhs: f64) -> SubElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v -= rhs;
        }
        out
    }
}

impl Mul<f64> for SubElementField {
    type Output = SubElementField;
    fn mul(mut self, rhs: f64) -> SubElementField {
        for v in &mut self.values {
            *v *= rhs;
        }
        self
    }
}

impl Mul<f64> for &SubElementField {
    type Output = SubElementField;
    fn mul(self, rhs: f64) -> SubElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v *= rhs;
        }
        out
    }
}

impl Div<f64> for SubElementField {
    type Output = SubElementField;
    fn div(mut self, rhs: f64) -> SubElementField {
        for v in &mut self.values {
            *v /= rhs;
        }
        self
    }
}

impl Div<f64> for &SubElementField {
    type Output = SubElementField;
    fn div(self, rhs: f64) -> SubElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v /= rhs;
        }
        out
    }
}

// ─── ElementField (aggregate) ──────────────────────────────────────────────

/// Aggregate of [`SubElementField`] — one per subspace of a
/// [`FiniteElementSpace`], in the same order.
///
/// Mirrors the [`FiniteElementSpace`] / [`SubFiniteElementSpace`] hierarchy: an
/// `ElementField` is to `FiniteElementSpace` what a `SubElementField` is
/// to `SubFiniteElementSpace`. The component lists captured by the underlying
/// sub-fields may differ from one subspace to the next.
#[derive(Serialize, Deserialize, Default)]
pub struct ElementField {
    subs: Vec<Handle<SubElementField>>,
}

crate::impl_aggregate!(ElementField, SubElementField, subfield, "subfield(s)");
crate::impl_aggregate_dump!(ElementField);

impl ElementField {
    /// Build an `ElementField` on `fespace` with the **same** `components`
    /// on every subspace.
    ///
    /// `fespace` must have at least one subspace.
    pub fn new(fespace: &FiniteElementSpace, components: Vec<String>) -> Result<Self> {
        let n_sub = fespace.len();
        if n_sub == 0 {
            return Err(PyrucastError::Message(
                "ElementField: FE space has no subspace".into(),
            ));
        }
        crate::containers::field::check_components("SubElementField", &components)?;
        let mut subs = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            let sub = fespace.get(i)?;
            let sf = SubElementField::new(sub, components.clone())?;
            subs.push(insert(sf));
        }
        Ok(Self { subs })
    }

    /// The [`SubElementField`] whose FE subspace handle matches `fespace`.
    ///
    /// The per-zone field-matching primitive shared by the assembly and
    /// behaviour operators. Errors if no sub-field of the aggregate lives on
    /// `fespace` (a missing per-zone material / deformation is always a
    /// caller error, never a silent skip).
    pub(crate) fn sub_for_fespace(
        &self,
        fespace: &Handle<SubFiniteElementSpace>,
    ) -> Result<Handle<SubElementField>> {
        for h in self {
            let matches = {
                let f = read(h)?.fespace();
                f.index() == fespace.index() && f.generation() == fespace.generation()
            };
            if matches {
                return Ok(h.clone());
            }
        }
        Err(PyrucastError::Message(format!(
            "no SubElementField in this ElementField matches the \
             SubFiniteElementSpace at slot {} (generation {})",
            fespace.index(),
            fespace.generation()
        )))
    }

    /// Build an `ElementField` with an explicit `components` list per
    /// subspace. `components_per_subspace.len()` must equal
    /// `fespace.len()`.
    pub fn with(
        fespace: &FiniteElementSpace,
        components_per_subspace: &[Vec<String>],
    ) -> Result<Self> {
        let n_sub = fespace.len();
        if n_sub == 0 {
            return Err(PyrucastError::Message(
                "ElementField: FE space has no subspace".into(),
            ));
        }
        if components_per_subspace.len() != n_sub {
            return Err(PyrucastError::Message(format!(
                "ElementField: {} component list(s) supplied for {} subspace(s)",
                components_per_subspace.len(),
                n_sub
            )));
        }
        let mut subs = Vec::with_capacity(n_sub);
        for (i, comps) in components_per_subspace.iter().enumerate() {
            let sub = fespace.get(i)?;
            let sf = SubElementField::new(sub, comps.clone())?;
            subs.push(insert(sf));
        }
        Ok(Self { subs })
    }

    /// Visualize this field on its own support: each zone knows its
    /// submesh through its FE subspace, so the mesh is reconstructed
    /// (shared, not copied) and coloured by `component` — per-element
    /// nodal fit of the Gauss values, discontinuities preserved.
    /// `smooth` is the subdivision level (`0` = flat per cell).
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        smooth: usize,
    ) -> Result<()> {
        let mut mesh = crate::containers::mesh::Mesh::empty();
        for h in self {
            let fespace = read(h)?.fespace();
            let sm = read(&fespace)?.submesh();
            mesh.add_sub(sm)?;
        }
        crate::viz::render_mesh_with_field(
            &mesh,
            crate::viz::FieldArg::Element(self),
            component,
            scale,
            smooth,
            view,
            save,
        )
    }

}

/// Zero-copy view of an [`ElementField`]'s zones — the
/// [`crate::containers::field::FieldView`] aggregate view specialised
/// to Gauss-point fields (built by `Field::view`). A zone is resolved
/// from its support submesh by handle identity (idx + generation),
/// the reciprocal of [`ElementField::sub_for_fespace`].
pub(crate) type ElementFieldView = crate::containers::field::FieldView<SubElementField>;

impl ElementFieldView {
    /// The zone supported on `submesh` (matched through the zone's FE
    /// subspace), or `None` if no zone lives on it.
    // Consumed by the viz layer (feature-gated).
    #[allow(dead_code)]
    pub(crate) fn zone_for_submesh(
        &self,
        submesh: &Handle<crate::containers::mesh::SubMesh>,
    ) -> Result<Option<&SubElementField>> {
        for z in &self.zones {
            let sm = read(&z.fespace())?.submesh();
            if sm.index() == submesh.index() && sm.generation() == submesh.generation() {
                return Ok(Some(z));
            }
        }
        Ok(None)
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
    use crate::containers::finite_element_space::Interpolation;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::mesh::Node;
    use crate::containers::finite_element_space::QuadratureRule;
    use crate::store::{insert, read, write};

    fn make_tri3_subfespace() -> Handle<SubFiniteElementSpace> {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        insert(SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap())
    }

    fn make_multi_cell_tri3_subfespace(n_cells: usize) -> Handle<SubFiniteElementSpace> {
        // n_cells triangles sharing a common apex, like a fan from origin.
        let cfg = insert(Configuration::new(2).unwrap());
        let apex = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mut perimeter = Vec::with_capacity(n_cells + 1);
        for i in 0..=n_cells {
            let t = i as f64 / n_cells as f64;
            perimeter.push(Node::create_in(cfg.clone(), &[1.0, t]).unwrap());
        }
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            for i in 0..n_cells {
                sm.add_cell(&[apex.id(), perimeter[i].id(), perimeter[i + 1].id()])
                    .unwrap();
            }
            insert(sm)
        };
        insert(SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap())
    }

    fn make_mesh_with_tri_and_qua() -> Mesh {
        let cfg = insert(Configuration::new(2).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_qua).unwrap();
        mesh
    }

    // ── SubElementField ─────────────────────────────────────────────────────

    #[test]
    fn sub_new_zero_initialized() {
        let sub = make_tri3_subfespace();
        let f = SubElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        assert_eq!(f.cell_count(), 1);
        assert_eq!(f.gauss_count(), 3);
        assert_eq!(f.component_count(), 2);
        for g in 0..3 {
            assert_eq!(f.get(0, g, 0).unwrap(), 0.0);
            assert_eq!(f.get(0, g, 1).unwrap(), 0.0);
        }
    }

    #[test]
    fn sub_new_rejects_empty_components() {
        let sub = make_tri3_subfespace();
        assert!(SubElementField::new(sub, vec![]).is_err());
    }

    #[test]
    fn sub_new_rejects_duplicate_components() {
        let sub = make_tri3_subfespace();
        assert!(SubElementField::new(sub, vec!["E".into(), "E".into()]).is_err());
    }

    #[test]
    fn sub_get_set_roundtrip() {
        let sub = make_multi_cell_tri3_subfespace(3);
        let mut f =
            SubElementField::new(sub, vec!["sigma_xx".into(), "sigma_yy".into()]).unwrap();
        assert_eq!(f.cell_count(), 3);
        f.set(0, 0, 0, 1.0).unwrap();
        f.set(1, 2, 1, -3.5).unwrap();
        assert_eq!(f.get(0, 0, 0).unwrap(), 1.0);
        assert_eq!(f.get(1, 2, 1).unwrap(), -3.5);
        assert_eq!(f.get(0, 0, 1).unwrap(), 0.0);
    }

    #[test]
    fn sub_value_and_set_value_by_name() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["T".into(), "P".into()]).unwrap();
        f.set_value(0, 1, "P", 42.0).unwrap();
        assert_eq!(f.value(0, 1, "P").unwrap(), 42.0);
        assert!(f.value(0, 0, "unknown").is_err());
        assert!(f.set_value(0, 0, "unknown", 1.0).is_err());
    }

    #[test]
    fn sub_point_values_returns_all_components() {
        let sub = make_tri3_subfespace();
        let mut f =
            SubElementField::new(sub, vec!["a".into(), "b".into(), "c".into()]).unwrap();
        f.set(0, 1, 0, 1.0).unwrap();
        f.set(0, 1, 1, 2.0).unwrap();
        f.set(0, 1, 2, 3.0).unwrap();
        assert_eq!(f.point_values(0, 1).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sub_out_of_bounds_errors() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.get(99, 0, 0).is_err());
        assert!(f.get(0, 99, 0).is_err());
        assert!(f.get(0, 0, 99).is_err());
        assert!(f.set(99, 0, 0, 0.0).is_err());
        assert!(f.point_values(99, 0).is_err());
    }

    #[test]
    fn sub_set_uniform_fills_every_point_of_one_component() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let mut f = SubElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        f.set_uniform("E", 210e9).unwrap();
        for cell in 0..2 {
            for g in 0..3 {
                assert_eq!(f.get(cell, g, 0).unwrap(), 210e9);
                assert_eq!(f.get(cell, g, 1).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn sub_set_cell_uniform_touches_only_one_cell() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let mut f = SubElementField::new(sub, vec!["rho".into()]).unwrap();
        f.set_cell_uniform(1, "rho", 7800.0).unwrap();
        for g in 0..3 {
            assert_eq!(f.get(0, g, 0).unwrap(), 0.0);
            assert_eq!(f.get(1, g, 0).unwrap(), 7800.0);
        }
    }

    #[test]
    fn sub_from_uniform_per_component_constructor() {
        let sub = make_tri3_subfespace();
        let f = SubElementField::from_uniform_per_component(
            sub,
            vec!["E".into(), "nu".into(), "rho".into()],
            &[210e9, 0.3, 7800.0],
        )
        .unwrap();
        for g in 0..3 {
            assert_eq!(f.get(0, g, 0).unwrap(), 210e9);
            assert_eq!(f.get(0, g, 1).unwrap(), 0.3);
            assert_eq!(f.get(0, g, 2).unwrap(), 7800.0);
        }
    }

    #[test]
    fn sub_from_uniform_per_component_length_mismatch_errors() {
        let sub = make_tri3_subfespace();
        assert!(SubElementField::from_uniform_per_component(
            sub,
            vec!["a".into(), "b".into()],
            &[1.0]
        )
        .is_err());
    }

    #[test]
    fn sub_component_scalar_ops_isolate_components() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["a".into(), "b".into()]).unwrap();
        f.set_uniform("a", 10.0).unwrap();
        f.set_uniform("b", 1.0).unwrap();
        f.add_to_component("a", 5.0).unwrap();
        f.sub_to_component("a", 2.0).unwrap();
        f.mul_to_component("a", 3.0).unwrap();
        f.div_to_component("a", 13.0).unwrap();
        // a went 10 → 15 → 13 → 39 → 3.0
        for g in 0..3 {
            assert!((f.get(0, g, 0).unwrap() - 3.0).abs() < 1e-12);
            assert_eq!(f.get(0, g, 1).unwrap(), 1.0); // unchanged
        }
    }

    #[test]
    fn sub_component_scalar_div_by_zero_errors() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.div_to_component("x", 0.0).is_err());
    }

    #[test]
    fn sub_component_scalar_unknown_name_errors() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.add_to_component("missing", 1.0).is_err());
        assert!(f.sub_to_component("missing", 1.0).is_err());
        assert!(f.mul_to_component("missing", 1.0).is_err());
        assert!(f.div_to_component("missing", 1.0).is_err());
    }

    #[test]
    fn sub_clone_is_independent() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        f.set(0, 0, 0, 1.0).unwrap();
        let g = f.clone();
        f.set(0, 0, 0, 99.0).unwrap();
        assert_eq!(g.get(0, 0, 0).unwrap(), 1.0);
    }

    #[test]
    fn sub_operator_add_f64_reference_keeps_self_intact() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        f.set(0, 1, 0, 4.0).unwrap();
        let g = &f + 10.0;
        assert_eq!(g.get(0, 1, 0).unwrap(), 14.0);
        assert_eq!(f.get(0, 1, 0).unwrap(), 4.0); // f unchanged
    }

    #[test]
    fn sub_operator_chained_sub_mul_div() {
        let sub = make_tri3_subfespace();
        let mut f = SubElementField::new(sub, vec!["x".into()]).unwrap();
        f.set_uniform("x", 12.0).unwrap();
        let g = (f.clone() - 2.0) * 3.0 / 2.0;
        for gp in 0..3 {
            assert!((g.get(0, gp, 0).unwrap() - 15.0).abs() < 1e-12);
        }
    }

    #[test]
    fn sub_debug_and_display() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let f = SubElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        let d = format!("{:?}", f);
        assert!(d.contains("SubElementField"));
        assert!(d.contains("cell_count"));
        assert!(d.contains("E"));
        let s = format!("{}", f);
        assert!(s.contains("SubElementField"));
        assert!(s.contains("2 cell(s)"));
        assert!(s.contains("3 gauss"));
        assert!(s.contains("2 component(s)"));
        assert!(s.contains("E, nu"));
    }

    // ── ElementField (aggregate) ────────────────────────────────────────────

    #[test]
    fn ef_new_creates_one_subfield_per_subspace() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
        assert_eq!(ef.len(), 2);

        // TRI3: 1 cell × 3 gauss × 2 components
        {
            let s = read(&ef.get(0).unwrap()).unwrap();
            assert_eq!(s.cell_count(), 1);
            assert_eq!(s.gauss_count(), 3);
            assert_eq!(s.component_count(), 2);
            assert_eq!(s.components(), &["E", "nu"]);
        }

        // QUA4: 1 cell × 4 gauss × 2 components
        {
            let s = read(&ef.get(1).unwrap()).unwrap();
            assert_eq!(s.cell_count(), 1);
            assert_eq!(s.gauss_count(), 4);
            assert_eq!(s.component_count(), 2);
        }
    }

    #[test]
    fn ef_with_supports_per_subspace_components() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let comps = vec![vec!["k".into()], vec!["E".into(), "nu".into()]];
        let ef = ElementField::with(&fes, &comps).unwrap();
        assert_eq!(ef.len(), 2);
        assert_eq!(read(&ef.get(0).unwrap()).unwrap().components(), &["k"]);
        assert_eq!(
            read(&ef.get(1).unwrap()).unwrap().components(),
            &["E", "nu"]
        );
    }

    #[test]
    fn ef_with_rejects_mismatched_length() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let comps_one = vec![vec!["k".into()]];
        assert!(ElementField::with(&fes, &comps_one).is_err());
        let comps_three = vec![
            vec!["k".into()],
            vec!["k".into()],
            vec!["k".into()],
        ];
        assert!(ElementField::with(&fes, &comps_three).is_err());
    }

    #[test]
    fn ef_subfield_out_of_bounds_errors() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        assert!(ef.get(5).is_err());
    }

    #[test]
    fn ef_aggregate_iter_and_index() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        // Iteration walks all subfields in order.
        let counts: Vec<usize> = ef
            .into_iter()
            .map(|h| read(h).unwrap().gauss_count())
            .collect();
        assert_eq!(counts, vec![3, 4]);
        // Indexing matches subfield().
        let _h = &ef[0];
    }

    #[test]
    fn ef_subfields_are_mutated_independently() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        write(&ef.get(0).unwrap()).unwrap().set_uniform("k", 1.5).unwrap();
        write(&ef.get(1).unwrap()).unwrap().set_uniform("k", 2.5).unwrap();
        assert_eq!(
            read(&ef.get(0).unwrap()).unwrap().value(0, 0, "k").unwrap(),
            1.5
        );
        assert_eq!(
            read(&ef.get(1).unwrap()).unwrap().value(0, 0, "k").unwrap(),
            2.5
        );
    }

    #[test]
    fn ef_debug_and_display() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        let d = format!("{:?}", ef);
        assert!(d.contains("ElementField"));
        let s = format!("{}", ef);
        assert!(s.contains("ElementField"));
        assert!(s.contains("2 subfield"));
    }
}
