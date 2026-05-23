//! ElementField — multi-component values per `(cell, Gauss point)` on a
//! [`SubFESpace`].
//!
//! Where [`crate::node_field::NodeField`] stores values **at nodes**,
//! [`ElementField`] stores them **at the Gauss points of every cell** of a
//! finite-element subspace. It is the natural support for:
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
//! On construction the field captures the three dimensions of the host
//! `SubFESpace`:
//!
//! - `cell_count`  — number of cells (`SubMesh::cell_count` at that moment);
//! - `gauss_count` — number of Gauss points per cell;
//! - `component_count` — chosen by the caller.
//!
//! The internal buffer is sized accordingly and **never reallocated**. The
//! mesh topology underlying the FE space is expected to stay frozen for
//! the lifetime of the field (per the contract documented on
//! [`crate::fe_space::FiniteElementSpace`]). The Gauss-point coordinates
//! and weights are kept as reference data on the `SubFESpace` itself and
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
//! use pyrucast::configuration::Configuration;
//! use pyrucast::element_field::ElementField;
//! use pyrucast::element_type::ElementType;
//! use pyrucast::fe_space::{FiniteElementSpace, SubFESpace};
//! use pyrucast::interpolation::Interpolation;
//! use pyrucast::mesh::{Mesh, SubMesh};
//! use pyrucast::node::Node;
//! use pyrucast::quadrature::QuadratureRule;
//! use pyrucast::store::{insert, with};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
//! let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//! let mesh_h = insert(mesh);
//!
//! let fes = FiniteElementSpace::lagrange1(mesh_h).unwrap();
//! let sub_h = fes.subspace(0).unwrap();
//!
//! // Linear elasticity 2D — two material properties (E, nu).
//! let mut mat = ElementField::new(sub_h, vec!["E".into(), "nu".into()]).unwrap();
//! mat.set_uniform("E", 210e9).unwrap();
//! mat.set_uniform("nu", 0.3).unwrap();
//!
//! assert_eq!(mat.cell_count(), 1);
//! assert_eq!(mat.gauss_count(), 3);   // TRI3 Hammer
//! assert_eq!(mat.component_count(), 2);
//! assert_eq!(mat.value(0, 0, "E").unwrap(), 210e9);
//! ```

use crate::error::{PyrucastError, Result};
use crate::fe_space::SubFESpace;
use crate::store::{with, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ─── ElementField ──────────────────────────────────────────────────────────

/// Multi-component values per `(cell, Gauss point)` on a [`SubFESpace`].
///
/// Layout: flat row-major in the order *cell → gauss → component*
/// (see the module-level documentation).
#[derive(Serialize, Deserialize)]
pub struct ElementField {
    fespace: Handle<SubFESpace>,
    components: Vec<String>,
    /// Dimensions captured at construction; the buffer is never resized.
    n_cells: usize,
    n_gauss: usize,
    /// Flat row-major buffer of length `n_cells * n_gauss * components.len()`.
    values: Vec<f64>,
}

impl ElementField {
    /// Build a field on the given FE subspace with the supplied component
    /// names. Every value is initialized to `0.0`.
    ///
    /// # Errors
    ///
    /// - `components` is empty;
    /// - `components` contains a duplicate name.
    pub fn new(fespace: Handle<SubFESpace>, components: Vec<String>) -> Result<Self> {
        check_components(&components)?;
        let (n_cells, n_gauss) = with(&fespace, |s| -> Result<_> {
            Ok((s.cell_count()?, s.gauss_count()))
        })??;
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

    /// Convenience: build a field with a uniform value per component.
    ///
    /// `values_per_component` must have the same length as `components`.
    pub fn from_uniform_per_component(
        fespace: Handle<SubFESpace>,
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
        let mut field = Self::new(fespace, components)?;
        for (c, &v) in values_per_component.iter().enumerate() {
            field.fill_component(c, v);
        }
        Ok(field)
    }

    // ── Structural accessors ────────────────────────────────────────────────

    /// Handle to the underlying FE subspace (internal clone).
    pub fn fespace(&self) -> Handle<SubFESpace> {
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

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Component names, in order.
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Index of a named component, or `None` if absent.
    pub fn component_index(&self, name: &str) -> Option<usize> {
        self.components.iter().position(|c| c == name)
    }

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

    /// Set every `(cell, gauss)` entry of one component to the same value.
    ///
    /// Convenience for constant-per-domain material properties.
    pub fn set_uniform(&mut self, component: &str, value: f64) -> Result<()> {
        let c = self.component_index_or_err(component)?;
        self.fill_component(c, value);
        Ok(())
    }

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

    // ── Scalar operations on a single component ─────────────────────────────

    /// Add `scalar` to every entry of `component`.
    pub fn add_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let c = self.component_index_or_err(component)?;
        let n_comp = self.components.len();
        for i in 0..self.n_cells * self.n_gauss {
            self.values[i * n_comp + c] += scalar;
        }
        Ok(())
    }

    /// Subtract `scalar` from every entry of `component`.
    pub fn sub_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let c = self.component_index_or_err(component)?;
        let n_comp = self.components.len();
        for i in 0..self.n_cells * self.n_gauss {
            self.values[i * n_comp + c] -= scalar;
        }
        Ok(())
    }

    /// Multiply every entry of `component` by `scalar`.
    pub fn mul_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let c = self.component_index_or_err(component)?;
        let n_comp = self.components.len();
        for i in 0..self.n_cells * self.n_gauss {
            self.values[i * n_comp + c] *= scalar;
        }
        Ok(())
    }

    /// Divide every entry of `component` by `scalar` (errors on zero).
    pub fn div_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        if scalar == 0.0 {
            return Err(PyrucastError::Message(
                "div_to_component: division by zero".into(),
            ));
        }
        let c = self.component_index_or_err(component)?;
        let n_comp = self.components.len();
        for i in 0..self.n_cells * self.n_gauss {
            self.values[i * n_comp + c] /= scalar;
        }
        Ok(())
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn fill_component(&mut self, comp: usize, value: f64) {
        let n_comp = self.components.len();
        for i in 0..self.n_cells * self.n_gauss {
            self.values[i * n_comp + comp] = value;
        }
    }

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
                "ElementField: cell index {} ≥ cell_count {}",
                cell, self.n_cells
            )));
        }
        Ok(())
    }

    fn check_gauss(&self, gauss: usize) -> Result<()> {
        if gauss >= self.n_gauss {
            return Err(PyrucastError::Message(format!(
                "ElementField: gauss index {} ≥ gauss_count {}",
                gauss, self.n_gauss
            )));
        }
        Ok(())
    }

    fn check_comp(&self, comp: usize) -> Result<()> {
        if comp >= self.components.len() {
            return Err(PyrucastError::Message(format!(
                "ElementField: component index {} ≥ component_count {}",
                comp,
                self.components.len()
            )));
        }
        Ok(())
    }

    fn component_index_or_err(&self, component: &str) -> Result<usize> {
        self.component_index(component)
            .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", component)))
    }
}

fn check_components(components: &[String]) -> Result<()> {
    if components.is_empty() {
        return Err(PyrucastError::Message(
            "ElementField requires at least one component".into(),
        ));
    }
    for i in 0..components.len() {
        for j in (i + 1)..components.len() {
            if components[i] == components[j] {
                return Err(PyrucastError::Message(format!(
                    "duplicate component name: {}",
                    components[i]
                )));
            }
        }
    }
    Ok(())
}

// ─── Clone ─────────────────────────────────────────────────────────────────

impl Clone for ElementField {
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

impl fmt::Debug for ElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElementField")
            .field("cell_count", &self.n_cells)
            .field("gauss_count", &self.n_gauss)
            .field("components", &self.components)
            .finish()
    }
}

impl fmt::Display for ElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElementField: {} cell(s) × {} gauss × {} component(s) [{}]",
            self.n_cells,
            self.n_gauss,
            self.components.len(),
            self.components.join(", ")
        )
    }
}

// ─── Operators field OP f64 ────────────────────────────────────────────────
//
// Consuming versions (mutate self in place); reference versions clone first.

impl Add<f64> for ElementField {
    type Output = ElementField;
    fn add(mut self, rhs: f64) -> ElementField {
        for v in &mut self.values {
            *v += rhs;
        }
        self
    }
}

impl Add<f64> for &ElementField {
    type Output = ElementField;
    fn add(self, rhs: f64) -> ElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v += rhs;
        }
        out
    }
}

impl Sub<f64> for ElementField {
    type Output = ElementField;
    fn sub(mut self, rhs: f64) -> ElementField {
        for v in &mut self.values {
            *v -= rhs;
        }
        self
    }
}

impl Sub<f64> for &ElementField {
    type Output = ElementField;
    fn sub(self, rhs: f64) -> ElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v -= rhs;
        }
        out
    }
}

impl Mul<f64> for ElementField {
    type Output = ElementField;
    fn mul(mut self, rhs: f64) -> ElementField {
        for v in &mut self.values {
            *v *= rhs;
        }
        self
    }
}

impl Mul<f64> for &ElementField {
    type Output = ElementField;
    fn mul(self, rhs: f64) -> ElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v *= rhs;
        }
        out
    }
}

impl Div<f64> for ElementField {
    type Output = ElementField;
    fn div(mut self, rhs: f64) -> ElementField {
        for v in &mut self.values {
            *v /= rhs;
        }
        self
    }
}

impl Div<f64> for &ElementField {
    type Output = ElementField;
    fn div(self, rhs: f64) -> ElementField {
        let mut out = self.clone();
        for v in &mut out.values {
            *v /= rhs;
        }
        out
    }
}

// ─── Python binding ────────────────────────────────────────────────────────

#[cfg(feature = "python-api")]
mod python {
    use super::*;
    use crate::fe_space::PySubFESpace;
    use crate::store::insert;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Python wrapper for [`ElementField`].
    #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
    #[pyclass(name = "ElementField")]
    pub struct PyElementField {
        pub(crate) handle: Handle<ElementField>,
    }

    #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
    #[pymethods]
    impl PyElementField {
        /// `ElementField(subfespace, components)` — zero-initialized field.
        #[new]
        fn py_new(fespace: PyRef<PySubFESpace>, components: Vec<String>) -> PyResult<Self> {
            let field = ElementField::new(fespace.handle.clone(), components)?;
            Ok(Self { handle: insert(field) })
        }

        /// Alternate constructor: uniform value per component.
        #[classmethod]
        fn from_uniform_per_component(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            fespace: PyRef<PySubFESpace>,
            components: Vec<String>,
            values_per_component: Vec<f64>,
        ) -> PyResult<Self> {
            let field = ElementField::from_uniform_per_component(
                fespace.handle.clone(),
                components,
                &values_per_component,
            )?;
            Ok(Self { handle: insert(field) })
        }

        fn cell_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |f| f.cell_count())?)
        }

        fn gauss_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |f| f.gauss_count())?)
        }

        fn component_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |f| f.component_count())?)
        }

        fn components(&self) -> PyResult<Vec<String>> {
            Ok(with(&self.handle, |f| f.components().to_vec())?)
        }

        fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
            Ok(with(&self.handle, |f| f.component_index(name))?)
        }

        fn get(&self, cell: usize, gauss: usize, comp: usize) -> PyResult<f64> {
            Ok(with(&self.handle, |f| f.get(cell, gauss, comp))??)
        }

        fn set(&self, cell: usize, gauss: usize, comp: usize, value: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.set(cell, gauss, comp, value))??;
            Ok(())
        }

        fn value(&self, cell: usize, gauss: usize, component: &str) -> PyResult<f64> {
            Ok(with(&self.handle, |f| f.value(cell, gauss, component))??)
        }

        fn set_value(
            &self,
            cell: usize,
            gauss: usize,
            component: &str,
            value: f64,
        ) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| {
                f.set_value(cell, gauss, component, value)
            })??;
            Ok(())
        }

        fn point_values(&self, cell: usize, gauss: usize) -> PyResult<Vec<f64>> {
            Ok(with(&self.handle, |f| {
                f.point_values(cell, gauss).map(|s| s.to_vec())
            })??)
        }

        fn set_uniform(&self, component: &str, value: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.set_uniform(component, value))??;
            Ok(())
        }

        fn set_cell_uniform(&self, cell: usize, component: &str, value: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| {
                f.set_cell_uniform(cell, component, value)
            })??;
            Ok(())
        }

        fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.add_to_component(component, scalar))??;
            Ok(())
        }

        fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.sub_to_component(component, scalar))??;
            Ok(())
        }

        fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.mul_to_component(component, scalar))??;
            Ok(())
        }

        fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            crate::store::with_mut(&self.handle, |f| f.div_to_component(component, scalar))??;
            Ok(())
        }

        // ── Scalar operators (return a new field) ───────────────────────────

        fn __add__(&self, rhs: f64) -> PyResult<PyElementField> {
            let res = with(&self.handle, |f| f + rhs)?;
            Ok(PyElementField { handle: insert(res) })
        }

        fn __sub__(&self, rhs: f64) -> PyResult<PyElementField> {
            let res = with(&self.handle, |f| f - rhs)?;
            Ok(PyElementField { handle: insert(res) })
        }

        fn __mul__(&self, rhs: f64) -> PyResult<PyElementField> {
            let res = with(&self.handle, |f| f * rhs)?;
            Ok(PyElementField { handle: insert(res) })
        }

        fn __truediv__(&self, rhs: f64) -> PyResult<PyElementField> {
            let res = with(&self.handle, |f| f / rhs)?;
            Ok(PyElementField { handle: insert(res) })
        }

        /// `field[cell, gauss, "name"]` — raises ValueError if the component
        /// is unknown.
        fn __getitem__(&self, key: (usize, usize, String)) -> PyResult<f64> {
            let (cell, gauss, comp) = key;
            with(&self.handle, |f| f.value(cell, gauss, &comp))?
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        /// `field[cell, gauss, "name"] = value`.
        fn __setitem__(&self, key: (usize, usize, String), value: f64) -> PyResult<()> {
            let (cell, gauss, comp) = key;
            crate::store::with_mut(&self.handle, |f| f.set_value(cell, gauss, &comp, value))??;
            Ok(())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |f| format!("{:?}", f))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |f| format!("{}", f))?)
        }
    }
}

#[cfg(feature = "python-api")]
pub use python::PyElementField;

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Configuration;
    use crate::element_type::ElementType;
    use crate::fe_space::{FiniteElementSpace, SubFESpace};
    use crate::interpolation::Interpolation;
    use crate::mesh::{Mesh, SubMesh};
    use crate::node::Node;
    use crate::quadrature::QuadratureRule;
    use crate::store::insert;

    fn make_tri3_subfespace() -> Handle<SubFESpace> {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        insert(SubFESpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap())
    }

    fn make_multi_cell_tri3_subfespace(n_cells: usize) -> Handle<SubFESpace> {
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
        insert(SubFESpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap())
    }

    #[test]
    fn new_zero_initialized() {
        let sub = make_tri3_subfespace();
        let f = ElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        assert_eq!(f.cell_count(), 1);
        assert_eq!(f.gauss_count(), 3);
        assert_eq!(f.component_count(), 2);
        for g in 0..3 {
            assert_eq!(f.get(0, g, 0).unwrap(), 0.0);
            assert_eq!(f.get(0, g, 1).unwrap(), 0.0);
        }
    }

    #[test]
    fn new_rejects_empty_components() {
        let sub = make_tri3_subfespace();
        assert!(ElementField::new(sub, vec![]).is_err());
    }

    #[test]
    fn new_rejects_duplicate_components() {
        let sub = make_tri3_subfespace();
        assert!(ElementField::new(sub, vec!["E".into(), "E".into()]).is_err());
    }

    #[test]
    fn get_set_roundtrip() {
        let sub = make_multi_cell_tri3_subfespace(3);
        let mut f = ElementField::new(sub, vec!["sigma_xx".into(), "sigma_yy".into()]).unwrap();
        assert_eq!(f.cell_count(), 3);
        f.set(0, 0, 0, 1.0).unwrap();
        f.set(1, 2, 1, -3.5).unwrap();
        assert_eq!(f.get(0, 0, 0).unwrap(), 1.0);
        assert_eq!(f.get(1, 2, 1).unwrap(), -3.5);
        assert_eq!(f.get(0, 0, 1).unwrap(), 0.0);
    }

    #[test]
    fn value_and_set_value_by_name() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["T".into(), "P".into()]).unwrap();
        f.set_value(0, 1, "P", 42.0).unwrap();
        assert_eq!(f.value(0, 1, "P").unwrap(), 42.0);
        assert!(f.value(0, 0, "unknown").is_err());
        assert!(f.set_value(0, 0, "unknown", 1.0).is_err());
    }

    #[test]
    fn point_values_returns_all_components() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["a".into(), "b".into(), "c".into()]).unwrap();
        f.set(0, 1, 0, 1.0).unwrap();
        f.set(0, 1, 1, 2.0).unwrap();
        f.set(0, 1, 2, 3.0).unwrap();
        assert_eq!(f.point_values(0, 1).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn out_of_bounds_errors() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.get(99, 0, 0).is_err());
        assert!(f.get(0, 99, 0).is_err());
        assert!(f.get(0, 0, 99).is_err());
        assert!(f.set(99, 0, 0, 0.0).is_err());
        assert!(f.point_values(99, 0).is_err());
    }

    #[test]
    fn set_uniform_fills_every_point_of_one_component() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let mut f = ElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        f.set_uniform("E", 210e9).unwrap();
        for cell in 0..2 {
            for g in 0..3 {
                assert_eq!(f.get(cell, g, 0).unwrap(), 210e9);
                assert_eq!(f.get(cell, g, 1).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn set_cell_uniform_touches_only_one_cell() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let mut f = ElementField::new(sub, vec!["rho".into()]).unwrap();
        f.set_cell_uniform(1, "rho", 7800.0).unwrap();
        for g in 0..3 {
            assert_eq!(f.get(0, g, 0).unwrap(), 0.0);
            assert_eq!(f.get(1, g, 0).unwrap(), 7800.0);
        }
    }

    #[test]
    fn from_uniform_per_component_constructor() {
        let sub = make_tri3_subfespace();
        let f = ElementField::from_uniform_per_component(
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
    fn from_uniform_per_component_length_mismatch_errors() {
        let sub = make_tri3_subfespace();
        assert!(ElementField::from_uniform_per_component(
            sub,
            vec!["a".into(), "b".into()],
            &[1.0]
        )
        .is_err());
    }

    #[test]
    fn component_scalar_ops_isolate_components() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["a".into(), "b".into()]).unwrap();
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
    fn component_scalar_div_by_zero_errors() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.div_to_component("x", 0.0).is_err());
    }

    #[test]
    fn component_scalar_unknown_name_errors() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        assert!(f.add_to_component("missing", 1.0).is_err());
        assert!(f.sub_to_component("missing", 1.0).is_err());
        assert!(f.mul_to_component("missing", 1.0).is_err());
        assert!(f.div_to_component("missing", 1.0).is_err());
    }

    #[test]
    fn clone_is_independent() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        f.set(0, 0, 0, 1.0).unwrap();
        let g = f.clone();
        f.set(0, 0, 0, 99.0).unwrap();
        assert_eq!(g.get(0, 0, 0).unwrap(), 1.0);
    }

    #[test]
    fn operator_add_f64_reference_keeps_self_intact() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        f.set(0, 1, 0, 4.0).unwrap();
        let g = &f + 10.0;
        assert_eq!(g.get(0, 1, 0).unwrap(), 14.0);
        assert_eq!(f.get(0, 1, 0).unwrap(), 4.0); // f unchanged
    }

    #[test]
    fn operator_chained_sub_mul_div() {
        let sub = make_tri3_subfespace();
        let mut f = ElementField::new(sub, vec!["x".into()]).unwrap();
        f.set_uniform("x", 12.0).unwrap();
        let g = (f.clone() - 2.0) * 3.0 / 2.0;
        for gp in 0..3 {
            assert!((g.get(0, gp, 0).unwrap() - 15.0).abs() < 1e-12);
        }
    }

    #[test]
    fn debug_and_display() {
        let sub = make_multi_cell_tri3_subfespace(2);
        let f = ElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
        let d = format!("{:?}", f);
        assert!(d.contains("ElementField"));
        assert!(d.contains("cell_count"));
        assert!(d.contains("E"));
        let s = format!("{}", f);
        assert!(s.contains("ElementField"));
        assert!(s.contains("2 cell(s)"));
        assert!(s.contains("3 gauss"));
        assert!(s.contains("2 component(s)"));
        assert!(s.contains("E, nu"));
    }

    #[test]
    fn integrates_with_finite_element_space() {
        // Sanity: a Mesh+FE space can drive ElementField construction
        // without going through bare SubFESpace handles.
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(insert(mesh)).unwrap();
        let sub = fes.subspace(0).unwrap();
        let f = ElementField::new(sub, vec!["sigma".into()]).unwrap();
        assert_eq!(f.cell_count(), 1);
        assert_eq!(f.gauss_count(), 3);
    }
}
