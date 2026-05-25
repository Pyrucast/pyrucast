//! ElementField — multi-component values per `(cell, Gauss point)` on a
//! [`crate::finite_element_space::FiniteElementSpace`].
//!
//! Hierarchy mirroring [`crate::finite_element_space`]:
//!
//! - [`SubElementField`] — multi-component values per `(cell, Gauss point)`
//!   on a single [`crate::finite_element_space::SubFiniteElementSpace`]. Where
//!   [`crate::containers::node_field::NodeField`] stores values **at nodes**, a
//!   `SubElementField` stores them **at the Gauss points of every cell**
//!   of a finite-element subspace.
//! - [`ElementField`] — aggregate of `SubElementField`, one per subspace
//!   of a [`crate::finite_element_space::FiniteElementSpace`], in the same order.
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
//! [`crate::finite_element_space::FiniteElementSpace`]). The Gauss-point coordinates
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
//! use pyrucast::mesh::configuration::Configuration;
//! use pyrucast::containers::element_field::ElementField;
//! use pyrucast::mesh::element_type::ElementType;
//! use pyrucast::finite_element_space::FiniteElementSpace;
//! use pyrucast::mesh::Mesh;
//! use pyrucast::mesh::node::Node;
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
//! let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//!
//! // Linear elasticity 2D — two material properties (E, nu) on every subspace.
//! let mat = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
//! assert_eq!(mat.subfield_count(), 1);
//!
//! let sub0 = mat.subfield(0).unwrap();
//! with_mut(&sub0, |s| {
//!     s.set_uniform("E", 210e9).unwrap();
//!     s.set_uniform("nu", 0.3).unwrap();
//! })
//! .unwrap();
//!
//! with(&sub0, |s| {
//!     assert_eq!(s.cell_count(), 1);
//!     assert_eq!(s.gauss_count(), 3);   // TRI3 Hammer
//!     assert_eq!(s.component_count(), 2);
//!     assert_eq!(s.value(0, 0, "E").unwrap(), 210e9);
//! })
//! .unwrap();
//! ```

use crate::aggregate::Aggregate;
use crate::error::{PyrucastError, Result};
use crate::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::store::{insert, with, Handle};
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
        let mut field = Self::new(fespace, components)?;
        for (c, &v) in values_per_component.iter().enumerate() {
            field.fill_component(c, v);
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

    fn component_index_or_err(&self, component: &str) -> Result<usize> {
        self.component_index(component)
            .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", component)))
    }
}

fn check_components(components: &[String]) -> Result<()> {
    if components.is_empty() {
        return Err(PyrucastError::Message(
            "SubElementField requires at least one component".into(),
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
        f.debug_struct("SubElementField")
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
#[derive(Serialize, Deserialize)]
pub struct ElementField {
    subfields: Vec<Handle<SubElementField>>,
}

impl Aggregate for ElementField {
    type Sub = SubElementField;
    fn items(&self) -> &[Handle<SubElementField>] {
        &self.subfields
    }
    fn items_mut(&mut self) -> &mut Vec<Handle<SubElementField>> {
        &mut self.subfields
    }
}

impl ElementField {
    /// Build an `ElementField` on `fespace` with the **same** `components`
    /// on every subspace.
    ///
    /// `fespace` must have at least one subspace.
    pub fn new(fespace: &FiniteElementSpace, components: Vec<String>) -> Result<Self> {
        let n_sub = fespace.subspace_count();
        if n_sub == 0 {
            return Err(PyrucastError::Message(
                "ElementField: FE space has no subspace".into(),
            ));
        }
        check_components(&components)?;
        let mut subfields = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            let sub = fespace.subspace(i)?;
            let sf = SubElementField::new(sub, components.clone())?;
            subfields.push(insert(sf));
        }
        Ok(Self { subfields })
    }

    /// Build an `ElementField` with an explicit `components` list per
    /// subspace. `components_per_subspace.len()` must equal
    /// `fespace.subspace_count()`.
    pub fn with(
        fespace: &FiniteElementSpace,
        components_per_subspace: &[Vec<String>],
    ) -> Result<Self> {
        let n_sub = fespace.subspace_count();
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
        let mut subfields = Vec::with_capacity(n_sub);
        for (i, comps) in components_per_subspace.iter().enumerate() {
            let sub = fespace.subspace(i)?;
            let sf = SubElementField::new(sub, comps.clone())?;
            subfields.push(insert(sf));
        }
        Ok(Self { subfields })
    }

    /// Number of sub-fields (= number of subspaces of the host FE space).
    pub fn subfield_count(&self) -> usize {
        self.subfields.len()
    }

    /// Handle to the `i`-th sub-field (internal clone).
    pub fn subfield(&self, i: usize) -> Result<Handle<SubElementField>> {
        self.subfields.get(i).cloned().ok_or_else(|| {
            PyrucastError::Message(format!(
                "ElementField: subfield index {} out of bounds",
                i
            ))
        })
    }
}

impl std::ops::Index<usize> for ElementField {
    type Output = Handle<SubElementField>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.subfields[idx]
    }
}

impl<'a> IntoIterator for &'a ElementField {
    type Item = &'a Handle<SubElementField>;
    type IntoIter = std::slice::Iter<'a, Handle<SubElementField>>;
    fn into_iter(self) -> Self::IntoIter {
        self.subfields.iter()
    }
}

impl fmt::Debug for ElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElementField")
            .field("subfield_count", &self.subfields.len())
            .finish()
    }
}

impl fmt::Display for ElementField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ElementField: {} subfield(s)", self.subfields.len())
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::configuration::Configuration;
    use crate::mesh::element_type::ElementType;
    use crate::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
    use crate::finite_element_space::interpolation::Interpolation;
    use crate::mesh::{Mesh, SubMesh};
    use crate::mesh::node::Node;
    use crate::finite_element_space::quadrature::QuadratureRule;
    use crate::store::{insert, with, with_mut};

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
        let mut mesh = Mesh::new(cfg.clone());
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
        mesh.add_submesh(sm_tri).unwrap();
        mesh.add_submesh(sm_qua).unwrap();
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
        assert_eq!(ef.subfield_count(), 2);

        // TRI3: 1 cell × 3 gauss × 2 components
        with(&ef.subfield(0).unwrap(), |s| {
            assert_eq!(s.cell_count(), 1);
            assert_eq!(s.gauss_count(), 3);
            assert_eq!(s.component_count(), 2);
            assert_eq!(s.components(), &["E", "nu"]);
        })
        .unwrap();

        // QUA4: 1 cell × 4 gauss × 2 components
        with(&ef.subfield(1).unwrap(), |s| {
            assert_eq!(s.cell_count(), 1);
            assert_eq!(s.gauss_count(), 4);
            assert_eq!(s.component_count(), 2);
        })
        .unwrap();
    }

    #[test]
    fn ef_with_supports_per_subspace_components() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let comps = vec![vec!["k".into()], vec!["E".into(), "nu".into()]];
        let ef = ElementField::with(&fes, &comps).unwrap();
        assert_eq!(ef.subfield_count(), 2);
        with(&ef.subfield(0).unwrap(), |s| {
            assert_eq!(s.components(), &["k"]);
        })
        .unwrap();
        with(&ef.subfield(1).unwrap(), |s| {
            assert_eq!(s.components(), &["E", "nu"]);
        })
        .unwrap();
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
        let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        assert!(ef.subfield(5).is_err());
    }

    #[test]
    fn ef_aggregate_iter_and_index() {
        let mesh = make_mesh_with_tri_and_qua();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["k".into()]).unwrap();
        // Iteration walks all subfields in order.
        let counts: Vec<usize> = ef
            .into_iter()
            .map(|h| with(h, |s| s.gauss_count()).unwrap())
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
        with_mut(&ef.subfield(0).unwrap(), |s| s.set_uniform("k", 1.5).unwrap()).unwrap();
        with_mut(&ef.subfield(1).unwrap(), |s| s.set_uniform("k", 2.5).unwrap()).unwrap();
        with(&ef.subfield(0).unwrap(), |s| {
            assert_eq!(s.value(0, 0, "k").unwrap(), 1.5);
        })
        .unwrap();
        with(&ef.subfield(1).unwrap(), |s| {
            assert_eq!(s.value(0, 0, "k").unwrap(), 2.5);
        })
        .unwrap();
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
