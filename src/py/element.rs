//! Python wrapper for [`crate::atoms::Element`].

use crate::atoms::Element;
use crate::py::cell::PyCell;
use crate::py::node::PyNode;
use pyo3::prelude::*;

/// A single finite element — its geometry together with the interpolation
/// and quadrature of its space.
///
/// A read-only view obtained by indexing a finite-element subspace
/// (`fes[i][j]`); not constructed directly.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Element")]
pub struct PyElement {
    pub(crate) inner: Element,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElement {
    /// Index of this element within its subspace.
    #[getter]
    fn index(&self) -> PyResult<usize> {
        Ok(self.inner.index())
    }

    /// Number of nodes of this element.
    #[getter]
    fn nodes_per_cell(&self) -> PyResult<usize> {
        Ok(self.inner.nodes_per_cell()?)
    }

    /// Spatial dimension the element lives in.
    #[getter]
    fn space_dim(&self) -> PyResult<usize> {
        Ok(self.inner.space_dim()?)
    }

    /// Reference (parametric) dimension of the element.
    #[getter]
    fn ref_dim(&self) -> PyResult<usize> {
        Ok(self.inner.ref_dim()?)
    }

    /// Number of Gauss (quadrature) points.
    #[getter]
    fn gauss_count(&self) -> PyResult<usize> {
        Ok(self.inner.gauss_count())
    }

    /// Underlying mesh cell view.
    fn cell(&self) -> PyResult<PyCell> {
        let c = self.inner.cell()?;
        Ok(PyCell { inner: c })
    }

    /// Materialised nodes of this element (each refcounted on the
    /// Coords) — symmetric with `Cell.nodes()`.
    fn nodes(&self) -> PyResult<Vec<PyNode>> {
        let nodes = self.inner.cell()?.nodes()?;
        Ok(nodes.into_iter().map(PyNode::from_node).collect())
    }

    /// Reference coordinates of the `g`-th Gauss point.
    fn gauss_xi(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.inner.gauss_xi(g)?)
    }

    /// Weight of the `g`-th Gauss point.
    fn gauss_weight(&self, g: usize) -> PyResult<f64> {
        Ok(self.inner.gauss_weight(g)?)
    }

    /// `N_i(ξ_g)` at the `g`-th Gauss point (length `nodes_per_cell`).
    fn n_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.inner.n_at_g(g))
    }

    /// `∂N_i/∂ξ_j(ξ_g)` at the `g`-th Gauss point.
    ///
    /// Flat row-major: index `[i * ref_dim + j]`.
    fn dn_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.inner.dn_at_g(g)?)
    }

    /// Jacobian `J = ∂x/∂ξ` at the `g`-th Gauss point.
    fn jacobian(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.inner.jacobian(g)?)
    }

    /// `|J|` at the `g`-th Gauss point.
    fn det_jacobian(&self, g: usize) -> PyResult<f64> {
        Ok(self.inner.det_jacobian(g)?)
    }

    /// Physical derivatives `∂N_i/∂x_a` at the `g`-th Gauss point.
    fn dn_dx(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.inner.dn_dx(g)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

crate::impl_dump_pymethod!(value PyElement, inner);
