//! Python wrappers for [`crate::containers::finite_element_space::SubFiniteElementSpace`] and
//! [`crate::containers::finite_element_space::FiniteElementSpace`].

use crate::error::{PyrucastError, Result};
use crate::containers::finite_element_space::{Element, FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::finite_element_space::Interpolation;
use crate::py::element::PyElement;
use crate::py::mesh::PyMesh;
use crate::containers::finite_element_space::QuadratureRule;
use crate::store::{with, Handle};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

fn parse_interpolation(s: &str) -> PyResult<Interpolation> {
    Interpolation::from_name(s)
        .ok_or_else(|| PyValueError::new_err(format!("unknown interpolation: {s}")))
}

fn parse_quadrature(s: &str) -> PyResult<QuadratureRule> {
    QuadratureRule::from_name(s)
        .ok_or_else(|| PyValueError::new_err(format!("unknown quadrature rule: {s}")))
}

/// Python wrapper for [`SubFiniteElementSpace`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubFiniteElementSpace")]
pub struct PySubFiniteElementSpace {
    pub(crate) handle: Handle<SubFiniteElementSpace>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubFiniteElementSpace {
    #[getter]
    fn element_type(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| s.element_type())??.name().to_string())
    }

    #[getter]
    fn interpolation(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| s.interpolation().name().to_string())?)
    }

    #[getter]
    fn quadrature(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| s.quadrature().name().to_string())?)
    }

    #[getter]
    fn ref_dim(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.ref_dim())??)
    }

    #[getter]
    fn space_dim(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.space_dim())?)
    }

    #[getter]
    fn nodes_per_cell(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.nodes_per_cell())??)
    }

    fn cell_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.cell_count())??)
    }

    fn gauss_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.gauss_count())?)
    }

    /// Reference coordinates of the `g`-th Gauss point.
    fn gauss_xi(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |s| s.gauss_xi(g).map(|x| x.to_vec()))??)
    }

    /// Weight of the `g`-th Gauss point.
    fn gauss_weight(&self, g: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |s| s.gauss_weight(g))??)
    }

    /// `N_i(ξ_g)` at the `g`-th Gauss point (flat, length `nodes_per_cell`).
    fn n_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |s| s.n_at_g(g).map(|x| x.to_vec()))??)
    }

    /// `∂N_i/∂ξ_j(ξ_g)` at the `g`-th Gauss point.
    ///
    /// Flat row-major: index `[i * ref_dim + j]`.
    fn dn_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |s| s.dn_at_g(g).map(|x| x.to_vec()))??)
    }

    /// Jacobian `J = ∂x/∂ξ` of cell `cell_idx` at Gauss point `g`.
    ///
    /// Flat row-major buffer of length `space_dim × ref_dim`,
    /// indexed `[a * ref_dim + k]`.
    fn jacobian(&self, cell_idx: usize, g: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |s| s.jacobian(cell_idx, g))??)
    }

    /// `|J|` — `|det(J)|` if `space_dim == ref_dim`, else
    /// `sqrt(det(JᵀJ))`. Always non-negative.
    fn det_jacobian(&self, cell_idx: usize, g: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |s| s.det_jacobian(cell_idx, g))??)
    }

    /// Physical derivatives `∂N_i/∂x_a` of cell `cell_idx` at Gauss
    /// point `g`. Flat row-major buffer of length
    /// `nodes_per_cell × space_dim`, indexed `[i * space_dim + a]`.
    fn dn_dx(&self, cell_idx: usize, g: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |s| s.dn_dx(cell_idx, g))??)
    }

    /// Element view on the `i`-th cell of this subspace.
    fn element(&self, i: usize) -> PyResult<PyElement> {
        let el = Element::new(self.handle.clone(), i)?;
        Ok(PyElement { inner: el })
    }

    /// All elements as a Python list.
    fn elements(&self) -> PyResult<Vec<PyElement>> {
        let n = with(&self.handle, |s| s.cell_count())??;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(PyElement {
                inner: Element::new(self.handle.clone(), i)?,
            });
        }
        Ok(out)
    }

    /// `len(subspace)` → number of elements (= number of cells).
    fn __len__(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.cell_count())??)
    }

    /// `subspace[i]` → `Element` view on element `i`. Supports negative
    /// indices and raises `IndexError` out of range so
    /// `for el in subspace:` works.
    fn __getitem__(&self, idx: isize) -> PyResult<PyElement> {
        let n = with(&self.handle, |s| s.cell_count())?? as isize;
        let normalized = if idx < 0 { n + idx } else { idx };
        if normalized < 0 || normalized >= n {
            return Err(PyIndexError::new_err(format!(
                "subspace element index {idx} out of range (len={n})"
            )));
        }
        let el = Element::new(self.handle.clone(), normalized as usize)?;
        Ok(PyElement { inner: el })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{:?}", s))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{}", s))?)
    }
}

/// Python wrapper for [`FiniteElementSpace`].
///
/// Owns the `FiniteElementSpace` struct directly — no longer stored
/// in the global store. Identity is the Python object identity.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "FiniteElementSpace")]
pub struct PyFiniteElementSpace {
    pub(crate) inner: FiniteElementSpace,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyFiniteElementSpace {
    /// `FiniteElementSpace(mesh)` — Lagrange-1 + default Gauss for
    /// every submesh.
    ///
    /// `FiniteElementSpace(mesh, interpolation="LAGRANGE1", quadrature="GAUSS")`
    /// — same `(interpolation, quadrature)` applied to every submesh.
    #[new]
    #[pyo3(signature = (mesh, interpolation="LAGRANGE1", quadrature="GAUSS"))]
    fn py_new(
        mesh: PyRef<PyMesh>,
        interpolation: &str,
        quadrature: &str,
    ) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let quad = parse_quadrature(quadrature)?;
        let n_sub = mesh.inner.submesh_count();
        let choices: Vec<(Interpolation, QuadratureRule)> =
            (0..n_sub).map(|_| (interp, quad)).collect();
        let fes = FiniteElementSpace::with(&mesh.inner, &choices)?;
        Ok(Self { inner: fes })
    }

    /// Explicit `(interpolation, quadrature)` per submesh.
    #[classmethod]
    fn with_choices(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh: PyRef<PyMesh>,
        choices: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let parsed: Result<Vec<_>> = choices
            .iter()
            .map(|(i, q)| -> Result<(Interpolation, QuadratureRule)> {
                let interp = Interpolation::from_name(i).ok_or_else(|| {
                    PyrucastError::Message(format!("unknown interpolation: {i}"))
                })?;
                let quad = QuadratureRule::from_name(q).ok_or_else(|| {
                    PyrucastError::Message(format!("unknown quadrature rule: {q}"))
                })?;
                Ok((interp, quad))
            })
            .collect();
        let parsed = parsed?;
        let fes = FiniteElementSpace::with(&mesh.inner, &parsed)?;
        Ok(Self { inner: fes })
    }

    /// Convenience: same as `FiniteElementSpace(mesh)`.
    #[classmethod]
    fn lagrange1(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh: PyRef<PyMesh>,
    ) -> PyResult<Self> {
        let fes = FiniteElementSpace::lagrange1(&mesh.inner)?;
        Ok(Self { inner: fes })
    }

    /// `fes.element(sub_idx, cell_idx)` — Element view on a single cell.
    /// Mirrors `Mesh.cell(sub_idx, cell_idx)`.
    fn element(&self, sub_idx: usize, cell_idx: usize) -> PyResult<PyElement> {
        let el = self.inner.element(sub_idx, cell_idx)?;
        Ok(PyElement { inner: el })
    }

    /// `fes.elements(sub_idx)` — list of every Element in subspace `sub_idx`.
    /// Mirrors `Mesh.cells(sub_idx)`.
    fn elements(&self, sub_idx: usize) -> PyResult<Vec<PyElement>> {
        let iter = self.inner.elements(sub_idx)?;
        Ok(iter.map(|el| PyElement { inner: el }).collect())
    }
}

crate::impl_aggregate_pymethods!(PyFiniteElementSpace, PySubFiniteElementSpace, "FiniteElementSpace", subspace);
