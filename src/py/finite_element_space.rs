//! Python wrappers for [`crate::containers::finite_element_space::SubFiniteElementSpace`] and
//! [`crate::containers::finite_element_space::FiniteElementSpace`].

use crate::atoms::Element;
use crate::atoms::Interpolation;
use crate::atoms::QuadratureRule;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::handle::Handle;
use crate::py::element::PyElement;
use crate::py::mesh::{PyMesh, PySubMesh};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// One subspace of a `FiniteElementSpace`: the finite elements over a
/// single submesh, with their interpolation and quadrature.
///
/// A read-only view obtained by indexing the parent (`fes[i]`); index it
/// further (`fes[i][j]`) to reach an `Element`. Not constructed directly.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubFiniteElementSpace")]
pub struct PySubFiniteElementSpace {
    pub(crate) handle: Handle<SubFiniteElementSpace>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubFiniteElementSpace {
    /// Element type name (e.g. `"TRI3"`).
    #[getter]
    fn element_type(&self) -> String {
        self.handle.read().element_type().name().to_string()
    }

    /// Interpolation name (e.g. `"LAGRANGE1"`).
    #[getter]
    fn interpolation(&self) -> String {
        self.handle.read().interpolation().name().to_string()
    }

    /// Quadrature-rule name (e.g. `"GAUSS"`).
    #[getter]
    fn quadrature(&self) -> String {
        self.handle.read().quadrature().name().to_string()
    }

    /// The submesh this subspace is built on — shared, not copied, and sealed
    /// for as long as the subspace holds it. `fes[i].submesh()` is
    /// `fes.mesh()[i]` without rebuilding the mesh.
    fn submesh(&self) -> PySubMesh {
        PySubMesh {
            handle: self.handle.read().submesh(),
        }
    }

    /// Reference (parametric) dimension of the elements.
    #[getter]
    fn ref_dim(&self) -> PyResult<usize> {
        Ok(self.handle.read().ref_dim()?)
    }

    /// Spatial dimension the elements live in.
    #[getter]
    fn space_dim(&self) -> usize {
        self.handle.read().space_dim()
    }

    /// Whether the underlying geometry is a body of revolution — inherited from
    /// the `Coords`, so a body and its boundary can never disagree.
    #[getter]
    fn is_axisymmetric(&self) -> bool {
        self.handle.read().is_axisymmetric()
    }

    /// Number of nodes per element.
    #[getter]
    fn nodes_per_cell(&self) -> PyResult<usize> {
        Ok(self.handle.read().nodes_per_cell()?)
    }

    /// Number of elements (cells) in this subspace.
    #[getter]
    fn cell_count(&self) -> usize {
        self.handle.read().cell_count()
    }

    /// Number of Gauss (quadrature) points per element.
    #[getter]
    fn gauss_count(&self) -> usize {
        self.handle.read().gauss_count()
    }

    /// Reference coordinates of the `g`-th Gauss point.
    fn gauss_xi(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().gauss_xi(g)?.to_vec())
    }

    /// Weight of the `g`-th Gauss point.
    fn gauss_weight(&self, g: usize) -> PyResult<f64> {
        Ok(self.handle.read().gauss_weight(g)?)
    }

    /// `N_i(ξ_g)` at the `g`-th Gauss point (flat, length `nodes_per_cell`).
    fn n_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().n_at_g(g)?.to_vec())
    }

    /// `∂N_i/∂ξ_j(ξ_g)` at the `g`-th Gauss point.
    ///
    /// Flat row-major: index `[i * ref_dim + j]`.
    fn dn_at_g(&self, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().dn_at_g(g)?.to_vec())
    }

    /// Jacobian `J = ∂x/∂ξ` of cell `cell_idx` at Gauss point `g`.
    ///
    /// Flat row-major buffer of length `space_dim × ref_dim`,
    /// indexed `[a * ref_dim + k]`.
    fn jacobian(&self, cell_idx: usize, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().jacobian(cell_idx, g)?)
    }

    /// `|J|` — `|det(J)|` if `space_dim == ref_dim`, else
    /// `sqrt(det(JᵀJ))`. Always non-negative.
    fn det_jacobian(&self, cell_idx: usize, g: usize) -> PyResult<f64> {
        Ok(self.handle.read().det_jacobian(cell_idx, g)?)
    }

    /// Physical derivatives `∂N_i/∂x_a` of cell `cell_idx` at Gauss
    /// point `g`. Flat row-major buffer of length
    /// `nodes_per_cell × space_dim`, indexed `[i * space_dim + a]`.
    fn dn_dx(&self, cell_idx: usize, g: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().dn_dx(cell_idx, g)?)
    }

    /// Element view on the `i`-th cell of this subspace.
    fn element(&self, i: usize) -> PyResult<PyElement> {
        let el = Element::new(self.handle.clone(), i)?;
        Ok(PyElement { inner: el })
    }

    /// All elements as a Python list.
    fn elements(&self) -> PyResult<Vec<PyElement>> {
        let n = self.handle.read().cell_count();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(PyElement {
                inner: Element::new(self.handle.clone(), i)?,
            });
        }
        Ok(out)
    }

    /// `len(subspace)` → number of elements (= number of cells).
    fn __len__(&self) -> usize {
        self.handle.read().cell_count()
    }

    /// `subspace[i]` → `Element` view on element `i`. Supports negative
    /// indices and raises `IndexError` out of range so
    /// `for el in subspace:` works.
    fn __getitem__(&self, idx: isize) -> PyResult<PyElement> {
        let n = self.handle.read().cell_count() as isize;
        let normalized = if idx < 0 { n + idx } else { idx };
        if normalized < 0 || normalized >= n {
            return Err(PyIndexError::new_err(format!(
                "subspace element index {idx} out of range (len={n})"
            )));
        }
        let el = Element::new(self.handle.clone(), normalized as usize)?;
        Ok(PyElement { inner: el })
    }
}

/// The finite-element discretization of a mesh: one subspace per submesh,
/// each carrying an interpolation and a quadrature rule.
///
/// Build from a `Mesh` with `FiniteElementSpace(mesh)` (or
/// `.with_choices(...)` per submesh); index it (`fes[i]`) to reach a
/// `SubFiniteElementSpace`, compose several with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "FiniteElementSpace")]
pub struct PyFiniteElementSpace {
    pub(crate) inner: FiniteElementSpace,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyFiniteElementSpace {
    /// `FiniteElementSpace(mesh)` — on every submesh, the Lagrange
    /// interpolation of its own degree (`LAGRANGE1` on a HEX8, `LAGRANGE2`
    /// on a HEX20 or a QUA8) and the default Gauss quadrature.
    ///
    /// `FiniteElementSpace(mesh, interpolation="LAGRANGE2", quadrature="GAUSS")`
    /// — an interpolation given applies to every submesh; omitting it keeps
    /// each submesh's own degree. `"GAUSS"` is the default quadrature.
    #[new]
    #[pyo3(signature = (mesh, interpolation=None, quadrature=None))]
    fn py_new(
        mesh: PyRef<PyMesh>,
        interpolation: Option<Interpolation>,
        quadrature: Option<QuadratureRule>,
    ) -> PyResult<Self> {
        let fes = match (interpolation, quadrature) {
            (None, None) => FiniteElementSpace::lagrange(&mesh.inner)?,
            _ => {
                let quad = quadrature.unwrap_or(QuadratureRule::Gauss);
                let choices: Vec<(Interpolation, QuadratureRule)> = (&mesh.inner)
                    .into_iter()
                    .map(|sm| {
                        let own = sm.read().element_type().as_kind().degree();
                        (
                            interpolation.or(own).unwrap_or(Interpolation::Lagrange1),
                            quad,
                        )
                    })
                    .collect();
                FiniteElementSpace::with(&mesh.inner, &choices)?
            }
        };
        Ok(Self { inner: fes })
    }

    /// Explicit `(interpolation, quadrature)` per submesh.
    #[classmethod]
    fn with_choices(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh: PyRef<PyMesh>,
        choices: Vec<(Interpolation, QuadratureRule)>,
    ) -> PyResult<Self> {
        let fes = FiniteElementSpace::with(&mesh.inner, &choices)?;
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

    /// `fes.mesh()` — the `Mesh` this space spans: one submesh per subspace, in
    /// order, so `fes.mesh()[i]` is `fes[i].submesh()` (shared handles, sealed
    /// while captured). Two subspaces on one submesh list it twice. The reverse
    /// of building a `FiniteElementSpace(mesh)`.
    fn mesh(&self) -> PyResult<PyMesh> {
        Ok(PyMesh {
            inner: self.inner.mesh()?,
        })
    }
}

crate::impl_aggregate_pymethods!(
    PyFiniteElementSpace,
    PySubFiniteElementSpace,
    "FiniteElementSpace",
    subspace,
    FiniteElementSpace,
    r#"
class PyFiniteElementSpace:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubFiniteElementSpace"]:
        """`fes[i]` → the `SubFiniteElementSpace` of zone i: the elements over
        one submesh, with their interpolation and quadrature. Index it further
        (`fes[i][j]`) to reach an `Element`."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyFiniteElementSpace"]:
        """`fes[i:j:k]` → a fresh `FiniteElementSpace` over the sliced
        subspaces, shared with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyFiniteElementSpace"] | pyo3_stub_gen.RustType["PySubFiniteElementSpace"]) -> pyo3_stub_gen.RustType["PyFiniteElementSpace"]:
        """`fes | other` → a fresh `FiniteElementSpace` spanning the subspaces
        of both, in first-seen order and deduplicated by object identity."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubFiniteElementSpace"]) -> pyo3_stub_gen.RustType["PyFiniteElementSpace"]:
        """`subspace | fes` — the mirror of `fes | subspace`, differing only in
        that the lone subspace comes first."""
    "#,
    r#"
class PySubFiniteElementSpace:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubFiniteElementSpace"]) -> pyo3_stub_gen.RustType["PyFiniteElementSpace"]:
        """`subspace | subspace` → a fresh `FiniteElementSpace` spanning both."""
    "#
);
crate::impl_dump_pymethod!(handle PySubFiniteElementSpace, handle);
