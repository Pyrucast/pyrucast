//! Python wrappers for [`crate::mesh::SubMesh`] and [`crate::mesh::Mesh`].

use crate::mesh::configuration::NodeId;
use crate::mesh::element_type::ElementType;
use crate::mesh::{Mesh, SubMesh};
use crate::py::configuration::PyConfiguration;
use crate::py::node::PyNode;
use crate::store::{insert, with, with_mut, Handle};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// Python wrapper for [`SubMesh`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubMesh")]
pub struct PySubMesh {
    pub(crate) handle: Handle<SubMesh>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubMesh {
    #[new]
    fn py_new(config: PyRef<PyConfiguration>, element_type: &str) -> PyResult<Self> {
        let et = ElementType::from_name(element_type).ok_or_else(|| {
            PyValueError::new_err(format!("unknown element type: {element_type}"))
        })?;
        let cfg_handle = config.handle.clone();
        let sm = SubMesh::new(cfg_handle, et);
        Ok(Self { handle: insert(sm) })
    }

    #[getter]
    fn element_type(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| s.element_type().name().to_string())?)
    }

    fn add_cell(&self, nodes: Vec<u32>) -> PyResult<usize> {
        let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
        let idx = with_mut(&self.handle, move |s| s.add_cell(&nodes_typed))??;
        Ok(idx)
    }

    fn cell_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.cell_count())?)
    }

    /// Face colour as an `(r, g, b)` tuple of bytes.
    #[getter]
    fn face_color(&self) -> PyResult<(u8, u8, u8)> {
        let c = with(&self.handle, |s| s.face_color())?;
        Ok((c.r, c.g, c.b))
    }

    /// Set the face colour from an `(r, g, b)` tuple of bytes.
    #[setter]
    fn set_face_color(&self, rgb: (u8, u8, u8)) -> PyResult<()> {
        with_mut(&self.handle, |s| {
            s.set_face_color(crate::mesh::color::RgbColor::new(rgb.0, rgb.1, rgb.2))
        })?;
        Ok(())
    }

    /// Visualize this submesh.
    ///
    /// - `save=None`: interactive window (requires `viz-interactive`).
    /// - `save="<path>.png"` or `.svg`: image file.
    /// - `view`: optional `(yaw, pitch, scale)` triple; default is iso.
    /// - `show_axes`: draw the X/Y/Z orientation gizmo in the bottom-left
    ///   corner (default `True`). In the interactive window, the key
    ///   `A` toggles it at runtime.
    /// - `field`: optional `NodeField` whose values colour each cell
    ///   (per-cell value = mean over the cell's nodes of the chosen
    ///   component). Default `None` ⇒ uniform face colour.
    /// - `component`: component name to display when `field` is set
    ///   (defaults to the field's first component). In the
    ///   interactive window, click the top button or press `Tab` to
    ///   cycle through every component.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
    ) -> PyResult<()> {
        let mut view = view
            .map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
                show_axes,
            })
            .unwrap_or_else(crate::viz::View::default);
        view.show_axes = show_axes;
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                let sm_handle = self.handle.clone();
                let field_handle = f.handle.clone();
                crate::store::with(&sm_handle, |s| {
                    crate::store::with(&field_handle, |fld| {
                        s.plot_with_field(Some(view), save_ref, fld, comp_ref)
                    })?
                })??;
            }
            None => {
                with(&self.handle, |s| s.plot(Some(view), save_ref))??;
            }
        }
        Ok(())
    }

    /// `len(submesh)` → number of cells.
    fn __len__(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |s| s.cell_count())?)
    }

    /// `submesh[i]` → `Cell` view on cell i. Supports negative
    /// indices and raises `IndexError` out of range so
    /// `for cell in submesh:` works.
    fn __getitem__(&self, idx: isize) -> PyResult<crate::py::cell::PyCell> {
        let n = with(&self.handle, |s| s.cell_count())? as isize;
        let normalized = if idx < 0 { n + idx } else { idx };
        if normalized < 0 || normalized >= n {
            return Err(PyIndexError::new_err(format!(
                "submesh index {idx} out of range (len={n})"
            )));
        }
        let cell = crate::mesh::cell::Cell::new(self.handle.clone(), normalized as usize)?;
        Ok(crate::py::cell::PyCell::from_cell(cell))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{:?}", s))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{}", s))?)
    }
}

/// Python wrapper for [`Mesh`].
///
/// Owns the `Mesh` struct directly — `Mesh` is no longer kept in the
/// global store. Identity is the Python object identity itself
/// (two Python references to the same `PyMesh` see the same submeshes).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner: Mesh,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMesh {
    /// `Mesh(config)` — empty mesh.
    /// `Mesh(config, element_type)` — mesh with one pre-created submesh.
    #[new]
    #[pyo3(signature = (config, element_type=None))]
    fn py_new(config: PyRef<PyConfiguration>, element_type: Option<&str>) -> PyResult<Self> {
        let cfg = config.handle.clone();
        let mesh = match element_type {
            Some(et_str) => {
                let et = ElementType::from_name(et_str).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown element type: {et_str}"))
                })?;
                Mesh::with_element_type(cfg, et)
            }
            None => Mesh::empty(),
        };
        Ok(Self { inner: mesh })
    }

    fn add_submesh(&mut self, sm: PyRef<PySubMesh>) -> PyResult<()> {
        let sm_handle = sm.handle.clone();
        self.inner.add_submesh(sm_handle)?;
        Ok(())
    }

    fn add_cell(&mut self, nodes: Vec<u32>) -> PyResult<usize> {
        let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
        Ok(self.inner.add_cell(&nodes_typed)?)
    }

    #[getter]
    fn element_type(&self) -> PyResult<Option<String>> {
        if self.inner.submesh_count() == 1 {
            let h = self.inner.submesh(0)?;
            Ok(Some(with(&h, |sm| sm.element_type().name().to_string())?))
        } else {
            Ok(None)
        }
    }

    fn element_types(&self) -> PyResult<Vec<String>> {
        let types = self.inner.element_types()?;
        Ok(types.into_iter().map(|et| et.name().to_string()).collect())
    }

    fn cell_counts(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.cell_counts()?)
    }

    fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> PyResult<PyNode> {
        let node = self.inner.node(submesh_idx, cell_idx, node_idx)?;
        Ok(PyNode::from_node(node))
    }

    #[classmethod]
    fn from_live_nodes(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        config: PyRef<PyConfiguration>,
    ) -> PyResult<Self> {
        let mesh = Mesh::from_live_nodes(config.handle.clone())?;
        Ok(Self { inner: mesh })
    }

    #[classmethod]
    fn line_seg2(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        a: PyRef<PyNode>,
        b: PyRef<PyNode>,
        n_elems: usize,
    ) -> PyResult<Self> {
        let mesh = Mesh::line_seg2(a.as_node(), b.as_node(), n_elems)?;
        Ok(Self { inner: mesh })
    }

    #[classmethod]
    fn circle_seg2(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        center: PyRef<PyNode>,
        normal: Vec<f64>,
        radius: f64,
        n_elems: usize,
    ) -> PyResult<Self> {
        let mesh = Mesh::circle_seg2(center.as_node(), &normal, radius, n_elems)?;
        Ok(Self { inner: mesh })
    }

    #[classmethod]
    fn sweep_qua4(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh_a: PyRef<PyMesh>,
        mesh_b: PyRef<PyMesh>,
        n_layers: usize,
    ) -> PyResult<Self> {
        let mesh = Mesh::sweep_qua4(&mesh_a.inner, &mesh_b.inner, n_layers)?;
        Ok(Self { inner: mesh })
    }

    #[classmethod]
    fn extrude(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh: PyRef<PyMesh>,
        direction: Vec<f64>,
        n_layers: usize,
    ) -> PyResult<Self> {
        let result = Mesh::extrude(&mesh.inner, &direction, n_layers)?;
        Ok(Self { inner: result })
    }

    #[classmethod]
    #[pyo3(signature = (contour, element_type, max_edge_length=None, min_angle_deg=None))]
    fn fill_surface(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        contour: PyRef<PyMesh>,
        element_type: &str,
        max_edge_length: Option<f64>,
        min_angle_deg: Option<f64>,
    ) -> PyResult<Self> {
        let et = ElementType::from_name(element_type).ok_or_else(|| {
            PyValueError::new_err(format!("unknown element type: {element_type}"))
        })?;
        let refinement = if max_edge_length.is_some() || min_angle_deg.is_some() {
            Some(crate::ops::mesher::triangulation::RefinementOptions {
                max_edge_length,
                min_angle_deg,
            })
        } else {
            None
        };
        let mesh = Mesh::fill_surface(&contour.inner, et, refinement)?;
        Ok(Self { inner: mesh })
    }

    fn __add__(&self, other: PyRef<PyMesh>) -> PyResult<PyMesh> {
        let mesh = self.inner.merge(&other.inner);
        Ok(PyMesh { inner: mesh })
    }

    /// Merge submeshes of the same type and drop duplicate cells.
    /// Returns a new mesh with one submesh per element type.
    fn consolidate(&self) -> PyResult<PyMesh> {
        let mesh = self.inner.consolidate()?;
        Ok(PyMesh { inner: mesh })
    }

    fn submesh_count(&self) -> PyResult<usize> {
        Ok(self.inner.submesh_count())
    }

    /// Return the submesh at index `idx` as a `SubMesh` wrapper.
    /// The returned object shares storage with the parent mesh, so
    /// mutating it (e.g. setting `face_color`) is visible through
    /// the mesh too.
    fn submesh(&self, idx: usize) -> PyResult<PySubMesh> {
        let h = self.inner.submesh(idx)?;
        Ok(PySubMesh { handle: h })
    }

    /// `mesh.cell(submesh_idx, cell_idx)` → `Cell` view; same thing
    /// as `mesh[submesh_idx][cell_idx]`.
    fn cell(
        &self,
        submesh_idx: usize,
        cell_idx: usize,
    ) -> PyResult<crate::py::cell::PyCell> {
        let cell = self.inner.cell(submesh_idx, cell_idx)?;
        Ok(crate::py::cell::PyCell::from_cell(cell))
    }

    fn cell_count(&self) -> PyResult<usize> {
        Ok(self.inner.cell_count()?)
    }

    /// Visualize this mesh (every submesh in its own colour, or
    /// coloured by a `NodeField` if `field` is supplied). See
    /// `SubMesh.plot` for the meaning of `view`, `save`, `show_axes`,
    /// `field` and `component`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
    ) -> PyResult<()> {
        let mut view = view
            .map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
                show_axes,
            })
            .unwrap_or_else(crate::viz::View::default);
        view.show_axes = show_axes;
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                let field_handle = f.handle.clone();
                crate::store::with(&field_handle, |fld| {
                    self.inner.plot_with_field(Some(view), save_ref, fld, comp_ref)
                })??;
            }
            None => {
                self.inner.plot(Some(view), save_ref)?;
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

crate::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh");
