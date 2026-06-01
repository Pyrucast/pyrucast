//! Python wrappers for [`crate::containers::mesh::SubMesh`] and [`crate::containers::mesh::Mesh`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::py::configuration::PyConfiguration;
use crate::py::node::PyNode;
use crate::store::{with, with_mut, Handle};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Resolve a `Handle<SubMesh>` from either a `SubMesh` view or a
/// **unitary** `Mesh` (the parent→sub coercion — see `CONVENTIONS.md`,
/// « Agrégats : un ou plusieurs »). Used wherever an API takes a single
/// submesh support (`NodeField`, `Matrix.block`, …). A multi-submesh
/// `Mesh` is rejected with a clear error.
pub(crate) fn submesh_handle(obj: &Bound<'_, PyAny>) -> PyResult<Handle<SubMesh>> {
    if let Ok(sm) = obj.extract::<PyRef<PySubMesh>>() {
        Ok(sm.handle.clone())
    } else if let Ok(mesh) = obj.extract::<PyRef<PyMesh>>() {
        Ok(mesh.inner.unit()?)
    } else {
        Err(PyTypeError::new_err("expected a SubMesh or a unitary Mesh"))
    }
}

/// A **view** into one submesh of a `Mesh` — the cells of a single element
/// type. Obtained by indexing (`mesh[i]`); never constructed directly.
/// Build at the parent level instead: `Mesh(config, element_type)` for a
/// single zone, composed with `+` for several.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubMesh")]
pub struct PySubMesh {
    pub(crate) handle: Handle<SubMesh>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubMesh {
    #[getter]
    fn element_type(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| s.element_type().name().to_string())?)
    }

    fn add_cell(&self, nodes: Vec<PyRef<'_, PyNode>>) -> PyResult<usize> {
        let nodes_typed: Vec<NodeId> = nodes.iter().map(|n| n.as_node().id()).collect();
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
            s.set_face_color(crate::containers::mesh::RgbColor::new(rgb.0, rgb.1, rgb.2))
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
    /// - `vmin` / `vmax`: pin the bottom / top of the colour scale (and
    ///   of the colorbar drawn on the right). Either may be left `None`
    ///   to track the data's own min / max for that bound.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
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
        let scale = crate::viz::ColorScale { vmin, vmax };
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                let sm_handle = self.handle.clone();
                let field_handle = f.handle.clone();
                crate::store::with(&sm_handle, |s| {
                    crate::store::with(&field_handle, |fld| {
                        s.plot_with_field(Some(view), save_ref, fld, comp_ref, scale)
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
        let cell = crate::containers::mesh::Cell::new(self.handle.clone(), normalized as usize)?;
        Ok(crate::py::cell::PyCell::from_cell(cell))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{:?}", s))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{}", s))?)
    }
}

/// A geometric mesh: a collection of submeshes, each holding the cells of a
/// single element type.
///
/// Build with `Mesh(config, element_type)` for one zone, compose several
/// with `+`; index it (`mesh[i]`) to reach a `SubMesh`.
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
                Mesh::from_submesh(SubMesh::new(cfg, et))
            }
            None => Mesh::empty(),
        };
        Ok(Self { inner: mesh })
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
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
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
        let scale = crate::viz::ColorScale { vmin, vmax };
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                let field_handle = f.handle.clone();
                crate::store::with(&field_handle, |fld| {
                    self.inner.plot_with_field(Some(view), save_ref, fld, comp_ref, scale)
                })??;
            }
            None => {
                self.inner.plot(Some(view), save_ref)?;
            }
        }
        Ok(())
    }

}

crate::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh", submesh);
