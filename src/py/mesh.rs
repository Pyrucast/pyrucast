//! Python wrappers for [`crate::containers::mesh::SubMesh`] and [`crate::containers::mesh::Mesh`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::py::configuration::PyConfiguration;
use crate::py::node::PyNode;
use crate::store::{read, write, Handle};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Parse an optional `cmap` name into a [`crate::viz::Colormap`].
/// `None` ⇒ the default (Viridis); an unknown name is a `ValueError`
/// that lists the accepted names.
#[cfg(feature = "viz")]
fn parse_cmap(name: Option<String>) -> PyResult<crate::viz::Colormap> {
    match name {
        None => Ok(crate::viz::Colormap::default()),
        Some(n) => crate::viz::Colormap::from_name(&n).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown colormap {n:?} (available: {:?})",
                crate::viz::Colormap::names()
            ))
        }),
    }
}

/// Resolve a `Handle<SubMesh>` from either a `SubMesh` view or a
/// **unitary** `Mesh` (the parent→sub coercion — see `CONVENTIONS.md`,
/// « Agrégats : un ou plusieurs »). Used wherever an API takes a single
/// submesh support (`Matrix.block`, …). A multi-submesh
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
    /// Element type name of this submesh (e.g. `"TRI3"`).
    #[getter]
    fn element_type(&self) -> PyResult<String> {
        Ok(read(&self.handle)?.element_type().name().to_string())
    }

    /// Append a cell from its list of nodes; returns the new cell's index.
    fn add_cell(&self, nodes: Vec<PyRef<'_, PyNode>>) -> PyResult<usize> {
        let nodes_typed: Vec<NodeId> = nodes.iter().map(|n| n.as_node().id()).collect();
        let idx = write(&self.handle)?.add_cell(&nodes_typed)?;
        Ok(idx)
    }

    /// Number of cells in this submesh.
    fn cell_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.cell_count())
    }

    /// Face colour as an `(r, g, b)` tuple of bytes.
    #[getter]
    fn face_color(&self) -> PyResult<(u8, u8, u8)> {
        let c = read(&self.handle)?.face_color();
        Ok((c.r, c.g, c.b))
    }

    /// Set the face colour from an `(r, g, b)` tuple of bytes.
    #[setter]
    fn set_face_color(&self, rgb: (u8, u8, u8)) -> PyResult<()> {
        write(&self.handle)?
            .set_face_color(crate::containers::mesh::RgbColor::new(rgb.0, rgb.1, rgb.2));
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
    /// - `cmap`: colour scale name — `"viridis"` (default), `"jet"`,
    ///   `"coolwarm"`, `"hot"` or `"gray"`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None, cmap=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
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
        let scale = crate::viz::ColorScale {
            cmap: parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                let sm_handle = self.handle.clone();
                crate::store::read(&sm_handle)?
                    .plot_with_field(Some(view), save_ref, &f.inner, comp_ref, scale)?;
            }
            None => {
                read(&self.handle)?.plot(Some(view), save_ref)?;
            }
        }
        Ok(())
    }

    /// `len(submesh)` → number of cells.
    fn __len__(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.cell_count())
    }

    /// `submesh[i]` → `Cell` view on cell i. Supports negative
    /// indices and raises `IndexError` out of range so
    /// `for cell in submesh:` works.
    fn __getitem__(&self, idx: isize) -> PyResult<crate::py::cell::PyCell> {
        let n = read(&self.handle)?.cell_count() as isize;
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
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
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

    /// Element type name of each submesh, in order.
    fn element_types(&self) -> PyResult<Vec<String>> {
        let types = self.inner.element_types()?;
        Ok(types.into_iter().map(|et| et.name().to_string()).collect())
    }

    /// Number of cells in each submesh, in order.
    fn cell_counts(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.cell_counts()?)
    }

    /// The `node_idx`-th node of cell `cell_idx` in submesh `submesh_idx`.
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

    /// Total number of cells across all submeshes.
    fn cell_count(&self) -> PyResult<usize> {
        Ok(self.inner.cell_count()?)
    }

    /// Visualize this mesh (every submesh in its own colour, or
    /// coloured by a `NodeField` if `field` is supplied). See
    /// `SubMesh.plot` for the meaning of `view`, `save`, `show_axes`,
    /// `field` and `component`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None, cmap=None))]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<PyRef<crate::py::node_field::PyNodeField>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
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
        let scale = crate::viz::ColorScale {
            cmap: parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        let save_ref = save.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                self.inner
                    .plot_with_field(Some(view), save_ref, &f.inner, comp_ref, scale)?;
            }
            None => {
                self.inner.plot(Some(view), save_ref)?;
            }
        }
        Ok(())
    }

}

crate::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh", submesh);
crate::impl_aggregate_sub_add!(PySubMesh, PyMesh);
crate::impl_dump_pymethod!(handle PySubMesh, handle);
