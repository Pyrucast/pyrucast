//! Python wrappers for [`crate::containers::mesh::SubMesh`] and [`crate::containers::mesh::Mesh`].

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::py::coords::PyCoords;
use crate::py::node::PyNode;
use crate::store::{insert, read, write, Handle};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Borrow a Python `NodeField` or `ElementField` as a [`crate::viz::FieldArg`].
#[cfg(feature = "viz")]
fn with_field_arg<R>(
    obj: &Bound<'_, PyAny>,
    f: impl FnOnce(crate::viz::FieldArg<'_>) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(nf) = obj.extract::<PyRef<crate::py::node_field::PyNodeField>>() {
        f(crate::viz::FieldArg::Node(&nf.inner))
    } else if let Ok(ef) = obj.extract::<PyRef<crate::py::element_field::PyElementField>>() {
        f(crate::viz::FieldArg::Element(&ef.inner))
    } else {
        Err(PyTypeError::new_err(
            "field must be a NodeField or an ElementField",
        ))
    }
}

/// Parse an optional `cmap` name into a [`crate::viz::Colormap`].
/// `None` ⇒ the default (Viridis); an unknown name is a `ValueError`
/// that lists the accepted names.
#[cfg(feature = "viz")]
pub(crate) fn parse_cmap(name: Option<String>) -> PyResult<crate::viz::Colormap> {
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

/// Resolve the geometry rendering style from the `wireframe` flag,
/// rejecting it when a field is also given (a field always colours faces,
/// so a wireframe makes no sense).
#[cfg(feature = "viz")]
fn mesh_style(wireframe: bool, has_field: bool) -> PyResult<crate::viz::MeshStyle> {
    if wireframe && has_field {
        return Err(PyValueError::new_err(
            "wireframe=True has no meaning with a field: a field colours the \
             cell faces. Drop field, or set wireframe=False.",
        ));
    }
    Ok(if wireframe {
        crate::viz::MeshStyle::Wireframe
    } else {
        crate::viz::MeshStyle::Surface
    })
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
/// Build at the parent level instead: `Mesh(coords, element_type)` for a
/// single zone, composed with `|` for several.
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

    /// Whether this submesh is sealed: `True` once it is used by a
    /// finite-element space, field or matrix, after which `add_cell` fails.
    #[getter]
    fn is_sealed(&self) -> PyResult<bool> {
        Ok(read(&self.handle)?.is_sealed())
    }

    /// Deep-copy into a fresh, **unsealed** SubMesh with the same
    /// connectivity — the way to keep editing after this one has been sealed.
    fn duplicate(&self) -> PyResult<PySubMesh> {
        let copy = read(&self.handle)?.duplicate()?;
        Ok(PySubMesh {
            handle: insert(copy),
        })
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
        write(&self.handle)?.set_face_color(crate::atoms::RgbColor::new(rgb.0, rgb.1, rgb.2));
        Ok(())
    }

    /// Visualize this submesh.
    ///
    /// - `save=None`: interactive window (requires `viz-interactive`).
    /// - `save="<path>.png"`, `.svg` or `.svgz`: image file. `.svgz` is the
    ///   same SVG gzipped — around a tenth of the bytes on disk.
    /// - `view`: optional `(yaw, pitch, scale)` triple; default is iso.
    /// - `show_axes`: draw the X/Y/Z orientation gizmo in the bottom-left
    ///   corner (default `True`). In the interactive window, the key
    ///   `A` toggles it at runtime.
    /// - `field`: optional `NodeField` **or** `ElementField` whose values
    ///   colour each cell. Node field: per-cell nodal values read
    ///   directly. Element field: nodal values fitted per element from
    ///   the Gauss values (least squares local to the cell — the
    ///   discontinuities between elements stay visible; with a single
    ///   Gauss point the colour is constant per element).
    ///   Default `None` ⇒ uniform face colour.
    /// - `component`: component name to display when `field` is set
    ///   (defaults to the field's first component). In the
    ///   interactive window, click the top button or press `Tab` to
    ///   cycle through every component.
    /// - `vmin` / `vmax`: pin the bottom / top of the colour scale (and
    ///   of the colorbar drawn on the right). Either may be left `None`
    ///   to track the data's own min / max for that bound.
    /// - `cmap`: colour scale name — `"viridis"` (default), `"jet"`,
    ///   `"coolwarm"`, `"hot"` or `"gray"`.
    /// - `smooth`: subdivision level of the interpolated rendering
    ///   (default `4`): the colour follows the shape functions inside
    ///   each element. `0` = one flat colour per cell.
    /// - `wireframe`: when `True`, draw every element edge as a line
    ///   (interior edges of volume cells included) instead of the opaque
    ///   outer skin. Geometry only — combining it with `field` raises
    ///   `ValueError`, since a field always colours the faces.
    /// - `revolve`: on an **axisymmetric** geometry (`Coords.axisymmetric()`),
    ///   sweep the `(r, z)` meridian section into the body of revolution it
    ///   describes, instead of drawing the flat section (default `False`).
    ///   `ValueError` on a non-axisymmetric geometry. In the interactive
    ///   window, the top-left button or the `R` key toggles it at runtime.
    /// - `revolve_angle`: swept angle in degrees, in `]0, 360]` (default
    ///   `360`). A partial sweep cuts the body open and shows the meridian
    ///   section — and the field on it — at both ends.
    /// - `title`: optional figure name. It titles the interactive window
    ///   and is drawn centred as a caption at the bottom of a saved
    ///   PNG/SVG (default `None` ⇒ no caption, default window title).
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, wireframe=false, revolve=false, revolve_angle=360.0, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<Bound<'_, PyAny>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
        smooth: usize,
        wireframe: bool,
        revolve: bool,
        revolve_angle: f64,
        title: Option<String>,
    ) -> PyResult<()> {
        let style = mesh_style(wireframe, field.is_some())?;
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        let save_ref = save.as_deref();
        let title_ref = title.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                with_field_arg(&f, |arg| {
                    crate::viz::render_submesh_with_field(
                        &self.handle,
                        arg,
                        comp_ref,
                        scale,
                        smooth,
                        Some(view),
                        save_ref,
                        title_ref,
                    )?;
                    Ok(())
                })?;
            }
            None => {
                read(&self.handle)?.plot_styled(Some(view), save_ref, style, title_ref)?;
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
        let cell = crate::atoms::Cell::new(self.handle.clone(), normalized as usize)?;
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
/// Build with `Mesh(coords, element_type)` for one zone, compose several
/// with `|`; index it (`mesh[i]`) to reach a `SubMesh`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner: Mesh,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMesh {
    /// `Mesh(coords)` — empty mesh.
    /// `Mesh(coords, element_type)` — mesh with one pre-created submesh.
    #[new]
    #[pyo3(signature = (coords, element_type=None))]
    fn py_new(coords: PyRef<PyCoords>, element_type: Option<&str>) -> PyResult<Self> {
        let coords = coords.handle.clone();
        let mesh = match element_type {
            Some(et_str) => {
                let et = ElementType::from_name(et_str).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown element type: {et_str}"))
                })?;
                Mesh::from_submesh(SubMesh::new(coords, et))
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

    /// Deep-copy the whole mesh into fresh, **unsealed** submeshes with the
    /// same connectivity — editable again even if the source was sealed by
    /// consumers. Nodes are shared (same `Coords`).
    fn duplicate(&self) -> PyResult<PyMesh> {
        Ok(Self {
            inner: self.inner.duplicate()?,
        })
    }

    /// The `node_idx`-th node of cell `cell_idx` in submesh `submesh_idx`.
    fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> PyResult<PyNode> {
        let node = self.inner.node(submesh_idx, cell_idx, node_idx)?;
        Ok(PyNode::from_node(node))
    }

    /// A points (POI1) mesh with one point per node in `nodes`.
    ///
    /// The `Coords` comes from the nodes themselves — every `Node` carries its
    /// own — so no `Coords` argument is needed. A **named constructor**, hence
    /// a classmethod: it builds its own type, from atoms.
    #[classmethod]
    fn poi1_from_nodes(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        nodes: Vec<PyRef<PyNode>>,
    ) -> PyResult<PyMesh> {
        let ns: Vec<crate::atoms::Node> = nodes.iter().map(|n| n.as_node().clone()).collect();
        Ok(PyMesh {
            inner: crate::containers::mesh::Mesh::poi1_from_nodes(&ns)?,
        })
    }

    /// The mesh node closest (Euclidean distance) to `point`.
    ///
    /// `point` must have the mesh coordinate dimension. Only nodes referenced
    /// by a cell are considered; ties break to the smaller node id.
    fn nearest_node(&self, point: Vec<f64>) -> PyResult<PyNode> {
        let node = self.inner.nearest_node(&point)?;
        Ok(PyNode::from_node(node))
    }

    /// `mesh.cell(submesh_idx, cell_idx)` → `Cell` view; same thing
    /// as `mesh[submesh_idx][cell_idx]`.
    fn cell(&self, submesh_idx: usize, cell_idx: usize) -> PyResult<crate::py::cell::PyCell> {
        let cell = self.inner.cell(submesh_idx, cell_idx)?;
        Ok(crate::py::cell::PyCell::from_cell(cell))
    }

    /// Total number of cells across all submeshes.
    fn cell_count(&self) -> PyResult<usize> {
        Ok(self.inner.cell_count()?)
    }

    /// The `Coords` this mesh hangs off (all submeshes share it).
    ///
    /// A safety net to get the handle back when it has been dropped on the
    /// Python side — e.g. after `read_gmsh(coords, …)` if `coords` went out
    /// of scope. Raises if the mesh has no submesh yet (no `Coords` to take).
    fn coords(&self) -> PyResult<PyCoords> {
        Ok(PyCoords {
            handle: self.inner.coords()?,
        })
    }

    /// Visualize this mesh (every submesh in its own colour, or
    /// coloured by a `NodeField` / `ElementField` if `field` is
    /// supplied). See
    /// `SubMesh.plot` for the meaning of `view`, `save`, `show_axes`,
    /// `field`, `component`, `wireframe`, `revolve`, `revolve_angle` and
    /// `title`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, field=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, wireframe=false, revolve=false, revolve_angle=360.0, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        field: Option<Bound<'_, PyAny>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
        smooth: usize,
        wireframe: bool,
        revolve: bool,
        revolve_angle: f64,
        title: Option<String>,
    ) -> PyResult<()> {
        let style = mesh_style(wireframe, field.is_some())?;
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        let save_ref = save.as_deref();
        let title_ref = title.as_deref();
        match field {
            Some(f) => {
                let comp_ref = component.as_deref();
                with_field_arg(&f, |arg| {
                    self.inner.plot_with_field(
                        Some(view),
                        save_ref,
                        arg,
                        comp_ref,
                        scale,
                        smooth,
                        title_ref,
                    )?;
                    Ok(())
                })?;
            }
            None => {
                self.inner
                    .plot_styled(Some(view), save_ref, style, title_ref)?;
            }
        }
        Ok(())
    }
}

crate::impl_aggregate_pymethods!(
    PyMesh,
    PySubMesh,
    "Mesh",
    submesh,
    Mesh,
    r#"
class PyMesh:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubMesh"]:
        """`mesh[i]` → the `SubMesh` view of zone i, i.e. the cells of one
        element type (negative indices supported). Iterating a `Mesh` yields
        the same views, one per zone."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`mesh[i:j:k]` → a fresh `Mesh` over the sliced zones, sharing their
        cells with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyMesh"] | pyo3_stub_gen.RustType["PySubMesh"]) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`mesh | other` → a fresh `Mesh` holding the zones of both, in
        first-seen order. A zone already present (same store slot) is not added
        twice; the others are shared, not copied. All zones must hang off the
        same `Coords`."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubMesh"]) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`submesh | mesh` — the mirror of `mesh | submesh`, differing only in
        that the lone zone comes first."""
    "#,
    r#"
class PySubMesh:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubMesh"]) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`submesh | submesh` → a fresh `Mesh` holding both zones. The usual
        way to build a multi-element-type mesh from single-type ones."""
    "#
);
crate::impl_dump_pymethod!(handle PySubMesh, handle);
