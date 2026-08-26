//! Python wrappers for [`crate::containers::evolution::SubEvolution`] and
//! [`crate::containers::evolution::Evolution`].

use crate::containers::element_field::ElementField;
use crate::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue};
use crate::containers::node_field::NodeField;
use crate::py::element_field::{PyElementField, PySubElementField};
// Used by the viz-gated `plot` methods and — even without `viz` — by the
// `gen_stub_pymethods` macro, which reads those methods' signatures to emit
// the `.pyi` stub. Hence the import must also be present under `stub-gen`.
use crate::handle::Handle;
#[cfg(any(feature = "viz", feature = "stub-gen"))]
use crate::py::mesh::PyMesh;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Extract one tabulated value (`float` / `SubNodeField` / `SubElementField`)
/// from a Python object.
fn extract_sub_value(value: &Bound<'_, PyAny>) -> PyResult<SubValue> {
    if let Ok(f) = value.extract::<f64>() {
        return Ok(SubValue::Scalar(f));
    }
    if let Ok(s) = value.extract::<PyRef<PySubNodeField>>() {
        return Ok(SubValue::Node((*s.handle.read()).clone()));
    }
    if let Ok(s) = value.extract::<PyRef<PySubElementField>>() {
        return Ok(SubValue::Element((*s.handle.read()).clone()));
    }
    Err(PyTypeError::new_err(
        "evolution value must be a float, a SubNodeField or a SubElementField",
    ))
}

/// Wrap an interpolated [`SubValue`] as the matching Python object.
fn sub_value_to_py(py: Python<'_>, value: SubValue) -> PyResult<Py<PyAny>> {
    match value {
        SubValue::Scalar(s) => Ok(PyFloat::new(py, s).into_any().unbind()),
        SubValue::Node(f) => Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(f),
            },
        )?
        .into_any()),
        SubValue::Element(f) => Ok(Py::new(
            py,
            PySubElementField {
                handle: Handle::new(f),
            },
        )?
        .into_any()),
    }
}

// ─── SubEvolution (view + builder) ──────────────────────────────────────────

/// One tabulated curve: a sorted list of abscissas with the matching values
/// (a float, a `SubNodeField` or a `SubElementField`), interpolated linearly.
///
/// Build with `SubEvolution(samples, out_of_range="error")` where `samples`
/// is a list of `(abscissa, value)` pairs; compose several curves into an
/// `Evolution` with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubEvolution")]
pub struct PySubEvolution {
    pub(crate) handle: Handle<SubEvolution>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubEvolution {
    /// `SubEvolution(samples, out_of_range="error", abscissa_type=None,
    /// ordinate_type=None)` — one curve. `samples` is a list of
    /// `(abscissa, value)`; the values must all be of the same kind (and, for
    /// fields, on the same support). `out_of_range` defaults to `"error"`.
    /// `abscissa_type` names the physical type of
    /// the abscissa (used to label plots and to select a field component when a
    /// field is interpolated); `ordinate_type` names the value's type (scalar
    /// curves only).
    #[new]
    #[pyo3(signature = (samples, out_of_range=None, abscissa_type=None, ordinate_type=None))]
    fn py_new(
        py: Python<'_>,
        samples: Vec<(f64, Py<PyAny>)>,
        out_of_range: Option<OutOfRange>,
        abscissa_type: Option<String>,
        ordinate_type: Option<String>,
    ) -> PyResult<Self> {
        let oor = out_of_range.unwrap_or(OutOfRange::Error);
        let mut pairs = Vec::with_capacity(samples.len());
        for (x, v) in samples {
            pairs.push((x, extract_sub_value(v.bind(py))?));
        }
        let mut sub = SubEvolution::new(pairs, oor)?;
        sub.set_abscissa_type(abscissa_type);
        sub.set_ordinate_type(ordinate_type)?;
        Ok(Self {
            handle: Handle::new(sub),
        })
    }

    /// The abscissa's physical type, or `None`.
    fn abscissa_type(&self) -> PyResult<Option<String>> {
        Ok(self.handle.read().abscissa_type().map(str::to_string))
    }

    /// The ordinate's physical type (scalar curves), or `None`.
    fn ordinate_type(&self) -> PyResult<Option<String>> {
        Ok(self.handle.read().ordinate_type().map(str::to_string))
    }

    /// Number of samples.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.handle.read().len())
    }

    /// The sorted abscissas.
    fn abscissas(&self) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().abscissas().to_vec())
    }

    /// The stored out-of-range policy name.
    fn out_of_range(&self) -> PyResult<String> {
        Ok(self.handle.read().out_of_range().name().to_string())
    }

    /// Interpolate at `x`.
    ///
    /// - `x` a **float** → a float / `SubNodeField` / `SubElementField` (the
    ///   curve's value at `x`);
    /// - `x` a **`SubNodeField`** or **`SubElementField`** → a field of the
    ///   same support whose every entry is this **scalar** curve evaluated at
    ///   the input's `abscissa_type` component (the curve as a transfer
    ///   function). The output component is named after `ordinate_type`.
    ///
    /// `out_of_range` (`"error"` / `"clamp"` / `"extrapolate"`) overrides the
    /// stored policy for this query.
    #[pyo3(signature = (x, out_of_range=None))]
    fn interpolate(
        &self,
        py: Python<'_>,
        x: Py<PyAny>,
        out_of_range: Option<OutOfRange>,
    ) -> PyResult<Py<PyAny>> {
        let policy = out_of_range;
        let x = x.bind(py);
        let sub = self.handle.read();
        if let Ok(v) = x.extract::<f64>() {
            return sub_value_to_py(py, sub.interpolate(v, policy)?);
        }
        if let Ok(f) = x.extract::<PyRef<PySubNodeField>>() {
            let field = (*f.handle.read()).clone();
            let out = sub.interpolate_field(&field, policy)?;
            return Ok(Py::new(
                py,
                PySubNodeField {
                    handle: Handle::new(out),
                },
            )?
            .into_any());
        }
        if let Ok(f) = x.extract::<PyRef<PySubElementField>>() {
            let field = (*f.handle.read()).clone();
            let out = sub.interpolate_field(&field, policy)?;
            return Ok(Py::new(
                py,
                PySubElementField {
                    handle: Handle::new(out),
                },
            )?
            .into_any());
        }
        Err(PyTypeError::new_err(
            "interpolate expects a float, a SubNodeField or a SubElementField",
        ))
    }

    /// Plot this single curve — see `Evolution.plot`. A scalar curve draws an
    /// X-Y line; a field curve renders with a frame slider.
    /// `revolve` / `revolve_angle` sweep an axisymmetric plot into its body
    /// of revolution — see `SubMesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, mesh=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, frame=None, revolve=false, revolve_angle=360.0, x_label=None, y_label=None, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        mesh: Option<PyRef<'_, PyMesh>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<crate::viz::Colormap>,
        smooth: usize,
        frame: Option<usize>,
        revolve: bool,
        revolve_angle: f64,
        x_label: Option<String>,
        y_label: Option<String>,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: cmap.unwrap_or_default(),
            vmin,
            vmax,
        };
        // Clone out so no read lock is held during rendering.
        let sub = (*self.handle.read()).clone();
        sub.plot(
            Some(view),
            save.as_deref(),
            mesh.as_ref().map(|m| &m.inner),
            component.as_deref(),
            scale,
            smooth,
            frame,
            x_label.as_deref(),
            y_label.as_deref(),
            title.as_deref(),
        )?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.handle.read()))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", *self.handle.read()))
    }
}

crate::impl_dump_pymethod!(handle PySubEvolution, handle);

// ─── Evolution (aggregate) ──────────────────────────────────────────────────

/// An aggregate of tabulated curves (one `SubEvolution` per zone), interpolated
/// linearly against a variable (often time).
///
/// Build high-level with `Evolution(steps, out_of_range="error")` where each
/// step is `(abscissa, value)` and `value` is a whole `NodeField`,
/// `ElementField` or float (whole fields are transposed into one curve per
/// zone); or low-level by composing `SubEvolution`s with `|`. Index it
/// (`evolution[i]`) to reach a `SubEvolution`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Evolution")]
pub struct PyEvolution {
    pub(crate) inner: Evolution,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyEvolution {
    /// `Evolution(steps, out_of_range="error")` — build from whole values
    /// (`out_of_range` defaults to `"error"`)
    /// tabulated at each abscissa. `steps` is a list of `(abscissa, value)`
    /// where `value` is a float, a `NodeField` or an `ElementField`; the kind
    /// is taken from the first step. Whole fields are transposed into one
    /// curve per zone.
    #[new]
    #[pyo3(signature = (steps, out_of_range=None, abscissa_type=None, ordinate_type=None))]
    fn py_new(
        py: Python<'_>,
        steps: Vec<(f64, Py<PyAny>)>,
        out_of_range: Option<OutOfRange>,
        abscissa_type: Option<String>,
        ordinate_type: Option<String>,
    ) -> PyResult<Self> {
        let oor = out_of_range.unwrap_or(OutOfRange::Error);
        if steps.is_empty() {
            return Err(PyValueError::new_err(
                "Evolution: at least one step is required",
            ));
        }
        let first = steps[0].1.bind(py);

        let mut inner = if first.extract::<f64>().is_ok() {
            let mut pairs = Vec::with_capacity(steps.len());
            for (x, v) in &steps {
                pairs.push((*x, v.bind(py).extract::<f64>()?));
            }
            Evolution::from_scalars(pairs, oor)?
        } else if first.extract::<PyRef<PyNodeField>>().is_ok() {
            let mut fields: Vec<PyRef<PyNodeField>> = Vec::with_capacity(steps.len());
            for (_, v) in &steps {
                fields.push(v.bind(py).extract::<PyRef<PyNodeField>>()?);
            }
            let refs: Vec<(f64, &NodeField)> = steps
                .iter()
                .map(|(x, _)| *x)
                .zip(fields.iter().map(|f| &f.inner))
                .collect();
            Evolution::from_node_fields(&refs, oor)?
        } else if first.extract::<PyRef<PyElementField>>().is_ok() {
            let mut fields: Vec<PyRef<PyElementField>> = Vec::with_capacity(steps.len());
            for (_, v) in &steps {
                fields.push(v.bind(py).extract::<PyRef<PyElementField>>()?);
            }
            let refs: Vec<(f64, &ElementField)> = steps
                .iter()
                .map(|(x, _)| *x)
                .zip(fields.iter().map(|f| &f.inner))
                .collect();
            Evolution::from_element_fields(&refs, oor)?
        } else {
            return Err(PyTypeError::new_err(
                "Evolution step value must be a float, a NodeField or an ElementField",
            ));
        };
        inner.set_abscissa_type(abscissa_type)?;
        inner.set_ordinate_type(ordinate_type)?;
        Ok(Self { inner })
    }

    /// The stored out-of-range policy name.
    fn out_of_range(&self) -> PyResult<String> {
        Ok(self.inner.out_of_range().name().to_string())
    }

    /// The abscissa's physical type, or `None`.
    fn abscissa_type(&self) -> PyResult<Option<String>> {
        Ok(self.inner.abscissa_type()?)
    }

    /// The ordinate's physical type (scalar evolutions), or `None`.
    fn ordinate_type(&self) -> PyResult<Option<String>> {
        Ok(self.inner.ordinate_type()?)
    }

    /// Interpolate at `x`.
    ///
    /// - `x` a **float** → every curve interpolated and regrouped: a
    ///   `list[float]` for scalars, a `NodeField` / `ElementField` for fields;
    /// - `x` a **`NodeField`** or **`ElementField`** → that field mapped
    ///   through the (single, scalar) curve as a transfer function: each entry
    ///   of the input's `abscissa_type` component is looked up, yielding a field
    ///   of one component (named after `ordinate_type`) on the same support.
    ///
    /// `out_of_range` overrides the stored policy for this query.
    #[pyo3(signature = (x, out_of_range=None))]
    fn interpolate(
        &self,
        py: Python<'_>,
        x: Py<PyAny>,
        out_of_range: Option<OutOfRange>,
    ) -> PyResult<Py<PyAny>> {
        let policy = out_of_range;
        let x = x.bind(py);
        if let Ok(v) = x.extract::<f64>() {
            return match self.inner.interpolate(v, policy)? {
                Interpolated::Scalars(v) => Ok(PyList::new(py, v)?.into_any().unbind()),
                Interpolated::Node(f) => Ok(Py::new(py, PyNodeField { inner: f })?.into_any()),
                Interpolated::Element(f) => {
                    Ok(Py::new(py, PyElementField { inner: f })?.into_any())
                }
            };
        }
        if let Ok(f) = x.extract::<PyRef<PyNodeField>>() {
            let out = self.inner.interpolate_node_field(&f.inner, policy)?;
            return Ok(Py::new(py, PyNodeField { inner: out })?.into_any());
        }
        if let Ok(f) = x.extract::<PyRef<PyElementField>>() {
            let out = self.inner.interpolate_element_field(&f.inner, policy)?;
            return Ok(Py::new(py, PyElementField { inner: out })?.into_any());
        }
        Err(PyTypeError::new_err(
            "interpolate expects a float, a NodeField or an ElementField",
        ))
    }

    /// Plot the evolution. A **scalar** evolution draws an X-Y curve (one line
    /// per zone); a **field** evolution renders like `Mesh.plot(field=...)`
    /// with a frame slider (drag, or ← / →) picking the tabulated value.
    ///
    /// `save` writes a PNG/SVG (a single `frame`, default = last for fields);
    /// omit it for the interactive window. `mesh` supplies the surface for
    /// field evolutions (node frames default to a point cloud). `x_label` /
    /// `y_label` / `title` label the curve.
    /// `revolve` / `revolve_angle` sweep an axisymmetric plot into its body
    /// of revolution — see `SubMesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, mesh=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, frame=None, revolve=false, revolve_angle=360.0, x_label=None, y_label=None, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        mesh: Option<PyRef<'_, PyMesh>>,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<crate::viz::Colormap>,
        smooth: usize,
        frame: Option<usize>,
        revolve: bool,
        revolve_angle: f64,
        x_label: Option<String>,
        y_label: Option<String>,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: cmap.unwrap_or_default(),
            vmin,
            vmax,
        };
        self.inner.plot(
            Some(view),
            save.as_deref(),
            mesh.as_ref().map(|m| &m.inner),
            component.as_deref(),
            scale,
            smooth,
            frame,
            x_label.as_deref(),
            y_label.as_deref(),
            title.as_deref(),
        )?;
        Ok(())
    }
}

crate::impl_aggregate_pymethods!(
    PyEvolution,
    PySubEvolution,
    "Evolution",
    sub_evolution,
    Evolution,
    r#"
class PyEvolution:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubEvolution"]:
        """`evolution[i]` → the `SubEvolution` of zone i: one tabulated curve,
        interpolated linearly against the variable."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyEvolution"]:
        """`evolution[i:j:k]` → a fresh `Evolution` holding the sliced curves,
        shared with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyEvolution"] | pyo3_stub_gen.RustType["PySubEvolution"]) -> pyo3_stub_gen.RustType["PyEvolution"]:
        """`evolution | other` → a fresh `Evolution` holding the curves of both,
        in first-seen order and deduplicated by object identity."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubEvolution"]) -> pyo3_stub_gen.RustType["PyEvolution"]:
        """`sub_evolution | evolution` — the mirror of
        `evolution | sub_evolution`, differing only in that the lone curve
        comes first."""
    "#,
    r#"
class PySubEvolution:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubEvolution"]) -> pyo3_stub_gen.RustType["PyEvolution"]:
        """`sub_evolution | sub_evolution` → a fresh `Evolution` holding both
        curves — the low-level way to build a multi-zone evolution."""
    "#
);
