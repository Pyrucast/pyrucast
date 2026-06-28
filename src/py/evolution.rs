//! Python wrappers for [`crate::containers::evolution::SubEvolution`] and
//! [`crate::containers::evolution::Evolution`].

use crate::containers::element_field::ElementField;
use crate::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue};
use crate::containers::node_field::NodeField;
use crate::py::element_field::{PyElementField, PySubElementField};
// Used by the viz-gated `plot` methods and — even without `viz` — by the
// `gen_stub_pymethods` macro, which reads those methods' signatures to emit
// the `.pyi` stub. Hence the import must also be present under `stub-gen`.
#[cfg(any(feature = "viz", feature = "stub-gen"))]
use crate::py::mesh::PyMesh;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use crate::store::{insert, read, Handle};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse an optional policy name into an optional [`OutOfRange`] override.
fn parse_policy(name: Option<&str>) -> PyResult<Option<OutOfRange>> {
    match name {
        Some(s) => Ok(Some(OutOfRange::from_name(s)?)),
        None => Ok(None),
    }
}

/// Extract one tabulated value (`float` / `SubNodeField` / `SubElementField`)
/// from a Python object.
fn extract_sub_value(value: &Bound<'_, PyAny>) -> PyResult<SubValue> {
    if let Ok(f) = value.extract::<f64>() {
        return Ok(SubValue::Scalar(f));
    }
    if let Ok(s) = value.extract::<PyRef<PySubNodeField>>() {
        return Ok(SubValue::Node((*read(&s.handle)?).clone()));
    }
    if let Ok(s) = value.extract::<PyRef<PySubElementField>>() {
        return Ok(SubValue::Element((*read(&s.handle)?).clone()));
    }
    Err(PyTypeError::new_err(
        "evolution value must be a float, a SubNodeField or a SubElementField",
    ))
}

/// Wrap an interpolated [`SubValue`] as the matching Python object.
fn sub_value_to_py(py: Python<'_>, value: SubValue) -> PyResult<Py<PyAny>> {
    match value {
        SubValue::Scalar(s) => Ok(PyFloat::new(py, s).into_any().unbind()),
        SubValue::Node(f) => {
            Ok(Py::new(py, PySubNodeField { handle: insert(f) })?.into_any())
        }
        SubValue::Element(f) => {
            Ok(Py::new(py, PySubElementField { handle: insert(f) })?.into_any())
        }
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
    /// `SubEvolution(samples, out_of_range="error")` — one curve. `samples`
    /// is a list of `(abscissa, value)`; the values must all be of the same
    /// kind (and, for fields, on the same support).
    #[new]
    #[pyo3(signature = (samples, out_of_range="error"))]
    fn py_new(py: Python<'_>, samples: Vec<(f64, Py<PyAny>)>, out_of_range: &str) -> PyResult<Self> {
        let oor = OutOfRange::from_name(out_of_range)?;
        let mut pairs = Vec::with_capacity(samples.len());
        for (x, v) in samples {
            pairs.push((x, extract_sub_value(v.bind(py))?));
        }
        Ok(Self {
            handle: insert(SubEvolution::new(pairs, oor)?),
        })
    }

    /// Number of samples.
    fn __len__(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.len())
    }

    /// The sorted abscissas.
    fn abscissas(&self) -> PyResult<Vec<f64>> {
        Ok(read(&self.handle)?.abscissas().to_vec())
    }

    /// The stored out-of-range policy name.
    fn out_of_range(&self) -> PyResult<String> {
        Ok(read(&self.handle)?.out_of_range().name().to_string())
    }

    /// Interpolate at `x`. Returns a float, a `SubNodeField` or a
    /// `SubElementField`. `out_of_range` (`"error"` / `"clamp"` /
    /// `"extrapolate"`) overrides the stored policy for this query.
    #[pyo3(signature = (x, out_of_range=None))]
    fn interpolate(
        &self,
        py: Python<'_>,
        x: f64,
        out_of_range: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let policy = parse_policy(out_of_range)?;
        let value = read(&self.handle)?.interpolate(x, policy)?;
        sub_value_to_py(py, value)
    }

    /// Plot this single curve — see `Evolution.plot`. A scalar curve draws an
    /// X-Y line; a field curve renders with a frame slider.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, mesh=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, frame=None, x_label=None, y_label=None, title=None))]
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
        cmap: Option<String>,
        smooth: usize,
        frame: Option<usize>,
        x_label: Option<String>,
        y_label: Option<String>,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = build_view(view, show_axes);
        let scale = crate::viz::ColorScale {
            cmap: crate::py::mesh::parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        // Clone out of the store so no read lock is held during rendering.
        let sub = (*read(&self.handle)?).clone();
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
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
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
    /// tabulated at each abscissa. `steps` is a list of `(abscissa, value)`
    /// where `value` is a float, a `NodeField` or an `ElementField`; the kind
    /// is taken from the first step. Whole fields are transposed into one
    /// curve per zone.
    #[new]
    #[pyo3(signature = (steps, out_of_range="error"))]
    fn py_new(py: Python<'_>, steps: Vec<(f64, Py<PyAny>)>, out_of_range: &str) -> PyResult<Self> {
        let oor = OutOfRange::from_name(out_of_range)?;
        if steps.is_empty() {
            return Err(PyValueError::new_err(
                "Evolution: at least one step is required",
            ));
        }
        let first = steps[0].1.bind(py);

        if first.extract::<f64>().is_ok() {
            let mut pairs = Vec::with_capacity(steps.len());
            for (x, v) in &steps {
                pairs.push((*x, v.bind(py).extract::<f64>()?));
            }
            return Ok(Self {
                inner: Evolution::from_scalars(pairs, oor)?,
            });
        }

        if first.extract::<PyRef<PyNodeField>>().is_ok() {
            let mut fields: Vec<PyRef<PyNodeField>> = Vec::with_capacity(steps.len());
            for (_, v) in &steps {
                fields.push(v.bind(py).extract::<PyRef<PyNodeField>>()?);
            }
            let refs: Vec<(f64, &NodeField)> = steps
                .iter()
                .map(|(x, _)| *x)
                .zip(fields.iter().map(|f| &f.inner))
                .collect();
            return Ok(Self {
                inner: Evolution::from_node_fields(&refs, oor)?,
            });
        }

        if first.extract::<PyRef<PyElementField>>().is_ok() {
            let mut fields: Vec<PyRef<PyElementField>> = Vec::with_capacity(steps.len());
            for (_, v) in &steps {
                fields.push(v.bind(py).extract::<PyRef<PyElementField>>()?);
            }
            let refs: Vec<(f64, &ElementField)> = steps
                .iter()
                .map(|(x, _)| *x)
                .zip(fields.iter().map(|f| &f.inner))
                .collect();
            return Ok(Self {
                inner: Evolution::from_element_fields(&refs, oor)?,
            });
        }

        Err(PyTypeError::new_err(
            "Evolution step value must be a float, a NodeField or an ElementField",
        ))
    }

    /// The stored out-of-range policy name.
    fn out_of_range(&self) -> PyResult<String> {
        Ok(self.inner.out_of_range().name().to_string())
    }

    /// Interpolate every curve at `x` and regroup the results: a `list[float]`
    /// for scalars, a `NodeField` or `ElementField` for fields. `out_of_range`
    /// overrides the stored policy for this query.
    #[pyo3(signature = (x, out_of_range=None))]
    fn interpolate(
        &self,
        py: Python<'_>,
        x: f64,
        out_of_range: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let policy = parse_policy(out_of_range)?;
        match self.inner.interpolate(x, policy)? {
            Interpolated::Scalars(v) => Ok(PyList::new(py, v)?.into_any().unbind()),
            Interpolated::Node(f) => Ok(Py::new(py, PyNodeField { inner: f })?.into_any()),
            Interpolated::Element(f) => Ok(Py::new(py, PyElementField { inner: f })?.into_any()),
        }
    }

    /// Plot the evolution. A **scalar** evolution draws an X-Y curve (one line
    /// per zone); a **field** evolution renders like `Mesh.plot(field=...)`
    /// with a frame slider (drag, or ← / →) picking the tabulated value.
    ///
    /// `save` writes a PNG/SVG (a single `frame`, default = last for fields);
    /// omit it for the interactive window. `mesh` supplies the surface for
    /// field evolutions (node frames default to a point cloud). `x_label` /
    /// `y_label` / `title` label the curve.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, mesh=None, component=None, vmin=None, vmax=None, cmap=None, smooth=4, frame=None, x_label=None, y_label=None, title=None))]
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
        cmap: Option<String>,
        smooth: usize,
        frame: Option<usize>,
        x_label: Option<String>,
        y_label: Option<String>,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = build_view(view, show_axes);
        let scale = crate::viz::ColorScale {
            cmap: crate::py::mesh::parse_cmap(cmap)?,
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

/// Build a [`crate::viz::View`] from an optional `(yaw, pitch, scale)` triple.
#[cfg(feature = "viz")]
fn build_view(view: Option<(f64, f64, f64)>, show_axes: bool) -> crate::viz::View {
    let mut v = view
        .map(|(yaw, pitch, scale)| crate::viz::View {
            yaw,
            pitch,
            scale,
            target: None,
            show_axes,
        })
        .unwrap_or_default();
    v.show_axes = show_axes;
    v
}

crate::impl_aggregate_pymethods!(PyEvolution, PySubEvolution, "Evolution", sub_evolution, Evolution);
