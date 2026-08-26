//! PyO3 wrappers — Python bindings for every public container of pyrucast.
//!
//! Each Rust container (`mesh`, `finite_element_space`, `model`, ...) has its mirror
//! file here defining the `Py*` classes and `#[pymethods]` impls. The
//! aim is to keep the container modules focused on Rust data + algorithms
//! and concentrate the FFI surface in one place.

pub mod archive;
pub mod arrays;
pub mod cell;
pub mod coords;
pub mod element;
pub mod element_field;
pub mod evolution;
pub mod finite_element_space;
pub mod matrix;
pub mod mesh;
pub mod model;
pub mod node;
pub mod node_field;
pub mod ops;
pub mod signals;

/// Build the [`View`](crate::viz::View) every `plot()` shares from the
/// arguments they all take: the optional `(yaw, pitch, scale)` triple (`None`
/// ⇒ the iso default), the gizmo flag, and the axisymmetric revolution.
///
/// `revolve_angle` is only read when `revolve` is on; an angle outside
/// `]0, 360]` is a `ValueError`.
#[cfg(feature = "viz")]
pub(crate) fn build_view(
    view: Option<(f64, f64, f64)>,
    show_axes: bool,
    revolve: bool,
    revolve_angle: f64,
) -> pyo3::PyResult<crate::viz::View> {
    let mut v = view
        .map(|(yaw, pitch, scale)| crate::viz::View {
            yaw,
            pitch,
            scale,
            target: None,
            show_axes,
            revolve: None,
        })
        .unwrap_or_default();
    v.show_axes = show_axes;
    v.revolve = match revolve {
        true => Some(crate::viz::Revolve::new(revolve_angle)?),
        false => None,
    };
    Ok(v)
}

/// Read a [`Named`](crate::named::Named) value from its name, as a Python
/// `ValueError` on failure.
///
/// Almost every wrapper takes the Rust type directly — the `FromPyObject` impl
/// of [`named_enum!`](crate::named_enum) does the reading. This helper serves
/// the handful of parameters that must stay `str` because pyo3 carries a
/// **literal default** for them (`element_type="SEG2"`): a default is a Rust
/// expression of the parameter's type, and only a `str` default renders as
/// readable Python in the generated stub.
#[cfg(feature = "python-api")]
pub(crate) fn named<T: crate::named::Named>(name: &str) -> pyo3::PyResult<T> {
    T::parse(name).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
