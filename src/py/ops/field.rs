//! Python wrappers for [`crate::ops::field`] — the operators polymorphic
//! over the field flavour, which give back a field of the caller's own kind.

use crate::handle::Handle;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::mesh::PyMesh;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Extract a component-name argument that is either a single `str` or a list of
/// `str` (e.g. the result of `model.primal_vars()`) into a `Vec<String>`.
pub(crate) fn extract_names(arg: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = arg.extract::<String>() {
        return Ok(vec![name]);
    }
    if let Ok(names) = arg.extract::<Vec<String>>() {
        return Ok(names);
    }
    Err(PyTypeError::new_err(
        "components: expected a str or a list of str",
    ))
}

/// Node-by-node (or point-by-point) scalar product of two fields — Cast3M's
/// `PSCA`. Returns a **new field** of the same flavour as the inputs, carrying
/// a single `"psca"` component whose value at each node/point is `∑_c xᵣ,c·yᵣ,c`
/// (reduction over components only, the support is kept).
///
/// `x` and `y` must be the same flavour (`NodeField` / `SubNodeField` /
/// `ElementField` / `SubElementField`), sit on the same support/decomposition,
/// and carry the same components (aligned by name).
///
/// For the **global** scalar product (a single float over the whole field),
/// see `pyrucast.measure.xty`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn psca(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be NodeFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyNodeField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be ElementFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyElementField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubNodeFields"))?;
        let out = a.handle.read().pscal(&*b.handle.read())?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(out),
            },
        )?
        .into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubElementFields"))?;
        let out = a.handle.read().pscal(&*b.handle.read())?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: Handle::new(out),
            },
        )?
        .into_any());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

// ── Element-wise unary maths (numpy-style) ──────────────────────────────────
//
// `pyrucast.cos(field)`, `pyrucast.exp(field)`, … apply a scalar function to
// every value of a field, returning a **new** field of the same type. They
// accept any of the four field flavours (`NodeField` / `SubNodeField` /
// `ElementField` / `SubElementField`) and mirror `crate::ops::field::*`.
// Results are unguarded (numpy-like): `log` of ≤ 0 → `-inf`/`nan`, etc.

/// Emit the method form of an element-wise unary op for an **aggregate**
/// flavour. The receiver's type being known, the method short-circuits the
/// four-branch dispatch of the free function and returns a **precise** type
/// instead of `Any`. It carries the free function's own doc — `$doc` is
/// substituted before the attribute macros read the item, so pyo3 (hence
/// `__doc__`) and pyo3-stub-gen (hence the `.pyi`) both see a real literal.
macro_rules! py_field_unary_aggregate {
    ($T:ident, $name:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $name(&self) -> PyResult<$T> {
                Ok($T {
                    inner: crate::ops::field::$name(&self.inner)?,
                })
            }
        }
    };
}

/// Same, for a **sub-container** flavour: the value is read through the handle,
/// and the result gets a handle of its own.
macro_rules! py_field_unary_sub {
    ($T:ident, $name:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $name(&self) -> PyResult<$T> {
                let out = crate::ops::field::$name(&*self.handle.read())?;
                Ok($T {
                    handle: Handle::new(out),
                })
            }
        }
    };
}

/// Generate a `#[pyfunction] $name(field)` that dispatches over the four field
/// wrapper types and applies the matching `ops::field::$name`, **and** the four
/// methods that are its « sujet » face. The documentation is written once, at
/// the call site, and reaches the free function and the four methods alike.
macro_rules! py_field_unary {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            use crate::ops::field::$name as op;
            if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
                return Ok(Py::new(
                    py,
                    PyNodeField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
                return Ok(Py::new(
                    py,
                    PyElementField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
                let out = op(&*f.handle.read())?;
                return Ok(Py::new(
                    py,
                    PySubNodeField {
                        handle: Handle::new(out),
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
                let out = op(&*f.handle.read())?;
                return Ok(Py::new(
                    py,
                    PySubElementField {
                        handle: Handle::new(out),
                    },
                )?
                .into_any());
            }
            Err(PyTypeError::new_err(
                "expected a NodeField, SubNodeField, ElementField or SubElementField",
            ))
        }

        py_field_unary_aggregate!(PyNodeField, $name, $doc);
        py_field_unary_aggregate!(PyElementField, $name, $doc);
        py_field_unary_sub!(PySubNodeField, $name, $doc);
        py_field_unary_sub!(PySubElementField, $name, $doc);
    };
}

py_field_unary!(abs, "Element-wise absolute value of a field.");
py_field_unary!(
    sqrt,
    "Element-wise square root of a field (`nan` for negatives)."
);
py_field_unary!(exp, "Element-wise exponential `eˣ` of a field.");
py_field_unary!(
    log,
    "Element-wise natural logarithm of a field (`-inf`/`nan` for ≤ 0)."
);
py_field_unary!(log10, "Element-wise base-10 logarithm of a field.");
py_field_unary!(cos, "Element-wise cosine of a field (radians).");
py_field_unary!(sin, "Element-wise sine of a field (radians).");
py_field_unary!(tan, "Element-wise tangent of a field (radians).");
py_field_unary!(sinh, "Element-wise hyperbolic sine of a field.");
py_field_unary!(cosh, "Element-wise hyperbolic cosine of a field.");
py_field_unary!(tanh, "Element-wise hyperbolic tangent of a field.");

// ── Composantes : filtrer, renommer ─────────────────────────────────────────
//
// Ces deux verbes n'ont **pas** de fonction libre : ils sont des méthodes et
// rien d'autre, la forme canonique elle-même (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Leur documentation n'a donc personne à qui
// renvoyer — elle est écrite ici, une fois par verbe, et la macro la pose sur
// les quatre saveurs. Le littéral traverse pyo3 et pyo3-stub-gen : l'aide
// complète part dans `__doc__` comme dans le stub.

/// Les deux verbes, pour une saveur **agrégat** : la valeur est tenue en
/// propre, le trait `Field` opère dessus directement.
macro_rules! py_field_components_aggregate {
    ($T:ident, $doc_filter:literal, $doc_rename:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc_filter]
            fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<$T> {
                use crate::containers::field::Field;
                let wanted = extract_names(components)?;
                Ok($T {
                    inner: self.inner.filter_components(wanted.as_slice())?,
                })
            }

            #[doc = $doc_rename]
            fn rename_component(&self, old: &str, new: &str) -> PyResult<$T> {
                use crate::containers::field::Field;
                Ok($T {
                    inner: self.inner.rename_component(old, new)?,
                })
            }
        }
    };
}

/// Les mêmes, pour une saveur **sous-conteneur** : la valeur est lue à travers
/// le handle, et `SubField` nomme le filtrage `select_components`.
macro_rules! py_field_components_sub {
    ($T:ident, $doc_filter:literal, $doc_rename:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc_filter]
            fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<$T> {
                use crate::containers::field::SubField;
                let wanted = extract_names(components)?;
                let out = self.handle.read().select_components(wanted.as_slice())?;
                Ok($T {
                    handle: Handle::new(out),
                })
            }

            #[doc = $doc_rename]
            fn rename_component(&self, old: &str, new: &str) -> PyResult<$T> {
                use crate::containers::field::SubField;
                let out = self.handle.read().rename_component(old, new)?;
                Ok($T {
                    handle: Handle::new(out),
                })
            }
        }
    };
}

/// Distribue les deux textes aux quatre saveurs. Le chapeau existe pour qu'ils
/// ne soient écrits **qu'une fois** : passés en `literal`, ils sont substitués
/// avant que pyo3 et pyo3-stub-gen ne lisent l'item, donc les deux y voient un
/// vrai texte. Une constante `const` ne conviendrait pas — un attribut `doc`
/// n'accepte qu'un littéral ou une expansion de macro, jamais un chemin.
macro_rules! py_field_components {
    ($doc_filter:literal, $doc_rename:literal) => {
        py_field_components_aggregate!(PyNodeField, $doc_filter, $doc_rename);
        py_field_components_aggregate!(PyElementField, $doc_filter, $doc_rename);
        py_field_components_sub!(PySubNodeField, $doc_filter, $doc_rename);
        py_field_components_sub!(PySubElementField, $doc_filter, $doc_rename);
    };
}

py_field_components!(
    "Keep only the named components, in the order given.\n\
     \n\
     `components` is a single name or a list of names (e.g. the result of\n\
     `model.primal_vars()`). Returns a **new** field of the caller's own kind,\n\
     sharing its support; the original is untouched. Errors if a requested name\n\
     is absent — filtering never invents a component.",
    "Rename one component, `old` to `new`, leaving every other untouched.\n\
     \n\
     Returns a **new** field of the caller's own kind, on the same support. The\n\
     component order is kept — renaming is not reordering. Errors if `old` is\n\
     absent, or if `new` is already taken: a name is how a component is\n\
     addressed, so two of them cannot share one."
);

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs polymorphes (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Le type du receveur étant connu, la méthode
// court-circuite le dispatch à quatre branches de la fonction libre et rend un
// type **précis** au lieu de `Any`. `psca` n'y figure pas : symétrique.
//
// Les onze maths élémentaires ne sont pas ici : elles naissent de
// `py_field_unary!`, avec la doc de leur fonction libre. Ne restent ci-dessous
// que les verbes dont la méthode a un corps propre.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// Voir `pyrucast.node_field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyNodeField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyNodeField {
            inner: crate::ops::node_field::mask(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_nodes(&self.inner, &band, components)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// Voir `pyrucast.element_field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyElementField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyElementField {
            inner: crate::ops::element_field::mask(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_cells(&self.inner, &band, components)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubNodeField {
    /// Voir `pyrucast.node_field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PySubNodeField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        let out = crate::ops::node_field::mask_sub(&self.handle.read(), &band, components);
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_sub_nodes(&self.handle.read(), &band, components)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubElementField {
    /// Voir `pyrucast.element_field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PySubElementField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        let out = crate::ops::element_field::mask_sub(&self.handle.read(), &band, components);
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_sub_cells(&self.handle.read(), &band, components)?,
        })
    }
}
