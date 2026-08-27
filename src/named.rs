//! Types whose values have a **name** — the textual form a user writes.
//!
//! A dozen enums cross the API as strings: an element type (`"TRI3"`), a
//! material symmetry (`"orthotropic"`), a colour map (`"viridis"`), a plastic
//! law (`"drucker_prager"`). Each one used to carry its own hand-written
//! parser, and the three things a parser decides — how to compare, what
//! alternative spellings to accept, what to say when it fails — had drifted
//! apart: `"lag1"` was accepted where `"OFF"` was not, not by decision but by
//! how each `match` happened to be written.
//!
//! [`Named`] holds the decisions in one place:
//!
//! - **the policy** — leading and trailing spaces are trimmed, ASCII case is
//!   ignored. It applies to canonical names *and* to aliases, which is what the
//!   hand-written parsers could not guarantee;
//! - **the vocabulary** — [`VALUES`](Named::VALUES) and [`name`](Named::name)
//!   give the canonical spelling, [`aliases`](Named::aliases) the accepted
//!   alternatives. Both are **data**, so `name()` is canonical by construction
//!   rather than by discipline, and a type can be asked what it accepts;
//! - **the message** — [`parse`](Named::parse) says what was not understood and
//!   lists what would have been.
//!
//! Two ways to read a name, because both are needed:
//! [`from_name`](Named::from_name) asks *whether* the name is valid and has no
//! opinion, [`parse`](Named::parse) requires that it be.

use crate::error::{PyrucastError, Result};

/// A type whose values are designated by name in the API.
///
/// ```
/// use pyrucast::atoms::ElementType;
/// use pyrucast::named::Named;
///
/// // La casse et les espaces de bord ne comptent pas.
/// assert_eq!(ElementType::from_name(" tri3 "), Some(ElementType::TRI3));
/// assert_eq!(ElementType::from_name("TRI3"), Some(ElementType::TRI3));
/// // Un nom inconnu reste inconnu.
/// assert_eq!(ElementType::from_name("TRI4"), None);
/// // `name` est la forme canonique, et la réciproque exacte de `from_name`.
/// assert_eq!(ElementType::TRI3.name(), "TRI3");
/// ```
pub trait Named: Copy + Sized + 'static {
    /// What this family of names designates, for error messages — `"element
    /// type"`, `"symmetry"`, `"plastic law"`. Lowercase, no article: it is
    /// inserted into `unknown {label} '…'`.
    const LABEL: &'static str;

    /// Every value, in a stable order. The canonical names are read from it,
    /// through [`name`](Self::name).
    ///
    /// Named `VALUES` and not `ALL` on purpose: several of these types already
    /// carry an inherent `ALL` of their own, and two associated constants of
    /// the same name on one type resolve in a way that is easy to misread.
    const VALUES: &'static [Self];

    /// The canonical name of this value — the one the API prints back.
    fn name(self) -> &'static str;

    /// Alternative spellings accepted on input, never printed back. Empty by
    /// default: most types have exactly one name per value.
    ///
    /// This is the *vocabulary*, proper to each type — `LAG1` for
    /// `LAGRANGE1`, `off` for `none`. It is deliberately **not** unified; only
    /// the way it is handled is.
    fn aliases() -> &'static [(&'static str, Self)] {
        &[]
    }

    /// The value named `s`, or `None` — spaces trimmed, ASCII case ignored,
    /// canonical names first and aliases next.
    ///
    /// ```
    /// use pyrucast::atoms::Interpolation;
    /// use pyrucast::named::Named;
    ///
    /// // Un alias suit la même politique qu'un nom canonique : c'est
    /// // précisément ce que les analyseurs écrits à la main ne garantissaient
    /// // pas.
    /// assert_eq!(Interpolation::from_name("lag1"), Some(Interpolation::Lagrange1));
    /// assert_eq!(Interpolation::from_name("LAGRANGE1"), Some(Interpolation::Lagrange1));
    /// ```
    fn from_name(s: &str) -> Option<Self> {
        let wanted = s.trim();
        Self::VALUES
            .iter()
            .copied()
            .find(|v| v.name().eq_ignore_ascii_case(wanted))
            .or_else(|| {
                Self::aliases()
                    .iter()
                    .find(|(alias, _)| alias.eq_ignore_ascii_case(wanted))
                    .map(|(_, v)| *v)
            })
    }

    /// The value named `s`, or an error naming what was not understood and
    /// listing what would have been.
    ///
    /// ```
    /// use pyrucast::atoms::ElementType;
    /// use pyrucast::named::Named;
    ///
    /// assert_eq!(ElementType::parse("qua4")?, ElementType::QUA4);
    /// // Le message porte le nom refusé *et* les noms acceptés.
    /// let err = ElementType::parse("QUA5").unwrap_err().to_string();
    /// assert!(err.contains("QUA5") && err.contains("QUA4"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    fn parse(s: &str) -> Result<Self> {
        Self::from_name(s).ok_or_else(|| {
            PyrucastError::Message(format!(
                "unknown {} '{}' (expected {})",
                Self::LABEL,
                s.trim(),
                Self::names()
            ))
        })
    }

    /// The canonical names, `|`-joined — for error messages. Aliases are left
    /// out: they are accepted, not advertised.
    ///
    /// ```
    /// use pyrucast::models::symmetry::MaterialSymmetry;
    /// use pyrucast::named::Named;
    ///
    /// assert_eq!(MaterialSymmetry::names(), "isotropic|orthotropic|anisotropic");
    /// ```
    fn names() -> String {
        Self::VALUES
            .iter()
            .map(|v| v.name())
            .collect::<Vec<_>>()
            .join("|")
    }
}

/// Implement the Python conversion of a [`Named`] type: a Python `str` in,
/// the value out — or a `ValueError` naming what was not understood.
///
/// The whole point is that a `#[pyfunction]` can then take the Rust type
/// **directly**, so its Python signature is its Rust signature and the wrapper
/// has nothing left to decide. The error is built as a `ValueError` rather than
/// going through `From<PyrucastError>`, which would raise a `RuntimeError`: a
/// name that is not understood is a bad argument, and Python callers catch it
/// as such.
///
/// ```
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::named::Named;
/// // Le trait est la seule chose que la macro demande au type.
/// assert_eq!(ElasticityModel::from_name("SOLID"), Some(ElasticityModel::Solid));
/// ```
#[macro_export]
macro_rules! named_enum {
    ($($ty:ty),+ $(,)?) => { $(
        #[cfg(feature = "python-api")]
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for $ty {
            type Error = ::pyo3::PyErr;

            // `::pyo3::PyErr` plutôt que `Self::Error` : un énuméré qui porte
            // une variante `Error` — `OutOfRange` en a une — rendrait le type
            // associé ambigu.
            fn extract(
                obj: ::pyo3::Borrowed<'a, 'py, ::pyo3::PyAny>,
            ) -> ::std::result::Result<Self, ::pyo3::PyErr> {
                let name: String = obj.extract()?;
                <$ty as $crate::named::Named>::parse(&name)
                    .map_err(|e| ::pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }
        }

        #[cfg(feature = "stub-gen")]
        impl ::pyo3_stub_gen::PyStubType for $ty {
            fn type_output() -> ::pyo3_stub_gen::TypeInfo {
                ::pyo3_stub_gen::TypeInfo::builtin("str")
            }
        }
    )+ };
}

named_enum!(
    crate::atoms::ElementType,
    crate::atoms::Interpolation,
    crate::atoms::QuadratureRule,
    crate::models::Physics,
    crate::models::damage::law::DamageLaw,
    crate::models::elasticity::ElasticityModel,
    crate::models::plasticity::law::PlasticLaw,
    crate::models::shell::ShellModel,
    crate::models::symmetry::MaterialSymmetry,
    crate::containers::evolution::OutOfRange,
    crate::ops::mesh::FrontRelax,
);

#[cfg(feature = "viz")]
named_enum!(crate::viz::Colormap);
