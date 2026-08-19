//! Unique error type of pyrucast.
//!
//! Every public function returns [`Result<T>`]. On the Python side,
//! [`PyrucastError`] is converted automatically into an exception
//! (`RuntimeError`).

use std::fmt;

/// Standard result type used throughout the library.
///
/// ```
/// # use pyrucast::{PyrucastError, Result};
/// # use pyrucast::coords::Coords;
/// // Toute la bibliothèque rend ce `Result` : une erreur porte un message
/// // qui nomme l'opération et ce qui a manqué, jamais un code nu.
/// fn repere(dim: u8) -> Result<Coords> {
///     Coords::new(dim)
/// }
/// assert!(repere(2).is_ok());
/// let e = repere(0).unwrap_err();
/// assert!(matches!(e, PyrucastError::Message(_)));
/// assert!(e.to_string().contains("dim"));
/// ```
pub type Result<T> = std::result::Result<T, PyrucastError>;

/// Single error type for pyrucast.
///
/// # Example
///
/// ```
/// use pyrucast::PyrucastError;
///
/// let e = PyrucastError::MeshSealed;
/// assert!(e.to_string().starts_with("submesh is sealed"));
/// ```
#[derive(Debug)]
pub enum PyrucastError {
    /// I/O error (file save/load).
    Io(String),
    /// (De)serialization failure.
    Serialization(String),
    /// The computation was cancelled by its [`crate::interrupt::Cancel`]
    /// token (e.g. a `Ctrl+C`, a timeout, or an external stop flag). On the
    /// Python side this surfaces as `KeyboardInterrupt`.
    Interrupted,
    /// A structural mutation was attempted on a **sealed** [`crate::containers::mesh::SubMesh`].
    /// A submesh is sealed the first time a non-mesh object (finite-element
    /// space, field, matrix, …) captures it, after which its connectivity is
    /// frozen so downstream objects cannot be left inconsistent.
    MeshSealed,
    /// Generic error with a message.
    Message(String),
}

impl fmt::Display for PyrucastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyrucastError::Io(m) => write!(f, "I/O error: {m}"),
            PyrucastError::Serialization(m) => write!(f, "serialization error: {m}"),
            PyrucastError::Interrupted => write!(f, "computation interrupted"),
            PyrucastError::MeshSealed => write!(
                f,
                "submesh is sealed: it is already used by a finite-element space, \
                 field or matrix and can no longer be modified"
            ),
            PyrucastError::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for PyrucastError {}

impl From<std::io::Error> for PyrucastError {
    fn from(e: std::io::Error) -> Self {
        PyrucastError::Io(e.to_string())
    }
}

#[cfg(feature = "python-api")]
impl From<PyrucastError> for pyo3::PyErr {
    fn from(e: PyrucastError) -> Self {
        match e {
            // A cancelled computation must surface as the Python-native
            // KeyboardInterrupt, not a generic RuntimeError.
            PyrucastError::Interrupted => {
                pyo3::exceptions::PyKeyboardInterrupt::new_err(e.to_string())
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        }
    }
}
