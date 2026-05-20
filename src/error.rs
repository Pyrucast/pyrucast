//! Unique error type of pyrucast.
//!
//! Every public function returns [`Result<T>`]. On the Python side,
//! [`PyrucastError`] is converted automatically into an exception
//! (`RuntimeError`).

use std::fmt;

/// Standard result type used throughout the library.
pub type Result<T> = std::result::Result<T, PyrucastError>;

/// Single error type for pyrucast.
///
/// # Example
///
/// ```
/// use pyrucast::PyrucastError;
///
/// let e = PyrucastError::StaleHandle;
/// assert_eq!(
///     e.to_string(),
///     "stale handle (slot freed or generation mismatch)"
/// );
/// ```
#[derive(Debug)]
pub enum PyrucastError {
    /// I/O error (disk swap, file save/load).
    Io(String),
    /// (De)serialization failure.
    Serialization(String),
    /// Stale handle: slot freed or generation mismatch.
    StaleHandle,
    /// Generic error with a message.
    Message(String),
}

impl fmt::Display for PyrucastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyrucastError::Io(m) => write!(f, "I/O error: {m}"),
            PyrucastError::Serialization(m) => write!(f, "serialization error: {m}"),
            PyrucastError::StaleHandle => {
                write!(f, "stale handle (slot freed or generation mismatch)")
            }
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

#[cfg(feature = "extension-module")]
impl From<PyrucastError> for pyo3::PyErr {
    fn from(e: PyrucastError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}
