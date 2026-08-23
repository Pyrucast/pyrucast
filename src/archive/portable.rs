//! Portable serialization — the backbone of file save/load.
//!
//! A single mechanism (`serde` + `bincode`) produces a binary format that
//! is **identical on Linux and Windows**:
//!
//! - normalized little-endian integers (independent of host endianness);
//! - `usize` encoded **by value**, as a variable-width integer, so it travels
//!   between 32- and 64-bit platforms; a value too large for the reader's
//!   `usize` is refused rather than silently truncated;
//! - IEEE-754 `f64`.
//!
//! These are the guarantees of bincode's `standard()` configuration, which
//! fixes the byte order in the *configuration* rather than inheriting the
//! host's — a file written on Linux reads on Windows and back.
//!
//! No OS-dependent data (absolute paths, separators) ever ends up in the
//! payload: slot identifiers are opaque.

use crate::error::{PyrucastError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Objects that can be serialized to / deserialized from a portable
/// binary buffer.
///
/// Automatically implemented for every `Serialize + DeserializeOwned`.
///
/// # Example
///
/// ```
/// use pyrucast::archive::Portable;
///
/// #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
/// struct Pt {
///     x: f64,
///     y: f64,
/// }
///
/// let a = Pt { x: 1.5, y: -2.0 };
/// let bytes = a.to_bytes().unwrap();
/// let b = Pt::from_bytes(&bytes).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Portable: Sized {
    /// Serialize into a portable binary buffer.
    fn to_bytes(&self) -> Result<Vec<u8>>;
    /// Rebuild from a portable binary buffer.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

impl<T> Portable for T
where
    T: Serialize + DeserializeOwned,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| PyrucastError::Serialization(e.to_string()))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // `decode_from_slice` rend aussi le nombre d'octets lus ; un enregistrement
        // porte sa propre longueur en amont, donc on n'en a pas l'usage ici.
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map(|(value, _read)| value)
            .map_err(|e| PyrucastError::Serialization(e.to_string()))
    }
}
