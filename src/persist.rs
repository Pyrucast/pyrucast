//! Portable serialization — the backbone of file save/load.
//!
//! A single mechanism (`serde` + `bincode`) produces a binary format that
//! is **identical on Linux and Windows**:
//!
//! - normalized little-endian integers (independent of host endianness);
//! - `usize` encoded on 64 bits (portable between 32/64-bit platforms);
//! - IEEE-754 `f64`.
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
/// use pyrucast::persist::Persist;
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
pub trait Persist: Sized {
    /// Serialize into a portable binary buffer.
    fn to_bytes(&self) -> Result<Vec<u8>>;
    /// Rebuild from a portable binary buffer.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

impl<T> Persist for T
where
    T: Serialize + DeserializeOwned,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| PyrucastError::Serialization(e.to_string()))
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| PyrucastError::Serialization(e.to_string()))
    }
}
