//! Reading a Python array without copying it.
//!
//! gmsh hands its node table and connectivity over as numpy arrays — and those
//! are *views* on gmsh's own memory, not copies (`gmsh.py` wraps the C pointer
//! with `numpy.ctypeslib.as_array` and frees it from a weakref finalizer). It
//! would be a shame to then walk them element by element through CPython.
//!
//! So we take the **buffer protocol** (PEP 3118): one call gives the raw
//! pointer, the item format and the shape, and holds the exporter to its
//! promise not to move the block while the view is held. That is what
//! [`pyo3::buffer::PyBuffer`] wraps, and why the crate's `abi3` floor is
//! CPython 3.11 — `PyObject_GetBuffer` entered the limited API only there.
//!
//! Not everything implements the protocol: a plain `list` does not, and gmsh
//! falls back to lists when numpy is absent. Those go through pyo3's ordinary
//! sequence conversion instead, which copies. Callers see a `&[T]` either way.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::prelude::*;

/// A slice that is either borrowed straight from a Python buffer or owned
/// because the object had no buffer to lend.
///
/// [`Borrowed::as_slice`] is the only thing callers need; which arm they got
/// is a performance detail, not a semantic one.
pub enum Borrowed<'a, T: Element> {
    /// A view on the exporter's memory. Nothing was copied.
    View {
        /// Held, never read: while this lives, the exporter has promised not
        /// to move or free the block. Releasing it early would leave `view`
        /// resting on nothing but the argument that no Python code can run in
        /// between — true today, and one future edit away from being false.
        _guard: PyBuffer<T>,
        view: &'a [T],
    },
    /// A copy, because the object exposes no buffer (a `list`, a generator…).
    Owned(Vec<T>),
}

impl<T: Element> Borrowed<'_, T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::View { view, .. } => view,
            Self::Owned(v) => v,
        }
    }
}

/// Borrow `obj`'s contents as a contiguous `&[T]`, copying only if it exposes
/// no buffer.
///
/// The buffer path is taken when the object exports a **C-contiguous,
/// one-dimensional** block whose item format matches `T` — which is what numpy
/// gives for a plain 1-D array of the matching dtype. Anything else (a list, a
/// strided view, a mismatched dtype) falls back to `extract()`, so a caller
/// never has to care which it got.
///
/// The returned slice borrows from `obj`, so it cannot outlive the reference
/// the caller holds.
pub fn borrow<'a, T>(obj: &'a Bound<'_, PyAny>) -> PyResult<Borrowed<'a, T>>
where
    T: Element + Copy + for<'b, 'py> FromPyObject<'b, 'py>,
{
    if let Ok(guard) = PyBuffer::<T>::get(obj)
        && guard.dimensions() == 1
        && guard.is_c_contiguous()
        && let Some(cells) = guard.as_slice(obj.py())
    {
        let (ptr, len) = (cells.as_ptr(), cells.len());
        // SAFETY: `ReadOnlyCell<T>` is `#[repr(transparent)]` over
        // `UnsafeCell<T>`, itself `#[repr(transparent)]` over `T`, so the two
        // slices have the same layout. The block belongs to the exporter, not
        // to `guard`, so moving `guard` into the returned value leaves the
        // pointer valid — and `guard` living exactly as long as the view is
        // what keeps the exporter from moving it. Read only, never written.
        let view = unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), len) };
        return Ok(Borrowed::View {
            _guard: guard,
            view,
        });
    }
    Ok(Borrowed::Owned(obj.extract::<Vec<T>>()?))
}
