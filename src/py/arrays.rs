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

// ─── Arrays out ──────────────────────────────────────────────────────────────

/// The struct-module code numpy reads as its canonical `int64`: `"l"` where a
/// C `long` has 64 bits (Linux, macOS), `"q"` where it has 32 (Windows).
/// Handing `"q"` everywhere would give `longlong` arrays, which some consumers
/// (medcoupling) refuse as "not INT64" although the bytes are identical.
const INT64_FORMAT: &std::ffi::CStr = if std::mem::size_of::<std::ffi::c_long>() == 8 {
    c"l"
} else {
    c"q"
};
const INT64_CODE: &str = if std::mem::size_of::<std::ffi::c_long>() == 8 {
    "l"
} else {
    "q"
};

/// What a [`PyArray`] owns: the buffer an exporter built, moved in whole.
pub enum ArrayData {
    F64(Vec<f64>),
    I64(Vec<i64>),
}

/// A read-only one-dimensional array handed out by pyrucast — node tags,
/// coordinates, connectivities, field values.
///
/// It **owns** the buffer the export built and lends it through the buffer
/// protocol: `numpy.asarray(a)` and `memoryview(a)` read it **without a
/// copy**, and pyrucast does not need numpy for that. `list(a)` or
/// `a.tolist()` copy it into Python numbers.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(frozen, name = "Array")]
pub struct PyArray {
    data: ArrayData,
    /// Element count, as the buffer protocol wants it: `shape` points here.
    len: isize,
}

impl PyArray {
    pub fn f64(values: Vec<f64>) -> Self {
        let len = values.len() as isize;
        Self {
            data: ArrayData::F64(values),
            len,
        }
    }

    pub fn i64(values: Vec<i64>) -> Self {
        let len = values.len() as isize;
        Self {
            data: ArrayData::I64(values),
            len,
        }
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyArray {
    fn __len__(&self) -> usize {
        self.len as usize
    }

    /// Item `i` (negative counts from the end) — what makes the array a
    /// sequence, so `list(a)` and `for x in a` work. Slow per element: bulk
    /// reads go through `numpy.asarray(a)`.
    fn __getitem__<'py>(&self, py: Python<'py>, i: isize) -> PyResult<Bound<'py, PyAny>> {
        let k = if i < 0 { i + self.len } else { i };
        if k < 0 || k >= self.len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Array index out of range",
            ));
        }
        let k = k as usize;
        match &self.data {
            ArrayData::F64(v) => Ok(v[k].into_pyobject(py)?.into_any()),
            ArrayData::I64(v) => Ok(v[k].into_pyobject(py)?.into_any()),
        }
    }

    /// The item format of the buffer: `"d"` (float64), or the int64 code of
    /// the platform (`"l"` where a C `long` is 64-bit, `"q"` elsewhere).
    #[getter]
    fn format(&self) -> &'static str {
        match self.data {
            ArrayData::F64(_) => "d",
            ArrayData::I64(_) => INT64_CODE,
        }
    }

    /// A copy as a Python list.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.data {
            ArrayData::F64(v) => v.into_pyobject(py).map(Bound::into_any),
            ArrayData::I64(v) => v.into_pyobject(py).map(Bound::into_any),
        }
    }

    fn __repr__(&self) -> String {
        format!("Array(len={}, format='{}')", self.len, self.format())
    }
}

// The buffer protocol, kept out of the stub: its raw pointers mean nothing to
// a type checker, and `numpy.asarray` / `memoryview` are how it is used.
#[pymethods]
impl PyArray {
    /// Lend the buffer, read-only, one-dimensional, C-contiguous.
    ///
    /// # Safety
    ///
    /// Called by CPython with a valid `view`. The pointers stored in it — the
    /// data, `shape` (this object's `len`) and the static format string — stay
    /// valid while `view.obj` holds a reference to this object, which the
    /// protocol guarantees until the view is released; the object is frozen,
    /// so neither the vector nor `len` can move or change meanwhile.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyBufferError;
        use pyo3::ffi;
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("pyrucast arrays are read-only"));
        }
        let this = slf.get();
        let (ptr, itemsize, format): (*const std::ffi::c_void, isize, &'static std::ffi::CStr) =
            match &this.data {
                ArrayData::F64(v) => (v.as_ptr().cast(), 8, c"d"),
                ArrayData::I64(v) => (v.as_ptr().cast(), 8, INT64_FORMAT),
            };
        let shape = std::ptr::addr_of!(this.len).cast_mut();
        // SAFETY: see the method's contract above.
        unsafe {
            (*view).buf = ptr.cast_mut();
            (*view).len = this.len * itemsize;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                format.as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                shape
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }
}

/// A node order named in Python: `"pyrucast"` (or `"vtk"`), `"gmsh"`, `"med"`.
pub fn parse_order(order: &str) -> PyResult<crate::ops::mesh::NodeOrder> {
    use crate::ops::mesh::NodeOrder;
    match order.trim().to_ascii_lowercase().as_str() {
        "pyrucast" | "vtk" => Ok(NodeOrder::Pyrucast),
        "gmsh" => Ok(NodeOrder::Gmsh),
        "med" => Ok(NodeOrder::Med),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown node order '{other}' (expected 'pyrucast', 'gmsh' or 'med')"
        ))),
    }
}
