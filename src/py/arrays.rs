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

use crate::handle::{Handle, Loan};
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{c_int, c_void, CStr};

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
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let this = slf.get();
        check_request(view, flags, &[this.len], "pyrucast arrays")?;
        let (buf, format): (*const c_void, &'static CStr) = match &this.data {
            ArrayData::F64(v) => (v.as_ptr().cast(), c"d"),
            ArrayData::I64(v) => (v.as_ptr().cast(), INT64_FORMAT),
        };
        // SAFETY: see the method's contract above. In one dimension the step is
        // the item size, so `strides` points at `view.itemsize` itself.
        unsafe {
            fill_view(
                view,
                flags,
                Block {
                    buf,
                    format,
                    ndim: 1,
                    shape: std::ptr::addr_of!(this.len).cast_mut(),
                    strides: std::ptr::addr_of_mut!((*view).itemsize),
                },
                std::ptr::null_mut(),
                slf.into_any(),
            );
        }
        Ok(())
    }
}

// ─── Lending a zone's values ─────────────────────────────────────────────────

/// Refuse what a lent block cannot give: a null view, a **writable** view
/// (everything pyrucast lends is read-only), or a Fortran-ordered one of a
/// block that is C-ordered in more than one non-trivial dimension.
fn check_request(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    shape: &[isize],
    what: &str,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("View is null"));
    }
    if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err(format!("{what} are read-only")));
    }
    if (flags & ffi::PyBUF_F_CONTIGUOUS) == ffi::PyBUF_F_CONTIGUOUS
        && shape.iter().filter(|&&n| n > 1).count() > 1
    {
        return Err(PyBufferError::new_err(format!(
            "{what} are C-contiguous, not Fortran-contiguous"
        )));
    }
    Ok(())
}

/// A read-only, C-contiguous block of 8-byte items, as a view describes it.
struct Block {
    buf: *const c_void,
    format: &'static CStr,
    ndim: c_int,
    /// `ndim` sizes and `ndim` steps in bytes; both stay valid until the
    /// view is released.
    shape: *mut isize,
    strides: *mut isize,
}

/// Fill `view` with `block`, giving the reader what its `flags` ask for: the
/// format with `PyBUF_FORMAT`, the shape with `PyBUF_ND`, the steps with
/// `PyBUF_STRIDES`. A reader asking for none sees the block as flat bytes.
///
/// # Safety
///
/// `view` is the valid `Py_buffer` CPython hands to `__getbuffer__`, already
/// checked by [`check_request`]. What `block` points at stays valid until the
/// view is released, `internal` is what `__releasebuffer__` will find in
/// `view.internal`, and `owner` — stored in `view.obj` — keeps the exporter
/// alive until then.
unsafe fn fill_view(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    block: Block,
    internal: *mut c_void,
    owner: Bound<'_, PyAny>,
) {
    const ITEMSIZE: isize = 8;
    let wants = |flag: c_int| (flags & flag) == flag;
    // SAFETY: the caller's contract.
    unsafe {
        let count: isize = (0..block.ndim as usize)
            .map(|k| *block.shape.add(k))
            .product();
        (*view).buf = block.buf.cast_mut();
        (*view).len = count * ITEMSIZE;
        (*view).readonly = 1;
        (*view).itemsize = ITEMSIZE;
        (*view).format = if wants(ffi::PyBUF_FORMAT) {
            block.format.as_ptr().cast_mut()
        } else {
            std::ptr::null_mut()
        };
        let nd = wants(ffi::PyBUF_ND);
        (*view).ndim = if nd { block.ndim } else { 1 };
        (*view).shape = if nd {
            block.shape
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if wants(ffi::PyBUF_STRIDES) {
            block.strides
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = internal;
        (*view).obj = owner.into_ptr();
    }
}

/// What one buffer view on a zone keeps until CPython releases it: its
/// [`Loan`], and the arrays `view.shape` and `view.strides` point into (the
/// first `ndim` entries of each are used). It lives boxed in `view.internal`,
/// the one member of the view CPython leaves to the exporter.
///
/// `Loan` stays in [`crate::handle`], pure Rust; this box is what the C
/// protocol adds to it.
struct KeptByView<T> {
    _loan: Loan<T>,
    shape: [isize; 3],
    strides: [isize; 3],
}

/// Lend the `f64` values of the object behind `handle` to `view`, read-only
/// and without a copy, until CPython calls [`release_values`].
///
/// `layout` reads, under the read lock, where the values start and the sizes
/// of their first `ndim` dimensions (at most 3), the last one varying fastest.
/// The lock is released at once: from then on, the [`Loan`] is what keeps a
/// [`write_checked`](crate::handle::Handle::write_checked) from writing under
/// the view.
///
/// # Safety
///
/// `view` is the valid `Py_buffer` CPython hands to `__getbuffer__`, and
/// `owner` the Python object it is asked of, which holds `handle`. The values
/// must not move while nothing writes the object: their buffer is never
/// reallocated after construction.
pub(crate) unsafe fn lend_values<T>(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    owner: Bound<'_, PyAny>,
    handle: &Handle<T>,
    what: &str,
    layout: impl FnOnce(&T) -> (*const f64, [usize; 3], usize),
) -> PyResult<()> {
    let loan = handle.lend();
    let (buf, dims, ndim) = layout(&handle.read());
    let mut shape = [0isize; 3];
    let mut strides = [0isize; 3];
    let mut step = 8isize;
    for k in (0..ndim).rev() {
        shape[k] = dims[k] as isize;
        strides[k] = step;
        step *= shape[k];
    }
    check_request(view, flags, &shape[..ndim], what)?;
    let kept = Box::into_raw(Box::new(KeptByView {
        _loan: loan,
        shape,
        strides,
    }));
    // SAFETY: the box lives until `release_values`, so the shape and steps the
    // view points into do too; the caller's contract covers the rest.
    unsafe {
        fill_view(
            view,
            flags,
            Block {
                buf: buf.cast(),
                format: c"d",
                ndim: ndim as c_int,
                shape: (*kept).shape.as_mut_ptr(),
                strides: (*kept).strides.as_mut_ptr(),
            },
            kept.cast(),
            owner,
        );
    }
    Ok(())
}

/// End a view opened by [`lend_values`]: free its box, and with it the
/// [`Loan`], so the object's count of views goes down by one.
///
/// # Safety
///
/// `view` is a view [`lend_values`] filled for a `Handle<T>`, released once.
pub(crate) unsafe fn release_values<T>(view: *mut ffi::Py_buffer) {
    // SAFETY: the caller's contract — `internal` is the box `lend_values` made.
    unsafe {
        drop(Box::from_raw((*view).internal.cast::<KeptByView<T>>()));
    }
}
