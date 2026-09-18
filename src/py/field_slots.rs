//! What the field slots share: two functions, not a macro.
//!
//! `__pow__` and `__richcmp__` exist in four copies — one per field flavour —
//! and must be written in the module declaring their `#[pyclass]` (pyo3
//! generates for a slot an `unsafe fn` trampoline that edition 2024 no longer
//! covers outside that module). What they have in common is not their shape,
//! which is a handful of lines, but their **semantics**: which value band a
//! comparison stands for, and why a modulo is refused. That is what lives here,
//! called by the eight methods.

use crate::atoms::Band;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

/// The value band this comparison stands for: `>= x` bounds from below, `< x`
/// from above, and so on.
///
/// `None` when the comparison has no band — `==` and `!=`, which are not
/// thresholds, and any right-hand side that is not a scalar. The caller then
/// returns `NotImplemented`, leaving Python to look for the reflected operation.
pub(crate) fn band_of(op: CompareOp, other: &Bound<'_, PyAny>) -> PyResult<Option<Band>> {
    let Ok(x) = other.extract::<f64>() else {
        return Ok(None);
    };
    let band = match op {
        CompareOp::Ge => Band::new(Some(x), None, None, None),
        CompareOp::Gt => Band::new(None, Some(x), None, None),
        CompareOp::Le => Band::new(None, None, Some(x), None),
        CompareOp::Lt => Band::new(None, None, None, Some(x)),
        CompareOp::Eq | CompareOp::Ne => return Ok(None),
    }?;
    Ok(Some(band))
}

/// Refuses the ternary `pow(x, y, z)` form: a modulo makes no sense on floats,
/// and accepting it silently would drop it from the computation.
pub(crate) fn reject_modulo(modulo: &Bound<'_, PyAny>) -> PyResult<()> {
    if modulo.is_none() {
        return Ok(());
    }
    Err(PyTypeError::new_err(
        "field ** exponent does not support a modulo argument",
    ))
}
