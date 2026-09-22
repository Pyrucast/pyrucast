//! Python wrapper for [`crate::spill`] — what the user-space swap has done.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Counters of the spilling allocator, as a dictionary.
///
/// Keys: `threshold` (bytes from which an allocation spills, `None` when
/// spilling is off), `count` (blocks spilled so far), `largest`, `live` (bytes
/// mapped right now) and `peak`.
///
/// Spilling is configured before the process starts, through
/// `PYRUCAST_SPILL_DIR` and `PYRUCAST_SPILL_MIN`; on a build that cannot spill
/// at all — the feature absent, or a platform other than Unix — every count is
/// zero and `threshold` is `None`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn spill_stats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    #[cfg(all(unix, feature = "spill"))]
    {
        let s = crate::spill::stats();
        d.set_item(
            "threshold",
            (s.threshold != usize::MAX).then_some(s.threshold),
        )?;
        d.set_item("count", s.count)?;
        d.set_item("largest", s.largest)?;
        d.set_item("live", s.live)?;
        d.set_item("peak", s.peak)?;
    }
    #[cfg(not(all(unix, feature = "spill")))]
    {
        d.set_item("threshold", None::<usize>)?;
        for key in ["count", "largest", "live", "peak"] {
            d.set_item(key, 0usize)?;
        }
    }
    Ok(d.unbind())
}
