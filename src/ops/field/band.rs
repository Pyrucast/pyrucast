//! Value-range predicate shared by [`select`](super::select) and
//! [`mask`](super::mask): a `[lower, upper]` band whose two sides are each
//! open, inclusive (`>=` / `<=`), or strict (`>` / `<`).
//!
//! Built from the four comparison bounds `ge` / `gt` / `le` / `lt` — one
//! lower bound (`ge` **or** `gt`) and one upper (`le` **or** `lt`), mirroring
//! Python's comparison operators one for one.

use crate::error::{PyrucastError, Result};

/// A value band, each side open / inclusive / strict. At least one bound set.
#[derive(Clone, Copy)]
pub struct Band {
    min: Option<f64>,
    max: Option<f64>,
    /// `true` ⇒ the lower bound is strict (`v > min` instead of `v >= min`).
    strict_min: bool,
    /// `true` ⇒ the upper bound is strict (`v < max` instead of `v <= max`).
    strict_max: bool,
}

impl Band {
    /// Build from the four comparison bounds:
    /// - `ge` ⇒ `v >= ge`, `gt` ⇒ `v > gt` (give at most one lower bound);
    /// - `le` ⇒ `v <= le`, `lt` ⇒ `v < lt` (give at most one upper bound).
    ///
    /// At least one bound overall, and the lower must not exceed the upper.
    pub fn new(ge: Option<f64>, gt: Option<f64>, le: Option<f64>, lt: Option<f64>) -> Result<Self> {
        let (min, strict_min) = match (ge, gt) {
            (Some(_), Some(_)) => {
                return Err(PyrucastError::Message(
                    "value band: give at most one lower bound (`ge` or `gt`, not both)".into(),
                ));
            }
            (Some(v), None) => (Some(v), false),
            (None, Some(v)) => (Some(v), true),
            (None, None) => (None, false),
        };
        let (max, strict_max) = match (le, lt) {
            (Some(_), Some(_)) => {
                return Err(PyrucastError::Message(
                    "value band: give at most one upper bound (`le` or `lt`, not both)".into(),
                ));
            }
            (Some(v), None) => (Some(v), false),
            (None, Some(v)) => (Some(v), true),
            (None, None) => (None, false),
        };
        if min.is_none() && max.is_none() {
            return Err(PyrucastError::Message(
                "value band: at least one bound (`ge`/`gt`/`le`/`lt`) must be given".into(),
            ));
        }
        if let (Some(lo), Some(hi)) = (min, max) {
            if lo > hi {
                return Err(PyrucastError::Message(format!(
                    "value band: lower bound ({lo}) is greater than upper bound ({hi})"
                )));
            }
        }
        Ok(Band {
            min,
            max,
            strict_min,
            strict_max,
        })
    }

    /// Whether `v` lies in the band (a missing bound leaves that side open).
    pub fn contains(&self, v: f64) -> bool {
        let lo_ok = self
            .min
            .is_none_or(|lo| if self.strict_min { v > lo } else { v >= lo });
        let hi_ok = self
            .max
            .is_none_or(|hi| if self.strict_max { v < hi } else { v <= hi });
        lo_ok && hi_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_vs_strict() {
        // ge/le inclusive: bounds are members.
        let b = Band::new(Some(1.0), None, Some(3.0), None).unwrap();
        assert!(b.contains(1.0) && b.contains(3.0) && b.contains(2.0));
        assert!(!b.contains(0.999) && !b.contains(3.001));
        // gt/lt strict: bounds excluded.
        let b = Band::new(None, Some(1.0), None, Some(3.0)).unwrap();
        assert!(!b.contains(1.0) && !b.contains(3.0));
        assert!(b.contains(2.0));
    }

    #[test]
    fn open_sides() {
        assert!(Band::new(Some(0.0), None, None, None)
            .unwrap()
            .contains(1e9));
        assert!(Band::new(None, None, Some(0.0), None)
            .unwrap()
            .contains(-1e9));
    }

    #[test]
    fn rejects_bad_bounds() {
        assert!(Band::new(None, None, None, None).is_err());
        assert!(Band::new(Some(5.0), None, Some(1.0), None).is_err());
        // Two lower bounds, or two upper bounds, are contradictory.
        assert!(Band::new(Some(0.0), Some(0.0), None, None).is_err());
        assert!(Band::new(None, None, Some(0.0), Some(0.0)).is_err());
    }
}
