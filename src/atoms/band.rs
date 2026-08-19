//! Value-range predicate: a `[lower, upper]` band whose two sides are each
//! open, inclusive (`>=` / `<=`), or strict (`>` / `<`).
//!
//! An atom by the usual test — no part of a band is a band — and it lives here
//! rather than with one of its users because it has several, in different
//! modules: [`node_field::mask`](fn@crate::ops::node_field::mask),
//! [`element_field::mask`](fn@crate::ops::element_field::mask) and
//! [`mesh::select_nodes`](crate::ops::mesh::select_nodes).
//!
//! Built from the four comparison bounds `ge` / `gt` / `le` / `lt` — one
//! lower bound (`ge` **or** `gt`) and one upper (`le` **or** `lt`), mirroring
//! Python's comparison operators one for one.

use crate::error::{PyrucastError, Result};

/// A value band, each side open / inclusive / strict. At least one bound set.
///
/// ```
/// # use pyrucast::atoms::Band;
/// // Les quatre bornes sont optionnelles et s'excluent deux à deux :
/// // `ge`/`gt` d'un côté, `le`/`lt` de l'autre.
/// let bande = Band::new(Some(0.0), None, None, Some(1.0)).unwrap(); // 0 ≤ v < 1
/// assert!(bande.contains(0.0));
/// assert!(!bande.contains(1.0));
/// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::Band;
    /// // 20 ≤ v ≤ 80, bornes inclusives.
    /// let bande = Band::new(Some(20.0), None, Some(80.0), None).unwrap();
    /// assert!(bande.contains(20.0) && bande.contains(80.0));
    /// // Bornes contradictoires : refusé.
    /// assert!(Band::new(Some(1.0), Some(2.0), None, None).is_err());
    /// ```
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
        if let (Some(lo), Some(hi)) = (min, max)
            && lo > hi
        {
            return Err(PyrucastError::Message(format!(
                "value band: lower bound ({lo}) is greater than upper bound ({hi})"
            )));
        }
        Ok(Band {
            min,
            max,
            strict_min,
            strict_max,
        })
    }

    /// Whether `v` lies in the band (a missing bound leaves that side open).
    ///
    /// ```
    /// # use pyrucast::atoms::Band;
    /// let stricte = Band::new(None, Some(0.0), None, Some(1.0)).unwrap(); // 0 < v < 1
    /// assert!(stricte.contains(0.5));
    /// assert!(!stricte.contains(0.0)); // borne exclue
    /// ```
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
