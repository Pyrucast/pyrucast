//! Node-by-node scalar product of two fields — Cast3M `PSCA`.
//!
//! A **free function and not a method**, unlike most of what a field can do:
//! it takes two containers as peers, which the rule files under `ops`. It is
//! also symmetric — `psca(a, b)` and `psca(b, a)` are the same field — so it
//! carries no method form either, for the same reason `merge` does not.
//!
//! Generic over the four field flavours through
//! [`crate::containers::field::Pscal`], exactly as the element-wise
//! maths are generic through `MapValues`.

use crate::containers::field::Pscal;
use crate::error::Result;

/// Node-by-node (or point-by-point) scalar product: a **new field** of the
/// same flavour, carrying a single `"psca"` component whose value at each
/// node/point is `∑_c x_c·y_c` — a reduction over components only, the support
/// is kept.
///
/// `x` and `y` must sit on the same support/decomposition and carry the same
/// components, aligned by name. For the **global** scalar product (one float
/// over the whole field), see `xty`.
pub fn psca<T: Pscal>(x: &T, y: &T) -> Result<T> {
    x.pscal_with(y)
}
