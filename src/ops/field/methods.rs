//! Delegation methods — the "subject" face of the polymorphic operators.
//!
//! See `CONVENTIONS.md` § "The verb also exposed as a method". The free
//! function stays the canonical form; these methods contain no logic.
//!
//! This module's operators return a field of the kind received, so the
//! four flavours (`NodeField` / `SubNodeField` / `ElementField` /
//! `SubElementField`) carry the same methods. `psca` is not among them: the
//! scalar product is **symmetric**, `a.psca(b)` would suggest that the order
//! matters.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;

/// Generates, for one field flavour, the eleven element-wise maths. The mask is
/// no longer here: `mask` produces a determined container, so it has joined
/// `ops::node_field` and `ops::element_field`.
macro_rules! field_methods {
    ($T:ty) => {
        impl $T {
            /// Absolute value, see [`field::abs`](fn@crate::ops::field::abs).
            pub fn abs(&self) -> Result<Self> {
                crate::ops::field::abs(self)
            }
            /// Square root, see [`field::sqrt`](fn@crate::ops::field::sqrt).
            pub fn sqrt(&self) -> Result<Self> {
                crate::ops::field::sqrt(self)
            }
            /// Exponential, see [`field::exp`](fn@crate::ops::field::exp).
            pub fn exp(&self) -> Result<Self> {
                crate::ops::field::exp(self)
            }
            /// Natural logarithm, see [`field::log`](fn@crate::ops::field::log).
            pub fn log(&self) -> Result<Self> {
                crate::ops::field::log(self)
            }
            /// Decimal logarithm, see [`field::log10`](fn@crate::ops::field::log10).
            pub fn log10(&self) -> Result<Self> {
                crate::ops::field::log10(self)
            }
            /// Cosine, see [`field::cos`](fn@crate::ops::field::cos).
            pub fn cos(&self) -> Result<Self> {
                crate::ops::field::cos(self)
            }
            /// Sine, see [`field::sin`](fn@crate::ops::field::sin).
            pub fn sin(&self) -> Result<Self> {
                crate::ops::field::sin(self)
            }
            /// Tangent, see [`field::tan`](fn@crate::ops::field::tan).
            pub fn tan(&self) -> Result<Self> {
                crate::ops::field::tan(self)
            }
            /// Hyperbolic sine, see [`field::sinh`](fn@crate::ops::field::sinh).
            pub fn sinh(&self) -> Result<Self> {
                crate::ops::field::sinh(self)
            }
            /// Hyperbolic cosine, see [`field::cosh`](fn@crate::ops::field::cosh).
            pub fn cosh(&self) -> Result<Self> {
                crate::ops::field::cosh(self)
            }
            /// Hyperbolic tangent, see [`field::tanh`](fn@crate::ops::field::tanh).
            pub fn tanh(&self) -> Result<Self> {
                crate::ops::field::tanh(self)
            }
        }
    };
}

field_methods!(NodeField);
field_methods!(ElementField);
field_methods!(SubNodeField);
field_methods!(SubElementField);
