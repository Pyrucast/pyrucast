//! **Generic** operators over the field flavour — they take any field and give
//! back one of the *same* kind.
//!
//! Their product is a container — always — but not a *determined* one: `abs`
//! yields a `NodeField` or an `ElementField` depending on what it is given, so
//! the "one module per produced container" rule does not select a module. They
//! are therefore grouped by **domain**, the third case of the rule, alongside
//! "one module per produced container" and "grouped by activity when nothing
//! is produced".
//!
//! Beware the false friend: two **monomorphic** functions of one family each
//! have a determined product and file normally. `mask` used to live here for
//! that mistaken reason; it is really two functions, and they now sit in
//! [`node_field::mask`](fn@crate::ops::node_field::mask) and
//! [`element_field::mask`](fn@crate::ops::element_field::mask). Only genuinely
//! generic code (`pub fn f<T: MapValues>(x: &T) -> Result<T>`) belongs here.
//!
//! What is left is the element-wise scalar maths ([`abs`](fn@abs),
//! [`sqrt`](fn@sqrt), …) and [`psca`](fn@psca), one generic function each.
//! The maths carry a method on the four flavours (`f.sqrt()`); `psca` does
//! not — it takes two containers as peers and is symmetric, so it is a free
//! function only, like `merge`.

pub mod elementwise;
pub mod methods;
pub mod psca;

pub use elementwise::{abs, cos, cosh, exp, log, log10, sin, sinh, sqrt, tan, tanh};
pub use psca::psca;
