//! Operators that are **polymorphic over the field flavour** — they take
//! any field and give back one of the *same* kind.
//!
//! Their product is their argument's own type, so the "one module per
//! produced container" rule cannot place them: `mask` on a `NodeField`
//! yields a `NodeField`, on an `ElementField` an `ElementField`. They are
//! therefore grouped by **domain** rather than by product — the third case of
//! the rule, alongside "one module per produced container" and "grouped by
//! activity when nothing is produced".
//!
//! They are full-fledged free functions, and each also carries a method on the
//! four field flavours (`f.sqrt()`, `f.mask(ge=…)`), like any other operator
//! meeting the three conditions.
//!
//! - [`Band`] — the shared `ge`/`gt`/`le`/`lt` comparison band;
//! - [`mask_nodes`](fn@mask_nodes) & co. — 0/1 indicator of the same shape
//!   (Cast3M `MASQUE`). Its sibling
//!   [`select`](crate::ops::mesh::select_nodes), which extracts the passing
//!   *support* instead of rewriting the values, produces a `Mesh` and lives
//!   with the mesh operators;
//! - the element-wise scalar maths ([`abs`](fn@abs), [`sqrt`](fn@sqrt), …).

pub mod band;
pub mod elementwise;
pub mod mask;
pub mod methods;

pub use band::Band;
pub use elementwise::{abs, cos, cosh, exp, log, log10, sin, sinh, sqrt, tan, tanh};
pub use mask::{mask_cells, mask_nodes, mask_sub_cells, mask_sub_nodes};
