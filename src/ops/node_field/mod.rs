//! Operators that **produce** a [`NodeField`](crate::containers::node_field::NodeField).
//!
//! Two families meet here, because both end on a nodal field:
//!
//! - *derivations* — [`positions`](fn@positions) (a mesh's geometry read
//!   as a field), [`divergence`](fn@divergence) (the nodal `∇·` of a
//!   per-element field), [`restrict`](fn@restrict) /
//!   [`restrict_like`](fn@restrict_like), [`merge`](fn@merge),
//!   [`consolidate`](fn@consolidate);
//! - *nodal assembly* — [`internal_forces`](fn@internal_forces) and
//!   [`external_forces`](fn@external_forces), the two sides of the balance
//!   `Σ f_int = Σ f_ext` whose gap is the residual. They are assemblies like
//!   the ones in [`crate::ops::matrix`], but their result is a vector, not an
//!   operator. A distributed load is no longer among them: it is a sub-model
//!   ([`crate::models::flux`]), and its term comes out of `external_forces`
//!   with every other given term.
//!
//! Resolution (`A · x = b`) also produces a nodal field; it keeps its own
//! module, [`crate::ops::solver`] — the single named exception to the
//! "one module per produced container" rule.

pub mod consolidate;
pub mod divergence;
pub mod external_forces;
pub mod internal_forces;
pub mod mask;
pub mod merge;
pub mod methods;
pub mod positions;
pub mod restrict;

pub use consolidate::consolidate;
pub use divergence::divergence;
pub use external_forces::external_forces;
pub use internal_forces::{internal_forces, internal_forces_continuum};
pub use mask::{mask, mask_sub};
pub use merge::merge;
pub use positions::positions;
pub use restrict::{restrict, restrict_like};
