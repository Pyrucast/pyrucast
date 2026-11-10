//! Operators that **produce** an
//! [`Evolution`](crate::containers::evolution::Evolution) from other
//! containers — a curve computed, not declared.
//!
//! An evolution is mostly **declared**, through the named constructors of the
//! type (`Evolution::from_scalars`, `from_node_fields`, …): the samples are
//! given, the evolution only holds them. The operators here compute their
//! samples from other containers:
//!
//! - [`along`](fn@along) — a field read along a line of points, one curve per
//!   component against the distance travelled.

pub mod along;
pub mod methods;

pub use along::along;
