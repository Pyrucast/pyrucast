//! Measures — operators that consume containers and produce a **number**,
//! not a container.
//!
//! The "one module per produced container" rule is silent here by
//! construction: nothing is produced. What lands in this module is grouped
//! by activity, like [`crate::ops::geom`] (geometric queries) and
//! [`crate::ops::export`] (side effects).
//!
//! [`integral`](fn@integral) is the quadrature of a nodal field over a
//! finite-element space (Cast3M `INTG`); [`integral_element`](fn@integral_element)
//! is its per-element counterpart, which carries its own measure and needs
//! no space.

pub mod integral;

pub use integral::{integral, integral_element};
