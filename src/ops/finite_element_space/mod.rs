//! Operators that **produce** a
//! [`FiniteElementSpace`](crate::containers::finite_element_space::FiniteElementSpace)
//! from another: [`restrict`](fn@restrict), a space cut down to the cells of a
//! mesh, each subspace keeping its interpolation and its quadrature.

pub mod methods;
pub mod restrict;

pub use restrict::restrict;
