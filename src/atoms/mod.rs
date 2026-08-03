//! Atoms — the indivisible types of pyrucast.
//!
//! An atom is a type of which **no part is of its own nature**: half a
//! [`Node`] is not a node, half a [`Cell`] is a bag of nodes rather than a
//! cell, half an [`ElementType`] is nothing at all. That is exactly what
//! sets them apart from the divisible operands of [`crate::containers`],
//! whose halves are still meshes, still fields, still matrices.
//!
//! The distinction is not decorative — it is the one the API convention
//! rests on: **only a container can be the subject of an operator**. An
//! atom composes (`node | node` yields a `Mesh`) but never decomposes, so
//! it appears as an *argument* and never as a `self`. Reading the module
//! tree is therefore enough to know where an operation may live.
//!
//! Two kinds of atom live here, and they are not quite alike:
//!
//! - **designators** — [`Node`], [`Cell`], [`Element`]: they carry an
//!   identity (a store handle, or an index into a container) and name one
//!   piece of something bigger;
//! - **values** — [`ElementType`], [`Point2`]/[`Point3`],
//!   [`Vector2`]/[`Vector3`], [`RgbColor`], [`Band`]: `Copy`, identity-free,
//!   they *are* their content.
//!
//! [`crate::coords::Coords`] belongs to neither: it is the coordinate
//! **store**, and it lives at the crate root next to [`crate::store`].

pub mod band;
pub mod cell;
pub mod color;
pub mod element;
pub mod element_type;
pub mod node;
pub mod point;

// Flat re-exports: an atom is reachable as `atoms::Node`, `atoms::Cell`, …
// rather than through its defining sub-module.
pub use band::Band;
pub use cell::{Cell, CellIter};
pub use color::RgbColor;
pub use element::{Element, ElementIter};
pub use element_type::ElementType;
pub use node::{Node, NodeId};
pub use point::{Point2, Point3, Vector2, Vector3};
