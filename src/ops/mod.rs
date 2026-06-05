//! Themed operators — functions that take containers in and return
//! containers (or derived data) out.
//!
//! The directory is the home for everything that does not naturally
//! belong to a single container's inherent methods. The split is by
//! theme rather than by container, so a `gradient(mesh, field)` lives
//! next to `interp_to_gauss`, not in `mesh.rs` *or* `node_field.rs`.
//!
//! Themes:
//! - [`build`]: construct containers (fills, refinements, transforms).
//! - [`geom`]: geometric measures (bbox, centroid, area, jacobian
//!   helpers, normals).
//! - [`field`]: field-on-mesh operations (gradient, divergence,
//!   interpolation, projection, restriction).
//! - [`assemble`]: build a [`crate::containers::matrix::Matrix`] /
//!   [`crate::containers::node_field::NodeField`] from a `Model`. The per-physics
//!   integrands live under [`crate::models`]; this layer wires them
//!   together.
//! - [`behavior`]: integrate the constitutive law of a `Model` (Cast3m
//!   `COMP`) — the exact, possibly non-linear counterpart of the
//!   `assemble::stiffness` linearization.
//! - [`solver`]: solve `A · x = b` (currently a single dense LU
//!   back-end in [`solver::lu`]; sparse direct and iterative
//!   back-ends will live alongside it).
//!
//! Each sub-module starts empty — new operators should land here from
//! day one; legacy code migrates opportunistically.

pub mod assemble;
pub mod behavior;
pub mod build;
pub mod field;
pub mod geom;
pub mod mesher;
pub mod solver;
