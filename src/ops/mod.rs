//! Operators — the functions that consume containers and produce
//! containers (or derived data).
//!
//! # How this directory is split
//!
//! **A module gathers the operators that produce the same container, and
//! is named after it.** An operation is filed by its *output*, never by
//! its input: `gradient(field, fespace)` produces an `ElementField`, so it
//! lives in [`element_field`] — next to `deformation` and
//! `interp_to_gauss`, not in [`mesh`] nor in [`node_field`].
//!
//! - [`mesh`] — meshers, transforms, selections: everything yielding a `Mesh`.
//! - [`model`] — the physics declarations: everything yielding a `Model`.
//! - [`node_field`] — nodal derivations and nodal assembly.
//! - [`element_field`] — kinematics, material data, constitutive law.
//! - [`matrix`] — the assemblers proper (`stiffness`, `mass`, `tangent`, …).
//! - [`coords`] — the two operators that write back into the coordinates.
//!
//! A third case: the **generic** operator, whose product is a container —
//! always — but not a *determined* one. `abs` yields a `NodeField` or an
//! `ElementField` depending on what it is given, so the rule does not select
//! one module. Those are grouped by domain instead:
//!
//! - [`field`] — anything that takes a field and gives back one of the same
//!   flavour: the value-band mask, component filtering and renaming, and the
//!   element-wise scalar maths.
//!
//! Operators that produce **no** container are grouped by activity, the
//! product rule having nothing to say about them:
//!
//! - [`measure`] — integrals and other reductions to a number;
//! - [`geom`] — geometric queries (locate, project, nearest);
//! - [`export`] — writing to external formats.
//!
//! # The one exception
//!
//! [`solver`] produces a `NodeField` and would belong to [`node_field`].
//! It keeps its own name because **several distinct families produce a
//! nodal field** — derivation, assembly, resolution — and only resolution
//! is looked up by its own name. It is the single module named after an
//! activity while producing a container, and it is meant to stay the only
//! one.
//!
//! # Two corollaries worth remembering
//!
//! A module never holds two operators differing only by the container they
//! act on: the qualifier belongs to the module name, not to the function
//! name. Hence three fusions named [`mesh::consolidate`](fn@mesh::consolidate),
//! [`node_field::consolidate`](fn@node_field::consolidate) and [`element_field::consolidate`](fn@element_field::consolidate), rather
//! than three suffixed functions in one place.
//!
//! Nothing here produces a `FiniteElementSpace` or an `Evolution`, and that
//! is not an oversight: those are **declared** through named constructors on
//! the type itself, never built by transformation. A `Model` is not in that
//! company: its declarations form a *catalogue* of physics — two dozen of
//! them and growing — which is a module's job, not a type's surface. They
//! live in [`model`], filed by product like every other operator.

pub mod coords;
pub mod element_field;
pub mod export;
pub mod field;
pub mod geom;
pub mod matrix;
pub mod measure;
pub mod mesh;
pub mod model;
pub mod node_field;
pub mod solver;

// Shared assembly machinery, used by the matrix assemblers and the nodal
// ones alike (`node_field::flux`, `node_field::internal_forces`). Not a
// theme — no operator lives here.
pub mod coloring;
pub mod scatter;
