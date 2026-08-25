//! Operators that **produce** a [`Model`](crate::containers::model::Model) —
//! the physics declarations, one per family.
//!
//! Each operator consumes the *parent* support (a `FiniteElementSpace`, or the
//! meshes a constraint relates) and returns a `Model` spanning **all** of it:
//! one sub-model per subspace. A single-subspace support gives the unit case;
//! several give one zone each. Heterogeneous physics compose with
//! [`Aggregate::union`](crate::aggregate::Aggregate::union) (Python `|`), never
//! by building a `SubModel` and attaching it by hand.
//!
//! - *field physics* — [`heat_conduction()`], [`fick()`], [`radiation()`],
//!   [`follower_pressure()`], [`boundary_transfer()`],
//!   [`interface_transfer()`];
//! - *solid mechanics* — [`elasticity()`], [`plasticity_with_law()`],
//!   [`damage_with_law()`] and their named shorthands;
//! - *structural elements* — [`truss()`], [`bernoulli()`], [`timoshenko()`],
//!   [`shell()`];
//! - *constraints* — [`dirichlet()`], [`mpc()`], [`embedded()`], [`contact()`].
//!
//! None of these is exposed as a method: their first argument is the
//! **support** the model spans, not a subject being transformed (see
//! `CONVENTIONS.md` § « Le verbe exposé aussi en méthode », condition 1).

pub mod bernoulli;
pub mod boundary_transfer;
pub mod contact;
pub mod damage;
pub mod dirichlet;
pub mod elasticity;
pub mod embedded;
pub mod fick;
pub mod follower_pressure;
pub mod heat_conduction;
pub mod interface_transfer;
pub mod mpc;
pub mod plasticity;
pub mod radiation;
pub mod shell;
pub mod timoshenko;
pub mod truss;

pub use bernoulli::bernoulli;
pub use boundary_transfer::boundary_transfer;
pub use contact::contact;
pub use damage::{damage_with_law, mazars};
pub use dirichlet::dirichlet;
pub use elasticity::{elasticity, elasticity_with_symmetry};
pub use embedded::embedded;
pub use fick::{fick, fick_with_symmetry};
pub use follower_pressure::follower_pressure;
pub use heat_conduction::{heat_conduction, heat_conduction_with_symmetry};
pub use interface_transfer::interface_transfer;
pub use mpc::mpc;
pub use plasticity::{plasticity_perfect, plasticity_with_law};
pub use radiation::radiation;
pub use shell::shell;
pub use timoshenko::timoshenko;
pub use truss::truss;
