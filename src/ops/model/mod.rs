//! Operators that **produce** a [`Model`] —
//! the physics declarations, one per family.
//!
//! Each operator consumes the *parent* support (a `FiniteElementSpace`, or the
//! meshes a constraint relates) and returns a `Model` spanning **all** of it:
//! one sub-model per subspace. A single-subspace support gives the unit case;
//! several give one zone each. Heterogeneous physics compose with
//! [`Aggregate::union`] (Python `|`), never
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

use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::handle::Handle;

/// Build a `Model` covering **every** subspace of `fes`, one sub-model per
/// subspace — the shape twelve of these operators share.
///
/// Deliberately a function and not a macro: the closure is type-checked at its
/// definition, and go-to-definition still lands on real code.
///
/// `pub(crate)`, and it matters: `tests/python/test_mirror_completeness.py`
/// reads the `pub fn` of an `ops` module root and would demand a Python binding
/// for it. It is machinery, not an operator.
pub(crate) fn spanning(
    fes: &FiniteElementSpace,
    mut make: impl FnMut(Handle<SubFiniteElementSpace>) -> Result<SubModel>,
) -> Result<Model> {
    let mut model = Model::empty();
    for zone in fes {
        model.add_sub(Handle::new(make(zone.clone())?))?;
    }
    Ok(model)
}

/// Build a `Model` holding exactly one sub-model — the shape of the four
/// constraints, which are carried by user-supplied meshes rather than by a
/// finite-element space, and so have no subspaces to sweep.
pub(crate) fn single(sub: SubModel) -> Result<Model> {
    let mut model = Model::empty();
    model.add_sub(Handle::new(sub))?;
    Ok(model)
}

pub mod contact;
pub mod damage;
pub mod dirichlet;
pub mod elasticity;
pub mod embedded;
pub mod fick;
pub mod heat_conduction;
pub mod interface_transfer;
pub mod mpc;
pub mod plasticity;

pub use crate::models::bernoulli::bernoulli;
pub use crate::models::boundary_transfer::boundary_transfer;
pub use crate::models::follower_pressure::follower_pressure;
pub use crate::models::radiation::radiation;
pub use crate::models::shell::shell;
pub use crate::models::timoshenko::timoshenko;
pub use crate::models::truss::truss;
pub use contact::contact;
pub use damage::{damage_with_law, mazars};
pub use dirichlet::dirichlet;
pub use elasticity::{elasticity, elasticity_with_symmetry};
pub use embedded::embedded;
pub use fick::{fick, fick_with_symmetry};
pub use heat_conduction::{heat_conduction, heat_conduction_with_symmetry};
pub use interface_transfer::interface_transfer;
pub use mpc::mpc;
pub use plasticity::{plasticity_perfect, plasticity_with_law};
