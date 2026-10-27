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
//!   [`boundary_transfer()`], [`interface_transfer()`];
//! - *solid mechanics* — [`elasticity()`], [`plasticity()`], [`damage()`]
//!   and their named shorthands;
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

/// Build a `Model` covering **every** subspace of `fespace`, one sub-model per
/// subspace — the shape of every operator on a finite-element space: those
/// [`physics_operator!`](crate::physics_operator) generates and the two generic
/// laws, [`plasticity()`] and [`damage()`].
///
/// Deliberately a function and not a macro: the closure is type-checked at its
/// definition, and go-to-definition still lands on real code.
///
/// `pub(crate)`, and it matters: `tests/python/test_mirror_completeness.py`
/// reads the `pub fn` of an `ops` module root and would demand a Python binding
/// for it. It is machinery, not an operator.
pub(crate) fn spanning(
    fespace: &FiniteElementSpace,
    mut make: impl FnMut(Handle<SubFiniteElementSpace>) -> Result<SubModel>,
) -> Result<Model> {
    let mut model = Model::empty();
    for zone in fespace {
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
pub mod embedded;
pub mod interface_transfer;
pub mod mpc;
pub mod plasticity;

pub use crate::models::bernoulli::bernoulli;
pub use crate::models::boundary_transfer::boundary_transfer;
pub use crate::models::damage::damage_tc::damage_tc;
pub use crate::models::damage::mazars::mazars;
pub use crate::models::damage::sic_sic::damage_sic_sic;
pub use crate::models::elasticity::elasticity;
pub use crate::models::fick::fick;
pub use crate::models::flux::flux;
pub use crate::models::heat_conduction::heat_conduction;
pub use crate::models::plasticity::drucker_prager::drucker_prager;
pub use crate::models::plasticity::gurson::gurson;
pub use crate::models::plasticity::ottosen::ottosen;
pub use crate::models::plasticity::viscous::creep_blackburn;
pub use crate::models::plasticity::viscous::creep_lemaitre;
pub use crate::models::plasticity::viscous::creep_norton;
pub use crate::models::plasticity::viscous::viscoplasticity_chaboche;
pub use crate::models::plasticity::viscous::viscoplasticity_lemaitre_chaboche;
pub use crate::models::plasticity::von_mises::plasticity_isotropic;
pub use crate::models::plasticity::von_mises::plasticity_perfect;
pub use crate::models::radiation::radiation;
pub use crate::models::shell::shell;
pub use crate::models::timoshenko::timoshenko;
pub use crate::models::truss::truss;
pub use contact::contact;
pub use damage::damage;
pub use dirichlet::dirichlet;
pub use embedded::embedded;
pub use interface_transfer::interface_transfer;
pub use mpc::mpc;
pub use plasticity::plasticity;
