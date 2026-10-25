//! Delegation methods — the "subject" face of this module's operators.
//!
//! See `CONVENTIONS.md` § "The verb also exposed as a method": the free
//! function stays the canonical form and carries the documentation; these
//! methods contain no logic. The `impl`s live here rather than in
//! `containers/` — a container must not depend on an operator, and Rust allows
//! an inherent `impl` in any module of the crate that defines the type.
//!
//! **Not** exposed here, for want of meaning for every instance of the type:
//! `internal_forces` and `external_forces` (their subject is the model, not a
//! field), and `merge`, which is symmetric — `a | b` is already its form.

use crate::containers::element_field::ElementField;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::Result;

impl NodeField {
    /// See [`node_field::consolidate`](fn@crate::ops::node_field::consolidate).
    pub fn consolidate(&self) -> Result<NodeField> {
        crate::ops::node_field::consolidate(self)
    }

    /// See [`node_field::restrict`](fn@crate::ops::node_field::restrict).
    pub fn restrict(&self, mesh: &Mesh) -> Result<NodeField> {
        crate::ops::node_field::restrict(self, mesh)
    }

    /// See [`node_field::restrict_like`](fn@crate::ops::node_field::restrict_like).
    pub fn restrict_like(&self, target: &NodeField) -> Result<NodeField> {
        crate::ops::node_field::restrict_like(self, target)
    }
}

impl ElementField {
    /// See [`node_field::divergence`](fn@crate::ops::node_field::divergence).
    pub fn divergence(&self, prefix: &str) -> Result<NodeField> {
        crate::ops::node_field::divergence(self, prefix)
    }
}

impl Mesh {
    /// See [`node_field::positions`](fn@crate::ops::node_field::positions).
    pub fn positions(&self, components: Option<Vec<String>>) -> Result<NodeField> {
        crate::ops::node_field::positions(self, components)
    }
}
