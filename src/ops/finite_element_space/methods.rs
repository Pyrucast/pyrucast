//! Delegation methods — the "subject" face of this module's operators.
//!
//! A free function meeting the three conditions of `CONVENTIONS.md`
//! (§ "The verb also exposed as a method") is **also** exposed as a method of
//! its subject, to allow chaining. These methods contain **no** logic: they
//! call the free function, receiver included. The free function stays the
//! canonical form and carries the documentation.

use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::mesh::Mesh;

impl FiniteElementSpace {
    /// See [`finite_element_space::restrict`](fn@crate::ops::finite_element_space::restrict).
    pub fn restrict(&self, mesh: &Mesh) -> FiniteElementSpace {
        crate::ops::finite_element_space::restrict(self, mesh)
    }
}
