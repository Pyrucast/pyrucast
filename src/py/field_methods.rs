//! The field verbs whose call serves **two classes at once**.
//!
//! The four component mutators (`add_`, `sub_`, `mul_`, `div_to_component`) hold
//! for both flavours of a family, and therefore belong to no class module: their
//! call lives here, and its documentation with it.
//!
//! The macros they call are in `field_macros.rs`. The other field verbs are
//! written by hand in `node_field.rs` and `element_field.rs`, or call a macro
//! there when they are slots.
//!
//! The documentation stays **distinct per family**: the operation on an aggregate
//! bears on the zones defining the component, the one on a sub-field on its
//! single zone.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};

// ─── The four component mutators ────────────────────────────────────────────

crate::impl_field_mutator_pymethod! {
    /// Add `scalar` to `component` on every zone that defines it.
    [PyNodeField, PyElementField], add_to_component
}
crate::impl_subfield_mutator_pymethod! {
    /// Add `scalar` to every value of `component` (in place).
    [PySubNodeField, PySubElementField], add_to_component
}

crate::impl_field_mutator_pymethod! {
    /// Subtract `scalar` from `component` on every zone that defines it.
    [PyNodeField, PyElementField], sub_to_component
}
crate::impl_subfield_mutator_pymethod! {
    /// Subtract `scalar` from every value of `component` (in place).
    [PySubNodeField, PySubElementField], sub_to_component
}

crate::impl_field_mutator_pymethod! {
    /// Multiply `component` by `scalar` on every zone that defines it.
    [PyNodeField, PyElementField], mul_to_component
}
crate::impl_subfield_mutator_pymethod! {
    /// Multiply every value of `component` by `scalar` (in place).
    [PySubNodeField, PySubElementField], mul_to_component
}

crate::impl_field_mutator_pymethod! {
    /// Divide `component` by `scalar` on every zone that defines it.
    [PyNodeField, PyElementField], div_to_component
}
crate::impl_subfield_mutator_pymethod! {
    /// Divide every value of `component` by `scalar` (in place).
    [PySubNodeField, PySubElementField], div_to_component
}
