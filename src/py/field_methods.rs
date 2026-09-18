//! Les verbes de champ dont l'appel sert **deux classes à la fois**.
//!
//! Les quatre mutateurs de composante (`add_`, `sub_`, `mul_`, `div_to_component`)
//! valent pour les deux saveurs d'une famille, et n'appartiennent donc à aucun
//! module de classe : leur appel vit ici, et sa documentation avec lui.
//!
//! Les macros qu'ils appellent sont dans `field_macros.rs`. Les autres verbes
//! de champ sont écrits à la main dans `node_field.rs` et `element_field.rs`,
//! ou y appellent une macro quand ce sont des slots.
//!
//! La documentation reste **distincte par famille** : l'opération sur un
//! agrégat porte sur les zones qui définissent la composante, celle sur un
//! sous-champ sur sa seule zone.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};

// ─── Les quatre mutateurs de composante ─────────────────────────────────────

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
