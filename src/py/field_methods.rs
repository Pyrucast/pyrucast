//! Les verbes que les quatre saveurs de champ portent à l'identique.
//!
//! Miroir de `src/ops/field/methods.rs`, qui joue ce rôle côté Rust. Ces
//! méthodes n'ont **pas de fonction libre** — ce sont des méthodes de conteneur
//! et rien d'autre — donc `#[py_op]` n'a rien à en dériver : elles passent par
//! des `macro_rules!`, comme les maths élémentaires et les verbes de
//! composantes.
//!
//! **Une macro par famille d'accès**, et chacune connaît ses deux types plutôt
//! que de les recevoir : un appel les sert tous les deux. Ce qui les sépare est
//! écrit une fois dans la macro — le trait (`Field` contre `SubField`) et
//! l'accès (`self.inner` contre `self.handle.read()`).
//!
//! La documentation, elle, reste **distincte par famille** : l'opération sur un
//! agrégat porte sur les zones qui définissent la composante, celle sur un
//! sous-champ sur son support. Les deux coïncident — un agrégat se lit comme la
//! concaténation de ses zones, nœuds d'interface comptés autant de fois qu'ils
//! sont stockés — mais un texte commun ferait lire au propriétaire d'un
//! sous-champ un avertissement sur des zones multiples qu'il n'a pas.
//!
//! Les littéraux terminent leurs lignes par `\n\` et non par `\` seul : le
//! second mangerait le saut de ligne et rendrait un pavé d'un seul tenant dans
//! le stub `.pyi`, que les IDE affichent tel quel.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::prelude::*;

/// Les verbes de **lecture** des deux agrégats : la valeur est tenue en propre,
/// le trait `Field` opère dessus directement.
///
/// Trois formes, selon l'argument : `optional:` pour un verbe dont la
/// composante peut être omise (`min`, `max`), `named:` pour un verbe qui l'exige
/// (`sum`), `components:` pour l'accesseur sans argument.
macro_rules! py_field_read {
    (optional: $nom:ident, $doc:literal) => {
        py_field_read!(@one PyNodeField, optional: $nom, $doc);
        py_field_read!(@one PyElementField, optional: $nom, $doc);
    };
    (named: $nom:ident, $doc:literal) => {
        py_field_read!(@one PyNodeField, named: $nom, $doc);
        py_field_read!(@one PyElementField, named: $nom, $doc);
    };
    (components: $doc:literal) => {
        py_field_read!(@components PyNodeField, $doc);
        py_field_read!(@components PyElementField, $doc);
    };

    (@one $T:ident, optional: $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            #[pyo3(signature = (component=None))]
            fn $nom(&self, component: Option<&str>) -> PyResult<f64> {
                use crate::containers::field::Field;
                Ok(Field::$nom(&self.inner, component)?)
            }
        }
    };
    (@one $T:ident, named: $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self, component: &str) -> PyResult<f64> {
                use crate::containers::field::Field;
                Ok(Field::$nom(&self.inner, component)?)
            }
        }
    };
    (@components $T:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn components(&self) -> PyResult<Vec<String>> {
                use crate::containers::field::Field;
                Ok(Field::components(&self.inner))
            }
        }
    };
}

/// Les mêmes, pour les deux **sous-conteneurs** : la valeur est lue à travers le
/// handle, et c'est `SubField` qui opère.
macro_rules! py_subfield_read {
    (optional: $nom:ident, $doc:literal) => {
        py_subfield_read!(@one PySubNodeField, optional: $nom, $doc);
        py_subfield_read!(@one PySubElementField, optional: $nom, $doc);
    };
    (named: $nom:ident, $doc:literal) => {
        py_subfield_read!(@one PySubNodeField, named: $nom, $doc);
        py_subfield_read!(@one PySubElementField, named: $nom, $doc);
    };
    (components: $doc:literal) => {
        py_subfield_read!(@components PySubNodeField, $doc);
        py_subfield_read!(@components PySubElementField, $doc);
    };
    (count: $nom:ident, $doc_nodal:literal, $doc_element:literal) => {
        py_subfield_read!(@count PySubNodeField, $nom, $doc_nodal);
        py_subfield_read!(@count PySubElementField, $nom, $doc_element);
    };
    (index: $nom:ident, $doc:literal) => {
        py_subfield_read!(@index PySubNodeField, $nom, $doc);
        py_subfield_read!(@index PySubElementField, $nom, $doc);
    };

    (@one $T:ident, optional: $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            #[pyo3(signature = (component=None))]
            fn $nom(&self, component: Option<&str>) -> PyResult<f64> {
                use crate::containers::field::SubField;
                Ok(SubField::$nom(&*self.handle.read(), component)?)
            }
        }
    };
    (@one $T:ident, named: $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self, component: &str) -> PyResult<f64> {
                use crate::containers::field::SubField;
                Ok(SubField::$nom(&*self.handle.read(), component)?)
            }
        }
    };
    (@components $T:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn components(&self) -> PyResult<Vec<String>> {
                use crate::containers::field::SubField;
                Ok(self.handle.read().components().to_vec())
            }
        }
    };
    (@count $T:ident, $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self) -> PyResult<usize> {
                use crate::containers::field::SubField;
                Ok(self.handle.read().$nom())
            }
        }
    };
    (@index $T:ident, $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self, name: &str) -> PyResult<Option<usize>> {
                use crate::containers::field::SubField;
                Ok(self.handle.read().$nom(name))
            }
        }
    };
}

/// Les verbes d'**écriture** des deux agrégats : la mutation descend aux zones
/// qui définissent la composante, à travers le trait `Field`.
macro_rules! py_field_write {
    (scalar: $nom:ident, $doc:literal) => {
        py_field_write!(@one PyNodeField, $nom, $doc);
        py_field_write!(@one PyElementField, $nom, $doc);
    };
    (@one $T:ident, $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self, component: &str, scalar: f64) -> PyResult<()> {
                use crate::containers::field::Field;
                self.inner.$nom(component, scalar)?;
                Ok(())
            }
        }
    };
}

/// Les mêmes pour les deux **sous-conteneurs**. Seule charpente à prendre un
/// **write** guard : la mutation est en place, sur la zone elle-même.
macro_rules! py_subfield_write {
    (scalar: $nom:ident, $doc:literal) => {
        py_subfield_write!(@one PySubNodeField, $nom, $doc);
        py_subfield_write!(@one PySubElementField, $nom, $doc);
    };
    (@one $T:ident, $nom:ident, $doc:literal) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pymethods]
        impl $T {
            #[doc = $doc]
            fn $nom(&self, component: &str, scalar: f64) -> PyResult<()> {
                use crate::containers::field::SubField;
                self.handle.write().$nom(component, scalar)?;
                Ok(())
            }
        }
    };
}

// ─── Les quatre verbes de lecture ───────────────────────────────────────────
//
// Huit appels pour seize méthodes. Les textes sont ceux d'avant, chacun
// complété de ce que l'autre sorte disait et qui valait pour les deux : la
// somme mentionne les deux supports (nœuds, points de Gauss) et son cas
// d'erreur, `min` renvoie au sous-champ sans nommer une sorte.

py_field_read!(
    optional: min,
    "Smallest value of `component` across the zones defining it — or, called\n\
     without a component, the smallest value of the **whole** field, every\n\
     component of every zone pooled (see the sub-field's `min` for what\n\
     pooling means)."
);
py_subfield_read!(
    optional: min,
    "Smallest value of the named `component` — or, called without one, the\n\
     smallest value of the **whole** field, every component pooled. Pooling\n\
     reads the field as the flat list of its values: on components carrying\n\
     different units it answers \"the smallest number in there\", not a\n\
     physical quantity."
);

py_field_read!(
    optional: max,
    "Largest value of `component` across the zones defining it — or, called\n\
     without a component, the largest value of the **whole** field (see `min`)."
);
py_subfield_read!(
    optional: max,
    "Largest value of the named `component` — or, called without one, the\n\
     largest value of the **whole** field, every component pooled (see `min`)."
);

py_field_read!(
    named: sum,
    "Sum of `component` across the zones defining it (Σ over the whole field)\n\
     — the resultant of a nodal force field, one component at a time. A node\n\
     carried by several zones counts once per zone that stores it. Errors if\n\
     no zone defines the component."
);
py_subfield_read!(
    named: sum,
    "Sum of the named `component` over the support — Σ over the nodes, or over\n\
     the Gauss points for a field by elements. The resultant of a nodal force\n\
     field, one component at a time. An empty support sums to `0.0`."
);

py_field_read!(components: "Union of the zones' component names, first-seen order.");
py_subfield_read!(components: "Component names, in order.");

// ─── Les deux accesseurs propres aux sous-conteneurs ────────────────────────
//
// Sans contrepartie côté agrégat : un agrégat n'a pas de nombre de composantes
// unique — ses zones peuvent en porter des jeux différents — ni d'index global.
// C'est ce que permet une macro par famille : servir une famille seule, là où
// une macro à quatre saveurs n'aurait pas pu les prendre.
//
// `get` et `value` restent écrits à la main : leur clé diffère par sorte,
// `(node_idx, comp_idx)` contre `(cell, gauss, comp)`.

py_subfield_read!(
    count: component_count,
    "Number of components stored per node.",
    "Number of components stored per point."
);
py_subfield_read!(index: component_index, "Index of component `name`, or `None` if unknown.");

// ─── Les quatre mutateurs de composante ─────────────────────────────────────
//
// Même signature partout, et une documentation qui ne varie qu'avec le mode
// d'accès : l'agrégat descend la mutation aux zones qui définissent la
// composante, le sous-champ l'applique en place sur la sienne.
//
// `set`, `set_value` et `__setitem__` restent écrits à la main : leur clé
// diffère par sorte, comme celle de `get` et `value`.

py_field_write!(
    scalar: add_to_component,
    "Add `scalar` to `component` on every zone that defines it."
);
py_subfield_write!(
    scalar: add_to_component,
    "Add `scalar` to every value of `component` (in place)."
);

py_field_write!(
    scalar: sub_to_component,
    "Subtract `scalar` from `component` on every zone that defines it."
);
py_subfield_write!(
    scalar: sub_to_component,
    "Subtract `scalar` from every value of `component` (in place)."
);

py_field_write!(
    scalar: mul_to_component,
    "Multiply `component` by `scalar` on every zone that defines it."
);
py_subfield_write!(
    scalar: mul_to_component,
    "Multiply every value of `component` by `scalar` (in place)."
);

py_field_write!(
    scalar: div_to_component,
    "Divide `component` by `scalar` on every zone that defines it."
);
py_subfield_write!(
    scalar: div_to_component,
    "Divide every value of `component` by `scalar` (in place)."
);
