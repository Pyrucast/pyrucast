//! Les verbes que les quatre saveurs de champ portent à l'identique.
//!
//! Miroir de `src/ops/field/methods.rs`, qui joue ce rôle côté Rust. Ces
//! méthodes n'ont **pas de fonction libre** — ce sont des méthodes de conteneur
//! et rien d'autre — donc `#[py_op]` n'a rien à en dériver : elles passent par
//! des `macro_rules!`, comme les maths élémentaires et les verbes de
//! composantes.
//!
//! **Une forme unique**, sans exception à retenir : chaque macro est exportée et
//! reçoit la **liste** des saveurs qu'elle sert. Une liste de deux depuis ce
//! module pour les verbes ordinaires, une liste d'un seul depuis
//! `node_field.rs` et `element_field.rs` pour les **slots** (`__add__`,
//! `__pow__`, `__richcmp__`…) — ceux-là engendrent chez pyo3 un trampoline
//! `unsafe fn` qui en appelle un autre, et l'édition 2024 ne couvre plus
//! implicitement ce corps : l'avertissement `unsafe_op_in_unsafe_fn` tombe dès
//! que l'`impl` vit hors du module qui déclare le `#[pyclass]`.
//!
//! La liste plutôt qu'un type unique : elle laisse la documentation écrite **une
//! fois** là où un type par appel l'aurait fait recopier deux fois par verbe.
//!
//! Ce qui sépare les deux familles est écrit une fois dans chaque macro — le
//! trait (`Field` contre `SubField`) et l'accès (`self.inner` contre
//! `self.handle.read()`, ou `.write()` pour une mutation).
//!
//! **Pas de `PyResult` inutile** : `components`, `component_count` et
//! `component_index` rendent leur valeur nue, leurs homologues Rust ne pouvant
//! pas échouer. Le stub est identique — pyo3 traduit `T` et `PyResult<T>` de la
//! même façon — mais le lecteur cesse de se demander quand l'appel échoue.
//! `component_index` garde en revanche son `Option` : `None` y dit « cette
//! composante n'existe pas », qu'aucun indice sentinelle ne dirait.
//!
//! La documentation reste **distincte par famille** : l'opération sur un agrégat
//! porte sur les zones qui définissent la composante, celle sur un sous-champ
//! sur son support. Les deux coïncident — un agrégat se lit comme la
//! concaténation de ses zones, nœuds d'interface comptés autant de fois qu'ils
//! sont stockés — mais un texte commun ferait lire au propriétaire d'un
//! sous-champ un avertissement sur des zones multiples qu'il n'a pas.
//!
//! Les littéraux terminent leurs lignes par `\n\` et non par `\` seul : le
//! second mangerait le saut de ligne et rendrait un pavé d'un seul tenant dans
//! le stub `.pyi`, que les IDE affichent tel quel.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};

/// Les verbes de **lecture** des agrégats : la valeur est tenue en propre, le
/// trait `Field` opère dessus directement.
///
/// Quatre formes selon l'argument et la faillibilité : `optional:` (`min`,
/// `max` — la composante peut être omise), `named:` (`sum` — elle est exigée),
/// `components:` (sans argument, infaillible).
#[macro_export]
macro_rules! py_field_read {
    (optional: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                #[pyo3(signature = (component=None))]
                fn $nom(&self, component: Option<&str>) -> ::pyo3::PyResult<f64> {
                    use $crate::containers::field::Field;
                    Ok(Field::$nom(&self.inner, component)?)
                }
            }
        )+
    };
    (named: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, component: &str) -> ::pyo3::PyResult<f64> {
                    use $crate::containers::field::Field;
                    Ok(Field::$nom(&self.inner, component)?)
                }
            }
        )+
    };
    (components: [$($T:ident),+ $(,)?], $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn components(&self) -> Vec<String> {
                    use $crate::containers::field::Field;
                    Field::components(&self.inner)
                }
            }
        )+
    };
}

/// Les mêmes pour les **sous-conteneurs** : la valeur est lue à travers le
/// handle, et c'est `SubField` qui opère.
///
/// Deux formes de plus, sans contrepartie côté agrégat : `count:` et `index:`.
/// Un agrégat n'a ni nombre de composantes unique — ses zones peuvent en porter
/// des jeux différents — ni index global.
#[macro_export]
macro_rules! py_subfield_read {
    (optional: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                #[pyo3(signature = (component=None))]
                fn $nom(&self, component: Option<&str>) -> ::pyo3::PyResult<f64> {
                    use $crate::containers::field::SubField;
                    Ok(SubField::$nom(&*self.handle.read(), component)?)
                }
            }
        )+
    };
    (named: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, component: &str) -> ::pyo3::PyResult<f64> {
                    use $crate::containers::field::SubField;
                    Ok(SubField::$nom(&*self.handle.read(), component)?)
                }
            }
        )+
    };
    (components: [$($T:ident),+ $(,)?], $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn components(&self) -> Vec<String> {
                    use $crate::containers::field::SubField;
                    self.handle.read().components().to_vec()
                }
            }
        )+
    };
    (count: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self) -> usize {
                    use $crate::containers::field::SubField;
                    self.handle.read().$nom()
                }
            }
        )+
    };
    (index: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, name: &str) -> Option<usize> {
                    use $crate::containers::field::SubField;
                    self.handle.read().$nom(name)
                }
            }
        )+
    };
}

/// Les verbes d'**écriture** des agrégats : la mutation descend aux zones qui
/// définissent la composante, à travers le trait `Field`.
#[macro_export]
macro_rules! py_field_write {
    (scalar: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                    use $crate::containers::field::Field;
                    self.inner.$nom(component, scalar)?;
                    Ok(())
                }
            }
        )+
    };
}

/// Les mêmes pour les **sous-conteneurs**. Seule charpente à prendre un
/// **write** guard : la mutation est en place, sur la zone elle-même.
#[macro_export]
macro_rules! py_subfield_write {
    (scalar: [$($T:ident),+ $(,)?], $nom:ident, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                    use $crate::containers::field::SubField;
                    self.handle.write().$nom(component, scalar)?;
                    Ok(())
                }
            }
        )+
    };
}

/// Un opérateur arithmétique des agrégats : un renvoi vers `binary`, le
/// dispatcheur que chaque saveur définit dans son bloc inhérent.
///
/// Appelée depuis `node_field.rs` et `element_field.rs` avec une liste d'un
/// seul type, contrainte des slots oblige (voir l'en-tête de ce fichier).
#[macro_export]
macro_rules! py_field_transform {
    (op: [$($T:ident),+ $(,)?], $nom:ident, $f:expr, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<$T> {
                    self.binary(rhs, $f)
                }
            }
        )+
    };
    (pow: [$($T:ident),+ $(,)?], $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn __pow__(
                    &self,
                    exponent: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                    modulo: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                ) -> ::pyo3::PyResult<$T> {
                    use ::pyo3::types::PyAnyMethods;
                    if !modulo.is_none() {
                        return Err(::pyo3::exceptions::PyTypeError::new_err(
                            "field ** exponent does not support a modulo argument",
                        ));
                    }
                    self.binary(exponent, |a, b| a.powf(b))
                }
            }
        )+
    };
}

/// La même pour les **sous-conteneurs**, qui passent par `scalar_or_combine`.
#[macro_export]
macro_rules! py_subfield_transform {
    (op: [$($T:ident),+ $(,)?], $nom:ident, $f:expr, $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<$T> {
                    self.scalar_or_combine(rhs, $f)
                }
            }
        )+
    };
    (pow: [$($T:ident),+ $(,)?], $doc:literal) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T {
                #[doc = $doc]
                fn __pow__(
                    &self,
                    exponent: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                    modulo: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                ) -> ::pyo3::PyResult<$T> {
                    use ::pyo3::types::PyAnyMethods;
                    if !modulo.is_none() {
                        return Err(::pyo3::exceptions::PyTypeError::new_err(
                            "field ** exponent does not support a modulo argument",
                        ));
                    }
                    self.scalar_or_combine(exponent, |a, b| a.powf(b))
                }
            }
        )+
    };
}

// ─── Les verbes de lecture ──────────────────────────────────────────────────

py_field_read!(
    optional: [PyNodeField, PyElementField], min,
    "Smallest value of `component` across the zones defining it — or, called\n\
     without a component, the smallest value of the **whole** field, every\n\
     component of every zone pooled (see the sub-field's `min` for what\n\
     pooling means)."
);
py_subfield_read!(
    optional: [PySubNodeField, PySubElementField], min,
    "Smallest value of the named `component` — or, called without one, the\n\
     smallest value of the **whole** field, every component pooled. Pooling\n\
     reads the field as the flat list of its values: on components carrying\n\
     different units it answers \"the smallest number in there\", not a\n\
     physical quantity."
);

py_field_read!(
    optional: [PyNodeField, PyElementField], max,
    "Largest value of `component` across the zones defining it — or, called\n\
     without a component, the largest value of the **whole** field (see `min`)."
);
py_subfield_read!(
    optional: [PySubNodeField, PySubElementField], max,
    "Largest value of the named `component` — or, called without one, the\n\
     largest value of the **whole** field, every component pooled (see `min`)."
);

py_field_read!(
    named: [PyNodeField, PyElementField], sum,
    "Sum of `component` across the zones defining it (Σ over the whole field)\n\
     — the resultant of a nodal force field, one component at a time. A node\n\
     carried by several zones counts once per zone that stores it. Errors if\n\
     no zone defines the component."
);
py_subfield_read!(
    named: [PySubNodeField, PySubElementField], sum,
    "Sum of the named `component` over the support — Σ over the nodes, or over\n\
     the Gauss points for a field by elements. The resultant of a nodal force\n\
     field, one component at a time. An empty support sums to `0.0`."
);

py_field_read!(
    components: [PyNodeField, PyElementField],
    "Union of the zones' component names, first-seen order."
);
py_subfield_read!(
    components: [PySubNodeField, PySubElementField],
    "Component names, in order."
);

py_subfield_read!(
    count: [PySubNodeField, PySubElementField], component_count,
    "Number of components stored per node, or per Gauss point for a field by\n\
     elements."
);
py_subfield_read!(
    index: [PySubNodeField, PySubElementField], component_index,
    "Index of component `name`, or `None` if unknown — no default index would\n\
     say \"absent\" without being mistaken for a real one."
);

// ─── Les quatre mutateurs de composante ─────────────────────────────────────

py_field_write!(
    scalar: [PyNodeField, PyElementField], add_to_component,
    "Add `scalar` to `component` on every zone that defines it."
);
py_subfield_write!(
    scalar: [PySubNodeField, PySubElementField], add_to_component,
    "Add `scalar` to every value of `component` (in place)."
);

py_field_write!(
    scalar: [PyNodeField, PyElementField], sub_to_component,
    "Subtract `scalar` from `component` on every zone that defines it."
);
py_subfield_write!(
    scalar: [PySubNodeField, PySubElementField], sub_to_component,
    "Subtract `scalar` from every value of `component` (in place)."
);

py_field_write!(
    scalar: [PyNodeField, PyElementField], mul_to_component,
    "Multiply `component` by `scalar` on every zone that defines it."
);
py_subfield_write!(
    scalar: [PySubNodeField, PySubElementField], mul_to_component,
    "Multiply every value of `component` by `scalar` (in place)."
);

py_field_write!(
    scalar: [PyNodeField, PyElementField], div_to_component,
    "Divide `component` by `scalar` on every zone that defines it."
);
py_subfield_write!(
    scalar: [PySubNodeField, PySubElementField], div_to_component,
    "Divide every value of `component` by `scalar` (in place)."
);
