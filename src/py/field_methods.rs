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
//! **Trois macros pour les deux familles** — `py_field_read!`,
//! `py_field_write!`, `py_field_transform!` —, dont chaque appel commence par
//! le nom de la famille servie : `Field` pour les agrégats, `SubField` pour les
//! sous-conteneurs. C'est aussi le trait qui opère. Ce qui sépare les deux
//! familles (l'accès `self.inner` contre `self.handle.read()` ou `.write()`,
//! l'emballage du résultat, le nom du dispatcheur…) est écrit une seule fois,
//! dans la table `py_field_family!`.
//!
//! Les appels, eux, restent un par famille : leurs documentations diffèrent
//! légitimement (voir plus bas). Fusionner aussi les bras entre eux
//! (`optional:`, `named:`, `count:`, `index:` ne diffèrent que par arguments et
//! retour) ferait de la macro un mini-langage, plus mutualisé et moins lisible
//! au site d'appel — refusé.
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
//! **La documentation s'écrit en `///`**, en tête d'appel, comme sur un item :
//! chaque macro la capture ligne à ligne (`$(#[doc = $doc:literal])*`), et
//! chaque ligne reste un vrai littéral pour pyo3 comme pour pyo3-stub-gen.
//! Les appels prennent donc des accolades, et rustfmt ne les reformate pas.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};

/// Tout ce qui sépare les deux familles, et rien d'autre : une ligne par
/// aspect et par famille. `Field` désigne les agrégats, qui tiennent leur
/// valeur en propre (`self.inner`) ; `SubField` les sous-conteneurs, qui la
/// lisent à travers leur handle.
///
/// Les trois macros publiques y renvoient au lieu de dédoubler leurs bras.
/// Elles lui passent `self` **depuis leur propre corps** : `self` est
/// hygiénique, et seul un `self` écrit par la macro qui déclare la méthode
/// désigne son receveur — venu du site d'appel, il ne compilerait pas
/// (`E0424`).
#[doc(hidden)]
#[macro_export]
macro_rules! py_field_family {
    // La valeur, empruntée en lecture — ou en écriture, pour une mutation en place.
    (Field, read, $s:tt) => {
        &$s.inner
    };
    (SubField, read, $s:tt) => {
        &*$s.handle.read()
    };
    (Field, write, $s:tt) => {
        $s.inner
    };
    (SubField, write, $s:tt) => {
        $s.handle.write()
    };
    // Un résultat emballé dans la saveur du receveur (`Self`).
    (Field, wrap, $T:ident, $e:expr) => {
        $T { inner: $e }
    };
    (SubField, wrap, $T:ident, $e:expr) => {
        $T {
            handle: $crate::handle::Handle::new($e),
        }
    };
    // Le dispatcheur des opérateurs, que chaque saveur définit dans son bloc
    // inhérent.
    (Field, combine, $s:tt, $rhs:expr, $f:expr) => {
        $s.binary($rhs, $f)
    };
    (SubField, combine, $s:tt, $rhs:expr, $f:expr) => {
        $s.scalar_or_combine($rhs, $f)
    };
    // `SubField` nomme `select_components` ce que `Field` appelle
    // `filter_components` — la seule divergence de nom entre les deux traits,
    // absorbée ici plutôt que laissée au lecteur.
    (Field, filter, $s:tt, $names:expr) => {
        $s.inner.filter_components($names)
    };
    (SubField, filter, $s:tt, $names:expr) => {
        $s.handle.read().select_components($names)
    };
    // Le masque : `mask_sub` ne rend pas de `Result` — la zone est seule, il
    // n'y a pas d'agrégat à reconstruire, donc rien à refuser.
    (Field, mask, $op:path, $s:tt, $band:expr) => {
        $op(&$s.inner, $band, None)?
    };
    (SubField, mask, $op:path, $s:tt, $band:expr) => {
        $op(&$s.handle.read(), $band, None)
    };
}

/// Pose un même bloc de méthodes sur chaque type de la liste. `stub` le
/// décore pour pyo3-stub-gen, `bare` non (voir le bras `richcmp:`).
///
/// C'est ce détour qui permet d'écrire la documentation en `///` au site
/// d'appel. Une doc capturée ligne à ligne (`$(#[doc = $doc:literal])*`) ne
/// peut pas se répéter à l'intérieur de la boucle sur les types — rustc
/// refuse deux répétitions de longueurs différentes au même niveau. Le bloc
/// entier, lui, arrive ici comme **un seul** `tt`, que la boucle recopie
/// librement ; et son corps nomme son type `Self`, puisqu'il ne connaît pas
/// le `$T` qui le portera.
#[doc(hidden)]
#[macro_export]
macro_rules! py_field_impl {
    (stub [$($T:ident),+ $(,)?] $corps:tt) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T $corps
        )+
    };
    (bare [$($T:ident),+ $(,)?] $corps:tt) => {
        $(
            #[::pyo3::pymethods]
            impl $T $corps
        )+
    };
}

/// Les verbes de **lecture** des deux familles. Après la documentation, le
/// premier jeton nomme la famille — `Field` ou `SubField`, le trait qui opère.
///
/// ```ignore
/// py_field_read! {
///     /// Sum of `component` across the zones defining it.
///     Field named: [PyNodeField, PyElementField], sum
/// }
/// ```
///
/// Trois formes communes selon l'argument et la faillibilité : `optional:`
/// (`min`, `max` — la composante peut être omise), `named:` (`sum` — elle est
/// exigée), `components:` (sans argument, infaillible). Deux propres aux
/// sous-conteneurs, `count:` et `index:` : un agrégat n'a ni nombre de
/// composantes unique — ses zones peuvent en porter des jeux différents — ni
/// index global.
#[macro_export]
macro_rules! py_field_read {
    ($(#[doc = $doc:literal])* $F:ident optional: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            #[pyo3(signature = (component=None))]
            fn $nom(&self, component: Option<&str>) -> ::pyo3::PyResult<f64> {
                use $crate::containers::field::$F;
                Ok($F::$nom($crate::py_field_family!($F, read, self), component)?)
            }
        });
    };
    ($(#[doc = $doc:literal])* $F:ident named: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str) -> ::pyo3::PyResult<f64> {
                use $crate::containers::field::$F;
                Ok($F::$nom($crate::py_field_family!($F, read, self), component)?)
            }
        });
    };
    ($(#[doc = $doc:literal])* $F:ident components: [$($T:ident),+ $(,)?]) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn components(&self) -> Vec<String> {
                use $crate::containers::field::$F;
                // `Field` rend un `Vec`, `SubField` une tranche : `into`
                // convient aux deux.
                $F::components($crate::py_field_family!($F, read, self)).into()
            }
        });
    };
    ($(#[doc = $doc:literal])* SubField count: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> usize {
                use $crate::containers::field::SubField;
                self.handle.read().$nom()
            }
        });
    };
    ($(#[doc = $doc:literal])* SubField index: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, name: &str) -> Option<usize> {
                use $crate::containers::field::SubField;
                self.handle.read().$nom(name)
            }
        });
    };
}

/// Les verbes d'**écriture** des deux familles : sur un agrégat, la mutation
/// descend aux zones qui définissent la composante ; sur un sous-conteneur,
/// elle a lieu en place, sous un **write** guard — la seule charpente à en
/// prendre un.
#[macro_export]
macro_rules! py_field_write {
    ($(#[doc = $doc:literal])* $F:ident scalar: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                use $crate::containers::field::$F;
                $crate::py_field_family!($F, write, self).$nom(component, scalar)?;
                Ok(())
            }
        });
    };
}

/// La charpente « lire, puis emballer dans le même type », pour les deux
/// familles, en cinq bras : `op:` et `pow:` renvoient au dispatcheur de la
/// saveur ; `unary:` applique une math élémentaire de `ops::field` ;
/// `components:` filtre et renomme ; `richcmp:` rend un masque.
///
/// `op:`, `pow:` et `richcmp:` sont des **slots**, appelés depuis
/// `node_field.rs` et `element_field.rs` avec une liste d'un seul type (voir
/// l'en-tête de ce fichier). Les deux autres sont des méthodes ordinaires, que
/// rien n'oblige à expanser chez le `pyclass` : `py/ops/field.rs` les appelle
/// avec les deux saveurs à la fois, depuis les chapeaux qui portent aussi la
/// fonction libre.
///
/// `components:` porte deux méthodes, donc deux documentations, chacune
/// placée devant le nom de la méthode qu'elle décrit.
#[macro_export]
macro_rules! py_field_transform {
    ($(#[doc = $doc:literal])* $F:ident op: [$($T:ident),+ $(,)?], $nom:ident, $f:expr) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                $crate::py_field_family!($F, combine, self, rhs, $f)
            }
        });
    };
    ($(#[doc = $doc:literal])* $F:ident pow: [$($T:ident),+ $(,)?]) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn __pow__(
                &self,
                exponent: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                modulo: &::pyo3::Bound<'_, ::pyo3::PyAny>,
            ) -> ::pyo3::PyResult<Self> {
                use ::pyo3::types::PyAnyMethods;
                if !modulo.is_none() {
                    return Err(::pyo3::exceptions::PyTypeError::new_err(
                        "field ** exponent does not support a modulo argument",
                    ));
                }
                $crate::py_field_family!($F, combine, self, exponent, |a, b| a.powf(b))
            }
        });
    };
    ($(#[doc = $doc:literal])* $F:ident unary: [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> ::pyo3::PyResult<Self> {
                let out = $crate::ops::field::$nom($crate::py_field_family!($F, read, self))?;
                Ok($crate::py_field_family!($F, wrap, Self, out))
            }
        });
    };
    (
        $F:ident components: [$($T:ident),+ $(,)?]
        $(#[doc = $doc_filter:literal])* filter_components
        $(#[doc = $doc_rename:literal])* rename_component
    ) => {
        $crate::py_field_impl!(stub [$($T),+] {
            $(#[doc = $doc_filter])*
            fn filter_components(
                &self,
                components: &::pyo3::Bound<'_, ::pyo3::PyAny>,
            ) -> ::pyo3::PyResult<Self> {
                use $crate::containers::field::$F;
                let wanted = $crate::py::ops::field::extract_names(components)?;
                let out = $crate::py_field_family!($F, filter, self, wanted.as_slice())?;
                Ok($crate::py_field_family!($F, wrap, Self, out))
            }

            $(#[doc = $doc_rename])*
            fn rename_component(&self, old: &str, new: &str) -> ::pyo3::PyResult<Self> {
                use $crate::containers::field::$F;
                let out = $F::rename_component($crate::py_field_family!($F, read, self), old, new)?;
                Ok($crate::py_field_family!($F, wrap, Self, out))
            }
        });
    };
    ($(#[doc = $doc:literal])* $F:ident richcmp: [$($T:ident),+ $(,)?], $op:path) => {
        // Seul bras sans `gen_stub_pymethods` : `__richcmp__` est une graphie
        // propre à pyo3, là où CPython expose `__ge__`/`__gt__`/`__le__`/
        // `__lt__`. Ce sont ces quatre noms que le stub déclare à la main ; les
        // engendrer ici aussi les compterait deux fois.
        $crate::py_field_impl!(bare [$($T),+] {
            $(#[doc = $doc])*
            fn __richcmp__(
                &self,
                py: ::pyo3::Python<'_>,
                other: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                op: ::pyo3::pyclass::CompareOp,
            ) -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> {
                use ::pyo3::pyclass::CompareOp;
                use ::pyo3::types::PyAnyMethods;
                let Ok(x) = other.extract::<f64>() else {
                    return Ok(py.NotImplemented());
                };
                let band = match op {
                    CompareOp::Ge => $crate::atoms::Band::new(Some(x), None, None, None),
                    CompareOp::Gt => $crate::atoms::Band::new(None, Some(x), None, None),
                    CompareOp::Le => $crate::atoms::Band::new(None, None, Some(x), None),
                    CompareOp::Lt => $crate::atoms::Band::new(None, None, None, Some(x)),
                    CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
                }?;
                let out = $crate::py_field_family!($F, mask, $op, self, &band);
                Ok(::pyo3::Py::new(py, $crate::py_field_family!($F, wrap, Self, out))?.into_any())
            }
        });
    };
}

// ─── Les verbes de lecture ──────────────────────────────────────────────────

py_field_read! {
    /// Smallest value of `component` across the zones defining it — or, called
    /// without a component, the smallest value of the **whole** field, every
    /// component of every zone pooled (see the sub-field's `min` for what
    /// pooling means).
    Field optional: [PyNodeField, PyElementField], min
}
py_field_read! {
    /// Smallest value of the named `component` — or, called without one, the
    /// smallest value of the **whole** field, every component pooled. Pooling
    /// reads the field as the flat list of its values: on components carrying
    /// different units it answers "the smallest number in there", not a
    /// physical quantity.
    SubField optional: [PySubNodeField, PySubElementField], min
}

py_field_read! {
    /// Largest value of `component` across the zones defining it — or, called
    /// without a component, the largest value of the **whole** field (see `min`).
    Field optional: [PyNodeField, PyElementField], max
}
py_field_read! {
    /// Largest value of the named `component` — or, called without one, the
    /// largest value of the **whole** field, every component pooled (see `min`).
    SubField optional: [PySubNodeField, PySubElementField], max
}

py_field_read! {
    /// Sum of `component` across the zones defining it (Σ over the whole field)
    /// — the resultant of a nodal force field, one component at a time. A node
    /// carried by several zones counts once per zone that stores it. Errors if
    /// no zone defines the component.
    Field named: [PyNodeField, PyElementField], sum
}
py_field_read! {
    /// Sum of the named `component` over the support — Σ over the nodes, or over
    /// the Gauss points for a field by elements. The resultant of a nodal force
    /// field, one component at a time. An empty support sums to `0.0`.
    SubField named: [PySubNodeField, PySubElementField], sum
}

py_field_read! {
    /// Union of the zones' component names, first-seen order.
    Field components: [PyNodeField, PyElementField]
}
py_field_read! {
    /// Component names, in order.
    SubField components: [PySubNodeField, PySubElementField]
}

py_field_read! {
    /// Number of components stored per node, or per Gauss point for a field by
    /// elements.
    SubField count: [PySubNodeField, PySubElementField], component_count
}
py_field_read! {
    /// Index of component `name`, or `None` if unknown — no default index would
    /// say "absent" without being mistaken for a real one.
    SubField index: [PySubNodeField, PySubElementField], component_index
}

// ─── Les quatre mutateurs de composante ─────────────────────────────────────

py_field_write! {
    /// Add `scalar` to `component` on every zone that defines it.
    Field scalar: [PyNodeField, PyElementField], add_to_component
}
py_field_write! {
    /// Add `scalar` to every value of `component` (in place).
    SubField scalar: [PySubNodeField, PySubElementField], add_to_component
}

py_field_write! {
    /// Subtract `scalar` from `component` on every zone that defines it.
    Field scalar: [PyNodeField, PyElementField], sub_to_component
}
py_field_write! {
    /// Subtract `scalar` from every value of `component` (in place).
    SubField scalar: [PySubNodeField, PySubElementField], sub_to_component
}

py_field_write! {
    /// Multiply `component` by `scalar` on every zone that defines it.
    Field scalar: [PyNodeField, PyElementField], mul_to_component
}
py_field_write! {
    /// Multiply every value of `component` by `scalar` (in place).
    SubField scalar: [PySubNodeField, PySubElementField], mul_to_component
}

py_field_write! {
    /// Divide `component` by `scalar` on every zone that defines it.
    Field scalar: [PyNodeField, PyElementField], div_to_component
}
py_field_write! {
    /// Divide every value of `component` by `scalar` (in place).
    SubField scalar: [PySubNodeField, PySubElementField], div_to_component
}
