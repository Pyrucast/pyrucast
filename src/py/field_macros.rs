//! Les trois formes de méthode que les champs répètent assez pour valoir une
//! macro.
//!
//! Une macro ne se justifie que par son **nombre d'expansions** : la méthode de
//! transformation est engendrée onze fois par famille, le slot binaire huit
//! fois, le mutateur de composante quatre. Les formes plus rares — `min`,
//! `sum`, `components`, `__pow__`, `__richcmp__`… — sont écrites à la main dans
//! `node_field.rs` et `element_field.rs`, où elles se lisent d'une traite.
//!
//! Chaque forme existe en deux macros, une par famille de champ :
//! `impl_field_*` pour les agrégats (`PyNodeField`, `PyElementField`), qui
//! tiennent leur valeur dans `self.inner` ; `impl_subfield_*` pour les
//! sous-conteneurs (`PySubNodeField`, `PySubElementField`), qui la lisent à
//! travers `self.handle`. Deux macros plutôt qu'un paramètre de famille : le
//! corps nomme alors son trait et son accès en clair, et se lit sans détour.
//!
//! Le nom d'une macro dit l'**item produit**, pas ce qu'on lui passe : elle lie
//! n'importe quelle fonction de la forme attendue, et l'inventaire du jour n'en
//! est pas une propriété.

/// Pose le même bloc de méthodes sur chaque type de la liste, dans un
/// `#[pymethods]` par type. C'est le seul code que les macros de ce fichier
/// partagent.
///
/// Arguments : `stub` ou `bare` selon que le bloc doit être décoré pour
/// pyo3-stub-gen, la liste des types, puis le bloc lui-même, accolades
/// comprises.
///
/// ```ignore
/// impl_pymethods_for_each!(stub [PyNodeField, PyElementField] {
///     fn zero(&self) -> f64 { 0.0 }
/// });
/// ```
///
/// **Le bloc arrive en un seul `tt`**, et non décomposé. C'est ce qui permet
/// d'écrire la documentation en `///` au site d'appel : capturée ligne à ligne
/// (`$(#[doc = $doc:literal])*`), elle ne peut pas se répéter à l'intérieur
/// d'une boucle sur les types, rustc refusant deux répétitions de longueurs
/// différentes au même niveau. Un `tt`, lui, se recopie librement.
///
/// **Le bloc nomme son type `Self`**, puisqu'il ignore lequel le portera.
///
/// `bare` n'est plus utilisé par ce fichier : il reste pour un bloc qu'il ne
/// faudrait pas déclarer au stub, comme l'est `__richcmp__`, dont CPython
/// expose les quatre noms `__ge__`/`__gt__`/`__le__`/`__lt__`.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_pymethods_for_each {
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

/// Engendre, sur un **agrégat**, une méthode sans argument qui rend un champ de
/// même saveur : elle applique à tout le champ la fonction de `ops::field` qui
/// porte son nom, laquelle ne prend que le champ.
///
/// Arguments : la documentation en `///`, la liste des types, le nom de la
/// méthode — qui est aussi celui de la fonction de `ops::field` appelée.
///
/// ```ignore
/// impl_field_transform_pymethod! {
///     /// Element-wise square root of a field (`nan` for negatives).
///     [PyNodeField, PyElementField], sqrt
/// }
/// ```
///
/// engendre sur chacun des deux types :
///
/// ```ignore
/// fn sqrt(&self) -> PyResult<Self> {
///     let out = ops::field::sqrt(&self.inner)?;
///     Ok(Self { inner: out })
/// }
/// ```
///
/// Les onze appels d'aujourd'hui — `sqrt`, `exp`, `cos`… — passent par
/// `define_polymorphic_pyfunction!` (`py/ops/field.rs`), qui engendre d'un coup
/// la fonction libre qui dispatche et les méthodes des quatre saveurs, avec un
/// texte écrit une fois.
#[macro_export]
macro_rules! impl_field_transform_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> ::pyo3::PyResult<Self> {
                let out = $crate::ops::field::$nom(&self.inner)?;
                Ok(Self { inner: out })
            }
        });
    };
}

/// La même, sur un **sous-conteneur** : la valeur est lue à travers le handle,
/// et le résultat reçoit un handle neuf.
#[macro_export]
macro_rules! impl_subfield_transform_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> ::pyo3::PyResult<Self> {
                let out = $crate::ops::field::$nom(&*self.handle.read())?;
                Ok(Self {
                    handle: $crate::handle::Handle::new(out),
                })
            }
        });
    };
}

/// Engendre, sur un **agrégat**, un opérateur binaire (`__add__`, `__sub__`,
/// `__mul__`, `__truediv__`) : il passe le membre droit et la closure au
/// dispatcheur `binary` que la saveur définit dans son bloc inhérent.
///
/// Arguments : la documentation en `///`, la liste des types, le nom du slot,
/// et la closure appliquée terme à terme.
///
/// ```ignore
/// impl_field_binary_pyslot! {
///     /// `field + other` — element-wise sum.
///     [PyNodeField], __add__, |a, b| a + b
/// }
/// ```
///
/// **À appeler depuis le module qui déclare le `#[pyclass]`**, avec une liste
/// d'un seul type : pyo3 engendre pour un slot un trampoline `unsafe fn` qui en
/// appelle un autre, et l'édition 2024 ne couvre plus implicitement ce corps —
/// `unsafe_op_in_unsafe_fn` se déclenche dès que l'`impl` vit ailleurs.
#[macro_export]
macro_rules! impl_field_binary_pyslot {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident, $f:expr) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                self.binary(rhs, $f)
            }
        });
    };
}

/// La même, sur un **sous-conteneur** : son dispatcheur s'appelle
/// `scalar_or_combine` — un scalaire s'applique partout, un autre sous-champ
/// se combine élément par élément.
#[macro_export]
macro_rules! impl_subfield_binary_pyslot {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident, $f:expr) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                self.scalar_or_combine(rhs, $f)
            }
        });
    };
}

/// Engendre, sur un **agrégat**, une mutation de composante par un scalaire
/// (`add_to_component` et ses trois voisines). La mutation descend aux zones
/// qui définissent la composante, et la méthode ne rend rien.
///
/// Arguments : la documentation en `///`, la liste des types, le nom de la
/// méthode — qui est aussi celui de la méthode du trait `Field` appelée.
///
/// ```ignore
/// impl_field_mutator_pymethod! {
///     /// Add `scalar` to `component` on every zone that defines it.
///     [PyNodeField, PyElementField], add_to_component
/// }
/// ```
#[macro_export]
macro_rules! impl_field_mutator_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                use $crate::containers::field::Field;
                self.inner.$nom(component, scalar)?;
                Ok(())
            }
        });
    };
}

/// La même, sur un **sous-conteneur** : la mutation a lieu en place, dans sa
/// zone à lui, sous le seul **write** guard que prennent ces macros.
#[macro_export]
macro_rules! impl_subfield_mutator_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                use $crate::containers::field::SubField;
                self.handle.write().$nom(component, scalar)?;
                Ok(())
            }
        });
    };
}
