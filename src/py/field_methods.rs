//! Les méthodes Python que les quatre saveurs de champ portent à l'identique.
//!
//! Aucune n'a de fonction libre dont `#[py_op]` la dériverait : ce sont des
//! méthodes de conteneur et rien d'autre. Deux `macro_rules!` les engendrent,
//! selon ce que fait le verbe : `py_field_write!` (muter une composante) et
//! `py_field_transform!` (rendre un champ). Deux macros cachées portent leur
//! travail commun : `py_field_impl!` pose un bloc de méthodes sur plusieurs
//! types, `py_field_family!` traduit ce qui sépare les deux familles de champ.
//!
//! Les verbes de lecture, eux, sont écrits à la main dans `node_field.rs` et
//! `element_field.rs` : une forme appelée une ou deux fois ne rembourse pas la
//! macro qui l'engendrerait.
//!
//! **Les deux familles.** Les agrégats (`PyNodeField`, `PyElementField`)
//! tiennent leur valeur en propre, dans `self.inner`, et le trait `Field`
//! opère dessus. Les sous-conteneurs (`PySubNodeField`, `PySubElementField`)
//! la lisent à travers `self.handle`, sous le trait `SubField`. Chaque appel
//! nomme la famille qu'il sert, et sert d'un coup les deux types de cette
//! famille.
//!
//! Un verbe demande donc **deux appels**, un par famille. C'est voulu : leurs
//! documentations diffèrent. Un agrégat parle des zones qui définissent la
//! composante ; un sous-champ n'a qu'un support et aucune zone à évoquer, et
//! lui servir le texte de l'agrégat le ferait se demander où sont ses zones.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};

/// Traduit un **aspect** en code, pour la famille demandée. C'est la seule
/// place où est écrit ce qui sépare un agrégat d'un sous-conteneur.
///
/// Arguments : la famille (`Field` ou `SubField`), le nom de l'aspect, puis ce
/// dont cet aspect a besoin. Rend une expression, à poser telle quelle dans le
/// corps d'une méthode :
///
/// ```ignore
/// let n = Field::components(py_field_family!(Field, read, self)).len();
/// ```
///
/// Chaque aspect tient en une ligne par famille, les deux l'une sous l'autre.
///
/// **`self` se passe en argument, et vient toujours du corps de la macro
/// appelante.** Un `self` est hygiénique : il ne désigne le receveur que s'il
/// a été écrit dans la même expansion que la méthode. Venu du site d'appel,
/// il ne compilerait pas (`E0424`).
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

/// Pose le même bloc de méthodes sur chaque type de la liste, dans un
/// `#[pymethods]` par type.
///
/// Arguments : `stub` ou `bare` selon que le bloc doit être décoré pour
/// pyo3-stub-gen, la liste des types, puis le bloc lui-même, accolades
/// comprises.
///
/// ```ignore
/// py_field_impl!(stub [PyNodeField, PyElementField] {
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
/// `bare` sert au seul `__richcmp__` : c'est une graphie pyo3, là où CPython
/// expose `__ge__`/`__gt__`/`__le__`/`__lt__`. Le stub déclare ces quatre noms
/// à la main, et décorer le bloc les compterait une seconde fois.
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

/// Engendre une méthode Python qui **mute** une composante, et ne rend rien.
///
/// Mêmes arguments que `py_field_read!`. Le bras unique, `scalar:`, donne à la
/// méthode la composante à modifier et le scalaire à lui appliquer.
///
/// ```ignore
/// py_field_write! {
///     /// Add `scalar` to `component` on every zone that defines it.
///     Field scalar: [PyNodeField, PyElementField], add_to_component
/// }
/// ```
///
/// La mutation d'un agrégat descend aux zones qui définissent la composante ;
/// celle d'un sous-conteneur a lieu dans sa zone à lui, sous le seul **write**
/// guard que prennent ces macros.
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

/// Engendre une méthode Python qui rend un **champ de la saveur du receveur**.
///
/// Arguments : la documentation en `///`, la famille, le bras suivi de `:`, la
/// liste des types, puis ce que le bras réclame — un nom de méthode, une
/// closure binaire, le chemin d'un opérateur.
///
/// ```ignore
/// py_field_transform! {
///     /// Element-wise square root of a field (`nan` for negatives).
///     SubField unary: [PySubNodeField, PySubElementField], sqrt
/// }
/// ```
///
/// engendre sur chacun des deux types :
///
/// ```ignore
/// fn sqrt(&self) -> PyResult<Self> {
///     let out = ops::field::sqrt(&*self.handle.read())?;
///     Ok(Self { handle: Handle::new(out) })
/// }
/// ```
///
/// Les bras : `op:` et `pow:` passent au dispatcheur d'opérateur de la saveur,
/// `unary:` applique une math élémentaire de `ops::field`, `components:` filtre
/// et renomme — deux méthodes, donc deux `///`, chacun devant le nom de la
/// méthode qu'il décrit —, `richcmp:` compare à un scalaire et rend un masque.
///
/// **`op:`, `pow:` et `richcmp:` sont des slots, à appeler depuis le module qui
/// déclare le `#[pyclass]`**, avec une liste d'un seul type. pyo3 leur engendre
/// un trampoline `unsafe fn` qui en appelle un autre ; l'édition 2024 ne couvre
/// plus implicitement ce corps, et `unsafe_op_in_unsafe_fn` se déclenche dès
/// que l'`impl` vit hors de ce module.
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
