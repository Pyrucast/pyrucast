//! Méthodes de délégation — la face « sujet » des opérateurs polymorphes.
//!
//! Voir `CONVENTIONS.md` § « Le verbe exposé aussi en méthode ». La fonction
//! libre reste la forme canonique ; ces méthodes ne contiennent aucune logique.
//!
//! Les opérateurs de ce module rendent un champ de la sorte reçue, donc les
//! quatre saveurs (`NodeField` / `SubNodeField` / `ElementField` /
//! `SubElementField`) portent les mêmes méthodes. `psca` n'y figure pas : le
//! produit scalaire est **symétrique**, `a.psca(b)` suggérerait que l'ordre
//! compte.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;

/// Génère, pour une saveur de champ, les onze maths élémentaires. Le masque
/// n'est plus ici : `mask` produit un conteneur déterminé, il a donc rejoint
/// `ops::node_field` et `ops::element_field`.
macro_rules! field_methods {
    ($T:ty) => {
        impl $T {
            /// Valeur absolue, voir [`field::abs`](fn@crate::ops::field::abs).
            pub fn abs(&self) -> Result<Self> {
                crate::ops::field::abs(self)
            }
            /// Racine carrée, voir [`field::sqrt`](fn@crate::ops::field::sqrt).
            pub fn sqrt(&self) -> Result<Self> {
                crate::ops::field::sqrt(self)
            }
            /// Exponentielle, voir [`field::exp`](fn@crate::ops::field::exp).
            pub fn exp(&self) -> Result<Self> {
                crate::ops::field::exp(self)
            }
            /// Logarithme népérien, voir [`field::log`](fn@crate::ops::field::log).
            pub fn log(&self) -> Result<Self> {
                crate::ops::field::log(self)
            }
            /// Logarithme décimal, voir [`field::log10`](fn@crate::ops::field::log10).
            pub fn log10(&self) -> Result<Self> {
                crate::ops::field::log10(self)
            }
            /// Cosinus, voir [`field::cos`](fn@crate::ops::field::cos).
            pub fn cos(&self) -> Result<Self> {
                crate::ops::field::cos(self)
            }
            /// Sinus, voir [`field::sin`](fn@crate::ops::field::sin).
            pub fn sin(&self) -> Result<Self> {
                crate::ops::field::sin(self)
            }
            /// Tangente, voir [`field::tan`](fn@crate::ops::field::tan).
            pub fn tan(&self) -> Result<Self> {
                crate::ops::field::tan(self)
            }
            /// Sinus hyperbolique, voir [`field::sinh`](fn@crate::ops::field::sinh).
            pub fn sinh(&self) -> Result<Self> {
                crate::ops::field::sinh(self)
            }
            /// Cosinus hyperbolique, voir [`field::cosh`](fn@crate::ops::field::cosh).
            pub fn cosh(&self) -> Result<Self> {
                crate::ops::field::cosh(self)
            }
            /// Tangente hyperbolique, voir [`field::tanh`](fn@crate::ops::field::tanh).
            pub fn tanh(&self) -> Result<Self> {
                crate::ops::field::tanh(self)
            }
        }
    };
}

field_methods!(NodeField);
field_methods!(ElementField);
field_methods!(SubNodeField);
field_methods!(SubElementField);
