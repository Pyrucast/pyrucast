//! Ce que les slots des champs partagent : deux fonctions, pas une macro.
//!
//! `__pow__` et `__richcmp__` existent en quatre exemplaires — un par saveur de
//! champ —, et doivent être écrits dans le module qui déclare leur `#[pyclass]`
//! (un slot engendre chez pyo3 un trampoline `unsafe fn` que l'édition 2024 ne
//! couvre plus hors de ce module). Ce qu'ils ont de commun n'est pas leur
//! forme, qui tient en quelques lignes, mais leur **sémantique** : quelle bande
//! de valeurs dit une comparaison, et pourquoi un modulo est refusé. C'est ce
//! qui vit ici, appelé par les huit méthodes.

use crate::atoms::Band;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

/// La bande de valeurs que dit cette comparaison : `>= x` borne par le bas,
/// `< x` par le haut, etc.
///
/// `None` quand la comparaison n'a pas de bande — `==` et `!=`, qui ne sont pas
/// des seuils, et tout membre droit qui n'est pas un scalaire. L'appelant rend
/// alors `NotImplemented`, laissant Python chercher l'opération réfléchie.
pub(crate) fn band_of(op: CompareOp, other: &Bound<'_, PyAny>) -> PyResult<Option<Band>> {
    let Ok(x) = other.extract::<f64>() else {
        return Ok(None);
    };
    let band = match op {
        CompareOp::Ge => Band::new(Some(x), None, None, None),
        CompareOp::Gt => Band::new(None, Some(x), None, None),
        CompareOp::Le => Band::new(None, None, Some(x), None),
        CompareOp::Lt => Band::new(None, None, None, Some(x)),
        CompareOp::Eq | CompareOp::Ne => return Ok(None),
    }?;
    Ok(Some(band))
}

/// Refuse la forme ternaire `pow(x, y, z)` : un modulo n'a pas de sens sur des
/// flottants, et l'accepter silencieusement le ferait disparaître du calcul.
pub(crate) fn reject_modulo(modulo: &Bound<'_, PyAny>) -> PyResult<()> {
    if modulo.is_none() {
        return Ok(());
    }
    Err(PyTypeError::new_err(
        "field ** exponent does not support a modulo argument",
    ))
}
