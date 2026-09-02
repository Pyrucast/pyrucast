//! Elastic laws — identity, contract, and the integration structure they share.
//!
//! `<physique>/law.rs` : the enum an archive stores, the trait the physics
//! drives, and the single `match` relating the two.

use crate::error::Result;
use crate::models::continuum::material::MatRead;
use crate::models::continuum::Continuum;
use crate::models::symmetry::MaterialSymmetry;
use serde::{Deserialize, Serialize};

/// Which elastic law an [`Elasticity`](super::Elasticity) evaluates.
///
/// One variant for now. The axis exists because it is the one the two sibling
/// physics already have, and because the next elastic law — a hyperelastic one —
/// is a law, not a physics: same DOFs, same modelling, same layouts.
///
/// New variants go at the end: `bincode` serialises the index.
///
/// ```
/// # use pyrucast::models::elasticity::law::ElasticLaw;
/// # use pyrucast::named::Named;
/// // Réciproque exacte de `from_name`, comme pour les deux autres familles.
/// assert!(ElasticLaw::ALL.iter().all(|l| ElasticLaw::from_name(l.name()) == Some(*l)));
/// assert_eq!(ElasticLaw::default(), ElasticLaw::Linear);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElasticLaw {
    /// `σ = D(matériau, symétrie) : ε` — the constitutive law is the elastic
    /// operator itself, and its tangent is constant.
    #[default]
    Linear,
}

impl ElasticLaw {
    /// The lowercase name (the inverse of
    /// [`from_name`](crate::named::Named::from_name)).
    ///
    /// ```
    /// # use pyrucast::models::elasticity::law::ElasticLaw;
    /// assert_eq!(ElasticLaw::Linear.name(), "linear");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
        }
    }

    /// Every law, in declaration order.
    ///
    /// ```
    /// # use pyrucast::models::elasticity::law::ElasticLaw;
    /// // Une seule pour l'instant : l'axe existe, il n'a qu'un point.
    /// assert_eq!(ElasticLaw::ALL, [ElasticLaw::Linear]);
    /// ```
    pub const ALL: [ElasticLaw; 1] = [Self::Linear];

    /// The behaviour behind this identity — **the only `match` per law**.
    pub(crate) fn as_law(self) -> &'static dyn StatelessLawKind {
        match self {
            Self::Linear => &super::linear::Linear,
        }
    }
}

impl crate::named::Named for ElasticLaw {
    const LABEL: &'static str = "elastic law";
    const VALUES: &'static [Self] = &Self::ALL;

    fn name(self) -> &'static str {
        ElasticLaw::name(self)
    }
}

impl std::fmt::Display for ElasticLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The **stateless** family of constitutive laws: `σ = f(ε)`, with no previous
/// state and no time increment.
///
/// The trait is named after that **integration structure**, not after a physical
/// family, because the structure is what fixes its signature — a strain in, a
/// stress out, and nothing to carry between steps. Linear elasticity lives here;
/// a hyperelastic law (`σ = ∂W/∂ε`), being path-independent, would join it. Its
/// siblings are `ReturnMapLawKind` (elastic predictor plus projection) and
/// `DirectUpdateLawKind` (state, but no predictor).
///
/// Same division of labour as [`SubModelKind`](crate::models::SubModelKind) one
/// level up: the enum [`ElasticLaw`] carries the **physical identity** — what an
/// archive stores — and the trait carries the **integration structure**, so a
/// single `match` ([`ElasticLaw::as_law`]) relates the two.
///
/// Adding a law: a unit struct and its `impl` in the law's own file, plus one
/// arm in `as_law`.
pub(crate) trait StatelessLawKind: Sync {
    /// The material components the law requires, for a symmetry and a space
    /// dimension. Owned, and read once per sub-model — never on a hot path.
    fn material_components(&self, symmetry: MaterialSymmetry, space_dim: usize) -> Vec<String>;

    /// `σ = f(ε)` at one Gauss point, written into `out`.
    ///
    /// `strain` is the engineering-Voigt strain already lifted out of the row;
    /// `mat` is this cell's material, read by position. No `prev`, no `dt`: that
    /// absence **is** the family.
    ///
    /// The `Result` is not ceremonial — an orthotropic or anisotropic contract
    /// can have a singular compliance, and that is what
    /// [`elastic_constitutive_into`](crate::models::symmetry::elastic_constitutive_into)
    /// reports. The isotropic path never takes it.
    fn stress(
        &self,
        strain: &[f64; 6],
        mat: &MatRead,
        continuum: &Continuum,
        symmetry: MaterialSymmetry,
        out: &mut [f64],
    ) -> Result<()>;

    /// Whether the tangent is independent of `ε`. Read **once per zone** — it
    /// decides whether the stiffness *is* the tangent, or merely its
    /// linearisation at `ε = 0`, and whether the law must emit `D_alg` as state.
    /// Never consulted at a Gauss point.
    fn is_linear(&self) -> bool {
        true
    }
}
