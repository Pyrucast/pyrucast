//! Linear elasticity, `σ = D : ε` — the law that does nothing but evaluate the
//! continuum's own elastic operator.
//!
//! Its triviality is in the kernel, not in whether the capability exists: the
//! same `D` is the elastic predictor of every return-map law and the undamaged
//! modulus of every damage law, which is why it lives in
//! [`continuum::elastic`](crate::models::continuum::elastic) and this file only
//! calls it.

use super::law::StatelessLawKind;
use crate::error::Result;
use crate::models::continuum::material::MatRead;
use crate::models::continuum::{elastic, Continuum};
use crate::models::owned_components;
use crate::models::symmetry::{self, MaterialSymmetry};

/// `σ = D(E, ν, symétrie) : ε`.
pub(crate) struct Linear;

impl StatelessLawKind for Linear {
    fn material_components(&self, symmetry: MaterialSymmetry, space_dim: usize) -> Vec<String> {
        owned_components(elastic::material_contract(symmetry, space_dim))
    }

    #[inline]
    fn stress(
        &self,
        strain: &[f64; 6],
        mat: &MatRead,
        continuum: &Continuum,
        symmetry: MaterialSymmetry,
        out: &mut [f64],
    ) -> Result<()> {
        let mut dmat = [[0.0_f64; 6]; 6];
        let v = symmetry::elastic_constitutive_into(
            mat.row,
            mat.idx,
            symmetry,
            continuum.kinematics(),
            continuum.space_dim(),
            &mut dmat,
        )?;
        for r in 0..v {
            out[r] = (0..v).map(|c| dmat[r][c] * strain[c]).sum();
        }
        Ok(())
    }
}
