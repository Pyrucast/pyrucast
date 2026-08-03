//! Thermal (free-dilation) strain `ε_th = α·(T − T_ref)` at the Gauss points —
//! the FE brick behind Cast3M's `EPTH`.
//!
//! Uncoupled thermomechanics: a prescribed temperature field produces a thermal
//! strain that is subtracted from the total strain, `σ = D:(ε − ε_th)`. This
//! operator only builds `ε_th`; the caller composes the rest with the existing
//! bricks (`integrate_behavior`, `internal_forces`, `deformation`,
//! `merge_field`).
//!
//! The temperature is taken **already at the Gauss points** (an
//! [`ElementField`], e.g. produced by
//! [`crate::ops::element_field::interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss)
//! from a nodal field); no interpolation happens here. The dilation coefficient
//! `alpha` is read from the material field, where it travels as an **optional**
//! component (see
//! [`crate::models::Domain::optional_material_components`]).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::ops::element_field::gradient::AXES;
use crate::store::{insert, read};

/// Temperature component read from the per-element temperature field.
const TEMPERATURE: &str = "T";
/// Thermal-expansion coefficient read from the material field.
const ALPHA: &str = "alpha";

/// Thermal strain `ε_th = α·(T − t_ref)` at the Gauss points of every subspace
/// of `fespace`.
///
/// - `temperature` carries the component `"T"` at the Gauss points (per
///   element), e.g. from
///   [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss);
/// - `material` carries `"alpha"` (the material field built with an `alpha`
///   pair — an optional component of the elastic material).
///
/// The result is the symmetric strain tensor in **tensor** convention, same
/// component layout as [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
/// (`eps_<ai><aj>` for `i ≤ j`): the normal components equal `α·(T − t_ref)`,
/// the shear components are zero. It aligns component-for-component with
/// `deformation`, so `deformation(u) − thermal_strain(…)` is a well-formed
/// mechanical strain.
///
/// Runs on the shared parallel driver
/// [`kernel::element_pointwise`](fn@crate::models::kernel::element_pointwise),
/// like [`crate::ops::element_field::behavior::integrate`].
pub fn thermal_strain(
    temperature: &ElementField,
    material: &ElementField,
    fespace: &FiniteElementSpace,
    t_ref: f64,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for sub in fespace {
        let temp_sub = temperature.sub_for_fespace(sub)?;
        // Resolve the material zone by `alpha`, so a shared fespace carrying
        // several component-disjoint material zones (e.g. thermal `k` +
        // mechanical `E`/`nu`/`alpha`) resolves the expansion zone without an
        // explicit consolidate.
        let mat_sub = material.sub_for_fespace_with(sub, &[ALPHA])?;
        let (space_dim, axisymmetric) = {
            let s = read(sub)?;
            (s.space_dim(), s.is_axisymmetric())
        };

        // Fail fast with an actionable message if the temperature lacks its
        // component (the material zone was already selected on `alpha`).
        if !read(&temp_sub)?
            .components()
            .iter()
            .any(|c| c == TEMPERATURE)
        {
            return Err(PyrucastError::Message(format!(
                "thermal_strain: the temperature field carries no '{TEMPERATURE}' component"
            )));
        }

        // Strain entries eps_<ai><aj> for i ≤ j; the diagonal (i == j) carries α·ΔT.
        let mut names = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        let mut pairs = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        for i in 0..space_dim {
            for j in i..space_dim {
                names.push(format!("eps_{}{}", AXES[i], AXES[j]));
                pairs.push((i, j));
            }
        }
        // A body of revolution dilates circumferentially too: the hoop `eps_zz`
        // is a fourth diagonal entry, matching the extra component
        // [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
        // produces there.
        if axisymmetric {
            names.push("eps_zz".to_string());
        }
        let n_diag = names.len();

        let sf = kernel::element_pointwise(
            sub,
            &temp_sub,
            Some(&mat_sub),
            names,
            |geom, input, material, g, out| {
                let cell = geom.cell;
                let material = material.expect("thermal_strain supplies a material field");
                let d_temp = input.value(cell, g, TEMPERATURE)? - t_ref;
                let eps_th = material.value(cell, g, ALPHA)? * d_temp;
                for (c, &(i, j)) in pairs.iter().enumerate() {
                    out[c] = if i == j { eps_th } else { 0.0 };
                }
                if axisymmetric {
                    out[n_diag - 1] = eps_th; // hoop `eps_zz`
                }
                Ok(())
            },
        )?;
        out.add_sub(insert(sf))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::store::{insert, read, write};

    fn tri3_fes_2d() -> FiniteElementSpace {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        FiniteElementSpace::lagrange1(&mesh).unwrap()
    }

    /// Uniform heating ΔT = 50, α = 1e-5 ⇒ ε_th normal = 5e-4, shear = 0.
    #[test]
    fn uniform_heating_gives_isotropic_normal_strain() {
        let fes = tri3_fes_2d();
        let temp = ElementField::new(&fes, vec!["T".into()]).unwrap();
        write(&temp.get(0).unwrap())
            .unwrap()
            .set_uniform("T", 70.0)
            .unwrap();
        let mat = ElementField::new(&fes, vec!["alpha".into()]).unwrap();
        write(&mat.get(0).unwrap())
            .unwrap()
            .set_uniform("alpha", 1e-5)
            .unwrap();

        let eps = thermal_strain(&temp, &mat, &fes, 20.0).unwrap();
        let s = read(&eps.get(0).unwrap()).unwrap();
        assert_eq!(
            s.components(),
            &[
                "eps_xx".to_string(),
                "eps_xy".to_string(),
                "eps_yy".to_string()
            ]
        );
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "eps_xx").unwrap() - 5e-4).abs() < 1e-15);
            assert!((s.value(0, g, "eps_yy").unwrap() - 5e-4).abs() < 1e-15);
            assert!(s.value(0, g, "eps_xy").unwrap().abs() < 1e-18);
        }
    }

    /// T = T_ref ⇒ no thermal strain.
    #[test]
    fn no_temperature_change_gives_zero_strain() {
        let fes = tri3_fes_2d();
        let temp = ElementField::new(&fes, vec!["T".into()]).unwrap();
        write(&temp.get(0).unwrap())
            .unwrap()
            .set_uniform("T", 20.0)
            .unwrap();
        let mat = ElementField::new(&fes, vec!["alpha".into()]).unwrap();
        write(&mat.get(0).unwrap())
            .unwrap()
            .set_uniform("alpha", 1e-5)
            .unwrap();

        let eps = thermal_strain(&temp, &mat, &fes, 20.0).unwrap();
        let s = read(&eps.get(0).unwrap()).unwrap();
        for g in 0..s.gauss_count() {
            for comp in ["eps_xx", "eps_xy", "eps_yy"] {
                assert!(s.value(0, g, comp).unwrap().abs() < 1e-18);
            }
        }
    }

    /// Missing `alpha` in the material ⇒ actionable error.
    #[test]
    fn missing_alpha_errors() {
        let fes = tri3_fes_2d();
        let temp = ElementField::new(&fes, vec!["T".into()]).unwrap();
        write(&temp.get(0).unwrap())
            .unwrap()
            .set_uniform("T", 50.0)
            .unwrap();
        // Material without alpha.
        let mat = ElementField::new(&fes, vec!["E".into()]).unwrap();

        let err = thermal_strain(&temp, &mat, &fes, 20.0).unwrap_err();
        assert!(format!("{err}").contains("alpha"), "unexpected: {err}");
    }
}
