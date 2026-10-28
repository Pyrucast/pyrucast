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
use crate::handle::Handle;
use crate::models::kernel;
use crate::ops::element_field::gradient::AXES;
use crate::ops::element_field::StrainMeasure;

/// Temperature component read from the per-element temperature field.
const TEMPERATURE: &str = "T";
/// Thermal-expansion coefficient read from the material field.
const ALPHA: &str = "alpha";

/// Thermal strain `ε_th = α·(T − t_ref)` at the Gauss points of every subspace
/// of `fespace`, in the strain `measure` a law reads.
///
/// - `temperature` carries the component `"T"` at the Gauss points (per
///   element), e.g. from
///   [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss);
/// - `material` carries `"alpha"` (the material field built with an `alpha`
///   pair — an optional component of the elastic material).
///
/// The result is the symmetric strain tensor in **tensor** convention, same
/// component layout as [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
/// (`<prefix>_<ai><aj>` for `i ≤ j`, the prefix that of `measure`): the strain
/// of a free isotropic dilation of stretch `1 + α·(T − t_ref)`, zero shear. Its
/// normal components are, exactly, `α·ΔT` for `linear`, `½((1 + α·ΔT)² − 1)`
/// for `green_lagrange` and `ln(1 + α·ΔT)` for `hencky` — the three agree to
/// first order. It aligns component-for-component with `deformation` under the
/// same measure, so `deformation(u) − thermal_strain(…)` is a well-formed
/// mechanical strain.
///
/// Errors: a measure that is not a strain a law reads (`gradient`,
/// Cauchy–Green, Euler–Almansi).
///
/// Runs on the shared parallel driver
/// [`kernel::element_pointwise`](fn@crate::models::kernel::element_pointwise),
/// like [`crate::ops::element_field::behavior::integrate`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // ε_th = α(T − T_ref)·I: purely spherical, and **zero at the
/// // température de référence**.
/// # let mut t = ElementField::new(&fes, vec!["T".into()])?;
/// # t.get(0)?.write().set_uniform("T", 120.0)?;
/// # let mut m = ElementField::new(&fes, vec!["alpha".into()])?;
/// # m.get(0)?.write().set_uniform("alpha", 1e-5)?;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// let eps = element_field::thermal_strain(&t, &m, &fes, 20.0, StrainMeasure::Linear)?;
/// assert!((eps.get(0)?.read().value(0, 0, "eps_xx")? - 1e-3).abs() < 1e-12);
/// assert!(eps.get(0)?.read().value(0, 0, "eps_xy")?.abs() < 1e-15);
/// // At T = T_ref, nothing.
/// let nulle = element_field::thermal_strain(&t, &m, &fes, 120.0, StrainMeasure::Linear)?;
/// assert!(nulle.get(0)?.read().value(0, 0, "eps_xx")?.abs() < 1e-15);
/// // Read by a law in Green–Lagrange, the same dilation is ½((1 + αΔT)² − 1).
/// let e = element_field::thermal_strain(&t, &m, &fes, 20.0, StrainMeasure::GreenLagrange)?;
/// assert!((e.get(0)?.read().value(0, 0, "E_xx")? - 0.5 * (1.001_f64.powi(2) - 1.0)).abs() < 1e-15);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn thermal_strain(
    temperature: &ElementField,
    material: &ElementField,
    fespace: &FiniteElementSpace,
    t_ref: f64,
    measure: StrainMeasure,
) -> Result<ElementField> {
    // The strain of the stretch `1 + a` in the measure, chosen once.
    let dilation: fn(f64) -> f64 = match measure {
        StrainMeasure::Linear => |a| a,
        StrainMeasure::GreenLagrange => |a| 0.5 * ((1.0 + a) * (1.0 + a) - 1.0),
        StrainMeasure::Hencky => |a| (1.0 + a).ln(),
        other => {
            return Err(PyrucastError::Message(format!(
                "thermal_strain: `{}` is not a strain a law reads — the measures are `linear`, \
                 `green_lagrange` and `hencky`",
                crate::named::Named::name(other)
            )))
        }
    };
    let prefix = measure.prefix();
    let mut out = ElementField::empty();
    for sub in fespace {
        let temp_sub = temperature.sub_for_fespace(sub)?;
        // Resolve the material zone by `alpha`, so a shared fespace carrying
        // several component-disjoint material zones (e.g. thermal `k` +
        // mechanical `E`/`nu`/`alpha`) resolves the expansion zone without an
        // explicit consolidate.
        let mat_sub = material.sub_for_fespace_with(sub, &[ALPHA.to_string()])?;
        let (space_dim, axisymmetric) = {
            let s = sub.read();
            (s.space_dim(), s.is_axisymmetric())
        };

        // Where the two values sit, resolved **once** for the zone — which is
        // also the check that they are there at all, with a message naming the
        // field rather than a bare component name.
        let i_temp = temp_sub
            .read()
            .resolve_components(&[TEMPERATURE], "thermal_strain: temperature")?[0]
            as usize;
        let i_alpha = mat_sub
            .read()
            .resolve_components(&[ALPHA], "thermal_strain: material")?[0]
            as usize;

        // Strain entries <prefix>_<ai><aj> for i ≤ j; the diagonal (i == j)
        // carries the dilation.
        let mut names = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        let mut pairs = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
        for i in 0..space_dim {
            for j in i..space_dim {
                names.push(format!("{prefix}_{}{}", AXES[i], AXES[j]));
                pairs.push((i, j));
            }
        }
        // A body of revolution dilates circumferentially too: the hoop `zz`
        // is a fourth diagonal entry, matching the extra component
        // [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
        // produces there.
        if axisymmetric {
            names.push(format!("{prefix}_zz"));
        }
        let n_diag = names.len();

        let sf = kernel::element_pointwise(
            sub,
            &temp_sub,
            // Thermal expansion reads no previous state: `α·ΔT` depends on the
            // temperature and the material, on nothing that happened before.
            // The driver wants a field, so it gets one it will never index.
            &temp_sub,
            &mat_sub,
            names,
            |_geom, _g, input, _prev, material, out| {
                let eps_th = dilation(material[i_alpha] * (input[i_temp] - t_ref));
                for (c, &(i, j)) in pairs.iter().enumerate() {
                    out[c] = if i == j { eps_th } else { 0.0 };
                }
                if axisymmetric {
                    out[n_diag - 1] = eps_th; // the hoop `zz`
                }
                Ok(())
            },
        )?;
        out.add_sub(Handle::new(sf))?;
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
    use crate::handle::Handle;

    fn tri3_fes_2d() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        temp.get(0).unwrap().write().set_uniform("T", 70.0).unwrap();
        let mat = ElementField::new(&fes, vec!["alpha".into()]).unwrap();
        mat.get(0)
            .unwrap()
            .write()
            .set_uniform("alpha", 1e-5)
            .unwrap();

        let eps = thermal_strain(&temp, &mat, &fes, 20.0, StrainMeasure::Linear).unwrap();
        let s = eps.get(0).unwrap().read();
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
        temp.get(0).unwrap().write().set_uniform("T", 20.0).unwrap();
        let mat = ElementField::new(&fes, vec!["alpha".into()]).unwrap();
        mat.get(0)
            .unwrap()
            .write()
            .set_uniform("alpha", 1e-5)
            .unwrap();

        let eps = thermal_strain(&temp, &mat, &fes, 20.0, StrainMeasure::Linear).unwrap();
        let s = eps.get(0).unwrap().read();
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
        temp.get(0).unwrap().write().set_uniform("T", 50.0).unwrap();
        // Material without alpha.
        let mat = ElementField::new(&fes, vec!["E".into()]).unwrap();

        let err = thermal_strain(&temp, &mat, &fes, 20.0, StrainMeasure::Linear).unwrap_err();
        assert!(format!("{err}").contains("alpha"), "unexpected: {err}");
    }
}
