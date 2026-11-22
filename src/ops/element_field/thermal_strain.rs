//! Thermal (free-dilation) strain `ε_th = α·(T − T_ref)` at the Gauss points of
//! every continuum law of a model.
//!
//! Uncoupled thermomechanics: a prescribed temperature field produces a thermal
//! strain that is subtracted from the total strain, `σ = D:(ε − ε_th)`. This
//! operator only builds `ε_th`; the caller composes the rest with the existing
//! bricks (`strain`, `integrate_behavior`, `internal_forces`, `merge_field`).
//!
//! The temperature is taken **already at the Gauss points** (an
//! [`ElementField`], e.g. produced by
//! [`crate::ops::element_field::interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss)
//! from a nodal field); no interpolation happens here. The dilation coefficient
//! `alpha` is read from the material field, where it travels as an **optional**
//! component (see
//! [`crate::models::Domain::optional_material_components`]).
//!
//! The model says, law by law, what the dilation is written as: the strain
//! measure the law reads (its value and its names) and the kinematics (whether
//! the out-of-plane direction of a 2-D law dilates).

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::model::Model;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::tensor::Kinematics;
use crate::ops::element_field::gradient::AXES;
use crate::ops::element_field::StrainMeasure;

/// Temperature component read from the per-element temperature field.
const TEMPERATURE: &str = "T";
/// Thermal-expansion coefficient read from the material field.
const ALPHA: &str = "alpha";

/// Thermal strain `ε_th = α·(T − t_ref)` at the Gauss points of every continuum
/// law of `model` (elasticity, plasticity, damage), in the strain each law
/// reads.
///
/// - `temperature` carries the component `"T"` at the Gauss points (per
///   element), e.g. from
///   [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss);
/// - `material` carries `"alpha"` (the material field built with an `alpha`
///   pair — an optional component of the continuum laws).
///
/// The result has one zone per continuum law, on its FE space, in **tensor**
/// convention with the component layout of
/// [`deformation`](fn@crate::ops::element_field::deformation) under the
/// measure the law reads (`<prefix>_<ai><aj>` for `i ≤ j`, then `<prefix>_zz`
/// in 2-D): it aligns component for component with
/// [`strain`](fn@crate::ops::element_field::strain), so `strain − thermal_strain`
/// is the mechanical strain the law reads.
///
/// It is the strain of a free isotropic dilation of stretch
/// `λ = 1 + α·(T − t_ref)`, zero shear. Its normal components are, exactly,
/// `α·ΔT` for a law reading `linear`, `½(λ² − 1)` for `green_lagrange` and
/// `ln λ` for `hencky` — the three agree to first order. In 2-D the
/// out-of-plane `zz` dilates in plane strain and on a body of revolution (the
/// hoop); in plane stress it is zero, the law solving its own `ε_zz` under
/// `σ_zz = 0`.
///
/// A sub-model with no continuum law (a conduction, a bar, a beam, a
/// constraint) contributes no zone.
///
/// Errors: `temperature` without `"T"` or `material` without `"alpha"` on a
/// continuum law's FE space.
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
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::ops::{element_field, model};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut t = ElementField::new(&fes, vec!["T".into()])?;
/// # t.get(0)?.write().set_uniform("T", 120.0)?;
/// # let mut m = ElementField::new(&fes, vec!["alpha".into()])?;
/// # m.get(0)?.write().set_uniform("alpha", 1e-5)?;
/// let elasticity = |kinematics, measure| {
///     model::elasticity(&fes, kinematics, MaterialSymmetry::Isotropic, measure)
/// };
/// // ε_th = α(T − T_ref)·I: purely spherical, and **zero at the reference
/// // temperature**. A long prismatic body dilates along its length too.
/// let plane_strain = elasticity(Kinematics::PlaneStrain, StrainMeasure::Linear)?;
/// let eps = element_field::thermal_strain(&plane_strain, &t, &m, 20.0)?;
/// let at = |c: &str| eps.get(0)?.read().value(0, 0, c);
/// assert!((at("eps_xx")? - 1e-3).abs() < 1e-12);
/// assert!((at("eps_zz")? - 1e-3).abs() < 1e-12);
/// assert!(at("eps_xy")?.abs() < 1e-15);
/// // A thin plate leaves its thickness to the law: no out-of-plane dilation.
/// let plate = elasticity(Kinematics::PlaneStress, StrainMeasure::Linear)?;
/// let eps = element_field::thermal_strain(&plate, &t, &m, 20.0)?;
/// assert_eq!(eps.get(0)?.read().value(0, 0, "eps_zz")?, 0.0);
/// // At T = T_ref, nothing.
/// let none = element_field::thermal_strain(&plane_strain, &t, &m, 120.0)?;
/// assert!(none.get(0)?.read().value(0, 0, "eps_xx")?.abs() < 1e-15);
/// // A law reading Green–Lagrange gets the same dilation as ½((1 + αΔT)² − 1).
/// let large = elasticity(Kinematics::PlaneStrain, StrainMeasure::GreenLagrange)?;
/// let e = element_field::thermal_strain(&large, &t, &m, 20.0)?;
/// let expect = 0.5 * (1.001_f64.powi(2) - 1.0);
/// assert!((e.get(0)?.read().value(0, 0, "E_xx")? - expect).abs() < 1e-15);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn thermal_strain(
    model: &Model,
    temperature: &ElementField,
    material: &ElementField,
    t_ref: f64,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for sub_model in model {
        // What the zone needs, read under the sub-model lock and released
        // before any field lock is taken.
        let zone = {
            let sub = sub_model.read();
            sub.as_kind().as_behavior().and_then(|b| {
                b.elastic_operator().map(|(continuum, _)| {
                    (
                        continuum.fespace(),
                        continuum.kinematics(),
                        continuum.measure(),
                    )
                })
            })
        };
        let Some((fespace, kinematics, measure)) = zone else {
            continue; // no continuum law: no zone
        };
        let zone = dilation_zone(&fespace, kinematics, measure, temperature, material, t_ref)?;
        out.add_sub(Handle::new(zone))?;
    }
    Ok(out)
}

/// [`thermal_strain`] on one continuum law's FE space.
fn dilation_zone(
    fespace: &Handle<SubFiniteElementSpace>,
    kinematics: Kinematics,
    measure: StrainMeasure,
    temperature: &ElementField,
    material: &ElementField,
    t_ref: f64,
) -> Result<SubElementField> {
    // The strain of the stretch `1 + a` in the law's measure, chosen once. A
    // continuum law reads `linear`, `green_lagrange` or `hencky` — its
    // constructor admits no other measure — so anything else is linear.
    let dilation: fn(f64) -> f64 = match measure {
        StrainMeasure::GreenLagrange => |a| 0.5 * ((1.0 + a) * (1.0 + a) - 1.0),
        StrainMeasure::Hencky => |a| (1.0 + a).ln(),
        _ => |a| a,
    };
    let temp_sub = temperature.sub_for_fespace(fespace)?;
    // Resolve the material zone by `alpha`, so a shared FE space carrying
    // several component-disjoint material zones (e.g. thermal `k` + mechanical
    // `E`/`nu`/`alpha`) resolves the expansion zone without an explicit
    // consolidate.
    let mat_sub = material.sub_for_fespace_with(fespace, &[ALPHA.to_string()])?;
    let space_dim = fespace.read().space_dim();

    // Where the two values sit, resolved **once** for the zone — which is
    // also the check that they are there at all, with a message naming the
    // field rather than a bare component name.
    let i_temp = temp_sub
        .read()
        .resolve_components(&[TEMPERATURE], "thermal_strain: temperature")?[0]
        as usize;
    let i_alpha = mat_sub
        .read()
        .resolve_components(&[ALPHA], "thermal_strain: material")?[0] as usize;

    // Strain entries <prefix>_<ai><aj> for i ≤ j, then the 2-D `zz`, each with
    // the share of the dilation it takes: 1 on a normal component, 0 on a
    // shear — and 0 on the out-of-plane normal of plane stress, whose `ε_zz`
    // the law solves. Settled here, so the Gauss point multiplies and tests
    // nothing.
    let prefix = measure.prefix();
    let mut names = Vec::with_capacity(7);
    let mut share = Vec::with_capacity(7);
    for i in 0..space_dim {
        for j in i..space_dim {
            names.push(format!("{prefix}_{}{}", AXES[i], AXES[j]));
            share.push(if i == j { 1.0 } else { 0.0 });
        }
    }
    if space_dim == 2 {
        names.push(format!("{prefix}_zz"));
        share.push(if kinematics == Kinematics::PlaneStress {
            0.0
        } else {
            1.0
        });
    }

    kernel::element_pointwise(
        fespace,
        &temp_sub,
        // Thermal expansion reads no previous state: `α·ΔT` depends on the
        // temperature and the material, on nothing that happened before.
        // The driver wants a field, so it gets one it will never index.
        &temp_sub,
        &mat_sub,
        names,
        |_geom, _g, input, _prev, material, out| {
            let eps_th = dilation(material[i_alpha] * (input[i_temp] - t_ref));
            for (o, &s) in out.iter_mut().zip(&share) {
                *o = s * eps_th;
            }
            Ok(())
        },
    )
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::ops::model;

    /// One TRI3 (2-D, off the axis when `axisymmetric`) or one TET4 (3-D).
    fn one_cell(dim: usize, axisymmetric: bool) -> FiniteElementSpace {
        let coords = Handle::new(if axisymmetric {
            Coords::axisymmetric()
        } else {
            Coords::new(dim as u8).unwrap()
        });
        let (ty, pts): (ElementType, Vec<Vec<f64>>) = if dim == 2 {
            let o = if axisymmetric { 1.0 } else { 0.0 };
            (
                ElementType::TRI3,
                vec![vec![o, 0.0], vec![o + 1.0, 0.0], vec![o, 1.0]],
            )
        } else {
            (
                ElementType::TET4,
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
            )
        };
        let ids: Vec<_> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ty);
        sm.add_cell(&ids).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    /// `T` everywhere, and a material carrying `alpha` (or `E` alone when
    /// `alpha` is `None`, the material that lacks it).
    fn fields(
        fes: &FiniteElementSpace,
        t: f64,
        alpha: Option<f64>,
    ) -> (ElementField, ElementField) {
        let temp = ElementField::new(fes, vec!["T".into()]).unwrap();
        temp.get(0).unwrap().write().set_uniform("T", t).unwrap();
        let (name, value) = alpha.map_or(("E", 1.0), |a| ("alpha", a));
        let mat = ElementField::new(fes, vec![name.into()]).unwrap();
        mat.get(0)
            .unwrap()
            .write()
            .set_uniform(name, value)
            .unwrap();
        (temp, mat)
    }

    fn elasticity(fes: &FiniteElementSpace, k: Kinematics, m: StrainMeasure) -> Model {
        model::elasticity(fes, k, MaterialSymmetry::Isotropic, m).unwrap()
    }

    /// Uniform heating ΔT = 50, α = 1e-5 ⇒ every dilated normal component is
    /// 5e-4, every shear 0 — the out-of-plane one dilating except in plane
    /// stress — in the layout of `deformation`.
    #[test]
    fn uniform_heating_dilates_each_kinematics_as_it_should() {
        let cases = [
            (one_cell(2, false), Kinematics::PlaneStrain, 4, 1.0),
            (one_cell(2, false), Kinematics::PlaneStress, 4, 0.0),
            (one_cell(2, true), Kinematics::Axisymmetric, 4, 1.0),
            (one_cell(3, false), Kinematics::Full3D, 6, 1.0),
        ];
        for (fes, k, count, zz_share) in cases {
            let (temp, mat) = fields(&fes, 70.0, Some(1e-5));
            let m = elasticity(&fes, k, StrainMeasure::Linear);
            let eps = thermal_strain(&m, &temp, &mat, 20.0).unwrap();
            let s = eps.get(0).unwrap().read();
            // The very names `deformation` gives the law's strain.
            let shape = crate::ops::element_field::deformation(
                &crate::containers::node_field::NodeField::new(
                    &fes.mesh().unwrap(),
                    (0..fes.get(0).unwrap().read().space_dim())
                        .map(crate::models::tensor::primal_name)
                        .collect(),
                )
                .unwrap(),
                &fes,
                StrainMeasure::Linear,
            )
            .unwrap();
            assert_eq!(s.components(), shape.get(0).unwrap().read().components());
            assert_eq!(s.components().len(), count, "{k:?}");
            for g in 0..s.gauss_count() {
                for c in s.components() {
                    let v = s.value(0, g, c).unwrap();
                    let expect = match c.as_str() {
                        "eps_zz" => zz_share * 5e-4,
                        "eps_xx" | "eps_yy" => 5e-4,
                        _ => 0.0,
                    };
                    assert!((v - expect).abs() < 1e-18, "{k:?} {c}: {v} ≠ {expect}");
                }
            }
        }
    }

    /// The dilation is written in the measure the law reads: `½(λ² − 1)` for
    /// Green–Lagrange, `ln λ` for Hencky, under the law's own prefix.
    #[test]
    fn the_dilation_follows_the_measure_of_the_law() {
        let fes = one_cell(2, false);
        let (temp, mat) = fields(&fes, 70.0, Some(1e-3)); // α·ΔT = 0.05
        let lambda: f64 = 1.05;
        for (measure, name, expect) in [
            (StrainMeasure::Linear, "eps_xx", 0.05),
            (
                StrainMeasure::GreenLagrange,
                "E_xx",
                0.5 * (lambda * lambda - 1.0),
            ),
            (StrainMeasure::Hencky, "H_zz", lambda.ln()),
        ] {
            let m = elasticity(&fes, Kinematics::PlaneStrain, measure);
            let eps = thermal_strain(&m, &temp, &mat, 20.0).unwrap();
            let v = eps.get(0).unwrap().read().value(0, 0, name).unwrap();
            assert!((v - expect).abs() < 1e-15, "{name}: {v} ≠ {expect}");
        }
    }

    /// T = T_ref ⇒ no thermal strain.
    #[test]
    fn no_temperature_change_gives_zero_strain() {
        let fes = one_cell(2, false);
        let (temp, mat) = fields(&fes, 20.0, Some(1e-5));
        let m = elasticity(&fes, Kinematics::PlaneStrain, StrainMeasure::Linear);
        let eps = thermal_strain(&m, &temp, &mat, 20.0).unwrap();
        let s = eps.get(0).unwrap().read();
        for g in 0..s.gauss_count() {
            for comp in s.components() {
                assert!(s.value(0, g, comp).unwrap().abs() < 1e-18);
            }
        }
    }

    /// A model with no continuum law yields an empty field.
    #[test]
    fn a_conduction_dilates_nothing() {
        let fes = one_cell(2, false);
        let (temp, mat) = fields(&fes, 70.0, Some(1e-5));
        let heat = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        assert_eq!(thermal_strain(&heat, &temp, &mat, 20.0).unwrap().len(), 0);
    }

    /// Missing `alpha` in the material ⇒ actionable error.
    #[test]
    fn missing_alpha_errors() {
        let fes = one_cell(2, false);
        let (temp, mat) = fields(&fes, 50.0, None);
        let m = elasticity(&fes, Kinematics::PlaneStrain, StrainMeasure::Linear);
        let err = thermal_strain(&m, &temp, &mat, 20.0).unwrap_err();
        assert!(format!("{err}").contains("alpha"), "unexpected: {err}");
    }
}
