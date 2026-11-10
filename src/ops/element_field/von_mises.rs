//! The equivalent stress of a stress field at its Gauss points — von Mises by
//! default, Hill's anisotropic form when given its six coefficients.
//!
//! Hill's 1948 form weighs each difference of normal stresses and each shear:
//!
//! ```text
//! σ_eq² = F(σyy − σzz)² + G(σzz − σxx)² + H(σxx − σyy)²
//!       + 2L σyz² + 2M σxz² + 2N σxy²
//! ```
//!
//! With `F = G = H = ½` and `L = M = N = 3/2` it is von Mises, `√(3 J₂)`
//! ([`VON_MISES`], the default).
//!
//! The stress components a field carries depend on the kinematics, which only
//! the model knows: a plane-stress field has no `sigma_zz` because it is zero,
//! a plane-strain one may lack it although it is not. The model is therefore
//! the first argument, as for
//! [`elastic_stress`](fn@crate::ops::element_field::elastic_stress).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::model::Model;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::tensor::Kinematics;

/// Hill's coefficients `[F, G, H, L, M, N]` that give von Mises: `½` on the
/// normal differences, `3/2` on the shears.
pub const VON_MISES: [f64; 6] = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5];

/// Name of the one component [`von_mises`] writes.
const OUTPUT: &str = "von_mises";

/// Slots of the full stress `[xx, yy, zz, yz, xz, xy]` that a kinematics
/// carries, with the names it carries them under. The slots it leaves out are
/// zero: `zz`, `yz`, `xz` in plane stress, `yz`, `xz` in plane strain and on a
/// body of revolution (whose `zz` is the hoop).
fn carried(space_dim: usize, kinematics: Kinematics) -> &'static [(usize, &'static str)] {
    match (space_dim, kinematics) {
        (2, Kinematics::PlaneStress) => &[(0, "sigma_xx"), (1, "sigma_yy"), (5, "sigma_xy")],
        (2, _) => &[
            (0, "sigma_xx"),
            (1, "sigma_yy"),
            (2, "sigma_zz"),
            (5, "sigma_xy"),
        ],
        _ => &[
            (0, "sigma_xx"),
            (1, "sigma_yy"),
            (2, "sigma_zz"),
            (3, "sigma_yz"),
            (4, "sigma_xz"),
            (5, "sigma_xy"),
        ],
    }
}

/// Equivalent stress of `stress` at the Gauss points of every continuum
/// sub-model of `model`: von Mises with [`VON_MISES`], Hill's anisotropic form
/// with any other `coefficients = [F, G, H, L, M, N]`.
///
/// `stress` carries the stresses the laws write (`sigma_xx`, …): the state of
/// [`integrate_behavior`](fn@crate::ops::element_field::behavior::integrate) or
/// the output of [`elastic_stress`](fn@crate::ops::element_field::elastic_stress)
/// will do, whatever else they carry. The kinematics of each sub-model says
/// which components to read:
///
/// | kinematics | read | taken as zero |
/// |---|---|---|
/// | `full_3d` | the six `sigma_*` | — |
/// | `plane_stress` | `sigma_xx`, `sigma_yy`, `sigma_xy` | `zz`, `yz`, `xz` |
/// | `plane_strain` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy` | `yz`, `xz` |
/// | `axisymmetric` | `sigma_xx`, `sigma_yy`, `sigma_zz` (the hoop), `sigma_xy` | `yz`, `xz` |
///
/// Returns one zone per continuum sub-model, on its FE space and Gauss points,
/// with the single component `"von_mises"`. A sub-model with no continuum
/// stress (a conduction, a bar, a beam, a constraint) contributes no zone.
///
/// Errors:
/// - coefficients for which the form can be negative: each of `L`, `M`, `N`,
///   `F + G`, `G + H`, `H + F` and `FG + GH + HF` must be `≥ 0`, and all finite;
/// - no zone of `stress` on a sub-model's FE space carrying the components its
///   kinematics reads — in plane strain, a stress without `sigma_zz` (the
///   plasticity laws carry it; an elastic stress does not).
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
/// # use pyrucast::ops::element_field::{self, StrainMeasure, VON_MISES};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// let plaque = model::elasticity(&fes, Kinematics::PlaneStress, MaterialSymmetry::Isotropic, StrainMeasure::Linear)?;
/// let sigma = ElementField::new(&fes, vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()])?;
/// sigma.get(0)?.write().set_uniform("sigma_xy", 100.0)?;
/// // Pure shear: von Mises is √3 times the shear stress.
/// let vm = element_field::von_mises(&plaque, &sigma, &VON_MISES)?;
/// assert!((vm.get(0)?.read().value(0, 0, "von_mises")? - 100.0 * 3f64.sqrt()).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn von_mises(
    model: &Model,
    stress: &ElementField,
    coefficients: &[f64; 6],
) -> Result<ElementField> {
    let [f, g, h, l, m, n] = *coefficients;
    let non_negative = [l, m, n, f + g, g + h, h + f, f * g + g * h + h * f];
    if coefficients.iter().any(|c| !c.is_finite()) || non_negative.iter().any(|&c| c < 0.0) {
        return Err(PyrucastError::Message(format!(
            "von_mises: the coefficients [F, G, H, L, M, N] = {coefficients:?} let the \
             equivalent stress go negative — L, M, N, F + G, G + H, H + F and FG + GH + HF \
             must all be ≥ 0 (von Mises is [0.5, 0.5, 0.5, 1.5, 1.5, 1.5])"
        )));
    }

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
                        continuum.space_dim(),
                        continuum.kinematics(),
                    )
                })
            })
        };
        let Some((fespace, space_dim, kinematics)) = zone else {
            continue; // no continuum stress: no zone
        };

        let reads = carried(space_dim, kinematics);
        let names: Vec<String> = reads.iter().map(|&(_, name)| name.to_string()).collect();
        let input =
            stress
                .sub_for_fespace_with(&fespace, &names)
                .map_err(|e| match kinematics {
                    Kinematics::PlaneStrain => PyrucastError::Message(format!(
                    "von_mises: a plane-strain stress needs `sigma_zz`, which is not zero there \
                     (the plasticity laws carry it, an elastic stress does not) — {e}"
                )),
                    _ => e,
                })?;
        // Where each carried component sits in the input row, resolved once.
        let refs: Vec<&str> = reads.iter().map(|&(_, name)| name).collect();
        let at = input
            .read()
            .resolve_components(&refs, "von_mises: stress")?;
        let slots: Vec<(usize, usize)> = reads
            .iter()
            .zip(&at)
            .map(|(&(slot, _), &i)| (slot, i as usize))
            .collect();

        let zone = kernel::element_pointwise(
            &fespace,
            &input,
            // The equivalent stress reads neither a previous state nor a
            // material; the driver wants fields, so it gets one it never indexes.
            &input,
            &input,
            vec![OUTPUT.to_string()],
            |_geom, _g, row, _prev, _mat, slot| {
                let mut s = [0.0_f64; 6];
                for &(k, i) in &slots {
                    s[k] = row[i];
                }
                let [xx, yy, zz, yz, xz, xy] = s;
                let q2 = f * (yy - zz).powi(2)
                    + g * (zz - xx).powi(2)
                    + h * (xx - yy).powi(2)
                    + 2.0 * (l * yz * yz + m * xz * xz + n * xy * xy);
                // A non-negative form can still round to a hair below zero.
                slot[0] = q2.max(0.0).sqrt();
                Ok(())
            },
        )?;
        out.add_sub(Handle::new(zone))?;
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
    use crate::models::symmetry::MaterialSymmetry;
    use crate::models::tensor;
    use crate::ops::element_field::StrainMeasure;
    use crate::ops::model;

    const FULL: [&str; 6] = [
        "sigma_xx", "sigma_yy", "sigma_zz", "sigma_yz", "sigma_xz", "sigma_xy",
    ];

    fn tet4() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<Node> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id(), n[3].id()])
            .unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    fn tri3() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    fn solid(fes: &FiniteElementSpace, kinematics: Kinematics) -> Model {
        model::elasticity(
            fes,
            kinematics,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
        .unwrap()
    }

    /// A uniform stress field carrying `names` with `values`.
    fn stress(fes: &FiniteElementSpace, names: &[&str], values: &[f64]) -> ElementField {
        let field = ElementField::new(fes, names.iter().map(|s| s.to_string()).collect()).unwrap();
        for (name, &v) in names.iter().zip(values) {
            field.get(0).unwrap().write().set_uniform(name, v).unwrap();
        }
        field
    }

    fn first(vm: &ElementField) -> f64 {
        vm.get(0).unwrap().read().value(0, 0, OUTPUT).unwrap()
    }

    #[test]
    fn uniaxial_tension_gives_the_applied_stress() {
        let fes = tet4();
        let s = stress(&fes, &FULL, &[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let vm = von_mises(&solid(&fes, Kinematics::Full3D), &s, &VON_MISES).unwrap();
        assert!((first(&vm) - 300.0).abs() < 1e-9);
    }

    #[test]
    fn default_coefficients_match_the_laws_von_mises() {
        let fes = tet4();
        let values = [120.0, -40.0, 75.0, 30.0, -12.0, 55.0];
        let s = stress(&fes, &FULL, &values);
        let vm = von_mises(&solid(&fes, Kinematics::Full3D), &s, &VON_MISES).unwrap();
        // The independent reference: √(3 J₂) of the plasticity laws.
        assert!((first(&vm) - tensor::von_mises_stress(&values)).abs() < 1e-9);
    }

    #[test]
    fn hill_coefficients_weigh_each_term() {
        let fes = tet4();
        // [xx, yy, zz, yz, xz, xy]
        let s = stress(&fes, &FULL, &[1.0, 2.0, 4.0, 0.5, 0.25, 0.1]);
        let vm = von_mises(
            &solid(&fes, Kinematics::Full3D),
            &s,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        // By hand: F·(2−4)² + G·(4−1)² + H·(1−2)² + 2(L·0.25 + M·0.0625 + N·0.01)
        //        = 4 + 18 + 3 + 2 + 0.625 + 0.12
        assert!((first(&vm) - 27.745_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn plane_stress_agrees_with_3d_when_sigma_zz_is_zero() {
        let (xx, yy, xy) = (80.0, -25.0, 40.0);
        let plane = tri3();
        let s2 = stress(&plane, &["sigma_xx", "sigma_yy", "sigma_xy"], &[xx, yy, xy]);
        let vm2 = von_mises(&solid(&plane, Kinematics::PlaneStress), &s2, &VON_MISES).unwrap();
        let space = tet4();
        let s3 = stress(&space, &FULL, &[xx, yy, 0.0, 0.0, 0.0, xy]);
        let vm3 = von_mises(&solid(&space, Kinematics::Full3D), &s3, &VON_MISES).unwrap();
        assert!((first(&vm2) - first(&vm3)).abs() < 1e-9);
    }

    #[test]
    fn plane_strain_reads_sigma_zz_and_refuses_a_stress_without_it() {
        let fes = tri3();
        let plate = solid(&fes, Kinematics::PlaneStrain);
        let with_zz = stress(
            &fes,
            &["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy"],
            &[100.0, 0.0, 30.0, 0.0],
        );
        let vm = von_mises(&plate, &with_zz, &VON_MISES).unwrap();
        // √(½[(0 − 30)² + (30 − 100)² + (100 − 0)²]) = √7900
        assert!((first(&vm) - 7900f64.sqrt()).abs() < 1e-9);

        let without = stress(
            &fes,
            &["sigma_xx", "sigma_yy", "sigma_xy"],
            &[100.0, 0.0, 0.0],
        );
        let err = von_mises(&plate, &without, &VON_MISES).unwrap_err();
        assert!(format!("{err}").contains("sigma_zz"), "unexpected: {err}");
    }

    #[test]
    fn coefficients_that_can_go_negative_are_refused() {
        let fes = tet4();
        let s = stress(&fes, &FULL, &[1.0; 6]);
        let model = solid(&fes, Kinematics::Full3D);
        for bad in [
            [0.5, 0.5, 0.5, -1.0, 1.5, 1.5],
            [-2.0, 0.5, 0.5, 1.5, 1.5, 1.5],
            [0.5, f64::NAN, 0.5, 1.5, 1.5, 1.5],
        ] {
            let err = von_mises(&model, &s, &bad).unwrap_err();
            assert!(format!("{err}").contains("≥ 0"), "unexpected: {err}");
        }
        // One negative F with a form that stays positive is Hill's own right.
        assert!(von_mises(&model, &s, &[-0.1, 1.0, 1.0, 1.5, 1.5, 1.5]).is_ok());
    }
}
