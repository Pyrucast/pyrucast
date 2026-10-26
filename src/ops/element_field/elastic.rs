//! The elastic operator of a model at its Gauss points, both ways: the stress
//! `σ = D:ε` of a strain, and the strain `ε = D⁻¹:σ` of a stress — Cast3M's
//! `ELAS`.
//!
//! `D` is the operator each continuum law **predicts with**, not the law's own
//! response: the Hooke matrix of an elasticity in its symmetry, the isotropic
//! `D` behind a plasticity's elastic predictor, the undamaged `D` a damage law
//! degrades (see [`Behavior::elastic_operator`](crate::models::Behavior::elastic_operator)). That is what an incremental
//! solver needs it for — turning an imposed strain increment (thermal, given)
//! into the stress it would cause if nothing yielded, or bringing a stress back
//! to the elastic strain that carries it under another set of moduli.
//!
//! Two functions rather than one that reads the direction off the component
//! names: the two directions are not the same operation (`D⁻¹` can be singular
//! where `D` cannot), and the state field a law returns carries `sigma_*` **and**
//! `eps_*` side by side, so the names alone could not say which was meant.
//! Both run the same private zone driver on
//! [`kernel::element_pointwise`](fn@crate::models::kernel::element_pointwise).
//!
//! A sub-model with no elastic operator (a conduction, a bar, a beam, a
//! constraint) contributes no zone, the way it contributes no term to
//! [`internal_forces`](fn@crate::ops::node_field::internal_forces).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::model::Model;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::continuum::{elastic, voigt};
use crate::models::kernel;
use crate::models::symmetry::elastic_constitutive_into;

/// Which way the operator is applied. Private: each public function is one
/// direction, and this only lets them share the zone driver.
#[derive(Clone, Copy)]
enum Direction {
    /// `σ = D:ε` — reads the strain, writes the stress.
    Stress,
    /// `ε = D⁻¹:σ` — reads the stress, writes the strain.
    Strain,
}

/// Elastic stress `σ = D:ε` of a strain field, at the Gauss points of every
/// continuum sub-model of `model`.
///
/// - `strain` carries the strain in **tensor** convention, the layout of
///   [`deformation`](fn@crate::ops::element_field::deformation) and
///   [`thermal_strain`](fn@crate::ops::element_field::thermal_strain)
///   (`eps_xx`, `eps_xy`, … ; the hoop `eps_zz` on a body of revolution);
/// - `materials` carries each law's elastic constants (`E`, `nu` for an
///   isotropic one, the orthotropic or anisotropic set otherwise), resolved
///   zone by zone by those components alone — a plasticity's `sigma_y` need
///   not be there.
///
/// Returns one zone per continuum sub-model, its components the stresses that
/// law writes (`sigma_xx`, `sigma_yy`, `sigma_xy` in plane stress). `D` is the
/// operator of [`Behavior::elastic_operator`](crate::models::Behavior::elastic_operator): for a damage law, the undamaged
/// one.
///
/// ```
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, model};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// // A plasticity: its elastic operator is the isotropic D, and `sigma_y`
/// // plays no part in it.
/// let plaque = model::plasticity_perfect(&fes, Kinematics::PlaneStress, StrainMeasure::Linear)?;
/// let materiau = element_field::material_field(&plaque, &[("E", 200.0), ("nu", 0.0), ("sigma_y", 1.0)])?;
/// let eps = ElementField::new(&fes, vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()])?;
/// eps.get(0)?.write().set_uniform("eps_xx", 1e-3)?;
/// // Far beyond the yield stress, yet purely elastic: σ = E·ε with ν = 0.
/// let sigma = element_field::elastic_stress(&plaque, &eps, &materiau)?;
/// assert!((sigma.get(0)?.read().value(0, 0, "sigma_xx")? - 0.2).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elastic_stress(
    model: &Model,
    strain: &ElementField,
    materials: &ElementField,
) -> Result<ElementField> {
    apply(model, strain, materials, Direction::Stress)
}

/// Elastic strain `ε = D⁻¹:σ` of a stress field, at the Gauss points of every
/// continuum sub-model of `model` — the inverse of [`elastic_stress`].
///
/// `stress` carries the stresses each law writes (`sigma_xx`, …, the state
/// field of [`integrate_behavior`](fn@crate::ops::element_field::behavior::integrate)
/// will do); `materials` the elastic constants, as for `elastic_stress`. The
/// result is in **tensor** convention, the layout of
/// [`deformation`](fn@crate::ops::element_field::deformation), so it subtracts
/// from a strain component for component. In plane stress the out-of-plane
/// `eps_zz`, which the plane law never reads, is not written.
///
/// Errors when `D` is singular at a point (`E = 0`, say): the stress then has no
/// elastic strain to go back to.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, model};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// let plaque = model::elasticity(&fes, Kinematics::PlaneStress, MaterialSymmetry::Isotropic, StrainMeasure::Linear)?;
/// let materiau = element_field::material_field(&plaque, &[("E", 100.0), ("nu", 0.25)])?;
/// let sigma = ElementField::new(&fes, vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()])?;
/// sigma.get(0)?.write().set_uniform("sigma_xx", 1.0)?;
/// let eps = element_field::elastic_strain(&plaque, &sigma, &materiau)?;
/// // Uniaxial plane stress: ε_xx = σ/E, ε_yy = −νσ/E.
/// let e = eps.get(0)?;
/// assert!((e.read().value(0, 0, "eps_xx")? - 0.01).abs() < 1e-14);
/// assert!((e.read().value(0, 0, "eps_yy")? + 0.0025).abs() < 1e-14);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elastic_strain(
    model: &Model,
    stress: &ElementField,
    materials: &ElementField,
) -> Result<ElementField> {
    apply(model, stress, materials, Direction::Strain)
}

/// The zone driver both directions share.
fn apply(
    model: &Model,
    field: &ElementField,
    materials: &ElementField,
    direction: Direction,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for h in model {
        // What the zone needs, read under the sub-model lock and released
        // before any field lock is taken.
        let zone = {
            let sub = h.read();
            sub.as_kind().as_behavior().and_then(|b| {
                b.elastic_operator().map(|(continuum, symmetry)| {
                    (
                        continuum.fespace(),
                        continuum.space_dim(),
                        continuum.kinematics(),
                        continuum.strain_reads(),
                        continuum.stress_names(),
                        symmetry,
                    )
                })
            })
        };
        let Some((fespace, space_dim, kinematics, strain_names, stress_names, symmetry)) = zone
        else {
            continue; // no elastic operator: no zone, as for internal forces
        };

        let (reads, writes) = match direction {
            Direction::Stress => (strain_names, stress_names),
            Direction::Strain => (stress_names, strain_names),
        };
        let contract: Vec<String> = elastic::material_contract(symmetry, space_dim)
            .iter()
            .map(|c| c.to_string())
            .collect();
        let input = field.sub_for_fespace_with(&fespace, &reads)?;
        let material = materials.sub_for_fespace_with(&fespace, &contract)?;

        // Where each value sits, resolved once for the zone.
        let read_refs: Vec<&str> = reads.iter().map(String::as_str).collect();
        let in_idx = input
            .read()
            .resolve_components(&read_refs, "elastic: input")?;
        let contract_refs: Vec<&str> = contract.iter().map(String::as_str).collect();
        let mat_idx = material
            .read()
            .resolve_components(&contract_refs, "elastic: material")?;
        let n_normal = voigt::normal_count(space_dim, kinematics);

        let zone = kernel::element_pointwise(
            &fespace,
            &input,
            // The operator reads no previous state; the driver wants a field,
            // so it gets one it will never index.
            &input,
            &material,
            writes,
            |_geom, _g, row, _prev, mat_row, out| {
                let mut d = [[0.0_f64; 6]; 6];
                let v = elastic_constitutive_into(
                    mat_row, &mat_idx, symmetry, kinematics, space_dim, &mut d,
                )?;
                match direction {
                    Direction::Stress => {
                        // Tensor strain → engineering Voigt, the convention `D` is in.
                        let eps = voigt::read_voigt_strain(row, &in_idx, space_dim, kinematics);
                        for (r, slot) in out.iter_mut().enumerate().take(v) {
                            *slot = (0..v).map(|c| d[r][c] * eps[c]).sum();
                        }
                    }
                    Direction::Strain => {
                        let mut x = [0.0_f64; 6];
                        for (r, &i) in in_idx.iter().enumerate() {
                            x[r] = row[i as usize];
                        }
                        solve_voigt(&mut d, &mut x, v)?;
                        // Engineering shears back to tensor convention.
                        for (r, slot) in out.iter_mut().enumerate().take(v) {
                            *slot = if r < n_normal { x[r] } else { 0.5 * x[r] };
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.add_sub(Handle::new(zone))?;
    }
    Ok(out)
}

/// Solve `D x = b` in place on the `v × v` leading block, Gaussian elimination
/// with partial pivoting — at most six unknowns, on the stack.
///
/// A pivot that vanishes against the matrix's own scale is a singular `D`: the
/// material gave no stiffness in some direction, and the stress has no elastic
/// strain to go back to.
fn solve_voigt(d: &mut [[f64; 6]; 6], b: &mut [f64; 6], v: usize) -> Result<()> {
    let scale = d[..v]
        .iter()
        .flat_map(|row| row[..v].iter())
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    for col in 0..v {
        let pivot = (col..v)
            .max_by(|&a, &b| d[a][col].abs().total_cmp(&d[b][col].abs()))
            .unwrap_or(col);
        if d[pivot][col].abs() <= 1e-14 * scale {
            return Err(PyrucastError::Message(
                "elastic_strain: the elastic operator is singular at a Gauss point \
                 (a zero modulus?) — the stress has no elastic strain"
                    .into(),
            ));
        }
        d.swap(col, pivot);
        b.swap(col, pivot);
        for r in col + 1..v {
            let f = d[r][col] / d[col][col];
            for c in col..v {
                d[r][c] -= f * d[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    for r in (0..v).rev() {
        let s: f64 = (r + 1..v).map(|c| d[r][c] * b[c]).sum();
        b[r] = (b[r] - s) / d[r][r];
    }
    Ok(())
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
    use crate::models::tensor::Kinematics;
    use crate::ops::element_field::behavior::integrate;
    use crate::ops::element_field::material_field;
    use crate::ops::element_field::StrainMeasure;
    use crate::ops::model;

    fn tri3(dim: usize, axisymmetric: bool) -> FiniteElementSpace {
        let coords = if axisymmetric {
            Handle::new(Coords::axisymmetric().unwrap())
        } else {
            Handle::new(Coords::new(dim as u8).unwrap())
        };
        let pts: Vec<Vec<f64>> = if dim == 2 {
            vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![1.0, 1.0]]
        } else {
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        };
        let n: Vec<Node> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let et = if dim == 2 {
            ElementType::TRI3
        } else {
            ElementType::TET4
        };
        let mut sm = SubMesh::new(coords, et);
        let ids: Vec<_> = n.iter().map(Node::id).collect();
        sm.add_cell(&ids).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    /// A strain with every component non-zero, in the layout of `deformation`.
    fn strain(fes: &FiniteElementSpace, names: &[&str]) -> ElementField {
        let f = ElementField::new(fes, names.iter().map(|s| s.to_string()).collect()).unwrap();
        for (k, name) in names.iter().enumerate() {
            f.get(0)
                .unwrap()
                .write()
                .set_uniform(name, 1e-3 * (k as f64 + 1.0))
                .unwrap();
        }
        f
    }

    /// The independent oracle: a linear elasticity's own law is `σ = D:ε`, so
    /// `elastic_stress` must reproduce `integrate_behavior` exactly, in every
    /// kinematics and symmetry.
    fn matches_the_linear_law(
        fes: &FiniteElementSpace,
        kinematics: Kinematics,
        symmetry: MaterialSymmetry,
        materials: &[(&str, f64)],
        names: &[&str],
    ) {
        let m = model::elasticity(fes, kinematics, symmetry, StrainMeasure::Linear).unwrap();
        let mat = material_field(&m, materials).unwrap();
        let eps = strain(fes, names);
        let ours = elastic_stress(&m, &eps, &mat).unwrap();
        let law = integrate(&m, &eps, None, &mat, None).unwrap();
        let (o, l) = (ours.get(0).unwrap(), law.get(0).unwrap());
        let (o, l) = (o.read(), l.read());
        for c in o.components() {
            for g in 0..o.gauss_count() {
                let (a, b) = (o.value(0, g, c).unwrap(), l.value(0, g, c).unwrap());
                assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{c}: {a} ≠ {b}");
            }
        }
    }

    #[test]
    fn stress_is_the_linear_law_in_every_kinematics() {
        let iso = [("E", 210e3), ("nu", 0.3)];
        let plane = ["eps_xx", "eps_xy", "eps_yy"];
        for k in [Kinematics::PlaneStress, Kinematics::PlaneStrain] {
            matches_the_linear_law(
                &tri3(2, false),
                k,
                MaterialSymmetry::Isotropic,
                &iso,
                &plane,
            );
        }
        matches_the_linear_law(
            &tri3(2, true),
            Kinematics::Axisymmetric,
            MaterialSymmetry::Isotropic,
            &iso,
            &["eps_xx", "eps_xy", "eps_yy", "eps_zz"],
        );
        matches_the_linear_law(
            &tri3(3, false),
            Kinematics::Full3D,
            MaterialSymmetry::Isotropic,
            &iso,
            &["eps_xx", "eps_xy", "eps_xz", "eps_yy", "eps_yz", "eps_zz"],
        );
    }

    #[test]
    fn stress_is_the_linear_law_for_an_orthotropic_material() {
        let ortho = [
            ("E_1", 150e3),
            ("E_2", 10e3),
            ("E_3", 10e3),
            ("nu_12", 0.3),
            ("nu_13", 0.3),
            ("nu_23", 0.4),
            ("G_12", 5e3),
            ("G_13", 5e3),
            ("G_23", 3.5e3),
            ("V1X", 0.6),
            ("V1Y", 0.8),
        ];
        matches_the_linear_law(
            &tri3(2, false),
            Kinematics::PlaneStrain,
            MaterialSymmetry::Orthotropic,
            &ortho,
            &["eps_xx", "eps_xy", "eps_yy"],
        );
    }

    /// `elastic_strain` undoes `elastic_stress`, shears included (the tensor ↔
    /// engineering factor of two is applied both ways).
    #[test]
    fn strain_inverts_stress() {
        for (fes, k, names) in [
            (
                tri3(2, false),
                Kinematics::PlaneStress,
                vec!["eps_xx", "eps_xy", "eps_yy"],
            ),
            (
                tri3(2, true),
                Kinematics::Axisymmetric,
                vec!["eps_xx", "eps_xy", "eps_yy", "eps_zz"],
            ),
            (
                tri3(3, false),
                Kinematics::Full3D,
                vec!["eps_xx", "eps_xy", "eps_xz", "eps_yy", "eps_yz", "eps_zz"],
            ),
        ] {
            let m = model::elasticity(&fes, k, MaterialSymmetry::Isotropic, StrainMeasure::Linear)
                .unwrap();
            let mat = material_field(&m, &[("E", 210e3), ("nu", 0.3)]).unwrap();
            let eps = strain(&fes, &names);
            let back = elastic_strain(&m, &elastic_stress(&m, &eps, &mat).unwrap(), &mat).unwrap();
            let (b, e) = (back.get(0).unwrap(), eps.get(0).unwrap());
            for name in &names {
                let (x, y) = (
                    b.read().value(0, 0, name).unwrap(),
                    e.read().value(0, 0, name).unwrap(),
                );
                assert!((x - y).abs() < 1e-15, "{k:?} {name}: {x} ≠ {y}");
            }
        }
    }

    /// A zero modulus leaves no elastic strain to go back to.
    #[test]
    fn singular_operator_is_an_error() {
        let fes = tri3(2, false);
        let m = model::elasticity(
            &fes,
            Kinematics::PlaneStress,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
        .unwrap();
        let mat = material_field(&m, &[("E", 0.0), ("nu", 0.3)]).unwrap();
        let sigma = strain(&fes, &["sigma_xx", "sigma_yy", "sigma_xy"]);
        let err = elastic_strain(&m, &sigma, &mat).unwrap_err();
        assert!(format!("{err}").contains("singular"), "{err}");
    }

    /// A model with no continuum law yields an empty field rather than an error.
    #[test]
    fn a_conduction_has_no_elastic_operator() {
        let fes = tri3(2, false);
        let m = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let mat = material_field(&m, &[("k", 1.0)]).unwrap();
        let eps = strain(&fes, &["eps_xx", "eps_xy", "eps_yy"]);
        assert!(elastic_stress(&m, &eps, &mat).unwrap().is_empty());
    }
}
