//! Large transformations in total Lagrangian: a law reading the Green–Lagrange
//! strain, its stress carried to the other measures, and the internal forces
//! written with the first Piola–Kirchhoff stress on the reference
//! configuration.
//!
//! Every oracle is independent of the operator it checks:
//!
//! - a **rigid rotation** strains nothing: no strain, no stress, no force —
//!   where the linear strain of the same motion is not zero;
//! - **objectivity**: rotating a stretched state rotates its Cauchy stress;
//! - **duality**: the internal forces are the derivative of the stored energy
//!   `∫ ½ E:S`, taken by finite differences;
//! - the **operator** of the stepping — the elastic stiffness and the
//!   geometric stiffness evaluated on the current configuration — is the
//!   derivative of those forces where theory says it is exactly: after a rigid
//!   rotation for the first, at frozen `S` for the second;
//! - the **conversions** against the formulas written out with `nalgebra`.

use nalgebra::Matrix3;
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::Matrix;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::coords as ops_coords;
use pyrucast::ops::element_field::behavior::integrate;
use pyrucast::ops::element_field::{
    deformation, material_field, strain, stress_measure, StrainMeasure, StressMeasure,
};
use pyrucast::ops::measure::integral_element;
use pyrucast::ops::node_field::{external_forces, internal_forces};
use pyrucast::ops::{matrix, model};
use pyrucast::Result;

const AXES: [&str; 2] = ["x", "y"];

/// Two distorted QUA4 side by side — nothing uniform, nothing aligned.
struct Patch {
    coords: Handle<Coords>,
    mesh: Mesh,
    fes: FiniteElementSpace,
    nodes: Vec<Node>,
}

fn patch() -> Result<Patch> {
    let coords = Handle::new(Coords::new(2)?);
    let points = [
        [0.0, 0.0],
        [1.1, 0.1],
        [2.0, 0.0],
        [0.1, 1.0],
        [1.0, 1.2],
        [2.1, 0.9],
    ];
    let nodes: Vec<Node> = points
        .iter()
        .map(|p| Node::create_in(coords.clone(), p))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    mesh.add_cell(&[nodes[0].id(), nodes[1].id(), nodes[4].id(), nodes[3].id()])?;
    mesh.add_cell(&[nodes[1].id(), nodes[2].id(), nodes[5].id(), nodes[4].id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok(Patch {
        coords,
        mesh,
        fes,
        nodes,
    })
}

/// Plane-strain elasticity reading `measure`, and its material.
fn solid(p: &Patch, measure: StrainMeasure) -> Result<(Model, ElementField)> {
    let solid = model::elasticity_with_symmetry(
        &p.fes,
        Kinematics::PlaneStrain,
        MaterialSymmetry::Isotropic,
        measure,
    )?;
    let materials = material_field(&solid, &[("E", 100.0), ("nu", 0.3)])?;
    Ok((solid, materials))
}

/// The nodal displacement `values[k]` at node `k`.
fn displacement(p: &Patch, values: &[[f64; 2]]) -> Result<NodeField> {
    let u = NodeField::new(&p.mesh, vec!["u_x".into(), "u_y".into()])?;
    {
        let mut zone = u.get(0)?.write();
        for (n, v) in p.nodes.iter().zip(values) {
            zone.set_value(n.id(), "u_x", v[0])?;
            zone.set_value(n.id(), "u_y", v[1])?;
        }
    }
    Ok(u)
}

/// The displacement of the affine motion `x = A X`.
fn affine(p: &Patch, a: [[f64; 2]; 2]) -> Result<Vec<[f64; 2]>> {
    p.nodes
        .iter()
        .map(|n| {
            let x = n.position()?;
            Ok([
                a[0][0] * x[0] + a[0][1] * x[1] - x[0],
                a[1][0] * x[0] + a[1][1] * x[1] - x[1],
            ])
        })
        .collect()
}

fn rotation(theta: f64) -> [[f64; 2]; 2] {
    let (s, c) = theta.sin_cos();
    [[c, -s], [s, c]]
}

/// The law's state at `u`: what `strain` and `integrate` give.
fn state(m: &Model, mat: &ElementField, u: &NodeField) -> Result<(ElementField, ElementField)> {
    let eps = strain(m, u, mat)?;
    let s = integrate(m, &eps, None, mat, None)?;
    Ok((eps, s))
}

/// The internal forces of the law's state `s`, with the gradient of `u`:
/// `∫ P·∇₀N` with `P = F S`.
fn forces(
    p: &Patch,
    m: &Model,
    mat: &ElementField,
    s: &ElementField,
    u: &NodeField,
) -> Result<NodeField> {
    let f = deformation(u, &p.fes, StrainMeasure::Gradient)?;
    let pk1 = stress_measure(s, &f, StressMeasure::FirstPiolaKirchhoff)?;
    internal_forces(m, &pk1, u, mat)
}

/// Every value of `component` in `field`, over all zones, cells and points.
fn values(field: &ElementField, component: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for zone in field {
        let z = zone.read();
        for cell in 0..z.cell_count() {
            for g in 0..z.gauss_count() {
                out.push(z.value(cell, g, component)?);
            }
        }
    }
    Ok(out)
}

fn largest(xs: &[f64]) -> f64 {
    xs.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// The nodal force on node `k`, axis `a`.
fn force(f: &NodeField, p: &Patch, k: usize, a: usize) -> Result<f64> {
    f.value(p.nodes[k].id(), &format!("f_{}", AXES[a]))
}

/// `K e_(k,a)`: the column `(k, a)` of `K`, every node and axis.
fn column(k_mat: &Matrix, p: &Patch, k: usize, a: usize) -> Result<Vec<f64>> {
    let mut unit = vec![[0.0; 2]; p.nodes.len()];
    unit[k][a] = 1.0;
    let e = displacement(p, &unit)?;
    let ke = (k_mat * &e)?;
    let mut out = Vec::new();
    for j in 0..p.nodes.len() {
        for b in 0..2 {
            out.push(force(&ke, p, j, b)?);
        }
    }
    Ok(out)
}

/// Evaluate `build` with the configuration `x = X + u` active, then restore the
/// reference.
fn on_current<T>(p: &Patch, u: &NodeField, build: impl FnOnce() -> Result<T>) -> Result<T> {
    let current = p.coords.write().add_config("current");
    let names = Some(vec!["u_x".to_string(), "u_y".to_string()]);
    ops_coords::displace_from(u, 0, current, names)?;
    p.coords.write().select(current)?;
    let out = build();
    p.coords.write().select(0)?;
    out
}

#[test]
fn a_rigid_rotation_strains_nothing() -> Result<()> {
    let p = patch()?;
    let u = displacement(&p, &affine(&p, rotation(std::f64::consts::FRAC_PI_3))?)?;

    let (gl, mat) = solid(&p, StrainMeasure::GreenLagrange)?;
    let (eps, s) = state(&gl, &mat, &u)?;
    for name in ["E_xx", "E_yy", "E_xy"] {
        assert!(largest(&values(&eps, name)?) < 1e-14, "{name}");
    }
    let f = forces(&p, &gl, &mat, &s, &u)?;
    for k in 0..p.nodes.len() {
        for a in 0..2 {
            assert!(force(&f, &p, k, a)?.abs() < 1e-12);
        }
    }
    let grad = deformation(&u, &p.fes, StrainMeasure::Gradient)?;
    let sigma = stress_measure(&s, &grad, StressMeasure::Cauchy)?;
    assert!(largest(&values(&sigma, "sigma_xx")?) < 1e-12);

    // The linear strain of the same motion is not zero: ε = (cos θ − 1) I.
    let (linear, mat_lin) = solid(&p, StrainMeasure::Linear)?;
    let eps_lin = strain(&linear, &u, &mat_lin)?;
    let spurious = largest(&values(&eps_lin, "eps_xx")?);
    assert!((spurious - 0.5).abs() < 1e-12, "ε_xx = {spurious}");
    Ok(())
}

#[test]
fn rotating_a_stretched_state_rotates_its_cauchy_stress() -> Result<()> {
    let p = patch()?;
    let (gl, mat) = solid(&p, StrainMeasure::GreenLagrange)?;
    let stretch = [[1.2, 0.1], [0.0, 0.9]];
    let r = rotation(0.7);
    let turned = [
        [
            r[0][0] * stretch[0][0] + r[0][1] * stretch[1][0],
            r[0][0] * stretch[0][1] + r[0][1] * stretch[1][1],
        ],
        [
            r[1][0] * stretch[0][0] + r[1][1] * stretch[1][0],
            r[1][0] * stretch[0][1] + r[1][1] * stretch[1][1],
        ],
    ];
    let cauchy = |a: [[f64; 2]; 2]| -> Result<ElementField> {
        let u = displacement(&p, &affine(&p, a)?)?;
        let (_, s) = state(&gl, &mat, &u)?;
        let grad = deformation(&u, &p.fes, StrainMeasure::Gradient)?;
        stress_measure(&s, &grad, StressMeasure::Cauchy)
    };
    let (before, after) = (cauchy(stretch)?, cauchy(turned)?);
    let read = |f: &ElementField, name: &str| -> Result<f64> { Ok(values(f, name)?[0]) };
    let s0 = nalgebra::Matrix2::new(
        read(&before, "sigma_xx")?,
        read(&before, "sigma_xy")?,
        read(&before, "sigma_xy")?,
        read(&before, "sigma_yy")?,
    );
    let rm = nalgebra::Matrix2::new(r[0][0], r[0][1], r[1][0], r[1][1]);
    let expected = rm * s0 * rm.transpose();
    let got = nalgebra::Matrix2::new(
        read(&after, "sigma_xx")?,
        read(&after, "sigma_xy")?,
        read(&after, "sigma_xy")?,
        read(&after, "sigma_yy")?,
    );
    assert!(s0.norm() > 1.0, "a stretched state is stressed");
    assert!((got - expected).norm() < 1e-10 * s0.norm());
    Ok(())
}

/// A moderate, non-affine displacement of the patch.
fn some_motion(p: &Patch) -> Result<Vec<[f64; 2]>> {
    let base = affine(p, [[1.15, 0.2], [-0.1, 0.95]])?;
    Ok(base
        .iter()
        .enumerate()
        .map(|(k, v)| {
            [
                v[0] + 0.03 * (k as f64).sin(),
                v[1] - 0.02 * (k as f64).cos(),
            ]
        })
        .collect())
}

#[test]
fn the_internal_forces_are_the_derivative_of_the_stored_energy() -> Result<()> {
    // Green–Lagrange checks B(F); Hencky checks, on top, T → S.
    for (measure, strain_name, stress_name) in [
        (StrainMeasure::GreenLagrange, "E", "S"),
        (StrainMeasure::Hencky, "H", "T"),
    ] {
        let p = patch()?;
        let (law, mat) = solid(&p, measure)?;
        // ∫ ½ ε:σ in the measure, tensor convention: the shear counts twice.
        let energy = |values_u: &[[f64; 2]]| -> Result<f64> {
            let u = displacement(&p, values_u)?;
            let (eps, s) = state(&law, &mat, &u)?;
            let (e, sz) = (eps.get(0)?.read().clone(), s.get(0)?.read().clone());
            let density = ElementField::new(&p.fes, vec!["w".into()])?;
            {
                let mut z = density.get(0)?.write();
                for cell in 0..z.cell_count() {
                    for g in 0..z.gauss_count() {
                        let pair = |ij: &str| -> Result<f64> {
                            Ok(e.value(cell, g, &format!("{strain_name}_{ij}"))?
                                * sz.value(cell, g, &format!("{stress_name}_{ij}"))?)
                        };
                        let w = 0.5 * (pair("xx")? + pair("yy")? + 2.0 * pair("xy")?);
                        z.set_value(cell, g, "w", w)?;
                    }
                }
            }
            integral_element(&density, "w")
        };

        let motion = some_motion(&p)?;
        let u = displacement(&p, &motion)?;
        let (_, s) = state(&law, &mat, &u)?;
        let f = forces(&p, &law, &mat, &s, &u)?;

        let h = 1e-6;
        let mut scale = 0.0_f64;
        let mut worst = 0.0_f64;
        for k in 0..p.nodes.len() {
            for a in 0..2 {
                let (mut plus, mut minus) = (motion.clone(), motion.clone());
                plus[k][a] += h;
                minus[k][a] -= h;
                let derivative = (energy(&plus)? - energy(&minus)?) / (2.0 * h);
                let fa = force(&f, &p, k, a)?;
                scale = scale.max(fa.abs());
                worst = worst.max((derivative - fa).abs());
            }
        }
        assert!(scale > 1.0, "the patch is loaded");
        assert!(
            worst < 1e-6 * scale,
            "{measure:?}: |∂W/∂u − f_int| = {worst}, scale {scale}"
        );
    }
    Ok(())
}

/// A unit HEX8 cube, `ν = 0` elasticity reading `measure`, stretched to `λ`
/// along x with nothing else moving — the exact uniaxial state when the
/// Poisson ratio vanishes. Returns the nominal force on the face `x = 1`.
fn uniaxial_force(measure: StrainMeasure, lambda: f64) -> Result<(f64, f64)> {
    let coords = Handle::new(Coords::new(3)?);
    let mut nodes = Vec::new();
    for z in [0.0, 1.0] {
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            nodes.push(Node::create_in(coords.clone(), &[x, y, z])?);
        }
    }
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
    let ids: Vec<_> = nodes.iter().map(|n| n.id()).collect();
    mesh.add_cell(&ids)?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let law = model::elasticity_with_symmetry(
        &fes,
        Kinematics::Full3D,
        MaterialSymmetry::Isotropic,
        measure,
    )?;
    let mat = material_field(&law, &[("E", 100.0), ("nu", 0.0)])?;
    let u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into(), "u_z".into()])?;
    {
        let mut z = u.get(0)?.write();
        for n in &nodes {
            z.set_value(n.id(), "u_x", (lambda - 1.0) * n.position()?[0])?;
        }
    }
    let (_, s) = state(&law, &mat, &u)?;
    let grad = deformation(&u, &fes, StrainMeasure::Gradient)?;
    let pk1 = stress_measure(&s, &grad, StressMeasure::FirstPiolaKirchhoff)?;
    let f = internal_forces(&law, &pk1, &u, &mat)?;
    let mut on_face = 0.0;
    for n in &nodes {
        if n.position()?[0] == 1.0 {
            on_face += f.value(n.id(), "f_x")?;
        }
    }
    Ok((on_face, values(&pk1, "P_xx")?[0]))
}

#[test]
fn a_uniaxial_stretch_gives_the_closed_form_nominal_force() -> Result<()> {
    // E = 100, A₀ = 1, λ = 1.5: Saint-Venant–Kirchhoff, S = E·E_xx with
    // E_xx = ½(λ² − 1), gives P = λ S; Hencky, T = E ln λ, S = T/λ², gives
    // P = E ln λ / λ.
    let (lambda, young) = (1.5_f64, 100.0);
    let expected = [
        (
            StrainMeasure::GreenLagrange,
            lambda * young * 0.5 * (lambda * lambda - 1.0),
        ),
        (StrainMeasure::Hencky, young * lambda.ln() / lambda),
    ];
    for (measure, nominal) in expected {
        let (force, p_xx) = uniaxial_force(measure, lambda)?;
        assert!(
            (p_xx - nominal).abs() < 1e-12 * nominal,
            "{measure:?}: P_xx = {p_xx}"
        );
        assert!(
            (force - nominal).abs() < 1e-12 * nominal,
            "{measure:?}: F = {force}"
        );
    }
    Ok(())
}

/// Finite-difference column `(k, a)` of `f(u)`, every node and axis.
fn fd_column(
    p: &Patch,
    motion: &[[f64; 2]],
    k: usize,
    a: usize,
    f: impl Fn(&NodeField) -> Result<NodeField>,
) -> Result<Vec<f64>> {
    let h = 1e-6;
    let (mut plus, mut minus) = (motion.to_vec(), motion.to_vec());
    plus[k][a] += h;
    minus[k][a] -= h;
    let (fp, fm) = (f(&displacement(p, &plus)?)?, f(&displacement(p, &minus)?)?);
    let mut out = Vec::new();
    for j in 0..p.nodes.len() {
        for b in 0..2 {
            out.push((force(&fp, p, j, b)? - force(&fm, p, j, b)?) / (2.0 * h));
        }
    }
    Ok(out)
}

fn compare(expected: &[f64], got: &[f64]) -> (f64, f64) {
    let scale = largest(expected);
    let worst = expected
        .iter()
        .zip(got)
        .fold(0.0_f64, |m, (e, g)| m.max((e - g).abs()));
    (worst, scale)
}

#[test]
fn after_a_rigid_rotation_the_stiffness_on_the_current_configuration_is_the_tangent() -> Result<()>
{
    let p = patch()?;
    let (gl, mat) = solid(&p, StrainMeasure::GreenLagrange)?;
    let motion = affine(&p, rotation(1.1))?;
    let u = displacement(&p, &motion)?;
    let k_current = on_current(&p, &u, || matrix::stiffness(&gl, &mat))?;

    for k in 0..p.nodes.len() {
        for a in 0..2 {
            let derivative = fd_column(&p, &motion, k, a, |v| {
                let (_, s) = state(&gl, &mat, v)?;
                forces(&p, &gl, &mat, &s, v)
            })?;
            let (worst, scale) = compare(&derivative, &column(&k_current, &p, k, a)?);
            assert!(
                worst < 1e-6 * scale,
                "column ({k}, {a}): {worst} of {scale}"
            );
        }
    }
    Ok(())
}

#[test]
fn the_geometric_stiffness_on_the_current_configuration_is_the_lagrangian_one() -> Result<()> {
    let p = patch()?;
    let (gl, mat) = solid(&p, StrainMeasure::GreenLagrange)?;
    let motion = some_motion(&p)?;
    let u = displacement(&p, &motion)?;
    let (_, frozen) = state(&gl, &mat, &u)?;
    let grad = deformation(&u, &p.fes, StrainMeasure::Gradient)?;
    let sigma = stress_measure(&frozen, &grad, StressMeasure::Cauchy)?;
    let k_sigma = on_current(&p, &u, || matrix::geometric(&gl, &mat, &sigma))?;

    // At frozen S, ∂/∂u ∫ (F S)·∇₀N is ∫ ∇₀N·S·∇₀N — the Lagrangian K_σ.
    for k in 0..p.nodes.len() {
        for a in 0..2 {
            let derivative = fd_column(&p, &motion, k, a, |v| forces(&p, &gl, &mat, &frozen, v))?;
            let (worst, scale) = compare(&derivative, &column(&k_sigma, &p, k, a)?);
            assert!(scale > 1e-3, "the state is stressed");
            assert!(
                worst < 1e-6 * scale,
                "column ({k}, {a}): {worst} of {scale}"
            );
        }
    }
    Ok(())
}

#[test]
fn at_rest_every_measure_is_the_same_and_current_is_the_reference() -> Result<()> {
    let p = patch()?;
    let (gl, mat) = solid(&p, StrainMeasure::GreenLagrange)?;
    let rest = displacement(&p, &vec![[0.0; 2]; p.nodes.len()])?;
    let grad = deformation(&rest, &p.fes, StrainMeasure::Gradient)?;
    let s = ElementField::new(&p.fes, vec!["S_xx".into(), "S_xy".into(), "S_yy".into()])?;
    {
        let mut z = s.get(0)?.write();
        z.set_uniform("S_xx", 3.0)?;
        z.set_uniform("S_xy", -2.0)?;
        z.set_uniform("S_yy", 5.0)?;
    }
    let pk1 = stress_measure(&s, &grad, StressMeasure::FirstPiolaKirchhoff)?;
    let sigma = stress_measure(&s, &grad, StressMeasure::Cauchy)?;
    for (name, want) in [("P_xx", 3.0), ("P_xy", -2.0), ("P_yx", -2.0), ("P_yy", 5.0)] {
        assert!(values(&pk1, name)?.iter().all(|v| (v - want).abs() < 1e-14));
    }
    for (name, want) in [("sigma_xx", 3.0), ("sigma_xy", -2.0), ("sigma_yy", 5.0)] {
        assert!(values(&sigma, name)?
            .iter()
            .all(|v| (v - want).abs() < 1e-14));
    }
    // At rest the law's forces are the small-strain ones: ∫ Bᵀ S.
    let (linear, mat_lin) = solid(&p, StrainMeasure::Linear)?;
    let as_sigma = stress_measure(&s, &grad, StressMeasure::Cauchy)?;
    let f_tl = internal_forces(&gl, &pk1, &rest, &mat)?;
    let f_lin = internal_forces(&linear, &as_sigma, &rest, &mat_lin)?;
    for k in 0..p.nodes.len() {
        for a in 0..2 {
            assert!((force(&f_tl, &p, k, a)? - force(&f_lin, &p, k, a)?).abs() < 1e-13);
        }
    }
    let k_current = on_current(&p, &rest, || matrix::stiffness(&gl, &mat))?;
    let k_reference = matrix::stiffness(&gl, &mat)?;
    assert_eq!(k_current.dense()?, k_reference.dense()?);
    Ok(())
}

#[test]
fn the_conversions_follow_their_formulas() -> Result<()> {
    let p = patch()?;
    let a = [[1.3, 0.4], [-0.2, 0.8]];
    let u = displacement(&p, &affine(&p, a)?)?;
    let grad = deformation(&u, &p.fes, StrainMeasure::Gradient)?;
    // A plane-strain second Piola–Kirchhoff stress, out-of-plane component
    // included.
    let s = ElementField::new(
        &p.fes,
        vec!["S_xx".into(), "S_xy".into(), "S_yy".into(), "S_zz".into()],
    )?;
    let s_m = Matrix3::new(7.0, -3.0, 0.0, -3.0, 2.0, 0.0, 0.0, 0.0, 4.0);
    {
        let mut z = s.get(0)?.write();
        for (name, (i, j)) in [
            ("S_xx", (0, 0)),
            ("S_xy", (0, 1)),
            ("S_yy", (1, 1)),
            ("S_zz", (2, 2)),
        ] {
            z.set_uniform(name, s_m[(i, j)])?;
        }
    }
    let f = Matrix3::new(a[0][0], a[0][1], 0.0, a[1][0], a[1][1], 0.0, 0.0, 0.0, 1.0);
    let j = f.determinant();
    let sigma_m = f * s_m * f.transpose() / j;
    let p_m = f * s_m;

    let sigma = stress_measure(&s, &grad, StressMeasure::Cauchy)?;
    let tau = stress_measure(&s, &grad, StressMeasure::Kirchhoff)?;
    let pk1 = stress_measure(&s, &grad, StressMeasure::FirstPiolaKirchhoff)?;
    let close = |f: &ElementField, name: &str, want: f64| -> Result<()> {
        for v in values(f, name)? {
            assert!(
                (v - want).abs() < 1e-12 * want.abs().max(1.0),
                "{name}: {v} ≠ {want}"
            );
        }
        Ok(())
    };
    for (suffix, (r, c)) in [
        ("xx", (0, 0)),
        ("xy", (0, 1)),
        ("yy", (1, 1)),
        ("zz", (2, 2)),
    ] {
        close(&sigma, &format!("sigma_{suffix}"), sigma_m[(r, c)])?;
        close(&tau, &format!("tau_{suffix}"), j * sigma_m[(r, c)])?;
    }
    for (suffix, (r, c)) in [
        ("xx", (0, 0)),
        ("xy", (0, 1)),
        ("yx", (1, 0)),
        ("yy", (1, 1)),
        ("zz", (2, 2)),
    ] {
        close(&pk1, &format!("P_{suffix}"), p_m[(r, c)])?;
    }
    // And back: every measure returns the S it came from.
    for from in [&sigma, &tau, &pk1] {
        let back = stress_measure(from, &grad, StressMeasure::SecondPiolaKirchhoff)?;
        for (suffix, (r, c)) in [
            ("xx", (0, 0)),
            ("xy", (0, 1)),
            ("yy", (1, 1)),
            ("zz", (2, 2)),
        ] {
            close(&back, &format!("S_{suffix}"), s_m[(r, c)])?;
        }
    }
    Ok(())
}

#[test]
fn the_measures_a_law_cannot_read_are_refused() -> Result<()> {
    let p = patch()?;
    let build = |kinematics, measure| {
        model::elasticity_with_symmetry(&p.fes, kinematics, MaterialSymmetry::Isotropic, measure)
    };
    // Hencky needs C_zz, which plane stress leaves to the law.
    assert!(build(Kinematics::PlaneStress, StrainMeasure::Hencky).is_err());
    assert!(build(Kinematics::PlaneStress, StrainMeasure::GreenLagrange).is_ok());
    // F is a kinematic quantity, not a strain a law reads.
    assert!(build(Kinematics::PlaneStrain, StrainMeasure::Gradient).is_err());
    // A stress field carrying two measures has no single one to convert.
    let both = ElementField::new(
        &p.fes,
        vec![
            "S_xx".into(),
            "S_xy".into(),
            "S_yy".into(),
            "sigma_xx".into(),
            "sigma_xy".into(),
            "sigma_yy".into(),
        ],
    )?;
    let rest = displacement(&p, &vec![[0.0; 2]; p.nodes.len()])?;
    let grad = deformation(&rest, &p.fes, StrainMeasure::Gradient)?;
    assert!(stress_measure(&both, &grad, StressMeasure::Cauchy).is_err());
    Ok(())
}

#[test]
fn displace_from_lays_the_current_configuration_in_absolute() -> Result<()> {
    let p = patch()?;
    let current = p.coords.write().add_config("current");
    let names = || Some(vec!["u_x".to_string(), "u_y".to_string()]);
    // Only some nodes move; the others stay at their reference position.
    let some = NodeField::new(
        &Mesh::from_submesh(SubMesh::poi1_from_nodes(&p.nodes[..2])?),
        vec!["u_x".into(), "u_y".into()],
    )?;
    some.get(0)?
        .write()
        .set_value(p.nodes[1].id(), "u_y", 0.25)?;
    for _ in 0..3 {
        ops_coords::displace_from(&some, 0, current, names())?;
    }
    p.coords.write().select(current)?;
    let moved = p.nodes[1].position()?;
    let still = p.nodes[5].position()?;
    p.coords.write().select(0)?;
    assert_eq!(moved, vec![1.1, 0.35]);
    assert_eq!(still, p.nodes[5].position()?);
    assert!(ops_coords::displace_from(&some, current, current, names()).is_err());
    Ok(())
}

#[test]
fn a_pressure_follows_the_normal_of_the_active_configuration() -> Result<()> {
    // A unit square and its contour, travelled counter-clockwise.
    let coords = Handle::new(Coords::new(2)?);
    let nodes: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p))
        .collect::<Result<_>>()?;
    let ids: Vec<_> = nodes.iter().map(|n| n.id()).collect();
    let mut body = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    body.add_cell(&ids)?;
    let mut contour = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for k in 0..4 {
        contour.add_cell(&[ids[k], ids[(k + 1) % 4]])?;
    }
    let solid_model = model::elasticity(
        &FiniteElementSpace::lagrange1(&body)?,
        Kinematics::PlaneStrain,
    )?;
    let load = model::flux(
        &FiniteElementSpace::lagrange1(&contour)?,
        &solid_model,
        "n".into(),
    )?;
    assert_eq!(load.dual_vars(), vec!["f_x".to_string(), "f_y".to_string()]);
    let density = material_field(&load, &[("phi_n", 2.0)])?;
    let at = |f: &NodeField, k: usize| -> Result<[f64; 2]> {
        Ok([f.value(ids[k], "f_x")?, f.value(ids[k], "f_y")?])
    };

    let reference = external_forces(&load, &density)?;
    // A closed contour: no resultant. Each corner gets φ·½ along each of its
    // two outward normals.
    let mut total = [0.0, 0.0];
    for k in 0..4 {
        let f = at(&reference, k)?;
        total[0] += f[0];
        total[1] += f[1];
    }
    assert!(total[0].abs() < 1e-14 && total[1].abs() < 1e-14);
    assert_eq!(at(&reference, 0)?, [-1.0, -1.0]);
    assert_eq!(at(&reference, 2)?, [1.0, 1.0]);

    // On a configuration turned by R, the load turns with the wall.
    let (s, c) = 0.9_f64.sin_cos();
    let u = NodeField::new(&body, vec!["u_x".into(), "u_y".into()])?;
    {
        let mut z = u.get(0)?.write();
        for n in &nodes {
            let x = n.position()?;
            z.set_value(n.id(), "u_x", c * x[0] - s * x[1] - x[0])?;
            z.set_value(n.id(), "u_y", s * x[0] + c * x[1] - x[1])?;
        }
    }
    let current = coords.write().add_config("current");
    ops_coords::displace_from(&u, 0, current, Some(vec!["u_x".into(), "u_y".into()]))?;
    coords.write().select(current)?;
    let turned = external_forces(&load, &density)?;
    coords.write().select(0)?;
    for k in 0..4 {
        let (f0, f) = (at(&reference, k)?, at(&turned, k)?);
        let expected = [c * f0[0] - s * f0[1], s * f0[0] + c * f0[1]];
        assert!((f[0] - expected[0]).abs() < 1e-12 && (f[1] - expected[1]).abs() < 1e-12);
    }
    Ok(())
}
