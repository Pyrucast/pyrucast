//! Consistent (algorithmic) tangent `K_t = ∫ Bᵀ D_alg B` (Cast3M `KTAN`) for
//! perfect-J2 plasticity, **validated against the finite-difference derivative of
//! the internal forces**: `K_t[i,j] = ∂f_int_i/∂u_j`. This is the ground-truth
//! oracle for `D_alg` — if the analytical algorithmic modulus is wrong, the
//! central-difference check fails.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::plasticity::law::PlasticLaw;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::behavior::integrate;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::element_field::{deformation, StrainMeasure};
use pyrucast::ops::matrix::tangent;
use pyrucast::ops::node_field::internal_forces;
use pyrucast::Result;

const AXES: [&str; 3] = ["x", "y", "z"];

/// Build a displacement `NodeField` from a flat per-node displacement table.
fn build_u(nodes: &[Node], dim: usize, disp: &[f64]) -> Result<NodeField> {
    let support = Handle::new(SubMesh::poi1_from_nodes(nodes)?);
    let comps: Vec<String> = (0..dim).map(|a| format!("u_{}", AXES[a])).collect();
    let mut u = SubNodeField::from_poi1(&support, comps)?;
    for (i, n) in nodes.iter().enumerate() {
        for a in 0..dim {
            u.set_value(n.id(), &format!("u_{}", AXES[a]), disp[i * dim + a])?;
        }
    }
    Ok(NodeField::from_sub(u))
}

/// Internal forces of the plastic step at displacement `disp`, as a flat vector
/// in `nodes × dim` order (dual components `f_*`).
fn internal_force_vec(
    model: &Model,
    fes: &FiniteElementSpace,
    materials: &pyrucast::containers::element_field::ElementField,
    nodes: &[Node],
    dim: usize,
    disp: &[f64],
) -> Result<Vec<f64>> {
    let u = build_u(nodes, dim, disp)?;
    let strain = deformation(&u, fes, StrainMeasure::Linear)?;
    let state = integrate(model, &strain, None, materials, None)?;
    let f = internal_forces(model, &state, &u, materials)?;
    let mut out = vec![0.0; nodes.len() * dim];
    for (i, n) in nodes.iter().enumerate() {
        for a in 0..dim {
            out[i * dim + a] = f.value(n.id(), &format!("f_{}", AXES[a]))?;
        }
    }
    Ok(out)
}

/// Assert the analytical tangent matches the central-difference derivative of the
/// internal forces at a plastic state.
fn check_tangent_fd(
    model: &Model,
    fes: &FiniteElementSpace,
    nodes: &[Node],
    dim: usize,
    base: &[f64],
) -> Result<()> {
    let materials = material_field(model, &[("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)])?;

    // The analytical tangent, evaluated at the converged state of `base`.
    let strain = deformation(&build_u(nodes, dim, base)?, fes, StrainMeasure::Linear)?;
    let kt = tangent(model, &materials, &strain, None, None)?;

    let ndof = nodes.len() * dim;
    let h = 1e-8;
    for j in 0..ndof {
        let mut dp = base.to_vec();
        let mut dm = base.to_vec();
        dp[j] += h;
        dm[j] -= h;
        let fp = internal_force_vec(model, fes, &materials, nodes, dim, &dp)?;
        let fm = internal_force_vec(model, fes, &materials, nodes, dim, &dm)?;
        let (jn, ja) = (j / dim, j % dim);
        for i in 0..ndof {
            let (in_, ia) = (i / dim, i % dim);
            let fd = (fp[i] - fm[i]) / (2.0 * h);
            let analytic = kt.get(
                nodes[in_].id(),
                &format!("f_{}", AXES[ia]),
                nodes[jn].id(),
                &format!("u_{}", AXES[ja]),
            )?;
            assert!(
                (fd - analytic).abs() < 1e-4 * (analytic.abs() + 1.0),
                "K_t[{i},{j}] analytic {analytic} vs FD {fd}"
            );
        }
    }
    Ok(())
}

fn unit_quad() -> Result<(FiniteElementSpace, [Node; 4])> {
    let coords = Handle::new(Coords::new(2)?);
    let n0 = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let n1 = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let n2 = Node::create_in(coords.clone(), &[1.0, 1.0])?;
    let n3 = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    mesh.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()])?;
    Ok((FiniteElementSpace::lagrange1(&mesh)?, [n0, n1, n2, n3]))
}

/// A multiaxial, well-past-yield displacement over the four corners
/// (coords (0,0),(1,0),(1,1),(0,1)), flattened as `[ux0,uy0, ux1,uy1, …]`.
fn plastic_disp_2d() -> [f64; 8] {
    let f = |x: f64, y: f64| [0.02 * x - 0.005 * y, 0.004 * x + 0.01 * y];
    let c = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut d = [0.0; 8];
    for (i, &(x, y)) in c.iter().enumerate() {
        let [ux, uy] = f(x, y);
        d[2 * i] = ux;
        d[2 * i + 1] = uy;
    }
    d
}

#[test]
fn tangent_matches_fd_plane_strain() -> Result<()> {
    let (fes, n) = unit_quad()?;
    let model = plasticity_model(&fes, Kinematics::PlaneStrain)?;
    check_tangent_fd(&model, &fes, &n, 2, &plastic_disp_2d())
}

#[test]
fn tangent_matches_fd_plane_stress() -> Result<()> {
    let (fes, n) = unit_quad()?;
    let model = plasticity_model(&fes, Kinematics::PlaneStress)?;
    check_tangent_fd(&model, &fes, &n, 2, &plastic_disp_2d())
}

/// Axisymmetric: the finite-difference oracle covers the whole revolved chain in
/// one shot — the hoop row `N_i/r` of `B`, its transpose in the internal forces,
/// the `[rr, zz, θθ, rz]` restriction of the 3-D algorithmic tangent, and the
/// `2πr` measure (which multiplies both sides, so it must be consistent).
#[test]
fn tangent_matches_fd_axisymmetric() -> Result<()> {
    let coords = Handle::new(Coords::axisymmetric()?);
    // Away from the axis, so the element is well away from r = 0.
    let corners = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)];
    let nodes: Vec<Node> = corners
        .iter()
        .map(|&(r, z)| Node::create_in(coords.clone(), &[r, z]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let model = plasticity_model(&fes, Kinematics::Axisymmetric)?;

    // Past-yield radial + axial displacement (the hoop strain u_r/r is large
    // here, so the θθ component genuinely drives the return map).
    let mut base = vec![0.0; nodes.len() * 2];
    for (i, &(r, z)) in corners.iter().enumerate() {
        base[2 * i] = 0.02 * r - 0.004 * z;
        base[2 * i + 1] = 0.003 * r + 0.015 * z;
    }
    check_tangent_fd(&model, &fes, &nodes, 2, &base)
}

#[test]
fn tangent_matches_fd_solid_hex() -> Result<()> {
    let coords = Handle::new(Coords::new(3)?);
    let p = |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]);
    let corners = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ];
    let nodes: Vec<Node> = corners
        .iter()
        .map(|&(x, y, z)| p(x, y, z))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
    mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let model = plasticity_model(&fes, Kinematics::Full3D)?;

    // Multiaxial past-yield displacement.
    let mut base = vec![0.0; nodes.len() * 3];
    for (i, &(x, y, z)) in corners.iter().enumerate() {
        base[3 * i] = 0.02 * x - 0.004 * y + 0.001 * z;
        base[3 * i + 1] = 0.003 * x + 0.015 * y - 0.002 * z;
        base[3 * i + 2] = -0.001 * x + 0.002 * y + 0.01 * z;
    }
    check_tangent_fd(&model, &fes, &nodes, 3, &base)
}

fn plasticity_model(fes: &FiniteElementSpace, model: Kinematics) -> Result<Model> {
    let mut m = Model::empty();
    m.add_sub(Handle::new(SubModel::plasticity(
        fes.get(0)?,
        model,
        PlasticLaw::Perfect,
        StrainMeasure::Linear,
    )?))?;
    Ok(m)
}

/// In the **elastic** regime (strain below yield) the consistent tangent equals
/// the elastic stiffness bit-for-bit-close.
#[test]
fn elastic_regime_tangent_equals_stiffness() -> Result<()> {
    use pyrucast::ops::matrix::stiffness;
    let (fes, n) = unit_quad()?;
    let model = plasticity_model(&fes, Kinematics::PlaneStrain)?;
    let materials = material_field(&model, &[("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)])?;

    // Tiny displacement ⇒ everywhere elastic.
    let base: Vec<f64> = plastic_disp_2d().iter().map(|d| d * 1e-4).collect();
    let strain = deformation(&build_u(&n, 2, &base)?, &fes, StrainMeasure::Linear)?;

    let kt = tangent(&model, &materials, &strain, None, None)?;
    let k = stiffness(&model, &materials)?;
    let tol = 1e-9;
    for i in 0..4 {
        for a in ["f_x", "f_y"] {
            for j in 0..4 {
                for b in ["u_x", "u_y"] {
                    let t = kt.get(n[i].id(), a, n[j].id(), b)?;
                    let s = k.get(n[i].id(), a, n[j].id(), b)?;
                    assert!(
                        (t - s).abs() < tol * (s.abs() + 1.0),
                        "{a}{i} {b}{j}: {t} ≠ {s}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// A linear physics answers a tangent request with its stiffness — exactly, and
/// without reading the deformation, which it has no law to differentiate. The
/// conductivity of the unit QUA4 is anchored on its closed form
/// `k/6·[4, −1, −2, −1]` so that a zero block on both sides cannot pass.
#[test]
fn a_linear_physics_tangent_is_its_stiffness() -> Result<()> {
    use pyrucast::containers::element_field::ElementField;
    use pyrucast::ops::matrix::stiffness;
    use pyrucast::ops::model;
    let (fes, n) = unit_quad()?;
    let conduction = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
    let k = 3.0;
    let materials = material_field(&conduction, &[("k", k)])?;

    let kt = tangent(&conduction, &materials, &ElementField::empty(), None, None)?;
    for (j, coef) in [4.0, -1.0, -2.0, -1.0].iter().enumerate() {
        let t = kt.get(n[0].id(), "q", n[j].id(), "T").unwrap();
        assert!((t - k * coef / 6.0).abs() < 1e-12, "q0 T{j}: {t}");
    }
    assert_eq!(kt.dense()?, stiffness(&conduction, &materials)?.dense()?);
    Ok(())
}

/// A constraint's relations are linear: they enter the tangent as they enter
/// the stiffness, multiplier rows included — without them a Newton iteration
/// on the tangent would lose its boundary conditions.
#[test]
fn constraints_enter_the_tangent() -> Result<()> {
    use pyrucast::containers::element_field::ElementField;
    use pyrucast::models::RelationSense;
    use pyrucast::ops::matrix::stiffness;
    use pyrucast::ops::{mesh, model};
    let (fes, n) = unit_quad()?;
    let conduction = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
    let imposed = mesh::poi1_from_nodes(&n[..1])?;
    let constrained = conduction.union(&model::dirichlet(
        &imposed,
        &conduction,
        "T",
        RelationSense::Equality,
    )?)?;
    let materials = material_field(&constrained, &[("k", 1.0)])?;

    let kt = tangent(&constrained, &materials, &ElementField::empty(), None, None)?;
    let k = stiffness(&constrained, &materials)?;
    // Four temperatures plus the multiplier: the constraint row is there.
    assert_eq!(kt.n_rows()?, 5);
    assert_eq!(kt.dense()?, k.dense()?);
    Ok(())
}

/// A non-linear physics with no tangent kernel of its own must say so: a zero
/// block would let a Newton iteration fail far from the cause.
#[test]
fn a_nonlinear_physics_without_tangent_errors() -> Result<()> {
    use pyrucast::models::damage::law::DamageLaw;
    use pyrucast::ops::model;
    let (fes, n) = unit_quad()?;
    let damage = model::damage(
        &fes,
        Kinematics::PlaneStress,
        DamageLaw::DamageTc,
        StrainMeasure::Linear,
    )?;
    let materials = material_field(
        &damage,
        &[
            ("E", 30_000.0),
            ("nu", 0.2),
            ("f_t", 3.0),
            ("f_c", 30.0),
            ("A_t", 0.9),
            ("A_c", 0.5),
        ],
    )?;
    let strain = deformation(&build_u(&n, 2, &[0.0; 8])?, &fes, StrainMeasure::Linear)?;

    let err = tangent(&damage, &materials, &strain, None, None).unwrap_err();
    assert!(
        err.to_string().contains("no consistent tangent"),
        "unexpected message: {err}"
    );
    Ok(())
}
