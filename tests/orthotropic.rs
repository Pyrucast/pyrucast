//! Orthotropic and anisotropic linear elasticity, exercised end-to-end.
//!
//! Uniaxial tension of the unit square `[0,1]²` (QUA4 grid), **plane stress**,
//! with the material axes along the global ones: rollers `u_x = 0` on the left
//! edge and `u_y = 0` on the bottom edge, uniform traction `S` on the right.
//! The stress state is `σ_xx = S`, `σ_yy = σ_zz = 0`, so the compliance gives
//! the exact answer directly,
//!
//! ```text
//! ε_xx = S / E_1        ⇒  u_x =  (S/E_1)·x
//! ε_yy = −ν_12 · S/E_1  ⇒  u_y = −(ν_12 S/E_1)·y
//! ```
//!
//! which Q1 reproduces nodally. The isotropic case is the special one where
//! `E_1 = E_2` and `ν_12 = ν`, so the same square also pins down the two
//! degeneracies that matter: **an orthotropic material fed isotropic constants
//! must behave isotropically whatever its frame**, and **an anisotropic material
//! fed the isotropic stiffness tensor must do the same**. Both go through the
//! full rotation machinery, so they are the sharpest end-to-end check of it.
//!
//! Single source for the « élasticité orthotrope » example of the mechanics book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::node_field::FluxDensity;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

/// Traction on the right edge.
const S: f64 = 2.0;
/// `N×N` QUA4 grid on the unit square.
const N: usize = 2;

#[test]
fn orthotropic_square_stretches_along_its_first_material_axis() -> Result<()> {
    const E1: f64 = 200.0; // stiff direction — aligned with global x
    const E2: f64 = 50.0; // compliant transverse direction
    const NU12: f64 = 0.25;

    let (grid, fes, coords) = unit_square()?;
    let model = clamped_model(&grid, &fes, MaterialSymmetry::Orthotropic)?;

    // The material axes travel through the material field like any other
    // coefficient: `V1` is the first orthotropy direction, here the global x.
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("E_1", E1),
            ("E_2", E2),
            ("E_3", E2),
            ("nu_12", NU12),
            ("nu_13", NU12),
            ("nu_23", 0.25),
            ("G_12", 30.0),
            ("G_13", 30.0),
            ("G_23", 30.0),
            ("V1X", 1.0),
            ("V1Y", 0.0),
        ],
    )?;

    let solution = solve_traction(&model, &materials, &grid, &coords)?;

    // σ_xx = S with σ_yy = σ_zz = 0 ⇒ the compliance row gives ε directly.
    let tol = 1e-10;
    let h = 1.0 / N as f64;
    for j in 0..=N {
        for i in 0..=N {
            let (x, y) = (i as f64 * h, j as f64 * h);
            let id = grid[j * (N + 1) + i].id();
            let ux = solution.value(id, "u_x")?;
            let uy = solution.value(id, "u_y")?;
            assert!((ux - S / E1 * x).abs() < tol, "u_x({x},{y}) = {ux}");
            assert!((uy + NU12 * S / E1 * y).abs() < tol, "u_y({x},{y}) = {uy}");
        }
    }
    Ok(())
}
// ANCHOR_END: example

/// Orthotropy with **equal** constants is isotropy — and must stay so whatever
/// the material frame. Rotating `V1` by 30° exercises the whole fourth-order
/// rotation through the assembly; anything mis-indexed in it would break the
/// invariance and show up here as a wrong displacement.
#[test]
fn orthotropy_with_isotropic_constants_ignores_its_frame() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;
    let g = E / (2.0 * (1.0 + NU));

    for angle_deg in [0.0_f64, 30.0, 90.0, 137.0] {
        let a = angle_deg.to_radians();
        let (grid, fes, coords) = unit_square()?;
        let model = clamped_model(&grid, &fes, MaterialSymmetry::Orthotropic)?;
        let materials = pyrucast::ops::element_field::material_field(
            &model,
            &[
                ("E_1", E),
                ("E_2", E),
                ("E_3", E),
                ("nu_12", NU),
                ("nu_13", NU),
                ("nu_23", NU),
                ("G_12", g),
                ("G_13", g),
                ("G_23", g),
                ("V1X", a.cos()),
                ("V1Y", a.sin()),
            ],
        )?;
        let solution = solve_traction(&model, &materials, &grid, &coords)?;

        let tol = 1e-9;
        let h = 1.0 / N as f64;
        for j in 0..=N {
            for i in 0..=N {
                let (x, y) = (i as f64 * h, j as f64 * h);
                let id = grid[j * (N + 1) + i].id();
                let ux = solution.value(id, "u_x")?;
                let uy = solution.value(id, "u_y")?;
                assert!(
                    (ux - S / E * x).abs() < tol,
                    "angle {angle_deg}°, u_x({x},{y}) = {ux}"
                );
                assert!(
                    (uy + NU * S / E * y).abs() < tol,
                    "angle {angle_deg}°, u_y({x},{y}) = {uy}"
                );
            }
        }
    }
    Ok(())
}

/// The anisotropic law fed the **isotropic** stiffness tensor must reproduce the
/// isotropic answer. This pins down the order in which the 21 upper-triangle
/// constants `C_11 … C_66` are read: any permutation of them would land the
/// shear moduli in the wrong Voigt slots and break the result.
#[test]
fn anisotropy_fed_the_isotropic_tensor_is_isotropic() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;
    // Isotropic 3-D stiffness (Voigt order [xx, yy, zz, yz, xz, xy]).
    let c = E / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let (d_n, d_off, g) = (c * (1.0 - NU), c * NU, c * (1.0 - 2.0 * NU) / 2.0);

    let (grid, fes, coords) = unit_square()?;
    let model = clamped_model(&grid, &fes, MaterialSymmetry::Anisotropic)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("C_11", d_n),
            ("C_12", d_off),
            ("C_13", d_off),
            ("C_14", 0.0),
            ("C_15", 0.0),
            ("C_16", 0.0),
            ("C_22", d_n),
            ("C_23", d_off),
            ("C_24", 0.0),
            ("C_25", 0.0),
            ("C_26", 0.0),
            ("C_33", d_n),
            ("C_34", 0.0),
            ("C_35", 0.0),
            ("C_36", 0.0),
            ("C_44", g),
            ("C_45", 0.0),
            ("C_46", 0.0),
            ("C_55", g),
            ("C_56", 0.0),
            ("C_66", g),
            ("V1X", 1.0),
            ("V1Y", 0.0),
        ],
    )?;
    let solution = solve_traction(&model, &materials, &grid, &coords)?;

    let tol = 1e-9;
    let h = 1.0 / N as f64;
    for j in 0..=N {
        for i in 0..=N {
            let (x, y) = (i as f64 * h, j as f64 * h);
            let id = grid[j * (N + 1) + i].id();
            assert!((solution.value(id, "u_x")? - S / E * x).abs() < tol);
            assert!((solution.value(id, "u_y")? + NU * S / E * y).abs() < tol);
        }
    }
    Ok(())
}

/// A null first material axis is a degenerate frame: it must be reported, not
/// silently turned into an arbitrary orientation.
#[test]
fn a_degenerate_material_frame_is_rejected() -> Result<()> {
    let (grid, fes, coords) = unit_square()?;
    let _ = coords;
    let model = clamped_model(&grid, &fes, MaterialSymmetry::Orthotropic)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("E_1", 200.0),
            ("E_2", 50.0),
            ("E_3", 50.0),
            ("nu_12", 0.25),
            ("nu_13", 0.25),
            ("nu_23", 0.25),
            ("G_12", 30.0),
            ("G_13", 30.0),
            ("G_23", 30.0),
            ("V1X", 0.0),
            ("V1Y", 0.0),
        ],
    )?;
    let err = pyrucast::ops::matrix::stiffness(&model, &materials).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("material frame"), "unexpected message: {msg}");
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The `N×N` QUA4 grid on `[0,1]²`, its FE space and its coordinate set.
fn unit_square() -> Result<(
    Vec<Node>,
    FiniteElementSpace,
    pyrucast::handle::Handle<Coords>,
)> {
    let h = 1.0 / N as f64;
    let coords = Handle::new(Coords::new(2)?);
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * h, j as f64 * h],
            )?);
        }
    }
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..N {
        for i in 0..N {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((grid, fes, coords))
}

/// Plane-stress elasticity of the given symmetry, with the two roller edges.
fn clamped_model(
    grid: &[Node],
    fes: &FiniteElementSpace,
    symmetry: MaterialSymmetry,
) -> Result<Model> {
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let roller = |nodes: &[Node], var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
        let multiplier = mesh::barycenter(&imposed)?;
        model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
    };
    let left: Vec<Node> = (0..=N).map(|j| grid[idx(0, j)].clone()).collect();
    let bottom: Vec<Node> = (0..=N).map(|i| grid[idx(i, 0)].clone()).collect();
    let mut model = model::elasticity_with_symmetry(fes, Kinematics::PlaneStress, symmetry)?;
    model = model.union(&roller(&left, "u_x", "f_x")?)?;
    model = model.union(&roller(&bottom, "u_y", "f_y")?)?;
    Ok(model)
}

/// Assemble, apply the uniform traction `S` on the right edge, and solve.
fn solve_traction(
    model: &Model,
    materials: &pyrucast::containers::element_field::ElementField,
    grid: &[Node],
    coords: &pyrucast::handle::Handle<Coords>,
) -> Result<NodeField> {
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let mut right_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        right_edge.add_cell(&[grid[idx(N, j)].id(), grid[idx(N, j + 1)].id()])?;
    }
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    let traction =
        pyrucast::ops::node_field::flux(&right_fes.get(0)?, FluxDensity::Uniform(S), "f_x")?;
    let rhs = NodeField::from_sub(traction);
    let stiffness = pyrucast::ops::matrix::stiffness(model, materials)?;
    solve(&stiffness, &rhs)
}
