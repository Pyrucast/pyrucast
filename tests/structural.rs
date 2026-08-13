//! Mass and geometric-stiffness of the structural (bar / beam) elements,
//! exercised end-to-end through the public API against closed forms.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node};
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::coords::Coords;
use pyrucast::store::insert;
use pyrucast::Result;

/// A single SEG2 bar from the origin to `(dx, dy)`, with its FE space + two nodes.
fn bar_2d(dx: f64, dy: f64) -> Result<(FiniteElementSpace, Node, Node, f64)> {
    let coords = insert(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[dx, dy])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded)?;
    Ok((fes, a, b, (dx * dx + dy * dy).sqrt()))
}

/// A one-component state field (`n` = axial force) on a bar's FE space.
fn axial_force(fes: &FiniteElementSpace, n: f64) -> Result<ElementField> {
    let sub = SubElementField::from_uniform_per_component(fes.get(0)?, vec!["n".into()], &[n])?;
    let mut f = ElementField::empty();
    f.add_sub(insert(sub))?;
    Ok(f)
}

#[test]
fn truss_consistent_mass_matches_closed_form() -> Result<()> {
    const RHO: f64 = 2.0;
    const AREA: f64 = 3.0;
    let (fes, a, b, len) = bar_2d(3.0, 4.0)?; // L = 5
    let model = Model::truss(&fes)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", 1.0), ("A", AREA), ("rho", RHO)],
    )?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    let tol = 1e-12;
    let base = RHO * AREA * len / 6.0; // ρAL/6
                                       // (ρAL/6)·[[2,1],[1,2]] on each translation component; block-diagonal.
    assert!((m.get(a.id(), "f_x", a.id(), "u_x")? - 2.0 * base).abs() < tol);
    assert!((m.get(a.id(), "f_x", b.id(), "u_x")? - base).abs() < tol);
    assert!((m.get(a.id(), "f_y", a.id(), "u_y")? - 2.0 * base).abs() < tol);
    assert!(m.get(a.id(), "f_x", a.id(), "u_y")?.abs() < tol);

    // Whole-matrix sum = space_dim · ρAL (each direction carries the full bar mass).
    let total: f64 = m.to_dmatrix()?.sum();
    assert!(
        (total - 2.0 * RHO * AREA * len).abs() < 1e-10,
        "total = {total}"
    );
    Ok(())
}

#[test]
fn truss_geometric_stiffens_transverse_only() -> Result<()> {
    const N: f64 = 7.0;
    let (fes, a, b, _l) = bar_2d(1.0, 0.0)?; // axis = x, L = 1
    let model = Model::truss(&fes)?;
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", 1.0), ("A", 1.0)])?;
    let state = axial_force(&fes, N)?;

    let kg = pyrucast::ops::matrix::geometric(&model, &materials, &state)?;
    let tol = 1e-12;
    // Axis is x ⇒ transverse is y: (N/L)[[1,-1],[-1,1]] on u_y, nothing on u_x.
    assert!((kg.get(a.id(), "f_y", a.id(), "u_y")? - N).abs() < tol);
    assert!((kg.get(a.id(), "f_y", b.id(), "u_y")? + N).abs() < tol);
    assert!(kg.get(a.id(), "f_x", a.id(), "u_x")?.abs() < tol);
    assert!(kg.get(a.id(), "f_x", a.id(), "u_y")?.abs() < tol);
    Ok(())
}

#[test]
fn truss_geometric_inclined_transverse_projector() -> Result<()> {
    const N: f64 = 5.0;
    let (fes, a, _b, len) = bar_2d(3.0, 4.0)?; // c = (0.6, 0.8)
    let model = Model::truss(&fes)?;
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", 1.0), ("A", 1.0)])?;
    let state = axial_force(&fes, N)?;

    let kg = pyrucast::ops::matrix::geometric(&model, &materials, &state)?;
    let (cx, cy) = (0.6, 0.8);
    let k = N / len;
    let tol = 1e-12;
    // Diagonal block at A = (N/L)·(I − c⊗c).
    assert!((kg.get(a.id(), "f_x", a.id(), "u_x")? - k * (1.0 - cx * cx)).abs() < tol);
    assert!((kg.get(a.id(), "f_x", a.id(), "u_y")? - k * (-cx * cy)).abs() < tol);
    assert!((kg.get(a.id(), "f_y", a.id(), "u_y")? - k * (1.0 - cy * cy)).abs() < tol);
    Ok(())
}

#[test]
fn truss_mass_requires_density() -> Result<()> {
    let (fes, _a, _b, _l) = bar_2d(1.0, 0.0)?;
    let model = Model::truss(&fes)?;
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", 1.0), ("A", 1.0)])?;
    assert!(pyrucast::ops::matrix::mass(&model, &materials).is_err());
    Ok(())
}

// ─── Frame (2-D beam-column) ────────────────────────────────────────────────

#[test]
fn frame_consistent_mass_matches_closed_form() -> Result<()> {
    const RHO: f64 = 2.0;
    const AREA: f64 = 3.0;
    const I: f64 = 0.5;
    const L: f64 = 2.0;
    let (fes, a, b, len) = bar_2d(L, 0.0)?; // horizontal ⇒ T = identity
    let model = Model::timoshenko(&fes)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("E", 1.0),
            ("A", AREA),
            ("I", I),
            ("G", 1.0),
            ("A_s", 1.0),
            ("rho", RHO),
        ],
    )?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    let tol = 1e-12;
    let ma = RHO * AREA * len / 6.0; // ρAL/6
                                     // The **axial** direction still carries the linear field's mass, which is
                                     // exact for what it interpolates — a bar stretches linearly.
    assert!((m.get(a.id(), "f_x", a.id(), "u_x")? - 2.0 * ma).abs() < tol);
    assert!((m.get(a.id(), "f_x", b.id(), "u_x")? - ma).abs() < tol);
    // The **transverse** direction takes the exact element's mass: a rigid
    // translation still carries exactly ρAL, and w now couples to the rotation.
    let total = m.get(a.id(), "f_y", a.id(), "u_y")?
        + m.get(a.id(), "f_y", b.id(), "u_y")?
        + m.get(b.id(), "f_y", a.id(), "u_y")?
        + m.get(b.id(), "f_y", b.id(), "u_y")?;
    assert!((total - RHO * AREA * len).abs() < 1e-10 * RHO * AREA * len);
    assert!(m.get(a.id(), "f_y", a.id(), "r_z")?.abs() > 1e-6);
    // No translation ↔ rotation coupling.
    assert!(m.get(a.id(), "f_x", a.id(), "r_z")?.abs() < tol);

    // The x direction carries the member's mass too — the sum over the whole
    // matrix is *not* an invariant of the exact element (it mixes translations
    // with rotations, which have different units), so the check is made
    // direction by direction, where it means something.
    let total_x = m.get(a.id(), "f_x", a.id(), "u_x")?
        + m.get(a.id(), "f_x", b.id(), "u_x")?
        + m.get(b.id(), "f_x", a.id(), "u_x")?
        + m.get(b.id(), "f_x", b.id(), "u_x")?;
    assert!((total_x - RHO * AREA * len).abs() < 1e-10 * RHO * AREA * len);
    Ok(())
}

#[test]
fn frame_geometric_stiffens_transverse_only() -> Result<()> {
    const N: f64 = 9.0;
    const L: f64 = 3.0;
    let (fes, a, b, len) = bar_2d(L, 0.0)?; // horizontal ⇒ transverse is u_y
    let model = Model::timoshenko(&fes)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", 1.0), ("A", 1.0), ("I", 1.0), ("G", 1.0), ("A_s", 1.0)],
    )?;
    let sub = SubElementField::from_uniform_per_component(fes.get(0)?, vec!["N".into()], &[N])?;
    let mut state = ElementField::empty();
    state.add_sub(insert(sub))?;

    let kg = pyrucast::ops::matrix::geometric(&model, &materials, &state)?;
    let tol = 1e-12;
    let g = N / len;
    assert!((kg.get(a.id(), "f_y", a.id(), "u_y")? - g).abs() < tol);
    assert!((kg.get(a.id(), "f_y", b.id(), "u_y")? + g).abs() < tol);
    // Axial and rotation carry no geometric term.
    assert!(kg.get(a.id(), "f_x", a.id(), "u_x")?.abs() < tol);
    assert!(kg.get(a.id(), "m_z", a.id(), "r_z")?.abs() < tol);
    Ok(())
}

// ─── Timoshenko (1-D beam, mass only) ───────────────────────────────────────

#[test]
fn timoshenko_consistent_mass_matches_closed_form() -> Result<()> {
    const RHO: f64 = 2.0;
    const AREA: f64 = 3.0;
    const I: f64 = 0.5;
    const L: f64 = 2.0;
    let coords = insert(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[L])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded)?;
    let model = Model::timoshenko(&fes)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("E", 1.0),
            ("I", I),
            ("G", 1.0),
            ("A_s", 1.0),
            ("A", AREA),
            ("rho", RHO),
        ],
    )?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    // The mass is now the **exact element's**, integrated from the same shape
    // functions as its stiffness. Two properties pin it, and neither is a
    // transcribed coefficient.
    //
    // 1. A rigid translation carries exactly the mass of the member. Nothing an
    //    interpolation does may change that, whatever `Φ`.
    let total = m.get(a.id(), "f_w", a.id(), "w")?
        + m.get(a.id(), "f_w", b.id(), "w")?
        + m.get(b.id(), "f_w", a.id(), "w")?
        + m.get(b.id(), "f_w", b.id(), "w")?;
    assert!(
        (total - RHO * AREA * L).abs() < 1e-10 * RHO * AREA * L,
        "rigid translation carries {total}, expected {}",
        RHO * AREA * L
    );
    // 2. Deflection and rotation are now **coupled** — the linear element's
    //    mass had no such term, and that absence is exactly what made it
    //    inconsistent with an exact stiffness.
    assert!(
        m.get(a.id(), "f_w", a.id(), "theta")?.abs() > 1e-6,
        "the exact mass must couple w and theta"
    );
    Ok(())
}

// ─── Frame3d (space frame) ──────────────────────────────────────────────────

/// A single SEG2 along global x in 3-D, with its FE space + two nodes.
fn bar_3d(len: f64) -> Result<(FiniteElementSpace, Node, Node)> {
    let coords = insert(Coords::new(3)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[len, 0.0, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    Ok((
        FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded)?,
        a,
        b,
    ))
}

fn frame3d_materials(model: &Model, iy: f64, iz: f64, area: f64, rho: f64) -> Result<ElementField> {
    pyrucast::ops::element_field::material_field(
        model,
        &[
            ("E", 1.0),
            ("A", area),
            ("I_y", iy),
            ("I_z", iz),
            ("J", 1.0),
            ("G", 1.0),
            ("A_sy", 1.0),
            ("A_sz", 1.0),
            ("rho", rho),
        ],
    )
}

#[test]
fn frame3d_consistent_mass_matches_closed_form() -> Result<()> {
    const RHO: f64 = 2.0;
    const AREA: f64 = 3.0;
    const IY: f64 = 0.4;
    const IZ: f64 = 0.7;
    const L: f64 = 2.0;
    let (fes, a, b) = bar_3d(L)?; // along x ⇒ local axes = global ⇒ T = identity
    let model = Model::timoshenko(&fes)?;
    let materials = frame3d_materials(&model, IY, IZ, AREA, RHO)?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    let tol = 1e-12;
    let diag = |sec: f64| 2.0 * RHO * sec * L / 6.0;
    // Axial and torsion keep the linear field's mass — exact for what those two
    // degrees of freedom actually interpolate.
    assert!((m.get(a.id(), "f_x", a.id(), "u_x")? - diag(AREA)).abs() < tol); // u'
    assert!((m.get(a.id(), "m_x", a.id(), "r_x")? - diag(IY + IZ)).abs() < tol); // torsion I_p
                                                                                 // The two bending planes take the exact element's mass. Each still carries
                                                                                 // exactly the member's mass under a rigid translation…
    for (dual, primal) in [("f_y", "u_y"), ("f_z", "u_z")] {
        let total = m.get(a.id(), dual, a.id(), primal)?
            + m.get(a.id(), dual, b.id(), primal)?
            + m.get(b.id(), dual, a.id(), primal)?
            + m.get(b.id(), dual, b.id(), primal)?;
        assert!(
            (total - RHO * AREA * L).abs() < 1e-10 * RHO * AREA * L,
            "{dual}: {total} vs {}",
            RHO * AREA * L
        );
    }
    // …and now couples the deflection to the rotation of its own plane, which
    // the linear mass never did.
    assert!(m.get(a.id(), "f_y", a.id(), "r_z")?.abs() > 1e-6);
    assert!(m.get(a.id(), "f_z", a.id(), "r_y")?.abs() > 1e-6);
    Ok(())
}

#[test]
fn frame3d_geometric_stiffens_both_transverse() -> Result<()> {
    const N: f64 = 8.0;
    const L: f64 = 4.0;
    let (fes, a, b) = bar_3d(L)?;
    let model = Model::timoshenko(&fes)?;
    let materials = frame3d_materials(&model, 1.0, 1.0, 1.0, 1.0)?;
    let sub = SubElementField::from_uniform_per_component(fes.get(0)?, vec!["N".into()], &[N])?;
    let mut state = ElementField::empty();
    state.add_sub(insert(sub))?;

    let kg = pyrucast::ops::matrix::geometric(&model, &materials, &state)?;
    let tol = 1e-12;
    let g = N / L;
    // Both transverse translations (u_y = v', u_z = w') are stiffened.
    assert!((kg.get(a.id(), "f_y", a.id(), "u_y")? - g).abs() < tol);
    assert!((kg.get(a.id(), "f_y", b.id(), "u_y")? + g).abs() < tol);
    assert!((kg.get(a.id(), "f_z", a.id(), "u_z")? - g).abs() < tol);
    // Axial and torsion carry no geometric term.
    assert!(kg.get(a.id(), "f_x", a.id(), "u_x")?.abs() < tol);
    assert!(kg.get(a.id(), "m_x", a.id(), "r_x")?.abs() < tol);
    Ok(())
}
