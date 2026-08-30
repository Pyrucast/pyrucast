//! HEX27 — the tri-quadratic brick, assembled end to end.
//!
//! It is the **widest** element pyrucast carries: 27 nodes, so 81 displacement
//! degrees of freedom in a cell, and 81 values in the `dN/dx` a kernel builds at
//! each Gauss point. Every kernel sizes that scratch with
//! `models::kernel::MAX_CELL_DOFS`, and the bound is proved once per zone when
//! the reference data is snapshotted — a narrower bound used to let the widest
//! element overflow a stack slice deep inside the elasticity kernel, which
//! reported an index range rather than a mesh.
//!
//! Two claims, on the two paths a cell goes through:
//!
//! - the **geometry** path — the volume of the `[0,2]³` cube, integrated through
//!   `det_j_w` over 27 shape functions, is 8;
//! - the **stiffness** path — the assembled 81×81 block is singular in the way
//!   an elastic stiffness must be: a rigid translation costs nothing, so every
//!   row sums to zero over the columns of its own direction, and the whole
//!   matrix therefore sums to zero.
//!
//! Runs under `cargo test`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, Interpolation};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::model;
use pyrucast::Result;

/// The reference layout of a HEX27 in pyrucast's own node order: the eight
/// corners, the twelve edge midpoints, the six face centres, then the body
/// centre. Mapped to the physical cube `[0,2]³` by `x = ξ + 1`.
const REF: [(f64, f64, f64); 27] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (0.0, -1.0, -1.0),
    (1.0, 0.0, -1.0),
    (0.0, 1.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, -1.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (-1.0, 0.0, 1.0),
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0),
];

/// One HEX27 cell on `[0,2]³`, as a tri-quadratic FE space.
fn cube() -> Result<FiniteElementSpace> {
    let coords = Handle::new(Coords::new(3)?);
    let mut ids = Vec::with_capacity(REF.len());
    for &(x, y, z) in &REF {
        ids.push(Node::create_in(coords.clone(), &[x + 1.0, y + 1.0, z + 1.0])?.id());
    }
    let mut sm = SubMesh::new(coords, ElementType::HEX27);
    sm.add_cell(&ids)?;
    FiniteElementSpace::new(&Mesh::from_submesh(sm), Interpolation::Lagrange2)
}

/// The geometry path: 27 shape functions integrate the cube's own volume.
#[test]
fn hex27_integrates_its_volume() -> Result<()> {
    let fes = cube()?;
    let field = ElementField::new(&fes, vec!["c".into()])?;
    field.get(0)?.write().set_uniform("c", 1.0)?;
    let volume = pyrucast::ops::measure::integral_element(&field, "c")?;
    assert!((volume - 8.0).abs() < 1e-12, "volume {volume} ≠ 8");
    Ok(())
}

/// The stiffness path: 81 degrees of freedom assembled without overflowing a
/// kernel's scratch, and singular under rigid translation.
#[test]
fn hex27_assembles_a_singular_stiffness() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;

    let fes = cube()?;
    let m = model::elasticity(&fes, Kinematics::Full3D)?;
    let materials = pyrucast::ops::element_field::material_field(&m, &[("E", E), ("nu", NU)])?;
    let k = pyrucast::ops::matrix::stiffness(&m, &materials)?;

    // 27 nodes × 3 displacement components.
    assert_eq!(k.n_rows()?, 81);
    assert_eq!(k.n_cols()?, 81);

    // A rigid translation along any axis costs no energy, so each row sums to
    // zero over the columns of that axis — and the whole matrix over all three.
    let total: f64 = k.dense()?.iter().sum();
    let scale: f64 = k.dense()?.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        total.abs() < 1e-9 * scale,
        "rigid modes carry {total} (largest entry {scale})"
    );
    Ok(())
}
