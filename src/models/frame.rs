//! Planar frame (portique) physics — an oriented Timoshenko beam carrying
//! **axial + bending + shear**.
//!
//! A `SEG2` element in a 2-D configuration with **three** DOFs per node: the
//! in-plane displacement `u_x, u_y` and the rotation `rz`. The local stiffness
//! combines a truss axial term `E·A/L`, a bending term `E·I` and a
//! **reduced**-shear term `G·A_s` (`B_s` evaluated at the element centre, like
//! the [`Timoshenko`](crate::models::timoshenko) beam), then is rotated to the
//! global frame by `K = Tᵀ K_loc T` where `T` is built from the element's
//! direction cosines — so any orientation in the plane works.
//!
//! Primal `u_x, u_y, rz`; dual `f_x, f_y, m_z`. Material components `E`, `A`,
//! `I`, `G`, `A_s`. Besides the stiffness it assembles the consistent **mass**
//! (translational `ρA` + rotary `ρI`, optional `rho`) and the **geometric
//! stiffness** under the axial force `N` (linear-element forms, rotated `Tᵀ·T`).
//!
//! ## What is left here
//!
//! Only the 2-D configuration's kernels, driven by
//! [`Timoshenko`](crate::models::timoshenko::Timoshenko). The physics itself
//! lives there: a plane frame is a beam in a 2-D configuration, not a model of
//! its own — everything that used to distinguish it (three DOFs per node, an
//! axial term, a rotation to the global axes) follows from the dimension of the
//! mesh. What remains is what is genuinely two-dimensional: the local closed
//! forms and the rotation that carries them to the global axes.

use crate::containers::element_field::SubElementField;
use crate::error::{PyrucastError, Result};
use crate::models::CellGeom;

/// Local 6×6 frame stiffness (DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`) from
/// `E·A`, `E·I`, `G·A_s` and length `L` — axial + bending + reduced shear.
fn local_stiffness(ea: f64, ei: f64, gas: f64, l: f64) -> [[f64; 6]; 6] {
    let mut k = [[0.0_f64; 6]; 6];
    // Axial (E·A/L) couples u'_A (0) and u'_B (3).
    let ka = ea / l;
    k[0][0] += ka;
    k[3][3] += ka;
    k[0][3] -= ka;
    k[3][0] -= ka;
    // Bending **and** shear together, from the exact Timoshenko block: it
    // solves the two coupled equations in closed form rather than
    // approximating them with a linear element and a reduced quadrature.
    let b = crate::models::beam::bending_4x4(ei, Some(gas), l);
    let idx = [1usize, 2, 4, 5]; // [w'_A, θ_A, w'_B, θ_B]
    for (a, &ia) in idx.iter().enumerate() {
        for (c, &ic) in idx.iter().enumerate() {
            k[ia][ic] += b[a][c];
        }
    }
    k
}

/// Rotation `T` (6×6) mapping global DOFs to local: per node
/// `[[c, s, 0], [−s, c, 0], [0, 0, 1]]`.
fn rotation(c: f64, s: f64) -> [[f64; 6]; 6] {
    let mut t = [[0.0_f64; 6]; 6];
    for node in 0..2 {
        let o = node * 3;
        t[o][o] = c;
        t[o][o + 1] = s;
        t[o + 1][o] = -s;
        t[o + 1][o + 1] = c;
        t[o + 2][o + 2] = 1.0;
    }
    t
}

/// `A · B` for 6×6 matrices.
fn matmul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = 0.0;
            for k in 0..6 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Element kernel: local frame stiffness `K = Tᵀ K_loc T` of one 2-node frame
/// element, written into `ke` (flat row-major 6×6, **node-major / variable-minor**
/// dof order `dof = node·3 + var`). Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(),
/// #     vec!["E".into(), "A".into(), "I".into(), "A_s".into(), "G".into()],
/// #     &[210000.0, 0.01, 1e-05, 0.008, 80000.0]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, s, ke| frame::element_stiffness(&geoms[0], m, ke),
/// )?;
/// // Portique plan : axial et flexion, ramenés aux axes globaux.
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (6, 6));
/// // La somme brute des entrées ne vaut pas zéro : les DDL mêlent
/// // translations et rotations, et seul le mode de **translation** est
/// // rigide. On vérifie plutôt la symétrie, propre à toute raideur.
/// let d = bloc.dense();
/// assert!((0..6).all(|i| (0..6).all(|j| (d[i * 6 + j] - d[j * 6 + i]).abs() < 1e-6)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let cell = geom.cell;
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    let (dx, dy) = (xb[0] - xa[0], xb[1] - xa[1]);
    let l = (dx * dx + dy * dy).sqrt();
    let (c, s) = (dx / l, dy / l);
    let ea = material.value(cell, 0, "E")? * material.value(cell, 0, "A")?;
    let ei = material.value(cell, 0, "E")? * material.value(cell, 0, "I")?;
    let gas = material.value(cell, 0, "G")? * material.value(cell, 0, "A_s")?;

    let kl = local_stiffness(ea, ei, gas, l);
    let t = rotation(c, s);
    // K_global = Tᵀ · K_loc · T.
    let kg = matmul(&transpose(&t), &matmul(&kl, &t));

    for r in 0..6 {
        for col in 0..6 {
            ke[r * 6 + col] = kg[r][col];
        }
    }
    Ok(())
}

/// Transpose of a 6×6 matrix.
fn transpose(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// Local 6×6 **consistent mass** (linear element): `(ρAL/6)[[2,1],[1,2]]` on each
/// translation (`u'`, `w'`) and `(ρIL/6)[[2,1],[1,2]]` on the rotation `θ`
/// (rotary inertia). No translation–rotation coupling for the linear kinematics.
fn local_mass(rho_a: f64, rho_i: f64, ei: f64, gas: Option<f64>, l: f64) -> [[f64; 6]; 6] {
    let mut m = [[0.0_f64; 6]; 6];
    // Axial: a bar's consistent mass, `(ρAL/6)[[2,1],[1,2]]`, exact for the
    // linear axial field the element really carries.
    let ma = rho_a * l / 6.0;
    m[0][0] += 2.0 * ma;
    m[3][3] += 2.0 * ma;
    m[0][3] += ma;
    m[3][0] += ma;
    // Bending: the exact element's own consistent mass, on [w'_A, θ_A, w'_B, θ_B].
    let b = crate::models::beam::mass_4x4(rho_a, rho_i, ei, gas, l);
    let idx = [1usize, 2, 4, 5];
    for (a, &ia) in idx.iter().enumerate() {
        for (c, &ic) in idx.iter().enumerate() {
            m[ia][ic] += b[a][c];
        }
    }
    m
}

/// Local 6×6 **geometric stiffness** of the beam-column under axial force `N`:
/// `(N/L)[[1,−1],[−1,1]]` on the transverse translations `w'` (indices 1, 4);
/// the axial and rotation DOFs carry no geometric term for the linear element.
fn local_geometric(n: f64, l: f64) -> [[f64; 6]; 6] {
    let mut k = [[0.0_f64; 6]; 6];
    let g = n / l;
    k[1][1] += g;
    k[4][4] += g;
    k[1][4] -= g;
    k[4][1] -= g;
    k
}

/// Length + direction cosines of one frame `SEG2` cell.
fn cell_frame(geom: &CellGeom) -> Result<(f64, f64, f64)> {
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    let (dx, dy) = (xb[0] - xa[0], xb[1] - xa[1]);
    let l = (dx * dx + dy * dy).sqrt();
    Ok((l, dx / l, dy / l))
}

/// Element kernel: local frame **consistent mass** `M = Tᵀ M_loc T` (Cast3M
/// `MASS`), from density `rho`, area `A` and second moment `I`. Same `ke` layout
/// as [`element_stiffness`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["rho".into(), "A".into(), "I".into(), "E".into(), "A_s".into()], &[3.0, 0.01, 1e-05, 210000.0, 0.008]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, s, ke| frame::element_mass(&geoms[0], m, ke),
/// )?;
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (6, 6));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_mass(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let cell = geom.cell;
    let (l, c, s) = cell_frame(geom)?;
    let rho = material.value(cell, 0, "rho").map_err(|_| {
        PyrucastError::Message(
            "Frame mass matrix: material component `rho` (density) is required".into(),
        )
    })?;
    let rho_a = rho * material.value(cell, 0, "A")?;
    let rho_i = rho * material.value(cell, 0, "I")?;
    let ei = material.value(cell, 0, "E")? * material.value(cell, 0, "I")?;
    // Absent shear constants mean `Φ = 0` — a Bernoulli beam's material
    // deliberately carries neither, and that absence *is* the statement that
    // there is no shear compliance. One mass kernel then serves both theories.
    let gas = match (material.value(cell, 0, "G"), material.value(cell, 0, "A_s")) {
        (Ok(g), Ok(a_s)) => Some(g * a_s),
        _ => None,
    };
    let ml = local_mass(rho_a, rho_i, ei, gas, l);
    let t = rotation(c, s);
    write_6x6(ke, &matmul(&transpose(&t), &matmul(&ml, &t)));
    Ok(())
}

/// Element kernel: local frame **geometric stiffness** `K_g = Tᵀ K_loc T`
/// (Cast3M `KSIG`), from the axial force `N` (state component `N`). Same `ke`
/// layout as [`element_stiffness`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["N".into()], &[100.0]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, Some(&mat),
///     |geoms, m, s, ke| frame::element_geometric(&geoms[0], s.unwrap(), ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// // C'est le signe de cette matrice qui décide de la charge critique.
/// assert!(total.abs() < 1e-6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_geometric(geom: &CellGeom, state: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let cell = geom.cell;
    let (l, c, s) = cell_frame(geom)?;
    let n = state.value(cell, 0, "N")?;
    let kl = local_geometric(n, l);
    let t = rotation(c, s);
    write_6x6(ke, &matmul(&transpose(&t), &matmul(&kl, &t)));
    Ok(())
}

/// Scatter a 6×6 matrix into the flat row-major `ke` (accumulating).
fn write_6x6(ke: &mut [f64], m: &[[f64; 6]; 6]) {
    for (r, row) in m.iter().enumerate() {
        for (col, v) in row.iter().enumerate() {
            ke[r * 6 + col] += v;
        }
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────
