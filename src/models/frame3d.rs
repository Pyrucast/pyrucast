//! 3-D Timoshenko frame (space frame) physics.
//!
//! A `SEG2` element in a 3-D configuration with **six** DOFs per node: the
//! displacement `u_x, u_y, u_z` and the rotation `r_x, r_y, r_z`. The local
//! 12×12 stiffness combines:
//!
//! - **axial** `E·A`,
//! - **torsion** `G·J` (about the beam axis),
//! - **bending** about the two principal axes (`E·I_z` with shear `G·A_sy`,
//!   `E·I_y` with shear `G·A_sz`), using the closed-form **Timoshenko** element
//!   (shear parameters `Φ`), which is nodally exact for end loads.
//!
//! It is rotated to the global frame by `K = Tᵀ K_loc T`. The section axes
//! (local `y'`, `z'`) are taken **automatically** from a global-Z reference
//! (global Y for a vertical beam), so no orientation data is needed — fine for
//! symmetric sections (`I_y = I_z`).
//!
//! Primal `u_x, u_y, u_z, r_x, r_y, r_z`; dual `f_x, f_y, f_z, m_x, m_y, m_z`.
//! Material components `E, A, I_y, I_z, J, G, A_sy, A_sz`. Besides the stiffness
//! it assembles the consistent **mass** (translational `ρA` + rotary
//! `ρI_p`/`ρI_y`/`ρI_z`, optional `rho`) and the **geometric stiffness** under
//! the axial force `N` — linear-element forms, rotated `Tᵀ·T`.
//!
//! ## What is left here
//!
//! Only the 3-D configuration's kernels, driven by
//! [`Timoshenko`](crate::models::timoshenko::Timoshenko). The physics itself
//! lives there: a space frame is a beam in a 3-D configuration, not a model of
//! its own. What remains is what is genuinely three-dimensional — the local axes
//! taken from a global-Z reference, the two bending planes with their sign
//! convention, the torsion, and the 12×12 rotation.

use crate::error::Result;
use crate::models::CellGeom;

// ─── Geometry helpers ────────────────────────────────────────────────────────

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Local axes `R = [x'; y'; z']` (rows, in global coords) from the beam
/// direction `d`. `x'` is along the beam; `y'`/`z'` come from a global-Z
/// reference (global Y if the beam is ~vertical) — automatic orientation.
fn local_axes(d: [f64; 3]) -> [[f64; 3]; 3] {
    let x = normalize(d);
    let z_ref = if x[2].abs() > 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let y = normalize(cross(z_ref, x)); // ⟂ x' and ref
    let z = cross(x, y); // completes the right-handed triad
    [x, y, z]
}

/// Local 12×12 Timoshenko frame stiffness, DOF order per node
/// `[u', v', w', θx', θy', θz']`.
#[allow(clippy::too_many_arguments)]
fn local_stiffness(
    ea: f64,
    gj: f64,
    e: f64,
    iy: f64,
    iz: f64,
    g: f64,
    asy: f64,
    asz: f64,
    l: f64,
) -> [[f64; 12]; 12] {
    let mut k = [[0.0_f64; 12]; 12];
    let a = ea / l; // axial
    let t = gj / l; // torsion
                    // Shear parameters for the two bending planes.
    let phi_y = 12.0 * e * iz / (g * asy * l * l); // x'-y' plane (I_z, A_sy)
    let phi_z = 12.0 * e * iy / (g * asz * l * l); // x'-z' plane (I_y, A_sz)

    let kv1 = 12.0 * e * iz / (l * l * l * (1.0 + phi_y));
    let kv2 = 6.0 * e * iz / (l * l * (1.0 + phi_y));
    let kv3 = (4.0 + phi_y) * e * iz / (l * (1.0 + phi_y));
    let kv4 = (2.0 - phi_y) * e * iz / (l * (1.0 + phi_y));

    let kw1 = 12.0 * e * iy / (l * l * l * (1.0 + phi_z));
    let kw2 = 6.0 * e * iy / (l * l * (1.0 + phi_z));
    let kw3 = (4.0 + phi_z) * e * iy / (l * (1.0 + phi_z));
    let kw4 = (2.0 - phi_z) * e * iy / (l * (1.0 + phi_z));

    // Upper triangle (then mirrored). DOFs: u'=0/6, v'=1/7, w'=2/8,
    // θx'=3/9, θy'=4/10, θz'=5/11.
    k[0][0] = a;
    k[0][6] = -a;
    k[6][6] = a;
    k[3][3] = t;
    k[3][9] = -t;
    k[9][9] = t;
    // x'-y' plane: v' (1, 7), θz' (5, 11).
    k[1][1] = kv1;
    k[1][5] = kv2;
    k[1][7] = -kv1;
    k[1][11] = kv2;
    k[5][5] = kv3;
    k[5][7] = -kv2;
    k[5][11] = kv4;
    k[7][7] = kv1;
    k[7][11] = -kv2;
    k[11][11] = kv3;
    // x'-z' plane: w' (2, 8), θy' (4, 10) — sign-flipped coupling.
    k[2][2] = kw1;
    k[2][4] = -kw2;
    k[2][8] = -kw1;
    k[2][10] = -kw2;
    k[4][4] = kw3;
    k[4][8] = kw2;
    k[4][10] = kw4;
    k[8][8] = kw1;
    k[8][10] = kw2;
    k[10][10] = kw3;

    for i in 0..12 {
        for j in (i + 1)..12 {
            k[j][i] = k[i][j];
        }
    }
    k
}

/// Rotation `T` (12×12): block-diagonal repetition of `R` over the four DOF
/// triples `[u_A], [θ_A], [u_B], [θ_B]`.
fn rotation(r: &[[f64; 3]; 3]) -> [[f64; 12]; 12] {
    let mut t = [[0.0_f64; 12]; 12];
    for blk in 0..4 {
        let o = blk * 3;
        for i in 0..3 {
            for j in 0..3 {
                t[o + i][o + j] = r[i][j];
            }
        }
    }
    t
}

fn matmul(a: &[[f64; 12]; 12], b: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0_f64; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            let mut acc = 0.0;
            for k in 0..12 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

fn transpose(a: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0_f64; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// Assemble `K = Tᵀ K_loc T` of every 3-D frame element into `k`, at
/// `(NodeId_i, dual_a) × (NodeId_j, primal_b)`.
/// Element kernel: local 3-D frame stiffness `K = Tᵀ K_loc T` of one 2-node
/// element, written into `ke` (flat row-major 12×12, **node-major /
/// variable-minor** dof order `dof = node·6 + var`). Pure and sequential —
/// driven in parallel by [`crate::models::kernel::assemble_block`].
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
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "G".into(), "A".into(), "I_y".into(), "I_z".into(), "A_sy".into(), "A_sz".into(), "J".into()],
/// #     &[210000.0, 80000.0, 0.01, 1e-05, 1e-05, 0.008, 0.008, 2e-05]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(), "m_x".to_string(), "m_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(), "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame3d;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     // Le noyau prend les constantes de section, pas le champ : c'est la
///     // physique qui lit son contrat, lui ne fait que les maths.
///     |geoms, m, s, ke| frame3d::element_stiffness(
///         &geoms[0], 210000.0 * 0.01, 80000.0 * 2e-05, 210000.0,
///         1e-05, 1e-05, 80000.0, 0.008, 0.008, ke),
/// )?;
/// // Portique spatial : axial, torsion et flexion autour de deux axes
/// // principaux — six DDL par nœud.
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (12, 12));
/// // La somme brute des entrées ne vaut pas zéro : les DDL mêlent
/// // translations et rotations, et seul le mode de **translation** est
/// // rigide. On vérifie plutôt la symétrie, propre à toute raideur.
/// let d = bloc.dense();
/// assert!((0..12).all(|i| (0..12).all(|j| (d[i * 12 + j] - d[j * 12 + i]).abs() < 1e-6)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn element_stiffness(
    geom: &CellGeom,
    ea: f64,
    gj: f64,
    e: f64,
    iy: f64,
    iz: f64,
    g: f64,
    a_sy: f64,
    a_sz: f64,
    ke: &mut [f64],
) -> Result<()> {
    let xa = geom.node_coord(0);
    let xb = geom.node_coord(1);
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    let kl = local_stiffness(ea, gj, e, iy, iz, g, a_sy, a_sz, l);
    let t = rotation(&local_axes(d));
    let kg = matmul(&transpose(&t), &matmul(&kl, &t)); // Tᵀ K_loc T

    for r in 0..12 {
        for c in 0..12 {
            ke[r * 12 + c] = kg[r][c];
        }
    }
    Ok(())
}

/// Local 12×12 **consistent mass** (linear element): `(ρAL/6)[[2,1],[1,2]]` on
/// each translation `u',v',w'`, and rotary inertia `(ρI L/6)[[2,1],[1,2]]` on
/// each rotation — polar `I_p = I_y + I_z` for the torsion `θx'`, `I_y` for
/// `θy'`, `I_z` for `θz'`. No translation–rotation coupling for the linear
/// kinematics.
/// `G·A_s`, or `None` when either constant is absent.
fn gas(g: Option<f64>, a_s: Option<f64>) -> Option<f64> {
    match (g, a_s) {
        (Some(g), Some(a)) => Some(g * a),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn local_mass(
    rho: f64,
    area: f64,
    iy: f64,
    iz: f64,
    e: f64,
    g: Option<f64>,
    a_sy: Option<f64>,
    a_sz: Option<f64>,
    l: f64,
) -> [[f64; 12]; 12] {
    let mut m = [[0.0_f64; 12]; 12];
    // Axial (0,6) and torsion (3,9): the linear field's consistent mass, exact
    // for what those two degrees of freedom actually interpolate.
    for (var, sec) in [(0usize, area), (3, iy + iz)] {
        let ms = rho * sec * l / 6.0;
        let (i, j) = (var, var + 6);
        m[i][i] += 2.0 * ms;
        m[j][j] += 2.0 * ms;
        m[i][j] += ms;
        m[j][i] += ms;
    }
    // The two bending planes take the exact element's mass, with the same index
    // maps and sign convention as the stiffness above.
    let b_xy = crate::models::beam::mass_4x4(rho * area, rho * iz, e * iz, gas(g, a_sy), l);
    for (a, &ia) in [1usize, 5, 7, 11].iter().enumerate() {
        for (c, &ic) in [1usize, 5, 7, 11].iter().enumerate() {
            m[ia][ic] += b_xy[a][c];
        }
    }
    let b_xz = crate::models::beam::mass_4x4(rho * area, rho * iy, e * iy, gas(g, a_sz), l);
    let sign = [1.0, -1.0, 1.0, -1.0];
    for (a, &ia) in [2usize, 4, 8, 10].iter().enumerate() {
        for (c, &ic) in [2usize, 4, 8, 10].iter().enumerate() {
            m[ia][ic] += sign[a] * sign[c] * b_xz[a][c];
        }
    }
    m
}

/// Local 12×12 **geometric stiffness** under axial force `N`:
/// `(N/L)[[1,−1],[−1,1]]` on each transverse translation `v'` (1, 7) and `w'`
/// (2, 8); the axial, torsion and bending-rotation DOFs carry no term for the
/// linear element.
fn local_geometric(n: f64, l: f64) -> [[f64; 12]; 12] {
    let mut k = [[0.0_f64; 12]; 12];
    let g = n / l;
    for &(i, j) in &[(1usize, 7usize), (2, 8)] {
        k[i][i] += g;
        k[j][j] += g;
        k[i][j] -= g;
        k[j][i] -= g;
    }
    k
}

/// Element kernel: local 3-D frame **consistent mass** `M = Tᵀ M_loc T` (Cast3M
/// `MASS`), from density `rho`, area `A` and second moments `I_y`, `I_z`. Same
/// `ke` layout as [`element_stiffness`].
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
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["rho".into(), "A".into(), "I_y".into(), "I_z".into(), "E".into(), "G".into(), "A_sy".into(), "A_sz".into()], &[3.0, 0.01, 1e-05, 1e-05, 210000.0, 80000.0, 0.008, 0.008]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(), "m_x".to_string(), "m_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(), "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame3d;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, s, ke| frame3d::element_mass(
///         &geoms[0], 3.0, 0.01, 1e-05, 1e-05, 210000.0,
///         Some(80000.0), Some(0.008), Some(0.008), ke),
/// )?;
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (12, 12));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn element_mass(
    geom: &CellGeom,
    rho: f64,
    area: f64,
    iy: f64,
    iz: f64,
    e: f64,
    g: Option<f64>,
    a_sy: Option<f64>,
    a_sz: Option<f64>,
    ke: &mut [f64],
) -> Result<()> {
    let xa = geom.node_coord(0);
    let xb = geom.node_coord(1);
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    let ml = local_mass(rho, area, iy, iz, e, g, a_sy, a_sz, l);
    let t = rotation(&local_axes(d));
    write_12x12(ke, &matmul(&transpose(&t), &matmul(&ml, &t)));
    Ok(())
}

/// Element kernel: local 3-D frame **geometric stiffness** `K_g = Tᵀ K_loc T`
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
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["N".into()], &[100.0]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(), "m_x".to_string(), "m_y".to_string(), "m_z".to_string()], vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(), "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::frame3d;
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, Some(&mat),
///     |geoms, m, s, ke| frame3d::element_geometric(&geoms[0], 100.0, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!(total.abs() < 1e-6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_geometric(geom: &CellGeom, n: f64, ke: &mut [f64]) -> Result<()> {
    let xa = geom.node_coord(0);
    let xb = geom.node_coord(1);
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    let kl = local_geometric(n, l);
    let t = rotation(&local_axes(d));
    write_12x12(ke, &matmul(&transpose(&t), &matmul(&kl, &t)));
    Ok(())
}

/// Scatter a 12×12 matrix into the flat row-major `ke` (accumulating).
fn write_12x12(ke: &mut [f64], m: &[[f64; 12]; 12]) {
    for (r, row) in m.iter().enumerate() {
        for (col, v) in row.iter().enumerate() {
            ke[r * 12 + col] += v;
        }
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────
