//! Discrete Kirchhoff — DKT on a triangle, DKQ on a quadrangle.
//!
//! Kirchhoff-Love says the normal fibre stays normal: `γ = ∇w + β = 0`, so the
//! rotation is no longer a field of its own and the curvature is a set of
//! **second** derivatives of the deflection alone. Imposed everywhere, that is a
//! fourth-order equation, and a conforming element for it needs a C¹ basis —
//! which on a general triangle or quadrangle does not exist in any usable form.
//!
//! The discrete Kirchhoff answer is to keep the rotation as the interpolated
//! field, and impose `γ = 0` **only at chosen points**. Nothing is ever
//! differentiated twice, the basis stays Lagrange, and the thin limit is exact
//! by construction rather than approached from a shear that has to be kept from
//! locking. That is the difference from [Reissner-Mindlin](super::thick), and it
//! is why the two coexist: one is right for a thick shell, the other for a thin
//! one, and neither is a special case of the other.
//!
//! ## The construction, in one place
//!
//! The rotation `β = (β_x, β_y)` is interpolated **quadratically** — the
//! six-function basis of a `TRI6`, the eight of a `QUA8` — over an element whose
//! geometry stays linear. Its mid-side values are then eliminated, one side at a
//! time, by three statements:
//!
//! ```text
//! γ = 0 at each corner                 β_i  = −∇w_i
//! γ_s = 0 at each mid-side             β_sk = −(3/2l)(w_j − w_i) − ¼(β_si + β_sj)
//! β_n linear along each side           β_nk = ½(β_ni + β_nj)
//! ```
//!
//! the second reading the mid-slope of the **cubic** the deflection follows
//! along a side — which is where the exactness of an Euler-Bernoulli beam enters
//! a plate, without a Hermite basis ever being assembled. After elimination each
//! mid-side function carries a fixed combination of the corner degrees of
//! freedom, and the whole element is described by five numbers per side:
//!
//! ```text
//! a = −x_ij/l²      b = ¾ x_ij y_ij/l²      c = (¼x_ij² − ½y_ij²)/l²
//! d = −y_ij/l²                              e = (¼y_ij² − ½x_ij²)/l²
//! ```
//!
//! `a` and `d` carry the deflection into the rotation (they are odd in the side
//! direction, hence the sign flip between a side's two ends); `b`, `c` and `e`
//! resolve a corner rotation onto the side's tangent and normal, and are even.
//!
//! DKT and DKQ differ **only** by the number of corners, the quadratic basis and
//! the quadrature — so they are one routine here, not two. Batoz's published
//! `H_x`, `H_y` tables for the two elements come straight back out of it.
//!
//! ## What it is not
//!
//! There is no shear strain, so there is no `Q` to report from a constitutive
//! law: the transverse force of a thin plate is a **reaction**, recovered from
//! the gradient of the moments. The behaviour therefore stops at the six
//! membrane and bending resultants.

use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::error::{PyrucastError, Result};
use crate::models::shell::{
    accumulate, local_coords, local_frame, membrane_and_drilling, to_global,
};
use crate::models::CellGeom;

use super::thick::bending_law;

/// The five coefficients of one side — see the module documentation.
#[derive(Clone, Copy, Default)]
struct SideCoeffs {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
}

/// The coefficients of the side running from corner `i` to corner `j`, from the
/// in-plane node coordinates.
fn side_coeffs(p: &[[f64; 2]], i: usize, j: usize, cell: usize) -> Result<SideCoeffs> {
    let (dx, dy) = (p[i][0] - p[j][0], p[i][1] - p[j][1]);
    let l2 = dx * dx + dy * dy;
    if l2 <= f64::EPSILON {
        return Err(PyrucastError::Message(format!(
            "Shell (kirchhoff): cell {cell} has a side of zero length between local nodes {i} \
             and {j}"
        )));
    }
    Ok(SideCoeffs {
        a: -dx / l2,
        b: 0.75 * dx * dy / l2,
        c: (0.25 * dx * dx - 0.5 * dy * dy) / l2,
        d: -dy / l2,
        e: (0.25 * dy * dy - 0.5 * dx * dx) / l2,
    })
}

/// The elimination of one rotation, as a matrix on the quadratic shape
/// functions: `[degree of freedom][shape function]`.
type Elimination = Vec<Vec<f64>>;

/// The two rotations as linear forms in the quadratic shape functions:
/// `β_x = Σ_m N_m · (Σ_q C_x[q][m] u_q)`, and the same for `β_y`.
///
/// `u` is the bending degree-of-freedom vector `[w, θ_x, θ_y]` per corner, and
/// `m` runs over corner shape functions first, mid-side ones after — the node
/// order of `TRI6` and `QUA8` alike, which is why their tables can be read
/// directly.
///
/// Holding the elimination as a **matrix on the shape functions**, rather than
/// as the assembled `H_x(ξ, η)` of the literature, is what makes the derivative
/// free: `∂H/∂ξ = C · ∂N/∂ξ` uses the same `C`, so no second table has to be
/// written or kept consistent with the first.
fn constraint_matrices(p: &[[f64; 2]], cell: usize) -> Result<(Elimination, Elimination)> {
    let n = p.len();
    let n_shape = 2 * n;
    let n_dof = 3 * n;
    let sides = (0..n)
        .map(|k| side_coeffs(p, k, (k + 1) % n, cell))
        .collect::<Result<Vec<_>>>()?;

    let mut cx = vec![vec![0.0; n_shape]; n_dof];
    let mut cy = vec![vec![0.0; n_shape]; n_dof];
    for corner in 0..n {
        // The two sides meeting at this corner: the one it starts (where it is
        // the `i` end of `x_ij`) and the one it ends.
        let (next, prev) = (corner, (corner + n - 1) % n);
        let (m_next, m_prev) = (n + next, n + prev);
        let (sn, sp) = (sides[next], sides[prev]);
        let (w, tx, ty) = (3 * corner, 3 * corner + 1, 3 * corner + 2);

        // `a` and `d` are odd in the side direction: the corner is the far end
        // of `prev`, so that side enters with the opposite sign.
        cx[w][m_next] += 1.5 * sn.a;
        cx[w][m_prev] -= 1.5 * sp.a;
        cx[tx][m_next] += sn.b;
        cx[tx][m_prev] += sp.b;
        cx[ty][corner] += 1.0;
        cx[ty][m_next] -= sn.c;
        cx[ty][m_prev] -= sp.c;

        cy[w][m_next] += 1.5 * sn.d;
        cy[w][m_prev] -= 1.5 * sp.d;
        cy[tx][corner] -= 1.0;
        cy[tx][m_next] += sn.e;
        cy[tx][m_prev] += sp.e;
        cy[ty][m_next] -= sn.b;
        cy[ty][m_prev] -= sp.b;
    }
    Ok((cx, cy))
}

/// The element types of a facet with `n` corners: the linear one carrying the
/// geometry, and the quadratic one carrying the rotations.
fn element_pair(n: usize, cell: usize) -> Result<(ElementType, ElementType)> {
    match n {
        3 => Ok((ElementType::TRI3, ElementType::TRI6)),
        4 => Ok((ElementType::QUA4, ElementType::QUA8)),
        _ => Err(PyrucastError::Message(format!(
            "Shell (kirchhoff): cell {cell} has {n} nodes — a discrete Kirchhoff facet is a \
             triangle (DKT) or a quadrangle (DKQ)"
        ))),
    }
}

/// The curvature operator `B_b` (3 × 6n) at Gauss point `g`, on the **shell**
/// degrees of freedom `[u, v, w, θ_x, θ_y, θ_z]` per node.
///
/// `κ = [∂β_x/∂x, ∂β_y/∂y, ∂β_x/∂y + ∂β_y/∂x]`, the same convention as
/// [Reissner-Mindlin](super::thick) — only the way `β` depends on the degrees of
/// freedom differs.
fn bending_b(
    geom: &CellGeom,
    p: &[[f64; 2]],
    cx: &[Vec<f64>],
    cy: &[Vec<f64>],
    g: usize,
) -> Result<Vec<Vec<f64>>> {
    let n = p.len();
    let cell = geom.cell;
    let (linear, quadratic) = element_pair(n, cell)?;
    let xi = geom.gauss_xi(g)?;

    // The Jacobian of the **geometry**, in the element's own plane: a flat facet
    // is a plane map, so this is an ordinary 2×2 — no manifold pseudo-inverse.
    let dn_geom = linear.as_kind().dshape(xi);
    let mut j = [[0.0_f64; 2]; 2];
    for (i, node) in p.iter().enumerate() {
        for (a, &coord) in node.iter().enumerate() {
            for (k, jrow) in j[a].iter_mut().enumerate() {
                *jrow += coord * dn_geom[i * 2 + k];
            }
        }
    }
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if det.abs() <= f64::EPSILON {
        return Err(PyrucastError::Message(format!(
            "Shell (kirchhoff): cell {cell} is degenerate — its in-plane Jacobian vanishes"
        )));
    }
    // `jinv[k][a] = ∂ξ_k/∂x_a`.
    let jinv = [
        [j[1][1] / det, -j[0][1] / det],
        [-j[1][0] / det, j[0][0] / det],
    ];

    let dn_rot = quadratic.as_kind().dshape(xi);
    let n_shape = 2 * n;
    let side = 6 * n;
    let mut b = vec![vec![0.0; side]; 3];
    for corner in 0..n {
        for (slot, &shell_dof) in [6 * corner + 2, 6 * corner + 3, 6 * corner + 4]
            .iter()
            .enumerate()
        {
            let q = 3 * corner + slot;
            // `∂β/∂ξ_k` for this degree of freedom, then carried to `x` and `y`.
            let dref = |c: &[Vec<f64>], k: usize| -> f64 {
                (0..n_shape).map(|m| c[q][m] * dn_rot[m * 2 + k]).sum()
            };
            let (bx_xi, bx_eta) = (dref(cx, 0), dref(cx, 1));
            let (by_xi, by_eta) = (dref(cy, 0), dref(cy, 1));
            let bx_dx = bx_xi * jinv[0][0] + bx_eta * jinv[1][0];
            let bx_dy = bx_xi * jinv[0][1] + bx_eta * jinv[1][1];
            let by_dx = by_xi * jinv[0][0] + by_eta * jinv[1][0];
            let by_dy = by_xi * jinv[0][1] + by_eta * jinv[1][1];

            b[0][shell_dof] = bx_dx;
            b[1][shell_dof] = by_dy;
            b[2][shell_dof] = bx_dy + by_dx;
        }
    }
    Ok(b)
}

/// The local element stiffness of one discrete-Kirchhoff facet, carried to the
/// global axes.
///
/// One [`CellGeom`], not two: there is no shear term, so there is no second
/// quadrature to integrate it at.
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
/// # use pyrucast::models::kernel::{assemble_block, reduce_cells};
/// # use pyrucast::models::shell;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into(), "h".into()],
/// #     &[210_000.0, 0.3, 0.01]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(),
/// #                    "m_x".to_string(), "m_y".to_string(), "m_z".to_string()],
/// #               vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(),
/// #                    "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::shell::kirchhoff;
/// // **Une** `CellGeom`, non deux : il n'y a pas de terme de cisaillement,
/// // donc pas de seconde quadrature pour l'intégrer.
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, _s, ke| kirchhoff::element_stiffness(&geoms[0], m, ke),
/// )?;
/// // Le bloc porte les six DDL de chaque nœud : 18 × 18 sur un TRI3.
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (18, 18));
/// // Et il est symétrique, comme toute raideur.
/// let d = bloc.dense();
/// assert!((0..18).all(|i| (0..18).all(|j| (d[i * 18 + j] - d[j * 18 + i]).abs() < 1e-6)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let n = geom.n_nodes;
    let side = 6 * n;
    let cell = geom.cell;
    let (e, nu, h) = (
        material.value(cell, 0, "E")?,
        material.value(cell, 0, "nu")?,
        material.value(cell, 0, "h")?,
    );
    let db = bending_law(e, nu, h);

    let frame = local_frame(geom)?;
    let p = local_coords(geom, &frame)?;
    element_pair(n, cell)?;
    let (cx, cy) = constraint_matrices(&p, cell)?;

    let mut local = vec![vec![0.0_f64; side]; side];
    membrane_and_drilling(geom, &frame, e, nu, h, &mut local)?;
    for g in 0..geom.n_gauss {
        let bb = bending_b(geom, &p, &cx, &cy, g)?;
        accumulate(&mut local, &bb, &db, geom.det_j_w(g), side);
    }

    to_global(&local, &frame, n, ke);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `H_x`, `H_y` of a DKT — Batoz's tables — must come back out
    /// of the general construction, on a triangle chosen so that no coefficient
    /// is symmetric with another.
    ///
    /// This is the test that pins the algebra down. Everything else about a
    /// plate element (rigid modes, a patch test, a textbook deflection) is
    /// satisfied by families of wrong elements too; the `a…e` coefficients and
    /// their placement are not.
    #[test]
    fn the_triangle_reproduces_the_published_tables() {
        let p = [[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]];
        let (cx, cy) = constraint_matrices(&p, 0).unwrap();
        // Batoz numbers the mid-sides 4 ↔ 2-3, 5 ↔ 3-1, 6 ↔ 1-2; here they are
        // 3 ↔ 1-2, 4 ↔ 2-3, 5 ↔ 3-1, so the published index k maps as follows.
        let (k4, k5, k6) = (4_usize, 5, 3);
        let s = |i: usize, j: usize| side_coeffs(&p, i, j, 0).unwrap();
        let (s4, s5, s6) = (s(1, 2), s(2, 0), s(0, 1));

        // H_x(1) = 1.5(a₆N₆ − a₅N₅), H_x(2) = b₅N₅ + b₆N₆, H_x(3) = N₁ − c₅N₅ − c₆N₆
        assert!((cx[0][k6] - 1.5 * s6.a).abs() < 1e-14);
        assert!((cx[0][k5] + 1.5 * s5.a).abs() < 1e-14);
        assert!((cx[1][k5] - s5.b).abs() < 1e-14);
        assert!((cx[1][k6] - s6.b).abs() < 1e-14);
        assert!((cx[2][0] - 1.0).abs() < 1e-14);
        assert!((cx[2][k5] + s5.c).abs() < 1e-14);
        assert!((cx[2][k6] + s6.c).abs() < 1e-14);
        // H_x(4) = 1.5(a₄N₄ − a₆N₆): the second corner, the next pair of sides.
        assert!((cx[3][k4] - 1.5 * s4.a).abs() < 1e-14);
        assert!((cx[3][k6] + 1.5 * s6.a).abs() < 1e-14);
        // H_y(1) = 1.5(d₆N₆ − d₅N₅), H_y(2) = −N₁ + e₅N₅ + e₆N₆, H_y(3) = −H_x(2)
        assert!((cy[0][k6] - 1.5 * s6.d).abs() < 1e-14);
        assert!((cy[0][k5] + 1.5 * s5.d).abs() < 1e-14);
        assert!((cy[1][0] + 1.0).abs() < 1e-14);
        assert!((cy[1][k5] - s5.e).abs() < 1e-14);
        assert!((cy[1][k6] - s6.e).abs() < 1e-14);
        assert!((cy[2][k5] + s5.b).abs() < 1e-14);
        assert!((cy[2][k6] + s6.b).abs() < 1e-14);
    }

    /// The same, for the quadrangle: Batoz's DKQ numbers the sides 5 ↔ 1-2,
    /// 6 ↔ 2-3, 7 ↔ 3-4, 8 ↔ 4-1, which is this construction's own order.
    #[test]
    fn the_quadrangle_reproduces_the_published_tables() {
        let p = [[0.0, 0.0], [2.0, 0.3], [2.2, 1.7], [-0.1, 1.4]];
        let (cx, cy) = constraint_matrices(&p, 0).unwrap();
        let s = |i: usize, j: usize| side_coeffs(&p, i, j, 0).unwrap();
        let (s5, s8) = (s(0, 1), s(3, 0));
        // H_x(1) = 1.5(a₅N₅ − a₈N₈), H_x(3) = N₁ − c₅N₅ − c₈N₈
        assert!((cx[0][4] - 1.5 * s5.a).abs() < 1e-14);
        assert!((cx[0][7] + 1.5 * s8.a).abs() < 1e-14);
        assert!((cx[2][0] - 1.0).abs() < 1e-14);
        assert!((cx[2][4] + s5.c).abs() < 1e-14);
        assert!((cx[2][7] + s8.c).abs() < 1e-14);
        // H_y(2) = −N₁ + e₅N₅ + e₈N₈
        assert!((cy[1][0] + 1.0).abs() < 1e-14);
        assert!((cy[1][4] - s5.e).abs() < 1e-14);
        assert!((cy[1][7] - s8.e).abs() < 1e-14);
    }

    /// A rigid rotation of the plate — `w = x`, and the rotation that goes with
    /// it — must leave the interpolated `β` constant, hence the curvature zero,
    /// at **every** point and not merely at the nodes.
    ///
    /// Kirchhoff reads `β = −∇w`, so `w = x` carries `β_x = −1`, i.e. `θ_y = −1`
    /// in the shell's own convention. Getting that sign backwards is the single
    /// likeliest slip in the whole construction, and it is invisible in the
    /// coefficient tables above.
    #[test]
    fn a_rigid_rotation_bends_nothing() {
        for p in [
            vec![[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]],
            vec![[0.0, 0.0], [2.0, 0.3], [2.2, 1.7], [-0.1, 1.4]],
        ] {
            let n = p.len();
            let (cx, cy) = constraint_matrices(&p, 0).unwrap();
            // u = [w, θ_x, θ_y] per corner, for w = x: θ_y = −1, θ_x = 0.
            let mut u = vec![0.0; 3 * n];
            for (corner, node) in p.iter().enumerate() {
                u[3 * corner] = node[0];
                u[3 * corner + 2] = -1.0;
            }
            let quadratic = element_pair(n, 0).unwrap().1;
            for &xi in &[[0.2, 0.3], [0.5, 0.1], [0.0, 0.0], [0.25, 0.25]] {
                let point = if n == 3 {
                    [xi[0], xi[1]]
                } else {
                    [2.0 * xi[0] - 0.5, 2.0 * xi[1] - 0.5]
                };
                let sh = quadratic.as_kind().shape(&point);
                let beta = |c: &Vec<Vec<f64>>| -> f64 {
                    (0..3 * n)
                        .map(|q| u[q] * (0..2 * n).map(|m| c[q][m] * sh[m]).sum::<f64>())
                        .sum()
                };
                assert!(
                    (beta(&cx) + 1.0).abs() < 1e-12,
                    "β_x = {} at {point:?}, expected −1",
                    beta(&cx)
                );
                assert!(
                    beta(&cy).abs() < 1e-12,
                    "β_y = {} at {point:?}, expected 0",
                    beta(&cy)
                );
            }
        }
    }
}
