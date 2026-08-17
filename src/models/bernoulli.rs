//! Euler-Bernoulli beams — 1-D, plane frame, space frame.
//!
//! The classical beam theory: plane sections stay plane **and normal** to the
//! deflected axis, so the section rotation is the slope, `θ = w'`, and there is
//! no transverse shear at all. That is the difference from
//! [Timoshenko](crate::models::timoshenko) and from the
//! [frames](crate::models::frame), which keep a shear compliance.
//!
//! ## Why a separate physics rather than a shear area of infinity
//!
//! One could reach Bernoulli by letting `G·A_s → ∞` in a Timoshenko element, and
//! the answer would be right in exact arithmetic. In floating point it is not:
//! the shear term then dominates the stiffness by many orders of magnitude and
//! the bending response drowns in it — the classic **shear locking**, arrived at
//! from the other side. Writing the theory directly avoids the whole question,
//! and it removes two material constants (`G`, `A_s`) that a Bernoulli model has
//! no business asking for.
//!
//! ## The element
//!
//! Hermite cubic interpolation of the deflection makes the element **nodally
//! exact** for any load that leaves the interior free of distributed forces —
//! which is why one element per member is enough for a frame:
//!
//! ```text
//!            ⎡  12   6L  −12   6L ⎤
//! K_b = EI/L³⎢  6L  4L²  −6L  2L² ⎥        DOFs [w_A, θ_A, w_B, θ_B]
//!            ⎢ −12  −6L   12  −6L ⎥
//!            ⎣  6L  2L²  −6L  4L² ⎦
//! ```
//!
//! Three configurations share it, differing only in what surrounds the bending:
//!
//! | model | DOFs per node | adds |
//! |---|---|---|
//! | `planar_1d` | `w`, `theta` | nothing — pure bending |
//! | `frame_2d` | `u_x, u_y, r_z` | the axial `EA/L`, and a rotation to the global axes |
//! | `frame_3d` | `u_x…u_z, r_x…r_z` | the axial, the torsion `GJ/L`, and bending about **two** principal axes |
//!
//! The 3-D local frame is taken automatically from a global-Z reference (global
//! Y for a vertical member), exactly as
//! [`frame3d`](crate::models::frame3d) does — fine for symmetric sections.

use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::owned_components;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::Handle;

pub use crate::models::beam::BeamModel;

/// The material a configuration needs. No `G`, no `A_s` where there is no
/// shear — asking for a constant a theory does not use is a way of inviting the
/// wrong one.
fn material_of(model: BeamModel) -> &'static [&'static str] {
    match model {
        BeamModel::Planar1d => &["E", "I"],
        BeamModel::Frame2d => &["E", "A", "I"],
        BeamModel::Frame3d => &["E", "A", "I_y", "I_z", "J", "G"],
    }
}

/// The section forces the behaviour reports — no shear force, there being no
/// shear strain to produce one.
fn behavior_of(model: BeamModel) -> &'static [&'static str] {
    match model {
        BeamModel::Planar1d => &["M"],
        BeamModel::Frame2d => &["N", "M"],
        BeamModel::Frame3d => &["N", "M_y", "M_z", "T"],
    }
}

/// Euler-Bernoulli beam physics on a `SEG2` FE subspace.
#[derive(Clone)]
pub struct Bernoulli {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) model: BeamModel,
}

impl Bernoulli {
    /// Euler-Bernoulli beam on a `SEG2` FE subspace. Errors unless the subspace
    /// is `SEG2` in a configuration matching `model`.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, element, axisymmetric, interpolation) = {
            let s = fespace.read();
            (
                s.submesh(),
                s.space_dim(),
                s.submesh().read().element_type(),
                s.is_axisymmetric(),
                s.interpolation(),
            )
        };
        if element != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "Bernoulli: a beam needs SEG2 elements, got {element:?}"
            )));
        }
        if !interpolation.is_hermite() {
            return Err(PyrucastError::Message(format!(
                "Bernoulli: the beam interpolates its deflection with cubic Hermite functions, so \
                 it needs a HERMITE3 subspace — got {interpolation}. Build it with \
                 `FiniteElementSpace::new(&mesh, Interpolation::Hermite3)`. A Lagrange subspace \
                 would carry a linear deflection, whose curvature is identically zero."
            )));
        }
        let model = BeamModel::from_space_dim(space_dim)
            .map_err(|e| PyrucastError::Message(format!("Bernoulli: {e}")))?;
        if axisymmetric {
            return Err(PyrucastError::Message(
                "Bernoulli: a segment in a meridian plane is a shell of revolution, not a beam"
                    .into(),
            ));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            model,
        })
    }

    /// The cell's length and its unit direction, from the node coordinates.
    fn axis(&self, geom: &CellGeom) -> Result<(f64, Vec<f64>)> {
        let d = geom.space_dim;
        let (a, b) = (geom.node_coord(0)?.to_vec(), geom.node_coord(1)?.to_vec());
        let delta: Vec<f64> = (0..d).map(|i| b[i] - a[i]).collect();
        let l = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if l <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "Bernoulli: cell {} has zero length",
                geom.cell
            )));
        }
        Ok((l, delta.iter().map(|v| v / l).collect()))
    }
}

impl SubModelKind for Bernoulli {
    fn primal_vars(&self) -> Vec<String> {
        self.model.primal().iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        self.model.dual().iter().map(|s| s.to_string()).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// Mass and geometric stiffness share the stiffness layout.
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    /// The geometric stiffness needs an axial force to be stiffened by, which a
    /// pure-bending configuration does not have.
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        match self.model {
            BeamModel::Planar1d => None,
            _ => self.stiffness_layout(),
        }
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Bernoulli declares a material_fespace");
        match self.model {
            BeamModel::Planar1d => planar_mass(geom, mat, ke),
            BeamModel::Frame2d => crate::models::frame::element_mass(geom, mat, ke),
            BeamModel::Frame3d => crate::models::frame3d::element_mass(geom, mat, ke),
        }
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("the geometric stiffness requires the axial force `N`");
        match self.model {
            BeamModel::Planar1d => Err(PyrucastError::Message(
                "Bernoulli: a pure-bending beam carries no axial force, so it has no geometric \
                 stiffness — use a 2-D or 3-D configuration"
                    .into(),
            )),
            BeamModel::Frame2d => crate::models::frame::element_geometric(geom, stress, ke),
            BeamModel::Frame3d => crate::models::frame3d::element_geometric(geom, stress, ke),
        }
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Bernoulli declares a material_fespace");
        let (l, dir) = self.axis(geom)?;
        let cell = geom.cell;
        let e = mat.value(cell, 0, "E")?;

        match self.model {
            BeamModel::Planar1d => {
                let local = bending_4x4(geom, e * mat.value(cell, 0, "I")?)?;
                copy(&local, ke, 4);
            }
            BeamModel::Frame2d => {
                let local = plane_frame(
                    geom,
                    e * mat.value(cell, 0, "A")?,
                    e * mat.value(cell, 0, "I")?,
                    l,
                )?;
                let t = rotation_2d(dir[0], dir[1]);
                congruent(&local, &t, ke, 6);
            }
            BeamModel::Frame3d => {
                let local = space_frame(
                    geom,
                    e * mat.value(cell, 0, "A")?,
                    e * mat.value(cell, 0, "I_y")?,
                    e * mat.value(cell, 0, "I_z")?,
                    mat.value(cell, 0, "G")? * mat.value(cell, 0, "J")?,
                    l,
                )?;
                let t = rotation_3d(&dir);
                congruent(&local, &t, ke, 12);
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Bernoulli"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Bernoulli({})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Bernoulli {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(material_of(self.model)))
    }

    /// `rho` for the mass, and — in the 1-D configuration — the full area `A`,
    /// which only the mass needs (a pure-bending beam's stiffness uses none).
    fn optional_material_components(&self) -> &'static [&'static str] {
        match self.model {
            BeamModel::Planar1d => &["A", "rho"],
            _ => &["rho"],
        }
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(behavior_of(self.model)
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// The section forces from the generalised strains — a **linear** law, as it
    /// is for every structural element: `N = EA·ε`, `M = EI·κ`, `T = GJ·φ'`.
    fn integrate_point(
        &self,
        geom: &CellGeom,
        input: &SubElementField,
        _prev: Option<&SubElementField>,
        material: Option<&SubElementField>,
        g: usize,
        _dt: Option<f64>,
        out: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Bernoulli declares a material_fespace");
        let cell = geom.cell;
        let e = mat.value(cell, 0, "E")?;
        let read_in = |name: &str| input.value(cell, g, name);
        match self.model {
            BeamModel::Planar1d => {
                out[0] = e * mat.value(cell, 0, "I")? * read_in("kappa")?;
            }
            BeamModel::Frame2d => {
                out[0] = e * mat.value(cell, 0, "A")? * read_in("eps")?;
                out[1] = e * mat.value(cell, 0, "I")? * read_in("kappa")?;
            }
            BeamModel::Frame3d => {
                out[0] = e * mat.value(cell, 0, "A")? * read_in("eps")?;
                out[1] = e * mat.value(cell, 0, "I_y")? * read_in("kappa_y")?;
                out[2] = e * mat.value(cell, 0, "I_z")? * read_in("kappa_z")?;
                out[3] = mat.value(cell, 0, "G")? * mat.value(cell, 0, "J")? * read_in("torsion")?;
            }
        }
        Ok(())
    }
}

// ─── The closed forms ───────────────────────────────────────────────────────

/// The Hermite bending stiffness over `[w_A, θ_A, w_B, θ_B]`,
/// `∫ EI (∂²N/∂x²)ᵀ(∂²N/∂x²) dx`.
///
/// This is the whole of Euler-Bernoulli: everything else in this file surrounds
/// it with an axial term, a torsion, or a rotation.
///
/// It is **integrated**, from the `HERMITE3` basis the subspace declares, and
/// not from the classical closed form. The two agree to machine precision — the
/// integrand is quadratic in ξ, so two Gauss points are exact — and
/// `tests/hermite.rs` asserts it. Integrating is nevertheless the right choice:
/// it leaves one source of truth, and it makes the declared interpolation
/// **load-bearing**. A basis that was wrong would now produce a wrong stiffness
/// and fail the beam tests, where before it could have been anything at all.
fn bending_4x4(geom: &CellGeom, ei: f64) -> Result<Vec<Vec<f64>>> {
    let mut k = vec![vec![0.0_f64; 4]; 4];
    for g in 0..geom.n_gauss {
        let b = geom.field_d2n_dx2(g)?;
        let w = geom.det_j_w(g)?;
        for a in 0..4 {
            for c in 0..4 {
                k[a][c] += ei * b[a] * b[c] * w;
            }
        }
    }
    Ok(k)
}

/// Plane frame, local DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`.
fn plane_frame(geom: &CellGeom, ea: f64, ei: f64, l: f64) -> Result<Vec<Vec<f64>>> {
    let mut k = vec![vec![0.0; 6]; 6];
    let ka = ea / l;
    k[0][0] = ka;
    k[3][3] = ka;
    k[0][3] = -ka;
    k[3][0] = -ka;
    let b = bending_4x4(geom, ei)?;
    let idx = [1usize, 2, 4, 5];
    for (a, &ia) in idx.iter().enumerate() {
        for (c, &ic) in idx.iter().enumerate() {
            k[ia][ic] += b[a][c];
        }
    }
    Ok(k)
}

/// Space frame, local DOFs `[u, v, w, θ_x, θ_y, θ_z]` per node.
///
/// The two bending planes are the Timoshenko closed form with `Φ = 0` — which is
/// exactly what dropping the shear compliance means.
fn space_frame(
    geom: &CellGeom,
    ea: f64,
    ei_y: f64,
    ei_z: f64,
    gj: f64,
    l: f64,
) -> Result<Vec<Vec<f64>>> {
    let mut k = vec![vec![0.0; 12]; 12];
    // Axial, on u_A (0) and u_B (6).
    let ka = ea / l;
    k[0][0] = ka;
    k[6][6] = ka;
    k[0][6] = -ka;
    k[6][0] = -ka;
    // Torsion, on θ_x (3) and (9).
    let kt = gj / l;
    k[3][3] = kt;
    k[9][9] = kt;
    k[3][9] = -kt;
    k[9][3] = -kt;

    // Bending in the x-y plane (deflection v, rotation θ_z), stiffness EI_z.
    let b_xy = bending_4x4(geom, ei_z)?;
    let idx_xy = [1usize, 5, 7, 11];
    for (a, &ia) in idx_xy.iter().enumerate() {
        for (c, &ic) in idx_xy.iter().enumerate() {
            k[ia][ic] += b_xy[a][c];
        }
    }
    // Bending in the x-z plane (deflection w, rotation θ_y), stiffness EI_y. The
    // sign of the rotation is opposite there — a positive θ_y bends towards −z —
    // so the coupling terms flip.
    let b_xz = bending_4x4(geom, ei_y)?;
    let idx_xz = [2usize, 4, 8, 10];
    let sign = [1.0, -1.0, 1.0, -1.0];
    for (a, &ia) in idx_xz.iter().enumerate() {
        for (c, &ic) in idx_xz.iter().enumerate() {
            k[ia][ic] += sign[a] * sign[c] * b_xz[a][c];
        }
    }
    Ok(k)
}

/// The 2-D rotation `T` (6×6) mapping global DOFs to local, per node
/// `[[c, s, 0], [−s, c, 0], [0, 0, 1]]`.
fn rotation_2d(c: f64, s: f64) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; 6]; 6];
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

/// The 3-D rotation `T` (12×12): the local triad on each of the four
/// translation/rotation triplets.
///
/// The section axes are taken from a global-Z reference (global Y for a member
/// within a thousandth of vertical), so no orientation data is needed — sound
/// for a symmetric section, and the same convention
/// [`frame3d`](crate::models::frame3d) uses.
fn rotation_3d(dir: &[f64]) -> Vec<Vec<f64>> {
    let x = [dir[0], dir[1], dir[2]];
    let reference = if x[2].abs() > 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let normalise = |v: [f64; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let y = normalise(cross(reference, x));
    let z = cross(x, y);
    let r = [x, y, z];

    let mut t = vec![vec![0.0; 12]; 12];
    for block in 0..4 {
        let o = block * 3;
        for i in 0..3 {
            for j in 0..3 {
                t[o + i][o + j] = r[i][j];
            }
        }
    }
    t
}

/// Copy a dense local matrix into the flat row-major `ke`.
fn copy(local: &[Vec<f64>], ke: &mut [f64], side: usize) {
    for i in 0..side {
        for j in 0..side {
            ke[i * side + j] += local[i][j];
        }
    }
}

/// `ke += Tᵀ · local · T` — the local matrix carried to the global axes.
fn congruent(local: &[Vec<f64>], t: &[Vec<f64>], ke: &mut [f64], side: usize) {
    // `local · T` first, then `Tᵀ · (…)`: two matrix products rather than a
    // triple loop, and the intermediate is what a reader can check by eye.
    let mut lt = vec![vec![0.0; side]; side];
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                acc += local[i][k] * t[k][j];
            }
            lt[i][j] = acc;
        }
    }
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                acc += t[k][i] * lt[k][j];
            }
            ke[i * side + j] += acc;
        }
    }
}

/// Consistent **mass** of a 1-D Euler-Bernoulli beam, on `[w_A, θ_A, w_B, θ_B]`.
///
/// The shared block with **no** shear compliance: `Φ = 0` there, and what comes
/// back is the classical `ρAL/420·[156, 22L, 54, −13L; …]`, which
/// `models::beam` asserts against that very table. Bernoulli therefore adds no
/// derivation of its own — it is the `Φ → 0` end of one.
fn planar_mass(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let cell = geom.cell;
    let (xa, xb) = (geom.node_coord(0)?, geom.node_coord(1)?);
    let l = (xb[0] - xa[0]).abs();
    let rho = material.value(cell, 0, "rho").map_err(|_| {
        PyrucastError::Message(
            "Bernoulli mass matrix: material component `rho` (density) is required".into(),
        )
    })?;
    let i = material.value(cell, 0, "I")?;
    let m = crate::models::beam::mass_4x4(
        rho * material.value(cell, 0, "A")?,
        rho * i,
        material.value(cell, 0, "E")? * i,
        None,
        l,
    );
    for (r, row) in m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            ke[r * 4 + c] += v;
        }
    }
    Ok(())
}
