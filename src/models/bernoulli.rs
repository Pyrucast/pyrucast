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
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Which configuration a Bernoulli beam is in — the kinematics, not the theory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamModel {
    /// Pure bending in a 1-D configuration: deflection and section rotation.
    Planar1d,
    /// Plane frame: axial + bending, rotated to the global axes.
    Frame2d,
    /// Space frame: axial, torsion and bending about two principal axes.
    Frame3d,
}

impl BeamModel {
    /// The configuration a beam is in, **read from the geometry**.
    ///
    /// It was once an argument, and it could only ever take the one value
    /// matching the mesh: every constructor rejected the others outright. An
    /// argument that can hold exactly one value carries no information — it
    /// only offers a way to contradict oneself — so the configuration is
    /// derived instead.
    ///
    /// Everything that distinguishes the three (degrees of freedom, material,
    /// section forces, whether there is an axial term, a torsion, a rotation to
    /// the global axes) follows from the dimension, and none of it is a choice.
    pub fn from_space_dim(space_dim: usize) -> Result<Self> {
        match space_dim {
            1 => Ok(Self::Planar1d),
            2 => Ok(Self::Frame2d),
            3 => Ok(Self::Frame3d),
            d => Err(PyrucastError::Message(format!(
                "a beam lives in a 1-, 2- or 3-D configuration, got {d}-D"
            ))),
        }
    }

    /// The lowercase name of the configuration — for messages and rendering.
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Planar1d => "planar_1d",
            Self::Frame2d => "frame_2d",
            Self::Frame3d => "frame_3d",
        }
    }

    fn primal(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["w", "theta"],
            Self::Frame2d => &["u_x", "u_y", "r_z"],
            Self::Frame3d => &["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"],
        }
    }

    fn dual(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["f_w", "m_theta"],
            Self::Frame2d => &["f_x", "f_y", "m_z"],
            Self::Frame3d => &["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"],
        }
    }

    /// The material a configuration needs. No `G`, no `A_s` where there is no
    /// shear — asking for a constant a theory does not use is a way of inviting
    /// the wrong one.
    fn material(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["E", "I"],
            Self::Frame2d => &["E", "A", "I"],
            Self::Frame3d => &["E", "A", "I_y", "I_z", "J", "G"],
        }
    }

    /// The section forces the behaviour reports.
    fn behavior(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["M"],
            Self::Frame2d => &["N", "M"],
            Self::Frame3d => &["N", "M_y", "M_z", "T"],
        }
    }
}

impl std::fmt::Display for BeamModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

/// Euler-Bernoulli beam physics on a `SEG2` FE subspace.
#[derive(Clone, Serialize, Deserialize)]
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
            let s = read(&fespace)?;
            (
                s.submesh(),
                s.space_dim(),
                read(&s.submesh())?.element_type(),
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
        let support = read(&submesh)?.to_poi1()?;
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
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
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

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(self.model.material())
    }

    fn optional_material_components(&self) -> &'static [&'static str] {
        &["rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(self
            .model
            .behavior()
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
