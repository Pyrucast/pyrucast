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
use crate::containers::field::ABSENT_COMPONENT;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::owned_components;
use crate::models::ZoneLayout;
use crate::models::{Behavior, CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::models::{ElementLayout, MatrixKind};
use serde::{Deserialize, Serialize};

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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let n: Vec<Node> = [[0.0], [1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::new(&maillage, Interpolation::Hermite3).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::bernoulli::Bernoulli;
/// // La poutre d'Euler-Bernoulli : flèche interpolée en Hermite cubique,
/// // d'où deux fonctions de forme par nœud.
/// let b = Bernoulli::new(zone.clone())?;
/// assert!(b.material_components().contains(&"E".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let n: Vec<Node> = [[0.0], [1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::new(&maillage, Interpolation::Hermite3).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::bernoulli::Bernoulli;
    /// // La poutre d'Euler-Bernoulli : flèche interpolée en Hermite cubique,
    /// // d'où deux fonctions de forme par nœud.
    /// let b = Bernoulli::new(zone.clone())?;
    /// assert!(b.material_components().contains(&"E".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    fn axis(&self, geom: &CellGeom) -> Result<(f64, [f64; 3])> {
        let d = geom.space_dim;
        // Deux emprunts immuables coexistent : rien à recopier, et la direction
        // tient dans trois nombres.
        let (a, b) = (geom.node_coord(0), geom.node_coord(1));
        let mut delta = [0.0_f64; 3];
        for i in 0..d {
            delta[i] = b[i] - a[i];
        }
        let l = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if l <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "Bernoulli: cell {} has zero length",
                geom.cell
            )));
        }
        for v in &mut delta {
            *v /= l;
        }
        Ok((l, delta))
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

    fn as_behavior(&self) -> Option<&dyn Behavior> {
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

    /// Internal forces `f = ∫ Bᵀ σ dx` — the **transpose** of the very `B` this
    /// physics' deformation operator applies
    /// ([`crate::models::beam::b_into`]), integrated against the section
    /// forces. The continuum default reads a Voigt stress tensor, which a field
    /// carrying `N` and `M` has never had.
    ///
    /// There is no shear ratio to read, and that is the theory speaking:
    /// Euler-Bernoulli *is* `Φ = 0`, so the shear rows of `B` vanish — and the
    /// beam reports no shear force to weigh them with. Its read list therefore
    /// stops where its section forces do, the remaining rows of `B` weighing
    /// nothing. The two absences are one statement.
    fn internal_force_reads(&self) -> Vec<String> {
        owned_components(behavior_of(self.model))
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        crate::models::beam::internal_force_into(self.model, &geoms[0], stress, lay, &[], fe);
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

    fn material_components(&self) -> Vec<String> {
        owned_components(material_of(self.model))
    }

    /// `rho` for the mass, and — in the 1-D configuration — the full area `A`,
    /// which only the mass needs (a pure-bending beam's stiffness uses none).
    fn optional_material_components(&self) -> &'static [&'static str] {
        match self.model {
            BeamModel::Planar1d => &["A", "rho"],
            _ => &["rho"],
        }
    }

    /// La raideur géométrique de la poutre lit son effort normal.
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        match kind {
            MatrixKind::Geometric => vec!["N".to_string()],
            _ => Vec::new(),
        }
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let row = material.row(geom.cell, 0);
        let m = |k: usize| row[lay.material[k] as usize];
        // `rho` ferme la liste des facultatives ; sans elle, pas de masse.
        let rho = optional(row, lay, lay.optional_material.len() - 1, "rho")?;
        // Une poutre de Bernoulli ne déclare **ni** `G` **ni** `A_s` : cette
        // absence *est* l'énoncé qu'il n'y a pas de souplesse au cisaillement.
        match self.model {
            // [E, I] + facultatives [A, rho]
            BeamModel::Planar1d => {
                let a = optional(row, lay, 0, "A")?;
                planar_mass(geom, rho * a, rho * m(1), m(0) * m(1), ke)
            }
            // [E, A, I]
            BeamModel::Frame2d => crate::models::frame::element_mass(
                geom,
                rho * m(1),
                rho * m(2),
                m(0) * m(2),
                None,
                ke,
            ),
            // [E, A, I_y, I_z, J, G]
            BeamModel::Frame3d => crate::models::frame3d::element_mass(
                geom,
                rho,
                m(1),
                m(2),
                m(3),
                m(0),
                None,
                None,
                None,
                ke,
            ),
        }
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: &SubElementField,
        lay: &ElementLayout,
        state: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        match self.model {
            BeamModel::Planar1d => Err(PyrucastError::Message(
                "Bernoulli: a pure-bending beam carries no axial force, so it has no geometric \
                 stiffness — use a 2-D or 3-D configuration"
                    .into(),
            )),
            _ => {
                let n = state.row(geom.cell, 0)[lay.state[0] as usize];
                match self.model {
                    BeamModel::Frame2d => crate::models::frame::element_geometric(geom, n, ke),
                    _ => crate::models::frame3d::element_geometric(geom, n, ke),
                }
            }
        }
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let (l, dir) = self.axis(geom)?;
        // Le contrat de chaque configuration (`material_of`), lu par les indices
        // que la zone a résolus une fois.
        let row = material.row(geom.cell, 0);
        let m = |k: usize| row[lay.material[k] as usize];
        let e = m(0);

        match self.model {
            // [E, I]
            BeamModel::Planar1d => {
                let local = bending_4x4(geom, e * m(1))?;
                copy(&local, ke, 4);
            }
            // [E, A, I]
            BeamModel::Frame2d => {
                let local = plane_frame(geom, e * m(1), e * m(2), l)?;
                let t = rotation_2d(dir[0], dir[1]);
                congruent(&local, &t, ke, 6);
            }
            // [E, A, I_y, I_z, J, G]
            BeamModel::Frame3d => {
                let local = space_frame(geom, e * m(1), e * m(2), e * m(3), m(5) * m(4), l)?;
                let t = rotation_3d(&dir);
                congruent(&local, &t, ke, 12);
            }
        }
        Ok(())
    }
}

impl Behavior for Bernoulli {
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        behavior_of(self.model)
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// The section forces from the generalised strains — a **linear** law, as it
    /// is for every structural element: `N = EA·ε`, `M = EI·κ`, `T = GJ·φ'`.
    fn deformation_reads(&self) -> Vec<String> {
        owned_components(match self.model {
            BeamModel::Planar1d => &["kappa"][..],
            BeamModel::Frame2d => &["eps", "kappa"][..],
            BeamModel::Frame3d => &["eps", "kappa_y", "kappa_z", "torsion"][..],
        })
    }

    fn integrate_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        // Both orders are this physics' own: `material_of` for the constants,
        // `deformation_reads` for the section strains.
        let m = |k: usize| material[lay.material[k] as usize];
        let e = |k: usize| deformation[lay.deformation[k] as usize];
        match self.model {
            BeamModel::Planar1d => {
                out[0] = m(0) * m(1) * e(0); // E·I·κ
            }
            BeamModel::Frame2d => {
                out[0] = m(0) * m(1) * e(0); // E·A·ε
                out[1] = m(0) * m(2) * e(1); // E·I·κ
            }
            BeamModel::Frame3d => {
                out[0] = m(0) * m(1) * e(0); // E·A·ε
                out[1] = m(0) * m(2) * e(1); // E·I_y·κ_y
                out[2] = m(0) * m(3) * e(2); // E·I_z·κ_z
                out[3] = m(5) * m(4) * e(3); // G·J·torsion
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
fn bending_4x4(geom: &CellGeom, ei: f64) -> Result<[[f64; 4]; 4]> {
    let mut k = [[0.0_f64; 4]; 4];
    let mut b_buf = [0.0_f64; MAX_CELL_DOFS];
    for g in 0..geom.n_gauss {
        let n = geom.field_d2n_dx2(g, &mut b_buf);
        let b = &b_buf[..n];
        let w = geom.det_j_w(g);
        for a in 0..4 {
            for c in 0..4 {
                k[a][c] += ei * b[a] * b[c] * w;
            }
        }
    }
    Ok(k)
}

/// Plane frame, local DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`.
fn plane_frame(geom: &CellGeom, ea: f64, ei: f64, l: f64) -> Result<[[f64; 6]; 6]> {
    let mut k = [[0.0_f64; 6]; 6];
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
) -> Result<[[f64; 12]; 12]> {
    let mut k = [[0.0_f64; 12]; 12];
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
fn rotation_2d(c: f64, s: f64) -> [[f64; 6]; 6] {
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

/// The 3-D rotation `T` (12×12): the local triad on each of the four
/// translation/rotation triplets.
///
/// The section axes are taken from a global-Z reference (global Y for a member
/// within a thousandth of vertical), so no orientation data is needed — sound
/// for a symmetric section, and the same convention
/// [`frame3d`](crate::models::frame3d) uses.
fn rotation_3d(dir: &[f64]) -> [[f64; 12]; 12] {
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

    let mut t = [[0.0_f64; 12]; 12];
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
fn copy<const N: usize>(local: &[[f64; N]; N], ke: &mut [f64], side: usize) {
    for i in 0..side {
        for j in 0..side {
            ke[i * side + j] += local[i][j];
        }
    }
}

/// `ke += Tᵀ · local · T` — the local matrix carried to the global axes.
fn congruent<const N: usize>(
    local: &[[f64; N]; N],
    t: &[[f64; N]; N],
    ke: &mut [f64],
    side: usize,
) {
    // `local · T` first, then `Tᵀ · (…)`: two matrix products rather than a
    // triple loop, and the intermediate is what a reader can check by eye.
    // Il vit sur la pile : une poutre en assemblait un par maille.
    let mut lt = [[0.0_f64; N]; N];
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
/// Une composante facultative, lue par l'indice que la zone a résolu — absente,
/// elle se nomme dans l'erreur plutôt que de valoir zéro en silence.
fn optional(row: &[f64], lay: &ElementLayout, slot: usize, name: &str) -> Result<f64> {
    match lay.optional_material[slot] {
        ABSENT_COMPONENT => Err(PyrucastError::Message(format!(
            "Bernoulli mass matrix: material component `{name}` is required"
        ))),
        i => Ok(row[i as usize]),
    }
}

fn planar_mass(geom: &CellGeom, rho_a: f64, rho_i: f64, ei: f64, ke: &mut [f64]) -> Result<()> {
    let (xa, xb) = (geom.node_coord(0), geom.node_coord(1));
    let l = (xb[0] - xa[0]).abs();
    let m = crate::models::beam::mass_4x4(rho_a, rho_i, ei, None, l);
    for (r, row) in m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            ke[r * 4 + c] += v;
        }
    }
    Ok(())
}

crate::physics_operator! {
    /// Euler-Bernoulli beam `Model` spanning **every** subspace of `fes`.
    /// Parent-level operator; material is supplied at assembly time.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::models::{Physics, RelationSense};
    /// # use pyrucast::ops::mesh;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # use pyrucast::atoms::Interpolation;
    /// # let mut b = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # b.add_cell(&[n[0].id(), n[1].id()])?;
    /// let poutres = FiniteElementSpace::new(&Mesh::from_submesh(b), Interpolation::Hermite3)?;
    /// let m = model::bernoulli(&poutres)?;
    /// assert_eq!(m.len(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn bernoulli(fes) via SubModel::bernoulli;
    python: "`model.bernoulli(fespace)` — the classical **Euler-Bernoulli** beam,\nwhere plane sections stay normal to the deflected axis and there is no\ntransverse shear at all.\n\nThe configuration follows the mesh: a 1-D `Coords` gives a pure-bending\nbeam (DOFs `w`, `theta`; material `E`, `I`), a 2-D one a plane frame\n(`u_x, u_y, r_z`; `+ A`), a 3-D one a space frame (six DOFs;\n`+ I_y, I_z, J, G`). Read them back with `model.primal_vars()`.\n\nThe deflection is interpolated by **cubic Hermite** functions, so the\nsubspace must be `HERMITE3` — build it with\n`FiniteElementSpace(mesh, interpolation=\"HERMITE3\")`. That basis is what\nmakes the element **nodally exact** wherever the interior carries no\ndistributed load, so one element per member suffices for a frame; a\nLagrange subspace would carry a linear deflection, of zero curvature, and\nis refused.\n\nPrefer `timoshenko` for a stocky member, where the shear compliance\nmatters. Reaching Bernoulli by making the shear area huge would work in\nexact arithmetic and lock in floating point, which is why this is a\nphysics of its own rather than a limiting case."
}
