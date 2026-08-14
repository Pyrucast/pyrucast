//! Shells — the physics, for any shell formulation.
//!
//! A shell is a **surface** carrying membrane forces *and* bending moments. Its
//! elements are manifolds (`ref_dim = 2` in a 3-D space), which is exactly the
//! case the continuum guard of [`crate::models::elasticity`] rejects: a solid
//! kernel would build `B` from the tangential gradient and be rank-deficient
//! through the thickness. A shell has its own kernel, and its own kinematics.
//!
//! As everywhere else in this crate, the **formulation is an attribute**
//! ([`ShellModel`]) rather than a physics of its own: the DOFs, the local frame,
//! the membrane law and the rotation to the global axes are shared, and only the
//! bending/shear treatment differs.
//!
//! | formulation | transverse shear | elements | when |
//! |---|---|---|---|
//! | `thick` (Reissner-Mindlin) | yes, reduced-integrated | TRI3, QUA4 | the general case |
//! | `kirchhoff` (DKT/DKQ) | imposed zero at discrete points | TRI3, QUA4 | thin shells, no locking by construction |
//!
//! ## Six DOFs per node, and the drilling one
//!
//! The natural kinematics of a shell is five degrees of freedom — three
//! translations and two rotations of the normal fibre. But the fifth and sixth
//! are only distinguishable in the **local** frame, and a global assembler
//! numbers DOFs by name. So the element carries six, `u_x…u_z, r_x…r_z`, exactly
//! like [`frame3d`](crate::models::frame3d) — which also lets a shell and a
//! space frame share nodes with no adaptor at all.
//!
//! The sixth, the rotation about the normal, is the **drilling** DOF, and a flat
//! facet has no physical stiffness against it: left alone it makes the element
//! matrix singular. It is tied instead to the membrane's own in-plane rotation,
//!
//! ```text
//! ω_z = ½(∂v/∂x − ∂u/∂y),      K_drill = α·G·h·∫ (θ_z − ω_z)² dA
//! ```
//!
//! which is a physically meaningful statement (the drilling rotation should
//! follow the material's) rather than a numerical prop. A plain diagonal penalty
//! would be the tempting shortcut and would be **wrong**: it resists a rigid
//! rotation of the whole facet about its normal, which costs no energy.
//!
//! ## The local frame is per element
//!
//! Nodal DOFs are global, so the rotation from local to global axes must be one
//! matrix per **element**, not per Gauss point. It is built from the node
//! coordinates — the first edge, and the facet normal — which makes it exact for
//! a flat facet and a well-behaved average for a slightly warped one.

pub mod kirchhoff;
pub mod thick;

use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::{
    Interpolation, QuadratureRule, SubFiniteElementSpace,
};
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};

/// Primal DOF names — three translations and three rotations, as for a space
/// frame, so the two share nodes directly.
const PRIMAL: [&str; 6] = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];
/// Dual DOF names (force and moment).
const DUAL: [&str; 6] = ["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"];
/// Required material: the elastic constants and the thickness.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu", "h"];
/// The generalised forces a shell behaviour reports: membrane, bending, and —
/// where the formulation has one — transverse shear.
const BEHAVIOR: [&str; 8] = [
    "N_xx", "N_yy", "N_xy", "M_xx", "M_yy", "M_xy", "Q_xz", "Q_yz",
];
/// The six a formulation without transverse shear reports: `BEHAVIOR` cut short.
const BEHAVIOR_NO_SHEAR: usize = 6;

/// Which shell formulation a [`Shell`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellModel {
    /// Reissner-Mindlin: the transverse shear is a degree of freedom of its own,
    /// integrated **reduced** against locking. The general case.
    #[default]
    Thick,
    /// Discrete Kirchhoff (DKT on TRI3, DKQ on QUA4): the transverse shear is
    /// **imposed zero** at discrete points rather than integrated, so the thin
    /// limit is exact by construction and there is nothing left to lock.
    ///
    /// > New variants go at the **end** — `bincode` serialises the index.
    Kirchhoff,
}

impl ShellModel {
    /// Parse from a lowercase tag (`"thick"`, `"kirchhoff"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "thick" => Some(Self::Thick),
            "kirchhoff" => Some(Self::Kirchhoff),
            _ => None,
        }
    }

    /// The lowercase tag (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Thick => "thick",
            Self::Kirchhoff => "kirchhoff",
        }
    }

    /// The accepted tags, `|`-joined — for error messages.
    pub fn tag_list() -> String {
        ["thick", "kirchhoff"].join("|")
    }

    /// Whether the formulation carries a transverse shear at all. A discrete
    /// Kirchhoff element does not: it has no shear strain, no shear subspace and
    /// no shear force to report.
    pub fn has_transverse_shear(self) -> bool {
        matches!(self, Self::Thick)
    }
}

impl std::fmt::Display for ShellModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

/// Shell physics on a **surface** FE subspace in 3-D.
#[derive(Clone, Serialize, Deserialize)]
pub struct Shell {
    /// Full-quadrature subspace — membrane and bending.
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// Reduced-quadrature subspace over the **same** submesh — the transverse
    /// shear, whose full integration is what locks a thin shell. `None` for a
    /// formulation that has no transverse shear to integrate.
    pub(crate) shear: Option<Handle<SubFiniteElementSpace>>,
    /// POI1 support over the unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) model: ShellModel,
}

impl Shell {
    /// Shell physics on a surface FE subspace. Errors unless the subspace is a
    /// **manifold** of TRI3 or QUA4 in a 3-D configuration.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ShellModel) -> Result<Self> {
        let (submesh, space_dim, ref_dim, element) = {
            let s = read(&fespace)?;
            let sm = s.submesh();
            let element = read(&sm)?.element_type();
            (sm, s.space_dim(), s.ref_dim()?, element)
        };
        if space_dim != 3 || ref_dim != 2 {
            return Err(PyrucastError::Message(format!(
                "Shell: a shell is a surface in space — expected a 2-D element in a 3-D \
                 configuration, got a {ref_dim}-D element in {space_dim}-D"
            )));
        }
        if !matches!(element, ElementType::TRI3 | ElementType::QUA4) {
            return Err(PyrucastError::Message(format!(
                "Shell ({model}): expected TRI3 or QUA4, got {element:?}"
            )));
        }
        // The shear subspace shares the mesh and differs only by quadrature. It
        // is built here rather than asked for: nothing about it is the caller's
        // to choose, and `element_matrix` reads the two `CellGeom` as **one**
        // cell — an invariant worth establishing by construction rather than
        // validating. A discrete Kirchhoff element has no shear term at all, so
        // it declares none: a second subspace would be an integration the
        // formulation never does.
        let shear = model.has_transverse_shear().then(|| {
            SubFiniteElementSpace::new(
                submesh.clone(),
                Interpolation::Lagrange1,
                QuadratureRule::Reduced,
            )
            .map(insert)
        });
        let shear = match shear {
            Some(r) => Some(r?),
            None => None,
        };
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            shear,
            support,
            model,
        })
    }
}

impl SubModelKind for Shell {
    fn primal_vars(&self) -> Vec<String> {
        PRIMAL.iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        DUAL.iter().map(|s| s.to_string()).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    /// **Two** FE subspaces where there is a shear to integrate: the full
    /// quadrature drives the cell loop and the membrane/bending terms, the
    /// reduced one the transverse shear. That is the whole of the
    /// multi-quadrature layout, and this is currently its only user.
    ///
    /// A discrete Kirchhoff element declares **one**, having no second
    /// integration to do — which is the layout saying what the formulation is,
    /// rather than a shape it has to fit.
    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        let mut fespaces = vec![self.fespace.clone()];
        fespaces.extend(self.shear.clone());
        Some(MatrixLayout {
            fespaces,
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
        let mat = material.expect("Shell declares a material_fespace");
        match self.model {
            ShellModel::Thick => thick::element_stiffness(&geoms[0], &geoms[1], mat, ke),
            ShellModel::Kirchhoff => kirchhoff::element_stiffness(&geoms[0], mat, ke),
        }
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Shell"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Shell({})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Shell {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    /// `rho` for a mass matrix, `k_s` to override the shear-correction factor
    /// (`5/6` by default, the value for a homogeneous rectangular section) —
    /// the latter only where there is a shear to correct.
    fn optional_material_components(&self) -> &'static [&'static str] {
        match self.model {
            ShellModel::Thick => &["rho", "k_s"],
            ShellModel::Kirchhoff => &["rho"],
        }
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// Membrane and bending always; the two shear forces only where the
    /// formulation has a shear strain to derive them from. A discrete Kirchhoff
    /// element does not: its `Q` is a **reaction**, recovered from the gradient
    /// of the moments, not a constitutive product — reporting it here would
    /// state a law that does not exist.
    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let n = if self.model.has_transverse_shear() {
            BEHAVIOR.len()
        } else {
            BEHAVIOR_NO_SHEAR
        };
        Ok(BEHAVIOR[..n].iter().map(|s| s.to_string()).collect())
    }

    /// The generalised forces from the generalised strains — a linear law, as
    /// for every structural element: `N = D_m·ε`, `M = D_b·κ`, `Q = D_s·γ`.
    ///
    /// The strains come in as the components a shell-deformation operator would
    /// produce (`eps_xx`, `kappa_xx`, `gamma_xz`, …), all in the **local** frame.
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
        let mat = material.expect("Shell declares a material_fespace");
        let cell = geom.cell;
        let (e, nu, h) = (
            mat.value(cell, 0, "E")?,
            mat.value(cell, 0, "nu")?,
            mat.value(cell, 0, "h")?,
        );
        let dm = thick::membrane_law(e, nu, h);
        let db = thick::bending_law(e, nu, h);

        let eps = [
            input.value(cell, g, "eps_xx")?,
            input.value(cell, g, "eps_yy")?,
            input.value(cell, g, "eps_xy")?,
        ];
        let kappa = [
            input.value(cell, g, "kappa_xx")?,
            input.value(cell, g, "kappa_yy")?,
            input.value(cell, g, "kappa_xy")?,
        ];
        for i in 0..3 {
            out[i] = (0..3).map(|j| dm[i][j] * eps[j]).sum();
            out[3 + i] = (0..3).map(|j| db[i][j] * kappa[j]).sum();
        }
        if self.model.has_transverse_shear() {
            let ds = thick::shear_law(e, nu, h, thick::shear_factor(mat, cell));
            let gamma = [
                input.value(cell, g, "gamma_xz")?,
                input.value(cell, g, "gamma_yz")?,
            ];
            for i in 0..2 {
                out[6 + i] = ds * gamma[i];
            }
        }
        Ok(())
    }
}

// ─── The local frame ────────────────────────────────────────────────────────

/// The element's local triad `[e₁, e₂, n]`, each a unit vector in global
/// coordinates.
///
/// Built from the **node coordinates** — the first edge and the facet normal —
/// rather than from a Gauss point, because the nodal DOFs it rotates are one set
/// per element. Exact for a flat facet; for a slightly warped quadrilateral it is
/// the natural average, and warping beyond that is a meshing question.
pub fn local_frame(geom: &CellGeom) -> Result<[[f64; 3]; 3]> {
    let p0 = geom.node_coord(0)?.to_vec();
    let p1 = geom.node_coord(1)?.to_vec();
    let p2 = geom.node_coord(2)?.to_vec();
    let a: [f64; 3] = std::array::from_fn(|i| p1[i] - p0[i]);
    let b: [f64; 3] = std::array::from_fn(|i| p2[i] - p0[i]);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let unit = |v: [f64; 3], what: &str| -> Result<[f64; 3]> {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "Shell: cell {} is degenerate ({what} is null)",
                geom.cell
            )));
        }
        Ok([v[0] / n, v[1] / n, v[2] / n])
    };
    let e1 = unit(a, "the first edge")?;
    let n = unit(cross, "the normal")?;
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    Ok([e1, e2, n])
}

/// Shape-function derivatives with respect to the **local** in-plane axes at
/// Gauss point `g`, `[i][a]` for `a ∈ {0, 1}`.
///
/// The tangential gradient [`CellGeom::dn_dx`] already lies in the tangent
/// plane, so projecting it on `e₁`, `e₂` is exactly the local derivative — no
/// inverse, no second Jacobian.
pub fn local_derivatives(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    g: usize,
) -> Result<Vec<[f64; 2]>> {
    let dn = geom.dn_dx(g)?;
    Ok((0..geom.n_nodes)
        .map(|i| {
            let grad = [dn[i * 3], dn[i * 3 + 1], dn[i * 3 + 2]];
            [
                (0..3).map(|k| grad[k] * frame[0][k]).sum(),
                (0..3).map(|k| grad[k] * frame[1][k]).sum(),
            ]
        })
        .collect())
}

/// The node coordinates in the element's own plane, `[i] = (x, y)`.
///
/// The first node is the origin and the first edge the `x` axis, so a flat facet
/// is described exactly by two numbers per node. This is what a formulation
/// needs when its basis is written on the **geometry** rather than on the
/// reference element — a discrete-Kirchhoff element, whose side coefficients are
/// built from edge lengths and directions.
pub fn local_coords(geom: &CellGeom, frame: &[[f64; 3]; 3]) -> Result<Vec<[f64; 2]>> {
    let origin = geom.node_coord(0)?.to_vec();
    (0..geom.n_nodes)
        .map(|i| {
            let p = geom.node_coord(i)?;
            let d: [f64; 3] = std::array::from_fn(|k| p[k] - origin[k]);
            Ok([
                (0..3).map(|k| d[k] * frame[0][k]).sum(),
                (0..3).map(|k| d[k] * frame[1][k]).sum(),
            ])
        })
        .collect()
}

/// Weight of the drilling constraint, relative to `G·h`.
///
/// Small enough not to stiffen the shell, large enough to remove the
/// singularity. The constraint it weights is physical (`θ_z` should follow the
/// membrane rotation), so the answer is insensitive to the exact value over
/// several decades — which is what one wants from a regularisation.
const DRILLING_WEIGHT: f64 = 1e-3;

/// The membrane and drilling terms, at full quadrature — the part of a flat
/// facet that owes nothing to the bending theory, and which every formulation
/// therefore shares.
///
/// `local` is the `6 n × 6 n` matrix in the element frame, accumulated into.
pub fn membrane_and_drilling(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    e: f64,
    nu: f64,
    h: f64,
    local: &mut [Vec<f64>],
) -> Result<()> {
    let n = geom.n_nodes;
    let side = 6 * n;
    let dm = thick::membrane_law(e, nu, h);
    let g_mod = e / (2.0 * (1.0 + nu));

    for g in 0..geom.n_gauss {
        let dn = local_derivatives(geom, frame, g)?;
        let shape = geom.n_at_g(g)?;
        let w = geom.det_j_w(g)?;

        // Membrane `ε` on (u, v) — local DOFs 6i+0, 6i+1.
        let mut bm = vec![vec![0.0; side]; 3];
        // Drilling residual `θ_z − ω_z` on (u, v, θ_z).
        let mut bd = vec![0.0; side];
        for i in 0..n {
            let (dx, dy) = (dn[i][0], dn[i][1]);
            let (u, v, tz) = (6 * i, 6 * i + 1, 6 * i + 5);
            bm[0][u] = dx;
            bm[1][v] = dy;
            bm[2][u] = dy;
            bm[2][v] = dx;

            // ω_z = ½(∂v/∂x − ∂u/∂y), so the residual picks up its negative.
            bd[u] = 0.5 * dy;
            bd[v] = -0.5 * dx;
            bd[tz] = shape[i];
        }
        accumulate(local, &bm, &dm, w, side);
        // The drilling constraint is a scalar: its « law » is one coefficient.
        let kd = DRILLING_WEIGHT * g_mod * h * w;
        for a in 0..side {
            if bd[a] == 0.0 {
                continue;
            }
            for b in 0..side {
                local[a][b] += kd * bd[a] * bd[b];
            }
        }
    }
    Ok(())
}

/// `local += Bᵀ D B · w` for a 3-component strain.
pub fn accumulate(local: &mut [Vec<f64>], b: &[Vec<f64>], d: &[[f64; 3]; 3], w: f64, side: usize) {
    // `D B` first: three rows, so the inner loop stays short and the intermediate
    // is what a reader can check against the law above.
    let mut db = vec![vec![0.0; side]; 3];
    for (r, row) in db.iter_mut().enumerate() {
        for (c, e) in row.iter_mut().enumerate() {
            *e = (0..3).map(|k| d[r][k] * b[k][c]).sum();
        }
    }
    for a in 0..side {
        let contributes = (0..3).any(|k| b[k][a] != 0.0);
        if !contributes {
            continue;
        }
        for bcol in 0..side {
            let acc: f64 = (0..3).map(|k| b[k][a] * db[k][bcol]).sum();
            local[a][bcol] += acc * w;
        }
    }
}

/// Carry a local element matrix to the global axes: `ke += Tᵀ K_loc T`, with `T`
/// block-diagonal, the local triad on each translation and rotation triplet.
pub fn to_global(local: &[Vec<f64>], frame: &[[f64; 3]; 3], n_nodes: usize, ke: &mut [f64]) {
    let side = 6 * n_nodes;
    // `T` maps global DOFs to local: its rows are the local axes.
    let t = |row: usize, col: usize| -> f64 {
        let (br, bc) = (row / 3, col / 3);
        if br != bc {
            return 0.0;
        }
        frame[row % 3][col % 3]
    };
    let mut lt = vec![vec![0.0; side]; side];
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                let tkj = t(k, j);
                if tkj != 0.0 {
                    acc += local[i][k] * tkj;
                }
            }
            lt[i][j] = acc;
        }
    }
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                let tki = t(k, i);
                if tki != 0.0 {
                    acc += tki * lt[k][j];
                }
            }
            ke[i * side + j] += acc;
        }
    }
}
