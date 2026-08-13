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
//! | `thin` (Kirchhoff-Love) | none at all | TRI6, QUA8, QUA9 | thin shells, needs second derivatives |
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
/// The generalised forces a shell behaviour reports: membrane, bending, shear.
const BEHAVIOR: [&str; 8] = [
    "N_xx", "N_yy", "N_xy", "M_xx", "M_yy", "M_xy", "Q_xz", "Q_yz",
];

/// Which shell formulation a [`Shell`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellModel {
    /// Reissner-Mindlin: the transverse shear is a degree of freedom of its own,
    /// integrated **reduced** against locking. The general case.
    #[default]
    Thick,
}

impl ShellModel {
    /// Parse from a lowercase tag (`"thick"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "thick" => Some(Self::Thick),
            _ => None,
        }
    }

    /// The lowercase tag (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Thick => "thick",
        }
    }

    /// The accepted tags, `|`-joined — for error messages.
    pub fn tag_list() -> String {
        ["thick"].join("|")
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
    /// shear, whose full integration is what locks a thin shell.
    pub(crate) shear: Handle<SubFiniteElementSpace>,
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
        // The shear subspace shares the mesh and differs only by quadrature —
        // the same multi-quadrature pattern as the Timoshenko beam.
        let shear = insert(SubFiniteElementSpace::new(
            submesh.clone(),
            Interpolation::Lagrange1,
            QuadratureRule::Reduced,
        )?);
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

    /// **Two** FE subspaces: the full quadrature drives the cell loop and the
    /// membrane/bending terms, the reduced one the transverse shear. Exactly the
    /// multi-quadrature layout the Timoshenko beam introduced.
    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone(), self.shear.clone()],
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
    /// (`5/6` by default, the value for a homogeneous rectangular section).
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["rho", "k_s"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(BEHAVIOR.iter().map(|s| s.to_string()).collect())
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
        let ds = thick::shear_law(e, nu, h, thick::shear_factor(mat, cell));

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
        let gamma = [
            input.value(cell, g, "gamma_xz")?,
            input.value(cell, g, "gamma_yz")?,
        ];
        for i in 0..3 {
            out[i] = (0..3).map(|j| dm[i][j] * eps[j]).sum();
            out[3 + i] = (0..3).map(|j| db[i][j] * kappa[j]).sum();
        }
        for i in 0..2 {
            out[6 + i] = ds * gamma[i];
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
