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
use crate::containers::field::ABSENT_COMPONENT;
use crate::containers::finite_element_space::{
    Interpolation, QuadratureRule, SubFiniteElementSpace,
};
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::owned_components;
use crate::models::ElementLayout;
use crate::models::ZoneLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Primal DOF names — three translations and three rotations, as for a space
/// frame, so the two share nodes directly.
const PRIMAL: [&str; 6] = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];
/// Dual DOF names (force and moment).
const DUAL: [&str; 6] = ["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"];
/// Required material: the elastic constants and the thickness.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu", "h"];

/// La largeur maximale d'une coque : quatre nœuds (QUA4) à six DDL. `Shell::new`
/// refuse tout autre élément que TRI3 ou QUA4, donc la borne est acquise à la
/// construction — et tous les tampons d'un noyau tiennent sur la **pile**.
pub(crate) const MAX_SHELL_DOFS: usize = 24;

/// Une matrice de coque, `MAX_SHELL_DOFS²` à plat.
pub(crate) type ShellMatrix = [f64; MAX_SHELL_DOFS * MAX_SHELL_DOFS];
/// Les dérivées locales aux nœuds d'une coque.
pub(crate) type ShellNodes2 = [[f64; 2]; 4];

/// A shell's generalised strain-displacement matrix, on the stack.
///
/// Rows are the generalised strains in the order [`Shell::deformation_reads`]
/// names them; columns are the facet's **local** degrees of freedom, six per
/// node (`6i + k`). Nine by twenty-four is a `QUA4`'s size, and a formulation
/// with no transverse shear fills the leading block.
pub(crate) type ShellB = [[f64; MAX_SHELL_DOFS]; SHELL_STRAINS];

/// Rows of a [`ShellB`]: three membrane, three bending, one drilling, two
/// transverse shear.
pub(crate) const SHELL_STRAINS: usize = 9;
/// Where each block of rows begins. The first seven are integrated at **full**
/// quadrature and the last two at the **reduced** point, which is why the
/// drilling row sits between the bending and the shear rather than last: the
/// order is the order of integration.
const MEMBRANE_ROW: usize = 0;
const BENDING_ROW: usize = 3;
const DRILL_ROW: usize = 6;
pub(crate) const SHEAR_ROW: usize = 7;
/// The generalised strains a shell carries, in the row order of its `B` —
/// paired **by position** with the forces below.
const STRAINS: [&str; SHELL_STRAINS] = [
    "eps_xx", "eps_yy", "eps_xy", "kappa_xx", "kappa_yy", "kappa_xy", "drill", "gamma_xz",
    "gamma_yz",
];

/// The generalised forces a shell behaviour reports: membrane, bending, the
/// **drilling** moment, and — where the formulation has one — transverse shear.
///
/// `M_drill` is conjugate to the drilling residual `θ_z − ω_z`, and it is an
/// internal force like any other: the constraint does work, so a residual that
/// omitted it would not equal `K·u`. It was missing only because nothing had
/// ever asked a shell for its internal forces.
const BEHAVIOR: [&str; SHELL_STRAINS] = [
    "N_xx", "N_yy", "N_xy", "M_xx", "M_yy", "M_xy", "M_drill", "Q_xz", "Q_yz",
];
/// The seven a formulation without transverse shear reports: `BEHAVIOR` cut
/// short at the rows the reduced quadrature would have carried.
const BEHAVIOR_NO_SHEAR: usize = SHEAR_ROW;

/// Which shell formulation a [`Shell`] uses.
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// // Reissner-Mindlin porte le cisaillement transverse comme DDL propre ;
/// // Kirchhoff discret l'annule en des points choisis, ce qui rend la
/// // limite mince exacte et supprime tout blocage.
/// assert!(ShellModel::Thick.has_transverse_shear());
/// assert!(!ShellModel::Kirchhoff.has_transverse_shear());
/// ```
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
    /// The lowercase name (the inverse of
    /// [`from_name`](crate::named::Named::from_name)).
    ///
    /// ```
    /// # use pyrucast::models::shell::{self, ShellModel};
    /// assert_eq!(ShellModel::Thick.name(), "thick");
    /// # use pyrucast::named::Named;
    /// assert_eq!(ShellModel::from_name(ShellModel::Thick.name()), Some(ShellModel::Thick));
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Thick => "thick",
            Self::Kirchhoff => "kirchhoff",
        }
    }

    /// Whether the formulation carries a transverse shear at all. A discrete
    /// Kirchhoff element does not: it has no shear strain, no shear subspace and
    /// no shear force to report.
    ///
    /// ```
    /// # use pyrucast::models::shell::{self, ShellModel};
    /// // C'est ce qui décide s'il faut une **seconde** quadrature, réduite,
    /// // pour intégrer le cisaillement sans bloquer.
    /// assert!(ShellModel::Thick.has_transverse_shear());
    /// assert!(!ShellModel::Kirchhoff.has_transverse_shear());
    /// ```
    pub fn has_transverse_shear(self) -> bool {
        matches!(self, Self::Thick)
    }
}

impl crate::named::Named for ShellModel {
    const LABEL: &'static str = "shell model";
    const VALUES: &'static [Self] = &[Self::Thick, Self::Kirchhoff];

    fn name(self) -> &'static str {
        ShellModel::name(self)
    }
}

impl ShellModel {
    /// The **generalised strains** this formulation carries, in the row order of
    /// its `B` — and therefore in the order its section forces are conjugate to.
    ///
    /// The first seven are integrated at full quadrature (membrane, bending,
    /// drilling); the last two, the transverse shear, at the reduced point. A
    /// formulation without one simply stops after the seventh.
    ///
    /// ```
    /// # use pyrucast::models::shell::{self, ShellModel};
    /// assert_eq!(ShellModel::Thick.strains().len(), 9);
    /// // Kirchhoff discret n'a pas de cisaillement transverse : sa liste
    /// // s'arrête là où la quadrature réduite commençait.
    /// assert_eq!(ShellModel::Kirchhoff.strains().len(), 7);
    /// assert_eq!(ShellModel::Kirchhoff.strains()[6], "drill");
    /// ```
    pub fn strains(self) -> &'static [&'static str] {
        &STRAINS[..if self.has_transverse_shear() {
            SHELL_STRAINS
        } else {
            SHEAR_ROW
        }]
    }
}

impl std::fmt::Display for ShellModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Shell physics on a **surface** FE subspace in 3-D.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::shell::{Shell, ShellModel};
/// // Une coque vit sur une surface plongée en 3-D et porte l'épaisseur
/// // dans son matériau.
/// let s = Shell::new(zone.clone(), ShellModel::Thick)?;
/// assert!(s.material_components().contains(&"h".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(3).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::shell::{Shell, ShellModel};
    /// // Une coque vit sur une surface plongée en 3-D et porte l'épaisseur
    /// // dans son matériau.
    /// let s = Shell::new(zone.clone(), ShellModel::Thick)?;
    /// assert!(s.material_components().contains(&"h".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ShellModel) -> Result<Self> {
        let (submesh, space_dim, ref_dim, element) = {
            let s = fespace.read();
            let sm = s.submesh();
            let element = sm.read().element_type();
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
            .map(Handle::new)
        });
        let shear = match shear {
            Some(r) => Some(r?),
            None => None,
        };
        let support = submesh.read().to_poi1()?;
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

    /// Internal forces `f = ∫ Bᵀ σ dA` — the **transpose** of the `B` this
    /// physics builds its stiffness from (`b_into`, `shear_b_into`),
    /// integrated against the generalised forces. The continuum default reads a
    /// Voigt stress tensor, which a field carrying `N`, `M` and `Q` has never
    /// had.
    ///
    /// The read list is the behaviour's own output, in its own order, so the
    /// `k`-th force is conjugate to the `k`-th row of `B`.
    fn internal_force_reads(&self) -> Vec<String> {
        Domain::behavior_output_components(self)
    }

    /// The two quadratures again, on the other side of the same `B`: membrane,
    /// bending and drilling at the full Gauss points, the transverse shear at
    /// the **reduced** one — the point its stiffness integrates it at, and
    /// therefore the only point at which `∫ Bᵀσ` can equal `K·u`.
    ///
    /// The shear force is element-constant, a shell-deformation operator having
    /// sampled its strain at that same reduced point, so which full Gauss row it
    /// is read from does not matter.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let full = &geoms[0];
        let (n, cell) = (full.n_nodes, full.cell);
        let side = 6 * n;
        let frame = local_frame(full)?;
        let setup = bending_setup(self.model, full, &frame)?;
        // Les forces s'accumulent dans le repère **local**, celui où `B` est
        // écrit ; la rotation vers les axes globaux vient à la fin, une fois.
        let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];
        let mut local = [0.0_f64; MAX_SHELL_DOFS];
        for g in 0..full.n_gauss {
            b_into(full, &frame, &setup, g, &mut b)?;
            let row = stress.row(cell, g);
            let w = full.det_j_w(g);
            for (k, brow) in b[..SHEAR_ROW].iter().enumerate() {
                let s = row[lay[k] as usize] * w;
                for i in 0..side {
                    local[i] += brow[i] * s;
                }
            }
        }
        if let Some(reduced) = geoms.get(1) {
            let row = stress.row(cell, 0);
            for g in 0..reduced.n_gauss {
                shear_b_into(reduced, &frame, g, &mut b)?;
                let w = reduced.det_j_w(g);
                for (k, brow) in b[SHEAR_ROW..].iter().enumerate() {
                    let s = row[lay[SHEAR_ROW + k] as usize] * w;
                    for i in 0..side {
                        local[i] += brow[i] * s;
                    }
                }
            }
        }
        // `Tᵀ f_loc` : la transposée de la rotation qu'ont subie les DDL, par
        // triplet — translations puis rotations de chaque nœud.
        for blk in 0..(side / 3) {
            let o = blk * 3;
            for i in 0..3 {
                fe[o + i] += (0..3).map(|k| frame[k][i] * local[o + k]).sum::<f64>();
            }
        }
        Ok(())
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
        let n = self.support.read().cell_count();
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

    fn material_components(&self) -> Vec<String> {
        owned_components(MATERIAL_COMPONENTS)
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
    fn behavior_output_components(&self) -> Vec<String> {
        let n = if self.model.has_transverse_shear() {
            BEHAVIOR.len()
        } else {
            BEHAVIOR_NO_SHEAR
        };
        BEHAVIOR[..n].iter().map(|s| s.to_string()).collect()
    }

    /// The generalised forces from the generalised strains — a linear law, as
    /// for every structural element: `N = D_m·ε`, `M = D_b·κ`, `Q = D_s·γ`.
    ///
    /// The strains come in as the components a shell-deformation operator would
    /// produce (`eps_xx`, `kappa_xx`, `gamma_xz`, …), all in the **local** frame.
    fn deformation_reads(&self) -> Vec<String> {
        owned_components(self.model.strains())
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
        let (e, nu, h) = (
            material[lay.material[0] as usize],
            material[lay.material[1] as usize],
            material[lay.material[2] as usize],
        );
        let dm = thick::membrane_law(e, nu, h);
        let db = thick::bending_law(e, nu, h);
        let d = |k: usize| deformation[lay.deformation[k] as usize];

        let eps = [d(0), d(1), d(2)];
        let kappa = [d(3), d(4), d(5)];
        for i in 0..3 {
            out[i] = (0..3).map(|j| dm[i][j] * eps[j]).sum();
            out[3 + i] = (0..3).map(|j| db[i][j] * kappa[j]).sum();
        }
        // Le moment de vrillage : `α·G·h·(θ_z − ω_z)`, la loi d'une contrainte
        // dont la déformation est un seul nombre.
        out[DRILL_ROW] = drilling_law(e, nu, h) * d(DRILL_ROW);
        if self.model.has_transverse_shear() {
            // `k_s` overrides the 5/6 of a homogeneous rectangular section; which
            // of the two applies is a fact of the zone, settled in the layout.
            let k_s = match lay.optional_material[1] {
                ABSENT_COMPONENT => 5.0 / 6.0,
                i => material[i as usize],
            };
            let ds = thick::shear_law(e, nu, h, k_s);
            for i in 0..2 {
                out[SHEAR_ROW + i] = ds * d(SHEAR_ROW + i);
            }
        }
        Ok(())
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        match self.model {
            ShellModel::Thick => thick::element_stiffness(&geoms[0], &geoms[1], material, lay, ke),
            ShellModel::Kirchhoff => kirchhoff::element_stiffness(&geoms[0], material, lay, ke),
        }
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
/// // Un triade orthonormée `[e₁, e₂, n]`, bâtie sur les **nœuds** — la
/// // première arête et la normale de la facette — non sur un point de
/// // Gauss : les DDL qu'elle fait tourner sont un jeu par élément.
/// reduce_cells(&zone, |geom| {
///     let f = shell::local_frame(geom)?;
///     assert!((f[0][0] - 1.0).abs() < 1e-12); // e₁ suit la première arête
///     assert!((f[2][2] - 1.0).abs() < 1e-12); // n est la normale, ici +z
///     let dot: f64 = (0..3).map(|k| f[0][k] * f[1][k]).sum();
///     assert!(dot.abs() < 1e-12);
///     Ok(0.0)
/// })?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn local_frame(geom: &CellGeom) -> Result<[[f64; 3]; 3]> {
    // Trois emprunts immuables coexistent : rien à recopier.
    let (p0, p1, p2) = (geom.node_coord(0), geom.node_coord(1), geom.node_coord(2));
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
///
pub(crate) fn local_derivatives_into(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    g: usize,
    out: &mut ShellNodes2,
) -> Result<()> {
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    let dn = &mut dn_buf[..geom.n_nodes * geom.space_dim];
    geom.dn_dx(g, dn)?;
    for (i, o) in out[..geom.n_nodes].iter_mut().enumerate() {
        let grad = [dn[i * 3], dn[i * 3 + 1], dn[i * 3 + 2]];
        *o = [
            (0..3).map(|k| grad[k] * frame[0][k]).sum(),
            (0..3).map(|k| grad[k] * frame[1][k]).sum(),
        ];
    }
    Ok(())
}

/// The node coordinates in the element's own plane, `[i] = (x, y)`.
///
/// The first node is the origin and the first edge the `x` axis, so a flat facet
/// is described exactly by two numbers per node. This is what a formulation
/// needs when its basis is written on the **geometry** rather than on the
/// reference element — a discrete-Kirchhoff element, whose side coefficients are
/// built from edge lengths and directions.
///
pub(crate) fn local_coords_into(geom: &CellGeom, frame: &[[f64; 3]; 3], out: &mut ShellNodes2) {
    let origin = geom.node_coord(0);
    for (i, o) in out[..geom.n_nodes].iter_mut().enumerate() {
        let p = geom.node_coord(i);
        let d: [f64; 3] = std::array::from_fn(|k| p[k] - origin[k]);
        *o = [
            (0..3).map(|k| d[k] * frame[0][k]).sum(),
            (0..3).map(|k| d[k] * frame[1][k]).sum(),
        ];
    }
}

/// Weight of the drilling constraint, relative to `G·h`.
///
/// Small enough not to stiffen the shell, large enough to remove the
/// singularity. The constraint it weights is physical (`θ_z` should follow the
/// membrane rotation), so the answer is insensitive to the exact value over
/// several decades — which is what one wants from a regularisation.
const DRILLING_WEIGHT: f64 = 1e-3;

/// The drilling modulus `α·G·h` — the « law » of a constraint whose strain is a
/// single number.
///
/// It sits beside [`thick::membrane_law`] and the rest because it is one of
/// them: a modulus, on the `D` side, with nothing of the geometry in it.
pub(crate) fn drilling_law(e: f64, nu: f64, h: f64) -> f64 {
    DRILLING_WEIGHT * e / (2.0 * (1.0 + nu)) * h
}

/// What a formulation settles **once per cell** before it can write a bending
/// row of `B`.
///
/// Nothing at all for [Reissner-Mindlin](thick), whose fibre rotation is an
/// interpolated field and whose curvature is therefore its plain gradient; the
/// mid-side elimination for [discrete Kirchhoff](kirchhoff), which is where that
/// formulation's whole content lives.
// `Direct` carries nothing and `Discrete` a kilobyte and a half of elimination.
// Boxing would even the two out, at the price of one heap allocation **per
// cell**, in a loop where everything else — the 24×24 element matrix included —
// lives on the stack. The waste is a stack offset; the cure would be a malloc.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BendingSetup {
    /// The rotation is a field of its own: nothing to eliminate first.
    Direct,
    /// The discrete-Kirchhoff elimination, and the in-plane geometry it was
    /// built from.
    Discrete(kirchhoff::Setup),
}

/// The per-cell setup a formulation needs, settled from the geometry alone.
pub(crate) fn bending_setup(
    model: ShellModel,
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
) -> Result<BendingSetup> {
    Ok(match model {
        ShellModel::Thick => BendingSetup::Direct,
        ShellModel::Kirchhoff => BendingSetup::Discrete(kirchhoff::Setup::new(geom, frame)?),
    })
}

/// The cell's degrees of freedom in the element's **local** frame — the columns
/// `B` is written on.
///
/// `dofs` is a node-major gather of the six global components; `out` is the
/// `6n` local vector, each node's translation and rotation triple rotated by
/// the triad. The exact inverse of the `Tᵀ` an internal force comes back
/// through.
pub(crate) fn local_dofs(n_nodes: usize, frame: &[[f64; 3]; 3], dofs: &[f64], out: &mut [f64]) {
    for i in 0..n_nodes {
        for triple in 0..2 {
            let o = 6 * i + 3 * triple;
            for k in 0..3 {
                out[o + k] = (0..3).map(|c| frame[k][c] * dofs[o + c]).sum();
            }
        }
    }
}

/// The rows of `B` a facet integrates at **full** quadrature — membrane,
/// bending and drilling — in the element's local frame, at Gauss point `g`.
///
/// Every entry of the `7 × 6n` block is written, so a caller's buffer needs no
/// clearing between points.
///
/// This is the one place a shell says how its degrees of freedom become strains.
/// The stiffness integrates `Bᵀ D B` with it and the internal forces `Bᵀ σ`,
/// which is what keeps a residual and a tangent describing the same element —
/// and what the two formulations share, differing only in the bending rows.
pub(crate) fn b_into(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    setup: &BendingSetup,
    g: usize,
    b: &mut ShellB,
) -> Result<()> {
    let n = geom.n_nodes;
    let side = 6 * n;
    for row in b[..SHEAR_ROW].iter_mut() {
        row[..side].fill(0.0);
    }
    let mut dn: ShellNodes2 = [[0.0; 2]; 4];
    local_derivatives_into(geom, frame, g, &mut dn)?;
    let shape = geom.n_at_g(g);
    for i in 0..n {
        let (dx, dy) = (dn[i][0], dn[i][1]);
        let (u, v, tz) = (6 * i, 6 * i + 1, 6 * i + 5);
        // Membrane `ε` on the in-plane translations.
        b[MEMBRANE_ROW][u] = dx;
        b[MEMBRANE_ROW + 1][v] = dy;
        b[MEMBRANE_ROW + 2][u] = dy;
        b[MEMBRANE_ROW + 2][v] = dx;
        // The drilling residual `θ_z − ω_z`, with `ω_z = ½(∂v/∂x − ∂u/∂y)`, so
        // the residual picks up its negative.
        b[DRILL_ROW][u] = 0.5 * dy;
        b[DRILL_ROW][v] = -0.5 * dx;
        b[DRILL_ROW][tz] = shape[i];
    }
    match setup {
        BendingSetup::Direct => {
            // `κ` on the independent fibre rotation — local DOFs 6i+3, 6i+4.
            for i in 0..n {
                let (dx, dy) = (dn[i][0], dn[i][1]);
                let (tx, ty) = (6 * i + 3, 6 * i + 4);
                b[BENDING_ROW][ty] = dx;
                b[BENDING_ROW + 1][tx] = -dy;
                b[BENDING_ROW + 2][ty] = dy;
                b[BENDING_ROW + 2][tx] = -dx;
            }
        }
        BendingSetup::Discrete(dk) => dk.bending_into(geom, g, &mut b[BENDING_ROW..DRILL_ROW])?,
    }
    Ok(())
}

/// The transverse-shear rows of `B`, at Gauss point `g` of the **reduced**
/// geometry — the point the stiffness integrates them at, and the point a
/// deformation operator must sample them at for the two to agree.
pub(crate) fn shear_b_into(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    g: usize,
    b: &mut ShellB,
) -> Result<()> {
    let n = geom.n_nodes;
    let side = 6 * n;
    for row in b[SHEAR_ROW..].iter_mut() {
        row[..side].fill(0.0);
    }
    let mut dn: ShellNodes2 = [[0.0; 2]; 4];
    local_derivatives_into(geom, frame, g, &mut dn)?;
    let shape = geom.n_at_g(g);
    for i in 0..n {
        let (dx, dy) = (dn[i][0], dn[i][1]);
        let (wz, tx, ty) = (6 * i + 2, 6 * i + 3, 6 * i + 4);
        // `γ` on the deflection and the two fibre rotations.
        b[SHEAR_ROW][wz] = dx;
        b[SHEAR_ROW][ty] = shape[i];
        b[SHEAR_ROW + 1][wz] = dy;
        b[SHEAR_ROW + 1][tx] = -shape[i];
    }
    Ok(())
}

/// `local += Σ_r modulus · b_rᵀ b_r · w` — the law of a strain whose modulus is
/// a single number: the drilling constraint (one row), and the transverse shear
/// (two rows sharing one).
pub(crate) fn accumulate_scalar(
    local: &mut ShellMatrix,
    b: &[[f64; MAX_SHELL_DOFS]],
    modulus: f64,
    w: f64,
    side: usize,
) {
    let k = modulus * w;
    for row in b {
        for a in 0..side {
            if row[a] == 0.0 {
                continue;
            }
            for col in 0..side {
                local[a * MAX_SHELL_DOFS + col] += k * row[a] * row[col];
            }
        }
    }
}

/// `local += Bᵀ D B · w` for a 3-component strain.
///
pub(crate) fn accumulate(
    local: &mut ShellMatrix,
    b: &[[f64; MAX_SHELL_DOFS]],
    d: &[[f64; 3]; 3],
    w: f64,
    side: usize,
) {
    // `D B` first: three rows, so the inner loop stays short and the intermediate
    // is what a reader can check against the law above. It lives on the stack —
    // this runs at every Gauss point of every cell.
    let mut db = [[0.0_f64; MAX_SHELL_DOFS]; 3];
    for r in 0..3 {
        for c in 0..side {
            db[r][c] = (0..3).map(|k| d[r][k] * b[k][c]).sum();
        }
    }
    for a in 0..side {
        let contributes = (0..3).any(|k| b[k][a] != 0.0);
        if !contributes {
            continue;
        }
        for bcol in 0..side {
            let acc: f64 = (0..3).map(|k| b[k][a] * db[k][bcol]).sum();
            local[a * MAX_SHELL_DOFS + bcol] += acc * w;
        }
    }
}

/// Carry a local element matrix to the global axes: `ke += Tᵀ K_loc T`, with `T`
/// block-diagonal, the local triad on each translation and rotation triplet.
///
pub(crate) fn to_global(
    local: &ShellMatrix,
    frame: &[[f64; 3]; 3],
    n_nodes: usize,
    ke: &mut [f64],
) {
    let side = 6 * n_nodes;
    // `T` maps global DOFs to local: its rows are the local axes.
    let t = |row: usize, col: usize| -> f64 {
        let (br, bc) = (row / 3, col / 3);
        if br != bc {
            return 0.0;
        }
        frame[row % 3][col % 3]
    };
    let mut lt: ShellMatrix = [0.0; MAX_SHELL_DOFS * MAX_SHELL_DOFS];
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                let tkj = t(k, j);
                if tkj != 0.0 {
                    acc += local[i * MAX_SHELL_DOFS + k] * tkj;
                }
            }
            lt[i * MAX_SHELL_DOFS + j] = acc;
        }
    }
    for i in 0..side {
        for j in 0..side {
            let mut acc = 0.0;
            for k in 0..side {
                let tki = t(k, i);
                if tki != 0.0 {
                    acc += tki * lt[k * MAX_SHELL_DOFS + j];
                }
            }
            ke[i * side + j] += acc;
        }
    }
}

crate::physics_operator! {
    /// Shell `Model` spanning **every** subspace of a *surface* `fes`.
    /// Parent-level operator; material is supplied at assembly time.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::shell::ShellModel;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(3).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let m = model::shell(&fes, ShellModel::Thick)?;
    /// assert_eq!(m.len(), fes.len());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn shell(fes, model: ShellModel) via SubModel::shell;
    python: "`model.shell(fespace, model)` — a **shell**: a surface carrying membrane\nforces and bending moments, on a TRI3/QUA4 mesh in 3-D. Material `E`,\n`nu`, `h` (thickness), plus an optional `rho`.\n\n| `model` | transverse shear | when |\n|---|---|---|\n| `\"thick\"` | yes, integrated **reduced** | the general case |\n| `\"kirchhoff\"` | imposed zero at discrete points | thin shells |\n\nSix DOFs per node (`u_x…u_z, r_x…r_z`), as for `frame3d`, so a shell and\na space frame share nodes directly. The sixth — the **drilling** rotation\nabout the normal — is tied to the membrane's own in-plane rotation, which\nremoves the singularity a flat facet would otherwise have without\nresisting a rigid rotation of that facet.\n\n`\"thick\"` (Reissner-Mindlin) integrates the transverse shear at\n**reduced** quadrature: at full quadrature it would overwhelm the bending\nterm by `1/h²` as the shell thins and the element would refuse to bend at\nall (shear locking). It takes an optional `k_s`, the shear-correction\nfactor (`5/6` by default).\n\n`\"kirchhoff\"` (DKT on a triangle, DKQ on a quadrangle) has no transverse\nshear at all: `γ = 0` is imposed at the corners and along each side, so\nthe thin limit is exact by construction and there is nothing left to\nlock. It reports six generalised forces rather than eight — a thin plate\nhas no constitutive `Q`, only a reaction."
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::models::kernel::reduce_cells;

    /// Le triangle unité dans le plan `z = 0` — la facette la plus simple.
    fn facette() -> Handle<SubFiniteElementSpace> {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))
            .unwrap()
            .get(0)
            .unwrap()
    }

    /// `local += Bᵀ D B · w` : avec `B = I` sur trois colonnes et `D = I`, on
    /// retrouve `w` sur la diagonale.
    #[test]
    fn accumulate_is_b_transpose_d_b() {
        let mut local: ShellMatrix = [0.0; MAX_SHELL_DOFS * MAX_SHELL_DOFS];
        let mut b = [[0.0_f64; MAX_SHELL_DOFS]; 3];
        for (k, row) in b.iter_mut().enumerate() {
            row[k] = 1.0;
        }
        let d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        accumulate(&mut local, &b, &d, 2.0, 3);
        assert_eq!(local[MAX_SHELL_DOFS + 1], 2.0);
        assert_eq!(local[1], 0.0);
    }

    /// `ke += Tᵀ K_loc T` : un repère identité recopie la matrice locale.
    #[test]
    fn to_global_on_the_identity_frame_copies() {
        let identite = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut local: ShellMatrix = [0.0; MAX_SHELL_DOFS * MAX_SHELL_DOFS];
        local[0] = 5.0;
        let mut ke = vec![0.0; 36];
        to_global(&local, &identite, 1, &mut ke);
        assert_eq!(ke[0], 5.0);
    }

    /// Le gradient tangentiel est **déjà** dans le plan : le projeter sur
    /// `e₁`, `e₂` suffit — ni inverse, ni second jacobien. Et la partition de
    /// l'unité dérivée fait que les gradients somment à zéro.
    #[test]
    fn local_derivatives_sum_to_zero() {
        reduce_cells(&facette(), |geom| {
            let f = local_frame(geom)?;
            let mut d: ShellNodes2 = [[0.0; 2]; 4];
            local_derivatives_into(geom, &f, 0, &mut d)?;
            assert!((d[0][0] + d[1][0] + d[2][0]).abs() < 1e-12);
            Ok(0.0)
        })
        .unwrap();
    }

    /// Le premier nœud est l'origine, la première arête l'axe `x` : une facette
    /// plane tient en deux nombres par nœud.
    #[test]
    fn local_coords_put_the_first_edge_on_x() {
        reduce_cells(&facette(), |geom| {
            let f = local_frame(geom)?;
            let mut p: ShellNodes2 = [[0.0; 2]; 4];
            local_coords_into(geom, &f, &mut p);
            assert_eq!(p[0], [0.0, 0.0]);
            assert!((p[1][0] - 1.0).abs() < 1e-12 && p[1][1].abs() < 1e-12);
            Ok(0.0)
        })
        .unwrap();
    }

    /// Chaque bloc de lignes de `B` lit **ses** degrés de liberté : la membrane
    /// les translations dans le plan, la flexion les rotations de fibre, le
    /// vrillage la rotation autour de la normale. La flèche hors plan
    /// n'apparaît dans aucun des trois — elle est au cisaillement, intégré
    /// ailleurs.
    #[test]
    fn each_block_of_b_reads_its_own_degrees_of_freedom() {
        reduce_cells(&facette(), |geom| {
            let f = local_frame(geom)?;
            let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];
            b_into(geom, &f, &BendingSetup::Direct, 0, &mut b)?;
            // Membrane : `u` du premier nœud, jamais sa flèche `w`.
            assert!(b[MEMBRANE_ROW][0] != 0.0);
            assert_eq!(b[MEMBRANE_ROW][2], 0.0);
            // Flexion : les rotations de fibre, jamais les translations.
            assert!(b[BENDING_ROW + 2][3] != 0.0 || b[BENDING_ROW + 2][4] != 0.0);
            assert_eq!(b[BENDING_ROW][0], 0.0);
            // Vrillage : `θ_z` vaut la fonction de forme, et une rotation
            // **d'ensemble** autour de la normale ne coûte rien — c'est ce
            // qu'une pénalité diagonale aurait manqué.
            assert!(b[DRILL_ROW][5] != 0.0);
            let rigide: f64 = (0..geom.n_nodes).map(|i| b[DRILL_ROW][6 * i + 5]).sum();
            assert!((rigide - 1.0).abs() < 1e-12); // partition de l'unité
            Ok(0.0)
        })
        .unwrap();
    }

    /// Les lignes de cisaillement, elles, lisent la flèche — et c'est le point
    /// réduit qui les porte.
    #[test]
    fn the_shear_rows_read_the_deflection() {
        reduce_cells(&facette(), |geom| {
            let f = local_frame(geom)?;
            let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];
            shear_b_into(geom, &f, 0, &mut b)?;
            assert!(b[SHEAR_ROW][2] != 0.0 || b[SHEAR_ROW + 1][2] != 0.0);
            Ok(0.0)
        })
        .unwrap();
    }
}
