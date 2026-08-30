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
/// The generalised forces a shell behaviour reports: membrane, bending, and —
/// where the formulation has one — transverse shear.
const BEHAVIOR: [&str; 8] = [
    "N_xx", "N_yy", "N_xy", "M_xx", "M_yy", "M_xy", "Q_xz", "Q_yz",
];
/// The six a formulation without transverse shear reports: `BEHAVIOR` cut short.
const BEHAVIOR_NO_SHEAR: usize = 6;

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
        let mut names = owned_components(&[
            "eps_xx", "eps_yy", "eps_xy", "kappa_xx", "kappa_yy", "kappa_xy",
        ]);
        if self.model.has_transverse_shear() {
            names.extend(owned_components(&["gamma_xz", "gamma_yz"]));
        }
        names
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
        if self.model.has_transverse_shear() {
            // `k_s` overrides the 5/6 of a homogeneous rectangular section; which
            // of the two applies is a fact of the zone, settled in the layout.
            let k_s = match lay.optional_material[1] {
                ABSENT_COMPONENT => 5.0 / 6.0,
                i => material[i as usize],
            };
            let ds = thick::shear_law(e, nu, h, k_s);
            let gamma = [d(6), d(7)];
            for i in 0..2 {
                out[6 + i] = ds * gamma[i];
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
    let p0 = geom.node_coord(0).to_vec();
    let p1 = geom.node_coord(1).to_vec();
    let p2 = geom.node_coord(2).to_vec();
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
/// // Le gradient tangentiel est **déjà** dans le plan : le projeter sur
/// // e₁, e₂ suffit — ni inverse, ni second jacobien.
/// reduce_cells(&zone, |geom| {
///     let f = shell::local_frame(geom)?;
///     let d = shell::local_derivatives(geom, &f, 0)?;
///     assert_eq!(d.len(), 3);
///     // Partition de l'unité dérivée : les gradients somment à zéro.
///     assert!((d[0][0] + d[1][0] + d[2][0]).abs() < 1e-12);
///     Ok(0.0)
/// })?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn local_derivatives(
    geom: &CellGeom,
    frame: &[[f64; 3]; 3],
    g: usize,
) -> Result<Vec<[f64; 2]>> {
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    let dn = &mut dn_buf[..geom.n_nodes * geom.space_dim];
    geom.dn_dx(g, dn)?;
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
/// // Le premier nœud est l'origine, la première arête l'axe x : une
/// // facette plane tient en deux nombres par nœud, ce dont a besoin une
/// // formulation écrite sur la **géométrie** — un Kirchhoff discret.
/// reduce_cells(&zone, |geom| {
///     let f = shell::local_frame(geom)?;
///     let p = shell::local_coords(geom, &f)?;
///     assert_eq!(p[0], [0.0, 0.0]);
///     assert!((p[1][0] - 1.0).abs() < 1e-12 && p[1][1].abs() < 1e-12);
///     Ok(0.0)
/// })?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn local_coords(geom: &CellGeom, frame: &[[f64; 3]; 3]) -> Result<Vec<[f64; 2]>> {
    let origin = geom.node_coord(0).to_vec();
    (0..geom.n_nodes)
        .map(|i| {
            let p = geom.node_coord(i);
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
/// // La part que toute formulation de facette plane partage : elle ne doit
/// // rien à la théorie de flexion. Accumulée dans la matrice locale 6n×6n.
/// reduce_cells(&zone, |geom| {
///     let f = shell::local_frame(geom)?;
///     let mut local = vec![vec![0.0; 18]; 18];
///     shell::membrane_and_drilling(geom, &f, 210_000.0, 0.3, 0.01, &mut local)?;
///     // Les DDL de translation dans le plan sont chargés…
///     assert!(local[0][0] > 0.0);
///     // …et rien n'a touché la flèche hors plan, qui relève de la flexion.
///     assert_eq!(local[2][2], 0.0);
///     Ok(0.0)
/// })?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
        let shape = geom.n_at_g(g);
        let w = geom.det_j_w(g);

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
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// // `local += Bᵀ D B · w`. Avec B = I sur les trois premières colonnes et
/// // D l'identité, on retrouve `w` sur la diagonale.
/// let mut local = vec![vec![0.0; 3]; 3];
/// let b: Vec<Vec<f64>> = (0..3)
///     .map(|k| (0..3).map(|j| if j == k { 1.0 } else { 0.0 }).collect())
///     .collect();
/// let d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// shell::accumulate(&mut local, &b, &d, 2.0, 3);
/// assert_eq!(local[1][1], 2.0);
/// assert_eq!(local[0][1], 0.0);
/// ```
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
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// // `ke += Tᵀ K_loc T`, T bloc-diagonale : la triade locale sur chaque
/// // triplet de translation et de rotation. Repère identité ⇒ recopie.
/// let identite = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let mut local = vec![vec![0.0; 6]; 6];
/// local[0][0] = 5.0;
/// let mut ke = vec![0.0; 36];
/// shell::to_global(&local, &identite, 1, &mut ke);
/// assert_eq!(ke[0], 5.0);
/// ```
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
