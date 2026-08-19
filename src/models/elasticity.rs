//! Linear (small-strain) elasticity — `K = ∫ Bᵀ D B dΩ`.
//!
//! Works in 2-D (TRI3 / QUA4) and 3-D (TET4 / HEX8). 2-D supports **plane
//! stress**, **plane strain** and **axisymmetric**; 3-D is the full solid.
//! Voigt convention, with **engineering** shear `γ = 2ε` and stress in the
//! matching order:
//!
//! | model | Voigt vector |
//! |---|---|
//! | plane stress / plane strain | `[εxx, εyy, γxy]` |
//! | axisymmetric | `[εrr, εzz, εθθ, γrz]`, named `[εxx, εyy, εzz, γxy]` |
//! | solid | `[εxx, εyy, εzz, γyz, γxz, γxy]` |
//!
//! The axisymmetric naming follows Cast3M: `x = r`, `y = z` (axis of
//! revolution) and the **`zz` component is the hoop** `θθ`, whose strain is
//! `ε_θθ = u_r / r`. It requires an axisymmetric geometry
//! ([`Coords::axisymmetric`](crate::coords::Coords::axisymmetric)),
//! which is also what puts the `2πr` in the integration measure.
//!
//! Primal `u_x, u_y(, u_z)` (displacement), dual `f_x, …` (nodal force).
//! Material components `E` (Young) and `nu` (Poisson).

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::owned_components;
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by **isotropic** linear elasticity.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu"];
/// Orthotropic constants plus the in-plane material axis (2-D).
const ORTHOTROPIC_2D: &[&str] = &[
    "E_1", "E_2", "E_3", "nu_12", "nu_13", "nu_23", "G_12", "G_13", "G_23", "V1X", "V1Y",
];
/// Orthotropic constants plus the two material axes (3-D).
const ORTHOTROPIC_3D: &[&str] = &[
    "E_1", "E_2", "E_3", "nu_12", "nu_13", "nu_23", "G_12", "G_13", "G_23", "V1X", "V1Y", "V1Z",
    "V2X", "V2Y", "V2Z",
];
/// The 21 anisotropic constants plus the in-plane material axis (2-D).
const ANISOTROPIC_2D: &[&str] = &[
    "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_22", "C_23", "C_24", "C_25", "C_26", "C_33",
    "C_34", "C_35", "C_36", "C_44", "C_45", "C_46", "C_55", "C_56", "C_66", "V1X", "V1Y",
];
/// The 21 anisotropic constants plus the two material axes (3-D).
const ANISOTROPIC_3D: &[&str] = &[
    "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_22", "C_23", "C_24", "C_25", "C_26", "C_33",
    "C_34", "C_35", "C_36", "C_44", "C_45", "C_46", "C_55", "C_56", "C_66", "V1X", "V1Y", "V1Z",
    "V2X", "V2Y", "V2Z",
];

/// The material contract of a symmetry in a space of dimension `space_dim`:
/// the constants of the law, followed by the frame components it needs. Because
/// the assembler resolves a material zone by its **required component set**
/// ([`crate::ops::matrix::assemble_kind`]), these disjoint contracts let an
/// isotropic and an orthotropic zone live on one mesh without any consolidation.
fn material_contract(symmetry: MaterialSymmetry, space_dim: usize) -> &'static [&'static str] {
    match (symmetry, space_dim) {
        (MaterialSymmetry::Isotropic, _) => MATERIAL_COMPONENTS,
        (MaterialSymmetry::Orthotropic, 2) => ORTHOTROPIC_2D,
        (MaterialSymmetry::Orthotropic, _) => ORTHOTROPIC_3D,
        (MaterialSymmetry::Anisotropic, 2) => ANISOTROPIC_2D,
        (MaterialSymmetry::Anisotropic, _) => ANISOTROPIC_3D,
    }
}

/// Which 2-D assumption (or 3-D solid) to use for the constitutive matrix.
///
/// ```
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// // La cinématique choisie décide du nombre de composantes de Voigt.
/// assert_eq!(elasticity::constitutive(210e3, 0.3, ElasticityModel::PlaneStress, 2).len(), 3);
/// assert_eq!(elasticity::constitutive(210e3, 0.3, ElasticityModel::Axisymmetric, 2).len(), 4);
/// assert_eq!(elasticity::constitutive(210e3, 0.3, ElasticityModel::Solid, 3).len(), 6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElasticityModel {
    /// 2-D plane stress (thin plate loaded in its plane).
    PlaneStress,
    /// 2-D plane strain (long prismatic body, `εzz = 0`).
    PlaneStrain,
    /// 2-D meridian plane of a body of revolution: four Voigt components, the
    /// hoop strain `ε_θθ = u_r / r` among them. Requires an axisymmetric
    /// geometry.
    Axisymmetric,
    /// Full 3-D solid.
    Solid,
}

impl ElasticityModel {
    /// Parse from a lowercase tag (`"plane_stress"`, `"plane_strain"`,
    /// `"axisymmetric"`, `"solid"`).
    ///
    /// ```
    /// # use pyrucast::models::elasticity::{self, ElasticityModel};
    /// assert_eq!(ElasticityModel::from_tag("plane_stress"),
    ///            Some(ElasticityModel::PlaneStress));
    /// assert_eq!(ElasticityModel::from_tag("coque"), None);
    /// ```
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "plane_stress" => Some(Self::PlaneStress),
            "plane_strain" => Some(Self::PlaneStrain),
            "axisymmetric" => Some(Self::Axisymmetric),
            "solid" => Some(Self::Solid),
            _ => None,
        }
    }

    /// Whether this model carries the hoop (θθ) component — i.e. is
    /// [`Axisymmetric`](Self::Axisymmetric).
    ///
    /// ```
    /// # use pyrucast::models::elasticity::{self, ElasticityModel};
    /// // Seul le plan méridien d'un corps de révolution porte la déformation
    /// // orthoradiale ε_θθ = u_r / r.
    /// assert!(ElasticityModel::Axisymmetric.is_axisymmetric());
    /// assert!(!ElasticityModel::PlaneStrain.is_axisymmetric());
    /// ```
    pub fn is_axisymmetric(self) -> bool {
        self == Self::Axisymmetric
    }
}

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}
/// Voigt component count: 3 in 2-D plane, **4** axisymmetric (the hoop joins
/// them), 6 in 3-D.
fn voigt_size(space_dim: usize, model: ElasticityModel) -> usize {
    match (space_dim, model) {
        (2, ElasticityModel::Axisymmetric) => 4,
        (2, _) => 3,
        _ => 6,
    }
}
/// Stress component names in Voigt order. Axisymmetric names the hoop `θθ`
/// component `sigma_zz`, after Cast3M (`x = r`, `y = z`).
fn stress_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    match (space_dim, model) {
        (2, ElasticityModel::Axisymmetric) => vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_xy".into(),
        ],
        (2, _) => vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
        _ => vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ],
    }
}

/// Reject an FE subspace whose elements are a **manifold** in their space
/// (`ref_dim < space_dim`) for a continuum-mechanics physics named `label`.
///
/// The continuum kernels build `B` from `∂N_i/∂x_a`, which on a manifold is the
/// *tangent* gradient: the resulting `Bᵀ D B` would be rank-deficient in the
/// normal direction and silently meaningless. A boundary sub-mesh (`SEG2` in
/// 2-D, `TRI3` in 3-D) is a support for loads
/// ([`flux`](fn@crate::ops::node_field::flux)) or convection, not a solid — and a
/// structural element (bar, beam) is a different physics with its own kernel.
/// Shared by [`Elasticity`], [`Plasticity`](crate::models::plasticity) and
/// [`Mazars`](crate::models::damage).
pub(crate) fn check_continuum_dimensions(
    label: &str,
    space_dim: usize,
    ref_dim: usize,
) -> Result<()> {
    if ref_dim != space_dim {
        return Err(PyrucastError::Message(format!(
            "{label}: a {ref_dim}-D element in a {space_dim}-D space is a manifold, not a \
             solid — a boundary mesh carries loads (flux, convection), and a bar or beam \
             is a structural physics of its own (truss, frame, timoshenko)"
        )));
    }
    Ok(())
}

/// Linear-elasticity physics on an FE subspace.
///
/// Material data is supplied at assembly time via
/// [`crate::ops::matrix::stiffness`], not stored here — `E`, `nu` for the
/// isotropic default, the orthotropic or anisotropic constants plus the material
/// axes otherwise (see [`crate::models::symmetry`]).
///
/// Two orthogonal axes: `model` is the **kinematic** hypothesis (plane stress,
/// plane strain, axisymmetric, solid) and `symmetry` is the **material** one.
/// They combine freely — an orthotropic axisymmetric body is as ordinary as an
/// isotropic plane one.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::elasticity::{Elasticity, ElasticityModel};
/// let e = Elasticity::new(zone.clone(), ElasticityModel::PlaneStress)?;
/// assert_eq!(e.material_components(), Some(vec!["E".to_string(), "nu".to_string()]));
/// // La dilatation thermique est **facultative** : sans `alpha`, le modèle
/// // s'assemble sans elle.
/// assert!(e.optional_material_components().contains(&"alpha"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Elasticity {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
    pub(crate) symmetry: MaterialSymmetry,
}

impl Elasticity {
    /// **Isotropic** linear elasticity on an FE subspace, with the given
    /// 2-D/3-D model. Errors if `model` is inconsistent with the space dimension.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::elasticity::{Elasticity, ElasticityModel};
    /// let e = Elasticity::new(zone.clone(), ElasticityModel::PlaneStress)?;
    /// assert_eq!(e.material_components(), Some(vec!["E".to_string(), "nu".to_string()]));
    /// // La dilatation thermique est **facultative** : sans `alpha`, le modèle
    /// // s'assemble sans elle.
    /// assert!(e.optional_material_components().contains(&"alpha"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        Self::with_symmetry(fespace, model, MaterialSymmetry::Isotropic)
    }

    /// Linear elasticity with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::elasticity::{Elasticity, ElasticityModel};
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// // Le constructeur général : une symétrie orthotrope élargit le contrat
    /// // matériau, qui porte alors les modules **et** les axes.
    /// let o = Elasticity::with_symmetry(
    ///     zone.clone(), ElasticityModel::PlaneStress, MaterialSymmetry::Orthotropic)?;
    /// assert!(o.material_components().unwrap().len() > 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        let (submesh, space_dim, ref_dim, axisymmetric) = {
            let s = fespace.read();
            (
                s.submesh(),
                s.space_dim(),
                s.ref_dim()?,
                s.is_axisymmetric(),
            )
        };
        check_continuum_dimensions("Elasticity", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (2, ElasticityModel::Axisymmetric) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Elasticity: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ solid)"
            )));
        }
        // The model and the geometry must agree **both ways**: the 2πr measure
        // comes from the Coords while the hoop row comes from the model, so a
        // mismatch would silently mix a plane constitutive law with a revolved
        // measure (or the reverse) and quietly produce wrong results.
        if axisymmetric != model.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Elasticity: model {model:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` model (its integrals already \
                     carry the 2πr factor)"
                )
            } else {
                "Elasticity: the `axisymmetric` model requires an axisymmetric geometry \
                 (build the Coords with Coords::axisymmetric)"
                    .into()
            }));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
            symmetry,
        })
    }
}

impl SubModelKind for Elasticity {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
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

    /// The consistent mass matrix shares the stiffness layout (same fespace,
    /// support, DOF numbering) — only the kernel differs.
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    /// The geometric (initial-stress) stiffness shares the stiffness layout.
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("geometric stiffness requires the current stress field");
        element_geometric(geom, stress, ke)
    }

    /// Linear elasticity: the consistent tangent **is** the elastic stiffness.
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        _state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        element_stiffness(geom, mat, self.model, self.symmetry, ke)
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        element_stiffness(geom, mat, self.model, self.symmetry, ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        element_mass(geom, mat, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Elasticity"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Elasticity({:?}, {})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model, self.symmetry
        )
    }
}

impl Domain for Elasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(material_contract(
            self.symmetry,
            self.space_dim,
        )))
    }

    /// `alpha` (thermal-expansion coefficient) — accepted through the material
    /// field when doing thermomechanics, never required for a plain elastic
    /// assembly. Consumed by
    /// [`crate::ops::element_field::thermal_strain`](fn@crate::ops::element_field::thermal_strain).
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha", "rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(stress_names(self.space_dim, self.model))
    }

    /// Linear stress σ = D·ε at one Gauss point (material constants per cell).
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
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let dmat = symmetry::elastic_constitutive(mat, cell, self.symmetry, self.model, d)?;
        let strain = voigt_strain(&|name| input.value(cell, g, name), d, self.model)?;
        for (r, drow) in dmat.iter().enumerate() {
            out[r] = drow.iter().zip(&strain).map(|(dv, s)| dv * s).sum();
        }
        Ok(())
    }
}

/// Isotropic constitutive (Voigt) matrix `D` from `E`, `nu` and the model.
///
/// ```
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// // Contraintes planes : σ_zz = 0, la souplesse hors plan est condensée.
/// // Déformations planes : ε_zz = 0, le matériau est plus **raide**.
/// let cp = elasticity::constitutive(210e3, 0.3, ElasticityModel::PlaneStress, 2);
/// let dp = elasticity::constitutive(210e3, 0.3, ElasticityModel::PlaneStrain, 2);
/// assert!(dp[0][0] > cp[0][0]);
/// // En contraintes planes, D₀₀ = E/(1−ν²).
/// assert!((cp[0][0] - 210e3 / (1.0 - 0.09)).abs() < 1e-6);
/// // Le bloc de cisaillement vaut μ dans les deux cas (Voigt de l'ingénieur).
/// assert!((cp[2][2] - dp[2][2]).abs() < 1e-6);
/// ```
pub fn constitutive(e: f64, nu: f64, model: ElasticityModel, space_dim: usize) -> Vec<Vec<f64>> {
    match (space_dim, model) {
        (2, ElasticityModel::PlaneStress) => {
            let c = e / (1.0 - nu * nu);
            vec![
                vec![c, c * nu, 0.0],
                vec![c * nu, c, 0.0],
                vec![0.0, 0.0, c * (1.0 - nu) / 2.0],
            ]
        }
        (2, ElasticityModel::PlaneStrain) => {
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            vec![
                vec![c * (1.0 - nu), c * nu, 0.0],
                vec![c * nu, c * (1.0 - nu), 0.0],
                vec![0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
            ]
        }
        (2, ElasticityModel::Axisymmetric) => {
            // Voigt order [rr, zz, θθ, rz]: the three normal directions are
            // mutually orthogonal, so the 3×3 normal block is the isotropic one
            // (as in plane strain, with θθ restored) and `rz` is the lone shear.
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let (d_n, d_off) = (c * (1.0 - nu), c * nu);
            vec![
                vec![d_n, d_off, d_off, 0.0],
                vec![d_off, d_n, d_off, 0.0],
                vec![d_off, d_off, d_n, 0.0],
                vec![0.0, 0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
            ]
        }
        _ => {
            // 3-D solid (Voigt order [xx, yy, zz, yz, xz, xy]).
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let g = c * (1.0 - 2.0 * nu) / 2.0;
            let mut d = vec![vec![0.0; 6]; 6];
            for i in 0..3 {
                for j in 0..3 {
                    d[i][j] = if i == j { c * (1.0 - nu) } else { c * nu };
                }
            }
            d[3][3] = g;
            d[4][4] = g;
            d[5][5] = g;
            d
        }
    }
}

/// Voigt **engineering** strain from the tensor strain components produced by
/// [`crate::ops::element_field::deformation`] (`eps_xx`, `eps_xy`, …), reading each
/// component by name through `eps`. Off-diagonals become `γ = 2ε`.
fn voigt_strain(
    eps: &dyn Fn(&str) -> Result<f64>,
    space_dim: usize,
    model: ElasticityModel,
) -> Result<Vec<f64>> {
    if space_dim == 2 && model.is_axisymmetric() {
        // [εrr, εzz, εθθ, γrz] — the hoop `eps_zz` is produced by
        // `ops::element_field::deformation` on an axisymmetric space.
        Ok(vec![
            eps("eps_xx")?,
            eps("eps_yy")?,
            eps("eps_zz")?,
            2.0 * eps("eps_xy")?,
        ])
    } else if space_dim == 2 {
        Ok(vec![eps("eps_xx")?, eps("eps_yy")?, 2.0 * eps("eps_xy")?])
    } else {
        Ok(vec![
            eps("eps_xx")?,
            eps("eps_yy")?,
            eps("eps_zz")?,
            2.0 * eps("eps_yz")?,
            2.0 * eps("eps_xz")?,
            2.0 * eps("eps_xy")?,
        ])
    }
}

/// Strain-displacement matrix `B` (Voigt) from `∂N_i/∂x_a` (`dn_dx`, layout
/// `[i*space_dim + a]`). Shape `voigt_size × (space_dim·nodes)`, node-major
/// columns (matching [`DofOrdering::NodesThenVars`]).
///
/// `hoop` carries the axisymmetric extra: `Some((N, r))` — the shape values and
/// the radius at the Gauss point — adds the fourth row `ε_θθ = Σ_i N_i u_{r,i} / r`
/// and orders the rows `[rr, zz, θθ, rz]`. `None` gives the plane / solid `B`.
fn b_matrix(
    dn_dx: &[f64],
    n_nodes: usize,
    space_dim: usize,
    hoop: Option<(&[f64], f64)>,
) -> Vec<Vec<f64>> {
    let v = match hoop {
        Some(_) => 4,
        None => voigt_size(space_dim, ElasticityModel::PlaneStrain),
    };
    let dofs = space_dim * n_nodes;
    let mut b = vec![vec![0.0; dofs]; v];
    let dn = |i: usize, a: usize| dn_dx[i * space_dim + a];
    for i in 0..n_nodes {
        if let Some((n, r)) = hoop {
            let (cr, cz) = (2 * i, 2 * i + 1);
            b[0][cr] = dn(i, 0); // εrr
            b[1][cz] = dn(i, 1); // εzz
            b[2][cr] = n[i] / r; // εθθ = u_r / r
            b[3][cr] = dn(i, 1); // γrz
            b[3][cz] = dn(i, 0);
        } else if space_dim == 2 {
            let (cx, cy) = (2 * i, 2 * i + 1);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cx] = dn(i, 1); // γxy
            b[2][cy] = dn(i, 0);
        } else {
            let (cx, cy, cz) = (3 * i, 3 * i + 1, 3 * i + 2);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cz] = dn(i, 2); // εzz
            b[3][cy] = dn(i, 2); // γyz
            b[3][cz] = dn(i, 1);
            b[4][cx] = dn(i, 2); // γxz
            b[4][cz] = dn(i, 0);
            b[5][cx] = dn(i, 1); // γxy
            b[5][cy] = dn(i, 0);
        }
    }
    b
}

/// Element kernel: local stiffness `K_e = Σ_g (Bᵀ D B) |J| w` of one cell,
/// written into `ke` (flat row-major, side `space_dim·n_nodes`, **node-major /
/// component-minor** dof order `dof = node·space_dim + component`). Pure and
/// sequential — driven in parallel by [`crate::models::kernel::assemble_block`].
/// Reused as-is by [`crate::models::plasticity`] and [`crate::models::damage`]
/// (their iteration operator is the elastic stiffness).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
/// #                vec!["u_x".to_string(), "u_y".to_string()]);
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?);
/// // Le noyau d'élément, tel que le pilote l'appelle. La matrice de
/// // raideur d'un solide libre est **singulière** : ses lignes somment à
/// // zéro, un mouvement de corps rigide n'engendrant aucune force.
/// let (duals, primals) = vars();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, Some(&mat), None,
///     |geoms, m, _s, ke| elasticity::element_stiffness(
///         &geoms[0], m.unwrap(), ElasticityModel::PlaneStress,
///         MaterialSymmetry::Isotropic, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!(total.abs() < 1e-6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    model: ElasticityModel,
    symmetry: MaterialSymmetry,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    let dofs = space_dim * n_nodes;
    // Constants read at Gauss 0 — constant material per cell.
    let d = symmetry::elastic_constitutive(material, geom.cell, symmetry, model, space_dim)?;
    let v = d.len();
    for g in 0..geom.n_gauss {
        // On a body of revolution the hoop row needs `N` and `r` at this point.
        let hoop = if model.is_axisymmetric() {
            Some((geom.n_at_g(g)?, geom.radius(g)?))
        } else {
            None
        };
        let b = b_matrix(&geom.dn_dx(g)?, n_nodes, space_dim, hoop);
        // DB = D·B  (voigt × dofs).
        let mut db = vec![vec![0.0; dofs]; v];
        for r in 0..v {
            for c in 0..dofs {
                let mut acc = 0.0;
                for w in 0..v {
                    acc += d[r][w] * b[w][c];
                }
                db[r][c] = acc;
            }
        }
        let w = geom.det_j_w(g)?;
        for r in 0..dofs {
            for c in 0..dofs {
                let mut acc = 0.0;
                for vv in 0..v {
                    acc += b[vv][r] * db[vv][c];
                }
                ke[r * dofs + c] += acc * w;
            }
        }
    }
    Ok(())
}

/// Element kernel: local **consistent mass** `M_e = Σ_g ρ (Nᵀ N) |J| w` of one
/// cell, written into `ke` (same flat row-major, **node-major / component-minor**
/// dof order as [`element_stiffness`]). The vector shape-function matrix is
/// block-diagonal, so `M[(i,a),(j,b)] = δ_ab ρ ∫ N_i N_j`. Density `ρ` is read
/// from the material component `rho` (constant per cell). Pure and sequential,
/// law-independent — reused as-is by [`crate::models::plasticity`] and
/// [`crate::models::damage`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
/// #                vec!["u_x".to_string(), "u_y".to_string()]);
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["rho".into()], &[2.0])?);
/// // La masse totale se retrouve dans la somme des entrées, une fois par
/// // direction : ρ × aire × space_dim.
/// let (duals, primals) = vars();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, Some(&mat), None,
///     |geoms, m, _s, ke| elasticity::element_mass(&geoms[0], m.unwrap(), ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!((total - 2.0 * 0.5 * 2.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_mass(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    let dofs = space_dim * n_nodes;
    let rho = material.value(geom.cell, 0, "rho").map_err(|_| {
        PyrucastError::Message(
            "Elasticity mass matrix: material component `rho` (density) is required".into(),
        )
    })?;
    for g in 0..geom.n_gauss {
        let n = geom.n_at_g(g)?;
        let w = geom.det_j_w(g)? * rho;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let m = n[i] * n[j] * w;
                for a in 0..space_dim {
                    let r = i * space_dim + a;
                    let c = j * space_dim + a;
                    ke[r * dofs + c] += m;
                }
            }
        }
    }
    Ok(())
}

/// Element kernel: local **geometric (initial-stress) stiffness**
///   `Kg[(i,a),(j,b)] = δ_ab Σ_g Σ_cd (∂N_i/∂x_c) σ_cd (∂N_j/∂x_e) |J| w`
/// of one cell, written into `ke` (same flat, node-major / component-minor dof
/// order as [`element_stiffness`]). The scalar `∇N_i·σ·∇N_j` is applied to each
/// displacement component's diagonal block (`δ_ab`). The current Cauchy stress
/// `σ` (Voigt-named) is read from `state` per Gauss point. Pure and sequential,
/// law-independent — reused as-is by [`crate::models::plasticity`] and
/// [`crate::models::damage`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
/// #                vec!["u_x".to_string(), "u_y".to_string()]);
/// # let etat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(),
/// #     vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
/// #     &[100.0, 0.0, 0.0])?);
/// // La raideur **géométrique**, celle du flambement : elle vient de l'état
/// // de contrainte, non du matériau. Sous traction elle est définie
/// // positive ; c'est son signe qui décide de la charge critique.
/// let (duals, primals) = vars();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, None, Some(&etat),
///     |geoms, _m, s, ke| elasticity::element_geometric(&geoms[0], s.unwrap(), ke),
/// )?;
/// // Elle est singulière elle aussi : les modes rigides n'y coûtent rien.
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!(total.abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_geometric(geom: &CellGeom, stress: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let d = geom.space_dim;
    let dofs = d * n_nodes;
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?; // [i * d + c]
        let w = geom.det_j_w(g)?;
        let sig = crate::models::voigt_stress_matrix(stress, geom.cell, g, d)?; // [c * d + e]
                                                                                // On a body of revolution the hoop strain's own non-linear part,
                                                                                // ½(u_r/r)², contributes `σ_θθ N_i N_j / r²` on the radial diagonal —
                                                                                // the initial-stress counterpart of the `N_i / r` row of `B`.
        let hoop = if geom.axisymmetric {
            let r = geom.radius(g)?;
            Some((
                geom.n_at_g(g)?,
                stress.value(geom.cell, g, "sigma_zz")? / (r * r),
            ))
        } else {
            None
        };
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                // Scalar gᵢⱼ = Σ_{c,e} (∂N_i/∂x_c) σ_ce (∂N_j/∂x_e).
                let mut gij = 0.0;
                for c in 0..d {
                    for e in 0..d {
                        gij += dn[i * d + c] * sig[c * d + e] * dn[j * d + e];
                    }
                }
                gij *= w;
                // Same scalar on every component's diagonal block (δ_ab).
                for a in 0..d {
                    ke[(i * d + a) * dofs + (j * d + a)] += gij;
                }
                if let Some((n, s_hoop)) = hoop {
                    ke[(i * d) * dofs + (j * d)] += s_hoop * n[i] * n[j] * w;
                }
            }
        }
    }
    Ok(())
}

/// Names of the **consistent-tangent** state components a non-linear physics
/// (plasticity, Mazars) emits: the upper triangle of the symmetric `v×v`
/// algorithmic modulus `D_alg` in the model's engineering-Voigt order, named
/// `ktan_{i}_{j}` for `i ≤ j`. `v = 3` in 2-D plane, `4` axisymmetric, `6` in
/// 3-D — so 6, 10 or 21 names.
/// The tangent assembler reads them back with [`read_tangent_matrix`].
///
/// ```
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// // Le triangle supérieur d'un module symétrique v×v : 6, 10 ou 21 noms.
/// assert_eq!(elasticity::tangent_component_names(2, ElasticityModel::PlaneStress).len(), 6);
/// assert_eq!(elasticity::tangent_component_names(2, ElasticityModel::Axisymmetric).len(), 10);
/// assert_eq!(elasticity::tangent_component_names(3, ElasticityModel::Solid).len(), 21);
/// assert_eq!(elasticity::tangent_component_names(2, ElasticityModel::PlaneStress)[0],
///            "ktan_0_0");
/// ```
pub fn tangent_component_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    let v = voigt_size(space_dim, model);
    let mut names = Vec::with_capacity(v * (v + 1) / 2);
    for i in 0..v {
        for j in i..v {
            names.push(format!("ktan_{i}_{j}"));
        }
    }
    names
}

/// Reconstruct the symmetric `v×v` consistent tangent `D_alg` at `(cell, g)` from
/// the `ktan_{i}_{j}` state components emitted by the constitutive integrator.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
/// #                vec!["u_x".to_string(), "u_y".to_string()]);
/// // Le producteur écrit `ktan_i_j`, le consommateur les relit ici : c'est
/// // tout le contrat entre l'intégrateur constitutif et l'assembleur.
/// let noms = elasticity::tangent_component_names(2, ElasticityModel::PlaneStress);
/// let d0 = elasticity::constitutive(210e3, 0.3, ElasticityModel::PlaneStress, 2);
/// let mut etat = SubElementField::new(zone.clone(), noms.clone())?;
/// let mut k = 0;
/// for i in 0..3 {
///     for j in i..3 {
///         etat.set_uniform(&noms[k], d0[i][j])?;
///         k += 1;
///     }
/// }
/// // Relu, le module est **symétrique** et identique à ce qu'on a écrit.
/// let d = elasticity::read_tangent_matrix(&etat, 0, 0, 2, ElasticityModel::PlaneStress)?;
/// assert_eq!(d, d0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn read_tangent_matrix(
    state: &SubElementField,
    cell: usize,
    g: usize,
    space_dim: usize,
    model: ElasticityModel,
) -> Result<Vec<Vec<f64>>> {
    let v = voigt_size(space_dim, model);
    let mut d = vec![vec![0.0; v]; v];
    for i in 0..v {
        for j in i..v {
            let val = state.value(cell, g, &format!("ktan_{i}_{j}"))?;
            d[i][j] = val;
            d[j][i] = val;
        }
    }
    Ok(d)
}

/// Element kernel: local **consistent tangent** `K_t = Σ_g Bᵀ D_alg B |J| w` of
/// one cell, with the per-Gauss algorithmic modulus `D_alg` read from `state`
/// (the constitutive integrator's `ktan_*` output). Same `ke` layout as
/// [`element_stiffness`]; law-independent given `D_alg`, so plasticity and Mazars
/// share it — only the `D_alg` they produce differs.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity::{self, ElasticityModel};
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
/// #                vec!["u_x".to_string(), "u_y".to_string()]);
/// // Indépendante de la loi une fois `D_alg` donné : plasticité et Mazars
/// // partagent ce noyau, seul le module algorithmique qu'elles produisent
/// // diffère. Avec le module **élastique** en entrée, on retrouve la
/// // raideur élastique.
/// # let noms = elasticity::tangent_component_names(2, ElasticityModel::PlaneStress);
/// # let d0 = elasticity::constitutive(210e3, 0.3, ElasticityModel::PlaneStress, 2);
/// # let mut e = SubElementField::new(zone.clone(), noms.clone())?;
/// # let mut k = 0;
/// # for i in 0..3 { for j in i..3 { e.set_uniform(&noms[k], d0[i][j])?; k += 1; } }
/// # let etat = Handle::new(e);
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?);
/// let (duals, primals) = vars();
/// let tangente = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals.clone(), primals.clone(),
///     DofOrdering::NodesThenVars, true, None, Some(&etat),
///     |geoms, _m, s, ke| elasticity::element_tangent_from_state(
///         &geoms[0], s.unwrap(), ElasticityModel::PlaneStress, ke),
/// )?;
/// let raideur = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, Some(&mat), None,
///     |geoms, m, _s, ke| elasticity::element_stiffness(
///         &geoms[0], m.unwrap(), ElasticityModel::PlaneStress,
///         MaterialSymmetry::Isotropic, ke),
/// )?;
/// assert_eq!(tangente.dense(), raideur.dense());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_tangent_from_state(
    geom: &CellGeom,
    state: &SubElementField,
    model: ElasticityModel,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    let dofs = space_dim * n_nodes;
    let v = voigt_size(space_dim, model);
    for g in 0..geom.n_gauss {
        // Same hoop row as `element_stiffness` on a body of revolution.
        let hoop = if model.is_axisymmetric() {
            Some((geom.n_at_g(g)?, geom.radius(g)?))
        } else {
            None
        };
        let b = b_matrix(&geom.dn_dx(g)?, n_nodes, space_dim, hoop);
        let d = read_tangent_matrix(state, geom.cell, g, space_dim, model)?;
        // DB = D·B (voigt × dofs), then Kᵉ += Bᵀ (DB) · |J| w.
        let mut db = vec![vec![0.0; dofs]; v];
        for (r, dbr) in db.iter_mut().enumerate() {
            for (c, dbrc) in dbr.iter_mut().enumerate() {
                let mut acc = 0.0;
                for w in 0..v {
                    acc += d[r][w] * b[w][c];
                }
                *dbrc = acc;
            }
        }
        let w = geom.det_j_w(g)?;
        for r in 0..dofs {
            for c in 0..dofs {
                let mut acc = 0.0;
                for vv in 0..v {
                    acc += b[vv][r] * db[vv][c];
                }
                ke[r * dofs + c] += acc * w;
            }
        }
    }
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn unit_quad(model: ElasticityModel) -> Elasticity {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Elasticity::new(fes.get(0).unwrap(), model).unwrap()
    }

    #[test]
    fn vars_and_model_validation() {
        let el = unit_quad(ElasticityModel::PlaneStress);
        assert_eq!(el.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(el.dual_vars(), vec!["f_x", "f_y"]);
        // 2-D space cannot be Solid.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert!(Elasticity::new(fes.get(0).unwrap(), ElasticityModel::Solid).is_err());
    }

    #[test]
    fn plane_stress_constitutive_known_values() {
        let (e, nu) = (1.0, 0.25);
        let d = constitutive(e, nu, ElasticityModel::PlaneStress, 2);
        let c = e / (1.0 - nu * nu);
        assert!((d[0][0] - c).abs() < 1e-12);
        assert!((d[0][1] - c * nu).abs() < 1e-12);
        assert!((d[2][2] - c * (1.0 - nu) / 2.0).abs() < 1e-12);
        assert!((d[2][2] - e / (2.0 * (1.0 + nu))).abs() < 1e-12); // = G
    }

    /// COMP: uniaxial tensor strain `εxx = ε₀` in plane stress gives
    /// `σxx = E/(1-ν²)·ε₀`, `σyy = ν·σxx`, `σxy = 0`.
    #[test]
    fn integrate_behavior_plane_stress_uniaxial() {
        let (e, nu, eps0) = (210.0, 0.3, 0.001);
        let el = unit_quad(ElasticityModel::PlaneStress);
        let mut mat =
            SubElementField::new(el.fespace.clone(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", e).unwrap();
        mat.set_uniform("nu", nu).unwrap();
        let mat = Handle::new(mat);

        let mut strain = SubElementField::new(
            el.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = Handle::new(strain);

        let out = el
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-9);
            assert!((out.value(0, g, "sigma_yy").unwrap() - c * nu * eps0).abs() < 1e-9);
            assert!(out.value(0, g, "sigma_xy").unwrap().abs() < 1e-9);
        }
    }

    /// Element stiffness is symmetric and the rigid-body modes are in its
    /// kernel (zero row sums per axis).
    #[test]
    fn element_stiffness_symmetric_and_rigid_body_free() {
        let el = unit_quad(ElasticityModel::PlaneStrain);
        let mut mat =
            SubElementField::new(el.fespace.clone(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", 200.0).unwrap();
        mat.set_uniform("nu", 0.3).unwrap();
        let mat = Handle::new(mat);
        let blocks = el.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = el.support.read().connectivity().to_vec();
        let tol = 1e-9;
        // Symmetry K[(i,f_a),(j,u_b)] == K[(j,f_b),(i,u_a)].
        for &ni in &nodes {
            for &nj in &nodes {
                for a in ["x", "y"] {
                    for b in ["x", "y"] {
                        let lhs = k.get(ni, &format!("f_{a}"), nj, &format!("u_{b}"));
                        let rhs = k.get(nj, &format!("f_{b}"), ni, &format!("u_{a}"));
                        assert!((lhs - rhs).abs() < tol);
                    }
                }
            }
        }
        // A uniform translation in x ⇒ zero force everywhere (row sum = 0).
        for &ni in &nodes {
            let row: f64 = nodes.iter().map(|&nj| k.get(ni, "f_x", nj, "u_x")).sum();
            assert!(row.abs() < tol, "row sum {row} ≠ 0");
        }
    }
}
