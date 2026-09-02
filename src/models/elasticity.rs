//! Linear (small-strain) elasticity — `K = ∫ Bᵀ D B dΩ`.
//!
//! The **law**; the modelling it stands on is
//! [`Continuum`], and the elastic operator
//! it evaluates is [`continuum::elastic`](crate::models::continuum::elastic) —
//! shared with the return-map and damage families, which use it as their elastic
//! predictor.
//!
//! Works in 2-D (TRI3 / QUA4) and 3-D (TET4 / HEX8). 2-D supports **plane
//! stress**, **plane strain** and **axisymmetric**; 3-D is the full solid.
//!
//! Primal `u_x, u_y(, u_z)` (displacement), dual `f_x, …` (nodal force).
//! Material components `E` (Young) and `nu` (Poisson) in the isotropic case.

pub mod law;
mod linear;

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::continuum::material::MatRead;
use crate::models::continuum::{elastic, voigt, Continuum};
use crate::models::symmetry::MaterialSymmetry;
use crate::models::tensor::{dual_name, primal_name, Kinematics};
use crate::models::ZoneLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::models::{ElementLayout, MatrixKind};
use law::{ElasticLaw, StatelessLawKind};
use serde::{Deserialize, Serialize};

/// Linear-elasticity physics on an FE subspace.
///
/// Material data is supplied at assembly time via
/// [`crate::ops::matrix::stiffness`], not stored here — `E`, `nu` for the
/// isotropic default, the orthotropic or anisotropic constants plus the material
/// axes otherwise (see [`crate::models::symmetry`]).
///
/// Two orthogonal axes: the **kinematic** hypothesis (plane stress, plane
/// strain, axisymmetric, solid) belongs to the
/// [`Continuum`] it holds, and `symmetry`
/// is the **material** one. They combine freely — an orthotropic axisymmetric
/// body is as ordinary as an isotropic plane one.
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
/// # use pyrucast::models::elasticity::Elasticity;
/// # use pyrucast::models::tensor::Kinematics;
/// let e = Elasticity::new(zone.clone(), Kinematics::PlaneStress)?;
/// assert_eq!(e.material_components(), vec!["E".to_string(), "nu".to_string()]);
/// // La dilatation thermique est **facultative** : sans `alpha`, le modèle
/// // s'assemble sans elle.
/// assert!(e.optional_material_components().contains(&"alpha"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Elasticity {
    pub(crate) continuum: Continuum,
    pub(crate) symmetry: MaterialSymmetry,
    pub(crate) law: ElasticLaw,
}

impl Elasticity {
    /// **Isotropic** linear elasticity on an FE subspace, with the given
    /// 2-D/3-D kinematics. Errors if `kinematics` is inconsistent with the space dimension.
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
    /// # use pyrucast::models::elasticity::Elasticity;
    /// # use pyrucast::models::tensor::Kinematics;
    /// let e = Elasticity::new(zone.clone(), Kinematics::PlaneStress)?;
    /// assert_eq!(e.material_components(), vec!["E".to_string(), "nu".to_string()]);
    /// // La dilatation thermique est **facultative** : sans `alpha`, le modèle
    /// // s'assemble sans elle.
    /// assert!(e.optional_material_components().contains(&"alpha"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, kinematics: Kinematics) -> Result<Self> {
        Self::with_symmetry(fespace, kinematics, MaterialSymmetry::Isotropic)
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
    /// # use pyrucast::models::elasticity::Elasticity;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// // Le constructeur général : une symétrie orthotrope élargit le contrat
    /// // matériau, qui porte alors les modules **et** les axes.
    /// let o = Elasticity::with_symmetry(
    ///     zone.clone(), Kinematics::PlaneStress, MaterialSymmetry::Orthotropic)?;
    /// assert!(o.material_components().len() > 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        kinematics: Kinematics,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        Ok(Self {
            continuum: Continuum::new(fespace, kinematics, "Elasticity")?,
            symmetry,
            law: ElasticLaw::Linear,
        })
    }
}

impl SubModelKind for Elasticity {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.continuum.space_dim()).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.continuum.space_dim()).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.continuum.fespace()],
            support: self.continuum.support(),
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

    /// Linear elasticity: the consistent tangent **is** the elastic stiffness.
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
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
        let n = self.continuum.support().read().cell_count();
        format!(
            "SubModel<Elasticity({:?}, {})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.continuum.kinematics(),
            self.symmetry
        )
    }
}

impl Domain for Elasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.continuum.fespace()
    }

    fn material_components(&self) -> Vec<String> {
        self.law
            .as_law()
            .material_components(self.symmetry, self.continuum.space_dim())
    }

    /// `alpha` (thermal-expansion coefficient) — accepted through the material
    /// field when doing thermomechanics, never required for a plain elastic
    /// assembly. Consumed by
    /// [`crate::ops::element_field::thermal_strain`](fn@crate::ops::element_field::thermal_strain).
    fn optional_material_components(&self) -> &'static [&'static str] {
        elastic::OPTIONAL_COMPONENTS
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.continuum.fespace()
    }

    /// The stress, plus — for a law whose tangent depends on `ε` — the
    /// algorithmic modulus the tangent assembler will read back. A linear law
    /// emits the stress alone: its tangent **is** its stiffness.
    fn behavior_output_components(&self) -> Vec<String> {
        let mut comps = self.continuum.stress_names();
        if !self.law.as_law().is_linear() {
            comps.extend(self.continuum.tangent_component_names());
        }
        comps
    }

    /// Linear stress σ = D·ε at one Gauss point (material constants per cell).
    fn deformation_reads(&self) -> Vec<String> {
        self.continuum.strain_reads()
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
        let strain = voigt::read_voigt_strain(
            deformation,
            &lay.deformation,
            self.continuum.space_dim(),
            self.continuum.kinematics(),
        );
        let mat = MatRead::new(material, &lay.material, &lay.optional_material);
        // Dispatch **statique** : un `match` sur une énumération `Copy` ne coûte
        // rien et laisse le noyau s'inliner. Ce qu'on a mesuré, et non supposé :
        // remplacer `as_law()` par ce `match` n'a **rien** changé (p = 0,86) —
        // le surcoût venait de la frontière de module que le noyau doit
        // franchir, et c'est `#[inline]` sur `Linear::stress` qui l'a rendu.
        // Le `match` reste parce qu'il est gratuit et qu'il tient la boucle la
        // plus chaude hors du dispatch dynamique par construction, plutôt que
        // par la bonne volonté de l'optimiseur. `as_law()` sert les
        // déclarations de **zone**, où son coût est nul.
        match self.law {
            ElasticLaw::Linear => {
                linear::Linear.stress(&strain, &mat, &self.continuum, self.symmetry, out)
            }
        }
    }

    /// The geometric stiffness reads the current stress; the consistent tangent,
    /// the algorithmic moduli the integrator wrote. Both are declared here, once
    /// per zone, so the kernels below index instead of searching.
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        match kind {
            // La tangente d'une loi **linéaire** est sa raideur : elle lit le
            // matériau, et surtout pas des modules algorithmiques qu'aucun
            // intégrateur n'a écrits ici. Déclarer ces lectures ferait échouer
            // la résolution de zone sur un état qui n'a aucune raison de les
            // porter. Une loi non linéaire, elle, les produit et les relit.
            MatrixKind::Tangent if self.law.as_law().is_linear() => Vec::new(),
            _ => self.continuum.element_state_reads(kind),
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
        self.continuum.element_geometric(&geoms[0], state, lay, ke)
    }

    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        state: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        // A linear law's consistent tangent **is** its elastic stiffness, so it
        // reads the material and ignores the moduli it also declared. A law whose
        // tangent depends on `ε` wrote `D_alg` into the state; read it back.
        //
        // Ce `is_linear()` est lu **par maille**, non par zone — la signature du
        // trait est par maille. C'est sans conséquence : l'appel s'amortit sur
        // l'intégration complète d'une maille (`n_gauss × v × dofs²`), là où au
        // point de Gauss il aurait dominé.
        if self.law.as_law().is_linear() {
            self.continuum
                .element_stiffness(&geoms[0], material, lay, self.symmetry, ke)
        } else {
            self.continuum
                .element_tangent_from_state(&geoms[0], state, lay, ke)
        }
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        self.continuum
            .element_stiffness(&geoms[0], material, lay, self.symmetry, ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        self.continuum.element_mass(&geoms[0], material, lay, ke)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest state of `d` on its material — the `prev` of a first step,
    /// which the behaviour operator materializes for a caller who has none.
    fn rest<D: Domain>(d: &D, mat: &Handle<SubElementField>) -> Handle<SubElementField> {
        Handle::new(d.initial_state(&mat.read()).unwrap())
    }
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn unit_quad(kinematics: Kinematics) -> Elasticity {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Elasticity::new(fes.get(0).unwrap(), kinematics).unwrap()
    }

    #[test]
    fn vars_and_model_validation() {
        let el = unit_quad(Kinematics::PlaneStress);
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
        assert!(Elasticity::new(fes.get(0).unwrap(), Kinematics::Full3D).is_err());
    }

    #[test]
    fn plane_stress_constitutive_known_values() {
        let (e, nu) = (1.0, 0.25);
        let d = elastic::constitutive(e, nu, Kinematics::PlaneStress, 2);
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
        let el = unit_quad(Kinematics::PlaneStress);
        let mut mat =
            SubElementField::new(el.continuum.fespace(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", e).unwrap();
        mat.set_uniform("nu", nu).unwrap();
        let mat = Handle::new(mat);

        let mut strain = SubElementField::new(
            el.continuum.fespace(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = Handle::new(strain);

        let out = el
            .integrate_behavior(&strain, &rest(&el, &mat), &mat, 0.0)
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
        let el = unit_quad(Kinematics::PlaneStrain);
        let mut mat =
            SubElementField::new(el.continuum.fespace(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", 200.0).unwrap();
        mat.set_uniform("nu", 0.3).unwrap();
        let mat = Handle::new(mat);
        let blocks = el.build_stiffness_blocks(&mat).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = el.continuum.support().read().connectivity().to_vec();
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
