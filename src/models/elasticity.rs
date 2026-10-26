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
use crate::containers::matrix::Symmetry;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::continuum::material::MatRead;
use crate::models::continuum::{elastic, voigt, Continuum};
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::tensor::{dual_name, primal_name, Kinematics};
use crate::models::ZoneLayout;
use crate::models::{
    Behavior, CellGeom, Domain, MatrixLayout, Physics, StrainOperator, SubModelKind,
};
use crate::models::{ElementLayout, MatrixKind};
use crate::ops::element_field::StrainMeasure;
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
/// // Thermal expansion is **optional**: without `alpha`, the model assembles
/// // without it.
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
    /// // Thermal expansion is **optional**: without `alpha`, the model assembles
    /// // without it.
    /// assert!(e.optional_material_components().contains(&"alpha"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, kinematics: Kinematics) -> Result<Self> {
        Self::with_symmetry(
            fespace,
            kinematics,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
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
    /// // The general constructor: an orthotropic symmetry widens the material
    /// // contract, which then carries the moduli **and** the axes.
    /// # use pyrucast::ops::element_field::StrainMeasure;
    /// let o = Elasticity::with_symmetry(
    ///     zone.clone(), Kinematics::PlaneStress, MaterialSymmetry::Orthotropic,
    ///     StrainMeasure::Linear)?;
    /// assert!(o.material_components().len() > 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        kinematics: Kinematics,
        symmetry: MaterialSymmetry,
        measure: StrainMeasure,
    ) -> Result<Self> {
        Ok(Self {
            continuum: Continuum::new(fespace, kinematics, "Elasticity")?
                .with_measure(measure, "Elasticity")?,
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

    fn as_behavior(&self) -> Option<&dyn Behavior> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.continuum.fespace()],
            support: self.continuum.support(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetry: Symmetry::Full,
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

    /// The geometric stiffness reads the current stress; the tangent reads nothing
    /// any more — it is evaluated at the point.
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        self.continuum.element_state_reads(kind)
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

impl Behavior for Elasticity {
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.continuum.fespace()
    }

    /// The law **is** its elastic operator, in whatever symmetry it was built.
    fn elastic_operator(&self) -> Option<(&Continuum, MaterialSymmetry)> {
        Some((&self.continuum, self.symmetry))
    }

    fn behavior_output_components(&self) -> Vec<String> {
        self.continuum.stress_names()
    }

    /// Linear stress σ = D·ε at one Gauss point (material constants per cell).
    fn deformation_reads(&self) -> Vec<String> {
        self.continuum.strain_reads()
    }

    fn strain_operator(&self) -> StrainOperator {
        StrainOperator::Deformation(self.continuum.measure())
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
        // **Static** dispatch: a `match` on a `Copy` enumeration costs
        // rien et laisse le noyau s'inliner. Ce qu'on a mesuré, et non supposé :
        // replacing `as_law()` with this `match` changed **nothing** (p = 0.86) —
        // the overhead came from the module boundary the kernel must cross, and
        // `#[inline]` on `Linear::stress` is what gave it back.
        // The `match` stays because it is free and keeps the hottest loop out of
        // dynamic dispatch by construction, rather than by the optimizer's good
        // will. `as_law()` serves the **zone** declarations, where its cost is nil.

        match self.law {
            ElasticLaw::Linear => {
                linear::Linear.stress(&strain, &mat, &self.continuum, self.symmetry, out)
            }
        }
    }

    /// The consistent tangent of a linear law is its elastic modulus — constant,
    /// but a tangent all the same, and it travels the same road as the others.
    fn tangent_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        _deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        d: &mut [[f64; 6]; 6],
    ) -> Result<()> {
        symmetry::elastic_constitutive_into(
            material,
            &lay.material,
            self.symmetry,
            self.continuum.kinematics(),
            self.continuum.space_dim(),
            d,
        )?;
        Ok(())
    }

    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        lay: &ZoneLayout,
        deformation: &SubElementField,
        prev: &SubElementField,
        material: &SubElementField,
        dt: f64,
        ke: &mut [f64],
    ) -> Result<()> {
        self.continuum
            .element_tangent_of(self, geoms, lay, deformation, prev, material, dt, ke)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest state of `d` on its material — the `prev` of a first step,
    /// which the behaviour operator materializes for a caller who has none.
    fn rest<B: Behavior>(b: &B, mat: &Handle<SubElementField>) -> Handle<SubElementField> {
        Handle::new(b.initial_state(&mat.read()).unwrap())
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
                        let lhs = k.get(ni, &format!("f_{a}"), nj, &format!("u_{b}")).unwrap();
                        let rhs = k.get(nj, &format!("f_{b}"), ni, &format!("u_{a}")).unwrap();
                        assert!((lhs - rhs).abs() < tol);
                    }
                }
            }
        }
        // A uniform translation in x ⇒ zero force everywhere (row sum = 0).
        for &ni in &nodes {
            let row: f64 = nodes
                .iter()
                .map(|&nj| k.get(ni, "f_x", nj, "u_x").unwrap())
                .sum();
            assert!(row.abs() < tol, "row sum {row} ≠ 0");
        }
    }
}

crate::physics_operator! {
    /// Linear-elasticity `Model` spanning **every** subspace of `fes` — one
    /// [`SubModel::Elasticity`] per [`SubFiniteElementSpace`], with the same
    /// 2-D/3-D `kinematics` for all. Parent-level operator.
    ///
    /// `symmetry` is the **material** axis, independent of the kinematic one:
    /// `E`, `nu` (isotropic), the per-direction moduli (orthotropic) or the 21
    /// constants `C_11 … C_66` (anisotropic), the two oriented ones with the
    /// material axes. `strain_measure` is the strain the law reads — `Linear`,
    /// `GreenLagrange` or `Hencky` — and names the stress it writes (`sigma_*`,
    /// `S_*`, `T_*`). The material is supplied at assembly time.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::ops::element_field::StrainMeasure;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let m = model::elasticity(
    ///     &fes, Kinematics::PlaneStress, MaterialSymmetry::Isotropic, StrainMeasure::Linear)?;
    /// assert_eq!(m.dual_vars(), vec!["f_x".to_string(), "f_y".to_string()]);
    /// // The material axis is independent of the kinematic one.
    /// let ortho = model::elasticity(
    ///     &fes, Kinematics::PlaneStress, MaterialSymmetry::Orthotropic, StrainMeasure::Linear)?;
    /// let reads = ortho.get(0).unwrap().read().material_components().unwrap();
    /// assert!(reads.contains(&"E_1".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn elasticity(
        fes,
        kinematics: Kinematics,
        symmetry: MaterialSymmetry = MaterialSymmetry::Isotropic,
        strain_measure: StrainMeasure = StrainMeasure::Linear,
    ) via SubModel::elasticity;
    python: "`model.elasticity(fespace, kinematics, symmetry=\"isotropic\",\nstrain_measure=\"linear\")` — linear-elasticity model spanning every subspace of\n`fespace`. `kinematics` is `\"plane_stress\"`, `\"plane_strain\"` or\n`\"axisymmetric\"` (2-D), or `\"full_3d\"` (3-D). DOFs are the vector\ndisplacement `u_x, u_y(, u_z)`; material is supplied at assembly time.\n\n`\"axisymmetric\"` requires a geometry built with `Coords.axisymmetric()`\n(`x = r`, `y = z`): the hoop strain `ε_θθ = u_r / r` comes from the kinematics,\nthe `2πr` integration measure from the geometry, and the two must agree.\n\n`symmetry` is the **material** axis, independent of the kinematic one:\n`\"isotropic\"` (the default, material `E`, `nu`), `\"orthotropic\"`\n(`E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23`) or\n`\"anisotropic\"` (the 21 constants `C_11 … C_66`). The two oriented ones\nalso require the material axes — `V1X, V1Y` in 2-D, `V1X…V1Z, V2X…V2Z`\nin 3-D — which are orthonormalised internally.\n\n`strain_measure` is the strain the law reads: `\"linear\"` (small strains,\nthe default), `\"green_lagrange\"` (large rotations: Hooke on `E` is\nSaint-Venant–Kirchhoff) or `\"hencky\"` (`½ ln C`). The stress it writes is\nthe dual, named after it: `sigma_*`, `S_*` or `T_*`."
}
