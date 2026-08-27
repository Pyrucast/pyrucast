//! Scalar and orthotropic **damage** — the physics, for any damage law.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`], and the **same
//! elastic stiffness** as iteration operator. The constitutive update is a
//! *secant* scalar-damage law: the stress is the elastic (effective) stress
//! scaled by `(1 − D)`, with `D ∈ [0, 1)` a scalar damage built from the
//! equivalent strain.
//!
//! Equivalent strain `ε̃ = √(Σ ⟨ε_I⟩₊²)` (positive parts of the principal
//! strains). Damage grows with the history variable `κ = maxₜ ε̃`, initialised
//! at the threshold `eps_d0`. Two damage branches `D_t` (tension) and `D_c`
//! (compression) are blended by weights `α_t`, `α_c` derived from the
//! tension/compression split of the effective stress:
//!
//! ```text
//! D_t = 1 − eps_d0(1−A_t)/κ − A_t / exp(B_t (κ − eps_d0))
//! D_c = 1 − eps_d0(1−A_c)/κ − A_c / exp(B_c (κ − eps_d0))
//! D   = α_t D_t + α_c D_c            (shear coefficient β fixed to 1)
//! σ   = (1 − D) · D_el : ε
//! ```
//!
//! Material components `E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`. The
//! single internal variable `kappa` comes in as the previous-step state `prev`
//! (`κ(A)`, floored at `eps_d0` in the update, so the zero of the rest state is
//! fine) and out as the updated `VAR1`, alongside the scalar `damage`. The
//! effective stress is a function of the current total strain `ε(B)` — damage
//! mechanics has no strain increment; only `κ` is history.
//!
//! The equivalent strain is built from the **principal strains of the full 3-D
//! tensor**, so the 2-D models differ only in how that tensor is reconstructed:
//! plane strain forces `ε_zz = 0`, plane stress derives it, and **axisymmetric**
//! reads the measured hoop `ε_θθ = u_r/r`.
//!
//! As for plasticity, the Newton loop driving the load increments lives in
//! Python, not in Rust; this module provides the point-wise update only.

pub mod damage_tc;
pub mod mazars;
pub mod sic_sic;

pub mod law;

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::elasticity::{self};
use crate::models::owned_components;
use crate::models::plasticity::law::MAX_INTERNAL_VARS;
use crate::models::tensor::Kinematics;
use crate::models::tensor::{dual_name, primal_name};
use crate::models::tensor::{stress_names, voigt_stress};
use crate::models::ZoneLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use law::{DamageLaw, MatRead};
use serde::{Deserialize, Serialize};

/// Damage on an FE subspace. Same supports as
/// [`crate::models::elasticity::Elasticity`]; material is supplied at
/// assembly / integration time.
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
/// # use pyrucast::models::damage::Damage;
/// # use pyrucast::models::tensor::Kinematics;
/// // Mazars par défaut : un seuil et deux branches, traction et compression.
/// let d = Damage::new(zone.clone(), Kinematics::PlaneStress)?;
/// assert!(d.material_components().contains(&"eps_d0".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Damage {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) kinematics: Kinematics,
    pub(crate) law: DamageLaw,
}

impl Damage {
    /// **Mazars** damage on an FE subspace — the default law.
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
    /// # use pyrucast::models::damage::Damage;
    /// # use pyrucast::models::tensor::Kinematics;
    /// // Mazars par défaut : un seuil et deux branches, traction et compression.
    /// let d = Damage::new(zone.clone(), Kinematics::PlaneStress)?;
    /// assert!(d.material_components().contains(&"eps_d0".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, kinematics: Kinematics) -> Result<Self> {
        Self::with_law(fespace, kinematics, DamageLaw::Mazars)
    }

    /// Damage with an explicit law, on an FE subspace with the given 2-D/3-D
    /// kinematics. Errors if
    /// `kinematics` is inconsistent with the space dimension (same rule as
    /// [`crate::models::elasticity::Elasticity::new`]).
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
    /// # use pyrucast::models::damage::{Damage};
    /// # use pyrucast::models::damage::law::{DamageLaw};
    /// # use pyrucast::models::tensor::Kinematics;
    /// // La loi explicite. Damage-TC suit deux endommagements, donc réclame
    /// // deux résistances là où Mazars n'en demande qu'une.
    /// let tc = Damage::with_law(
    ///     zone.clone(), Kinematics::PlaneStress, DamageLaw::DamageTc)?;
    /// assert!(tc.material_components().contains(&"f_t".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_law(
        fespace: Handle<SubFiniteElementSpace>,
        kinematics: Kinematics,
        law: DamageLaw,
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
        crate::models::elasticity::check_continuum_dimensions("Damage", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, kinematics) {
            (2, Kinematics::PlaneStress | Kinematics::PlaneStrain) => true,
            (2, Kinematics::Axisymmetric) => true,
            (3, Kinematics::Full3D) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Damage: kinematics {kinematics:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ full_3d)"
            )));
        }
        // Same two-way agreement as `Elasticity::new`.
        if axisymmetric != kinematics.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Damage: kinematics {kinematics:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` kinematics"
                )
            } else {
                "Damage: the `axisymmetric` kinematics requires an axisymmetric geometry \
                 (build the Coords with Coords::axisymmetric)"
                    .into()
            }));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            kinematics,
            law,
        })
    }
}

impl SubModelKind for Damage {
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

    /// The consistent mass matrix shares the stiffness layout (mass is
    /// law-independent).
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    /// The geometric stiffness shares the stiffness layout (initial-stress term
    /// is law-independent given the current stress).
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // Iteration operator = elastic (undamaged) stiffness. Reuse the
        // elasticity element kernel; it reads only `E` and `nu`.
        let mat = material;
        elasticity::element_stiffness(
            geom,
            mat,
            self.kinematics,
            crate::models::symmetry::MaterialSymmetry::Isotropic,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material;
        elasticity::element_mass(geom, mat, ke)
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: &SubElementField,
        state: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state;
        elasticity::element_geometric(geom, stress, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Damage"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Damage({:?}, {})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.kinematics, self.law
        )
    }
}

impl Domain for Damage {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Vec<String> {
        owned_components(self.law.material_components(self.space_dim))
    }

    /// `alpha` (thermal expansion) and `rho` (density) — the same pair
    /// [`elasticity`] accepts, and for the same
    /// reasons.
    ///
    /// `alpha` is read by an **ancillary** operator,
    /// [`thermal_strain`](fn@crate::ops::element_field::thermal_strain), which
    /// subtracts the expansion before the mechanical law sees anything: the
    /// return mapping never touches it. Leaving it out therefore excluded
    /// thermal expansion from plasticity and damage for no reason at all —
    /// `material_field` drops a component the physics does not declare, so the
    /// operator then found no zone carrying it.
    ///
    /// `rho` is required only by the mass matrix, never by the
    /// stiffness/behaviour assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha", "rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        let mut comps = stress_names(self.space_dim, self.kinematics);
        comps.push("damage".into());
        // The law's own history and per-direction damages.
        comps.extend(self.law.internal_names());
        comps
    }

    /// One damage step at a Gauss point. Output layout = stress (Voigt, `v`) +
    /// the reported `damage` + the law's own internal variables.
    fn deformation_reads(&self) -> Vec<String> {
        elasticity::strain_reads(self.space_dim, self.kinematics)
    }

    /// The law's own internal variables — `κ` for Mazars, one per direction for
    /// a woven composite — read back from what this physics wrote last step.
    fn state_reads(&self) -> Vec<String> {
        self.law.internal_names()
    }

    fn integrate_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        let d = self.space_dim;
        let read = MatRead {
            row: material,
            idx: &lay.material,
        };
        // `nu` is the second component of every damage contract.
        let eps = read_strain(
            deformation,
            &lay.deformation,
            d,
            read.get(1),
            self.kinematics,
        );
        // Always present: the state at rest is materialized once, before the
        // first step, so no law has to recognise « no state yet » here.
        let mut vars = [0.0_f64; MAX_INTERNAL_VARS];
        let n_vars = lay.state.len();
        for (k, v) in vars[..n_vars].iter_mut().enumerate() {
            *v = prev[lay.state[k] as usize];
        }

        let update = self.law.update(&eps, &vars[..n_vars], &read, d)?;
        let v = stress_names(d, self.kinematics).len();
        for r in 0..v {
            out[r] = voigt_stress(&update.sigma, d, self.kinematics, r);
        }
        out[v] = update.damage;
        for (i, value) in update.vars.iter().enumerate() {
            out[v + 1 + i] = *value;
        }
        Ok(())
    }
}

// The constitutive cores live in [`crate::models::damage`]'s submodules, one
// per law — shared helpers (Lamé, the elastic stress, the positive part) in the
// module root below. What remains here is the physics: the DOFs, the layouts,
// and the plumbing between the field components and the full-3-D strain the
// laws work in.

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// The full 3-D strain a damage law sees, from the deformation row.
///
/// Plane stress derives the out-of-plane `ε_zz` from the in-plane pair (there is
/// no law solve here — damage is secant); axisymmetry reads the *measured* hoop.
fn read_strain(
    deformation: &[f64],
    idx: &[u32],
    space_dim: usize,
    nu: f64,
    kinematics: Kinematics,
) -> [f64; 6] {
    let mut eps = [0.0; 6];
    let slots: &[usize] = if space_dim == 2 && kinematics.is_axisymmetric() {
        &[0, 1, 2, 5]
    } else if space_dim == 2 {
        &[0, 1, 5]
    } else {
        &[0, 1, 2, 3, 4, 5]
    };
    for (r, &slot) in slots.iter().enumerate() {
        eps[slot] = deformation[idx[r] as usize];
    }
    if space_dim == 2 && kinematics == Kinematics::PlaneStress {
        eps[2] = -nu / (1.0 - nu) * (eps[0] + eps[1]);
    }
    eps
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest state of `d` on its material — the `prev` of a first step,
    /// which the behaviour operator materializes for a caller who has none.
    fn rest<D: Domain>(d: &D, mat: &Handle<SubElementField>) -> Handle<SubElementField> {
        Handle::new(d.initial_state(&mat.read()).unwrap())
    }
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn unit_quad(kinematics: Kinematics) -> Damage {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Damage::new(fes.get(0).unwrap(), kinematics).unwrap()
    }

    fn material(mz: &Damage) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            mz.fespace.clone(),
            mazars::MATERIAL.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        mat.set_uniform("E", 30_000.0).unwrap(); // ~ concrete (MPa)
        mat.set_uniform("nu", 0.2).unwrap();
        mat.set_uniform("eps_d0", 1e-4).unwrap();
        mat.set_uniform("A_t", 0.8).unwrap();
        mat.set_uniform("B_t", 20_000.0).unwrap();
        mat.set_uniform("A_c", 1.4).unwrap();
        mat.set_uniform("B_c", 1_900.0).unwrap();
        Handle::new(mat)
    }

    fn strain_field(mz: &Damage, eps_xx: f64) -> Handle<SubElementField> {
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        s.set_uniform("eps_xx", eps_xx).unwrap();
        Handle::new(s)
    }

    #[test]
    fn vars_and_material() {
        let mz = unit_quad(Kinematics::PlaneStress);
        assert_eq!(mz.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(mz.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(mz.material_components(), owned_components(mazars::MATERIAL));
    }

    /// Below the damage threshold the response is elastic: D = 0 and σ_xx is
    /// the linear plane-stress stress.
    #[test]
    fn undamaged_below_threshold() {
        let mz = unit_quad(Kinematics::PlaneStress);
        let mat = material(&mz);
        let eps0 = 1e-5; // < eps_d0 = 1e-4
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, &rest(&mz, &mat), &mat, 0.0)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap().abs() < 1e-14);
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-6);
        }
    }

    /// Above the threshold in tension, damage develops (0 < D < 1) and the
    /// stress is reduced below the elastic prediction.
    #[test]
    fn damages_in_tension() {
        let mz = unit_quad(Kinematics::PlaneStress);
        let mat = material(&mz);
        let eps0 = 5e-4; // > eps_d0
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, &rest(&mz, &mat), &mat, 0.0)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            let d = out.value(0, g, "damage").unwrap();
            assert!(d > 0.0 && d < 1.0, "D = {d}");
            // Damaged stress strictly below the elastic prediction.
            assert!(out.value(0, g, "sigma_xx").unwrap() < c * eps0);
            assert!(out.value(0, g, "kappa").unwrap() >= eps0 - 1e-12);
        }
    }

    /// History variable κ is monotone: unloading to a smaller strain does not
    /// reduce κ, and does not heal damage.
    #[test]
    fn kappa_is_monotone() {
        let mz = unit_quad(Kinematics::PlaneStress);
        let mat = material(&mz);
        // Load to 5e-4.
        let s1 = strain_field(&mz, 5e-4);
        let st1 = mz
            .integrate_behavior(&s1, &rest(&mz, &mat), &mat, 0.0)
            .unwrap();
        let k1 = st1.value(0, 0, "kappa").unwrap();
        let d1 = st1.value(0, 0, "damage").unwrap();

        // Unload to 2e-4, feeding the step-1 state (κ) via `prev`.
        let prev = Handle::new(st1);
        let s2 = strain_field(&mz, 2e-4);
        let st2 = mz.integrate_behavior(&s2, &prev, &mat, 0.0).unwrap();
        assert!((st2.value(0, 0, "kappa").unwrap() - k1).abs() < 1e-12);
        // Damage unchanged on unloading (same κ).
        assert!((st2.value(0, 0, "damage").unwrap() - d1).abs() < 1e-9);
    }

    /// Solid 3-D uniaxial tension also triggers tensile damage.
    #[test]
    fn solid_3d_damages() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]).unwrap();
        let n = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
            p(1.0, 0.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 1.0),
        ];
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
        mesh.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
            .unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mz = Damage::new(fes.get(0).unwrap(), Kinematics::Full3D).unwrap();
        let mat = material(&mz);
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            ["xx", "yy", "zz", "yz", "xz", "xy"]
                .iter()
                .map(|x| format!("eps_{x}"))
                .collect(),
        )
        .unwrap();
        s.set_uniform("eps_xx", 5e-4).unwrap();
        let s = Handle::new(s);
        let out = mz
            .integrate_behavior(&s, &rest(&mz, &mat), &mat, 0.0)
            .unwrap();
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap() > 0.0);
        }
    }
}
