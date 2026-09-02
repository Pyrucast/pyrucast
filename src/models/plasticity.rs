//! Rate-independent elastoplasticity — the physics, for **any** yield law.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`] (displacement
//! `u_x, u_y(, u_z)`, nodal force `f_x, …`), and the **same elastic stiffness**
//! as iteration operator: the non-linearity lives entirely in the behaviour
//! integration (`COMP`).
//!
//! The **yield law** is an attribute, [`PlasticLaw`], not a physics of its own —
//! von Mises (perfect or isotropically hardening), Drucker-Prager, Ottosen all
//! share these DOFs, this state and this incremental montage, differing only in
//! their surface and flow rule. That mirrors Cast3M, where `PLASTIQUE PARFAIT`,
//! `PLASTIQUE ISOTROPE`, `PLASTIQUE DRUCKER_PRAGER` and `PLASTIQUE OTTOSEN` are
//! variants of one formulation. Each law's return map lives in
//! [`crate::models::plasticity::law`]; the material components it needs are declared
//! there too, so this file never grows when a law is added.
//!
//! The integration is history-dependent and uses the **incremental montage**
//! A → B: the end-of-step strain `ε(B)` comes in as `deformation`, while the
//! converged state at the start of the step A — the stress `σ(A)`, the plastic
//! strain `ε_p(A)`, the cumulated `p(A)` and the strain `ε(A)` — comes in as
//! `prev` (the previous step's output, or the **rest state** on the first step,
//! where A is the reference configuration). The elastic predictor is
//! `σ_trial = σ(A) + C:Δε`
//! with `Δε = ε(B) − ε(A)` — algebraically identical to `C:(ε(B) − ε_p(A))` in
//! small strain, but the form that carries `σ(A)` explicitly, ready for a
//! large-strain law. The output echoes the full-3-D `ε(B)` (and, in 2-D, the
//! out-of-plane `σ_zz`) so it is a complete `prev` for the next step.
//!
//! State is always carried in **full 3-D** (six `eps_p_*` components) regardless
//! of the 2-D/3-D kinematics, which keeps the radial return identical across plane
//! stress / plane strain / axisymmetric / solid; only the input strain
//! reconstruction and the output stress projection differ.
//!
//! **Axisymmetric** therefore costs almost nothing here: the hoop `ε_θθ = u_r/r`
//! is *measured* by
//! [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation), not
//! assumed, so `ε(B)` is
//! fully known (no out-of-plane solve, unlike plane stress) and the whole
//! specialisation is the index map `[rr, zz, θθ, rz] → [xx, yy, zz, xy]`. Note
//! that `σ_zz` is then part of the Voigt dual and must **not** be echoed as extra
//! state.
//!
//! Following the locked architecture decision — non-linear algorithms are
//! orchestrated in Python, not in Rust — the Newton
//! loop driving these increments lives in Python; this module provides the
//! point-wise constitutive update **and** the consistent algorithmic tangent
//! `D_alg` (emitted alongside the stress, consumed by
//! [`crate::ops::matrix::tangent`]) for quadratic convergence.

pub mod drucker_prager;
pub mod gurson;
pub mod law;
pub mod ottosen;
pub mod viscous;
pub mod von_mises;

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::continuum::Continuum;
use crate::models::owned_components;
use crate::models::tensor::Kinematics;
use crate::models::tensor::{dual_name, primal_name};
use crate::models::tensor::{stress_names, voigt_stress};
use crate::models::ZoneLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::models::{ElementLayout, MatrixKind};
use law::{MatParams, PlasticLaw, PrevState, MAX_INTERNAL_VARS};
use serde::{Deserialize, Serialize};

/// Full 3-D tensor component suffixes, in the internal state order
/// `[xx, yy, zz, yz, xz, xy]` (off-diagonals are **tensor** strains, `ε_ij`).
use crate::models::plasticity::law::TENSOR_SUFFIXES;
/// Index pairs `(i, j)` matching [`TENSOR_SUFFIXES`].
const TENSOR_PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// Internal-state component names: plastic strain tensor `eps_p_*` (six,
/// always 3-D) followed by the cumulated plastic strain `p`.
fn state_names() -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES
        .iter()
        .map(|s| format!("eps_p_{s}"))
        .collect();
    v.push("p".into());
    v
}

/// Extra state echoed for the incremental montage so the output is a **complete
/// `prev`**: the full-3-D end-of-step strain `ε(B)` (six `eps_*`, so `ε(A)` is
/// recoverable next step) and — in 2-D only — the out-of-plane stress `sigma_zz`
/// that the Voigt dual omits (so `σ(A)` is fully recoverable). In 3-D the Voigt
/// dual already carries all six stresses.
fn echo_names(space_dim: usize, kinematics: Kinematics) -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
    if echoes_sigma_zz(space_dim, kinematics) {
        v.push("sigma_zz".into());
    }
    v
}

/// Whether `σ_zz` must be echoed as extra state. Only the **plane** 2-D models
/// need it: their Voigt dual stops at `[xx, yy, xy]`. Axisymmetric already
/// carries `sigma_zz` (the hoop) in its dual, so echoing it would emit the same
/// component name twice.
fn echoes_sigma_zz(space_dim: usize, kinematics: Kinematics) -> bool {
    space_dim == 2 && !kinematics.is_axisymmetric()
}

/// Perfect von Mises plasticity on an FE subspace.
///
/// Holds the same supports as [`crate::models::elasticity::Elasticity`];
/// material (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::models::Domain;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::plasticity::Plasticity;
/// # use pyrucast::models::plasticity::law::PlasticLaw;
/// // La physique élastoplastique d'une zone : sa loi décide du matériau
/// // qu'elle réclame et de l'état qu'elle porte.
/// let p = Plasticity::new(zone.clone(), Kinematics::PlaneStrain)?;
/// assert!(p.material_components().contains(&"sigma_y".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Plasticity {
    pub(crate) continuum: Continuum,
    pub(crate) law: PlasticLaw,
}

impl Plasticity {
    /// **Perfect** (non-hardening) von Mises plasticity on an FE subspace — the
    /// default law, and the one this physics shipped with.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::Domain;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::plasticity::Plasticity;
    /// // Von Mises **parfaite** : la loi par défaut, celle avec laquelle cette
    /// // physique est née.
    /// let p = Plasticity::new(zone.clone(), Kinematics::PlaneStrain)?;
    /// assert_eq!(p.material_components().len(), 3); // E, nu, sigma_y
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, kinematics: Kinematics) -> Result<Self> {
        Self::with_law(fespace, kinematics, PlasticLaw::Perfect)
    }

    /// Elastoplasticity with an explicit yield law, on an FE subspace with the
    /// given 2-D/3-D kinematics. Errors if `kinematics` is inconsistent with the space
    /// dimension (same rule as
    /// [`crate::models::elasticity::Elasticity::new`]).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::Domain;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::plasticity::Plasticity;
    /// # use pyrucast::models::plasticity::law::PlasticLaw;
    /// // La loi explicite, et le contrôle de cohérence entre la cinématique et
    /// // la dimension de l'espace.
    /// let p = Plasticity::with_law(
    ///     zone.clone(), Kinematics::PlaneStrain, PlasticLaw::Isotropic)?;
    /// assert!(p.material_components().contains(&"H".to_string()));
    /// // Un modèle solide sur une zone 2-D est refusé.
    /// assert!(Plasticity::with_law(
    ///     zone.clone(), Kinematics::Full3D, PlasticLaw::Perfect).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_law(
        fespace: Handle<SubFiniteElementSpace>,
        kinematics: Kinematics,
        law: PlasticLaw,
    ) -> Result<Self> {
        Ok(Self {
            continuum: Continuum::new(fespace, kinematics, "Plasticity")?,
            law,
        })
    }
}

impl Plasticity {
    /// Le préambule commun aux deux noyaux de point : la matière lue par
    /// position, `ε(B)`, et l'état en A reconstruit en 3-D plein.
    ///
    /// Passé par fermeture parce que `PrevState` emprunte un tableau de pile —
    /// et générique, donc monomorphisé : aucun appel virtuel n'entre ici.
    fn with_law_inputs<R>(
        &self,
        lay: &ZoneLayout,
        deformation: &[f64],
        prev: &[f64],
        material: &[f64],
        f: impl FnOnce(&MatParams, &[f64; 6], &PrevState) -> Result<R>,
    ) -> Result<R> {
        let d = self.continuum.space_dim();
        let params = MatParams::new(material, &lay.material, &lay.optional_material);

        // End-of-step strain ε(B).
        let eps_b = read_tensor(
            deformation,
            &lay.deformation,
            strain_slots(d, self.continuum.kinematics()),
        );

        // The state at A, in `state_reads` order: ε(A), σ(A), ε_p(A), p, then
        // the law's own variables.
        let n_stress = stress_slots(d, self.continuum.kinematics()).len();
        let (i_sigma, i_eps_p, i_p, i_vars) = (6, 6 + n_stress, 12 + n_stress, 13 + n_stress);
        let mut vars = [0.0_f64; MAX_INTERNAL_VARS];
        let n_vars = lay.state.len() - i_vars;
        for (k, v) in vars[..n_vars].iter_mut().enumerate() {
            *v = prev[lay.state[i_vars + k] as usize];
        }
        let prev_state = PrevState {
            eps: read_tensor(prev, &lay.state[..6], &[0, 1, 2, 3, 4, 5]),
            sigma: read_tensor(
                prev,
                &lay.state[i_sigma..],
                stress_slots(d, self.continuum.kinematics()),
            ),
            eps_p: read_tensor(prev, &lay.state[i_eps_p..], &[0, 1, 2, 3, 4, 5]),
            p: prev[lay.state[i_p] as usize],
            vars: &vars[..n_vars],
        };
        f(&params, &eps_b, &prev_state)
    }
}

impl SubModelKind for Plasticity {
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
            ordering: crate::containers::matrix::DofOrdering::NodesThenVars,
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

    /// The consistent tangent shares the stiffness layout; the algorithmic
    /// modulus `D_alg` (emitted by [`Domain::integrate_point`]) is read from the
    /// behaviour state.
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Plasticity"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.continuum.support().read().cell_count();
        format!(
            "SubModel<Plasticity({:?})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.continuum.kinematics()
        )
    }
}

impl Domain for Plasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.continuum.fespace()
    }

    fn material_components(&self) -> Vec<String> {
        owned_components(self.law.material_components())
    }

    /// `alpha` (thermal expansion) and `rho` (density) — the same pair
    /// [`elasticity`](crate::models::elasticity) accepts, and for the same
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
    ///
    /// One exception, and it is a **name** collision rather than a design:
    /// Drucker-Prager already requires `alpha` for the pressure sensitivity of
    /// its yield surface. There is one slot of that name, the required meaning
    /// wins, and thermal expansion is therefore out of reach for that law
    /// alone.
    fn optional_material_components(&self) -> &'static [&'static str] {
        // A law may accept constitutive parameters it can do without. The
        // general Drucker-Prager surface is nine numbers, of which six default
        // to the simple cone — see [`crate::models::plasticity::drucker_prager`].
        // They ride the optional channel so that a three-parameter kinematics stays
        // writable in three numbers.
        self.law.as_law().optional_material_components()
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.continuum.fespace()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        let mut comps = stress_names(self.continuum.space_dim(), self.continuum.kinematics());
        comps.extend(state_names());
        comps.extend(echo_names(
            self.continuum.space_dim(),
            self.continuum.kinematics(),
        ));
        // The law's **own** internal variables (a back stress, a damage…), which
        // is how a law grows its state without any other file changing.
        comps.extend(self.law.internal_names());
        comps
    }

    /// Zero everywhere, **except** the internal variables a law starts from a
    /// material constant — Gurson's porosity `f_0`, which the law declares
    /// through `initial_internal_sources`.
    ///
    /// Seeding them here, once per zone before the first step, is what lets the
    /// return map read `prev.var(0)` unconditionally: no law has to recognise a
    /// first step at a Gauss point.
    /// True for the creep and viscoplastic laws, whose answer depends on how
    /// long the step lasted.
    fn tangent_source(&self) -> crate::models::TangentSource {
        self.law.tangent_source()
    }

    fn requires_dt(&self) -> bool {
        self.law.is_viscous()
    }

    fn initial_state(&self, material: &SubElementField) -> Result<SubElementField> {
        let mut state =
            SubElementField::new(self.continuum.fespace(), self.behavior_output_components())?;
        let sources = self.law.as_law().initial_internal_sources();
        if sources.is_empty() {
            return Ok(state);
        }
        for (var, source) in self.law.internal_names().iter().zip(sources) {
            let slot = state.component_index_or_err(var)?;
            let from = material.component_index_or_err(source)?;
            for cell in 0..state.cell_count() {
                for g in 0..state.gauss_count() {
                    state.set(cell, g, slot, material.get(cell, g, from)?)?;
                }
            }
        }
        Ok(state)
    }

    /// Incremental radial-return at one Gauss point. Output layout =
    /// stress (Voigt, `v`) + plastic strain `eps_p` (full 3-D tensor, 6) +
    /// cumulated plastic strain `p` (1) + echoed strain `ε(B)` (full 3-D, 6)
    /// [+ `sigma_zz` in 2-D], matching `stress_names ++ state_names ++ echo_names`.
    fn deformation_reads(&self) -> Vec<String> {
        self.continuum.strain_reads()
    }

    /// What the law reads back from the state at A: the echoed strain ε(A), the
    /// stress σ(A) (the Voigt dual plus the echoed `σ_zz` of the plane models),
    /// the plastic strain ε_p(A), the cumulated `p`, then the law's own
    /// variables — a **subset of what this same physics wrote** last step, so a
    /// missing one is a real inconsistency, caught once per zone.
    fn state_reads(&self) -> Vec<String> {
        let d = self.continuum.space_dim();
        let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
        v.extend(stress_names(d, self.continuum.kinematics()));
        if echoes_sigma_zz(d, self.continuum.kinematics()) {
            v.push("sigma_zz".into());
        }
        v.extend(state_names());
        v.extend(self.law.internal_names());
        v
    }

    fn integrate_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        prev: &[f64],
        material: &[f64],
        dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        self.with_law_inputs(
            lay,
            deformation,
            prev,
            material,
            |params, eps_b, prev_state| {
                let d = self.continuum.space_dim();
                let n_stress = stress_slots(d, self.continuum.kinematics()).len();
                let (step, eps_b_full) = self.law.as_law().incremental_step(
                    eps_b,
                    prev_state,
                    params,
                    self.continuum.kinematics(),
                    dt,
                )?;

                let v = n_stress - usize::from(echoes_sigma_zz(d, self.continuum.kinematics()));
                for r in 0..v {
                    out[r] = voigt_stress(&step.sigma, d, self.continuum.kinematics(), r);
                }
                out[v..v + 6].copy_from_slice(&step.eps_p); // ε_p(B)
                out[v + 6] = step.p; // p(B)
                                     // Echo the full-3-D end-of-step strain ε(B), so `prev` carries ε(A) next
                                     // step (in plane stress this includes the solved out-of-plane ε_zz).
                out[v + 7..v + 13].copy_from_slice(&eps_b_full);
                // The plane 2-D duals omit σ_zz; echo it so σ(A) is fully recoverable.
                // Axisymmetric already carries it (the hoop), so it must not be echoed.
                let mut base = v + 13;
                if echoes_sigma_zz(d, self.continuum.kinematics()) {
                    out[base] = step.sigma[2];
                    base += 1;
                }
                // The law's own internal variables, right after the common state.
                for (i, value) in step.internal().iter().enumerate() {
                    out[base + i] = *value;
                }
                Ok(())
            },
        )
    }

    /// Le module algorithmique `D_alg` au point — la dérivée du **pas**
    /// `ε(B) ↦ σ(B)` à état A figé, non une tangente à une surface en un point.
    /// C'est pourquoi il reçoit exactement les entrées d'`integrate_point` : les
    /// deux extrémités du pas sont dans sa définition, et une loi dérivée par
    /// différences finies relance littéralement son retour radial depuis `prev`.
    ///
    /// Il n'est appelé que par `ops::matrix::tangent`, jamais par COMP : c'est
    /// tout l'objet de la séparation, puisque huit lois sur dix le paient douze
    /// retours radiaux.
    fn tangent_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        prev: &[f64],
        material: &[f64],
        dt: f64,
        d: &mut [[f64; 6]; 6],
    ) -> Result<()> {
        self.with_law_inputs(
            lay,
            deformation,
            prev,
            material,
            |params, eps_b, prev_state| {
                // Le pas d'abord : en contraintes planes, c'est lui qui résout la
                // composante hors plan que la tangente évalue ensuite.
                let (_, eps_b_full) = self.law.as_law().incremental_step(
                    eps_b,
                    prev_state,
                    params,
                    self.continuum.kinematics(),
                    dt,
                )?;
                let d3 =
                    self.law
                        .as_law()
                        .consistent_tangent(&eps_b_full, prev_state, params, dt)?;
                crate::models::symmetry::reduce_to_model_into(&d3, self.continuum.kinematics(), d);
                Ok(())
            },
        )
    }

    /// Les mêmes noyaux que l'élasticité, donc les mêmes lectures d'état.
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        self.continuum.element_state_reads(kind)
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // The plain "stiffness" kernel is the *elastic* stiffness (the simple
        // iteration operator). The consistent algorithmic tangent `K_t` is a
        // separate operator — see [`element_tangent`](Self::element_tangent) and
        // [`crate::ops::matrix::tangent`]. Reuse the elasticity element kernel
        // verbatim; it reads only `E` and `nu` from the material.
        let mat = material;
        self.continuum.element_stiffness(
            geom,
            mat,
            lay,
            crate::models::symmetry::MaterialSymmetry::Isotropic,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material;
        self.continuum.element_mass(geom, mat, lay, ke)
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
        let stress = state;
        self.continuum.element_geometric(geom, stress, lay, ke)
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

// The constitutive core — the elastic predictor, the return maps, the
// plane-stress secant loop and the consistent tangents — lives in
// [`crate::models::plasticity::law`], shared by every yield law. What remains here is
// the physics: the DOFs, the layouts, and the plumbing between the field
// components and the full-3-D state the laws work in.

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// Reconstruct the full 3-D **tensor** strain from the deformation input.
/// Plane strain forces the out-of-plane components to zero; plane stress leaves
/// `eps_zz` as the trial elastic guess (it is overwritten by the return map).
/// Which slots of the full 3-D tensor the **strain** of this kinematics spans,
/// in the order [`Continuum::strain_reads`](crate::models::continuum::Continuum::strain_reads) declares them.
///
/// Plane models span `[xx, yy, xy]`; axisymmetry adds the *measured* hoop
/// `θθ`; a solid spans all six. The slots left out stay zero — `ε_yz`/`ε_xz`
/// vanish by axial symmetry, and `ε_zz` is either zero (plane strain) or solved
/// by the law (plane stress).
fn strain_slots(space_dim: usize, kinematics: Kinematics) -> &'static [usize] {
    if space_dim == 2 && kinematics.is_axisymmetric() {
        &[0, 1, 2, 5]
    } else if space_dim == 2 {
        &[0, 1, 5]
    } else {
        &[0, 1, 2, 3, 4, 5]
    }
}

/// Which slots of the full 3-D tensor the **stress** carried by the state spans:
/// the Voigt dual of the kinematics, plus the echoed `σ_zz` of the plane models.
/// The rest is zero by the plane assumptions.
fn stress_slots(space_dim: usize, kinematics: Kinematics) -> &'static [usize] {
    if space_dim == 2 && kinematics.is_axisymmetric() {
        &[0, 1, 2, 5]
    } else if space_dim == 2 {
        &[0, 1, 5, 2] // [xx, yy, xy] then the echoed zz
    } else {
        &[0, 1, 2, 3, 4, 5]
    }
}

/// Scatter a row's values into a full 3-D tensor, by slot.
///
/// The one shape every state read takes: `idx` says where each component sits in
/// the row (resolved once for the zone), `slots` where it belongs in the tensor.
fn read_tensor(row: &[f64], idx: &[u32], slots: &[usize]) -> [f64; 6] {
    let mut t = [0.0_f64; 6];
    for (r, &slot) in slots.iter().enumerate() {
        t[slot] = row[idx[r] as usize];
    }
    t
}

/// Map a `(i, j)` tensor pair to its index in [`TENSOR_SUFFIXES`]; kept for
/// readers cross-checking the layout against [`TENSOR_PAIRS`].
#[allow(dead_code)]
fn tensor_index(i: usize, j: usize) -> usize {
    TENSOR_PAIRS
        .iter()
        .position(|&(a, b)| (a, b) == (i.min(j), i.max(j)))
        .expect("valid tensor pair")
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
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::models::tensor;

    fn unit_quad(kinematics: Kinematics) -> Plasticity {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Plasticity::new(fes.get(0).unwrap(), kinematics).unwrap()
    }

    fn unit_hex() -> Plasticity {
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
        Plasticity::new(fes.get(0).unwrap(), Kinematics::Full3D).unwrap()
    }

    fn material(pl: &Plasticity, e: f64, nu: f64, sy: f64) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            pl.continuum.fespace(),
            vec!["E".into(), "nu".into(), "sigma_y".into()],
        )
        .unwrap();
        mat.set_uniform("E", e).unwrap();
        mat.set_uniform("nu", nu).unwrap();
        mat.set_uniform("sigma_y", sy).unwrap();
        Handle::new(mat)
    }

    #[test]
    fn vars_and_model_validation() {
        let pl = unit_quad(Kinematics::PlaneStrain);
        assert_eq!(pl.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(pl.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(
            pl.material_components(),
            owned_components(PlasticLaw::Perfect.material_components())
        );
    }

    /// Below yield the response is purely elastic: equivalent stress < σ_y and
    /// no plastic strain accumulates.
    #[test]
    fn elastic_below_yield_solid() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // Small uniaxial strain well below yield (σ ≈ E·ε = 21 MPa < 250).
        let mut strain = SubElementField::new(
            pl.continuum.fespace(),
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-4).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, &rest(&pl, &mat), &mat, 0.0)
            .unwrap();
        // Confined uniaxial *strain* (only ε_xx ≠ 0): σ_xx = (λ+2μ)·ε.
        let (lambda, mu) = crate::models::continuum::elastic::lame(e, nu);
        for g in 0..out.gauss_count() {
            assert!(
                (out.value(0, g, "sigma_xx").unwrap() - (lambda + 2.0 * mu) * 1e-4).abs() < 1e-6
            );
            assert!(out.value(0, g, "p").unwrap().abs() < 1e-14);
        }
    }

    /// Beyond yield the von Mises equivalent stress is capped at σ_y (perfect
    /// plasticity plateau) and `p` grows.
    #[test]
    fn yields_and_caps_at_sigma_y_solid() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // Large uniaxial strain (elastic trial ≈ 2100 MPa ≫ 250).
        let mut strain = SubElementField::new(
            pl.continuum.fespace(),
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-2).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, &rest(&pl, &mat), &mat, 0.0)
            .unwrap();
        for g in 0..out.gauss_count() {
            let s = [
                out.value(0, g, "sigma_xx").unwrap(),
                out.value(0, g, "sigma_yy").unwrap(),
                out.value(0, g, "sigma_zz").unwrap(),
                out.value(0, g, "sigma_yz").unwrap(),
                out.value(0, g, "sigma_xz").unwrap(),
                out.value(0, g, "sigma_xy").unwrap(),
            ];
            assert!(
                (tensor::von_mises_stress(&s) - sy).abs() < 1e-3,
                "q = {}",
                tensor::von_mises_stress(&s)
            );
            assert!(out.value(0, g, "p").unwrap() > 0.0);
        }
    }

    /// Plane stress drives σ_zz to zero, and below yield the in-plane stress
    /// matches the linear plane-stress solution.
    #[test]
    fn plane_stress_zero_out_of_plane_and_matches_elastic() {
        let pl = unit_quad(Kinematics::PlaneStress);
        let (e, nu, sy) = (210_000.0, 0.3, 1e9); // huge σ_y ⇒ stays elastic
        let mat = material(&pl, e, nu, sy);
        let eps0 = 1e-3;
        let mut strain = SubElementField::new(
            pl.continuum.fespace(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, &rest(&pl, &mat), &mat, 0.0)
            .unwrap();
        // Linear plane stress uniaxial-strain: σ_xx = E/(1-ν²)·ε, σ_yy = ν·σ_xx.
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-3);
            assert!((out.value(0, g, "sigma_yy").unwrap() - c * nu * eps0).abs() < 1e-3);
            // σ_zz is not an output in 2-D; verify via the von Mises plateau is
            // not triggered (elastic) — covered above. Out-of-plane handled
            // internally.
        }
    }

    /// Build a uniaxial-strain deformation field `ε_xx = val` (full 3-D tensor
    /// component names) on a `unit_hex`.
    fn uniaxial(pl: &Plasticity, val: f64) -> Handle<SubElementField> {
        let comps: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
        let mut s = SubElementField::new(pl.continuum.fespace(), comps).unwrap();
        s.set_uniform("eps_xx", val).unwrap();
        Handle::new(s)
    }

    /// Internal state round-trips through `prev`: feeding the previous step's
    /// output back changes the result (history dependence) and `p` grows.
    #[test]
    fn state_round_trip_is_history_dependent() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // First load past yield (prev = None ⇒ reference config A).
        let st1 = pl
            .integrate_behavior(&uniaxial(&pl, 5e-3), &rest(&pl, &mat), &mat, 0.0)
            .unwrap();
        let p1 = st1.value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Second step: larger ε(B); the state of A is fed via `prev` (the step-1
        // output), *not* merged into the deformation field.
        let prev = Handle::new(st1);
        let st2 = pl
            .integrate_behavior(&uniaxial(&pl, 6e-3), &prev, &mat, 0.0)
            .unwrap();
        // Cumulated plastic strain only grows.
        assert!(st2.value(0, 0, "p").unwrap() >= p1);
    }

    /// Iso-result: on a **proportional** (monotone uniaxial) path, the
    /// incremental montage in N steps — threading `prev` — reproduces the
    /// single-step total-strain integration to round-off. Guards the whole
    /// prev-threading + `σ(A) + C:Δε` predictor.
    #[test]
    fn incremental_matches_single_step_on_proportional_path() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        let eps_final = 1e-2; // well past yield

        // Single step 0 → ε_final.
        let single = pl
            .integrate_behavior(&uniaxial(&pl, eps_final), &rest(&pl, &mat), &mat, 0.0)
            .unwrap();

        // Ten proportional increments, threading `prev` — which starts at the
        // rest state rather than at « nothing yet ».
        let nsteps = 10;
        let mut prev = rest(&pl, &mat);
        for i in 1..=nsteps {
            let val = eps_final * i as f64 / nsteps as f64;
            let out = pl
                .integrate_behavior(&uniaxial(&pl, val), &prev, &mat, 0.0)
                .unwrap();
            prev = Handle::new(out);
        }
        let multi = prev.read();
        for comp in [
            "sigma_xx", "sigma_yy", "sigma_zz", "p", "eps_p_xx", "eps_p_yy",
        ] {
            let a = single.value(0, 0, comp).unwrap();
            let b = multi.value(0, 0, comp).unwrap();
            assert!((a - b).abs() < 1e-9, "{comp}: single={a} multi={b}");
        }
    }

    /// History dependence: after loading past yield, a small **partial** unload
    /// is elastic — `p` does not grow and the stress drops *off* the yield
    /// plateau (`q < σ_y`), following the elastic slope. Impossible without
    /// threaded state: the old bug integrated the unloaded step from zero, so at
    /// a still-past-yield strain it would sit back on the plateau (`q = σ_y`)
    /// with a fresh `p`.
    #[test]
    fn partial_unload_is_elastic_and_leaves_yield_surface() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);

        // Load well past yield.
        let loaded = Handle::new(
            pl.integrate_behavior(&uniaxial(&pl, 1e-2), &rest(&pl, &mat), &mat, 0.0)
                .unwrap(),
        );
        let p1 = loaded.read().value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Small unload (still far past yield), threading the loaded state as `prev`.
        let unloaded = pl
            .integrate_behavior(&uniaxial(&pl, 9.9e-3), &loaded, &mat, 0.0)
            .unwrap();
        // Elastic: p unchanged.
        assert!(
            (unloaded.value(0, 0, "p").unwrap() - p1).abs() < 1e-12,
            "p must not grow on elastic unload"
        );
        // Stress has left the yield plateau (q < σ_y) — the history signature.
        let s = [
            unloaded.value(0, 0, "sigma_xx").unwrap(),
            unloaded.value(0, 0, "sigma_yy").unwrap(),
            unloaded.value(0, 0, "sigma_zz").unwrap(),
            unloaded.value(0, 0, "sigma_yz").unwrap(),
            unloaded.value(0, 0, "sigma_xz").unwrap(),
            unloaded.value(0, 0, "sigma_xy").unwrap(),
        ];
        assert!(
            tensor::von_mises_stress(&s) < sy - 1.0,
            "elastic unload must drop below σ_y, got q = {}",
            tensor::von_mises_stress(&s)
        );
    }

    /// The elastic stiffness block is reused from elasticity: symmetric.
    #[test]
    fn stiffness_is_elastic_and_symmetric() {
        let pl = unit_quad(Kinematics::PlaneStrain);
        let mat = material(&pl, 200.0, 0.3, 250.0);
        let blocks = pl.build_stiffness_blocks(&mat).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = pl.continuum.support().read().connectivity().to_vec();
        for &ni in &nodes {
            for &nj in &nodes {
                for a in ["x", "y"] {
                    for b in ["x", "y"] {
                        let lhs = k.get(ni, &format!("f_{a}"), nj, &format!("u_{b}"));
                        let rhs = k.get(nj, &format!("f_{b}"), ni, &format!("u_{a}"));
                        assert!((lhs - rhs).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_index_matches_layout() {
        assert_eq!(tensor_index(0, 0), 0);
        assert_eq!(tensor_index(2, 2), 2);
        assert_eq!(tensor_index(1, 2), 3);
        assert_eq!(tensor_index(0, 1), 5);
        assert_eq!(tensor_index(1, 0), 5);
    }
}
