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
//! `prev` (the previous step's output; `None` on the first step, where A is the
//! reference configuration). The elastic predictor is `σ_trial = σ(A) + C:Δε`
//! with `Δε = ε(B) − ε(A)` — algebraically identical to `C:(ε(B) − ε_p(A))` in
//! small strain, but the form that carries `σ(A)` explicitly, ready for a
//! large-strain law. The output echoes the full-3-D `ε(B)` (and, in 2-D, the
//! out-of-plane `σ_zz`) so it is a complete `prev` for the next step.
//!
//! State is always carried in **full 3-D** (six `eps_p_*` components) regardless
//! of the 2-D/3-D model, which keeps the radial return identical across plane
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
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::owned_components;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use law::{MatParams, PlasticLaw, PrevState};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Full 3-D tensor component suffixes, in the internal state order
/// `[xx, yy, zz, yz, xz, xy]` (off-diagonals are **tensor** strains, `ε_ij`).
use crate::models::plasticity::law::TENSOR_SUFFIXES;
/// Index pairs `(i, j)` matching [`TENSOR_SUFFIXES`].
const TENSOR_PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// Where each **axisymmetric** Voigt slot `[rr, zz, θθ, rz]` sits in the full
/// 3-D order [`TENSOR_SUFFIXES`] (`[xx, yy, zz, yz, xz, xy]`). The whole
/// axisymmetric specialisation of this law is this one index map: the state and
/// the radial return stay full 3-D, only the projection in and out changes.
const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order for the given space dimension —
/// matching [`crate::models::elasticity`] so downstream code is uniform.
fn stress_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    if space_dim == 2 && model.is_axisymmetric() {
        // [rr, zz, θθ, rz] — the hoop is `zz`, Cast3M naming.
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_xy".into(),
        ]
    } else if space_dim == 2 {
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()]
    } else {
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ]
    }
}

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
fn echo_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
    if echoes_sigma_zz(space_dim, model) {
        v.push("sigma_zz".into());
    }
    v
}

/// Whether `σ_zz` must be echoed as extra state. Only the **plane** 2-D models
/// need it: their Voigt dual stops at `[xx, yy, xy]`. Axisymmetric already
/// carries `sigma_zz` (the hoop) in its dual, so echoing it would emit the same
/// component name twice.
fn echoes_sigma_zz(space_dim: usize, model: ElasticityModel) -> bool {
    space_dim == 2 && !model.is_axisymmetric()
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
/// # use pyrucast::models::elasticity::ElasticityModel;
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
/// let p = Plasticity::new(zone.clone(), ElasticityModel::PlaneStrain)?;
/// assert!(p.material_components().unwrap().contains(&"sigma_y".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Plasticity {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
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
    /// # use pyrucast::models::elasticity::ElasticityModel;
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
    /// let p = Plasticity::new(zone.clone(), ElasticityModel::PlaneStrain)?;
    /// assert_eq!(p.material_components().unwrap().len(), 3); // E, nu, sigma_y
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        Self::with_law(fespace, model, PlasticLaw::Perfect)
    }

    /// Elastoplasticity with an explicit yield law, on an FE subspace with the
    /// given 2-D/3-D model. Errors if `model` is inconsistent with the space
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
    /// # use pyrucast::models::elasticity::ElasticityModel;
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
    ///     zone.clone(), ElasticityModel::PlaneStrain, PlasticLaw::Isotropic)?;
    /// assert!(p.material_components().unwrap().contains(&"H".to_string()));
    /// // Un modèle solide sur une zone 2-D est refusé.
    /// assert!(Plasticity::with_law(
    ///     zone.clone(), ElasticityModel::Solid, PlasticLaw::Perfect).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_law(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
        law: PlasticLaw,
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
        elasticity::check_continuum_dimensions("Plasticity", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (2, ElasticityModel::Axisymmetric) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Plasticity: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ solid)"
            )));
        }
        // Same two-way agreement as `Elasticity::new`: the 2πr measure comes
        // from the geometry, the hoop component from the model.
        if axisymmetric != model.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Plasticity: model {model:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` model"
                )
            } else {
                "Plasticity: the `axisymmetric` model requires an axisymmetric geometry \
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
            law,
        })
    }
}

impl SubModelKind for Plasticity {
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

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // The plain "stiffness" kernel is the *elastic* stiffness (the simple
        // iteration operator). The consistent algorithmic tangent `K_t` is a
        // separate operator — see [`element_tangent`](Self::element_tangent) and
        // [`crate::ops::matrix::tangent`]. Reuse the elasticity element kernel
        // verbatim; it reads only `E` and `nu` from the material.
        let mat = material.expect("Plasticity requires a material field");
        elasticity::element_stiffness(
            geom,
            mat,
            self.model,
            crate::models::symmetry::MaterialSymmetry::Isotropic,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Plasticity requires a material field");
        elasticity::element_mass(geom, mat, ke)
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
        elasticity::element_geometric(geom, stress, ke)
    }

    /// The consistent tangent shares the stiffness layout; the algorithmic
    /// modulus `D_alg` (emitted by [`Domain::integrate_point`]) is read from the
    /// behaviour state.
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let st = state.expect("consistent tangent requires the behaviour state (D_alg)");
        elasticity::element_tangent_from_state(geom, st, self.model, ke)
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
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Plasticity({:?})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Plasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(self.law.material_components()))
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
        // They ride the optional channel so that a three-parameter model stays
        // writable in three numbers.
        self.law.as_law().optional_material_components()
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim, self.model);
        comps.extend(state_names());
        comps.extend(echo_names(self.space_dim, self.model));
        // The law's **own** internal variables (a back stress, a damage…), which
        // is how a law grows its state without any other file changing.
        comps.extend(self.law.internal_names());
        // Consistent algorithmic tangent D_alg (upper triangle) — consumed by
        // the tangent assembler (`crate::ops::matrix::tangent`).
        comps.extend(elasticity::tangent_component_names(
            self.space_dim,
            self.model,
        ));
        Ok(comps)
    }

    /// Incremental radial-return at one Gauss point. Output layout =
    /// stress (Voigt, `v`) + plastic strain `eps_p` (full 3-D tensor, 6) +
    /// cumulated plastic strain `p` (1) + echoed strain `ε(B)` (full 3-D, 6)
    /// [+ `sigma_zz` in 2-D], matching `stress_names ++ state_names ++ echo_names`.
    fn integrate_point(
        &self,
        geom: &CellGeom,
        deformation: &SubElementField,
        prev: Option<&SubElementField>,
        material: Option<&SubElementField>,
        g: usize,
        dt: Option<f64>,
        out: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Plasticity declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let params = MatParams::new(mat, cell)?;

        // End-of-step strain ε(B).
        let eps_b = read_strain(deformation, cell, g, d, self.model)?;
        // Converged state at A from `prev` (all zero on the first step, where A
        // is the reference configuration: σ(A)=0, ε(A)=0, ε_p(A)=0, p(A)=0).
        let prev_state = PrevState {
            eps: read_prev_strain(prev, cell, g),
            sigma: read_prev_stress(prev, cell, g),
            eps_p: read_prev_plastic_strain(prev, cell, g),
            p: prev_opt(prev, cell, g, "p"),
            // The law's own variables, in the order it declared them. Left
            // **empty** on the first step, where `prev` is `None`: a law whose
            // state starts from a material constant (Gurson's initial porosity)
            // must be able to tell « no state yet » from « state that is zero ».
            vars: match prev {
                None => Vec::new(),
                Some(_) => self
                    .law
                    .internal_names()
                    .iter()
                    .map(|n| prev_opt(prev, cell, g, n))
                    .collect(),
            },
        };

        let (step, eps_b_full) =
            self.law
                .as_law()
                .incremental_step(&eps_b, &prev_state, &params, self.model, dt)?;

        let v = stress_names(d, self.model).len();
        for r in 0..v {
            out[r] = voigt_stress(&step.sigma, d, self.model, r);
        }
        out[v..v + 6].copy_from_slice(&step.eps_p); // ε_p(B)
        out[v + 6] = step.p; // p(B)
                             // Echo the full-3-D end-of-step strain ε(B), so `prev` carries ε(A) next
                             // step (in plane stress this includes the solved out-of-plane ε_zz).
        out[v + 7..v + 13].copy_from_slice(&eps_b_full);
        // The plane 2-D duals omit σ_zz; echo it so σ(A) is fully recoverable.
        // Axisymmetric already carries it (the hoop), so it must not be echoed.
        let mut base = v + 13;
        if echoes_sigma_zz(d, self.model) {
            out[base] = step.sigma[2];
            base += 1;
        }
        // The law's own internal variables, right after the common state.
        for (i, value) in step.vars.iter().enumerate() {
            out[base + i] = *value;
        }
        base += step.vars.len();

        // Consistent tangent D_alg at the converged step, evaluated at the solved
        // ε(B) (which carries the plane-stress ε_zz). Emitted (upper triangle)
        // right after the state, in `ktan_i_j` order.
        let d3 = self
            .law
            .as_law()
            .consistent_tangent(&eps_b_full, &prev_state, &params, dt)?;
        let dv = crate::models::symmetry::reduce_to_model(&d3, self.model);
        let mut idx = base;
        for i in 0..dv.len() {
            for j in i..dv.len() {
                out[idx] = dv[i][j];
                idx += 1;
            }
        }
        Ok(())
    }
}

// The constitutive core — the elastic predictor, the return maps, the
// plane-stress secant loop and the consistent tangents — lives in
// [`crate::models::plasticity::law`], shared by every yield law. What remains here is
// the physics: the DOFs, the layouts, and the plumbing between the field
// components and the full-3-D state the laws work in.

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// Read a component, returning `0.0` when it is absent (first step has no state).
fn read_opt(f: &SubElementField, cell: usize, g: usize, name: &str) -> f64 {
    if f.component_index(name).is_some() {
        f.value(cell, g, name).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Reconstruct the full 3-D **tensor** strain from the deformation input.
/// Plane strain forces the out-of-plane components to zero; plane stress leaves
/// `eps_zz` as the trial elastic guess (it is overwritten by the return map).
fn read_strain(
    f: &SubElementField,
    cell: usize,
    g: usize,
    space_dim: usize,
    model: ElasticityModel,
) -> Result<[f64; 6]> {
    let mut eps = [0.0; 6];
    if space_dim == 2 {
        eps[0] = f.value(cell, g, "eps_xx")?;
        eps[1] = f.value(cell, g, "eps_yy")?;
        eps[5] = f.value(cell, g, "eps_xy")?;
        if model.is_axisymmetric() {
            // The hoop ε_θθ = u_r/r is **measured**, not assumed: `deformation`
            // produces it on a body of revolution. So ε(B) is fully known here
            // — no plane assumption, no out-of-plane solve.
            eps[2] = f.value(cell, g, "eps_zz")?;
        }
        // eps_yz/xz stay 0 (axial symmetry ⇒ no orthoradial shear); for the
        // plane models eps_zz also stays 0 (plane strain) or is solved later
        // (plane stress).
    } else {
        for (k, suf) in TENSOR_SUFFIXES.iter().enumerate() {
            eps[k] = f.value(cell, g, &format!("eps_{suf}"))?;
        }
    }
    Ok(eps)
}

/// Read a component from the optional previous-state field `prev`, defaulting to
/// `0.0` when there is no previous step (`None`) or the component is absent.
fn prev_opt(prev: Option<&SubElementField>, cell: usize, g: usize, name: &str) -> f64 {
    prev.map_or(0.0, |f| read_opt(f, cell, g, name))
}

/// Full 3-D strain `ε(A)` echoed by the previous step (zero on the first step).
fn read_prev_strain(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("eps_{}", TENSOR_SUFFIXES[k])))
}

/// Full 3-D stress `σ(A)` from the previous step. Each Voigt slot is read by
/// name: `sigma_zz` comes from the 2-D echo (or the 3-D dual), and the shear
/// `sigma_yz`/`sigma_xz` are absent in 2-D (⇒ `0.0`), exactly the plane
/// assumptions.
fn read_prev_stress(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("sigma_{}", TENSOR_SUFFIXES[k])))
}

/// Previous plastic strain tensor `ε_p(A)` (VAR0), defaulting to zero.
fn read_prev_plastic_strain(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("eps_p_{}", TENSOR_SUFFIXES[k])))
}

/// Project the full 3-D stress to the model's Voigt slot `r`.
/// 2-D order is `[xx, yy, xy]`; 3-D is the full `[xx, yy, zz, yz, xz, xy]`.
fn voigt_stress(sigma: &[f64; 6], space_dim: usize, model: ElasticityModel, r: usize) -> f64 {
    if space_dim == 2 && model.is_axisymmetric() {
        sigma[AXI_TO_3D[r]]
    } else if space_dim == 2 {
        match r {
            0 => sigma[0],
            1 => sigma[1],
            _ => sigma[5],
        }
    } else {
        sigma[r]
    }
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
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::models::tensor;

    fn unit_quad(model: ElasticityModel) -> Plasticity {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Plasticity::new(fes.get(0).unwrap(), model).unwrap()
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
        Plasticity::new(fes.get(0).unwrap(), ElasticityModel::Solid).unwrap()
    }

    fn material(pl: &Plasticity, e: f64, nu: f64, sy: f64) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            pl.fespace.clone(),
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
        let pl = unit_quad(ElasticityModel::PlaneStrain);
        assert_eq!(pl.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(pl.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(
            pl.material_components(),
            Some(owned_components(PlasticLaw::Perfect.material_components()))
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
            pl.fespace.clone(),
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-4).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        // Confined uniaxial *strain* (only ε_xx ≠ 0): σ_xx = (λ+2μ)·ε.
        let (lambda, mu) = elasticity::lame(e, nu);
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
            pl.fespace.clone(),
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-2).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
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
        let pl = unit_quad(ElasticityModel::PlaneStress);
        let (e, nu, sy) = (210_000.0, 0.3, 1e9); // huge σ_y ⇒ stays elastic
        let mat = material(&pl, e, nu, sy);
        let eps0 = 1e-3;
        let mut strain = SubElementField::new(
            pl.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = Handle::new(strain);
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
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
        let mut s = SubElementField::new(pl.fespace.clone(), comps).unwrap();
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
            .integrate_behavior(&uniaxial(&pl, 5e-3), None, Some(&mat), None)
            .unwrap();
        let p1 = st1.value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Second step: larger ε(B); the state of A is fed via `prev` (the step-1
        // output), *not* merged into the deformation field.
        let prev = Handle::new(st1);
        let st2 = pl
            .integrate_behavior(&uniaxial(&pl, 6e-3), Some(&prev), Some(&mat), None)
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
            .integrate_behavior(&uniaxial(&pl, eps_final), None, Some(&mat), None)
            .unwrap();

        // Ten proportional increments, threading `prev`.
        let nsteps = 10;
        let mut prev: Option<Handle<SubElementField>> = None;
        for i in 1..=nsteps {
            let val = eps_final * i as f64 / nsteps as f64;
            let out = pl
                .integrate_behavior(&uniaxial(&pl, val), prev.as_ref(), Some(&mat), None)
                .unwrap();
            prev = Some(Handle::new(out));
        }
        let multi = prev.unwrap().read();
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
            pl.integrate_behavior(&uniaxial(&pl, 1e-2), None, Some(&mat), None)
                .unwrap(),
        );
        let p1 = loaded.read().value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Small unload (still far past yield), threading the loaded state as `prev`.
        let unloaded = pl
            .integrate_behavior(&uniaxial(&pl, 9.9e-3), Some(&loaded), Some(&mat), None)
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
        let pl = unit_quad(ElasticityModel::PlaneStrain);
        let mat = material(&pl, 200.0, 0.3, 250.0);
        let blocks = pl.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = pl.support.read().connectivity().to_vec();
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
