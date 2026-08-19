//! Radiation to infinity — the Stefan-Boltzmann boundary.
//!
//! A surface exchanging with a distant environment at `T_∞` radiates
//!
//! ```text
//! q·n = σ · ε · (T⁴ − T_∞⁴)
//! ```
//!
//! `σ` is the Stefan-Boltzmann constant and `ε` the emissivity. Primal `"T"`,
//! dual `"q"` — the **same** DOFs as
//! [`heat_conduction`](crate::models::heat_conduction), so a radiating boundary
//! couples straight into the conduction stiffness, exactly like
//! [`boundary_transfer`](crate::models::boundary_transfer).
//!
//! ## What makes it different from convection: it is non-linear
//!
//! Newton's law of cooling is linear in `T`, so convection contributes a
//! constant film matrix and nothing else. `T⁴` is not, which is why this physics
//! declares **three** things rather than one:
//!
//! | term | expression | role |
//! |---|---|---|
//! | stiffness | `4σεT_∞³ ∫ N_i N_j dΓ` | the **linearised** radiative film — a constant operator, the classic `h_r` |
//! | internal force | `∫ N_i σε(T⁴ − T_∞⁴) dΓ` | the residual, exact |
//! | tangent | `4σεT³ ∫ N_i N_j dΓ` | the consistent tangent at the current temperature |
//!
//! Linearising the stiffness about `T_∞` rather than about the current state is
//! what lets it stay a plain constant matrix: it is the operator to start a
//! Newton loop from (and, on its own, a perfectly usable Picard iteration). The
//! **tangent** carries the real non-linearity, reading `T` back from the state
//! that the behaviour integration produced — the same producer/consumer pairing
//! as the plastic `D_alg`.
//!
//! ## Nature
//!
//! It declares **two** natures, `[Thermal, Radiation]`. A radiating boundary is
//! part of a thermal problem — `filter("thermal")` must return it — while
//! `filter("radiation")` isolates the non-linear term on its own, to assemble or
//! inspect it apart.
//!
//! ## Units
//!
//! `sigma` defaults to the SI Stefan-Boltzmann constant, and `T` is then an
//! **absolute** temperature (Kelvin) — a fourth power has no invariance to
//! translate an origin through. In another unit system, supply `sigma`
//! explicitly as a material component.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::owned_components;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature) — shared with heat conduction.
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux) — shared with heat conduction.
pub const DUAL_VAR: &str = "q";
/// Required material components: the emissivity and the far-field temperature.
/// `T_inf` is **required**, not optional, because the linearised stiffness needs
/// it — there is no radiative operator without a reference temperature.
const MATERIAL_COMPONENTS: &[&str] = &["emis", "T_inf"];
/// Accepted but not required: the Stefan-Boltzmann constant, to override the SI
/// default when working in another unit system.
const OPTIONAL_COMPONENTS: &[&str] = &["sigma"];

/// The Stefan-Boltzmann constant, in SI (`W·m⁻²·K⁻⁴`).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Behaviour-**input** component: the temperature interpolated at the Gauss
/// points (via [`crate::ops::element_field::interp_to_gauss`]).
const INPUT_COMPONENT: &str = PRIMAL_VAR;
/// Behaviour-**output**: the radiated flux density, and the tangent coefficient
/// `4σεT³` that [`SubModelKind::element_tangent`] reads back.
const OUTPUT_FLUX: &str = "flux";
const OUTPUT_TANGENT: &str = "ktan";

/// Radiation to infinity on a boundary FE subspace.
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
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::radiation::Radiation;
/// // Rayonnement vers l'infini, sur un bord. Mêmes DDL que la conduction,
/// // d'où un couplage direct dans sa raideur.
/// let r = Radiation::new(zone.clone())?;
/// assert_eq!(r.primal_vars(), vec!["T".to_string()]);
/// assert!(r.material_components().unwrap().contains(&"emis".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Radiation {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the boundary's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
}

impl Radiation {
    /// Radiation physics on a **boundary** FE subspace (an edge mesh in 2-D, a
    /// surface mesh in 3-D). Like convection, it needs no normal: the direction
    /// is already consumed in writing `q·n = σε(T⁴ − T_∞⁴)`, and what remains
    /// under the integral is a scalar times the surface measure.
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
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::radiation::Radiation;
    /// // Rayonnement vers l'infini, sur un bord. Mêmes DDL que la conduction,
    /// // d'où un couplage direct dans sa raideur.
    /// let r = Radiation::new(zone.clone())?;
    /// assert_eq!(r.primal_vars(), vec!["T".to_string()]);
    /// assert!(r.material_components().unwrap().contains(&"emis".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let submesh = fespace.read().submesh();
        let support = submesh.read().to_poi1()?;
        Ok(Self { fespace, support })
    }

    fn layout(&self) -> MatrixLayout {
        MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        }
    }
}

impl SubModelKind for Radiation {
    fn primal_vars(&self) -> Vec<String> {
        vec![PRIMAL_VAR.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![DUAL_VAR.to_string()]
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(self.layout())
    }

    /// The consistent tangent shares the layout — same boundary, same DOFs; only
    /// the coefficient differs (`4σεT³` instead of `4σεT_∞³`).
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        Some(self.layout())
    }

    /// The **linearised** radiative film, about the far-field temperature:
    /// `h_r = 4σεT_∞³`, a constant. See the module docs for why the
    /// linearisation is taken there rather than at the current state.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Radiation declares a material_fespace");
        surface_mass(geom, ke, |g| {
            let (sigma, emis) = (
                sigma_of(mat, geom.cell, g)?,
                mat.value(geom.cell, g, "emis")?,
            );
            let t_inf = mat.value(geom.cell, g, "T_inf")?;
            Ok(4.0 * sigma * emis * t_inf.powi(3))
        })
    }

    /// The consistent tangent `4σεT³ ∫ N_i N_j`, reading the coefficient from
    /// the state the behaviour integration produced.
    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let st = state.expect("the radiation tangent requires the current state field");
        surface_mass(geom, ke, |g| st.value(geom.cell, g, OUTPUT_TANGENT))
    }

    /// Internal nodal fluxes `q_i = ∫ N_i · flux dΓ` — weighted by `N`, not by
    /// `Bᵀ`, as for convection: the integrand is a flux **density** on the
    /// boundary, not a gradient-conjugate quantity.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g)?;
            let w = geom.det_j_w(g)?;
            let flux = stress.value(geom.cell, g, OUTPUT_FLUX)?;
            for i in 0..geom.n_nodes {
                fe[i] += shape[i] * flux * w;
            }
        }
        Ok(())
    }

    /// Both natures: a radiating boundary belongs to the thermal problem **and**
    /// is selectable on its own.
    fn physics(&self) -> &'static [Physics] {
        &[Physics::Thermal, Physics::Radiation]
    }

    fn label(&self) -> &'static str {
        "Radiation"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Radiation>\n  primal var(s): {PRIMAL_VAR}\n  \
             dual var(s):   {DUAL_VAR}\n  support: {n} node(s)"
        )
    }
}

impl Domain for Radiation {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(MATERIAL_COMPONENTS))
    }

    fn optional_material_components(&self) -> &'static [&'static str] {
        OPTIONAL_COMPONENTS
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec![OUTPUT_FLUX.to_string(), OUTPUT_TANGENT.to_string()])
    }

    /// `σε(T⁴ − T_∞⁴)` and its derivative `4σεT³`, at one Gauss point. Emitting
    /// both in one pass is the producer half of the tangent pairing: the kernel
    /// that knows the law is the one that differentiates it.
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
        let mat = material.expect("Radiation declares a material_fespace");
        let cell = geom.cell;
        let sigma = sigma_of(mat, cell, g)?;
        let emis = mat.value(cell, g, "emis")?;
        let t_inf = mat.value(cell, g, "T_inf")?;
        let t = input.value(cell, g, INPUT_COMPONENT)?;
        out[0] = sigma * emis * (t.powi(4) - t_inf.powi(4));
        out[1] = 4.0 * sigma * emis * t.powi(3);
        Ok(())
    }
}

/// The Stefan-Boltzmann constant for this cell: the material's own if it carries
/// one, the SI value otherwise.
fn sigma_of(material: &SubElementField, cell: usize, g: usize) -> Result<f64> {
    Ok(match material.component_index("sigma") {
        Some(_) => material.value(cell, g, "sigma")?,
        None => STEFAN_BOLTZMANN,
    })
}

/// `∫_Γ coeff(g) · N_i N_j dΓ` over one boundary cell — the surface mass matrix
/// weighted by a per-Gauss coefficient. Both radiative operators are this
/// integral; only the coefficient differs, so they share it.
fn surface_mass(
    geom: &CellGeom,
    ke: &mut [f64],
    coeff: impl Fn(usize) -> Result<f64>,
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    for g in 0..geom.n_gauss {
        let shape = geom.n_at_g(g)?;
        let w = geom.det_j_w(g)? * coeff(g)?;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                ke[i * n_nodes + j] += shape[i] * shape[j] * w;
            }
        }
    }
    Ok(())
}
