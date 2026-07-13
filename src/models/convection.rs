//! Surface convection (Robin / film) boundary — Newton's law of cooling.
//!
//! On a boundary `Γ`, the outward heat flux obeys `q·n = h·(T − T_ext)` (film
//! law, `h` the convection coefficient, `T_ext` the ambient temperature). Its
//! boundary term in the weak form of heat conduction,
//!
//! ```text
//! ∮_Γ (q·n) δT dΓ = ∮_Γ h·(T − T_ext) δT dΓ,
//! ```
//!
//! splits into a **film matrix** and an **external-temperature load**:
//!
//! ```text
//! K_ij = h ∫_Γ N_i N_j dΓ   (this sub-model, into the stiffness),
//! f_i  = h·T_ext ∫_Γ N_i dΓ (a right-hand side, built with
//!                            crate::ops::assemble::flux — not stored here).
//! ```
//!
//! Primal `"T"`, dual `"q"` — the **same** DOFs as
//! [`crate::models::heat_conduction::HeatConduction`], so a `Convection`
//! sub-model on a boundary mesh **couples straight into** the conduction
//! stiffness. The film coefficient `h` is read from the material
//! [`SubElementField`] component [`MATERIAL_COMPONENT`], exactly as conduction
//! reads `k`.
//!
//! **No normal is needed.** The normal is already consumed in passing from
//! `q·n` to `h·(T − T_ext)`; what remains under the integral is a scalar times
//! the surface measure `dΓ = |J|`, which
//! [`CellGeom::det_j_w`](crate::models::kernel::CellGeom::det_j_w) returns as
//! `√det(JᵀJ)` — a magnitude, invariant under the boundary mesh's orientation
//! (winding). Contrast a pressure or a signed flux, where the normal direction
//! matters.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::models::{CellGeom, Domain, Physics, StiffnessLayout, SubModelKind};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature) — shared with heat conduction, so the film
/// term couples into the same equations.
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux) — shared with heat conduction.
pub const DUAL_VAR: &str = "q";
/// Required material component on the [`SubElementField`]: the convection
/// (film) coefficient `h`.
pub const MATERIAL_COMPONENT: &str = "h";
/// Material contract returned by [`Domain::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Behaviour-**input** component: the temperature interpolated at the Gauss
/// points (e.g. via [`crate::ops::field::interp_to_gauss`]).
const INPUT_COMPONENT: &str = PRIMAL_VAR;
/// Behaviour-**output** component: the weak-form convective flux density
/// `h·T` (see [`Convection::integrate_point`]).
const OUTPUT_COMPONENT: &str = "flux";

/// Surface convection (Robin / film) on a boundary FE subspace.
///
/// - primal variable: `"T"` (temperature, columns).
/// - dual variable:   `"q"` (heat flux row labels).
/// - Material data (film coefficient `"h"`) is **not** stored here; it is
///   supplied at assembly time via [`crate::ops::assemble::stiffness`], read
///   from the boundary cells of the material field.
#[derive(Clone, Serialize, Deserialize)]
pub struct Convection {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh, built
    /// once at construction. Reused as the row/col support of every assembled
    /// film block — no per-assembly rebuild.
    pub(crate) support: Handle<SubMesh>,
}

impl Convection {
    /// Convection physics on a boundary FE subspace (an edge mesh in 2-D, a
    /// surface mesh in 3-D). Builds the stable POI1 [`SubMesh`] covering the
    /// subspace's unique nodes (reused as the row/col support of every
    /// assembled block).
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let submesh = read(&fespace)?.submesh();
        let support = insert(read(&submesh)?.to_poi1()?);
        Ok(Self { fespace, support })
    }
}

impl SubModelKind for Convection {
    fn primal_vars(&self) -> Vec<String> {
        vec![PRIMAL_VAR.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![DUAL_VAR.to_string()]
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        Some(StiffnessLayout {
            fespaces: vec![self.fespace.clone()],
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
        let geom = &geoms[0];
        element_film_matrix(
            geom,
            material.expect("Convection requires a material field"),
            ke,
        )
    }

    /// Internal nodal fluxes `q_i = ∫ N_i · (h·T) dΓ` of one cell — the
    /// **`N`-weighted** boundary counterpart of the `Bᵀ` continuum default
    /// (a film term integrates shape values, not their gradients). For the
    /// linear film law this equals `(K_conv · T)_i`, so it fits the
    /// « internal forces == K·u » invariant. Single dual variable `q`.
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
            let flux = stress.value(geom.cell, g, OUTPUT_COMPONENT)?;
            for i in 0..geom.n_nodes {
                fe[i] += shape[i] * flux * w;
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Thermal]
    }

    fn label(&self) -> &'static str {
        "Convection"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Convection>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

impl Domain for Convection {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec![OUTPUT_COMPONENT.to_string()])
    }

    /// Linear film law: weak-form convective flux density `flux = h·T` at one
    /// Gauss point, from the interpolated temperature (input component `"T"`).
    /// This is the quantity the assembled film matrix integrates
    /// (`∫ N_i·flux = (K_conv·T)_i`), mirroring how
    /// [`crate::models::heat_conduction`] outputs the weak-form `k·∇T`; the
    /// external-temperature part `h·T_ext` lives in the load, not here. No
    /// internal state (`VAR0`/`VAR1` empty).
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
        let mat = material.expect("Convection declares a material_fespace ⇒ material is supplied");
        let cell = geom.cell;
        let h = mat.value(cell, g, MATERIAL_COMPONENT)?;
        let t = input.value(cell, g, INPUT_COMPONENT)?;
        out[0] = h * t;
        Ok(())
    }
}

/// Element kernel: local film matrix of one boundary cell,
///   `K_local[i, j] = Σ_g h(g) · N_i(ξ_g) · N_j(ξ_g) · |J|_g · w_g`,
/// written into `ke` (flat row-major, side `n_nodes`, `ke[i * n_nodes + j]`).
/// A shape-weighted surface mass matrix scaled by the film coefficient `h`.
/// Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
pub fn element_film_matrix(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    for g in 0..geom.n_gauss {
        let shape = geom.n_at_g(g)?;
        let det_j_w = geom.det_j_w(g)?;
        let h = material.value(geom.cell, g, MATERIAL_COMPONENT)?;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                ke[i * n_nodes + j] += h * shape[i] * shape[j] * det_j_w;
            }
        }
    }
    Ok(())
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node};
    use crate::store::insert;

    /// Convection on a single SEG2 boundary "edge" of length `L` in a 2-D
    /// `Coords` (a 1-D-embedded edge integrates as a line via the manifold
    /// Jacobian).
    fn seg2_edge(length: f64) -> Convection {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Convection::new(fes.get(0).unwrap()).unwrap()
    }

    fn material(conv: &Convection, h: f64) -> Handle<SubElementField> {
        let mut m =
            SubElementField::new(conv.fespace.clone(), vec![MATERIAL_COMPONENT.to_string()])
                .unwrap();
        m.set_uniform(MATERIAL_COMPONENT, h).unwrap();
        insert(m)
    }

    #[test]
    fn vars_match_heat_conduction() {
        let conv = seg2_edge(1.0);
        assert_eq!(conv.primal_vars(), vec!["T"]);
        assert_eq!(conv.dual_vars(), vec!["q"]);
    }

    /// The film matrix of a SEG2 edge is `h` times the consistent line mass
    /// matrix `(L/6)·[[2,1],[1,2]]`; its symmetric and its full sum is `h·L`
    /// (`∫∫ h N_i N_j = h·L`).
    #[test]
    fn film_matrix_is_h_times_line_mass() {
        let (h, len) = (5.0, 2.0);
        let conv = seg2_edge(len);
        let mat = material(&conv, h);
        let blocks = conv.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let nodes: Vec<_> = read(&conv.support).unwrap().connectivity().to_vec();
        let (a, b) = (nodes[0], nodes[1]);
        let tol = 1e-12;
        // Consistent line mass: diagonal h·L/3, off-diagonal h·L/6.
        assert!((k.get(a, "q", a, "T") - h * len / 3.0).abs() < tol);
        assert!((k.get(b, "q", b, "T") - h * len / 3.0).abs() < tol);
        assert!((k.get(a, "q", b, "T") - h * len / 6.0).abs() < tol);
        // Symmetry.
        assert!((k.get(a, "q", b, "T") - k.get(b, "q", a, "T")).abs() < tol);
        // Full sum = h·L.
        let sum = k.get(a, "q", a, "T")
            + k.get(a, "q", b, "T")
            + k.get(b, "q", a, "T")
            + k.get(b, "q", b, "T");
        assert!((sum - h * len).abs() < tol);
    }

    /// COMP on the linear film law returns the weak-form flux density `h·T`.
    #[test]
    fn integrate_behavior_returns_weak_form_flux() {
        let (h, temp) = (3.0, 4.0);
        let conv = seg2_edge(1.0);
        let mat = material(&conv, h);

        let mut input =
            SubElementField::new(conv.fespace.clone(), vec![INPUT_COMPONENT.to_string()]).unwrap();
        input.set_uniform(INPUT_COMPONENT, temp).unwrap();
        let input = insert(input);

        let out = conv
            .integrate_behavior(&input, None, Some(&mat), None)
            .unwrap();
        assert_eq!(out.components(), &[OUTPUT_COMPONENT.to_string()]);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, OUTPUT_COMPONENT).unwrap() - h * temp).abs() < 1e-12);
        }
    }
}
