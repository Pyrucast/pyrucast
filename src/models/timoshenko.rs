//! Timoshenko beam physics — shear-deformable planar bending.
//!
//! A `SEG2` beam in a **1-D** configuration with two scalar DOFs per node: the
//! transverse deflection `w` and the section rotation `theta`. The stiffness is
//! the sum of a **bending** and a **shear** contribution,
//!
//! ```text
//! K_b = ∫ E·I (θ')²       dx     (full Gauss integration)
//! K_s = ∫ G·A_s (w' − θ)² dx     (reduced 1-point integration)
//! ```
//!
//! integrated on **two FE subspaces over the same mesh** — a full-Gauss one for
//! bending and a reduced (1-point) one for shear. Reduced integration of the
//! shear term is what prevents **shear locking** for slender beams.
//!
//! Primal `w, theta`; dual `f_w` (transverse force conjugate to `w`) and
//! `m_theta` (moment conjugate to `theta`). Material components `E`, `I`, `G`,
//! `A_s` (the shear area `κ·A`).
//!
//! The behaviour (`COMP`) returns the **section forces** `M = E·I·κ`,
//! `V = G·A_s·γ` from the section strains `(κ, γ)` produced by
//! [`crate::ops::field::beam_deformation`](fn@crate::ops::field::beam_deformation).

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::{
    Interpolation, QuadratureRule, SubFiniteElementSpace,
};
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::{ElementType, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, Domain, Physics, StiffnessLayout, SubModelKind};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};

/// Material components required by the Timoshenko beam.
const MATERIAL_COMPONENTS: &[&str] = &["E", "I", "G", "A_s"];
/// Primal DOF names (deflection, rotation).
const PRIMAL: [&str; 2] = ["w", "theta"];
/// Dual DOF names (transverse force, moment).
const DUAL: [&str; 2] = ["f_w", "m_theta"];

/// Timoshenko beam physics on a 1-D `SEG2` FE subspace.
///
/// Holds **two** subspaces over the same mesh: `bending` (the given full-Gauss
/// space) and `shear` (a reduced 1-point space built on the fly). Material
/// (`E`, `I`, `G`, `A_s`) is supplied at assembly time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Timoshenko {
    /// Full-Gauss subspace, used for the bending term and as material support.
    pub(crate) bending: Handle<SubFiniteElementSpace>,
    /// Reduced (1-point) subspace over the same mesh, used for the shear term.
    pub(crate) shear: Handle<SubFiniteElementSpace>,
    /// POI1 support over the unique nodes (row/col support of the block).
    pub(crate) support: Handle<SubMesh>,
}

impl Timoshenko {
    /// Timoshenko beam on a 1-D `SEG2` FE subspace (full Gauss). Builds the
    /// reduced shear subspace and the POI1 node support. Errors unless the
    /// subspace is `SEG2` in a 1-D configuration.
    pub fn new(bending: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, et) = {
            let s = read(&bending)?;
            (s.submesh(), s.space_dim(), s.element_type()?)
        };
        if et != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "Timoshenko: expected SEG2 elements, got {et}"
            )));
        }
        if space_dim != 1 {
            return Err(PyrucastError::Message(format!(
                "Timoshenko: pure planar (w, θ) beam requires a 1-D configuration, got {space_dim}-D"
            )));
        }
        let shear = insert(SubFiniteElementSpace::new(
            submesh.clone(),
            Interpolation::Lagrange1,
            QuadratureRule::Reduced,
        )?);
        let support = insert(read(&submesh)?.to_poi1()?);
        Ok(Self {
            bending,
            shear,
            support,
        })
    }
}

impl SubModelKind for Timoshenko {
    fn primal_vars(&self) -> Vec<String> {
        PRIMAL.iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        DUAL.iter().map(|s| s.to_string()).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        // Two-quadrature element: bending (full Gauss) + shear (reduced), two FE
        // subspaces over the same mesh. The multi-fespace layout drives both the
        // computed (parallel scatter) and literal paths from this one description.
        Some(StiffnessLayout {
            fespaces: vec![self.bending.clone(), self.shear.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// `K_b + K_s` of one beam element. `geoms[0]` is the full-Gauss bending
    /// subspace, `geoms[1]` the reduced 1-point shear subspace (same cell, same
    /// nodes). DOF vector `d = [w_0, θ_0, w_1, θ_1]`; bending strain
    /// `θ' = B_b·d` with `B_b = [0, dN_0, 0, dN_1]`, shear strain
    /// `γ = w' − θ = B_s·d` with `B_s = [dN_0, −N_0, dN_1, −N_1]`.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Timoshenko declares a material_fespace ⇒ material is supplied");
        let (bend, shear) = (&geoms[0], &geoms[1]);
        let cell = bend.cell;
        let ei = mat.value(cell, 0, "E")? * mat.value(cell, 0, "I")?;
        let gas = mat.value(cell, 0, "G")? * mat.value(cell, 0, "A_s")?;

        // Bending: B_b = [0, dN_0, 0, dN_1], coefficient E·I (full Gauss).
        for g in 0..bend.n_gauss {
            let dn = bend.dn_dx(g)?; // [dN_0/dx, dN_1/dx] (1-D)
            let bb = [0.0, dn[0], 0.0, dn[1]];
            accumulate(ke, &bb, ei * bend.det_j_w(g)?);
        }
        // Shear: B_s = [dN_0, −N_0, dN_1, −N_1], coefficient G·A_s (reduced).
        for g in 0..shear.n_gauss {
            let dn = shear.dn_dx(g)?;
            let n = shear.n_at_g(g)?;
            let bs = [dn[0], -n[0], dn[1], -n[1]];
            accumulate(ke, &bs, gas * shear.det_j_w(g)?);
        }
        Ok(())
    }

    /// Internal forces `f = ∫ (B_bᵀ M + B_sᵀ V) dx` of one beam — the transpose
    /// of [`element_matrix`](Self::element_matrix), applied to the section forces
    /// `(M, V)` instead of the strains. `geoms[0]` is the full-Gauss bending
    /// space (bending moment `M`), `geoms[1]` the reduced shear space (shear
    /// force `V`). DOF vector `[f_w0, m_theta0, f_w1, m_theta1]`, with
    /// `B_b = [0, dN_0, 0, dN_1]` and `B_s = [dN_0, −N_0, dN_1, −N_1]`. `V` is
    /// element-constant (read at the first Gauss point).
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let (bend, shear) = (&geoms[0], &geoms[1]);
        let cell = bend.cell;
        // Bending: B_b = [0, dN_0, 0, dN_1] · M (full Gauss).
        for g in 0..bend.n_gauss {
            let dn = bend.dn_dx(g)?; // [dN_0/dx, dN_1/dx] (1-D)
            let bb = [0.0, dn[0], 0.0, dn[1]];
            let mw = stress.value(cell, g, "M")? * bend.det_j_w(g)?;
            for (k, b) in bb.iter().enumerate() {
                fe[k] += b * mw;
            }
        }
        // Shear: B_s = [dN_0, −N_0, dN_1, −N_1] · V (reduced).
        let v = stress.value(cell, 0, "V")?; // element-constant
        for g in 0..shear.n_gauss {
            let dn = shear.dn_dx(g)?;
            let n = shear.n_at_g(g)?;
            let bs = [dn[0], -n[0], dn[1], -n[1]];
            let vw = v * shear.det_j_w(g)?;
            for (k, b) in bs.iter().enumerate() {
                fe[k] += b * vw;
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Timoshenko"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Timoshenko>\n  primal var(s): w, theta\n  dual var(s):   f_w, m_theta\n  \
             support: {n} node(s) (bending: full Gauss, shear: reduced)"
        )
    }
}

impl Domain for Timoshenko {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.bending.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.bending.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec!["M".to_string(), "V".to_string()])
    }

    /// Section forces at one Gauss point: `M = E·I·κ`, `V = G·A_s·γ`.
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
        let mat = material.expect("Timoshenko declares a material_fespace ⇒ material is supplied");
        let cell = geom.cell;
        let ei = mat.value(cell, 0, "E")? * mat.value(cell, 0, "I")?;
        let gas = mat.value(cell, 0, "G")? * mat.value(cell, 0, "A_s")?;
        out[0] = ei * input.value(cell, g, "kappa")?;
        out[1] = gas * input.value(cell, g, "gamma")?;
        Ok(())
    }
}

/// `Ke += coef · (B ⊗ B)` on the flat 4×4 element matrix (row-major,
/// node-major / variable-minor: DOF `k ↔ (node k/2, var k%2)`).
fn accumulate(ke: &mut [f64], b: &[f64; 4], coef: f64) {
    for r in 0..4 {
        for c in 0..4 {
            ke[r * 4 + c] += coef * b[r] * b[c];
        }
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, Mesh, Node, NodeId};
    use crate::store::insert;

    /// One SEG2 beam of length `L`, returns `(timoshenko, n0, n1, L)`.
    fn one_element(length: f64) -> (Timoshenko, NodeId, NodeId) {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let beam = Timoshenko::new(fes.get(0).unwrap()).unwrap();
        (beam, a.id(), b.id())
    }

    fn material(beam: &Timoshenko, e: f64, i: f64, g: f64, a_s: f64) -> Handle<SubElementField> {
        let mut m = SubElementField::new(
            beam.bending.clone(),
            vec!["E".into(), "I".into(), "G".into(), "A_s".into()],
        )
        .unwrap();
        m.set_uniform("E", e).unwrap();
        m.set_uniform("I", i).unwrap();
        m.set_uniform("G", g).unwrap();
        m.set_uniform("A_s", a_s).unwrap();
        insert(m)
    }

    #[test]
    fn vars_and_construction() {
        let (beam, ..) = one_element(2.0);
        assert_eq!(beam.primal_vars(), vec!["w", "theta"]);
        assert_eq!(beam.dual_vars(), vec!["f_w", "m_theta"]);
        // 2-D config is rejected (pure planar beam needs 1-D).
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert!(Timoshenko::new(fes.get(0).unwrap()).is_err());
    }

    /// Element stiffness matches the analytical reduced-integration matrix:
    /// `Kb = EI/L` on the (θ,θ) block, `Ks = GA_s · B_sᵀB_s · L`.
    #[test]
    fn element_stiffness_matches_analytical() {
        let (l, ei, gas) = (2.0, 6.0, 10.0);
        let (e, i, g, a_s) = (3.0, 2.0, 5.0, 2.0); // E·I = 6, G·A_s = 10
        let (beam, n0, n1) = one_element(l);
        let mat = material(&beam, e, i, g, a_s);
        let blocks = beam.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let tol = 1e-9;

        // Bending block (θ,θ): EI/L on the diagonal, −EI/L coupling.
        assert!((k.get(n0, "m_theta", n0, "theta") - (ei / l + gas * l / 4.0)).abs() < tol);
        assert!((k.get(n0, "m_theta", n1, "theta") - (-ei / l + gas * l / 4.0)).abs() < tol);
        // Shear: (w,w) = GA_s/L, (w,θ) coupling = GA_s/2 (sign per B_s).
        assert!((k.get(n0, "f_w", n0, "w") - gas / l).abs() < tol);
        assert!((k.get(n0, "f_w", n1, "w") + gas / l).abs() < tol);
        assert!((k.get(n0, "f_w", n0, "theta") - gas / 2.0).abs() < tol);
        assert!((k.get(n1, "f_w", n1, "theta") + gas / 2.0).abs() < tol);
        // Symmetry.
        assert!((k.get(n0, "f_w", n0, "theta") - k.get(n0, "m_theta", n0, "w")).abs() < tol);
    }
}
