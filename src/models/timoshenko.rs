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
//! [`crate::ops::field::beam_deformation`].

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::{
    Interpolation, QuadratureRule, SubFiniteElementSpace,
};
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{ElementType, NodeId, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::Physics;
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

impl Physics for Timoshenko {
    fn primal_vars(&self) -> Vec<String> {
        PRIMAL.iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        DUAL.iter().map(|s| s.to_string()).collect()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.bending.clone())
    }

    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let mat = material.expect("Timoshenko requires a material field");
        let mut block = SubMatrix::new(
            self.support.clone(),
            self.support.clone(),
            self.dual_vars(),
            self.primal_vars(),
            DofOrdering::NodesThenVars,
            true,
        )?;
        assemble_stiffness(&self.bending, &self.shear, mat, &mut block)?;
        Ok(vec![block])
    }

    fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.bending.clone())
    }

    fn integrate_behavior(
        &self,
        input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<SubElementField> {
        let mat = material.expect("Timoshenko declares a material_fespace ⇒ material is supplied");
        let (n_cells, n_g) = {
            let f = read(input)?;
            (f.cell_count(), f.gauss_count())
        };
        let mut out =
            SubElementField::new(self.bending.clone(), vec!["M".to_string(), "V".to_string()])?;
        let f = read(input)?;
        let m = read(mat)?;
        for cell in 0..n_cells {
            let ei = m.value(cell, 0, "E")? * m.value(cell, 0, "I")?;
            let gas = m.value(cell, 0, "G")? * m.value(cell, 0, "A_s")?;
            for g in 0..n_g {
                out.set(cell, g, 0, ei * f.value(cell, g, "kappa")?)?; // M = E·I·κ
                out.set(cell, g, 1, gas * f.value(cell, g, "gamma")?)?; // V = G·A_s·γ
            }
        }
        Ok(out)
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

/// Per-cell, per-Gauss integration data of a SEG2 subspace: the two nodal
/// derivatives `dN/dx`, the two nodal values `N`, and `|J|·w`.
struct GaussData {
    dn: Vec<[f64; 2]>, // [g] = [dN_0/dx, dN_1/dx]
    n: Vec<[f64; 2]>,  // [g] = [N_0, N_1]
    det_j_w: Vec<f64>,
}

/// Snapshot one SEG2 subspace into per-cell [`GaussData`] (+ connectivity).
fn gauss_data(fespace: &Handle<SubFiniteElementSpace>) -> Result<(Vec<GaussData>, Vec<NodeId>)> {
    let s = read(fespace)?;
    let n_cells = s.cell_count()?;
    let n_g = s.gauss_count();
    let conn: Vec<NodeId> = read(&s.submesh())?.connectivity().to_vec();
    let mut out = Vec::with_capacity(n_cells);
    for cell in 0..n_cells {
        let mut dn = Vec::with_capacity(n_g);
        let mut n = Vec::with_capacity(n_g);
        let mut det_j_w = Vec::with_capacity(n_g);
        for g in 0..n_g {
            let d = s.dn_dx(cell, g)?; // [dN_0/dx, dN_1/dx] (1-D)
            let sh = s.n_at_g(g)?; // [N_0, N_1]
            dn.push([d[0], d[1]]);
            n.push([sh[0], sh[1]]);
            det_j_w.push(s.det_jacobian(cell, g)? * s.gauss_weight(g)?);
        }
        out.push(GaussData { dn, n, det_j_w });
    }
    Ok((out, conn))
}

/// Assemble `K_b + K_s` of every beam element into `k`.
///
/// Bending strain `θ' = B_b·d` with `B_b = [0, dN_0, 0, dN_1]`; shear strain
/// `γ = w' − θ = B_s·d` with `B_s = [dN_0, −N_0, dN_1, −N_1]`, where the DOF
/// vector is `d = [w_0, θ_0, w_1, θ_1]`.
pub fn assemble_stiffness(
    bending: &Handle<SubFiniteElementSpace>,
    shear: &Handle<SubFiniteElementSpace>,
    material: &Handle<SubElementField>,
    k: &mut SubMatrix,
) -> Result<()> {
    let (bend, conn) = gauss_data(bending)?;
    let (shr, _) = gauss_data(shear)?;
    let n_cells = bend.len();

    let (eis, gas): (Vec<f64>, Vec<f64>) = {
        let m = read(material)?;
        let mut eis = Vec::with_capacity(n_cells);
        let mut gas = Vec::with_capacity(n_cells);
        for cell in 0..n_cells {
            eis.push(m.value(cell, 0, "E")? * m.value(cell, 0, "I")?);
            gas.push(m.value(cell, 0, "G")? * m.value(cell, 0, "A_s")?);
        }
        (eis, gas)
    };

    for cell in 0..n_cells {
        let nodes = [conn[2 * cell], conn[2 * cell + 1]];
        let mut ke = [[0.0_f64; 4]; 4];

        // Bending: B_b = [0, dN_0, 0, dN_1], coefficient E·I (full Gauss).
        for (g, &[dn0, dn1]) in bend[cell].dn.iter().enumerate() {
            let bb = [0.0, dn0, 0.0, dn1];
            let coef = eis[cell] * bend[cell].det_j_w[g];
            accumulate(&mut ke, &bb, coef);
        }
        // Shear: B_s = [dN_0, −N_0, dN_1, −N_1], coefficient G·A_s (reduced).
        for g in 0..shr[cell].dn.len() {
            let [dn0, dn1] = shr[cell].dn[g];
            let [n0, n1] = shr[cell].n[g];
            let bs = [dn0, -n0, dn1, -n1];
            let coef = gas[cell] * shr[cell].det_j_w[g];
            accumulate(&mut ke, &bs, coef);
        }

        // Scatter the 4×4 element matrix: DOF k ↔ (nodes[k/2], var[k%2]).
        for r in 0..4 {
            for c in 0..4 {
                k.add_entry(
                    nodes[r / 2],
                    DUAL[r % 2],
                    nodes[c / 2],
                    PRIMAL[c % 2],
                    ke[r][c],
                )?;
            }
        }
    }
    Ok(())
}

/// `Ke += coef · (B ⊗ B)`.
fn accumulate(ke: &mut [[f64; 4]; 4], b: &[f64; 4], coef: f64) {
    for r in 0..4 {
        for c in 0..4 {
            ke[r][c] += coef * b[r] * b[c];
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
    use crate::containers::mesh::{Configuration, Mesh, Node};
    use crate::store::insert;

    /// One SEG2 beam of length `L`, returns `(timoshenko, n0, n1, L)`.
    fn one_element(length: f64) -> (Timoshenko, NodeId, NodeId) {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
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
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
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
        assert!(
            (k.get(n0, "f_w", n0, "theta") - k.get(n0, "m_theta", n0, "w")).abs() < tol
        );
    }
}
