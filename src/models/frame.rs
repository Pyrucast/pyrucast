//! Planar frame (portique) physics — an oriented Timoshenko beam carrying
//! **axial + bending + shear**.
//!
//! A `SEG2` element in a 2-D configuration with **three** DOFs per node: the
//! in-plane displacement `u_x, u_y` and the rotation `rz`. The local stiffness
//! combines a truss axial term `E·A/L`, a bending term `E·I` and a
//! **reduced**-shear term `G·A_s` (`B_s` evaluated at the element centre, like
//! the [`Timoshenko`](crate::models::timoshenko) beam), then is rotated to the
//! global frame by `K = Tᵀ K_loc T` where `T` is built from the element's
//! direction cosines — so any orientation in the plane works.
//!
//! Primal `u_x, u_y, rz`; dual `f_x, f_y, m_z`. Material components `E`, `A`,
//! `I`, `G`, `A_s`. This v0 assembles the stiffness only.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::{ElementType, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, Domain, Physics, StiffnessLayout, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

const MATERIAL_COMPONENTS: &[&str] = &["E", "A", "I", "G", "A_s"];
/// Primal DOF names (in-plane displacement + rotation).
const PRIMAL: [&str; 3] = ["u_x", "u_y", "rz"];
/// Dual DOF names (in-plane force + moment).
const DUAL: [&str; 3] = ["f_x", "f_y", "m_z"];

/// Planar frame physics on a 2-D `SEG2` FE subspace.
///
/// Material data (`E`, `A`, `I`, `G`, `A_s`) is supplied at assembly time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Frame {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
}

impl Frame {
    /// Frame physics on a 2-D `SEG2` FE subspace. Errors unless the subspace is
    /// `SEG2` in a 2-D configuration.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, et) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim(), s.element_type()?)
        };
        if et != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "Frame: expected SEG2 elements, got {et}"
            )));
        }
        if space_dim != 2 {
            return Err(PyrucastError::Message(format!(
                "Frame: planar frame requires a 2-D configuration, got {space_dim}-D"
            )));
        }
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self { fespace, support })
    }
}

impl SubModelKind for Frame {
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
        element_stiffness(geom, material.expect("Frame requires a material field"), ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Frame"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Frame>\n  primal var(s): u_x, u_y, rz\n  dual var(s):   f_x, f_y, m_z\n  \
             support: {n} node(s)"
        )
    }
}

impl Domain for Frame {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// Section forces `N` (axial), `M` (bending), `V` (shear) — the linear law
    /// on the generalised strains `(eps, kappa, gamma)` produced by
    /// [`crate::ops::field::frame_deformation`](fn@crate::ops::field::frame_deformation).
    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec!["N".into(), "M".into(), "V".into()])
    }

    /// `N = E·A·ε`, `M = E·I·κ`, `V = G·A_s·γ` at one Gauss point, from the
    /// local section strains `(eps, kappa, gamma)`.
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
        let mat = material.expect("Frame declares a material_fespace ⇒ material is supplied");
        let cell = geom.cell;
        let ea = mat.value(cell, 0, "E")? * mat.value(cell, 0, "A")?;
        let ei = mat.value(cell, 0, "E")? * mat.value(cell, 0, "I")?;
        let gas = mat.value(cell, 0, "G")? * mat.value(cell, 0, "A_s")?;
        out[0] = ea * input.value(cell, g, "eps")?;
        out[1] = ei * input.value(cell, g, "kappa")?;
        out[2] = gas * input.value(cell, g, "gamma")?;
        Ok(())
    }
}

/// Local 6×6 frame stiffness (DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`) from
/// `E·A`, `E·I`, `G·A_s` and length `L` — axial + bending + reduced shear.
fn local_stiffness(ea: f64, ei: f64, gas: f64, l: f64) -> [[f64; 6]; 6] {
    let mut k = [[0.0_f64; 6]; 6];
    // Axial (E·A/L) couples u'_A (0) and u'_B (3).
    let ka = ea / l;
    k[0][0] += ka;
    k[3][3] += ka;
    k[0][3] -= ka;
    k[3][0] -= ka;
    // Bending (E·I) couples θ_A (2) and θ_B (5): B_b = [θ'].
    let kb = ei / l;
    k[2][2] += kb;
    k[5][5] += kb;
    k[2][5] -= kb;
    k[5][2] -= kb;
    // Shear (G·A_s), reduced: γ = w' − θ, B_s over [w'_A, θ_A, w'_B, θ_B].
    let bs = [-1.0 / l, -0.5, 1.0 / l, -0.5];
    let idx = [1usize, 2, 4, 5];
    for a in 0..4 {
        for b in 0..4 {
            k[idx[a]][idx[b]] += gas * l * bs[a] * bs[b];
        }
    }
    k
}

/// Rotation `T` (6×6) mapping global DOFs to local: per node
/// `[[c, s, 0], [−s, c, 0], [0, 0, 1]]`.
fn rotation(c: f64, s: f64) -> [[f64; 6]; 6] {
    let mut t = [[0.0_f64; 6]; 6];
    for node in 0..2 {
        let o = node * 3;
        t[o][o] = c;
        t[o][o + 1] = s;
        t[o + 1][o] = -s;
        t[o + 1][o + 1] = c;
        t[o + 2][o + 2] = 1.0;
    }
    t
}

/// `A · B` for 6×6 matrices.
fn matmul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = 0.0;
            for k in 0..6 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Element kernel: local frame stiffness `K = Tᵀ K_loc T` of one 2-node frame
/// element, written into `ke` (flat row-major 6×6, **node-major / variable-minor**
/// dof order `dof = node·3 + var`). Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let cell = geom.cell;
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    let (dx, dy) = (xb[0] - xa[0], xb[1] - xa[1]);
    let l = (dx * dx + dy * dy).sqrt();
    let (c, s) = (dx / l, dy / l);
    let ea = material.value(cell, 0, "E")? * material.value(cell, 0, "A")?;
    let ei = material.value(cell, 0, "E")? * material.value(cell, 0, "I")?;
    let gas = material.value(cell, 0, "G")? * material.value(cell, 0, "A_s")?;

    let kl = local_stiffness(ea, ei, gas, l);
    let t = rotation(c, s);
    // K_global = Tᵀ · K_loc · T.
    let kg = matmul(&transpose(&t), &matmul(&kl, &t));

    for r in 0..6 {
        for col in 0..6 {
            ke[r * 6 + col] = kg[r][col];
        }
    }
    Ok(())
}

/// Transpose of a 6×6 matrix.
fn transpose(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = a[j][i];
        }
    }
    out
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

    fn one_frame(ax: f64, ay: f64, bx: f64, by: f64) -> (Frame, NodeId, NodeId) {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[ax, ay]).unwrap();
        let b = Node::create_in(coords.clone(), &[bx, by]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let frame = Frame::new(fes.get(0).unwrap()).unwrap();
        (frame, a.id(), b.id())
    }

    fn material(
        frame: &Frame,
        e: f64,
        area: f64,
        i: f64,
        g: f64,
        a_s: f64,
    ) -> Handle<SubElementField> {
        let mut m = SubElementField::new(
            frame.fespace.clone(),
            vec!["E".into(), "A".into(), "I".into(), "G".into(), "A_s".into()],
        )
        .unwrap();
        for (c, v) in [("E", e), ("A", area), ("I", i), ("G", g), ("A_s", a_s)] {
            m.set_uniform(c, v).unwrap();
        }
        insert(m)
    }

    /// Horizontal element (α = 0, T = I): axial along x, bending/shear in y.
    #[test]
    fn horizontal_frame_decouples_axial_and_bending() {
        let l = 2.0;
        let (e, area, i, g, a_s) = (3.0, 4.0, 2.0, 5.0, 2.0);
        let (ea, ei, gas) = (e * area, e * i, g * a_s);
        let (frame, a, b) = one_frame(0.0, 0.0, l, 0.0);
        let mat = material(&frame, e, area, i, g, a_s);
        let blocks = frame.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let tol = 1e-9;
        // Axial in x.
        assert!((k.get(a, "f_x", a, "u_x") - ea / l).abs() < tol);
        assert!((k.get(a, "f_x", b, "u_x") + ea / l).abs() < tol);
        // Shear / bending in (u_y, rz).
        assert!((k.get(a, "f_y", a, "u_y") - gas / l).abs() < tol);
        assert!((k.get(a, "m_z", a, "rz") - (ei / l + gas * l / 4.0)).abs() < tol);
        // Axial and bending are decoupled for a horizontal element.
        assert!(k.get(a, "f_x", a, "u_y").abs() < tol);
        assert!(k.get(a, "f_x", a, "rz").abs() < tol);
    }

    /// Vertical element (α = 90°): the rotation puts the axial term on `u_y`.
    #[test]
    fn vertical_frame_axial_is_along_y() {
        let l = 2.0;
        let (e, area, i, g, a_s) = (3.0, 4.0, 2.0, 5.0, 2.0);
        let (ea, gas) = (e * area, g * a_s);
        let (frame, a, _b) = one_frame(0.0, 0.0, 0.0, l);
        let mat = material(&frame, e, area, i, g, a_s);
        let blocks = frame.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let tol = 1e-9;
        // Axial now along y, shear along x.
        assert!((k.get(a, "f_y", a, "u_y") - ea / l).abs() < tol);
        assert!((k.get(a, "f_x", a, "u_x") - gas / l).abs() < tol);
        // Still decoupled (axial ⟂ bending).
        assert!(k.get(a, "f_y", a, "u_x").abs() < tol);
    }
}
