//! 3-D Timoshenko frame (space frame) physics.
//!
//! A `SEG2` element in a 3-D configuration with **six** DOFs per node: the
//! displacement `u_x, u_y, u_z` and the rotation `r_x, r_y, r_z`. The local
//! 12×12 stiffness combines:
//!
//! - **axial** `E·A`,
//! - **torsion** `G·J` (about the beam axis),
//! - **bending** about the two principal axes (`E·I_z` with shear `G·A_sy`,
//!   `E·I_y` with shear `G·A_sz`), using the closed-form **Timoshenko** element
//!   (shear parameters `Φ`), which is nodally exact for end loads.
//!
//! It is rotated to the global frame by `K = Tᵀ K_loc T`. The section axes
//! (local `y'`, `z'`) are taken **automatically** from a global-Z reference
//! (global Y for a vertical beam), so no orientation data is needed — fine for
//! symmetric sections (`I_y = I_z`).
//!
//! Primal `u_x, u_y, u_z, r_x, r_y, r_z`; dual `f_x, f_y, f_z, m_x, m_y, m_z`.
//! Material components `E, A, I_y, I_z, J, G, A_sy, A_sz`. v0: stiffness only.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::{ElementType, SubMesh};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, HasMaterial, StiffnessLayout, SubModelKind};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};

const MATERIAL_COMPONENTS: &[&str] = &["E", "A", "I_y", "I_z", "J", "G", "A_sy", "A_sz"];
/// Primal DOF names (displacement + rotation), per node.
const PRIMAL: [&str; 6] = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];
/// Dual DOF names (force + moment), per node.
const DUAL: [&str; 6] = ["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"];

/// 3-D Timoshenko frame physics on a 3-D `SEG2` FE subspace.
///
/// Material data (`E, A, I_y, I_z, J, G, A_sy, A_sz`) is supplied at assembly
/// time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Frame3d {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
}

impl Frame3d {
    /// 3-D frame physics on a 3-D `SEG2` FE subspace. Errors unless the
    /// subspace is `SEG2` in a 3-D configuration.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, et) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim(), s.element_type()?)
        };
        if et != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "Frame3d: expected SEG2 elements, got {et}"
            )));
        }
        if space_dim != 3 {
            return Err(PyrucastError::Message(format!(
                "Frame3d: space frame requires a 3-D configuration, got {space_dim}-D"
            )));
        }
        let support = insert(read(&submesh)?.to_poi1()?);
        Ok(Self { fespace, support })
    }
}

impl SubModelKind for Frame3d {
    fn primal_vars(&self) -> Vec<String> {
        PRIMAL.iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        DUAL.iter().map(|s| s.to_string()).collect()
    }

    fn as_material(&self) -> Option<&dyn HasMaterial> {
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
        element_stiffness(
            geom,
            material.expect("Frame3d requires a material field"),
            ke,
        )
    }

    fn label(&self) -> &'static str {
        "Frame3d"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Frame3d>\n  primal var(s): u_x, u_y, u_z, r_x, r_y, r_z\n  \
             dual var(s):   f_x, f_y, f_z, m_x, m_y, m_z\n  support: {n} node(s)"
        )
    }
}

impl HasMaterial for Frame3d {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }
}

// ─── Geometry helpers ────────────────────────────────────────────────────────

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Local axes `R = [x'; y'; z']` (rows, in global coords) from the beam
/// direction `d`. `x'` is along the beam; `y'`/`z'` come from a global-Z
/// reference (global Y if the beam is ~vertical) — automatic orientation.
fn local_axes(d: [f64; 3]) -> [[f64; 3]; 3] {
    let x = normalize(d);
    let z_ref = if x[2].abs() > 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let y = normalize(cross(z_ref, x)); // ⟂ x' and ref
    let z = cross(x, y); // completes the right-handed triad
    [x, y, z]
}

/// Local 12×12 Timoshenko frame stiffness, DOF order per node
/// `[u', v', w', θx', θy', θz']`.
#[allow(clippy::too_many_arguments)]
fn local_stiffness(
    ea: f64,
    gj: f64,
    e: f64,
    iy: f64,
    iz: f64,
    g: f64,
    asy: f64,
    asz: f64,
    l: f64,
) -> [[f64; 12]; 12] {
    let mut k = [[0.0_f64; 12]; 12];
    let a = ea / l; // axial
    let t = gj / l; // torsion
                    // Shear parameters for the two bending planes.
    let phi_y = 12.0 * e * iz / (g * asy * l * l); // x'-y' plane (I_z, A_sy)
    let phi_z = 12.0 * e * iy / (g * asz * l * l); // x'-z' plane (I_y, A_sz)

    let kv1 = 12.0 * e * iz / (l * l * l * (1.0 + phi_y));
    let kv2 = 6.0 * e * iz / (l * l * (1.0 + phi_y));
    let kv3 = (4.0 + phi_y) * e * iz / (l * (1.0 + phi_y));
    let kv4 = (2.0 - phi_y) * e * iz / (l * (1.0 + phi_y));

    let kw1 = 12.0 * e * iy / (l * l * l * (1.0 + phi_z));
    let kw2 = 6.0 * e * iy / (l * l * (1.0 + phi_z));
    let kw3 = (4.0 + phi_z) * e * iy / (l * (1.0 + phi_z));
    let kw4 = (2.0 - phi_z) * e * iy / (l * (1.0 + phi_z));

    // Upper triangle (then mirrored). DOFs: u'=0/6, v'=1/7, w'=2/8,
    // θx'=3/9, θy'=4/10, θz'=5/11.
    k[0][0] = a;
    k[0][6] = -a;
    k[6][6] = a;
    k[3][3] = t;
    k[3][9] = -t;
    k[9][9] = t;
    // x'-y' plane: v' (1, 7), θz' (5, 11).
    k[1][1] = kv1;
    k[1][5] = kv2;
    k[1][7] = -kv1;
    k[1][11] = kv2;
    k[5][5] = kv3;
    k[5][7] = -kv2;
    k[5][11] = kv4;
    k[7][7] = kv1;
    k[7][11] = -kv2;
    k[11][11] = kv3;
    // x'-z' plane: w' (2, 8), θy' (4, 10) — sign-flipped coupling.
    k[2][2] = kw1;
    k[2][4] = -kw2;
    k[2][8] = -kw1;
    k[2][10] = -kw2;
    k[4][4] = kw3;
    k[4][8] = kw2;
    k[4][10] = kw4;
    k[8][8] = kw1;
    k[8][10] = kw2;
    k[10][10] = kw3;

    for i in 0..12 {
        for j in (i + 1)..12 {
            k[j][i] = k[i][j];
        }
    }
    k
}

/// Rotation `T` (12×12): block-diagonal repetition of `R` over the four DOF
/// triples `[u_A], [θ_A], [u_B], [θ_B]`.
fn rotation(r: &[[f64; 3]; 3]) -> [[f64; 12]; 12] {
    let mut t = [[0.0_f64; 12]; 12];
    for blk in 0..4 {
        let o = blk * 3;
        for i in 0..3 {
            for j in 0..3 {
                t[o + i][o + j] = r[i][j];
            }
        }
    }
    t
}

fn matmul(a: &[[f64; 12]; 12], b: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0_f64; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            let mut acc = 0.0;
            for k in 0..12 {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

fn transpose(a: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0_f64; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// Assemble `K = Tᵀ K_loc T` of every 3-D frame element into `k`, at
/// `(NodeId_i, dual_a) × (NodeId_j, primal_b)`.
/// Element kernel: local 3-D frame stiffness `K = Tᵀ K_loc T` of one 2-node
/// element, written into `ke` (flat row-major 12×12, **node-major /
/// variable-minor** dof order `dof = node·6 + var`). Pure and sequential —
/// driven in parallel by [`crate::models::kernel::assemble_block`].
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let cell = geom.cell;
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    // [E·A, G·J, E, I_y, I_z, G, A_sy, A_sz].
    let v = |c| material.value(cell, 0, c);
    let kl = local_stiffness(
        v("E")? * v("A")?,
        v("G")? * v("J")?,
        v("E")?,
        v("I_y")?,
        v("I_z")?,
        v("G")?,
        v("A_sy")?,
        v("A_sz")?,
        l,
    );
    let t = rotation(&local_axes(d));
    let kg = matmul(&transpose(&t), &matmul(&kl, &t)); // Tᵀ K_loc T

    for r in 0..12 {
        for c in 0..12 {
            ke[r * 12 + c] = kg[r][c];
        }
    }
    Ok(())
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

    fn one_beam(bx: f64, by: f64, bz: f64) -> (Frame3d, NodeId, NodeId) {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[bx, by, bz]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let frame = Frame3d::new(fes.get(0).unwrap()).unwrap();
        (frame, a.id(), b.id())
    }

    fn material(frame: &Frame3d) -> Handle<SubElementField> {
        let mut m = SubElementField::new(
            frame.fespace.clone(),
            MATERIAL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        for (c, v) in [
            ("E", 2.0),
            ("A", 3.0),
            ("I_y", 1.5),
            ("I_z", 2.5),
            ("J", 1.2),
            ("G", 0.8),
            ("A_sy", 4.0),
            ("A_sz", 5.0),
        ] {
            m.set_uniform(c, v).unwrap();
        }
        insert(m)
    }

    /// Horizontal X-beam (local = global): axial on x, torsion on r_x, the two
    /// bending planes decoupled from axial.
    #[test]
    fn horizontal_beam_axial_torsion_decoupled() {
        let l = 2.0;
        let (frame, a, b) = one_beam(l, 0.0, 0.0);
        let mat = material(&frame);
        let blocks = frame.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let tol = 1e-9;
        // E·A = 6, G·J = 0.96.
        assert!((k.get(a, "f_x", a, "u_x") - 6.0 / l).abs() < tol); // axial
        assert!((k.get(a, "f_x", b, "u_x") + 6.0 / l).abs() < tol);
        assert!((k.get(a, "m_x", a, "r_x") - 0.96 / l).abs() < tol); // torsion
                                                                     // Axial ⟂ everything else; the two bending planes are decoupled.
        assert!(k.get(a, "f_x", a, "u_y").abs() < tol);
        assert!(k.get(a, "f_x", a, "r_x").abs() < tol);
        assert!(k.get(a, "f_y", a, "u_z").abs() < tol);
        assert!(k.get(a, "f_y", a, "r_y").abs() < tol);
    }

    /// A beam along global Y: the rotation puts the axial term on `u_y`.
    #[test]
    fn vertical_beam_axial_is_along_y() {
        let l = 2.0;
        let (frame, a, _b) = one_beam(0.0, l, 0.0);
        let mat = material(&frame);
        let blocks = frame.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        assert!((k.get(a, "f_y", a, "u_y") - 6.0 / l).abs() < 1e-9); // axial on y
        assert!(k.get(a, "f_y", a, "u_x").abs() < 1e-9);
    }
}
