//! Generalised section strains of an oriented `SEG2` beam (frame) at the Gauss
//! points — the co-rotational counterpart of
//! [`crate::ops::field::beam_deformation`](fn@crate::ops::field::beam_deformation)
//! for the [`Frame`](crate::models::frame::Frame) (2-D) and
//! [`Frame3d`](crate::models::frame3d::Frame3d) (3-D) physics.
//!
//! Where `beam_deformation` works on a 1-D `(w, θ)` beam already aligned with
//! the axis, a frame element is oriented arbitrarily in space and carries the
//! **full** displacement + rotation at each node. This operator therefore:
//!
//! 1. builds the element's **local axes** (`x'` along the beam; `y'`/`z'` from
//!    the same automatic reference as the stiffness kernels), and
//! 2. rotates the two nodal displacement/rotation triples into that frame,
//!    then
//! 3. evaluates the generalised strains from the **local** DOFs, exactly as the
//!    Timoshenko section strains: axial `ε = u'`, curvature `κ = θ'` and
//!    **reduced** shear `γ = w' − θ` (shear rotation taken at the element
//!    centre to avoid shear locking).
//!
//! All strains are **element-constant** (linear `SEG2`), written at every Gauss
//! point. The output components are, per physics:
//!
//! | physics  | components (in order)                                    |
//! |----------|----------------------------------------------------------|
//! | 2-D frame| `eps, kappa, gamma`                                       |
//! | 3-D frame| `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z`       |
//!
//! Feed the result to [`crate::ops::behavior::integrate`]; the frame physics
//! turns it into the section forces (`N = E·A·ε`, `M = E·I·κ`, `V = G·A_s·γ`,
//! …).

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::Field;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::mesh::Coords;
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};

/// Output component names of the 2-D frame section strains.
const COMPONENTS_2D: &[&str] = &["eps", "kappa", "gamma"];
/// Output component names of the 3-D frame section strains.
const COMPONENTS_3D: &[&str] = &["eps", "kappa_y", "kappa_z", "torsion", "gamma_y", "gamma_z"];
/// Displacement + rotation DOFs read from the nodal field, per space dim.
const DOFS_2D: &[&str] = &["u_x", "u_y", "rz"];
const DOFS_3D: &[&str] = &["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];

/// Generalised section strains of a frame displacement/rotation `field` at the
/// Gauss points of every subspace of `fespace`.
///
/// The field must carry the frame DOFs of the configuration: `u_x, u_y, rz`
/// in 2-D, or `u_x, u_y, u_z, r_x, r_y, r_z` in 3-D. Each subspace must be
/// oriented `SEG2` in that same configuration (as required by the frame
/// physics).
pub fn frame_deformation(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(field)?;
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        out.add_sub(insert(subspace_frame_deformation(sub, &view, &components)?))?;
    }
    Ok(out)
}

/// Section strains on one oriented `SEG2` subspace. Dispatches on the
/// configuration's spatial dimension (2-D vs 3-D frame).
fn subspace_frame_deformation(
    fespace: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    components: &[String],
) -> Result<SubElementField> {
    let (space_dim, n_nodes, n_g, dofs, out_comps) = {
        let s = read(fespace)?;
        let space_dim = s.space_dim();
        let (dofs, out_comps): (&[&str], &[&str]) = match space_dim {
            2 => (DOFS_2D, COMPONENTS_2D),
            3 => (DOFS_3D, COMPONENTS_3D),
            d => {
                return Err(PyrucastError::Message(format!(
                "frame_deformation: frame element requires a 2-D or 3-D configuration, got {d}-D"
            )))
            }
        };
        (
            space_dim,
            s.nodes_per_cell()?,
            s.gauss_count(),
            dofs,
            out_comps,
        )
    };
    if n_nodes != 2 {
        return Err(PyrucastError::Message(format!(
            "frame_deformation: frame element must be SEG2 (2 nodes/cell), got {n_nodes}"
        )));
    }
    // The field must carry every frame DOF up front (a clearer error than a
    // missing-component failure deep in the per-cell loop).
    for required in dofs {
        if !components.iter().any(|c| c == required) {
            return Err(PyrucastError::Message(format!(
                "frame_deformation: the field must carry a '{required}' component (has: [{}])",
                components.join(", ")
            )));
        }
    }

    let submesh = read(fespace)?.submesh();
    let conn = read(&submesh)?.connectivity().to_vec();
    let coords_h: Handle<Coords> = read(&submesh)?.coords();
    let coords = read(&coords_h)?;

    let n_comp = out_comps.len();
    let mut field = SubElementField::new(
        fespace.clone(),
        out_comps.iter().map(|s| s.to_string()).collect(),
    )?;
    let n_cells = conn.len() / n_nodes;
    for cell in 0..n_cells {
        let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
        // Nodal DOFs and coordinates of the two endpoints.
        let xa = coords.coord(ids[0])?;
        let xb = coords.coord(ids[1])?;
        let da: Vec<f64> = dofs
            .iter()
            .map(|c| view.value(ids[0], c))
            .collect::<Result<_>>()?;
        let db: Vec<f64> = dofs
            .iter()
            .map(|c| view.value(ids[1], c))
            .collect::<Result<_>>()?;

        let strains = if space_dim == 2 {
            strains_2d(xa, xb, &da, &db)
        } else {
            strains_3d(xa, xb, &da, &db)
        };

        for g in 0..n_g {
            for (c, &v) in strains.iter().enumerate() {
                field.set(cell, g, c, v)?;
            }
        }
        debug_assert_eq!(strains.len(), n_comp);
    }
    Ok(field)
}

/// 2-D frame section strains `(eps, kappa, gamma)` in the element's local
/// frame, from the endpoint coordinates and the nodal DOFs `[u_x, u_y, rz]`.
///
/// Local kinematics on the length-`L` element (`u' = axial displ., w' =
/// transverse displ., θ = rotation`): axial `ε = (u'_B − u'_A)/L`, curvature
/// `κ = (θ_B − θ_A)/L`, reduced shear `γ = (w'_B − w'_A)/L − (θ_A + θ_B)/2`.
fn strains_2d(xa: &[f64], xb: &[f64], da: &[f64], db: &[f64]) -> Vec<f64> {
    let (dx, dy) = (xb[0] - xa[0], xb[1] - xa[1]);
    let l = (dx * dx + dy * dy).sqrt();
    let (c, s) = (dx / l, dy / l);
    // Rotate the in-plane displacement into local axial (u') / transverse (w').
    let ua = c * da[0] + s * da[1];
    let wa = -s * da[0] + c * da[1];
    let ub = c * db[0] + s * db[1];
    let wb = -s * db[0] + c * db[1];
    let (ta, tb) = (da[2], db[2]); // rotation is frame-invariant in 2-D

    let eps = (ub - ua) / l;
    let kappa = (tb - ta) / l;
    let gamma = (wb - wa) / l - 0.5 * (ta + tb);
    vec![eps, kappa, gamma]
}

/// 3-D frame section strains
/// `(eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z)` in the element's local
/// frame, from the endpoint coordinates and the nodal DOFs
/// `[u_x, u_y, u_z, r_x, r_y, r_z]`.
///
/// The local axes match the [`Frame3d`](crate::models::frame3d) stiffness
/// kernel (`x'` along the beam, `y'`/`z'` from an automatic global reference).
/// With local displacement `(u', v', w')` and rotation `(θx', θy', θz')`:
/// axial `ε = u'_,x`, torsion `= θx'_,x`, curvatures `κ_y = θy'_,x`,
/// `κ_z = θz'_,x`, reduced shears `γ_y = v'_,x − θz'`, `γ_z = w'_,x + θy'`
/// (centre rotations).
fn strains_3d(xa: &[f64], xb: &[f64], da: &[f64], db: &[f64]) -> Vec<f64> {
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    let r = local_axes(d); // rows: x', y', z' in global coords
    let ua = rotate(&r, &da[0..3]);
    let ra = rotate(&r, &da[3..6]);
    let ub = rotate(&r, &db[0..3]);
    let rb = rotate(&r, &db[3..6]);

    let eps = (ub[0] - ua[0]) / l; // u'_,x
    let torsion = (rb[0] - ra[0]) / l; // θx'_,x
    let kappa_y = (rb[1] - ra[1]) / l; // θy'_,x
    let kappa_z = (rb[2] - ra[2]) / l; // θz'_,x
                                       // Reduced shear (centre rotation): γ_y = v'_,x − θz', γ_z = w'_,x + θy'.
    let gamma_y = (ub[1] - ua[1]) / l - 0.5 * (ra[2] + rb[2]);
    let gamma_z = (ub[2] - ua[2]) / l + 0.5 * (ra[1] + rb[1]);
    vec![eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z]
}

// ─── 3-D geometry helpers (mirror `models::frame3d`) ─────────────────────────

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
/// direction `d` — identical to the [`Frame3d`](crate::models::frame3d)
/// stiffness kernel so strains and forces share one orientation convention.
fn local_axes(d: [f64; 3]) -> [[f64; 3]; 3] {
    let x = normalize(d);
    let z_ref = if x[2].abs() > 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let y = normalize(cross(z_ref, x));
    let z = cross(x, y);
    [x, y, z]
}

/// `R · v`: express the global 3-vector `v` in the local axes `R`.
fn rotate(r: &[[f64; 3]; 3], v: &[f64]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::store::insert;

    /// Build a one-cell frame FE space between two nodes plus a `(w,θ)`/DOF
    /// node field, returning `(fespace, node_a, node_b, field)`. `space_dim`
    /// selects the 2-D or 3-D DOF set.
    fn one_element(
        space_dim: u8,
        a: &[f64],
        b: &[f64],
    ) -> (FiniteElementSpace, Node, Node, Handle<SubMesh>) {
        let coords = insert(Coords::new(space_dim).unwrap());
        let na = Node::create_in(coords.clone(), a).unwrap();
        let nb = Node::create_in(coords.clone(), b).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[na.id(), nb.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let support = insert(SubMesh::poi1_from_nodes(&[na.clone(), nb.clone()]).unwrap());
        (fes, na, nb, support)
    }

    /// Horizontal 2-D element (local = global): pure axial stretch, pure
    /// curvature and pure reduced shear map straight through.
    #[test]
    fn horizontal_2d_strains() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(2, &[0.0, 0.0], &[l, 0.0]);
        let mut f =
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into(), "rz".into()])
                .unwrap();
        // u_x = x·0.5 (ε = 0.5), rz = x·0.25 (κ = 0.25), u_y = 0 (γ = −θ_centre).
        f.set_value(a.id(), "u_x", 0.0).unwrap();
        f.set_value(b.id(), "u_x", 0.5 * l).unwrap();
        f.set_value(a.id(), "rz", 0.0).unwrap();
        f.set_value(b.id(), "rz", 0.25 * l).unwrap();
        let f = NodeField::from_sub(f);

        let def = frame_deformation(&f, &fes).unwrap();
        let s = read(&def.get(0).unwrap()).unwrap();
        assert_eq!(s.components(), &["eps", "kappa", "gamma"]);
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "eps").unwrap() - 0.5).abs() < 1e-12);
            assert!((s.value(0, g, "kappa").unwrap() - 0.25).abs() < 1e-12);
            // w' = 0, θ_centre = (0 + 0.5)/2 = 0.25 ⇒ γ = −0.25.
            assert!((s.value(0, g, "gamma").unwrap() + 0.25).abs() < 1e-12);
        }
    }

    /// Vertical 2-D element (α = 90°): a global-y displacement is the *axial*
    /// stretch, a global-x displacement is the transverse (shear) one.
    #[test]
    fn vertical_2d_strains_are_rotated() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(2, &[0.0, 0.0], &[0.0, l]);
        let mut f =
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into(), "rz".into()])
                .unwrap();
        // Axial along global y.
        f.set_value(a.id(), "u_y", 0.0).unwrap();
        f.set_value(b.id(), "u_y", 0.5 * l).unwrap();
        let f = NodeField::from_sub(f);

        let def = frame_deformation(&f, &fes).unwrap();
        let s = read(&def.get(0).unwrap()).unwrap();
        assert!((s.value(0, 0, "eps").unwrap() - 0.5).abs() < 1e-12);
        // No rotation, no transverse displacement ⇒ κ = γ = 0.
        assert!(s.value(0, 0, "kappa").unwrap().abs() < 1e-12);
        assert!(s.value(0, 0, "gamma").unwrap().abs() < 1e-12);
    }

    /// Horizontal 3-D X-beam (local = global): axial on u_x, torsion on r_x,
    /// curvatures on r_y/r_z, shears on u_y/u_z.
    #[test]
    fn horizontal_3d_strains() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(3, &[0.0, 0.0, 0.0], &[l, 0.0, 0.0]);
        let dofs: Vec<String> = DOFS_3D.iter().map(|s| s.to_string()).collect();
        let mut f = SubNodeField::from_poi1(&support, dofs).unwrap();
        f.set_value(a.id(), "u_x", 0.0).unwrap();
        f.set_value(b.id(), "u_x", 0.5 * l).unwrap(); // ε = 0.5
        f.set_value(a.id(), "r_x", 0.0).unwrap();
        f.set_value(b.id(), "r_x", 0.3 * l).unwrap(); // torsion = 0.3
        f.set_value(a.id(), "r_z", 0.0).unwrap();
        f.set_value(b.id(), "r_z", 0.2 * l).unwrap(); // κ_z = 0.2
        let f = NodeField::from_sub(f);

        let def = frame_deformation(&f, &fes).unwrap();
        let s = read(&def.get(0).unwrap()).unwrap();
        assert_eq!(
            s.components(),
            &["eps", "kappa_y", "kappa_z", "torsion", "gamma_y", "gamma_z"]
        );
        assert!((s.value(0, 0, "eps").unwrap() - 0.5).abs() < 1e-12);
        assert!((s.value(0, 0, "torsion").unwrap() - 0.3).abs() < 1e-12);
        assert!((s.value(0, 0, "kappa_z").unwrap() - 0.2).abs() < 1e-12);
        assert!(s.value(0, 0, "kappa_y").unwrap().abs() < 1e-12);
        // v' = 0 but θz' at centre = (0 + 0.4)/2 = 0.2 ⇒ γ_y = −θz'_centre.
        assert!((s.value(0, 0, "gamma_y").unwrap() + 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_missing_dof() {
        let (fes, _a, _b, support) = one_element(2, &[0.0, 0.0], &[1.0, 0.0]);
        // Missing the rotation `rz`.
        let f = NodeField::from_sub(
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap(),
        );
        let err = frame_deformation(&f, &fes).unwrap_err();
        assert!(format!("{err}").contains("must carry a 'rz'"));
    }
}
