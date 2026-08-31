//! Generalised **section strains** of a beam, at the Gauss points — the
//! geometric producer of the behaviour input for
//! [`Timoshenko`](crate::models::timoshenko::Timoshenko) and
//! [`Bernoulli`](crate::models::bernoulli::Bernoulli).
//!
//! One operator for the three configurations, dispatching on the dimension of
//! the mesh exactly as the physics does. It was two — a `beam_deformation` for
//! the 1-D beam and a `beam_deformation` for the oriented ones — which was the
//! same irregularity, one layer down, as the three beam models it served.
//!
//! Per element it:
//!
//! 1. builds the element's **local axes** (`x'` along the beam; `y'`/`z'` from
//!    the same automatic reference as the stiffness kernels) — a no-op in 1-D,
//!    where there is nothing to rotate;
//! 2. rotates the nodal displacement/rotation into that frame; then
//! 3. evaluates the generalised strains from the **local** DOFs: axial
//!    `ε = u'`, curvature `κ = θ'`, and **reduced** shear `γ = w' − θ` with the
//!    rotation taken at the element centre.
//!
//! The **material** is required, `Φ = 12EI/(G·A_s·L²)` deciding how the
//! curvature is distributed. A material carrying no shear constants — a
//! [Bernoulli](crate::models::bernoulli) beam's, which asks for neither `G` nor
//! `A_s` — means `Φ = 0`, so the same operator serves both theories without
//! being told which. The material contract each declares already says it.
//!
//! | `Coords` | DOFs read | components produced |
//! |---|---|---|
//! | 1-D | `w, theta` | `kappa, gamma` |
//! | 2-D | `u_x, u_y, r_z` | `eps, kappa, gamma` |
//! | 3-D | six | `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z` |
//!
//! All strains are **element-constant**, written at every Gauss point. Sampling
//! the shear at the centre rather than at the full Gauss points avoids the
//! spurious oscillation that reduced integration exists to remove.
//!
//! > Since the beams moved to their **exact** closed form, the curvature of the
//! > real solution varies across the element while this recovery reports a mean.
//! > It is therefore an approximation owned by the formulation, not a
//! > consequence of a mis-declared basis — see
//! > [`crate::models::timoshenko`].
//!
//! Feed the result to [`crate::ops::element_field::behavior::integrate`]; the
//! beam physics turns it into the section forces.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::{Field, SubField};
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// Output component names of the 1-D beam section strains.
const COMPONENTS_1D: &[&str] = &["kappa", "gamma"];
/// Output component names of the 2-D frame section strains.
const COMPONENTS_2D: &[&str] = &["eps", "kappa", "gamma"];
/// Output component names of the 3-D frame section strains.
const COMPONENTS_3D: &[&str] = &["eps", "kappa_y", "kappa_z", "torsion", "gamma_y", "gamma_z"];
/// Displacement + rotation DOFs read from the nodal field, per space dim.
const DOFS_1D: &[&str] = &["w", "theta"];
const DOFS_2D: &[&str] = &["u_x", "u_y", "r_z"];
const DOFS_3D: &[&str] = &["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];

/// Generalised section strains of a frame displacement/rotation `field` at the
/// Gauss points of every subspace of `fespace`.
///
/// The field must carry the frame DOFs of the configuration: `u_x, u_y, rz`
/// in 2-D, or `u_x, u_y, u_z, r_x, r_y, r_z` in 3-D. Each subspace must be
/// oriented `SEG2` in that same configuration (as required by the frame
/// physics).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{element_field, mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::new(&maillage, Interpolation::ModelEmbedded)?;
/// # let support = mesh::poi1_from_nodes(&n)?;
/// # let u = NodeField::from_submesh(&support.get(0)?,
/// #     vec!["u_x".into(), "u_y".into(), "r_z".into()])?;
/// # let mut mat = ElementField::new(&fes,
/// #     vec!["E".into(), "A".into(), "I".into(), "G".into(), "A_s".into()])?;
/// # for c in ["E", "A", "I", "G", "A_s"] { mat.get(0)?.write().set_uniform(c, 1.0)?; }
/// // Les déformations **généralisées** d'une poutre : allongement,
/// // courbure, distorsion — non un tenseur, la section étant réduite à
/// // trois nombres.
/// u.get(0)?.write().set_value(n[1].id(), "u_x", 1.0)?;
/// let d = element_field::beam_deformation(&u, &fes, &mat)?;
/// assert_eq!(d.get(0)?.read().components(),
///            &["eps".to_string(), "kappa".to_string(), "gamma".to_string()]);
/// // Un allongement de 1 sur une portée de 2 : ε = 0,5, et rien d'autre.
/// assert!((d.get(0)?.read().value(0, 0, "eps")? - 0.5).abs() < 1e-12);
/// assert!(d.get(0)?.read().value(0, 0, "kappa")?.abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn beam_deformation(
    field: &NodeField,
    fespace: &FiniteElementSpace,
    material: &ElementField,
) -> Result<ElementField> {
    let components = Field::components(field);
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        let mat = material.sub_for_fespace(sub)?;
        out.add_sub(Handle::new(subspace_beam_deformation(
            sub,
            &view,
            &components,
            &mat,
        )?))?;
    }
    Ok(out)
}

/// Section strains on one oriented `SEG2` subspace. Dispatches on the
/// configuration's spatial dimension (2-D vs 3-D frame).
fn subspace_beam_deformation(
    fespace: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    components: &[String],
    material: &Handle<SubElementField>,
) -> Result<SubElementField> {
    // One read guard on the subspace, held for every property we read off it.
    let s = fespace.read();
    let space_dim = s.space_dim();
    let (dofs, out_comps): (&[&str], &[&str]) = match space_dim {
        1 => (DOFS_1D, COMPONENTS_1D),
        2 => (DOFS_2D, COMPONENTS_2D),
        3 => (DOFS_3D, COMPONENTS_3D),
        d => {
            return Err(PyrucastError::Message(format!(
                "beam_deformation: a beam lives in a 1-, 2- or 3-D configuration, got {d}-D"
            )))
        }
    };
    let n_nodes = s.nodes_per_cell()?;
    let n_g = s.gauss_count();
    if n_nodes != 2 {
        return Err(PyrucastError::Message(format!(
            "beam_deformation: a beam must be SEG2 (2 nodes/cell), got {n_nodes}"
        )));
    }
    // The field must carry every frame DOF up front (a clearer error than a
    // missing-component failure deep in the per-cell loop).
    for required in dofs {
        if !components.iter().any(|c| c == required) {
            return Err(PyrucastError::Message(format!(
                "beam_deformation: the field must carry a '{required}' component (has: [{}])",
                components.join(", ")
            )));
        }
    }

    // Hold the mesh guard over the whole loop and read the connectivity in
    // place (sequential loop ⇒ no need to copy it out).
    let mat = material.read();
    let submesh = s.submesh();
    let mesh = submesh.read();
    let conn = mesh.connectivity();
    let coords_h: Handle<Coords> = mesh.coords();
    let coords = coords_h.read();

    let n_comp = out_comps.len();
    let mut field = SubElementField::new(
        fespace.clone(),
        out_comps.iter().map(|s| s.to_string()).collect(),
    )?;
    // Les constantes de section, situées **une fois pour la zone** : un plan de
    // flexion en 1-D/2-D, deux en 3-D. Elles étaient cherchées par nom à chaque
    // point de Gauss de chaque maille.
    let slots: Vec<BendingSlots> = if space_dim == 3 {
        vec![
            BendingSlots::resolve(&mat, "I_z", "A_sy")?,
            BendingSlots::resolve(&mat, "I_y", "A_sz")?,
        ]
    } else {
        vec![BendingSlots::resolve(&mat, "I", "A_s")?]
    };

    let n_cells = conn.len() / n_nodes;
    for cell in 0..n_cells {
        let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
        // La ligne matériau de la maille : elle ne change pas d'un point de
        // Gauss au suivant.
        let row = mat.row(cell, 0);
        // Nodal DOFs and coordinates of the two endpoints.
        let xa = coords.position(ids[0])?;
        let xb = coords.position(ids[1])?;
        let da: Vec<f64> = dofs
            .iter()
            .map(|c| view.value(ids[0], c))
            .collect::<Result<_>>()?;
        let db: Vec<f64> = dofs
            .iter()
            .map(|c| view.value(ids[1], c))
            .collect::<Result<_>>()?;

        // Evaluated **at each Gauss point** from the element's own
        // interpolation: the curvature varies along an unloaded span (`M' = V`)
        // and only the shear is constant (`V' = 0`). The previous recovery
        // reported one element-constant value for both, because a linear
        // element has nothing else to report.
        for g in 0..n_g {
            let xi = s.gauss_xi(g)?[0];
            // SEG2's reference runs over [-1, 1]; the beam's own functions over
            // [0, 1].
            let t = 0.5 * (xi + 1.0);
            let strains = match space_dim {
                1 => strains_1d(&slots, row, xa, xb, &da, &db, t)?,
                2 => strains_2d(&slots, row, xa, xb, &da, &db, t)?,
                _ => strains_3d(&slots, row, xa, xb, &da, &db, t)?,
            };
            debug_assert_eq!(strains.len(), n_comp);
            for (c, &v) in strains.iter().enumerate() {
                field.set(cell, g, c, v)?;
            }
        }
    }
    Ok(field)
}

/// Where a bending plane's section constants sit in the material row, resolved
/// **once for the zone**.
///
/// A material carrying **no shear constants** has `Φ = 0`, and that is not a
/// fallback: each theory declares its own material contract, and
/// [Bernoulli](crate::models::bernoulli) deliberately asks for neither `G` nor
/// `A_s`. The absence *is* the statement that there is no shear compliance — so
/// this operator needs no model to tell the two theories apart, the material it
/// is handed having already said which.
///
/// Saying it in the layout rather than in the kernel matters twice: the question
/// is a fact of the zone, and it used to be answered by **swallowing an error**
/// — any failure at all, a wrong index included, read as « no shear ».
#[derive(Clone, Copy)]
struct BendingSlots {
    e: usize,
    i: usize,
    /// `(G, A_s)`, or `None` when the theory declares no shear compliance.
    shear: Option<(usize, usize)>,
}

impl BendingSlots {
    fn resolve(mat: &SubElementField, i_name: &str, a_s_name: &str) -> Result<Self> {
        let need = |name: &str| -> Result<usize> {
            mat.component_index(name).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "beam_deformation: the material carries no `{name}`"
                ))
            })
        };
        Ok(Self {
            e: need("E")?,
            i: need(i_name)?,
            shear: mat.component_index("G").zip(mat.component_index(a_s_name)),
        })
    }

    /// `Φ`, the ratio of bending to shear compliance, from this cell's row.
    fn phi(&self, row: &[f64], l: f64) -> f64 {
        let ei = row[self.e] * row[self.i];
        let gas = self.shear.map(|(g, a_s)| row[g] * row[a_s]);
        crate::models::beam::phi(ei, gas, l)
    }
}

/// 1-D beam section strains `(kappa, gamma)` at `t = x/L`, from the nodal DOFs
/// `[w, theta]`.
///
/// There is no local frame to build and no axial term: the axis *is* the mesh,
/// and a pure-bending beam has no direction to stretch along.
#[allow(clippy::too_many_arguments)]
fn strains_1d(
    slots: &[BendingSlots],
    row: &[f64],
    xa: &[f64],
    xb: &[f64],
    da: &[f64],
    db: &[f64],
    t: f64,
) -> Result<Vec<f64>> {
    let l = (xb[0] - xa[0]).abs();
    let ph = slots[0].phi(row, l);
    let d = [da[0], da[1], db[0], db[1]];
    let (kappa, gamma) = crate::models::beam::section_strains(ph, l, &d, t);
    Ok(vec![kappa, gamma])
}

/// 2-D frame section strains `(eps, kappa, gamma)` in the element's local
/// frame, at `t = x/L`, from the nodal DOFs `[u_x, u_y, r_z]`.
///
/// The axial strain stays **element-constant**, and correctly so: the axial
/// field of a bar really is linear.
#[allow(clippy::too_many_arguments)]
fn strains_2d(
    slots: &[BendingSlots],
    row: &[f64],
    xa: &[f64],
    xb: &[f64],
    da: &[f64],
    db: &[f64],
    t: f64,
) -> Result<Vec<f64>> {
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
    let ph = slots[0].phi(row, l);
    let (kappa, gamma) = crate::models::beam::section_strains(ph, l, &[wa, ta, wb, tb], t);
    Ok(vec![eps, kappa, gamma])
}

/// 3-D frame section strains
/// `(eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z)` in the element's local
/// frame, at `t = x/L`, from the nodal DOFs `[u_x, u_y, u_z, r_x, r_y, r_z]`.
///
/// The local axes match the [`frame3d`](crate::models::frame3d) stiffness
/// kernel (`x'` along the beam, `y'`/`z'` from an automatic global reference).
/// The two bending planes each carry their **own** `Φ`, through `A_sy` and
/// `A_sz`; the axial and the torsion stay element-constant, their fields being
/// genuinely linear.
#[allow(clippy::too_many_arguments)]
fn strains_3d(
    slots: &[BendingSlots],
    row: &[f64],
    xa: &[f64],
    xb: &[f64],
    da: &[f64],
    db: &[f64],
    t: f64,
) -> Result<Vec<f64>> {
    let d = [xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]];
    let l = norm(d);
    let r = local_axes(d); // rows: x', y', z' in global coords
    let ua = rotate(&r, &da[0..3]);
    let ra = rotate(&r, &da[3..6]);
    let ub = rotate(&r, &db[0..3]);
    let rb = rotate(&r, &db[3..6]);

    let eps = (ub[0] - ua[0]) / l; // u'_,x
    let torsion = (rb[0] - ra[0]) / l; // θx'_,x

    // x'-y' plane: deflection v', rotation θz' — the pair maps straight onto the
    // (w, θ) of the shared block, giving `κ_z = θz'_,x` and `γ_y = v'_,x − θz'`.
    let phi_y = slots[0].phi(row, l);
    let (kappa_z, gamma_y) =
        crate::models::beam::section_strains(phi_y, l, &[ua[1], ra[2], ub[1], rb[2]], t);
    // x'-z' plane: the rotation's sign is opposite — a positive θy' bends
    // towards −z — so it is fed negated. The curvature comes back negated with
    // it; the shear does not, `γ_z = w'_,x + θy'` being exactly what the flip
    // produces.
    let phi_z = slots[1].phi(row, l);
    let (minus_kappa_y, gamma_z) =
        crate::models::beam::section_strains(phi_z, l, &[ua[2], -ra[1], ub[2], -rb[1]], t);
    Ok(vec![
        eps,
        -minus_kappa_y,
        kappa_z,
        torsion,
        gamma_y,
        gamma_z,
    ])
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
/// direction `d` — identical to the [`frame3d`](crate::models::frame3d)
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
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Build a one-cell frame FE space between two nodes plus a `(w,θ)`/DOF
    /// node field, returning `(fespace, node_a, node_b, field)`. `space_dim`
    /// selects the 2-D or 3-D DOF set.
    fn one_element(
        space_dim: u8,
        a: &[f64],
        b: &[f64],
    ) -> (FiniteElementSpace, Node, Node, Handle<SubMesh>) {
        let coords = Handle::new(Coords::new(space_dim).unwrap());
        let na = Node::create_in(coords.clone(), a).unwrap();
        let nb = Node::create_in(coords.clone(), b).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[na.id(), nb.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&[na.clone(), nb.clone()]).unwrap());
        (fes, na, nb, support)
    }

    /// A uniform material on a subspace, with whatever components the
    /// configuration needs to build `Φ`.
    fn material(fes: &FiniteElementSpace, pairs: &[(&str, f64)]) -> ElementField {
        let names: Vec<String> = pairs.iter().map(|(c, _)| (*c).to_string()).collect();
        let mut m = SubElementField::new(fes.get(0).unwrap(), names).unwrap();
        for (c, v) in pairs {
            m.set_uniform(c, *v).unwrap();
        }
        let mut out = ElementField::empty();
        out.add_sub(Handle::new(m)).unwrap();
        out
    }

    /// `∫₀^L κ dx = θ(L) − θ(0)`, by construction of the curvature. It holds
    /// for **any** `Φ`, which is what makes it the right assertion: the old
    /// tests pinned a constant curvature, and a constant is precisely what the
    /// exact element does not have.
    fn integrated_curvature(
        s: &SubElementField,
        fes: &FiniteElementSpace,
        comp: &str,
        l: f64,
    ) -> f64 {
        let sub = fes.get(0).unwrap().read();
        (0..sub.gauss_count())
            .map(|g| s.value(0, g, comp).unwrap() * sub.gauss_weight(g).unwrap())
            .sum::<f64>()
            * l
            / 2.0 // SEG2's reference measure is 2, the element's length is L
    }

    /// Horizontal 2-D element (local = global): the axial strain is constant
    /// and exact, the curvature integrates to the rotation it came from, and
    /// the shear is constant along the span.
    #[test]
    fn horizontal_2d_strains() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(2, &[0.0, 0.0], &[l, 0.0]);
        let mut f =
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into(), "r_z".into()])
                .unwrap();
        f.set_value(a.id(), "u_x", 0.0).unwrap();
        f.set_value(b.id(), "u_x", 0.5 * l).unwrap();
        f.set_value(a.id(), "r_z", 0.0).unwrap();
        f.set_value(b.id(), "r_z", 0.25 * l).unwrap();
        let f = NodeField::from_sub(f);
        let mat = material(
            &fes,
            &[("E", 1.0), ("A", 1.0), ("I", 1.0), ("G", 1.0), ("A_s", 1.0)],
        );

        let def = beam_deformation(&f, &fes, &mat).unwrap();
        let s = def.get(0).unwrap().read();
        assert_eq!(s.components(), &["eps", "kappa", "gamma"]);
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "eps").unwrap() - 0.5).abs() < 1e-12);
        }
        // The curvature integrates to Δθ = 0.25·L.
        let total = integrated_curvature(&s, &fes, "kappa", l);
        assert!((total - 0.25 * l).abs() < 1e-10, "∫κ = {total}");
        // The shear is constant — an unloaded span carries a constant `V`.
        let g0 = s.value(0, 0, "gamma").unwrap();
        for g in 1..s.gauss_count() {
            assert!((s.value(0, g, "gamma").unwrap() - g0).abs() < 1e-12);
        }
    }

    /// The same element stood on end: the strains are read in the **local**
    /// frame, so rotating the member and its DOFs together changes nothing.
    #[test]
    fn vertical_2d_strains_are_rotated() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(2, &[0.0, 0.0], &[0.0, l]);
        let mut f =
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into(), "r_z".into()])
                .unwrap();
        // Axial is now along y.
        f.set_value(a.id(), "u_y", 0.0).unwrap();
        f.set_value(b.id(), "u_y", 0.5 * l).unwrap();
        f.set_value(a.id(), "r_z", 0.0).unwrap();
        f.set_value(b.id(), "r_z", 0.25 * l).unwrap();
        let f = NodeField::from_sub(f);
        let mat = material(
            &fes,
            &[("E", 1.0), ("A", 1.0), ("I", 1.0), ("G", 1.0), ("A_s", 1.0)],
        );

        let def = beam_deformation(&f, &fes, &mat).unwrap();
        let s = def.get(0).unwrap().read();
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "eps").unwrap() - 0.5).abs() < 1e-12);
        }
        let total = integrated_curvature(&s, &fes, "kappa", l);
        assert!((total - 0.25 * l).abs() < 1e-10, "∫κ = {total}");
    }

    /// 3-D: the axial and the torsion stay constant (their fields really are
    /// linear), and each bending plane's curvature integrates to its own
    /// rotation difference.
    #[test]
    fn horizontal_3d_strains() {
        let l = 2.0;
        let (fes, a, b, support) = one_element(3, &[0.0, 0.0, 0.0], &[l, 0.0, 0.0]);
        let mut f = SubNodeField::from_poi1(
            &support,
            ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
        .unwrap();
        f.set_value(b.id(), "u_x", 0.5 * l).unwrap();
        f.set_value(b.id(), "r_x", 0.1 * l).unwrap();
        f.set_value(b.id(), "r_z", 0.25 * l).unwrap();
        let _ = a;
        let f = NodeField::from_sub(f);
        let mat = material(
            &fes,
            &[
                ("E", 1.0),
                ("A", 1.0),
                ("I_y", 1.0),
                ("I_z", 1.0),
                ("G", 1.0),
                ("A_sy", 1.0),
                ("A_sz", 1.0),
            ],
        );

        let def = beam_deformation(&f, &fes, &mat).unwrap();
        let s = def.get(0).unwrap().read();
        for g in 0..s.gauss_count() {
            assert!((s.value(0, g, "eps").unwrap() - 0.5).abs() < 1e-12);
            assert!((s.value(0, g, "torsion").unwrap() - 0.1).abs() < 1e-12);
        }
        // The x'-y' plane bends: ∫κ_z = Δθ_z.
        let total = integrated_curvature(&s, &fes, "kappa_z", l);
        assert!((total - 0.25 * l).abs() < 1e-10, "∫κ_z = {total}");
    }

    /// The field must carry every DOF of the configuration, said up front
    /// rather than as a missing-component failure deep in the loop.
    #[test]
    fn rejects_missing_dof() {
        let (fes, _a, _b, support) = one_element(2, &[0.0, 0.0], &[1.0, 0.0]);
        // Missing the rotation `r_z`.
        let f = NodeField::from_sub(
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap(),
        );
        let mat = material(
            &fes,
            &[("E", 1.0), ("A", 1.0), ("I", 1.0), ("G", 1.0), ("A_s", 1.0)],
        );
        let err = beam_deformation(&f, &fes, &mat).unwrap_err();
        assert!(format!("{err}").contains("must carry a 'r_z'"));
    }
}
