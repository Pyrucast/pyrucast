//! Timoshenko-beam section strains of a `(w, theta)` nodal field at the Gauss
//! points — the geometric producer of the behaviour input for
//! [`crate::models::timoshenko::Timoshenko`].
//!
//! Per element it evaluates the **curvature** `κ = θ'` and the **shear strain**
//! `γ = w' − θ`. Both are taken **element-constant** — `θ'`/`w'` are constant
//! for the linear element, and `θ` is taken at the element centre (the
//! *reduced* point). Sampling `γ` at the centre rather than at the full Gauss
//! points avoids the spurious oscillating shear (the very thing reduced
//! integration removes), so the reported section forces are smooth. The
//! constant values are written at every Gauss point of `fespace`.
//!
//! Feed the result to [`crate::ops::element_field::behavior::integrate`]; the beam physics
//! turns it into the section forces `M = E·I·κ`, `V = G·A_s·γ`.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::{Field, SubField};
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::error::Result;
use crate::parallel::*;
use crate::store::{insert, read, Handle};

/// Beam section strains `(kappa, gamma)` of a `(w, theta)` node `field` at the
/// Gauss points of every subspace of `fespace`. The field must carry the
/// components `"w"` (deflection) and `"theta"` (rotation).
pub fn beam_deformation(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    // Validate the field carries the two beam components up front.
    let components = Field::components(field)?;
    for required in ["w", "theta"] {
        if !components.iter().any(|c| c == required) {
            return Err(crate::error::PyrucastError::Message(format!(
                "beam_deformation: the field must carry a '{required}' component (has: [{}])",
                components.join(", ")
            )));
        }
    }
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        out.add_sub(insert(subspace_beam_deformation(sub, &view)?))?;
    }
    Ok(out)
}

/// `(kappa, gamma)` on one 1-D `SEG2` subspace.
fn subspace_beam_deformation(
    fespace: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
) -> Result<SubElementField> {
    let s = read(fespace)?;
    let n_nodes = s.nodes_per_cell()?;
    let n_g = s.gauss_count();
    let submesh = s.submesh();
    let conn = read(&submesh)?.connectivity().to_vec();

    let mut field = SubElementField::new(fespace.clone(), vec!["kappa".into(), "gamma".into()])?;
    // Parallel per cell: each cell owns a disjoint 2·n_g chunk of the output,
    // written once; the FE space and field guards are shared, read-only.
    let s_ref: &SubFiniteElementSpace = &s;
    field
        .values_mut()
        .par_chunks_mut(2 * n_g)
        .with_min_len((MIN_PARALLEL_LEN / (2 * n_g).max(1)).max(1))
        .enumerate()
        .try_for_each(|(cell, chunk)| -> Result<()> {
            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            let w: Vec<f64> = ids
                .iter()
                .map(|&id| view.value(id, "w"))
                .collect::<Result<_>>()?;
            let th: Vec<f64> = ids
                .iter()
                .map(|&id| view.value(id, "theta"))
                .collect::<Result<_>>()?;

            // Linear element ⇒ θ', w' are constant: use the first Gauss point's
            // dN/dx. θ at the centre (reduced point) is the nodal average.
            let dn = s_ref.dn_dx(cell, 0)?; // [dN_0/dx, dN_1/dx] (1-D)
            let kappa: f64 = (0..n_nodes).map(|i| dn[i] * th[i]).sum(); // θ'
            let dwdx: f64 = (0..n_nodes).map(|i| dn[i] * w[i]).sum(); // w'
            let theta_centre: f64 = th.iter().sum::<f64>() / n_nodes as f64;
            let gamma = dwdx - theta_centre; // γ = w' − θ (reduced)

            for g in 0..n_g {
                chunk[2 * g] = kappa;
                chunk[2 * g + 1] = gamma;
            }
            Ok(())
        })?;
    Ok(field)
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
    use crate::store::insert;

    /// Pure bending kinematics `w = 0`, `θ = x` on a SEG2 ⇒ `κ = 1`, `γ = −θ`.
    #[test]
    fn curvature_and_shear_of_linear_rotation() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["w".into(), "theta".into()]).unwrap();
        f.set_value(a.id(), "theta", 0.0).unwrap();
        f.set_value(b.id(), "theta", 2.0).unwrap(); // θ = x
        let f = NodeField::from_sub(f);

        let def = beam_deformation(&f, &fes).unwrap();
        let s = read(&def.get(0).unwrap()).unwrap();
        assert_eq!(s.components(), &["kappa".to_string(), "gamma".to_string()]);
        for g in 0..s.gauss_count() {
            // θ' = 1 everywhere; γ = w' − θ_centre = 0 − (0+2)/2 = −1.
            assert!((s.value(0, g, "kappa").unwrap() - 1.0).abs() < 1e-12);
            assert!((s.value(0, g, "gamma").unwrap() + 1.0).abs() < 1e-12);
        }
    }
}
