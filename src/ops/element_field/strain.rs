//! The **generalized strain** of a model — the `B` of `ε = B·u`, sub-model by
//! sub-model: the input each law of the model reads, produced from the nodal
//! solution (Cast3M `EPSI`, extended to every physics).
//!
//! The adjoint of
//! [`crate::ops::node_field::internal_forces`](fn@crate::ops::node_field::internal_forces):
//!
//! ```text
//!   nodes                      Gauss points
//!   u      ──── strain (B) ───▶  ε
//!                                │  integrate_behavior (the law)
//!   f_int  ◀── internal_forces ─  σ
//!                (Bᵀ W)
//! ```
//!
//! The geometric producers — `deformation`, `gradient`, `interp_to_gauss`,
//! `beam_deformation`, `shell_deformation` — know an FE space, not a model.
//! Which one a law needs, and on which unknowns, is the law's own fact
//! ([`Behavior::strain_operator`](crate::models::Behavior::strain_operator));
//! this operator asks each law, then runs its producer on that law's space.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::StrainOperator;
use crate::ops::element_field::beam_deformation::subspace_beam_deformation;
use crate::ops::element_field::deformation::deformation_zone;
use crate::ops::element_field::gradient::gradient_zone;
use crate::ops::element_field::interp_to_gauss::interp_to_gauss_zone;
use crate::ops::element_field::shell_deformation::subspace_shell_deformation;

/// The input of every law of `model`, produced from the nodal `solution` —
/// one zone per sub-model with a law, on its behaviour space, carrying what
/// that law reads (see the module docs).
///
/// | sub-model | zone produced |
/// |---|---|
/// | continuum (elasticity, plasticity, damage), truss | small-strain tensor `eps_*` |
/// | conduction, diffusion | gradient `grad_T_*`, `grad_c_<species>_*` |
/// | radiation | the temperature `T` at the points |
/// | Bernoulli, Timoshenko beams | section strains `eps`, `kappa*`, `gamma*`, … |
/// | shell | `eps_*`, `kappa_*`, `drill`, `gamma_*` in the facet's frame |
///
/// Each law reads **its own** unknowns of `solution` by name — a solution
/// carrying displacement, temperature and multipliers serves every law at
/// once; an unknown the solution lacks reads as zero, the nodal convention.
/// Sub-models without a law (constraints, loads, boundary transfers) produce
/// nothing.
///
/// `materials` is read by the beams alone: how a Timoshenko element shares its
/// curvature with shear depends on `Φ = 12EI/(G·A_s·L²)`.
///
/// The result feeds [`behavior::integrate`](fn@crate::ops::element_field::behavior::integrate)
/// (`integrate_behavior` in Python) as is.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, mesh, model};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let mut mat = ElementField::new(&fes, vec!["k".into()])?;
/// # mat.get(0)?.write().set_uniform("k", 1.0)?;
/// let solide = model::elasticity(&fes, Kinematics::PlaneStrain, MaterialSymmetry::Isotropic, StrainMeasure::Linear)?;
/// let couple = solide.union(&model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?)?;
/// // One solution for both physics: u_x = 0.1·x and T = 5·y.
/// let sol = NodeField::from_submesh(&support.get(0)?,
///     vec!["u_x".into(), "u_y".into(), "T".into()])?;
/// sol.get(0)?.write().set_value(n[1].id(), "u_x", 0.2)?;
/// sol.get(0)?.write().set_value(n[2].id(), "T", 10.0)?;
/// let eps = element_field::strain(&couple, &sol, &mat)?;
/// // Each law gets its own input: a strain for the solid, a gradient for
/// // conduction.
/// assert_eq!(eps.len(), 2);
/// assert!((eps.get(0)?.read().value(0, 0, "eps_xx")? - 0.1).abs() < 1e-12);
/// assert!((eps.get(1)?.read().value(0, 0, "grad_T_y")? - 5.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn strain(
    model: &Model,
    solution: &NodeField,
    materials: &ElementField,
) -> Result<ElementField> {
    // One view of the solution for every law: building it indexes every node.
    let view = solution.view()?;
    let mut out = ElementField::empty();
    for h in model {
        let sub = h.read();
        let kind = sub.as_kind();
        let Some(law) = kind.as_behavior() else {
            continue; // no law, no input
        };
        let fespace = law.behavior_fespace();
        let unknowns = kind.primal_vars();
        let zone = match law.strain_operator() {
            StrainOperator::Deformation(measure) => {
                deformation_zone(&fespace, &view, &unknowns, measure)?
            }
            StrainOperator::Gradient => gradient_zone(&fespace, &view, &unknowns)?,
            StrainOperator::Value => interp_to_gauss_zone(&fespace, &view, &unknowns)?,
            StrainOperator::Beam => {
                let material = materials
                    .sub_for_fespace_with(&law.material_fespace(), &law.material_components())?;
                subspace_beam_deformation(&fespace, &view, &unknowns, &material)?
            }
            StrainOperator::Shell(formulation) => {
                subspace_shell_deformation(&fespace, &view, &unknowns, formulation)?
            }
        };
        out.add_sub(Handle::new(zone))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::{Field, SubField};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::models::tensor::Kinematics;
    use crate::models::RelationSense;
    use crate::ops::element_field::{self, StrainMeasure};
    use crate::ops::{mesh, model, node_field};

    /// Two TRI3 on the unit square and its four nodes.
    fn square() -> (FiniteElementSpace, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        sm.add_cell(&[n[0].id(), n[2].id(), n[3].id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
        (fes, n)
    }

    /// A solution `u_x = 0.1x + 0.3y`, `u_y = -0.2y`, `T = 4x - y`,
    /// `c_h = 2x + 3y`, with a stray multiplier that no law reads.
    fn solution(n: &[Node]) -> NodeField {
        let support = mesh::poi1_from_nodes(n).unwrap();
        let names = ["u_x", "u_y", "T", "c_h", "lambda"]
            .map(String::from)
            .to_vec();
        let sol = NodeField::from_submesh(&support.get(0).unwrap(), names).unwrap();
        let x = node_field::positions(&support, None).unwrap();
        {
            let mut z = sol.get(0).unwrap().write();
            let x = x.get(0).unwrap().read();
            for node in n {
                let (px, py) = (
                    x.value(node.id(), "X").unwrap(),
                    x.value(node.id(), "Y").unwrap(),
                );
                z.set_value(node.id(), "u_x", 0.1 * px + 0.3 * py).unwrap();
                z.set_value(node.id(), "u_y", -0.2 * py).unwrap();
                z.set_value(node.id(), "T", 4.0 * px - py).unwrap();
                z.set_value(node.id(), "c_h", 2.0 * px + 3.0 * py).unwrap();
                z.set_value(node.id(), "lambda", 99.0).unwrap();
            }
        }
        sol
    }

    fn unit_material(fes: &FiniteElementSpace) -> ElementField {
        let mat = ElementField::new(fes, vec!["k".into()]).unwrap();
        mat.get(0).unwrap().write().set_uniform("k", 1.0).unwrap();
        mat
    }

    /// The model-level strain of a solid is the geometric one of its
    /// displacement, value for value — the other unknowns ignored.
    #[test]
    fn a_solid_reads_the_strain_of_its_displacement() {
        let (fes, n) = square();
        let sol = solution(&n);
        let solid = model::elasticity(
            &fes,
            Kinematics::PlaneStrain,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
        .unwrap();
        let got = strain(&solid, &sol, &unit_material(&fes)).unwrap();
        let u = sol.filter_components(["u_x", "u_y"]).unwrap();
        let want = element_field::deformation(&u, &fes, StrainMeasure::Linear).unwrap();
        let (g, w) = (got.get(0).unwrap().read(), want.get(0).unwrap().read());
        assert_eq!(g.components(), w.components());
        assert_eq!(g.values(), w.values());
        assert!((g.value(1, 0, "eps_xy").unwrap() - 0.15).abs() < 1e-12);
    }

    /// Conduction reads the gradient of `T`.
    #[test]
    fn conduction_reads_the_gradient_of_its_temperature() {
        let (fes, n) = square();
        let sol = solution(&n);
        let conduction = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let got = strain(&conduction, &sol, &unit_material(&fes)).unwrap();
        let g = got.get(0).unwrap().read();
        assert_eq!(
            g.components(),
            &["grad_T_x".to_string(), "grad_T_y".to_string()]
        );
        assert!((g.value(0, 0, "grad_T_x").unwrap() - 4.0).abs() < 1e-12);
        assert!((g.value(1, 2, "grad_T_y").unwrap() + 1.0).abs() < 1e-12);
    }

    /// Diffusion reads the gradient of its own species' concentration.
    #[test]
    fn diffusion_reads_the_gradient_of_its_species() {
        let (fes, n) = square();
        let got = strain(
            &model::fick(&fes, "h", MaterialSymmetry::Isotropic).unwrap(),
            &solution(&n),
            &unit_material(&fes),
        );
        let got = got.unwrap();
        let g = got.get(0).unwrap().read();
        assert!((g.value(0, 0, "grad_c_h_x").unwrap() - 2.0).abs() < 1e-12);
        assert!((g.value(1, 0, "grad_c_h_y").unwrap() - 3.0).abs() < 1e-12);
    }

    /// Radiation, on the edge `x = 1`, reads `T = 4 − y` at its points: the
    /// symmetric rule averages it to its value at mid-edge, 3.5.
    #[test]
    fn radiation_reads_the_temperature_at_its_points() {
        let (fes, n) = square();
        let coords = n[0].coords();
        let mut edge = SubMesh::new(coords, ElementType::SEG2);
        edge.add_cell(&[n[1].id(), n[2].id()]).unwrap();
        let edge = FiniteElementSpace::lagrange1(&Mesh::from_submesh(edge)).unwrap();
        let target = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let rad = model::radiation(&edge, &target).unwrap();
        let got = strain(&rad, &solution(&n), &unit_material(&fes)).unwrap();
        let g = got.get(0).unwrap().read();
        assert_eq!(g.components(), &["T".to_string()]);
        let mean = (0..g.gauss_count())
            .map(|p| g.value(0, p, "T").unwrap())
            .sum::<f64>()
            / g.gauss_count() as f64;
        assert!((mean - 3.5).abs() < 1e-12, "{mean}");
    }

    /// Two laws on one mesh: the chain `strain → integrate → internal_forces`
    /// runs on the coupled model without consolidating anything — each consumer
    /// takes the zone it reads — and, the laws being linear, gives `K·u`.
    #[test]
    fn a_coupled_model_runs_the_whole_chain() {
        let (fes, n) = square();
        let sol = solution(&n);
        let solid = model::elasticity(
            &fes,
            Kinematics::PlaneStrain,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
        .unwrap();
        let coupled = solid
            .union(&model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap())
            .unwrap();
        let materials = element_field::material_field_per_sub_model(
            &coupled,
            &[&[("E", 200.0), ("nu", 0.3)], &[("k", 45.0)]],
        )
        .unwrap();
        let eps = strain(&coupled, &sol, &materials).unwrap();
        let state = element_field::behavior::integrate(&coupled, &eps, None, &materials, None);
        let state = state.unwrap();
        // The state at A, two zones on one space, is read back as well.
        let again =
            element_field::behavior::integrate(&coupled, &eps, Some(&state), &materials, None);
        assert!(again.is_ok(), "{:?}", again.err());
        let f_int = node_field::internal_forces(&coupled, &state, &sol, &materials).unwrap();
        let k = crate::ops::matrix::stiffness(&coupled, &materials).unwrap();
        let ku = k
            .mul_field(&sol.filter_components(coupled.primal_vars()).unwrap())
            .unwrap();
        for node in &n {
            for dual in ["f_x", "f_y", "q"] {
                let (got, want) = (
                    f_int.value(node.id(), dual).unwrap(),
                    ku.value(node.id(), dual).unwrap(),
                );
                assert!(
                    (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "{dual}: {got} vs {want}"
                );
            }
        }
    }

    /// Sub-models without a law contribute nothing; the order of the zones is
    /// the order of the laws.
    #[test]
    fn a_constraint_has_no_input() {
        let (fes, n) = square();
        let sol = solution(&n);
        let solid = model::elasticity(
            &fes,
            Kinematics::PlaneStrain,
            MaterialSymmetry::Isotropic,
            StrainMeasure::Linear,
        )
        .unwrap();
        let clamp = mesh::poi1_from_nodes(&n[..1]).unwrap();
        let support = model::dirichlet(&clamp, &solid, "u_x", RelationSense::Equality);
        let blocked = solid.union(&support.unwrap()).unwrap();
        let got = strain(&blocked, &sol, &unit_material(&fes)).unwrap();
        assert_eq!(got.len(), 1);
    }
}
