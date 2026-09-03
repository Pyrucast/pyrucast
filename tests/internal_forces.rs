//! `f_int == K·u` for the structural elements — the invariant that says a
//! residual and a stiffness describe the **same** element.
//!
//! `∫ Bᵀσ` and `∫ Bᵀ D B` are built from one `B`, so for a **linear** law the
//! two must agree exactly. Every structural law is linear — `N = EA·ε`,
//! `M = EI·κ`, `Q = GA_s·γ` — which is what makes this a test with no tolerance
//! to negotiate: the difference is rounding, and nothing else. A `B` transposed
//! wrongly, a rotation applied to one side only, a Gauss weight counted twice
//! would all show up here at once.
//!
//! Each case solves a clamped, loaded member and compares the internal forces
//! of the state it settles in with the assembled stiffness applied to that same
//! displacement. `Truss` is here as the **witness**: it is the one structural
//! element that already had a `Bᵀ`, so it must not move.
//!
//! The mirror of `tests/tangent.rs`, which checks the consistent tangent against
//! the finite-difference derivative of these same internal forces.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::Matrix;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::element_field::{beam_deformation, behavior, deformation, material_field};
use pyrucast::ops::node_field::internal_forces;
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{mesh, model};
use pyrucast::Result;

/// A chain of `SEG2` cells through `points`, and its FE space under
/// `interpolation`.
fn chain(
    points: &[&[f64]],
    interpolation: Interpolation,
) -> Result<(FiniteElementSpace, Vec<Node>)> {
    let coords = Handle::new(Coords::new(points[0].len() as u8)?);
    let nodes: Vec<Node> = points
        .iter()
        .map(|p| Node::create_in(coords.clone(), p))
        .collect::<Result<_>>()?;
    let mut m = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    for w in nodes.windows(2) {
        m.add_cell(&[w[0].id(), w[1].id()])?;
    }
    Ok((FiniteElementSpace::new(&m, interpolation)?, nodes))
}

/// `model` with every primal DOF of `node` held at zero — a clamped end.
fn clamp(model: &Model, node: &Node) -> Result<Model> {
    let mut out = model.subset(0..model.len())?;
    for (var, dual) in model.primal_vars().iter().zip(model.dual_vars()) {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesh::barycenter(&imposed)?;
        out = out.union(&model::dirichlet(
            var.clone(),
            dual,
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    Ok(out)
}

/// The displacement a clamped member settles in under `loads` at `node`.
fn solve_clamped(
    model: &Model,
    materials: &ElementField,
    clamped: &Node,
    loaded: &Node,
    loads: &[(&str, f64)],
) -> Result<NodeField> {
    let support = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(loaded))?);
    let mut rhs = SubNodeField::from_poi1(
        &support,
        loads.iter().map(|(c, _)| (*c).to_string()).collect(),
    )?;
    for (c, v) in loads {
        rhs.set_value(loaded.id(), c, *v)?;
    }
    let k = pyrucast::ops::matrix::stiffness(&clamp(model, clamped)?, materials)?;
    solve(&k, &NodeField::from_sub(rhs))
}

/// The primal degrees of freedom of `solution`, on the member's own nodes.
///
/// A solve returns the multipliers alongside them, and a deformation operator
/// counts the components it is handed.
fn primal_of(solution: &NodeField, nodes: &[Node], vars: &[String]) -> Result<NodeField> {
    let support = Handle::new(SubMesh::poi1_from_nodes(nodes)?);
    let mut u = SubNodeField::from_poi1(&support, vars.to_vec())?;
    for n in nodes {
        for v in vars {
            u.set_value(n.id(), v, solution.value(n.id(), v)?)?;
        }
    }
    Ok(NodeField::from_sub(u))
}

/// `‖f_int − K·u‖∞`, relative to `‖K·u‖∞`, over every node and dual variable.
fn residual_gap(k: &Matrix, f_int: &NodeField, u: &NodeField, nodes: &[Node]) -> Result<f64> {
    let f_k = k.mul_field(u)?;
    let (mut gap, mut scale) = (0.0_f64, 0.0_f64);
    for n in nodes {
        for var in f_k.get(0)?.read().components() {
            let want = f_k.value(n.id(), var)?;
            let got = f_int.value(n.id(), var)?;
            gap = gap.max((got - want).abs());
            scale = scale.max(want.abs());
        }
    }
    Ok(gap / scale)
}

/// Solve the clamped member, then check its internal forces against `K·u`.
///
/// `model` is the **bare** physics: the clamp is added for the solve alone, so
/// the stiffness compared against carries no multiplier block.
fn check_beam(
    fes: &FiniteElementSpace,
    model: &Model,
    pairs: &[(&str, f64)],
    nodes: &[Node],
    loads: &[(&str, f64)],
) -> Result<()> {
    let materials = material_field(model, pairs)?;
    let solution = solve_clamped(model, &materials, &nodes[0], nodes.last().unwrap(), loads)?;
    let u = primal_of(&solution, nodes, &model.primal_vars())?;
    let strain = beam_deformation(&u, fes, &materials)?;
    let state = behavior::integrate(model, &strain, None, &materials, None)?;
    let f_int = internal_forces(&state, model)?;
    let k = pyrucast::ops::matrix::stiffness(model, &materials)?;
    let gap = residual_gap(&k, &f_int, &u, nodes)?;
    assert!(gap < 1e-10, "f_int vs K·u: relative gap {gap:e}");
    Ok(())
}

// ─── Euler-Bernoulli, in the three configurations ───────────────────────────

#[test]
fn bernoulli_planar_1d_internal_forces_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(&[&[0.0], &[0.9], &[2.0]], Interpolation::Hermite3)?;
    check_beam(
        &fes,
        &model::bernoulli(&fes)?,
        &[("E", 210_000.0), ("I", 1.0e-4)],
        &nodes,
        &[("f_w", 50.0), ("m_theta", 12.0)],
    )
}

#[test]
fn bernoulli_frame_2d_internal_forces_match_k_times_u() -> Result<()> {
    // Incliné : la rotation vers les axes globaux doit se transposer avec le
    // reste, et une poutre horizontale ne le dirait pas.
    let (fes, nodes) = chain(
        &[&[0.0, 0.0], &[1.2, 0.9], &[2.4, 1.8]],
        Interpolation::Hermite3,
    )?;
    check_beam(
        &fes,
        &model::bernoulli(&fes)?,
        &[("E", 210_000.0), ("A", 1.0e-2), ("I", 1.0e-4)],
        &nodes,
        &[("f_x", 30.0), ("f_y", 50.0), ("m_z", 8.0)],
    )
}

#[test]
fn bernoulli_frame_3d_internal_forces_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(
        &[&[0.0, 0.0, 0.0], &[1.0, 0.6, 0.4], &[2.0, 1.2, 0.8]],
        Interpolation::Hermite3,
    )?;
    check_beam(
        &fes,
        &model::bernoulli(&fes)?,
        &[
            ("E", 210_000.0),
            ("A", 1.0e-2),
            ("I_y", 1.0e-4),
            ("I_z", 2.0e-4),
            ("J", 1.5e-4),
            ("G", 80_000.0),
        ],
        &nodes,
        // Les deux plans de flexion, l'axial et la torsion, tous excités.
        &[
            ("f_x", 30.0),
            ("f_y", 50.0),
            ("f_z", 20.0),
            ("m_x", 9.0),
            ("m_y", 7.0),
            ("m_z", 8.0),
        ],
    )
}

// ─── Timoshenko, in the three configurations ────────────────────────────────

#[test]
fn timoshenko_planar_1d_internal_forces_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(&[&[0.0], &[0.9], &[2.0]], Interpolation::ModelEmbedded)?;
    check_beam(
        &fes,
        &model::timoshenko(&fes)?,
        // Trapue : `Φ` pèse, et une section élancée le cacherait.
        &[
            ("E", 210_000.0),
            ("I", 1.0e-4),
            ("G", 80_000.0),
            ("A_s", 8.0e-3),
        ],
        &nodes,
        &[("f_w", 50.0), ("m_theta", 12.0)],
    )
}

#[test]
fn timoshenko_frame_2d_internal_forces_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(
        &[&[0.0, 0.0], &[1.2, 0.9], &[2.4, 1.8]],
        Interpolation::ModelEmbedded,
    )?;
    check_beam(
        &fes,
        &model::timoshenko(&fes)?,
        &[
            ("E", 210_000.0),
            ("A", 1.0e-2),
            ("I", 1.0e-4),
            ("G", 80_000.0),
            ("A_s", 8.0e-3),
        ],
        &nodes,
        &[("f_x", 30.0), ("f_y", 50.0), ("m_z", 8.0)],
    )
}

#[test]
fn timoshenko_frame_3d_internal_forces_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(
        &[&[0.0, 0.0, 0.0], &[1.0, 0.6, 0.4], &[2.0, 1.2, 0.8]],
        Interpolation::ModelEmbedded,
    )?;
    check_beam(
        &fes,
        &model::timoshenko(&fes)?,
        // Les deux plans portent leur **propre** `Φ` : des inerties et des
        // sections réduites différentes les distinguent, et un appariement
        // croisé se verrait.
        &[
            ("E", 210_000.0),
            ("A", 1.0e-2),
            ("I_y", 1.0e-4),
            ("I_z", 3.0e-4),
            ("J", 1.5e-4),
            ("G", 80_000.0),
            ("A_sy", 8.0e-3),
            ("A_sz", 5.0e-3),
        ],
        &nodes,
        &[
            ("f_x", 30.0),
            ("f_y", 50.0),
            ("f_z", 20.0),
            ("m_x", 9.0),
            ("m_y", 7.0),
            ("m_z", 8.0),
        ],
    )
}

// ─── The witness ────────────────────────────────────────────────────────────

/// `Truss` is the one structural element that already carried its own `Bᵀ`. It
/// must not have moved by a bit.
///
/// It is checked on an **imposed** displacement rather than a solved one: a
/// chain of bars is a mechanism transversally — pin one end and there is
/// nothing to solve — while `f_int == K·u` holds for *any* displacement, the
/// law being linear. Which makes this the stronger reading of the two.
#[test]
fn truss_internal_forces_still_match_k_times_u() -> Result<()> {
    let (fes, nodes) = chain(
        &[&[0.0, 0.0], &[1.2, 0.9], &[2.4, 1.8]],
        Interpolation::Lagrange1,
    )?;
    let model = model::truss(&fes)?;
    let materials = material_field(&model, &[("E", 210_000.0), ("A", 1.0e-2)])?;

    let support = Handle::new(SubMesh::poi1_from_nodes(&nodes)?);
    let mut imposed = SubNodeField::from_poi1(&support, model.primal_vars())?;
    for (i, n) in nodes.iter().enumerate() {
        for (j, v) in model.primal_vars().iter().enumerate() {
            imposed.set_value(n.id(), v, 1e-3 * (1.0 + i as f64) * (0.7 - 0.3 * j as f64))?;
        }
    }
    let u = NodeField::from_sub(imposed);

    let strain = deformation(&u, &fes)?;
    let state = behavior::integrate(&model, &strain, None, &materials, None)?;
    let f_int = internal_forces(&state, &model)?;
    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let gap = residual_gap(&k, &f_int, &u, &nodes)?;
    assert!(gap < 1e-10, "f_int vs K·u: relative gap {gap:e}");
    Ok(())
}
