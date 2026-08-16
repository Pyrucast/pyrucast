//! The yield laws of [`PlasticLaw`], each checked on what makes it *that* law.
//!
//! A plasticity test that only asserts « the stress went down » proves nothing.
//! Each law here is pinned by its **defining property**:
//!
//! | law | what is asserted |
//! |---|---|
//! | isotropic hardening | consistency: `q = σ_y + H·p` after the return, and `H = 0` reproduces the perfect law exactly |
//! | Drucker-Prager | the return lands on the cone, compression yields later than tension, and a strong hydrostatic tension collapses onto the apex |
//! | Ottosen | the return lands on the four-parameter surface, which is far weaker in tension than in compression |
//!
//! On top of that, **every** law has its consistent tangent checked against a
//! central difference of the internal forces — the same oracle as
//! `tests/tangent.rs`. That is what catches a hand-derived tangent that is
//! plausible but wrong, and it is the reason the Drucker-Prager derivation can
//! be trusted at all.
//!
//! Single source for the « lois d'écoulement » examples of the plasticity book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::models::plastic::PlasticLaw;
use pyrucast::ops::element_field::{behavior::integrate, deformation, material_field};
use pyrucast::ops::matrix::tangent;
use pyrucast::ops::node_field::internal_forces;
use pyrucast::store::{insert, read};
use pyrucast::Result;

const AXES: [&str; 3] = ["x", "y", "z"];

/// von Mises with linear isotropic hardening.
const ISOTROPIC: &[(&str, f64)] = &[
    ("E", 70_000.0),
    ("nu", 0.3),
    ("sigma_y", 200.0),
    ("H", 5_000.0),
];
/// A frictional, mildly dilatant material — `ψ < α`, so the flow is
/// non-associated.
const DRUCKER: &[(&str, f64)] = &[
    ("E", 20_000.0),
    ("nu", 0.2),
    ("friction", 0.3),
    ("k", 30.0),
    ("psi", 0.1),
];
/// Ottosen's classic concrete set, for a tensile/compressive strength ratio of
/// about 0.1.
const OTTOSEN: &[(&str, f64)] = &[
    ("E", 30_000.0),
    ("nu", 0.2),
    ("a", 1.2759),
    ("b", 3.1962),
    ("k_1", 11.7365),
    ("k_2", 0.9801),
    ("sigma_c", 30.0),
];

#[test]
fn isotropic_hardening_satisfies_its_consistency_condition() -> Result<()> {
    let cube = Cube::new(PlasticLaw::Isotropic, ISOTROPIC)?;
    // Well past yield, so the step is definitely plastic.
    let s = cube.stress(&uniaxial(0.02))?;
    let (q, p) = (von_mises(&s.sigma), s.p);
    assert!(p > 0.0, "the step must be plastic (p = {p})");
    let expected = 200.0 + 5_000.0 * p;
    assert!(
        (q - expected).abs() < 1e-6 * expected,
        "q = {q}, expected σ_y + H·p = {expected}"
    );
    Ok(())
}
// ANCHOR_END: example

/// With `H = 0` the hardening law **is** the perfect one — same stress, same
/// state. Two code paths that must agree, and the cheapest way to know they do.
#[test]
fn isotropic_hardening_with_no_hardening_is_the_perfect_law() -> Result<()> {
    let no_hardening: Vec<(&str, f64)> =
        vec![("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0), ("H", 0.0)];
    let hardening = Cube::new(PlasticLaw::Isotropic, &no_hardening)?;
    let perfect = Cube::new(
        PlasticLaw::Perfect,
        &[("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)],
    )?;
    let u = uniaxial(0.02);
    let (a, b) = (hardening.stress(&u)?, perfect.stress(&u)?);
    for i in 0..6 {
        assert!(
            (a.sigma[i] - b.sigma[i]).abs() < 1e-9,
            "component {i}: {} vs {}",
            a.sigma[i],
            b.sigma[i]
        );
    }
    assert!((a.p - b.p).abs() < 1e-12);
    Ok(())
}

#[test]
fn drucker_prager_returns_onto_its_cone() -> Result<()> {
    let cube = Cube::new(PlasticLaw::DruckerPrager, DRUCKER)?;
    let s = cube.stress(&uniaxial(0.02))?;
    assert!(s.p > 0.0, "the step must be plastic");
    let f = von_mises(&s.sigma) + 0.3 * trace(&s.sigma) - 30.0;
    assert!(f.abs() < 1e-6 * 30.0, "f = {f}, must be 0 on the cone");
    Ok(())
}

/// The defining property of a pressure-sensitive surface: the same shear is
/// admissible under compression and not under tension. von Mises cannot tell
/// them apart; this is exactly what `α` adds.
#[test]
fn drucker_prager_is_stronger_in_compression() -> Result<()> {
    let cube = Cube::new(PlasticLaw::DruckerPrager, DRUCKER)?;
    let tension = cube.stress(&uniaxial(0.01))?;
    let compression = cube.stress(&uniaxial(-0.01))?;
    assert!(
        von_mises(&compression.sigma) > von_mises(&tension.sigma) * 1.05,
        "compression {} should sustain more than tension {}",
        von_mises(&compression.sigma),
        von_mises(&tension.sigma)
    );
    Ok(())
}

/// A cone has a tip. Under strong hydrostatic tension the flank return would
/// push the equivalent stress negative, which is meaningless — the stress must
/// collapse onto the apex `I₁ = k/α` instead. This is the case a naive
/// implementation gets wrong, silently.
#[test]
fn drucker_prager_collapses_onto_its_apex() -> Result<()> {
    let cube = Cube::new(PlasticLaw::DruckerPrager, DRUCKER)?;
    // Pure hydrostatic tension, far beyond the tip.
    let s = cube.stress(&hydrostatic(0.05))?;
    assert!(
        von_mises(&s.sigma) < 1e-8,
        "the deviator must vanish at the apex (q = {})",
        von_mises(&s.sigma)
    );
    let expected = 30.0 / 0.3; // I₁ = k/α
    assert!(
        (trace(&s.sigma) - expected).abs() < 1e-6 * expected,
        "I₁ = {}, expected {expected}",
        trace(&s.sigma)
    );
    Ok(())
}

#[test]
fn ottosen_returns_onto_its_surface() -> Result<()> {
    let cube = Cube::new(PlasticLaw::Ottosen, OTTOSEN)?;
    let s = cube.stress(&uniaxial(0.01))?;
    assert!(s.p > 0.0, "the step must be plastic");
    assert!(
        ottosen_f(&s.sigma).abs() < 1e-6,
        "f = {}, must be 0 on the surface",
        ottosen_f(&s.sigma)
    );
    Ok(())
}

/// Concrete is roughly ten times weaker in tension than in compression, and it
/// is the **Lode-angle** term that expresses it — a pressure-sensitive but
/// Lode-blind surface could not.
#[test]
fn ottosen_is_far_weaker_in_tension() -> Result<()> {
    let cube = Cube::new(PlasticLaw::Ottosen, OTTOSEN)?;
    let tension = cube.stress(&uniaxial(0.01))?;
    let compression = cube.stress(&uniaxial(-0.01))?;
    let (t, c) = (trace(&tension.sigma).abs(), trace(&compression.sigma).abs());
    assert!(
        c > 3.0 * t,
        "compression should carry far more: {c} against {t}"
    );
    Ok(())
}

/// Every law's consistent tangent, against a central difference of the internal
/// forces. This is the check that makes a hand-derived tangent trustworthy —
/// and the reason Ottosen's is computed numerically rather than derived.
#[test]
fn every_law_has_a_consistent_tangent() -> Result<()> {
    // Drucker-Prager is driven in **shear**: uniaxial tension sends it straight
    // to the apex, where the stress is pinned and the tangent is legitimately
    // zero (checked separately below). The flank is the differentiable regime,
    // and the one worth validating.
    //
    // The tolerance differs by law, and the difference is the point. von Mises
    // and Drucker-Prager have **closed-form** returns, so their tangent — derived
    // or differentiated — is accurate to the difference step: 0.2 %.
    //
    // Ottosen is doubly numerical: its return map differentiates `f` to get the
    // normal, and the tangent then differentiates that whole iterative map. The
    // two error scales compound, and roughly 10 % is what it costs. That is
    // immaterial for Newton — which needs a tangent good enough to converge, not
    // one good to machine precision, and pays only in its convergence *rate* —
    // and it is more honest to state the figure than to loosen every tolerance
    // to the worst case.
    for (law, mat, disp, tol) in [
        (PlasticLaw::Isotropic, ISOTROPIC, uniaxial(0.02), 2e-3),
        (PlasticLaw::DruckerPrager, DRUCKER, shear(0.002), 2e-3),
        (PlasticLaw::Ottosen, OTTOSEN, uniaxial(0.01), 1e-1),
    ] {
        let cube = Cube::new(law, mat)?;
        cube.check_tangent(&disp, tol)
            .map_err(|e| pyrucast::PyrucastError::Message(format!("{law}: {e}")))?;
    }
    Ok(())
}

/// At the apex the return pins the stress, so the tangent is **zero** — and the
/// finite difference agrees. Returning the elastic modulus there would be the
/// tempting fallback and would make the tangent inconsistent with the return
/// map; this test is what forbids it.
#[test]
fn the_drucker_prager_apex_has_a_vanishing_tangent() -> Result<()> {
    let cube = Cube::new(PlasticLaw::DruckerPrager, DRUCKER)?;
    // Deep into hydrostatic tension: every Gauss point sits on the tip.
    cube.check_tangent(&hydrostatic(0.05), 2e-3)?;
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The converged state of one Gauss point.
struct State {
    /// Full 3-D stress, Voigt order `[xx, yy, zz, yz, xz, xy]`.
    sigma: [f64; 6],
    /// Cumulated plastic strain.
    p: f64,
}

fn von_mises(s: &[f64; 6]) -> f64 {
    let mean = (s[0] + s[1] + s[2]) / 3.0;
    let d = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]];
    (1.5 * (d[0] * d[0]
        + d[1] * d[1]
        + d[2] * d[2]
        + 2.0 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5])))
        .sqrt()
}

fn trace(s: &[f64; 6]) -> f64 {
    s[0] + s[1] + s[2]
}

/// Ottosen's yield function for the parameter set above — recomputed here rather
/// than imported, so the test checks the law against the *formula* and not
/// against itself.
fn ottosen_f(s: &[f64; 6]) -> f64 {
    let (a, b, k1, k2, sc) = (1.2759, 3.1962, 11.7365, 0.9801, 30.0);
    let mean = trace(s) / 3.0;
    let d = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]];
    let j2 = 0.5
        * (d[0] * d[0]
            + d[1] * d[1]
            + d[2] * d[2]
            + 2.0 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]));
    let j3 = d[0] * (d[1] * d[2] - d[3] * d[3]) - d[5] * (d[5] * d[2] - d[3] * d[4])
        + d[4] * (d[5] * d[3] - d[1] * d[4]);
    let cos3t = if j2 > 1e-30 {
        (3.0 * 3.0_f64.sqrt() / 2.0 * j3 / j2.powf(1.5)).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    let lambda = if cos3t >= 0.0 {
        k1 * ((k2 * cos3t).clamp(-1.0, 1.0).acos() / 3.0).cos()
    } else {
        k1 * (std::f64::consts::FRAC_PI_3 - (-k2 * cos3t).clamp(-1.0, 1.0).acos() / 3.0).cos()
    };
    a * j2 / (sc * sc) + lambda * j2.sqrt() / sc + b * trace(s) / sc - 1.0
}

/// Nodal displacements of a uniaxial strain `ε_xx = e` on the unit cube.
fn uniaxial(e: f64) -> Vec<f64> {
    CORNERS.iter().flat_map(|c| [e * c[0], 0.0, 0.0]).collect()
}

/// Nodal displacements of a pure **shear** `γ_xy = 2e` — deviatoric, so `I₁`
/// stays zero and a pressure-sensitive law stays on its flank rather than
/// running to the apex.
fn shear(e: f64) -> Vec<f64> {
    CORNERS
        .iter()
        .flat_map(|c| [e * c[1], e * c[0], 0.0])
        .collect()
}

/// Nodal displacements of a hydrostatic strain `ε = e·I`.
fn hydrostatic(e: f64) -> Vec<f64> {
    CORNERS
        .iter()
        .flat_map(|c| [e * c[0], e * c[1], e * c[2]])
        .collect()
}

const CORNERS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// A single HEX8 material point: one cell, one law, one material.
struct Cube {
    nodes: Vec<Node>,
    fes: FiniteElementSpace,
    model: Model,
    materials: ElementField,
}

impl Cube {
    fn new(law: PlasticLaw, material: &[(&str, f64)]) -> Result<Self> {
        let coords = insert(Coords::new(3)?);
        let nodes: Vec<Node> = CORNERS
            .iter()
            .map(|c| Node::create_in(coords.clone(), c))
            .collect::<Result<_>>()?;
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
        let fes = FiniteElementSpace::lagrange1(&mesh)?;
        let model = Model::plasticity_with_law(&fes, ElasticityModel::Solid, law)?;
        let materials = material_field(&model, material)?;
        Ok(Self {
            nodes,
            fes,
            model,
            materials,
        })
    }

    fn displacement(&self, disp: &[f64]) -> Result<NodeField> {
        let support = insert(SubMesh::poi1_from_nodes(&self.nodes)?);
        let comps: Vec<String> = (0..3).map(|a| format!("u_{}", AXES[a])).collect();
        let mut u = SubNodeField::from_poi1(&support, comps)?;
        for (i, n) in self.nodes.iter().enumerate() {
            for a in 0..3 {
                u.set_value(n.id(), &format!("u_{}", AXES[a]), disp[i * 3 + a])?;
            }
        }
        Ok(NodeField::from_sub(u))
    }

    /// The converged state at the first Gauss point.
    fn stress(&self, disp: &[f64]) -> Result<State> {
        let strain = deformation(&self.displacement(disp)?, &self.fes)?;
        let state = integrate(&self.model, &strain, None, &self.materials, None)?;
        let sub = read(&state.get(0)?)?;
        let names = ["xx", "yy", "zz", "yz", "xz", "xy"];
        let mut sigma = [0.0; 6];
        for (i, n) in names.iter().enumerate() {
            sigma[i] = sub.value(0, 0, &format!("sigma_{n}"))?;
        }
        Ok(State {
            sigma,
            p: sub.value(0, 0, "p")?,
        })
    }

    fn internal_force_vec(&self, disp: &[f64]) -> Result<Vec<f64>> {
        let strain = deformation(&self.displacement(disp)?, &self.fes)?;
        let state = integrate(&self.model, &strain, None, &self.materials, None)?;
        let f = internal_forces(&state, &self.model)?;
        let mut out = vec![0.0; self.nodes.len() * 3];
        for (i, n) in self.nodes.iter().enumerate() {
            for a in 0..3 {
                out[i * 3 + a] = f.value(n.id(), &format!("f_{}", AXES[a]))?;
            }
        }
        Ok(out)
    }

    /// `K_t[i,j] = ∂f_int_i/∂u_j`, by central differences.
    ///
    /// The comparison is against the **symmetrised** difference, because that is
    /// what `D_alg` can carry: the state field stores it as an upper triangle
    /// and reads it back mirrored. For an associated law the symmetrisation is a
    /// no-op and this is the full check; for a non-associated one (Drucker-
    /// Prager) it is the exact statement of what is promised.
    fn check_tangent(&self, base: &[f64], tol: f64) -> Result<()> {
        let strain = deformation(&self.displacement(base)?, &self.fes)?;
        let state = integrate(&self.model, &strain, None, &self.materials, None)?;
        let kt = tangent(&self.model, &self.materials, &state)?;

        let ndof = self.nodes.len() * 3;
        let h = 1e-8;
        let mut fd_matrix = vec![vec![0.0; ndof]; ndof];
        for j in 0..ndof {
            let mut dp = base.to_vec();
            let mut dm = base.to_vec();
            dp[j] += h;
            dm[j] -= h;
            let (fp, fm) = (self.internal_force_vec(&dp)?, self.internal_force_vec(&dm)?);
            for i in 0..ndof {
                fd_matrix[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }

        // Both orders are read — `[i][j]` **and** `[j][i]`, to symmetrise — and
        // each index is also split into a node and an axis. Iterating the rows
        // would give access to one order only, so the range loop is the honest
        // form here.
        #[allow(clippy::needless_range_loop)]
        for j in 0..ndof {
            let (jn, ja) = (j / 3, j % 3);
            for i in 0..ndof {
                let (in_, ia) = (i / 3, i % 3);
                let fd = 0.5 * (fd_matrix[i][j] + fd_matrix[j][i]);
                let analytic = kt.get(
                    self.nodes[in_].id(),
                    &format!("f_{}", AXES[ia]),
                    self.nodes[jn].id(),
                    &format!("u_{}", AXES[ja]),
                )?;
                if (fd - analytic).abs() > tol * (analytic.abs() + 1.0) {
                    return Err(pyrucast::PyrucastError::Message(format!(
                        "K_t[{i},{j}] analytic {analytic} vs symmetrised finite difference {fd}"
                    )));
                }
            }
        }
        Ok(())
    }
}
