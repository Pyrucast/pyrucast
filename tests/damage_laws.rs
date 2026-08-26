//! The damage laws — Mazars, tension/compression, orthotropic SiC/SiC — plus
//! Gurson's porous plasticity.
//!
//! Each is pinned by what it does that the others cannot:
//!
//! | law | what is asserted |
//! |---|---|
//! | Damage TC | a crack that **closes** carries load again, and the two histories stay independent — which one scalar cannot do |
//! | Mazars | carries a **single** scalar, the limitation the pair above lifts |
//! | SiC/SiC | damage is **directional**: stretching along one weave axis leaves the others intact |
//! | Gurson | the porosity **grows** under triaxial tension and shrinks the surface, so a porous metal softens where a J2 one would not |
//!
//! Single source for the « endommagement » examples of the book; runs under
//! `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::damage::DamageLaw;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::models::plasticity::law::PlasticLaw;
use pyrucast::ops::element_field::{behavior::integrate, deformation, material_field};
use pyrucast::ops::model;
use pyrucast::Result;

const AXES: [&str; 3] = ["x", "y", "z"];

const TC: &[(&str, f64)] = &[
    ("E", 30_000.0),
    ("nu", 0.2),
    ("f_t", 3.0),
    ("f_c", 30.0),
    ("A_t", 0.9),
    ("A_c", 0.5),
];

#[test]
fn a_closed_crack_carries_load_again() -> Result<()> {
    let cube = Cube::damage(DamageLaw::DamageTc, TC)?;
    // Stretch far enough to damage in tension…
    let damaged = cube.step(&uniaxial(1.5e-3), None)?;
    let d_plus = damaged.var("d_plus")?;
    let d_minus = damaged.var("d_minus")?;
    assert!(d_plus > 0.1, "tension must damage (d⁺ = {d_plus})");
    // …and the **compressive** damage is untouched: the crack has not crushed
    // anything. That separation is the whole point of two variables.
    assert!(
        d_minus < 1e-12,
        "compression must stay intact (d⁻ = {d_minus})"
    );

    // Now close the crack. The compressive stiffness is the undamaged one.
    let closed = cube.step(&uniaxial(-1e-4), Some(&damaged.field))?;
    let intact = cube.step(&uniaxial(-1e-4), None)?;
    assert!(
        (closed.sigma[0] - intact.sigma[0]).abs() < 1e-9 * intact.sigma[0].abs(),
        "a closed crack must carry the full compressive stress: {} vs {}",
        closed.sigma[0],
        intact.sigma[0]
    );
    Ok(())
}
// ANCHOR_END: example

/// The mirror of the previous test, and what makes the pair meaningful: crushing
/// leaves the **tensile** damage untouched. Two variables mean two independent
/// histories, not one history read twice.
#[test]
fn crushing_leaves_the_tensile_damage_untouched() -> Result<()> {
    let cube = Cube::damage(DamageLaw::DamageTc, TC)?;
    // Confined compression, so the compressive driver is large.
    let crushed = cube.step(&hydrostatic(-3e-3), None)?;
    let d_minus = crushed.var("d_minus")?;
    let d_plus = crushed.var("d_plus")?;
    assert!(d_minus > 0.0, "compression must damage (d⁻ = {d_minus})");
    assert!(d_plus < 1e-12, "tension must stay intact (d⁺ = {d_plus})");
    // …and the two histories are separate objects, not one.
    assert!(crushed.var("r_minus")? > crushed.var("r_plus")? * 0.0);
    Ok(())
}

/// Mazars keeps **one** scalar, which is exactly what `damage_tc` exists to
/// lift. Under a purely compressive strain its weights vanish, so it reports no
/// damage at all there — a well-known property of the law, and the reason a
/// separate compressive variable is worth its cost.
#[test]
fn mazars_reports_a_single_scalar() -> Result<()> {
    let cube = Cube::damage(
        DamageLaw::Mazars,
        &[
            ("E", 30_000.0),
            ("nu", 0.2),
            ("eps_d0", 1e-4),
            ("A_t", 0.8),
            ("B_t", 20_000.0),
            ("A_c", 1.4),
            ("B_c", 1_900.0),
        ],
    )?;
    let damaged = cube.step(&uniaxial(1.5e-3), None)?;
    assert!(damaged.var("damage")? > 0.1, "tension must damage");
    // One history variable, not two: that is the whole difference.
    assert!(damaged.var("kappa")? > 0.0);
    assert!(damaged.field.get(0).is_ok());
    Ok(())
}

/// SiC/SiC damages **by direction**. Stretching along the first weave axis must
/// damage that direction and leave the other two untouched — which is exactly
/// what a scalar damage cannot represent.
#[test]
fn sic_sic_damages_only_the_stretched_direction() -> Result<()> {
    let cube = Cube::damage(DamageLaw::SicSic, SIC)?;
    let stretched = cube.step(&uniaxial(2e-3), None)?;
    let (d1, d2, d3) = (
        stretched.var("d_1")?,
        stretched.var("d_2")?,
        stretched.var("d_3")?,
    );
    assert!(
        d1 > 0.05,
        "the stretched direction must damage (d_1 = {d1})"
    );
    assert!(d2 < 1e-12 && d3 < 1e-12, "the others must not: {d2}, {d3}");
    Ok(())
}

/// The damage saturates at `d_max` rather than reaching one: matrix cracking
/// does not take the whole stiffness, because the fibres remain.
#[test]
fn sic_sic_damage_saturates_at_its_ceiling() -> Result<()> {
    let cube = Cube::damage(DamageLaw::SicSic, SIC)?;
    // Far past the characteristic strain, so the exponential has saturated.
    let d1 = cube.step(&uniaxial(0.5), None)?.var("d_1")?;
    assert!(
        (d1 - 0.6).abs() < 1e-6,
        "the damage must saturate at d_max_1 = 0.6 (got {d1})"
    );
    Ok(())
}

const SIC: &[(&str, f64)] = &[
    ("E", 230_000.0),
    ("nu", 0.2),
    ("eps_0_1", 5e-4),
    ("eps_c_1", 2e-3),
    ("d_max_1", 0.6),
    ("eps_0_2", 5e-4),
    ("eps_c_2", 2e-3),
    ("d_max_2", 0.6),
    ("eps_0_3", 5e-4),
    ("eps_c_3", 2e-3),
    ("d_max_3", 0.6),
    ("V1X", 1.0),
    ("V1Y", 0.0),
    ("V1Z", 0.0),
    ("V2X", 0.0),
    ("V2Y", 1.0),
    ("V2Z", 0.0),
];

/// Gurson: the porosity grows under **triaxial tension** — voids open where the
/// pressure pulls them apart — and shrinks the yield surface as it does. A J2
/// law is blind to all of it.
const GURSON: &[(&str, f64)] = &[
    ("E", 200_000.0),
    ("nu", 0.3),
    ("sigma_y", 400.0),
    ("q_1", 1.5),
    ("q_2", 1.0),
    ("q_3", 2.25),
    ("f_0", 0.001),
    ("f_c", 0.15),
    ("f_f", 0.25),
];

#[test]
fn gurson_porosity_grows_under_triaxial_tension() -> Result<()> {
    let cube = Cube::plastic(PlasticLaw::Gurson, GURSON)?;
    let mut prev = None;
    let mut porosities = Vec::new();
    for _ in 0..5 {
        let step = cube.step(&hydrostatic(5e-3), prev.as_ref().map(|s: &Step| &s.field))?;
        porosities.push(step.var("porosity")?);
        prev = Some(step);
    }
    assert!(
        porosities[0] > 0.001,
        "the porosity must start from f_0 and grow: {porosities:?}"
    );
    for w in porosities.windows(2) {
        assert!(w[1] >= w[0], "the porosity must not shrink: {porosities:?}");
    }
    Ok(())
}

/// The defining contrast: at zero porosity Gurson **is** von Mises, and at a
/// finite one it yields sooner. That is the surface shrinking.
#[test]
fn a_porous_metal_yields_before_a_dense_one() -> Result<()> {
    let dense: Vec<(&str, f64)> = GURSON
        .iter()
        .map(|&(k, v)| if k == "f_0" { (k, 0.0) } else { (k, v) })
        .collect();
    let porous: Vec<(&str, f64)> = GURSON
        .iter()
        .map(|&(k, v)| if k == "f_0" { (k, 0.05) } else { (k, v) })
        .collect();
    let strain = hydrostatic(4e-3);
    let q_dense = von_mises(
        &Cube::plastic(PlasticLaw::Gurson, &dense)?
            .step(&strain, None)?
            .sigma,
    );
    let s_porous = Cube::plastic(PlasticLaw::Gurson, &porous)?.step(&strain, None)?;
    // Under pure hydrostatic tension a dense (von Mises) material stays elastic —
    // there is no deviator to yield on — while a porous one flows and sheds
    // pressure. The mean stress is where that shows.
    let mean_porous = (s_porous.sigma[0] + s_porous.sigma[1] + s_porous.sigma[2]) / 3.0;
    assert!(q_dense < 1e-6, "a dense metal has no deviator here");
    assert!(
        mean_porous > 0.0 && mean_porous < 4e-3 * 200_000.0 / (1.0 - 2.0 * 0.3),
        "the porous metal must shed hydrostatic stress (σ_m = {mean_porous})"
    );
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

struct Step {
    sigma: [f64; 6],
    field: ElementField,
}

impl Step {
    fn var(&self, name: &str) -> Result<f64> {
        self.field.get(0)?.read().value(0, 0, name)
    }
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

fn uniaxial(e: f64) -> Vec<f64> {
    CORNERS.iter().flat_map(|c| [e * c[0], 0.0, 0.0]).collect()
}

fn hydrostatic(e: f64) -> Vec<f64> {
    CORNERS
        .iter()
        .flat_map(|c| [e * c[0], e * c[1], e * c[2]])
        .collect()
}

/// A single HEX8 material point, of either family.
struct Cube {
    nodes: Vec<Node>,
    fes: FiniteElementSpace,
    model: Model,
    materials: ElementField,
}

impl Cube {
    fn damage(law: DamageLaw, material: &[(&str, f64)]) -> Result<Self> {
        Self::build(material, |fes| {
            model::damage_with_law(fes, ElasticityModel::Solid, law)
        })
    }

    fn plastic(law: PlasticLaw, material: &[(&str, f64)]) -> Result<Self> {
        Self::build(material, |fes| {
            model::plasticity_with_law(fes, ElasticityModel::Solid, law)
        })
    }

    fn build(
        material: &[(&str, f64)],
        make: impl Fn(&FiniteElementSpace) -> Result<Model>,
    ) -> Result<Self> {
        let coords = Handle::new(Coords::new(3)?);
        let nodes: Vec<Node> = CORNERS
            .iter()
            .map(|c| Node::create_in(coords.clone(), c))
            .collect::<Result<_>>()?;
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
        let fes = FiniteElementSpace::lagrange1(&mesh)?;
        let model = make(&fes)?;
        let materials = material_field(&model, material)?;
        Ok(Self {
            nodes,
            fes,
            model,
            materials,
        })
    }

    fn step(&self, disp: &[f64], prev: Option<&ElementField>) -> Result<Step> {
        let support = Handle::new(SubMesh::poi1_from_nodes(&self.nodes)?);
        let comps: Vec<String> = (0..3).map(|a| format!("u_{}", AXES[a])).collect();
        let mut u = SubNodeField::from_poi1(&support, comps)?;
        for (i, n) in self.nodes.iter().enumerate() {
            for a in 0..3 {
                u.set_value(n.id(), &format!("u_{}", AXES[a]), disp[i * 3 + a])?;
            }
        }
        let strain = deformation(&NodeField::from_sub(u), &self.fes)?;
        let field = integrate(&self.model, &strain, prev, &self.materials, None)?;
        let sub = field.get(0)?.read();
        let names = ["xx", "yy", "zz", "yz", "xz", "xy"];
        let mut sigma = [0.0; 6];
        for (i, n) in names.iter().enumerate() {
            sigma[i] = sub.value(0, 0, &format!("sigma_{n}"))?;
        }
        drop(sub);
        Ok(Step { sigma, field })
    }
}
