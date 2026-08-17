//! Follower pressure — a load that turns with the surface.
//!
//! The whole claim of a follower load is that its **direction depends on the
//! displacement**. So the tests do not check a value against a formula once;
//! they check that the load *moves* the way the surface does, and that it
//! degenerates to the dead load when the surface does not move.
//!
//! The test surface is the `x = 1` edge of the unit square — a unit-length
//! SEG2 wound so that its normal points along `+x`, i.e. outwards. A positive
//! pressure therefore pushes along `−x`.
//!
//! Three regimes pin the law down:
//!
//! | displacement | expected load |
//! |---|---|
//! | none | `(−p, 0)` — the dead load exactly |
//! | rigid rotation by `θ` | `(−p cosθ, −p sinθ)` — same magnitude, turned by `θ` |
//! | uniform stretch `λ` along the edge | `(−pλ, 0)` — the deformed **area** has grown |
//!
//! The second is the one a non-follower load gets wrong, and the third is the
//! one a follower load that only rotates its direction, forgetting the area
//! change, gets wrong.
//!
//! Single source for the « pression suiveuse » example of the mechanics book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::ops::element_field;
use pyrucast::store::Handle;
use pyrucast::Result;

const P: f64 = 3.0; // pressure

#[test]
fn an_undeformed_surface_gives_back_the_dead_load() -> Result<()> {
    let edge = Edge::new()?;
    let (fx, fy) = edge.load(|_x, _y| (0.0, 0.0))?;
    // Outward normal along +x, pressure pushing against it.
    assert!((fx + P).abs() < 1e-12, "f_x = {fx}");
    assert!(fy.abs() < 1e-12, "f_y = {fy}");
    Ok(())
}

#[test]
fn the_load_turns_with_the_surface() -> Result<()> {
    let edge = Edge::new()?;
    for angle_deg in [10.0_f64, 45.0, 90.0, 170.0] {
        let a = angle_deg.to_radians();
        let (c, s) = (a.cos(), a.sin());
        // A rigid rotation of the whole plane: u = (R − I)·x.
        let (fx, fy) = edge.load(|x, y| (c * x - s * y - x, s * x + c * y - y))?;
        // Same magnitude, turned by exactly the same angle.
        assert!(
            (fx + P * c).abs() < 1e-10 && (fy + P * s).abs() < 1e-10,
            "{angle_deg}°: ({fx}, {fy}), expected ({}, {})",
            -P * c,
            -P * s
        );
        let magnitude = (fx * fx + fy * fy).sqrt();
        assert!((magnitude - P).abs() < 1e-10, "magnitude {magnitude}");
    }
    Ok(())
}
// ANCHOR_END: example

/// Stretching the edge grows the surface the pressure acts on, so the total load
/// grows with it. That factor rides on the **magnitude** of the deformed normal,
/// which a follower load that only rotates its direction would normalise away.
#[test]
fn stretching_the_surface_grows_the_load() -> Result<()> {
    let edge = Edge::new()?;
    for lambda in [0.5_f64, 1.0, 1.5, 2.0] {
        // Stretch along the edge (y), leaving x alone.
        let (fx, fy) = edge.load(|_x, y| (0.0, (lambda - 1.0) * y))?;
        assert!(
            (fx + P * lambda).abs() < 1e-10,
            "λ = {lambda}: f_x = {fx}, expected {}",
            -P * lambda
        );
        assert!(fy.abs() < 1e-12, "f_y = {fy}");
    }
    Ok(())
}

/// A pressure acts on a surface. Handing it a cell that fills its space is a
/// modelling error — there is no normal to follow — and is reported as one.
#[test]
fn a_volumetric_subspace_is_rejected() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        .iter()
        .map(|c| Node::create_in(coords.clone(), c))
        .collect::<Result<_>>()?;
    let mut square = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    square.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())?;
    let err = Model::follower_pressure(&FiniteElementSpace::lagrange1(&square)?).unwrap_err();
    assert!(err.to_string().contains("not a boundary"), "{err}");
    Ok(())
}

/// Reversing the winding reverses the normal, hence the load. The orientation of
/// the boundary mesh is the user's statement of which side is « outside », and
/// nothing else can supply it.
#[test]
fn reversing_the_winding_reverses_the_load() -> Result<()> {
    let forward = Edge::new()?;
    let backward = Edge::reversed()?;
    let (fx, _) = forward.load(|_x, _y| (0.0, 0.0))?;
    let (bx, _) = backward.load(|_x, _y| (0.0, 0.0))?;
    assert!((fx + bx).abs() < 1e-12, "{fx} and {bx} must be opposite");
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The `x = 1` edge of the unit square, as a follower-pressure model.
struct Edge {
    nodes: Vec<Node>,
    fes: FiniteElementSpace,
    model: Model,
    materials: ElementField,
}

impl Edge {
    /// Wound bottom → top, so the normal points along `+x` (outwards).
    fn new() -> Result<Self> {
        Self::build(false)
    }

    /// Wound top → bottom: the same geometry, the opposite normal.
    fn reversed() -> Result<Self> {
        Self::build(true)
    }

    fn build(reversed: bool) -> Result<Self> {
        let coords = Handle::new(Coords::new(2)?);
        let bottom = Node::create_in(coords.clone(), &[1.0, 0.0])?;
        let top = Node::create_in(coords.clone(), &[1.0, 1.0])?;
        let nodes = if reversed {
            vec![top, bottom]
        } else {
            vec![bottom, top]
        };
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[nodes[0].id(), nodes[1].id()])?;
        let fes = FiniteElementSpace::lagrange1(&mesh)?;
        let model = Model::follower_pressure(&fes)?;
        let materials = element_field::material_field(&model, &[("p", P)])?;
        Ok(Self {
            nodes,
            fes,
            model,
            materials,
        })
    }

    /// The **total** nodal load for a displacement field given as `u(x, y)`.
    ///
    /// This is the follower pipeline end to end: the displacement is
    /// differentiated on the surface, the behaviour turns that into a traction,
    /// and the internal forces integrate it. Calling it again with another
    /// displacement is what « the load follows » means.
    fn load(&self, u: impl Fn(f64, f64) -> (f64, f64)) -> Result<(f64, f64)> {
        let sm = Handle::new(SubMesh::poi1_from_nodes(&self.nodes)?);
        let mut field = SubNodeField::from_poi1(&sm, vec!["u_x".to_string(), "u_y".to_string()])?;
        for n in &self.nodes {
            let p = n.position()?;
            let (ux, uy) = u(p[0], p[1]);
            field.set_value(n.id(), "u_x", ux)?;
            field.set_value(n.id(), "u_y", uy)?;
        }
        let displacement = NodeField::from_sub(field);

        let gradient = element_field::gradient(&displacement, &self.fes)?;
        let traction = element_field::behavior::integrate(
            &self.model,
            &gradient,
            None,
            &self.materials,
            None,
        )?;
        let forces = pyrucast::ops::node_field::internal_forces(&traction, &self.model)?;
        let mut total = (0.0, 0.0);
        for n in &self.nodes {
            total.0 += forces.value(n.id(), "f_x")?;
            total.1 += forces.value(n.id(), "f_y")?;
        }
        Ok(total)
    }
}
