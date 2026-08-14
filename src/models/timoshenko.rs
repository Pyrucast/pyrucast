//! Timoshenko beams — one physics, three configurations.
//!
//! The shear-deformable beam theory: plane sections stay plane but **not**
//! normal to the deflected axis, so the section rotation is a field of its own
//! and the transverse shear `γ = w' − θ` is a strain in its own right. That is
//! the whole difference from [Euler-Bernoulli](crate::models::bernoulli), where
//! `θ = w'` and no shear exists.
//!
//! ## Three configurations, no argument
//!
//! [`BeamModel`] is read from the mesh — see [`crate::models::beam`] for why
//! there is nothing to choose. What the *theory* adds on top of the shared
//! configuration is a shear compliance: material components `G` and `A_s` (the
//! shear area `κ·A`), and a shear force among the section forces.
//!
//! | `Coords` | DOFs per node | material | section forces |
//! |---|---|---|---|
//! | 1-D | `w`, `theta` | `E, I, G, A_s` | `M, V` |
//! | 2-D | `u_x, u_y, r_z` | `+ A` | `N, M, V` |
//! | 3-D | six | `E, A, I_y, I_z, J, G, A_sy, A_sz` | `N, M_y, M_z, T, V_y, V_z` |
//!
//! ## The element is exact, so the space owns no basis
//!
//! The stiffness is the closed form of the solution of the two coupled
//! equations on a span free of distributed load — [`crate::models::beam::bending_4x4`], driven
//! by `Φ = 12EI/(G·A_s·L²)`. It is **nodally exact** for end loads, so one
//! element per member suffices, and its shape functions are cubic in the
//! deflection and quadratic in the rotation.
//!
//! Those functions depend on the material through `Φ`, so no finite-element
//! space can tabulate them. The subspace therefore declares
//! [`ModelEmbedded`](crate::atoms::Interpolation::ModelEmbedded): the
//! formulation owns its interpolation, and says so rather than claiming a
//! Lagrange one it does not use.
//!
//! This replaces an earlier **linear** element with reduced shear integration,
//! which converged with mesh refinement instead of being exact. The two
//! theories now line up: Bernoulli integrates a Hermite basis it declares,
//! Timoshenko owns an exact one it declares owning.
//!
//! ## Mass
//!
//! The consistent mass is the **same element's**, integrated from the same
//! shape functions — so stiffness and mass finally describe one beam. Only the
//! axial and torsional degrees of freedom keep the linear field's
//! `(ρL/6)[[2,1],[1,2]]`, which is exact for what they actually interpolate.

use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::beam::{bending_4x4, mass_4x4, BeamModel};
use crate::models::owned_components;
use crate::models::{frame, frame3d, CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// The material a configuration needs. `G` and `A_s` are what the theory adds
/// over [Bernoulli](crate::models::bernoulli): the shear compliance.
fn material_of(model: BeamModel) -> &'static [&'static str] {
    match model {
        BeamModel::Planar1d => &["E", "I", "G", "A_s"],
        BeamModel::Frame2d => &["E", "A", "I", "G", "A_s"],
        BeamModel::Frame3d => &["E", "A", "I_y", "I_z", "J", "G", "A_sy", "A_sz"],
    }
}

/// The section forces the behaviour reports — a shear force in every one, the
/// theory having a shear strain to conjugate it with.
fn behavior_of(model: BeamModel) -> &'static [&'static str] {
    match model {
        BeamModel::Planar1d => &["M", "V"],
        BeamModel::Frame2d => &["N", "M", "V"],
        BeamModel::Frame3d => &["N", "M_y", "M_z", "T", "V_y", "V_z"],
    }
}

/// Timoshenko beam physics on a `SEG2` FE subspace.
#[derive(Clone, Serialize, Deserialize)]
pub struct Timoshenko {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the unique nodes (row/col support of the block).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) model: BeamModel,
}

impl Timoshenko {
    /// Timoshenko beam on a `SEG2` FE subspace. The configuration follows the
    /// dimension of the mesh; the subspace must be `MODEL_EMBEDDED`.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, et, axisymmetric, interpolation) = {
            let s = read(&fespace)?;
            (
                s.submesh(),
                s.space_dim(),
                s.element_type()?,
                s.is_axisymmetric(),
                s.interpolation(),
            )
        };
        if et != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "Timoshenko: a beam needs SEG2 elements, got {et}"
            )));
        }
        if !interpolation.is_model_embedded() {
            return Err(PyrucastError::Message(format!(
                "Timoshenko: this element is the exact beam, whose interpolation is \
                 cubic/quadratic and depends on the material through Φ = 12EI/(G·A_s·L²) — it is \
                 owned by the formulation, not by the space. Build the subspace with \
                 `Interpolation::ModelEmbedded`; got {interpolation}."
            )));
        }
        if axisymmetric {
            return Err(PyrucastError::Message(
                "Timoshenko: a segment in a meridian plane is a shell of revolution, not a beam"
                    .into(),
            ));
        }
        let model = BeamModel::from_space_dim(space_dim)
            .map_err(|e| PyrucastError::Message(format!("Timoshenko: {e}")))?;
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            model,
        })
    }

    /// The layout every block of this physics shares.
    fn layout(&self) -> MatrixLayout {
        MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        }
    }
}

impl SubModelKind for Timoshenko {
    fn primal_vars(&self) -> Vec<String> {
        self.model.primal().iter().map(|s| s.to_string()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        self.model.dual().iter().map(|s| s.to_string()).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(self.layout())
    }

    fn mass_layout(&self) -> Option<MatrixLayout> {
        Some(self.layout())
    }

    /// The geometric stiffness needs an axial force to be stiffened by, which a
    /// pure-bending configuration does not have.
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        match self.model {
            BeamModel::Planar1d => None,
            _ => Some(self.layout()),
        }
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Timoshenko declares a material_fespace");
        match self.model {
            BeamModel::Planar1d => planar_stiffness(geom, mat, ke),
            BeamModel::Frame2d => frame::element_stiffness(geom, mat, ke),
            BeamModel::Frame3d => frame3d::element_stiffness(geom, mat, ke),
        }
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Timoshenko declares a material_fespace");
        match self.model {
            BeamModel::Planar1d => planar_mass(geom, mat, ke),
            BeamModel::Frame2d => frame::element_mass(geom, mat, ke),
            BeamModel::Frame3d => frame3d::element_mass(geom, mat, ke),
        }
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("the geometric stiffness requires the axial force `N`");
        match self.model {
            BeamModel::Planar1d => Err(PyrucastError::Message(
                "Timoshenko: a pure-bending beam carries no axial force, so it has no geometric \
                 stiffness — use a 2-D or 3-D configuration"
                    .into(),
            )),
            BeamModel::Frame2d => frame::element_geometric(geom, stress, ke),
            BeamModel::Frame3d => frame3d::element_geometric(geom, stress, ke),
        }
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Timoshenko"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Timoshenko({})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Timoshenko {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(material_of(self.model)))
    }

    /// `rho` for the mass, and — in the 1-D configuration — the full area `A`,
    /// which only the mass needs (the stiffness uses the shear area).
    fn optional_material_components(&self) -> &'static [&'static str] {
        match self.model {
            BeamModel::Planar1d => &["A", "rho"],
            _ => &["rho"],
        }
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(behavior_of(self.model)
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// The section forces from the generalised strains — a **linear** law, as
    /// for every structural element.
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
        let mat = material.expect("Timoshenko declares a material_fespace");
        let cell = geom.cell;
        let v = |c| mat.value(cell, 0, c);
        let e = v("E")?;
        let gg = v("G")?;
        let read_in = |name: &str| input.value(cell, g, name);
        match self.model {
            BeamModel::Planar1d => {
                out[0] = e * v("I")? * read_in("kappa")?;
                out[1] = gg * v("A_s")? * read_in("gamma")?;
            }
            BeamModel::Frame2d => {
                out[0] = e * v("A")? * read_in("eps")?;
                out[1] = e * v("I")? * read_in("kappa")?;
                out[2] = gg * v("A_s")? * read_in("gamma")?;
            }
            BeamModel::Frame3d => {
                out[0] = e * v("A")? * read_in("eps")?;
                out[1] = e * v("I_y")? * read_in("kappa_y")?;
                out[2] = e * v("I_z")? * read_in("kappa_z")?;
                out[3] = gg * v("J")? * read_in("torsion")?;
                out[4] = gg * v("A_sy")? * read_in("gamma_y")?;
                out[5] = gg * v("A_sz")? * read_in("gamma_z")?;
            }
        }
        Ok(())
    }
}

// ─── The 1-D configuration's own kernels ────────────────────────────────────
//
// The plane and space frames delegate to [`crate::models::frame`] and
// [`crate::models::frame3d`], which hold the rotations to the global axes. A
// 1-D beam has no rotation to make, so its kernels are the bare blocks.

/// The exact bending stiffness of a 1-D beam, on `[w_A, θ_A, w_B, θ_B]`.
fn planar_stiffness(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let cell = geom.cell;
    let (xa, xb) = (geom.node_coord(0)?, geom.node_coord(1)?);
    let l = (xb[0] - xa[0]).abs();
    if l <= f64::EPSILON {
        return Err(PyrucastError::Message(format!(
            "Timoshenko: cell {cell} has zero length"
        )));
    }
    let ei = material.value(cell, 0, "E")? * material.value(cell, 0, "I")?;
    let gas = material.value(cell, 0, "G")? * material.value(cell, 0, "A_s")?;
    let k = bending_4x4(ei, Some(gas), l);
    for (r, row) in k.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            ke[r * 4 + c] += v;
        }
    }
    Ok(())
}

/// Consistent **mass** of a 1-D beam — translations `(ρAL/6)[[2,1],[1,2]]` and
/// rotary inertia `(ρIL/6)[[2,1],[1,2]]`, in DOF order `[w0, θ0, w1, θ1]`.
///
/// This is the *linear* element's mass, kept as it was; see the module note.
fn planar_mass(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let cell = geom.cell;
    let (xa, xb) = (geom.node_coord(0)?, geom.node_coord(1)?);
    let l = (xb[0] - xa[0]).abs();
    let rho = material.value(cell, 0, "rho").map_err(|_| {
        PyrucastError::Message(
            "Timoshenko mass matrix: material component `rho` (density) is required".into(),
        )
    })?;
    let i = material.value(cell, 0, "I")?;
    let ei = material.value(cell, 0, "E")? * i;
    let gas = material.value(cell, 0, "G")? * material.value(cell, 0, "A_s")?;
    let m = mass_4x4(
        rho * material.value(cell, 0, "A")?,
        rho * i,
        ei,
        Some(gas),
        l,
    );
    for (r, row) in m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            ke[r * 4 + c] += v;
        }
    }
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{Interpolation, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::store::insert;

    /// One `SEG2` from `a` to `b`, in whatever dimension the coordinates have.
    fn one_beam(a: &[f64], b: &[f64]) -> (Timoshenko, NodeId, NodeId) {
        let coords = insert(Coords::new(a.len() as u8).unwrap());
        let na = Node::create_in(coords.clone(), a).unwrap();
        let nb = Node::create_in(coords.clone(), b).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[na.id(), nb.id()]).unwrap();
        let fes = FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded).unwrap();
        let beam = Timoshenko::new(fes.get(0).unwrap()).unwrap();
        (beam, na.id(), nb.id())
    }

    fn material(beam: &Timoshenko, pairs: &[(&str, f64)]) -> Handle<SubElementField> {
        let names: Vec<String> = pairs.iter().map(|(c, _)| (*c).to_string()).collect();
        let mut m = SubElementField::new(beam.fespace.clone(), names).unwrap();
        for (c, v) in pairs {
            m.set_uniform(c, *v).unwrap();
        }
        insert(m)
    }

    /// The configuration is read from the mesh, and with it every DOF name.
    #[test]
    fn the_configuration_follows_the_dimension() {
        let (b1, _, _) = one_beam(&[0.0], &[2.0]);
        assert_eq!(b1.primal_vars(), ["w", "theta"]);
        let (b2, _, _) = one_beam(&[0.0, 0.0], &[2.0, 0.0]);
        assert_eq!(b2.primal_vars(), ["u_x", "u_y", "r_z"]);
        let (b3, _, _) = one_beam(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0]);
        assert_eq!(b3.primal_vars().len(), 6);
    }

    /// A Lagrange subspace is refused, and the message says why rather than
    /// leaving the caller to guess which space a beam wants.
    #[test]
    fn a_lagrange_subspace_is_refused() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let err = match Timoshenko::new(fes.get(0).unwrap()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a Lagrange subspace should be refused"),
        };
        assert!(err.contains("ModelEmbedded"), "message: {err}");
    }

    /// The exact element combines the bending and shear compliances **in
    /// series** — bend the member and shear it, and the two give way one after
    /// the other. That is the physical content of the `Φ` correction, and it
    /// holds in every configuration.
    #[test]
    fn the_tip_stiffness_is_the_two_compliances_in_series() {
        let l = 2.0;
        let (e, i, g, a_s) = (3.0, 2.0, 5.0, 2.0);
        let (ei, gas) = (e * i, g * a_s);
        let series = 1.0 / (l * l * l / (12.0 * ei) + l / gas);

        // 1-D: the bare block.
        let (beam, a, _) = one_beam(&[0.0], &[l]);
        let mat = material(&beam, &[("E", e), ("I", i), ("G", g), ("A_s", a_s)]);
        let k = &beam.build_stiffness_blocks(Some(&mat)).unwrap()[0];
        assert!((k.get(a, "f_w", a, "w") - series).abs() < 1e-9);

        // 2-D: the same block, rotated, and decoupled from the axial term.
        let (frame, a2, _) = one_beam(&[0.0, 0.0], &[l, 0.0]);
        let mat = material(
            &frame,
            &[("E", e), ("A", 4.0), ("I", i), ("G", g), ("A_s", a_s)],
        );
        let k = &frame.build_stiffness_blocks(Some(&mat)).unwrap()[0];
        assert!((k.get(a2, "f_y", a2, "u_y") - series).abs() < 1e-9);
        assert!((k.get(a2, "f_x", a2, "u_x") - e * 4.0 / l).abs() < 1e-9);
        assert!(k.get(a2, "f_x", a2, "u_y").abs() < 1e-9);
    }

    /// A vertical member puts the axial term on `u_y` — the local axes are read
    /// from the geometry, with no orientation data to supply.
    #[test]
    fn a_vertical_member_finds_its_own_axes() {
        let l = 2.0;
        let (e, area) = (3.0, 4.0);
        let (frame, a, _) = one_beam(&[0.0, 0.0], &[0.0, l]);
        let mat = material(
            &frame,
            &[("E", e), ("A", area), ("I", 2.0), ("G", 5.0), ("A_s", 2.0)],
        );
        let k = &frame.build_stiffness_blocks(Some(&mat)).unwrap()[0];
        assert!((k.get(a, "f_y", a, "u_y") - e * area / l).abs() < 1e-9);
        assert!(k.get(a, "f_y", a, "u_x").abs() < 1e-9);
    }

    /// A pure-bending beam has no axial force, so it declares no geometric
    /// stiffness — rather than assembling an empty one that would read as "this
    /// member cannot buckle".
    #[test]
    fn a_pure_bending_beam_declares_no_geometric_stiffness() {
        let (b1, _, _) = one_beam(&[0.0], &[2.0]);
        assert!(b1.geometric_layout().is_none());
        let (b2, _, _) = one_beam(&[0.0, 0.0], &[2.0, 0.0]);
        assert!(b2.geometric_layout().is_some());
    }
}
