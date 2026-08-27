//! Follower pressure — a load that turns with the surface it acts on.
//!
//! A pressure is always normal to the surface it presses. As the body deforms,
//! that surface **moves and tilts**, so the load direction moves with it: a
//! pressure is a *follower* load. Ignoring this is exact only in small
//! displacements; on an inflating membrane, a buckling shell or a rotating blade
//! it is not.
//!
//! ```text
//! t = −p · n(u)          n(u) the normal of the **deformed** surface
//! ```
//!
//! ## Why it is a model and not a load
//!
//! A dead load is built once with
//! [`flux`](fn@crate::ops::node_field::flux) and never looked at again. A follower
//! pressure cannot be: its direction depends on the current displacement, so it
//! has to be **recomputed at every residual evaluation**. That is precisely what
//! a physics does — it integrates a behaviour and contributes to the internal
//! forces — so it is one:
//!
//! ```text
//! u  ──gradient──▶  ∇_s u  ──integrate_behavior──▶  t(u)  ──internal_forces──▶  f(u)
//! ```
//!
//! The behaviour integration is where the direction is refreshed. Nothing else
//! in the pipeline changes.
//!
//! ## The deformed normal, from the deformed tangents
//!
//! The direction **and** the area change both come from the tangents of the
//! surface. If `a_k = ∂x/∂ξ_k` are the reference tangents, the deformed ones are
//!
//! ```text
//! ā_k = a_k + ∂u/∂ξ_k = a_k + (∇_s u)·a_k
//! ```
//!
//! and the normal times the area ratio is their cross product (their −90° turn
//! in 2-D) divided by the reference one:
//!
//! ```text
//! t = −p · (ā₁ × ā₂) / |a₁ × a₂|         (3-D)
//! t = −p · (ā_y, −ā_x) / |a|             (2-D)
//! ```
//!
//! Keeping the traction **referential** is what lets the internal-force integral
//! use the ordinary reference measure: the formulation stays total-Lagrangian,
//! and with no displacement it gives back `t = −p·N` exactly.
//!
//! ### Why not Nanson
//!
//! `n da = det(F)·F⁻ᵀ·N dA` is the textbook route, and it is the wrong one
//! **here**. On a manifold the tangential gradient has no component along the
//! normal, so `I + ∇_s u` is not a deformation gradient: a quarter-turn of the
//! surface sends its determinant to zero and the formula blows up on a
//! perfectly ordinary rotation. The tangents never degenerate that way — they
//! rotate with the surface — so they are what a surface load must be built on.
//!
//! ## Orientation is the mesh's business
//!
//! The normal follows the boundary mesh's **winding**
//! ([`CellGeom::normal`](crate::models::kernel::CellGeom::normal)). A positive
//! `p` pushes **against** it — compressive — so an outward-wound boundary gives
//! the usual sign. This is the one place where a boundary mesh's orientation
//! matters: contrast [`boundary_transfer`](crate::models::boundary_transfer), whose direction
//! is already consumed in writing `q·n` and which is therefore
//! orientation-blind.
//!
//! ## What it contributes
//!
//! Internal forces only. It declares a `stiffness_layout` — that is what the
//! internal-force scatter is driven from — but **contributes no matrix**: its
//! `contributions()` is empty for every kind. The load-correction stiffness
//! `∂f/∂u` (non-symmetric) is not implemented; a Newton loop converges without
//! it, more slowly.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::owned_components;
use crate::models::tensor::{dual_name, primal_name};
use crate::models::ZoneLayout;
use crate::models::{
    CellGeom, Contribution, Domain, MatrixKind, MatrixLayout, Physics, SubModelKind,
};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Required material component: the pressure.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::follower_pressure;
/// // La pression appliquée, fournie au moment de l'assemblage.
/// assert_eq!(follower_pressure::MATERIAL_COMPONENT, "p");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL_COMPONENT: &str = "p";
/// Material contract returned by [`Domain::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Behaviour-**output** components: the referential traction.
fn traction_names(space_dim: usize) -> Vec<String> {
    (0..space_dim).map(|a| format!("t_{}", AXES[a])).collect()
}

/// Follower pressure on a boundary FE subspace.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::follower_pressure::{self, FollowerPressure};
/// // Une pression qui tourne avec la surface : une seule constante.
/// let f = FollowerPressure::new(zone.clone())?;
/// assert_eq!(f.material_components(),
///            vec![follower_pressure::MATERIAL_COMPONENT.to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct FollowerPressure {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the boundary's unique nodes.
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
}

impl FollowerPressure {
    /// Follower pressure on a **boundary** FE subspace — an edge mesh in 2-D, a
    /// surface mesh in 3-D. Errors on anything else: a pressure acts on a
    /// surface, and a cell that fills its space has no normal to follow.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::follower_pressure::{self, FollowerPressure};
    /// // Une pression qui tourne avec la surface : une seule constante.
    /// let f = FollowerPressure::new(zone.clone())?;
    /// assert_eq!(f.material_components(),
    ///            vec![follower_pressure::MATERIAL_COMPONENT.to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, ref_dim) = {
            let s = fespace.read();
            (s.submesh(), s.space_dim(), s.ref_dim()?)
        };
        if ref_dim + 1 != space_dim {
            return Err(PyrucastError::Message(format!(
                "FollowerPressure: a {ref_dim}-D element in a {space_dim}-D space is not a \
                 boundary — a pressure acts on a surface (SEG2 in 2-D, TRI3/QUA4 in 3-D), \
                 and needs a normal to follow"
            )));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
        })
    }
}

impl SubModelKind for FollowerPressure {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    /// Declared so the internal-force scatter knows which subspace and support
    /// to run on — **not** to contribute a matrix. See
    /// [`contributions`](Self::contributions).
    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: false,
        })
    }

    /// A follower pressure contributes **no matrix at all** — it is a load, and
    /// its whole effect lives in the internal forces. Overriding this (rather
    /// than dropping the layout) is what lets it keep a `stiffness_layout` for
    /// the internal-force scatter without an assembler ever asking it for an
    /// element matrix it does not have.
    fn contributions(
        &self,
        _kind: MatrixKind,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        Ok(Vec::new())
    }

    /// The consistent nodal load `f_{i,a} = ∫_Γ N_i · t_a dΓ`, integrated on the
    /// **reference** surface — the traction already carries the area change.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = self.space_dim;
        let names = traction_names(d);
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g)?;
            let w = geom.det_j_w(g)?;
            for i in 0..geom.n_nodes {
                for (a, name) in names.iter().enumerate() {
                    fe[i * d + a] += shape[i] * stress.value(geom.cell, g, name)? * w;
                }
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "FollowerPressure"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<FollowerPressure>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

impl Domain for FollowerPressure {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Vec<String> {
        owned_components(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        traction_names(self.space_dim)
    }

    /// The referential traction at one Gauss point, from the deformed tangents.
    ///
    /// This is where the direction is refreshed: call it again with an updated
    /// displacement and the load has turned with the surface.
    fn deformation_reads(&self) -> Vec<String> {
        let d = self.space_dim;
        let mut names = Vec::with_capacity(d * d);
        for a in 0..d {
            for b in 0..d {
                names.push(format!("grad_u_{}_{}", AXES[a], AXES[b]));
            }
        }
        names
    }

    fn integrate_point(
        &self,
        geom: &CellGeom,
        g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        let (cell, d) = (geom.cell, self.space_dim);
        let p = material[lay.material[0] as usize];

        // ∇_s u, the tangential gradient of the displacement, row-major over
        // `deformation_reads`' own `(a, b)` order.
        let mut grad = [0.0_f64; 9];
        for k in 0..d * d {
            grad[k] = deformation[lay.deformation[k] as usize];
        }

        // The deformed tangents ā_k = a_k + (∇_s u)·a_k, and the reference
        // measure |a₁ × a₂| that turns the result into a *referential* traction.
        let reference = geom.tangents(g)?;
        let deformed: Vec<Vec<f64>> = reference
            .iter()
            .map(|a| {
                (0..d)
                    .map(|i| a[i] + (0..d).map(|j| grad[i * d + j] * a[j]).sum::<f64>())
                    .collect()
            })
            .collect();

        let n_ref = CellGeom::normal_from_tangents(&reference)?;
        let area_ref = n_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
        if area_ref <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "FollowerPressure: cell {cell} is degenerate at Gauss point {g} (null area)"
            )));
        }
        let n_def = CellGeom::normal_from_tangents(&deformed)?;

        // The pressure pushes **against** the normal, hence the minus sign; the
        // magnitude of `n_def` already carries the area change.
        for (a, o) in out.iter_mut().enumerate().take(d) {
            *o = -p * n_def[a] / area_ref;
        }
        Ok(())
    }
}

crate::physics_operator! {
    /// Follower-pressure `Model` spanning **every** subspace of a *boundary*
    /// `fes`. Parent-level operator; `p` is supplied at assembly time.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::models::{Physics, RelationSense};
    /// # use pyrucast::ops::mesh;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let mut bord = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # bord.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let fes_bord = FiniteElementSpace::lagrange1(&Mesh::from_submesh(bord))?;
    /// let m = model::follower_pressure(&fes_bord)?;
    /// assert_eq!(m.len(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn follower_pressure(fes) via SubModel::follower_pressure;
    python: "`model.follower_pressure(fespace)` — a pressure that **turns with the\nsurface** it acts on, on a *boundary* `fespace` (an edge mesh in 2-D, a\nsurface mesh in 3-D). Material: `p`, the pressure.\n\nUnlike a dead load built once with `flux(...)`, its direction depends on\nthe current displacement, so it is recomputed at each residual\nevaluation:\n\n```text\nu → element_field.gradient → integrate_behavior → node_field.internal_forces\n```\n\nIt contributes **no matrix** — only internal forces. A positive `p`\npushes *against* the boundary mesh's own normal, which follows its\nwinding: orienting the boundary outwards gives the usual compressive\nsign."
}
