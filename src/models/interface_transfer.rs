//! Exchange law across an **interface** between two meshes.
//!
//! Two bodies meeting along a surface do not, in general, share their nodes: an
//! imperfect contact, a coating, a joint, a membrane all let the field **jump**
//! across the interface while a flux crosses it in proportion to that jump:
//!
//! ```text
//! j·n = h · (c₁ − c₂)
//! ```
//!
//! `h` is the transfer coefficient (its inverse is the contact resistance). The
//! same law describes a thermal contact resistance, with `T` and `q` in place of
//! `c` and `j` — hence the [`TransferKind`] argument of
//! [`Model::interface_transfer`](crate::containers::model::Model::interface_transfer),
//! which picks the variable names and the physics nature and changes nothing
//! else: the mathematics is identical.
//!
//! ## Four blocks, two of them off-diagonal
//!
//! The weak form of the exchange term over the interface `Γ` is
//!
//! ```text
//! ∮_Γ h (c₁ − c₂)(δc₁ − δc₂) dΓ
//! ```
//!
//! which expands into a **2×2 block structure** on the two sides' DOFs:
//!
//! ```text
//! ⎡ +K  −K ⎤          with   K_ij = h ∫_Γ N_i N_j dΓ
//! ⎣ −K  +K ⎦
//! ```
//!
//! The two diagonal blocks are ordinary [`Contribution::Computed`] blocks — rows
//! and columns on one mesh. The two off-diagonal ones have their rows on one mesh
//! and their columns on the other, which is exactly
//! [`Contribution::Coupling`]. This physics is its first user.
//!
//! The **sign** rides on the kernel, not on a factor threaded through the
//! assembler: [`SubModelKind::element_matrix`] gives `+h∫N_iN_j` and
//! [`SubModelKind::coupling_element`] gives `−h∫N_iN_j`. Because each block picks
//! its kernel from its own contribution variant, the assembler needs to know
//! nothing about interfaces.
//!
//! ## Conformité
//!
//! The two sides must be **conforming**: same element type, same cell count,
//! cell `i` facing cell `i`, and node `k` of a cell facing node `k` of its
//! counterpart. That is checked geometrically at construction — the paired nodes
//! must be co-located — and reported rather than approximated. A non-matching
//! interface is a meshing problem; papering over it with a projection would be a
//! silent source of wrong fluxes.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::coords::Coords;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{
    CellGeom, Contribution, CouplingLayout, Domain, MatrixKind, MatrixLayout, Physics, SubModelKind,
};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Required material component: the transfer coefficient.
pub const MATERIAL_COMPONENT: &str = "h";
/// Material contract returned by [`Domain::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Which field the interface law transports — it fixes the variable names and
/// the physics nature, nothing else. The mathematics is identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferKind {
    /// Mass transfer: concentration `c`, flux `j`, nature `Diffusion`.
    Mass,
    /// Thermal contact resistance: temperature `T`, flux `q`, nature `Thermal`.
    Thermal,
}

impl TransferKind {
    /// Parse from a lowercase tag (`"mass"`, `"thermal"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "mass" => Some(Self::Mass),
            "thermal" => Some(Self::Thermal),
            _ => None,
        }
    }

    /// The lowercase tag (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Mass => "mass",
            Self::Thermal => "thermal",
        }
    }

    /// `(primal, dual)` variable names — deliberately the **same** as the bulk
    /// physics they border, so the interface term couples straight into it by
    /// union of DOFs.
    fn vars(self) -> (&'static str, &'static str) {
        match self {
            Self::Mass => ("c", "j"),
            Self::Thermal => ("T", "q"),
        }
    }

    fn physics(self) -> &'static [Physics] {
        match self {
            Self::Mass => &[Physics::Diffusion],
            Self::Thermal => &[Physics::Thermal],
        }
    }
}

/// Exchange law between two conforming boundary FE subspaces.
#[derive(Clone, Serialize, Deserialize)]
pub struct InterfaceTransfer {
    pub(crate) side_a: Handle<SubFiniteElementSpace>,
    pub(crate) side_b: Handle<SubFiniteElementSpace>,
    /// POI1 supports over each side's unique nodes.
    pub(crate) support_a: Handle<SubMesh>,
    pub(crate) support_b: Handle<SubMesh>,
    pub(crate) kind: TransferKind,
}

impl InterfaceTransfer {
    /// Exchange law across the interface between two **conforming** boundary FE
    /// subspaces. Errors unless the two sides match cell for cell and node for
    /// node, within `tol` of each other geometrically.
    pub fn new(
        side_a: Handle<SubFiniteElementSpace>,
        side_b: Handle<SubFiniteElementSpace>,
        kind: TransferKind,
        tol: f64,
    ) -> Result<Self> {
        let (mesh_a, mesh_b) = (read(&side_a)?.submesh(), read(&side_b)?.submesh());
        check_conforming_geometry(&mesh_a, &mesh_b, tol)?;
        let support_a = read(&mesh_a)?.to_poi1()?;
        let support_b = read(&mesh_b)?.to_poi1()?;
        Ok(Self {
            side_a,
            side_b,
            support_a,
            support_b,
            kind,
        })
    }

    /// The layout of a diagonal block, on one side of the interface.
    fn diagonal_layout(
        &self,
        fespace: &Handle<SubFiniteElementSpace>,
        support: &Handle<SubMesh>,
    ) -> MatrixLayout {
        MatrixLayout {
            fespaces: vec![fespace.clone()],
            support: support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        }
    }

    /// The layout of an off-diagonal block, rows on one side, columns on the other.
    fn coupling_layout(
        &self,
        row_fespace: &Handle<SubFiniteElementSpace>,
        row_support: &Handle<SubMesh>,
        col_fespace: &Handle<SubFiniteElementSpace>,
        col_support: &Handle<SubMesh>,
    ) -> CouplingLayout {
        CouplingLayout {
            fespaces: vec![row_fespace.clone()],
            col_fespaces: vec![col_fespace.clone()],
            row_support: row_support.clone(),
            col_support: col_support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
        }
    }
}

impl SubModelKind for InterfaceTransfer {
    fn primal_vars(&self) -> Vec<String> {
        vec![self.kind.vars().0.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.kind.vars().1.to_string()]
    }

    fn physics(&self) -> &'static [Physics] {
        self.kind.physics()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    /// The four blocks of the exchange law — two diagonal, two coupling. Nothing
    /// but the stiffness: an interface law adds no mass, no geometric stiffness.
    fn contributions(
        &self,
        kind: MatrixKind,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        if kind != MatrixKind::Stiffness {
            return Ok(Vec::new());
        }
        Ok(vec![
            Contribution::Computed(self.diagonal_layout(&self.side_a, &self.support_a)),
            Contribution::Computed(self.diagonal_layout(&self.side_b, &self.support_b)),
            Contribution::Coupling(self.coupling_layout(
                &self.side_a,
                &self.support_a,
                &self.side_b,
                &self.support_b,
            )),
            Contribution::Coupling(self.coupling_layout(
                &self.side_b,
                &self.support_b,
                &self.side_a,
                &self.support_a,
            )),
        ])
    }

    /// A diagonal block: `+h ∫_Γ N_i N_j dΓ`.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("InterfaceTransfer declares a material_fespace");
        exchange_matrix(geom, geom, mat, 1.0, ke)
    }

    /// An off-diagonal block: `−h ∫_Γ N_i^row N_j^col dΓ`. The sign lives here
    /// rather than in a factor, because the two kernels are already distinct.
    fn coupling_element(
        &self,
        _kind: MatrixKind,
        row_geoms: &[CellGeom],
        col_geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("InterfaceTransfer declares a material_fespace");
        exchange_matrix(&row_geoms[0], &col_geoms[0], mat, -1.0, ke)
    }

    /// Internal fluxes `q_i = ∫ N_i · flux dΓ` — weighted by `N`, not by `Bᵀ`,
    /// exactly as for convection: the interface integrand is a flux **density**,
    /// not a gradient-conjugate quantity.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g)?;
            let w = geom.det_j_w(g)?;
            let flux = stress.value(geom.cell, g, OUTPUT_COMPONENT)?;
            for i in 0..geom.n_nodes {
                fe[i] += shape[i] * flux * w;
            }
        }
        Ok(())
    }

    fn label(&self) -> &'static str {
        "InterfaceTransfer"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let (primal, dual) = self.kind.vars();
        let cells = read(&self.side_a).and_then(|f| f.cell_count()).unwrap_or(0);
        format!(
            "SubModel<InterfaceTransfer({})>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  interface: {cells} facing cell pair(s)",
            self.kind.to_tag()
        )
    }
}

/// Behaviour-**input** component: the field jump across the interface, or simply
/// the field interpolated at the Gauss points of one side.
const INPUT_COMPONENT: &str = "jump";
/// Behaviour-**output** component: the exchanged flux density `h·jump`.
const OUTPUT_COMPONENT: &str = "flux";

impl Domain for InterfaceTransfer {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.side_a.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.side_a.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec![OUTPUT_COMPONENT.to_string()])
    }

    /// The exchanged flux density `h·(c₁ − c₂)` at one Gauss point, from the jump
    /// supplied as input.
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
        let mat = material.expect("InterfaceTransfer declares a material_fespace");
        let h = mat.value(geom.cell, g, MATERIAL_COMPONENT)?;
        out[0] = h * input.value(geom.cell, g, INPUT_COMPONENT)?;
        Ok(())
    }
}

/// `sign · h ∫_Γ N_i^row N_j^col dΓ` over one facing cell pair.
///
/// The measure comes from the **row** side: on a conforming interface the two
/// sides carry the same surface, so either would do — taking the row side keeps
/// the diagonal and off-diagonal blocks integrated identically, which is what
/// makes the four blocks sum to a consistent operator.
fn exchange_matrix(
    row_geom: &CellGeom,
    col_geom: &CellGeom,
    material: &SubElementField,
    sign: f64,
    ke: &mut [f64],
) -> Result<()> {
    let n_col = col_geom.n_nodes;
    for g in 0..row_geom.n_gauss {
        let row_shape = row_geom.n_at_g(g)?;
        let col_shape = col_geom.n_at_g(g)?;
        let w = row_geom.det_j_w(g)?;
        let h = material.value(row_geom.cell, g, MATERIAL_COMPONENT)?;
        for i in 0..row_geom.n_nodes {
            for j in 0..n_col {
                ke[i * n_col + j] += sign * h * row_shape[i] * col_shape[j] * w;
            }
        }
    }
    Ok(())
}

/// Verify that two boundary sub-meshes face each other cell for cell **and node
/// for node**, their paired nodes being co-located within `tol`.
///
/// Node-for-node matters as much as cell-for-cell: the coupling kernel pairs
/// `N_i` of one side with `N_j` of the other at a shared Gauss point, which is
/// only meaningful if local node `k` of a cell faces local node `k` of its
/// counterpart.
fn check_conforming_geometry(
    mesh_a: &Handle<SubMesh>,
    mesh_b: &Handle<SubMesh>,
    tol: f64,
) -> Result<()> {
    let (a, b) = (read(mesh_a)?, read(mesh_b)?);
    if a.element_type() != b.element_type() {
        return Err(PyrucastError::Message(format!(
            "InterfaceTransfer: the two sides must carry the same element type — \
             {:?} facing {:?}",
            a.element_type(),
            b.element_type()
        )));
    }
    if a.cell_count() != b.cell_count() {
        return Err(PyrucastError::Message(format!(
            "InterfaceTransfer: the two sides must be conforming — {} cell(s) facing {}",
            a.cell_count(),
            b.cell_count()
        )));
    }
    let (coords_a_h, coords_b_h) = (a.coords(), b.coords());
    let (guard_a, guard_b) = (read(&coords_a_h)?, read(&coords_b_h)?);
    let (coords_a, coords_b): (&Coords, &Coords) = (&guard_a, &guard_b);
    for (k, (&na, &nb)) in a.connectivity().iter().zip(b.connectivity()).enumerate() {
        let (pa, pb) = (coords_a.position(na)?, coords_b.position(nb)?);
        let d2: f64 = pa
            .iter()
            .zip(pb)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>();
        if d2.sqrt() > tol {
            return Err(PyrucastError::Message(format!(
                "InterfaceTransfer: the interface is not node-conforming — the node pair at \
                 connectivity slot {k} is {:.3e} apart (tolerance {tol:.3e}). Local node `k` of \
                 a cell must face local node `k` of its counterpart.",
                d2.sqrt()
            )));
        }
    }
    Ok(())
}
