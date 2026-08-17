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
//! `h` is the transfer coefficient (its inverse is the contact resistance). What
//! is transferred is the **caller's** to say: the sub-model is given
//! `(primal, dual)` pairs — the same shape [`embedded`](crate::models::embedded)
//! and [`contact`](crate::models::contact) take — and derives its coefficients
//! `h_<primal>` and its fluxes `flux_<primal>` from them. The same law describes
//! a thermal contact resistance (`("T", "q")`), a coating on a diffusion
//! (`("c_H2", "j_H2")`) and a bonded joint of finite stiffness (the three
//! displacement pairs): the mathematics is identical, only the names change.
//!
//! ## When *not* to use it on displacements
//!
//! Tying two surfaces by making `h` large is a **penalty** method, and a
//! [`Mpc`](crate::models::mpc) does it exactly, without degrading the
//! conditioning. The test is where the number comes from: if `h` comes from a
//! measurement this is physics; if `h` was chosen "large enough", it wanted a
//! constraint. See [`transfer`](crate::models::transfer), the module the two
//! exchange laws share.
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
use crate::models::transfer::{
    coefficient_indices, coefficient_name, exchange_matrix, flux_name, internal_force, jump_name,
    material_contract, physics_slice,
};
use crate::models::{
    CellGeom, Contribution, CouplingLayout, Domain, MatrixKind, MatrixLayout, Physics, SubModelKind,
};
use crate::store::Handle;

/// Exchange law between two conforming boundary FE subspaces.
#[derive(Clone)]
pub struct InterfaceTransfer {
    pub(crate) side_a: Handle<SubFiniteElementSpace>,
    pub(crate) side_b: Handle<SubFiniteElementSpace>,
    /// POI1 supports over each side's unique nodes.
    pub(crate) support_a: Handle<SubMesh>,
    pub(crate) support_b: Handle<SubMesh>,
    /// The transferred quantities, as `(primal, dual)` pairs.
    pub(crate) components: Vec<(String, String)>,
    /// The physics nature this exchange belongs to — what `model.filter(…)`
    /// selects it by. Free variable names cannot imply it, so it is declared.
    pub(crate) physics: Physics,
}

impl InterfaceTransfer {
    /// Exchange law across the interface between two **conforming** boundary FE
    /// subspaces. Errors unless the two sides match cell for cell and node for
    /// node, within `tol` of each other geometrically.
    pub fn new(
        side_a: Handle<SubFiniteElementSpace>,
        side_b: Handle<SubFiniteElementSpace>,
        components: Vec<(String, String)>,
        physics: Physics,
        tol: f64,
    ) -> Result<Self> {
        material_contract("InterfaceTransfer", &components)?;
        let (mesh_a, mesh_b) = (side_a.read().submesh(), side_b.read().submesh());
        check_conforming_geometry(&mesh_a, &mesh_b, tol)?;
        let support_a = mesh_a.read().to_poi1()?;
        let support_b = mesh_b.read().to_poi1()?;
        Ok(Self {
            side_a,
            side_b,
            support_a,
            support_b,
            components,
            physics,
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
        self.components.iter().map(|(p, _)| p.clone()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        self.components.iter().map(|(_, d)| d.clone()).collect()
    }

    fn physics(&self) -> &'static [Physics] {
        physics_slice(self.physics)
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
        let coefficients = coefficient_indices(mat, &self.components)?;
        exchange_matrix(geom, geom, mat, &coefficients, 1.0, ke)
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
        let coefficients = coefficient_indices(mat, &self.components)?;
        exchange_matrix(&row_geoms[0], &col_geoms[0], mat, &coefficients, -1.0, ke)
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
        internal_force(&geoms[0], stress, &self.components, fe)
    }

    fn label(&self) -> &'static str {
        "InterfaceTransfer"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let cells = self.side_a.read().cell_count().unwrap_or(0);
        format!(
            "SubModel<InterfaceTransfer>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  interface: {cells} facing cell pair(s)"
        )
    }
}

impl Domain for InterfaceTransfer {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.side_a.clone()
    }

    /// One coefficient per transferred quantity, named after it — `h_T`,
    /// `h_c_H2`, `h_u_x`.
    fn material_components(&self) -> Option<Vec<String>> {
        Some(
            self.components
                .iter()
                .map(|(p, _)| coefficient_name(p))
                .collect(),
        )
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.side_a.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(self.components.iter().map(|(p, _)| flux_name(p)).collect())
    }

    /// The exchanged flux density `h·(a₁ − a₂)` at one Gauss point, from the jump
    /// supplied as input (`jump_<primal>`), one transferred quantity at a time.
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
        let cell = geom.cell;
        for (v, (primal, _)) in self.components.iter().enumerate() {
            let h = mat.value(cell, g, &coefficient_name(primal))?;
            out[v] = h * input.value(cell, g, &jump_name(primal))?;
        }
        Ok(())
    }
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
    let (a, b) = (mesh_a.read(), mesh_b.read());
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
    let (guard_a, guard_b) = (coords_a_h.read(), coords_b_h.read());
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
