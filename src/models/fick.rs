//! Fickian diffusion — assembly of the cell-wise stiffness
//! `K_ij = ∫ ∇N_i · D · ∇N_j dx`.
//!
//! The mass-transport twin of [`heat_conduction`](crate::models::heat_conduction):
//! same Laplacian, different variables. Primal `"c"` (concentration, columns),
//! dual `"j"` (mass flux, rows). Fick's first law is `j = −D·∇c`; as in
//! conduction, what is stored is the **weak-form** flux `D·∇c`, so that
//! `∫ Bᵀ·flux = K·c` and the behaviour matches the stiffness in the linear case.
//!
//! Its nature is [`Physics::Diffusion`], not `Thermal`: sharing an operator is
//! not sharing a physics, and a coupled thermo-diffusive model must be able to
//! select one without dragging in the other.
//!
//! The diffusivity obeys any [`MaterialSymmetry`]:
//!
//! | symmetry | material components |
//! |---|---|
//! | isotropic | `D` |
//! | orthotropic | `D_1`, `D_2`, `D_3` + the material axes |
//! | anisotropic | `D_11 … D_33` (symmetric) + the material axes |
//!
//! The transient (storage) term `∂c/∂t` is the mass matrix, `∫ poro N_i N_j` —
//! `poro` being the storage coefficient, the porosity for a species diffusing
//! through a porous solid. It is optional: a steady assembly never asks for it.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Column DOF name (concentration).
pub const PRIMAL_VAR: &str = "c";
/// Row DOF name (mass flux).
pub const DUAL_VAR: &str = "j";
/// The coefficient's name — the isotropic diffusivity, and the prefix of the
/// oriented ones (`D_1`, `D_11`, …).
pub const MATERIAL_COMPONENT: &str = "D";
/// Storage (capacity) coefficient, consumed **only** by the mass matrix.
const STORAGE_COMPONENTS: &[&str] = &["poro"];

/// Isotropic contract.
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];
/// Orthotropic diffusivities plus the in-plane material axis (2-D).
const ORTHOTROPIC_2D: &[&str] = &["D_1", "D_2", "D_3", "V1X", "V1Y"];
/// Orthotropic diffusivities plus the two material axes (3-D).
const ORTHOTROPIC_3D: &[&str] = &[
    "D_1", "D_2", "D_3", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];
/// The symmetric diffusivity tensor plus the in-plane material axis (2-D).
const ANISOTROPIC_2D: &[&str] = &["D_11", "D_12", "D_13", "D_22", "D_23", "D_33", "V1X", "V1Y"];
/// The symmetric diffusivity tensor plus the two material axes (3-D).
const ANISOTROPIC_3D: &[&str] = &[
    "D_11", "D_12", "D_13", "D_22", "D_23", "D_33", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];

/// The material contract of a symmetry in a space of dimension `space_dim`.
fn material_contract(symmetry: MaterialSymmetry, space_dim: usize) -> &'static [&'static str] {
    match (symmetry, space_dim) {
        (MaterialSymmetry::Isotropic, _) => MATERIAL_COMPONENTS,
        (MaterialSymmetry::Orthotropic, 2) => ORTHOTROPIC_2D,
        (MaterialSymmetry::Orthotropic, _) => ORTHOTROPIC_3D,
        (MaterialSymmetry::Anisotropic, 2) => ANISOTROPIC_2D,
        (MaterialSymmetry::Anisotropic, _) => ANISOTROPIC_3D,
    }
}

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];

/// Behaviour-**input** components (`grad_c_x`, …) — what
/// [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient)
/// names the gradient of a field whose component is [`PRIMAL_VAR`].
fn gradient_components(space_dim: usize) -> Vec<String> {
    (0..space_dim)
        .map(|a| format!("grad_{PRIMAL_VAR}_{}", AXES[a]))
        .collect()
}

/// Behaviour-**output** components (`j_x`, …) — the weak-form flux `D·∇c`. Named
/// after the dual variable rather than `flux_*`, so a model carrying both
/// conduction and diffusion keeps two unambiguous flux fields.
fn flux_components(space_dim: usize) -> Vec<String> {
    (0..space_dim)
        .map(|a| format!("{DUAL_VAR}_{}", AXES[a]))
        .collect()
}

/// Fickian diffusion on an FE subspace.
///
/// - primal variable: `"c"` (concentration, columns).
/// - dual variable:   `"j"` (mass flux, row labels).
/// - Material data is supplied at assembly time, not stored here.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fick {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) symmetry: MaterialSymmetry,
}

impl Fick {
    /// **Isotropic** Fickian diffusion on an FE subspace.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Self::with_symmetry(fespace, MaterialSymmetry::Isotropic)
    }

    /// Fickian diffusion with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case.
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        let (submesh, space_dim) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim())
        };
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            symmetry,
        })
    }
}

impl SubModelKind for Fick {
    fn primal_vars(&self) -> Vec<String> {
        vec![PRIMAL_VAR.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![DUAL_VAR.to_string()]
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// The storage (mass) matrix shares the diffusion layout — only the kernel
    /// differs.
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        element_stiffness(
            &geoms[0],
            material.expect("Fick requires a material field"),
            self.symmetry,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        element_storage(
            &geoms[0],
            material.expect("Fick requires a material field"),
            ke,
        )
    }

    /// Internal nodal fluxes `j_i = ∫ ∇N_i · flux dx` — `Bᵀ` applied to the
    /// weak-form flux, as in conduction. Single dual variable, so `fe[i]` per node.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = geom.space_dim;
        let names = flux_components(d);
        for g in 0..geom.n_gauss {
            let dn = geom.dn_dx(g)?;
            let w = geom.det_j_w(g)?;
            for i in 0..geom.n_nodes {
                let mut s = 0.0;
                for a in 0..d {
                    s += dn[i * d + a] * stress.value(geom.cell, g, &names[a])?;
                }
                fe[i] += s * w;
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Diffusion]
    }

    fn label(&self) -> &'static str {
        "Fick"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Fick({})>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)",
            self.symmetry
        )
    }
}

impl Domain for Fick {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(material_contract(self.symmetry, self.space_dim))
    }

    /// `poro` — required only by the storage (mass) matrix, never by the
    /// diffusion assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        STORAGE_COMPONENTS
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(flux_components(self.space_dim))
    }

    /// Fick's law: weak-form flux `D·∇c` at one Gauss point.
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
        let mat = material.expect("Fick declares a material_fespace ⇒ material is supplied");
        let (cell, space_dim) = (geom.cell, geom.space_dim);
        let names = gradient_components(space_dim);
        let d3 =
            symmetry::transport_tensor(mat, cell, g, self.symmetry, space_dim, MATERIAL_COMPONENT)?;
        for a in 0..space_dim {
            let mut acc = 0.0;
            for b in 0..space_dim {
                acc += d3[(a, b)] * input.value(cell, g, &names[b])?;
            }
            out[a] = acc;
        }
        Ok(())
    }
}

/// Element kernel: local diffusion matrix of one cell,
/// `K[i, j] = Σ_g ∇N_iᵀ · D · ∇N_j · |J|_g · w_g`, written into `ke` (flat
/// row-major, side `n_nodes`). Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    symmetry: MaterialSymmetry,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    // An oriented diffusivity is built once per cell; the isotropic scalar is
    // read at each Gauss point, so it may vary inside a cell.
    let tensor = if symmetry.has_frame() {
        Some(symmetry::transport_tensor(
            material,
            geom.cell,
            0,
            symmetry,
            space_dim,
            MATERIAL_COMPONENT,
        )?)
    } else {
        None
    };
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?;
        let det_j_w = geom.det_j_w(g)?;
        match &tensor {
            None => {
                let d = material.value(geom.cell, g, MATERIAL_COMPONENT)?;
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        let mut grad_dot = 0.0;
                        for a in 0..space_dim {
                            grad_dot += dn[i * space_dim + a] * dn[j * space_dim + a];
                        }
                        ke[i * n_nodes + j] += d * grad_dot * det_j_w;
                    }
                }
            }
            Some(d3) => {
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        let mut acc = 0.0;
                        for a in 0..space_dim {
                            for b in 0..space_dim {
                                acc += dn[i * space_dim + a] * d3[(a, b)] * dn[j * space_dim + b];
                            }
                        }
                        ke[i * n_nodes + j] += acc * det_j_w;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Element kernel: local **storage** (mass) matrix of one cell,
/// `C[i, j] = Σ_g poro · N_i N_j · |J|_g · w_g`. Same `ke` layout as
/// [`element_stiffness`].
pub fn element_storage(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let poro = material.value(geom.cell, 0, "poro").map_err(|_| {
        PyrucastError::Message(
            "Fick storage matrix: material component `poro` (storage coefficient) is required"
                .into(),
        )
    })?;
    for g in 0..geom.n_gauss {
        let n = geom.n_at_g(g)?;
        let det_j_w = geom.det_j_w(g)?;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                ke[i * n_nodes + j] += poro * n[i] * n[j] * det_j_w;
            }
        }
    }
    Ok(())
}
