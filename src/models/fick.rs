//! Fickian diffusion — assembly of the cell-wise stiffness
//! `K_ij = ∫ ∇N_i · D · ∇N_j dx`.
//!
//! The mass-transport twin of [`heat_conduction`](crate::models::heat_conduction):
//! same Laplacian, different variables. Fick's first law is `j = −D·∇c`; as in
//! conduction, what is stored is the **weak-form** flux `D·∇c`, so that
//! `∫ Bᵀ·flux = K·c` and the behaviour matches the stiffness in the linear case.
//!
//! ## Every name carries its species
//!
//! A diffusion problem rarely has one diffusing species, and two of them share
//! neither their concentration, their flux, nor their diffusivity. The species
//! is therefore **named at construction** and carried by every quantity that
//! belongs to it:
//!
//! ```text
//! species "H2"      primal  c_H2        dual  j_H2
//!                   material D_H2  (D_1_H2, D_11_H2, … when oriented)
//!                   behaviour  grad_c_H2_x → j_H2_x
//! ```
//!
//! Two `Fick` sub-models with different species then live on the **same mesh**
//! without colliding — their DOFs are distinct names, so the assembler puts them
//! side by side with no special case. It also removes the older ambiguity with
//! heat conduction, whose `T`/`q` no longer risk meeting a bare `c`/`j`.
//!
//! What is **not** suffixed is what belongs to the medium rather than to the
//! species: the storage coefficient `poro` and the material axes `V1X`, `V1Y`, …
//! A porous solid has one porosity and one weaving frame, whatever diffuses
//! through it.
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

/// Column DOF name of a species — `c_H2`.
pub fn primal_var(species: &str) -> String {
    format!("c_{species}")
}

/// Row DOF name of a species — `j_H2`.
pub fn dual_var(species: &str) -> String {
    format!("j_{species}")
}
/// Storage (capacity) coefficient, consumed **only** by the mass matrix.
const STORAGE_COMPONENTS: &[&str] = &["poro"];

/// Isotropic contract.
const MATERIAL_COMPONENTS: &[&str] = &["D"];
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

/// The material contract of a symmetry, **with the species** carried by every
/// diffusivity and by nothing else.
///
/// The species goes last (`D_1_H2`, not `D_H2_1`), so stripping it is one rule
/// whatever the symmetry. The axes `V1X…V2Z` keep their bare names: they are the
/// medium's frame, shared by everything that diffuses through it.
fn material_contract(symmetry: MaterialSymmetry, space_dim: usize, species: &str) -> Vec<String> {
    static_contract(symmetry, space_dim)
        .iter()
        .map(|name| {
            if name.starts_with('D') {
                format!("{name}_{species}")
            } else {
                (*name).to_string()
            }
        })
        .collect()
}

/// The species-free contract, one table per symmetry.
fn static_contract(symmetry: MaterialSymmetry, space_dim: usize) -> &'static [&'static str] {
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
/// names the gradient of a field whose component is [`primal_var`].
fn gradient_components(space_dim: usize, species: &str) -> Vec<String> {
    let primal = primal_var(species);
    (0..space_dim)
        .map(|a| format!("grad_{primal}_{}", AXES[a]))
        .collect()
}

/// Behaviour-**output** components (`j_x`, …) — the weak-form flux `D·∇c`. Named
/// after the dual variable rather than `flux_*`, so a model carrying both
/// conduction and diffusion keeps two unambiguous flux fields.
fn flux_components(space_dim: usize, species: &str) -> Vec<String> {
    let dual = dual_var(species);
    (0..space_dim)
        .map(|a| format!("{dual}_{}", AXES[a]))
        .collect()
}

/// Fickian diffusion of **one named species** on an FE subspace.
///
/// - primal variable: `c_<species>` (concentration, columns).
/// - dual variable:   `j_<species>` (mass flux, row labels).
/// - Material data is supplied at assembly time, not stored here.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fick {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) symmetry: MaterialSymmetry,
    /// The diffusing species, carried by every name this physics declares.
    pub(crate) species: String,
}

impl Fick {
    /// **Isotropic** Fickian diffusion of `species` on an FE subspace.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, species: &str) -> Result<Self> {
        Self::with_symmetry(fespace, MaterialSymmetry::Isotropic, species)
    }

    /// Fickian diffusion with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case.
    ///
    /// The species names the concentration, the flux and the diffusivity; an
    /// empty one is refused, since it would give back the bare `c`/`j` the
    /// suffix exists to avoid.
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        symmetry: MaterialSymmetry,
        species: &str,
    ) -> Result<Self> {
        if species.is_empty() {
            return Err(PyrucastError::Message(
                "Fick: a diffusing species must be named — its concentration, flux and \
                 diffusivity all carry the name (`c_H2`, `j_H2`, `D_H2`), which is what lets \
                 two species share a mesh"
                    .into(),
            ));
        }
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
            species: species.to_string(),
        })
    }
}

impl SubModelKind for Fick {
    fn primal_vars(&self) -> Vec<String> {
        vec![primal_var(&self.species)]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![dual_var(&self.species)]
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
            &self.species,
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
        let names = flux_components(d, &self.species);
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

    /// Every diffusivity carries the species and goes last (`D_H2`, `D_1_H2`,
    /// `D_11_H2`); the material axes `V1X…V2Z` keep their bare names, being the
    /// medium's frame rather than the species' business.
    fn material_components(&self) -> Option<Vec<String>> {
        Some(material_contract(
            self.symmetry,
            self.space_dim,
            &self.species,
        ))
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
        Ok(flux_components(self.space_dim, &self.species))
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
        let names = gradient_components(space_dim, &self.species);
        let prefix = format!("D_{}", self.species);
        let d3 = symmetry::transport_tensor(mat, cell, g, self.symmetry, space_dim, &prefix)?;
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
    species: &str,
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
            &format!("D_{species}"),
        )?)
    } else {
        None
    };
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?;
        let det_j_w = geom.det_j_w(g)?;
        match &tensor {
            None => {
                let d = material.value(geom.cell, g, &format!("D_{species}"))?;
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
