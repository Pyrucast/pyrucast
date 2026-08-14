//! Linear heat-conduction physics — assembly of the cell-wise stiffness
//! `K_ij = ∫ k · ∇N_i · ∇N_j dx`.
//!
//! Primal variable `"T"` (temperature, columns), dual `"q"` (heat flux,
//! rows). The conductivity is read from a [`SubElementField`] component
//! named [`MATERIAL_COMPONENT`].

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::models::owned_components;
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature).
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux).
pub const DUAL_VAR: &str = "q";
/// Required component on the material `SubElementField` (isotropic
/// conductivity).
pub const MATERIAL_COMPONENT: &str = "k";
/// Material contract returned by [`SubModelKind::material_components`] for the
/// isotropic default.
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];
/// Orthotropic conductivities plus the in-plane material axis (2-D).
const ORTHOTROPIC_2D: &[&str] = &["k_1", "k_2", "k_3", "V1X", "V1Y"];
/// Orthotropic conductivities plus the two material axes (3-D).
const ORTHOTROPIC_3D: &[&str] = &[
    "k_1", "k_2", "k_3", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];
/// The symmetric conductivity tensor plus the in-plane material axis (2-D).
const ANISOTROPIC_2D: &[&str] = &["k_11", "k_12", "k_13", "k_22", "k_23", "k_33", "V1X", "V1Y"];
/// The symmetric conductivity tensor plus the two material axes (3-D).
const ANISOTROPIC_3D: &[&str] = &[
    "k_11", "k_12", "k_13", "k_22", "k_23", "k_33", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];

/// The material contract of a symmetry in a space of dimension `space_dim` —
/// disjoint component sets, so an isotropic and an orthotropic conduction zone
/// resolve separately on one mesh (see [`crate::ops::matrix::assemble_kind`]).
fn material_contract(symmetry: MaterialSymmetry, space_dim: usize) -> &'static [&'static str] {
    match (symmetry, space_dim) {
        (MaterialSymmetry::Isotropic, _) => MATERIAL_COMPONENTS,
        (MaterialSymmetry::Orthotropic, 2) => ORTHOTROPIC_2D,
        (MaterialSymmetry::Orthotropic, _) => ORTHOTROPIC_3D,
        (MaterialSymmetry::Anisotropic, 2) => ANISOTROPIC_2D,
        (MaterialSymmetry::Anisotropic, _) => ANISOTROPIC_3D,
    }
}
/// Extra material components consumed **only** by the heat-capacity (mass)
/// matrix: density `rho` and specific heat `cp` (so the volumetric heat capacity
/// is `ρ·cp`). Optional — the conductivity assembly does not need them.
const CAPACITY_COMPONENTS: &[&str] = &["rho", "cp"];

/// Axis suffixes for the vector components of the deformation / flux at a
/// Gauss point, indexed by spatial direction (`x`, `y`, `z`).
const AXES: [&str; 3] = ["x", "y", "z"];

/// Deformation component names (`grad_T_x`, …), one per spatial direction —
/// the leading components of the behaviour-input field consumed by
/// [`SubModelKind::integrate_behavior`]. They match what
/// [`crate::ops::element_field::gradient`] names the gradient of a temperature field
/// whose component is [`PRIMAL_VAR`].
fn deformation_components(space_dim: usize) -> Vec<String> {
    (0..space_dim)
        .map(|a| format!("grad_{PRIMAL_VAR}_{}", AXES[a]))
        .collect()
}

/// Flux component names (`flux_x`, …), one per spatial direction. The value
/// stored is the **weak-form** flux `k·∇T` (such that `∫ Bᵀ·flux = K·T`,
/// hence the « COMP == stiffness » match in the linear case); the physical
/// Fourier flux is its opposite, `−k·∇T`.
fn flux_components(space_dim: usize) -> Vec<String> {
    (0..space_dim)
        .map(|a| format!("flux_{}", AXES[a]))
        .collect()
}

/// Linear heat conduction.
///
/// - primal variable: `"T"` (temperature, columns).
/// - dual variable:   `"q"` (heat flux row labels).
/// - Material data (conductivity `"k"`, …) is **not** stored here; it is
///   supplied at assembly time via [`crate::ops::matrix::stiffness`].
#[derive(Clone, Serialize, Deserialize)]
pub struct HeatConduction {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh,
    /// built once at construction. Reused as the row/col support of every
    /// assembled stiffness block — no per-assembly rebuild.
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) symmetry: MaterialSymmetry,
}

impl HeatConduction {
    /// **Isotropic** heat-conduction physics on an FE subspace. Builds the stable
    /// POI1 [`SubMesh`] covering the subspace's unique nodes (reused as the
    /// row/col support of every assembled block).
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Self::with_symmetry(fespace, MaterialSymmetry::Isotropic)
    }

    /// Heat conduction with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case. An
    /// orthotropic or anisotropic conductivity carries its material axes through
    /// the material field (see [`crate::models::symmetry`]).
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

impl SubModelKind for HeatConduction {
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

    /// The heat-capacity (mass) matrix shares the conductivity layout (same
    /// fespace, support, single `T` DOF per node) — only the kernel differs.
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        element_stiffness(
            geom,
            material.expect("HeatConduction requires a material field"),
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
        let geom = &geoms[0];
        element_capacity(
            geom,
            material.expect("HeatConduction requires a material field"),
            ke,
        )
    }

    /// Internal nodal fluxes `q_i = ∫ ∇N_i · flux dx` of one cell — `Bᵀ` applied
    /// to the weak-form flux, the scalar-transport counterpart of the
    /// continuum-mechanics default (and identical to
    /// [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence) of the
    /// flux vector). Single dual variable `q`, so `fe[i]` per node.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = geom.space_dim;
        let flux_names = flux_components(d);
        for g in 0..geom.n_gauss {
            let dn = geom.dn_dx(g)?; // [i * d + a]
            let w = geom.det_j_w(g)?;
            for i in 0..geom.n_nodes {
                let mut s = 0.0;
                for a in 0..d {
                    s += dn[i * d + a] * stress.value(geom.cell, g, &flux_names[a])?;
                }
                fe[i] += s * w;
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Thermal]
    }

    fn label(&self) -> &'static str {
        "HeatConduction"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<HeatConduction({})>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)",
            self.symmetry
        )
    }
}

impl Domain for HeatConduction {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(material_contract(
            self.symmetry,
            self.space_dim,
        )))
    }

    /// `rho` + `cp` — required only by the heat-capacity (mass) matrix, never by
    /// the conductivity assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        CAPACITY_COMPONENTS
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(flux_components(read(&self.fespace)?.space_dim()))
    }

    /// Linear constitutive law: weak-form flux = k·∇T at one Gauss point.
    /// (No internal-state variables — `VAR0`/`VAR1` are empty; a non-linear law
    /// would read trailing state components of `input` and write updated ones.)
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
        let mat =
            material.expect("HeatConduction declares a material_fespace ⇒ material is supplied");
        let (cell, space_dim) = (geom.cell, geom.space_dim);
        let grad_names = deformation_components(space_dim);
        let k3 =
            symmetry::transport_tensor(mat, cell, g, self.symmetry, space_dim, MATERIAL_COMPONENT)?;
        for a in 0..space_dim {
            let mut acc = 0.0;
            for b in 0..space_dim {
                acc += k3[(a, b)] * input.value(cell, g, &grad_names[b])?;
            }
            out[a] = acc;
        }
        Ok(())
    }
}

/// Element kernel: local conductivity matrix of one cell,
///   `K_local[i, j] = Σ_g k(g) · (∇N_i · ∇N_j)|_g · |J|_g · w_g`,
/// written into `ke` (flat row-major, side `n_nodes`, `ke[i * n_nodes + j]`).
/// Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    symmetry: MaterialSymmetry,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    // An oriented conductivity is built **once per cell** — its constants and its
    // material axes are cell-wise. Isotropy keeps reading its scalar at each
    // Gauss point, so a conductivity varying inside a cell still works.
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
                let k = material.value(geom.cell, g, MATERIAL_COMPONENT)?;
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        let mut grad_dot = 0.0;
                        for a in 0..space_dim {
                            grad_dot += dn[i * space_dim + a] * dn[j * space_dim + a];
                        }
                        ke[i * n_nodes + j] += k * grad_dot * det_j_w;
                    }
                }
            }
            // `∇N_iᵀ · K · ∇N_j` — the isotropic dot product is its `K = k·I` case.
            Some(k3) => {
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        let mut acc = 0.0;
                        for a in 0..space_dim {
                            for b in 0..space_dim {
                                acc += dn[i * space_dim + a] * k3[(a, b)] * dn[j * space_dim + b];
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

/// Element kernel: local **heat-capacity** (mass) matrix of one cell,
///   `C_local[i, j] = Σ_g ρ·cp · N_i N_j · |J|_g · w_g`,
/// written into `ke` (flat row-major, side `n_nodes`, `ke[i * n_nodes + j]`).
/// Density `rho` and specific heat `cp` are read per cell. Pure and sequential.
pub fn element_capacity(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let rho = material.value(geom.cell, 0, "rho").map_err(|_| {
        crate::error::PyrucastError::Message(
            "HeatConduction capacity matrix: material component `rho` (density) is required".into(),
        )
    })?;
    let cp = material.value(geom.cell, 0, "cp").map_err(|_| {
        crate::error::PyrucastError::Message(
            "HeatConduction capacity matrix: material component `cp` (specific heat) is required"
                .into(),
        )
    })?;
    let rho_cp = rho * cp;
    for g in 0..geom.n_gauss {
        let n = geom.n_at_g(g)?;
        let w = geom.det_j_w(g)? * rho_cp;
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                ke[i * n_nodes + j] += n[i] * n[j] * w;
            }
        }
    }
    Ok(())
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::store::insert;

    /// HeatConduction on a single SEG2 of length `L`.
    fn seg2_hc(length: f64) -> HeatConduction {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        HeatConduction::new(fes.get(0).unwrap()).unwrap()
    }

    #[test]
    fn behavior_fespace_is_the_physics_fespace() {
        let hc = seg2_hc(1.0);
        let fe = hc.behavior_fespace();
        assert_eq!(fe.index(), hc.fespace.index());
        assert_eq!(fe.generation(), hc.fespace.generation());
    }

    /// COMP on a linear law returns the weak-form flux `k·∇T` — the exact
    /// quantity the assembled stiffness integrates (`∫ Bᵀ·flux = K·T`). The
    /// deformation input (`grad_T_x`) is what [`crate::ops::element_field::gradient`]
    /// produces from a nodal temperature; here it is set directly.
    #[test]
    fn integrate_behavior_returns_weak_form_flux() {
        let hc = seg2_hc(2.0);
        let grad = 1.5; // e.g. ΔT / L = 3 / 2
        let k = 1.5;

        let mut def = SubElementField::new(hc.fespace.clone(), deformation_components(1)).unwrap();
        def.set_uniform("grad_T_x", grad).unwrap();
        let def = insert(def);

        let mut mat =
            SubElementField::new(hc.fespace.clone(), vec![MATERIAL_COMPONENT.to_string()]).unwrap();
        mat.set_uniform(MATERIAL_COMPONENT, k).unwrap();
        let mat = insert(mat);

        let flux = hc.integrate_behavior(&def, None, Some(&mat), None).unwrap();
        assert_eq!(flux.components(), &["flux_x".to_string()]);
        let expected = k * grad;
        for g in 0..flux.gauss_count() {
            assert!((flux.value(0, g, "flux_x").unwrap() - expected).abs() < 1e-12);
        }
    }
}
