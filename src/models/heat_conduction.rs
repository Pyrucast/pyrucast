//! Linear heat-conduction physics — assembly of the cell-wise stiffness
//! `K_ij = ∫ k · ∇N_i · ∇N_j dx`.
//!
//! Primal variable `"T"` (temperature, columns), dual `"q"` (heat flux,
//! rows). The conductivity is read from a [`SubElementField`] component
//! named [`MATERIAL_COMPONENT`].

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::NodeId;
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::NodeField;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::models::Physics;
use crate::store::{insert, with, Handle};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature).
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux).
pub const DUAL_VAR: &str = "q";
/// Required component on the material `SubElementField` (isotropic
/// conductivity).
pub const MATERIAL_COMPONENT: &str = "k";
/// Material contract returned by [`Physics::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Axis suffixes for the vector components of the deformation / flux at a
/// Gauss point, indexed by spatial direction (`x`, `y`, `z`).
const AXES: [&str; 3] = ["x", "y", "z"];

/// Deformation component names (`grad_T_x`, …), one per spatial direction.
/// These are the leading components of the behaviour-input field consumed
/// by [`Physics::integrate_behavior`].
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
    (0..space_dim).map(|a| format!("flux_{}", AXES[a])).collect()
}

/// Linear heat conduction.
///
/// - primal variable: `"T"` (temperature, columns).
/// - dual variable:   `"q"` (heat flux row labels).
/// - Material data (conductivity `"k"`, …) is **not** stored here; it is
///   supplied at assembly time via [`crate::ops::assemble::stiffness`].
#[derive(Clone, Serialize, Deserialize)]
pub struct HeatConduction {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh,
    /// built once at construction. Reused as the row/col support of every
    /// assembled stiffness block — no per-assembly rebuild.
    pub(crate) support: Handle<SubMesh>,
}

impl HeatConduction {
    /// Heat-conduction physics on an FE subspace. Builds the stable POI1
    /// [`SubMesh`] covering the subspace's unique nodes (reused as the
    /// row/col support of every assembled block).
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let submesh = with(&fespace, |s| s.submesh())?;
        let support = insert(with(&submesh, |s| s.to_poi1())??);
        Ok(Self { fespace, support })
    }
}

impl Physics for HeatConduction {
    fn primal_vars(&self) -> Vec<String> {
        vec![PRIMAL_VAR.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![DUAL_VAR.to_string()]
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.fespace.clone())
    }

    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let mat = material.expect("HeatConduction requires a material field");
        let mut block = SubMatrix::new(
            self.support.clone(),
            self.support.clone(),
            vec![DUAL_VAR.to_string()],
            vec![PRIMAL_VAR.to_string()],
            DofOrdering::NodesThenVars,
            true,
        )?;
        assemble_stiffness(&self.fespace, mat, &mut block)?;
        Ok(vec![block])
    }

    fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.fespace.clone())
    }

    fn compute_deformation(&self, solution: &Handle<NodeField>) -> Result<SubElementField> {
        // Snapshot the reference/geometry data we need from the FE space in
        // one critical section (∇N at every (cell, Gauss) + connectivity),
        // mirroring `assemble_stiffness`.
        struct Snap {
            n_cells: usize,
            n_g: usize,
            space_dim: usize,
            n_nodes: usize,
            conn: Vec<NodeId>,
            dn_dx: Vec<Vec<Vec<f64>>>, // [cell][g][i * space_dim + a]
        }
        let snap = with(&self.fespace, |s| -> Result<Snap> {
            let n_cells = s.cell_count()?;
            let n_nodes = s.nodes_per_cell()?;
            let n_g = s.gauss_count();
            let space_dim = s.space_dim();
            let submesh = s.submesh();
            let conn: Vec<NodeId> = with(&submesh, |sm| sm.connectivity().to_vec())?;
            let mut dn_dx = Vec::with_capacity(n_cells);
            for cell in 0..n_cells {
                let mut per_g = Vec::with_capacity(n_g);
                for g in 0..n_g {
                    per_g.push(s.dn_dx(cell, g)?);
                }
                dn_dx.push(per_g);
            }
            Ok(Snap { n_cells, n_g, space_dim, n_nodes, conn, dn_dx })
        })??;

        // Build the deformation field, then fill ∇T = Σ_i T_i ∇N_i per
        // (cell, Gauss, direction). The nodal T are read from `solution`
        // (a different store type, so a separate critical section).
        let mut field = SubElementField::new(
            self.fespace.clone(),
            deformation_components(snap.space_dim),
        )?;
        with(solution, |sol| -> Result<()> {
            for cell in 0..snap.n_cells {
                let ids = &snap.conn[cell * snap.n_nodes..(cell + 1) * snap.n_nodes];
                for g in 0..snap.n_g {
                    for a in 0..snap.space_dim {
                        let mut grad = 0.0;
                        for i in 0..snap.n_nodes {
                            let t_i = sol.value(ids[i], PRIMAL_VAR)?;
                            grad += t_i * snap.dn_dx[cell][g][i * snap.space_dim + a];
                        }
                        field.set(cell, g, a, grad)?;
                    }
                }
            }
            Ok(())
        })??;
        Ok(field)
    }

    fn integrate_behavior(
        &self,
        input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<SubElementField> {
        let mat = material
            .expect("HeatConduction declares a material_fespace ⇒ material is supplied");
        let space_dim = with(&self.fespace, |s| s.space_dim())?;
        let grad_names = deformation_components(space_dim);

        // `input` and `mat` are both `SubElementField`: the store Mutex is
        // per-type and non-reentrant, so snapshot them in *sequential*
        // critical sections — never nested.
        let (n_cells, n_g) = with(input, |f| (f.cell_count(), f.gauss_count()))?;
        let mut grads: Vec<f64> = Vec::with_capacity(n_cells * n_g * space_dim);
        with(input, |f| -> Result<()> {
            for cell in 0..n_cells {
                for g in 0..n_g {
                    for a in 0..space_dim {
                        grads.push(f.value(cell, g, &grad_names[a])?);
                    }
                }
            }
            Ok(())
        })??;
        let mut ks: Vec<f64> = Vec::with_capacity(n_cells * n_g);
        with(mat, |m| -> Result<()> {
            for cell in 0..n_cells {
                for g in 0..n_g {
                    ks.push(m.value(cell, g, MATERIAL_COMPONENT)?);
                }
            }
            Ok(())
        })??;

        // Linear constitutive law: weak-form flux = k·∇T at each point.
        // (No internal-state variables for this law — `VAR0`/`VAR1` are
        // empty; a non-linear law would read trailing state components of
        // `input` and append the updated ones to the output.)
        let mut out = SubElementField::new(self.fespace.clone(), flux_components(space_dim))?;
        for cell in 0..n_cells {
            for g in 0..n_g {
                let k = ks[cell * n_g + g];
                for a in 0..space_dim {
                    let grad = grads[(cell * n_g + g) * space_dim + a];
                    out.set(cell, g, a, k * grad)?;
                }
            }
        }
        Ok(out)
    }

    fn label(&self) -> &'static str {
        "HeatConduction"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = with(&self.support, |s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<HeatConduction>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

/// Assemble the heat-conduction stiffness contribution.
///
/// On each cell of `fespace`'s submesh, at each Gauss point `g`:
///   `K_local[i, j] += k(g) · (∇N_i · ∇N_j)|_g · |J|_g · w_g`
/// and the local block is written into `k` at
///   row = `(NodeId_i, "q")`, col = `(NodeId_j, "T")`.
pub fn assemble_stiffness(
    fespace: &Handle<SubFiniteElementSpace>,
    material: &Handle<SubElementField>,
    k: &mut SubMatrix,
) -> Result<()> {
    // Snapshot everything we need from the FE space and submesh in one
    // pass. We then drop the FE space lock before reading the material
    // (different store type, but better hygiene to keep critical
    // sections small).
    struct CellSnapshot {
        node_ids: Vec<NodeId>,
        dn_dx: Vec<Vec<f64>>, // [g][i * space_dim + a]
        det_j_w: Vec<f64>,    // |J|_g · w_g
    }

    let snapshots: Vec<CellSnapshot> = with(fespace, |s| -> Result<_> {
        let n_cells = s.cell_count()?;
        let n_nodes = s.nodes_per_cell()?;
        let n_g = s.gauss_count();
        let submesh = s.submesh();

        let conn: Vec<NodeId> = with(&submesh, |sm| sm.connectivity().to_vec())?;

        let mut out = Vec::with_capacity(n_cells);
        for cell in 0..n_cells {
            let ids = conn[cell * n_nodes..(cell + 1) * n_nodes].to_vec();
            let mut dn_dx: Vec<Vec<f64>> = Vec::with_capacity(n_g);
            let mut det_j_w: Vec<f64> = Vec::with_capacity(n_g);
            for g in 0..n_g {
                dn_dx.push(s.dn_dx(cell, g)?);
                det_j_w.push(s.det_jacobian(cell, g)? * s.gauss_weight(g)?);
            }
            out.push(CellSnapshot {
                node_ids: ids,
                dn_dx,
                det_j_w,
            });
        }
        Ok(out)
    })??;

    let space_dim = with(fespace, |s| s.space_dim())?;
    let n_nodes = with(fespace, |s| s.nodes_per_cell())??;
    let n_g = with(fespace, |s| s.gauss_count())?;

    // Read material conductivity once per (cell, gauss).
    let mut conductivities: Vec<Vec<f64>> = Vec::with_capacity(snapshots.len());
    with(material, |f| -> Result<()> {
        for cell in 0..snapshots.len() {
            let mut row = Vec::with_capacity(n_g);
            for g in 0..n_g {
                row.push(f.value(cell, g, MATERIAL_COMPONENT)?);
            }
            conductivities.push(row);
        }
        Ok(())
    })??;

    // Assemble cell by cell.
    for (cell, snap) in snapshots.iter().enumerate() {
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let mut k_ij = 0.0;
                for g in 0..n_g {
                    let mut grad_dot = 0.0;
                    for a in 0..space_dim {
                        grad_dot += snap.dn_dx[g][i * space_dim + a]
                            * snap.dn_dx[g][j * space_dim + a];
                    }
                    k_ij += conductivities[cell][g] * grad_dot * snap.det_j_w[g];
                }
                k.add_entry(snap.node_ids[i], DUAL_VAR, snap.node_ids[j], PRIMAL_VAR, k_ij)?;
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
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Configuration, ElementType, Mesh, Node};
    use crate::store::insert;

    /// HeatConduction on a single SEG2 of length `L`; returns the physics
    /// plus the two node ids `(a @ 0, b @ L)`.
    fn seg2_hc(length: f64) -> (HeatConduction, NodeId, NodeId) {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let hc = HeatConduction::new(fes.get(0).unwrap()).unwrap();
        (hc, a.id(), b.id())
    }

    /// A nodal temperature solution `T(a) = 0`, `T(b) = dt` on the HC
    /// support, inserted into the store.
    fn linear_solution(hc: &HeatConduction, a: NodeId, b: NodeId, dt: f64) -> Handle<NodeField> {
        let mut sol = NodeField::from_poi1(&hc.support, vec![PRIMAL_VAR.to_string()]).unwrap();
        sol.set_value(a, PRIMAL_VAR, 0.0).unwrap();
        sol.set_value(b, PRIMAL_VAR, dt).unwrap();
        insert(sol)
    }

    #[test]
    fn behavior_fespace_is_the_physics_fespace() {
        let (hc, _, _) = seg2_hc(1.0);
        let fe = hc.behavior_fespace().expect("HC has a behaviour");
        assert_eq!(fe.index(), hc.fespace.index());
        assert_eq!(fe.generation(), hc.fespace.generation());
    }

    /// ∇T of a linear field on a SEG2 is the constant `ΔT / L` at every
    /// Gauss point, in the single `grad_T_x` component.
    #[test]
    fn deformation_is_constant_gradient_on_seg2() {
        let length = 2.0;
        let dt = 3.0;
        let (hc, a, b) = seg2_hc(length);
        let sol = linear_solution(&hc, a, b, dt);

        let def = hc.compute_deformation(&sol).unwrap();
        assert_eq!(def.components(), &[format!("grad_{PRIMAL_VAR}_x")]);
        let grad = format!("grad_{PRIMAL_VAR}_x");
        let expected = dt / length;
        for g in 0..def.gauss_count() {
            assert!((def.value(0, g, &grad).unwrap() - expected).abs() < 1e-12);
        }
    }

    /// COMP on a linear law returns the weak-form flux `k·∇T` — the exact
    /// quantity the assembled stiffness integrates (`∫ Bᵀ·flux = K·T`).
    #[test]
    fn integrate_behavior_returns_weak_form_flux() {
        let length = 2.0;
        let dt = 3.0;
        let k = 1.5;
        let (hc, a, b) = seg2_hc(length);
        let sol = linear_solution(&hc, a, b, dt);
        let def = insert(hc.compute_deformation(&sol).unwrap());

        let mut mat =
            SubElementField::new(hc.fespace.clone(), vec![MATERIAL_COMPONENT.to_string()]).unwrap();
        mat.set_uniform(MATERIAL_COMPONENT, k).unwrap();
        let mat = insert(mat);

        let flux = hc.integrate_behavior(&def, Some(&mat)).unwrap();
        assert_eq!(flux.components(), &["flux_x".to_string()]);
        let expected = k * dt / length;
        for g in 0..flux.gauss_count() {
            assert!((flux.value(0, g, "flux_x").unwrap() - expected).abs() < 1e-12);
        }
    }
}
