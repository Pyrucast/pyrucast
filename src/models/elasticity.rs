//! Linear (small-strain) elasticity — `K = ∫ Bᵀ D B dΩ`.
//!
//! Works in 2-D (TRI3 / QUA4) and 3-D (TET4 / HEX8). 2-D supports **plane
//! stress** and **plane strain**; 3-D is the full solid. Voigt convention:
//! strain `[εxx, εyy, γxy]` (2-D) / `[εxx, εyy, εzz, γyz, γxz, γxy]` (3-D), with
//! **engineering** shear `γ = 2ε`; stress in the matching order.
//!
//! Primal `u_x, u_y(, u_z)` (displacement), dual `f_x, …` (nodal force).
//! Material components `E` (Young) and `nu` (Poisson).

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, Domain, Physics, StiffnessLayout, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by linear elasticity.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu"];

/// Which 2-D assumption (or 3-D solid) to use for the constitutive matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElasticityModel {
    /// 2-D plane stress (thin plate loaded in its plane).
    PlaneStress,
    /// 2-D plane strain (long prismatic body, `εzz = 0`).
    PlaneStrain,
    /// Full 3-D solid.
    Solid,
}

impl ElasticityModel {
    /// Parse from a lowercase tag (`"plane_stress"`, `"plane_strain"`, `"solid"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "plane_stress" => Some(Self::PlaneStress),
            "plane_strain" => Some(Self::PlaneStrain),
            "solid" => Some(Self::Solid),
            _ => None,
        }
    }
}

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}
/// Voigt component count: 3 in 2-D, 6 in 3-D.
fn voigt_size(space_dim: usize) -> usize {
    if space_dim == 2 {
        3
    } else {
        6
    }
}
/// Stress component names in Voigt order.
fn stress_names(space_dim: usize) -> Vec<String> {
    if space_dim == 2 {
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()]
    } else {
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ]
    }
}

/// Linear-elasticity physics on an FE subspace.
///
/// Material data (`E`, `nu`) is supplied at assembly time via
/// [`crate::ops::assemble::stiffness`], not stored here.
#[derive(Clone, Serialize, Deserialize)]
pub struct Elasticity {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
}

impl Elasticity {
    /// Linear elasticity on an FE subspace, with the given 2-D/3-D model.
    /// Errors if `model` is inconsistent with the space dimension.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        let (submesh, space_dim) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim())
        };
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Elasticity: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain, 3-D ⇒ solid)"
            )));
        }
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
        })
    }
}

impl SubModelKind for Elasticity {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        Some(StiffnessLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        element_stiffness(geom, mat, self.model, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Elasticity"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Elasticity({:?})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Elasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    /// `alpha` (thermal-expansion coefficient) — accepted through the material
    /// field when doing thermomechanics, never required for a plain elastic
    /// assembly. Consumed by
    /// [`crate::ops::field::thermal_strain`](fn@crate::ops::field::thermal_strain).
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(stress_names(self.space_dim))
    }

    /// Linear stress σ = D·ε at one Gauss point (material `E`, `nu` per cell).
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
        let mat = material.expect("Elasticity declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let e = mat.value(cell, 0, "E")?;
        let nu = mat.value(cell, 0, "nu")?;
        let dmat = constitutive(e, nu, self.model, d);
        let strain = voigt_strain(&|name| input.value(cell, g, name), d)?;
        for (r, drow) in dmat.iter().enumerate() {
            out[r] = drow.iter().zip(&strain).map(|(dv, s)| dv * s).sum();
        }
        Ok(())
    }
}

/// Isotropic constitutive (Voigt) matrix `D` from `E`, `nu` and the model.
pub fn constitutive(e: f64, nu: f64, model: ElasticityModel, space_dim: usize) -> Vec<Vec<f64>> {
    match (space_dim, model) {
        (2, ElasticityModel::PlaneStress) => {
            let c = e / (1.0 - nu * nu);
            vec![
                vec![c, c * nu, 0.0],
                vec![c * nu, c, 0.0],
                vec![0.0, 0.0, c * (1.0 - nu) / 2.0],
            ]
        }
        (2, ElasticityModel::PlaneStrain) => {
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            vec![
                vec![c * (1.0 - nu), c * nu, 0.0],
                vec![c * nu, c * (1.0 - nu), 0.0],
                vec![0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
            ]
        }
        _ => {
            // 3-D solid (Voigt order [xx, yy, zz, yz, xz, xy]).
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let g = c * (1.0 - 2.0 * nu) / 2.0;
            let mut d = vec![vec![0.0; 6]; 6];
            for i in 0..3 {
                for j in 0..3 {
                    d[i][j] = if i == j { c * (1.0 - nu) } else { c * nu };
                }
            }
            d[3][3] = g;
            d[4][4] = g;
            d[5][5] = g;
            d
        }
    }
}

/// Voigt **engineering** strain from the tensor strain components produced by
/// [`crate::ops::field::deformation`] (`eps_xx`, `eps_xy`, …), reading each
/// component by name through `eps`. Off-diagonals become `γ = 2ε`.
fn voigt_strain(eps: &dyn Fn(&str) -> Result<f64>, space_dim: usize) -> Result<Vec<f64>> {
    if space_dim == 2 {
        Ok(vec![eps("eps_xx")?, eps("eps_yy")?, 2.0 * eps("eps_xy")?])
    } else {
        Ok(vec![
            eps("eps_xx")?,
            eps("eps_yy")?,
            eps("eps_zz")?,
            2.0 * eps("eps_yz")?,
            2.0 * eps("eps_xz")?,
            2.0 * eps("eps_xy")?,
        ])
    }
}

/// Strain-displacement matrix `B` (Voigt) from `∂N_i/∂x_a` (`dn_dx`, layout
/// `[i*space_dim + a]`). Shape `voigt_size × (space_dim·nodes)`, node-major
/// columns (matching [`DofOrdering::NodesThenVars`]).
fn b_matrix(dn_dx: &[f64], n_nodes: usize, space_dim: usize) -> Vec<Vec<f64>> {
    let v = voigt_size(space_dim);
    let dofs = space_dim * n_nodes;
    let mut b = vec![vec![0.0; dofs]; v];
    let dn = |i: usize, a: usize| dn_dx[i * space_dim + a];
    for i in 0..n_nodes {
        if space_dim == 2 {
            let (cx, cy) = (2 * i, 2 * i + 1);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cx] = dn(i, 1); // γxy
            b[2][cy] = dn(i, 0);
        } else {
            let (cx, cy, cz) = (3 * i, 3 * i + 1, 3 * i + 2);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cz] = dn(i, 2); // εzz
            b[3][cy] = dn(i, 2); // γyz
            b[3][cz] = dn(i, 1);
            b[4][cx] = dn(i, 2); // γxz
            b[4][cz] = dn(i, 0);
            b[5][cx] = dn(i, 1); // γxy
            b[5][cy] = dn(i, 0);
        }
    }
    b
}

/// Element kernel: local stiffness `K_e = Σ_g (Bᵀ D B) |J| w` of one cell,
/// written into `ke` (flat row-major, side `space_dim·n_nodes`, **node-major /
/// component-minor** dof order `dof = node·space_dim + component`). Pure and
/// sequential — driven in parallel by [`crate::models::kernel::assemble_block`].
/// Reused as-is by [`crate::models::plasticity`] and [`crate::models::mazars`]
/// (their iteration operator is the elastic stiffness).
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    model: ElasticityModel,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    let dofs = space_dim * n_nodes;
    // E, nu read at Gauss 0 — constant material per cell.
    let d = constitutive(
        material.value(geom.cell, 0, "E")?,
        material.value(geom.cell, 0, "nu")?,
        model,
        space_dim,
    );
    let v = d.len();
    for g in 0..geom.n_gauss {
        let b = b_matrix(&geom.dn_dx(g)?, n_nodes, space_dim);
        // DB = D·B  (voigt × dofs).
        let mut db = vec![vec![0.0; dofs]; v];
        for r in 0..v {
            for c in 0..dofs {
                let mut acc = 0.0;
                for w in 0..v {
                    acc += d[r][w] * b[w][c];
                }
                db[r][c] = acc;
            }
        }
        let w = geom.det_j_w(g)?;
        for r in 0..dofs {
            for c in 0..dofs {
                let mut acc = 0.0;
                for vv in 0..v {
                    acc += b[vv][r] * db[vv][c];
                }
                ke[r * dofs + c] += acc * w;
            }
        }
    }
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId};
    use crate::store::insert;

    fn unit_quad(model: ElasticityModel) -> Elasticity {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Elasticity::new(fes.get(0).unwrap(), model).unwrap()
    }

    #[test]
    fn vars_and_model_validation() {
        let el = unit_quad(ElasticityModel::PlaneStress);
        assert_eq!(el.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(el.dual_vars(), vec!["f_x", "f_y"]);
        // 2-D space cannot be Solid.
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert!(Elasticity::new(fes.get(0).unwrap(), ElasticityModel::Solid).is_err());
    }

    #[test]
    fn plane_stress_constitutive_known_values() {
        let (e, nu) = (1.0, 0.25);
        let d = constitutive(e, nu, ElasticityModel::PlaneStress, 2);
        let c = e / (1.0 - nu * nu);
        assert!((d[0][0] - c).abs() < 1e-12);
        assert!((d[0][1] - c * nu).abs() < 1e-12);
        assert!((d[2][2] - c * (1.0 - nu) / 2.0).abs() < 1e-12);
        assert!((d[2][2] - e / (2.0 * (1.0 + nu))).abs() < 1e-12); // = G
    }

    /// COMP: uniaxial tensor strain `εxx = ε₀` in plane stress gives
    /// `σxx = E/(1-ν²)·ε₀`, `σyy = ν·σxx`, `σxy = 0`.
    #[test]
    fn integrate_behavior_plane_stress_uniaxial() {
        let (e, nu, eps0) = (210.0, 0.3, 0.001);
        let el = unit_quad(ElasticityModel::PlaneStress);
        let mut mat =
            SubElementField::new(el.fespace.clone(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", e).unwrap();
        mat.set_uniform("nu", nu).unwrap();
        let mat = insert(mat);

        let mut strain = SubElementField::new(
            el.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = insert(strain);

        let out = el
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-9);
            assert!((out.value(0, g, "sigma_yy").unwrap() - c * nu * eps0).abs() < 1e-9);
            assert!(out.value(0, g, "sigma_xy").unwrap().abs() < 1e-9);
        }
    }

    /// Element stiffness is symmetric and the rigid-body modes are in its
    /// kernel (zero row sums per axis).
    #[test]
    fn element_stiffness_symmetric_and_rigid_body_free() {
        let el = unit_quad(ElasticityModel::PlaneStrain);
        let mut mat =
            SubElementField::new(el.fespace.clone(), vec!["E".into(), "nu".into()]).unwrap();
        mat.set_uniform("E", 200.0).unwrap();
        mat.set_uniform("nu", 0.3).unwrap();
        let mat = insert(mat);
        let blocks = el.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = read(&el.support).unwrap().connectivity().to_vec();
        let tol = 1e-9;
        // Symmetry K[(i,f_a),(j,u_b)] == K[(j,f_b),(i,u_a)].
        for &ni in &nodes {
            for &nj in &nodes {
                for a in ["x", "y"] {
                    for b in ["x", "y"] {
                        let lhs = k.get(ni, &format!("f_{a}"), nj, &format!("u_{b}"));
                        let rhs = k.get(nj, &format!("f_{b}"), ni, &format!("u_{a}"));
                        assert!((lhs - rhs).abs() < tol);
                    }
                }
            }
        }
        // A uniform translation in x ⇒ zero force everywhere (row sum = 0).
        for &ni in &nodes {
            let row: f64 = nodes.iter().map(|&nj| k.get(ni, "f_x", nj, "u_x")).sum();
            assert!(row.abs() < tol, "row sum {row} ≠ 0");
        }
    }
}
