//! Truss (bar) physics — axial-force element in any spatial dimension.
//!
//! A 2-node `SEG2` bar carrying **axial force only**. Its orientation is read
//! from the node coordinates (direction cosines `c = (x_B − x_A)/L`), so the
//! same code works in 1-D, 2-D and 3-D: the global element stiffness is
//!
//! ```text
//! K_e = (E·A / L) · [[ c⊗c, −c⊗c ],
//!                    [ −c⊗c,  c⊗c ]]
//! ```
//!
//! Primal variables `u_x, u_y, …` (displacement, one per axis), dual `f_x, …`
//! (nodal force). Material components `E` (Young's modulus) and `A` (section
//! area), read from a [`SubElementField`].

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
pub(crate) const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by the truss physics.
const MATERIAL_COMPONENTS: &[&str] = &["E", "A"];

/// Primal (displacement) component name for axis `a`: `u_x`, `u_y`, `u_z`.
fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
/// Dual (force) component name for axis `a`: `f_x`, `f_y`, `f_z`.
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}
/// Strain-tensor component names (`eps_xx`, `eps_xy`, …) for `i ≤ j`, matching
/// what [`crate::ops::element_field::deformation`] produces — the behaviour input.
fn strain_names(space_dim: usize) -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..space_dim {
        for j in i..space_dim {
            names.push(format!("eps_{}{}", AXES[i], AXES[j]));
        }
    }
    names
}

/// Truss / bar physics on a `SEG2` FE subspace.
///
/// Material data (`E`, `A`) is supplied at assembly time via
/// [`crate::ops::matrix::stiffness`], not stored here.
#[derive(Clone, Serialize, Deserialize)]
pub struct Truss {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support covering the subspace's unique nodes (row/col support of
    /// every assembled block).
    pub(crate) support: Handle<SubMesh>,
    /// Spatial dimension (number of displacement components per node).
    pub(crate) space_dim: usize,
}

impl Truss {
    /// Truss physics on a `SEG2` FE subspace. Builds the stable POI1
    /// [`SubMesh`] over the subspace's unique nodes.
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, axisymmetric) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim(), s.is_axisymmetric())
        };
        // A segment in a meridian plane sweeps a cone of revolution, not a bar:
        // the kernel's `E·A/L` has no meaning there.
        if axisymmetric {
            return Err(PyrucastError::Message(
                "Truss: axisymmetric geometries are not supported — a segment in a \
                 meridian plane is a shell of revolution, not a bar"
                    .into(),
            ));
        }
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
        })
    }
}

impl SubModelKind for Truss {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
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

    /// Mass and geometric-stiffness blocks share the stiffness layout (same
    /// SEG2 fespace, node support, translational DOFs).
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn geometric_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        element_stiffness(geom, material.expect("Truss requires a material field"), ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        element_mass(geom, material.expect("Truss requires a material field"), ke)
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("truss geometric stiffness requires the axial force `n`");
        element_geometric(geom, stress, ke)
    }

    /// Internal forces `f = Bᵀ N` of one bar. `B` projects the nodal
    /// displacement onto the axis (`ε = (u_B − u_A)·c / L`), so its transpose
    /// spreads the axial force `N` back onto the two ends: `f_A = −N c`,
    /// `f_B = +N c` — the equilibrating end forces along the direction cosine
    /// `c`. `N` is element-constant (linear bar), read at the first Gauss point;
    /// the closed form mirrors [`element_stiffness`]'s analytic treatment (a
    /// SEG2 in space has no square isoparametric Jacobian).
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = self.space_dim;
        let c = cell_cosine(geom, d)?;
        let n = stress.value(geom.cell, 0, "n")?;
        for a in 0..d {
            fe[a] = -n * c[a]; // node A
            fe[d + a] = n * c[a]; // node B
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Truss"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Truss>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)"
        )
    }
}

impl Domain for Truss {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    /// `rho` (density) — required only by the mass matrix.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Ok(vec!["n".to_string()])
    }

    /// Axial force `N = E·A·ε_axial` at one Gauss point, with `ε_axial = cᵀ ε c`
    /// and `c` the cell's unit direction cosine (from its node coordinates).
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
        let mat = material.expect("Truss declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let c = cell_cosine(geom, d)?;
        let strain = strain_names(d);
        // (i,j) → flat strain-component index (symmetric, i ≤ j).
        let comp_index = |i: usize, j: usize| -> usize {
            let (i, j) = if i <= j { (i, j) } else { (j, i) };
            (0..i).map(|r| d - r).sum::<usize>() + (j - i)
        };
        let e = mat.value(cell, 0, "E")?;
        let a = mat.value(cell, 0, "A")?;
        let mut eps_axial = 0.0;
        for i in 0..d {
            for j in 0..d {
                let eps_ij = input.value(cell, g, &strain[comp_index(i, j)])?;
                eps_axial += c[i] * eps_ij * c[j];
            }
        }
        out[0] = e * a * eps_axial;
        Ok(())
    }
}

/// Unit direction cosine vector `c = (x_B − x_A)/L` of one `SEG2` cell, from its
/// two node coordinates.
fn cell_cosine(geom: &CellGeom, space_dim: usize) -> Result<Vec<f64>> {
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    let d: Vec<f64> = (0..space_dim).map(|a| xb[a] - xa[a]).collect();
    let len = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(d.iter().map(|v| v / len).collect())
}

/// Element kernel: local truss stiffness `K_e = (E·A/L)·[[c⊗c,−c⊗c],…]` of one
/// `SEG2`, written into `ke` (flat row-major, side `2·space_dim`, **node-major /
/// component-minor** dof order). `c` is the unit direction cosine from the cell's
/// node coordinates. Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    let c = cell_cosine(geom, sd)?;
    let len = {
        let xa = geom.node_coord(0)?;
        let xb = geom.node_coord(1)?;
        (0..sd).map(|a| (xb[a] - xa[a]).powi(2)).sum::<f64>().sqrt()
    };
    let k_ax = material.value(geom.cell, 0, "E")? * material.value(geom.cell, 0, "A")? / len;
    for ii in 0..2 {
        for jj in 0..2 {
            let sign = if ii == jj { 1.0 } else { -1.0 };
            for a in 0..sd {
                for b in 0..sd {
                    ke[(ii * sd + a) * side + (jj * sd + b)] = sign * k_ax * c[a] * c[b];
                }
            }
        }
    }
    Ok(())
}

/// Length of one `SEG2` cell from its two node coordinates.
fn cell_length(geom: &CellGeom, space_dim: usize) -> Result<f64> {
    let xa = geom.node_coord(0)?;
    let xb = geom.node_coord(1)?;
    Ok((0..space_dim)
        .map(|a| (xb[a] - xa[a]).powi(2))
        .sum::<f64>()
        .sqrt())
}

/// Element kernel: local **consistent mass** of one bar,
///   `M[(i,a),(j,b)] = δ_ab · (ρ A L / 6) · (2 if i==j else 1)`
/// (the linear-element mass `(ρAL/6)[[2,1],[1,2]]` on each translation
/// component), written into `ke` (same layout as [`element_stiffness`]). Reads
/// density `rho` and area `A`.
pub fn element_mass(geom: &CellGeom, material: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    let len = cell_length(geom, sd)?;
    let rho = material.value(geom.cell, 0, "rho").map_err(|_| {
        crate::error::PyrucastError::Message(
            "Truss mass matrix: material component `rho` (density) is required".into(),
        )
    })?;
    let a = material.value(geom.cell, 0, "A")?;
    let m = rho * a * len / 6.0;
    for ii in 0..2 {
        for jj in 0..2 {
            let coef = m * if ii == jj { 2.0 } else { 1.0 };
            for aa in 0..sd {
                ke[(ii * sd + aa) * side + (jj * sd + aa)] += coef;
            }
        }
    }
    Ok(())
}

/// Element kernel: local **geometric (initial-stress) stiffness** of one bar
/// under axial force `N`,
///   `K_g = (N / L) · [[P, −P], [−P, P]]`,   `P = I − c⊗c`
/// (the transverse projector, so only motion perpendicular to the bar axis is
/// stiffened). `N` is read from the state component `n`. Same `ke` layout as
/// [`element_stiffness`].
pub fn element_geometric(geom: &CellGeom, state: &SubElementField, ke: &mut [f64]) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    let c = cell_cosine(geom, sd)?;
    let len = cell_length(geom, sd)?;
    let k = state.value(geom.cell, 0, "n")? / len;
    for ii in 0..2 {
        for jj in 0..2 {
            let sign = if ii == jj { 1.0 } else { -1.0 };
            for a in 0..sd {
                for b in 0..sd {
                    let p = (if a == b { 1.0 } else { 0.0 }) - c[a] * c[b];
                    ke[(ii * sd + a) * side + (jj * sd + b)] += sign * k * p;
                }
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
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::store::insert;

    /// Truss on a single inclined SEG2 in 2-D, returns `(model, a_id, b_id)`.
    fn inclined_bar(e: f64, area: f64, dx: f64, dy: f64) -> (Truss, NodeId, NodeId, f64, [f64; 2]) {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[dx, dy]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let truss = Truss::new(fes.get(0).unwrap()).unwrap();
        let len = (dx * dx + dy * dy).sqrt();
        let _ = (e, area);
        (truss, a.id(), b.id(), len, [dx / len, dy / len])
    }

    fn material(truss: &Truss, e: f64, area: f64) -> Handle<SubElementField> {
        let mut m =
            SubElementField::new(truss.fespace.clone(), vec!["E".into(), "A".into()]).unwrap();
        m.set_uniform("E", e).unwrap();
        m.set_uniform("A", area).unwrap();
        insert(m)
    }

    #[test]
    fn vars_follow_space_dim() {
        let (truss, ..) = inclined_bar(1.0, 1.0, 3.0, 4.0);
        assert_eq!(truss.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(truss.dual_vars(), vec!["f_x", "f_y"]);
    }

    /// Inclined-bar global stiffness: `K[(I,f_a),(J,u_b)] = s(I,J)·(EA/L)·c_a·c_b`.
    #[test]
    fn inclined_bar_stiffness_matches_direction_cosines() {
        let (e, area) = (210.0, 2.0);
        let (truss, a, b, len, c) = inclined_bar(e, area, 3.0, 4.0);
        let mat = material(&truss, e, area);
        let blocks = truss.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let k_ax = e * area / len;
        let tol = 1e-9;
        // Diagonal block at A.
        assert!((k.get(a, "f_x", a, "u_x") - k_ax * c[0] * c[0]).abs() < tol);
        assert!((k.get(a, "f_x", a, "u_y") - k_ax * c[0] * c[1]).abs() < tol);
        assert!((k.get(a, "f_y", a, "u_y") - k_ax * c[1] * c[1]).abs() < tol);
        // Coupling A–B is the negative.
        assert!((k.get(a, "f_x", b, "u_x") + k_ax * c[0] * c[0]).abs() < tol);
        assert!((k.get(b, "f_y", a, "u_x") + k_ax * c[1] * c[0]).abs() < tol);
        assert!((k.get(b, "f_y", b, "u_y") - k_ax * c[1] * c[1]).abs() < tol);
    }

    /// COMP: a pure axial strain `ε = ε₀·(c⊗c)` gives axial force `N = EA·ε₀`.
    #[test]
    fn integrate_behavior_returns_axial_force() {
        let (e, area) = (100.0, 3.0);
        let (truss, _a, _b, _len, c) = inclined_bar(e, area, 3.0, 4.0);
        let mat = material(&truss, e, area);
        let eps0 = 0.01;

        let mut strain = SubElementField::new(truss.fespace.clone(), strain_names(2)).unwrap();
        // ε = ε₀ (c⊗c): eps_xx = ε₀ c_x², eps_xy = ε₀ c_x c_y, eps_yy = ε₀ c_y².
        strain.set_uniform("eps_xx", eps0 * c[0] * c[0]).unwrap();
        strain.set_uniform("eps_xy", eps0 * c[0] * c[1]).unwrap();
        strain.set_uniform("eps_yy", eps0 * c[1] * c[1]).unwrap();
        let strain = insert(strain);

        let out = truss
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        assert_eq!(out.components(), &["n".to_string()]);
        let expected = e * area * eps0;
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "n").unwrap() - expected).abs() < 1e-9);
        }
    }
}
