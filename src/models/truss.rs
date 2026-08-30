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
use crate::containers::field::ABSENT_COMPONENT;
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
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::models::{ElementLayout, MatrixKind};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
pub(crate) const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by the truss physics.
const MATERIAL_COMPONENTS: &[&str] = &["E", "A"];

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
/// # use pyrucast::models::truss::Truss;
/// // La barre : deux constantes, aucune flexion.
/// let t = Truss::new(zone.clone())?;
/// assert_eq!(t.material_components(), vec!["E".to_string(), "A".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// # use pyrucast::models::truss::Truss;
    /// // La barre : deux constantes, aucune flexion.
    /// let t = Truss::new(zone.clone())?;
    /// assert_eq!(t.material_components(), vec!["E".to_string(), "A".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, axisymmetric) = {
            let s = fespace.read();
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
        let support = submesh.read().to_poi1()?;
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

    /// Internal forces `f = Bᵀ N` of one bar. `B` projects the nodal
    /// displacement onto the axis (`ε = (u_B − u_A)·c / L`), so its transpose
    /// spreads the axial force `N` back onto the two ends: `f_A = −N c`,
    /// `f_B = +N c` — the equilibrating end forces along the direction cosine
    /// `c`. `N` is element-constant (linear bar), read at the first Gauss point;
    /// the closed form mirrors [`element_stiffness`]'s analytic treatment (a
    /// SEG2 in space has no square isoparametric Jacobian).
    fn internal_force_reads(&self) -> Vec<String> {
        vec!["n".to_string()]
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = self.space_dim;
        let mut c = [0.0_f64; 3];
        cell_cosine(geom, d, &mut c)?;
        let n = stress.row(geom.cell, 0)[lay[0] as usize];
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
        let n = self.support.read().cell_count();
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

    fn material_components(&self) -> Vec<String> {
        owned_components(MATERIAL_COMPONENTS)
    }

    /// `rho` (density) — required only by the mass matrix.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        vec!["n".to_string()]
    }

    /// Axial force `N = E·A·ε_axial` at one Gauss point, with `ε_axial = cᵀ ε c`
    /// and `c` the cell's unit direction cosine (from its node coordinates).
    fn deformation_reads(&self) -> Vec<String> {
        strain_names(self.space_dim)
    }

    fn integrate_point(
        &self,
        geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        let d = self.space_dim;
        // Sur la pile : ce noyau tourne à chaque point de Gauss.
        let mut c = [0.0_f64; 3];
        cell_cosine(geom, d, &mut c)?;
        // (i,j) → flat strain-component index (symmetric, i ≤ j), the order
        // `strain_names` declares and therefore the order of `lay.deformation`.
        let comp_index = |i: usize, j: usize| -> usize {
            let (i, j) = if i <= j { (i, j) } else { (j, i) };
            (0..i).map(|r| d - r).sum::<usize>() + (j - i)
        };
        let (e, a) = (
            material[lay.material[0] as usize],
            material[lay.material[1] as usize],
        );
        let mut eps_axial = 0.0;
        for i in 0..d {
            for j in 0..d {
                let eps_ij = deformation[lay.deformation[comp_index(i, j)] as usize];
                eps_axial += c[i] * eps_ij * c[j];
            }
        }
        out[0] = e * a * eps_axial;
        Ok(())
    }

    /// La raideur géométrique de la barre lit son effort normal.
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        match kind {
            MatrixKind::Geometric => vec!["n".to_string()],
            _ => Vec::new(),
        }
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        element_stiffness(&geoms[0], material, lay, ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        element_mass(&geoms[0], material, lay, ke)
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: &SubElementField,
        lay: &ElementLayout,
        state: &SubElementField,
        ke: &mut [f64],
    ) -> Result<()> {
        element_geometric(&geoms[0], state, lay, ke)
    }
}

/// Unit direction cosine vector `c = (x_B − x_A)/L` of one `SEG2` cell, from its
/// two node coordinates.
fn cell_cosine(geom: &CellGeom, space_dim: usize, out: &mut [f64; 3]) -> Result<f64> {
    let xa = geom.node_coord(0);
    let xb = geom.node_coord(1);
    for a in 0..space_dim {
        out[a] = xb[a] - xa[a];
    }
    let len = out[..space_dim].iter().map(|v| v * v).sum::<f64>().sqrt();
    // Le `Result` gardait le vide : on divisait par `len` sans jamais le
    // tester, et une maille dégénérée rendait `Ok(NaN)` — un résultat faux
    // qui traversait tout l'assemblage. Maintenant il garde quelque chose.
    if len <= f64::EPSILON {
        return Err(crate::error::PyrucastError::Message(format!(
            "Truss: cell {} has zero length",
            geom.cell
        )));
    }
    for v in &mut out[..space_dim] {
        *v /= len;
    }
    Ok(len)
}

/// Element kernel: local truss stiffness `K_e = (E·A/L)·[[c⊗c,−c⊗c],…]` of one
/// `SEG2`, written into `ke` (flat row-major, side `2·space_dim`, **node-major /
/// component-minor** dof order). `c` is the unit direction cosine from the cell's
/// node coordinates. Pure and sequential — driven in parallel by
/// [`crate::models::kernel::assemble_block`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "A".into()], &[210000.0, 0.01]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string()], vec!["u_x".to_string(), "u_y".to_string()]);
/// # use pyrucast::models::{truss, ElementLayout};
/// // `E` puis `A` : le champ suit le contrat, la table est l'identité.
/// let lay = ElementLayout {
///     material: vec![0, 1], optional_material: vec![], state: vec![],
/// };
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, s, ke| truss::element_stiffness(&geoms[0], m, &lay, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// // Une barre libre est singulière : la translation d'ensemble ne coûte rien.
/// assert!(total.abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    ke: &mut [f64],
) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    // Les cosinus directeurs **et** la longueur, d'un seul parcours.
    let mut c = [0.0_f64; 3];
    let len = cell_cosine(geom, sd, &mut c)?;
    // `E` then `A`, in the order `MATERIAL_COMPONENTS` declares.
    let row = material.row(geom.cell, 0);
    let k_ax = row[lay.material[0] as usize] * row[lay.material[1] as usize] / len;
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
/// Element kernel: local **consistent mass** of one bar,
///   `M[(i,a),(j,b)] = δ_ab · (ρ A L / 6) · (2 if i==j else 1)`
/// (the linear-element mass `(ρAL/6)[[2,1],[1,2]]` on each translation
/// component), written into `ke` (same layout as [`element_stiffness`]). Reads
/// density `rho` and area `A`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["rho".into(), "A".into()], &[3.0, 0.01]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string()], vec!["u_x".to_string(), "u_y".to_string()]);
/// # use pyrucast::models::{truss, ElementLayout};
/// # use pyrucast::containers::field::ABSENT_COMPONENT;
/// // La masse lit `A` (seconde composante requise) et `rho` (la seule
/// // facultative) ; `E` n'entre pas, et n'est pas dans ce champ-ci.
/// let lay = ElementLayout {
///     material: vec![ABSENT_COMPONENT, 1],
///     optional_material: vec![0],
///     state: vec![],
/// };
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, s, ke| truss::element_mass(&geoms[0], m, &lay, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// // La masse cohérente somme à ρ·A·L par direction : ici deux fois 0,06.
/// assert!((total - 2.0 * 3.0 * 0.01 * 2.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_mass(
    geom: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    ke: &mut [f64],
) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    let mut c = [0.0_f64; 3];
    let len = cell_cosine(geom, sd, &mut c)?;
    let row = material.row(geom.cell, 0);
    // `rho` is the bar's only optional component: without it there is no mass.
    let rho = match lay.optional_material[0] {
        ABSENT_COMPONENT => {
            return Err(crate::error::PyrucastError::Message(
                "Truss mass matrix: material component `rho` (density) is required".into(),
            ))
        }
        i => row[i as usize],
    };
    let a = row[lay.material[1] as usize];
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["n".into()], &[100.0]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string()], vec!["u_x".to_string(), "u_y".to_string()]);
/// # use pyrucast::models::{truss, ElementLayout};
/// // La raideur géométrique ne lit que l'effort normal `n`.
/// let lay = ElementLayout {
///     material: vec![], optional_material: vec![], state: vec![0],
/// };
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, Some(&mat),
///     |geoms, m, s, ke| truss::element_geometric(&geoms[0], s.unwrap(), &lay, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// // La raideur **géométrique** vient de l'effort normal, non du matériau.
/// // Elle est singulière elle aussi.
/// assert!(total.abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_geometric(
    geom: &CellGeom,
    state: &SubElementField,
    lay: &ElementLayout,
    ke: &mut [f64],
) -> Result<()> {
    let sd = geom.space_dim;
    let side = 2 * sd;
    let mut c = [0.0_f64; 3];
    let len = cell_cosine(geom, sd, &mut c)?;
    let k = state.row(geom.cell, 0)[lay.state[0] as usize] / len;
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

    /// The rest state of `d` on its material — the `prev` of a first step,
    /// which the behaviour operator materializes for a caller who has none.
    fn rest<D: Domain>(d: &D, mat: &Handle<SubElementField>) -> Handle<SubElementField> {
        Handle::new(d.initial_state(&mat.read()).unwrap())
    }
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Truss on a single inclined SEG2 in 2-D, returns `(model, a_id, b_id)`.
    fn inclined_bar(e: f64, area: f64, dx: f64, dy: f64) -> (Truss, NodeId, NodeId, f64, [f64; 2]) {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        Handle::new(m)
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
        let blocks = truss.build_stiffness_blocks(&mat).unwrap();
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
        let strain = Handle::new(strain);

        let out = truss
            .integrate_behavior(&strain, &rest(&truss, &mat), &mat, 0.0)
            .unwrap();
        assert_eq!(out.components(), &["n".to_string()]);
        let expected = e * area * eps0;
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "n").unwrap() - expected).abs() < 1e-9);
        }
    }
}

// ANCHOR: operator
crate::physics_operator! {
    /// Truss / bar `Model` spanning **every** subspace of `fes` — one
    /// [`SubModel::Truss`] per
    /// [`SubFiniteElementSpace`].
    /// Parent-level operator; material (`E`, `A`) is supplied at assembly time.
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
    /// # let mut b = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # b.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let barres = FiniteElementSpace::lagrange1(&Mesh::from_submesh(b))?;
    /// let m = model::truss(&barres)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn truss(fes) via SubModel::truss;
    python: "`model.truss(fespace)` — truss / bar (axial-force) model spanning\n**every** subspace of `fespace` (SEG2 elements). DOFs are the vector\ndisplacement `u_x, u_y(, u_z)`; the orientation is taken from the node\ncoordinates. Material (`E`, `A`) is supplied at assembly time."
}
// ANCHOR_END: operator
