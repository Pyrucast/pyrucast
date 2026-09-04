//! Linear heat-conduction physics — assembly of the cell-wise stiffness
//! `K_ij = ∫ k · ∇N_i · ∇N_j dx`.
//!
//! Primal variable `"T"` (temperature, columns), dual `"q"` (heat flux,
//! rows). The conductivity is read from a [`SubElementField`] component
//! named [`MATERIAL_COMPONENT`].

use crate::containers::element_field::SubElementField;
use crate::containers::field::ABSENT_COMPONENT;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::owned_components;
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::ElementLayout;
use crate::models::ZoneLayout;
use crate::models::{Behavior, CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature).
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
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
/// # use pyrucast::models::heat_conduction;
/// // La primale de la conduction : c'est ce nom que doit citer une loi de
/// // bord ou une contrainte pour s'y coupler.
/// assert_eq!(heat_conduction::PRIMAL_VAR, "T");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux).
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
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
/// # use pyrucast::models::heat_conduction;
/// assert_eq!(heat_conduction::DUAL_VAR, "q");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const DUAL_VAR: &str = "q";
/// Required component on the material `SubElementField` (isotropic
/// conductivity).
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
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
/// # use pyrucast::models::heat_conduction;
/// // La conductivité isotrope. L'orthotropie en demande davantage.
/// assert_eq!(heat_conduction::MATERIAL_COMPONENT, "k");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::heat_conduction::{self, HeatConduction};
/// let hc = HeatConduction::new(zone.clone())?;
/// assert_eq!(hc.primal_vars(), vec![heat_conduction::PRIMAL_VAR.to_string()]);
/// assert_eq!(hc.dual_vars(), vec![heat_conduction::DUAL_VAR.to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
    /// # use pyrucast::models::heat_conduction::HeatConduction;
    /// # use pyrucast::models::Domain;
    /// // Bâtit le POI1 stable des nœuds de la zone, réutilisé comme support de
    /// // ligne et de colonne de **tous** les blocs assemblés.
    /// let hc = HeatConduction::new(zone.clone())?;
    /// assert_eq!(hc.material_components(), vec!["k".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Self::with_symmetry(fespace, MaterialSymmetry::Isotropic)
    }

    /// Heat conduction with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case. An
    /// orthotropic or anisotropic conductivity carries its material axes through
    /// the material field (see [`crate::models::symmetry`]).
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
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
    /// # use pyrucast::models::heat_conduction::HeatConduction;
    /// # use pyrucast::models::Domain;
    /// // Le constructeur général, dont `new` est le cas isotrope.
    /// let ortho = HeatConduction::with_symmetry(zone.clone(), MaterialSymmetry::Orthotropic)?;
    /// assert!(ortho.material_components().len() > 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        let (submesh, space_dim) = {
            let s = fespace.read();
            (s.submesh(), s.space_dim())
        };
        let support = submesh.read().to_poi1()?;
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

    fn as_behavior(&self) -> Option<&dyn Behavior> {
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

    /// Internal nodal fluxes `q_i = ∫ ∇N_i · flux dx` of one cell — `Bᵀ` applied
    /// to the weak-form flux, the scalar-transport counterpart of the
    /// continuum-mechanics default (and identical to
    /// [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence) of the
    /// flux vector). Single dual variable `q`, so `fe[i]` per node.
    fn internal_force_reads(&self) -> Vec<String> {
        flux_components(self.space_dim)
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        _material: &SubElementField,
        _mat: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = geom.space_dim;
        let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
        for g in 0..geom.n_gauss {
            let dn = &mut dn_buf[..geom.n_nodes * d];
            geom.dn_dx(g, dn)?; // [i * d + a]
            let w = geom.det_j_w(g);
            // The flux row, sliced once: its bounds were settled with the zone,
            // so a node no longer re-proves them component by component.
            let row = stress.row(geom.cell, g);
            for i in 0..geom.n_nodes {
                let mut s = 0.0;
                for a in 0..d {
                    s += dn[i * d + a] * row[lay[a] as usize];
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
        let n = self.support.read().cell_count();
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

    fn material_components(&self) -> Vec<String> {
        owned_components(material_contract(self.symmetry, self.space_dim))
    }

    /// `rho` + `cp` — required only by the heat-capacity (mass) matrix, never by
    /// the conductivity assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        CAPACITY_COMPONENTS
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        element_stiffness(&geoms[0], material, lay, self.symmetry, ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        element_capacity(&geoms[0], material, lay, ke)
    }
}

impl Behavior for HeatConduction {
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        flux_components(self.fespace.read().space_dim())
    }

    /// Linear constitutive law: weak-form flux = k·∇T at one Gauss point.
    /// (No internal-state variables — `VAR0`/`VAR1` are empty; a non-linear law
    /// would read trailing state components of `input` and write updated ones.)
    fn deformation_reads(&self) -> Vec<String> {
        deformation_components(self.space_dim)
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
        let space_dim = geom.space_dim;
        let k3 = symmetry::transport_tensor_from(
            |k| material[lay.material[k] as usize],
            self.symmetry,
            space_dim,
        )?;
        for a in 0..space_dim {
            let mut acc = 0.0;
            for b in 0..space_dim {
                acc += k3[(a, b)] * deformation[lay.deformation[b] as usize];
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
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[2.0]).unwrap());
/// # use pyrucast::models::heat_conduction;
/// # use pyrucast::models::ElementLayout;
/// // Le champ est rangé dans l'ordre du contrat : la table est l'identité.
/// let lay = ElementLayout { material: vec![0], optional_material: vec![], state: vec![] };
/// // ∫ ∇Nᵀ k ∇N. Une conductivité constante donne une matrice singulière :
/// // un champ de température uniforme ne conduit rien.
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
///     &mat, None,
///     |geoms, m, _s, ke| heat_conduction::element_stiffness(
///         &geoms[0], m, &lay, MaterialSymmetry::Isotropic, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!(total.abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    geom: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    symmetry: MaterialSymmetry,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    let space_dim = geom.space_dim;
    let cell_row = material.row(geom.cell, 0);
    // An oriented conductivity is built **once per cell** — its constants and its
    // material axes are cell-wise. Isotropy keeps reading its scalar at each
    // Gauss point, so a conductivity varying inside a cell still works.
    let tensor = if symmetry.has_frame() {
        Some(symmetry::transport_tensor_by(
            cell_row,
            cell_row,
            &lay.material,
            symmetry,
            space_dim,
        )?)
    } else {
        None
    };
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    for g in 0..geom.n_gauss {
        let dn = &mut dn_buf[..n_nodes * geom.space_dim];
        geom.dn_dx(g, dn)?;
        let det_j_w = geom.det_j_w(g);
        match &tensor {
            None => {
                // The scalar conductivity, by the index the zone resolved.
                let k = material.row(geom.cell, g)[lay.material[0] as usize];
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
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["rho".into(), "cp".into()], &[3.0, 4.0]).unwrap());
/// # use pyrucast::models::heat_conduction;
/// # use pyrucast::models::ElementLayout;
/// // `rho` et `cp` sont les deux composantes **facultatives** du contrat de
/// // conduction : la conductivité ne les demande jamais, la capacité si.
/// let lay = ElementLayout {
///     material: vec![], optional_material: vec![0, 1], state: vec![],
/// };
/// // ∫ ρ c_p Nᵀ N : la capacité thermique. Sa somme vaut ρ·c_p × aire —
/// // la capacité de la maille entière.
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
///     &mat, None,
///     |geoms, m, _s, ke| heat_conduction::element_capacity(&geoms[0], m, &lay, ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!((total - 3.0 * 4.0 * 0.5).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_capacity(
    geom: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    ke: &mut [f64],
) -> Result<()> {
    let n_nodes = geom.n_nodes;
    // `rho` and `cp` are the two optional components of the conduction contract
    // (`CAPACITY_COMPONENTS`): the conductivity assembly never asks for them,
    // the capacity cannot do without.
    let row = material.row(geom.cell, 0);
    let read = |slot: usize, what: &str| -> Result<f64> {
        match lay.optional_material[slot] {
            ABSENT_COMPONENT => Err(crate::error::PyrucastError::Message(format!(
                "HeatConduction capacity matrix: material component `{}` ({what}) is required",
                CAPACITY_COMPONENTS[slot]
            ))),
            i => Ok(row[i as usize]),
        }
    };
    let rho_cp = read(0, "density")? * read(1, "specific heat")?;
    for g in 0..geom.n_gauss {
        let n = geom.n_at_g(g);
        let w = geom.det_j_w(g) * rho_cp;
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

    /// The rest state of `d` on its material — the `prev` of a first step,
    /// which the behaviour operator materializes for a caller who has none.
    fn rest<B: Behavior>(b: &B, mat: &Handle<SubElementField>) -> Handle<SubElementField> {
        Handle::new(b.initial_state(&mat.read()).unwrap())
    }
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// HeatConduction on a single SEG2 of length `L`.
    fn seg2_hc(length: f64) -> HeatConduction {
        let coords = Handle::new(Coords::new(1).unwrap());
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
        assert!(fe.same_object(&hc.fespace));
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
        let def = Handle::new(def);

        let mut mat =
            SubElementField::new(hc.fespace.clone(), vec![MATERIAL_COMPONENT.to_string()]).unwrap();
        mat.set_uniform(MATERIAL_COMPONENT, k).unwrap();
        let mat = Handle::new(mat);

        let flux = hc
            .integrate_behavior(&def, &rest(&hc, &mat), &mat, 0.0)
            .unwrap();
        assert_eq!(flux.components(), &["flux_x".to_string()]);
        let expected = k * grad;
        for g in 0..flux.gauss_count() {
            assert!((flux.value(0, g, "flux_x").unwrap() - expected).abs() < 1e-12);
        }
    }
}
