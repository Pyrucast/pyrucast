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
use crate::handle::Handle;
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::ZoneLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Column DOF name of a species — `c_H2`.
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
/// #     zone.clone(), vec!["D_H2".into()], &[2.0]).unwrap());
/// # use pyrucast::models::fick;
/// // L'espèce nomme la concentration : c'est ce qui permet à plusieurs
/// // diffusions de coexister dans un même modèle.
/// assert_eq!(fick::primal_var("H2"), "c_H2");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn primal_var(species: &str) -> String {
    format!("c_{species}")
}

/// Row DOF name of a species — `j_H2`.
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
/// #     zone.clone(), vec!["D_H2".into()], &[2.0]).unwrap());
/// # use pyrucast::models::fick;
/// assert_eq!(fick::dual_var("H2"), "j_H2");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
/// # use pyrucast::models::fick::{self, Fick};
/// // L'espèce nomme tout : la concentration, le flux et la diffusivité.
/// let f = Fick::new(zone.clone(), "H2")?;
/// assert_eq!(f.primal_vars(), vec![fick::primal_var("H2")]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// #     zone.clone(), vec!["D_H2".into()], &[2.0]).unwrap());
    /// # use pyrucast::models::fick::Fick;
    /// # use pyrucast::models::SubModelKind;
    /// let f = Fick::new(zone.clone(), "H2")?;
    /// assert_eq!(f.primal_vars(), vec!["c_H2".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, species: &str) -> Result<Self> {
        Self::with_symmetry(fespace, MaterialSymmetry::Isotropic, species)
    }

    /// Fickian diffusion with an explicit material symmetry — the general
    /// constructor, of which [`new`](Self::new) is the isotropic case.
    ///
    /// The species names the concentration, the flux and the diffusivity; an
    /// empty one is refused, since it would give back the bare `c`/`j` the
    /// suffix exists to avoid.
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
    /// #     zone.clone(), vec!["D_H2".into()], &[2.0]).unwrap());
    /// # use pyrucast::models::fick::Fick;
    /// # use pyrucast::models::SubModelKind;
    /// let f = Fick::with_symmetry(zone.clone(), MaterialSymmetry::Isotropic, "H2")?;
    /// assert_eq!(f.dual_vars(), vec!["j_H2".to_string()]);
    /// // Une espèce vide est refusée : elle rendrait les `c`/`j` nus que le
    /// // suffixe existe précisément pour éviter.
    /// assert!(Fick::with_symmetry(zone.clone(), MaterialSymmetry::Isotropic, "").is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
            let s = fespace.read();
            (s.submesh(), s.space_dim())
        };
        let support = submesh.read().to_poi1()?;
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
    fn internal_force_reads(&self) -> Vec<String> {
        flux_components(self.space_dim, &self.species)
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = geom.space_dim;
        for g in 0..geom.n_gauss {
            let dn = geom.dn_dx(g)?;
            let w = geom.det_j_w(g)?;
            for i in 0..geom.n_nodes {
                let mut s = 0.0;
                for a in 0..d {
                    s += dn[i * d + a] * stress.get(geom.cell, g, lay[a] as usize)?;
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
        let n = self.support.read().cell_count();
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
    fn material_components(&self) -> Vec<String> {
        material_contract(self.symmetry, self.space_dim, &self.species)
    }

    /// `poro` — required only by the storage (mass) matrix, never by the
    /// diffusion assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        STORAGE_COMPONENTS
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        flux_components(self.space_dim, &self.species)
    }

    /// Fick's law: weak-form flux `D·∇c` at one Gauss point.
    fn deformation_reads(&self) -> Vec<String> {
        gradient_components(self.space_dim, &self.species)
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
        let d3 = symmetry::transport_tensor_from(
            |k| material[lay.material[k] as usize],
            self.symmetry,
            space_dim,
        )?;
        for a in 0..space_dim {
            let mut acc = 0.0;
            for b in 0..space_dim {
                acc += d3[(a, b)] * deformation[lay.deformation[b] as usize];
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
/// #     zone.clone(), vec!["D_H2".into()], &[2.0]).unwrap());
/// # use pyrucast::models::fick;
/// // Le même laplacien que la conduction, avec la diffusivité de l'espèce.
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["j_H2".into()], vec!["c_H2".into()], DofOrdering::NodesThenVars, true,
///     Some(&mat), None,
///     |geoms, m, _s, ke| fick::element_stiffness(
///         &geoms[0], m.unwrap(), MaterialSymmetry::Isotropic, "H2", ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!(total.abs() < 1e-9); // une concentration uniforme ne diffuse pas
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
/// #     zone.clone(), vec!["poro".into()], &[3.0]).unwrap());
/// # use pyrucast::models::fick;
/// // Le pendant de la capacité thermique, côté transport de masse.
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["j_H2".into()], vec!["c_H2".into()], DofOrdering::NodesThenVars, true,
///     Some(&mat), None,
///     |geoms, m, _s, ke| fick::element_storage(&geoms[0], m.unwrap(), ke),
/// )?;
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!((total - 3.0 * 0.5).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
