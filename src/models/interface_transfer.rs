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
//! assembler: [`Domain::element_matrix`] gives `+h∫N_iN_j` and
//! [`Domain::coupling_element`] gives `−h∫N_iN_j`. Because each block picks
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

use crate::aggregate::Aggregate;
use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::coords::Coords;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::transfer::{
    coefficient_name, exchange_matrix, jump_name, material_contract, physics_slice,
};
use crate::models::ElementLayout;
use crate::models::{
    CellGeom, Contribution, CouplingLayout, Domain, MatrixKind, MatrixLayout, Physics, SubModelKind,
};
use serde::{Deserialize, Serialize};

/// Exchange law between two conforming boundary FE subspaces.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
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
/// # use pyrucast::models::interface_transfer::InterfaceTransfer;
/// # use pyrucast::models::Domain;
/// // Deux bords **conformes** — ici le même, ce qui suffit à montrer le
/// // contrat ; en pratique deux faces en vis-à-vis.
/// let i = InterfaceTransfer::new(zone.clone(), zone.clone(),
///     vec![("T".into(), "q".into())], Physics::Thermal, 1e-6)?;
/// assert_eq!(i.primal_vars(), vec!["T".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
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
    /// # use pyrucast::models::interface_transfer::InterfaceTransfer;
    /// # use pyrucast::models::Domain;
    /// // Deux bords **conformes** — ici le même, ce qui suffit à montrer le
    /// // contrat ; en pratique deux faces en vis-à-vis.
    /// let i = InterfaceTransfer::new(zone.clone(), zone.clone(),
    ///     vec![("T".into(), "q".into())], Physics::Thermal, 1e-6)?;
    /// assert_eq!(i.primal_vars(), vec!["T".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    /// Une interpolation de la primale aux points d'un côté.
    fn interp_side(
        fes: &Handle<SubFiniteElementSpace>,
        solution: &crate::containers::node_field::NodeField,
    ) -> Result<Handle<SubElementField>> {
        let mut one = crate::containers::finite_element_space::FiniteElementSpace::empty();
        one.add_sub(fes.clone())?;
        crate::ops::element_field::interp_to_gauss(solution, &one)?.get(0)
    }

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

    /// Internal fluxes `q_i = ∫ N_i · flux dΓ` — weighted by `N`, not by `Bᵀ`,
    /// exactly as for convection: the interface integrand is a flux **density**,
    /// not a gradient-conjugate quantity.
    /// Deux termes, un par côté : `∫ h·(a₁−a₂)·N` sur A et son opposé sur B.
    ///
    /// L'intégrale d'une différence est la différence des intégrales, et
    /// chacune se disperse sur **un** espace, le sien. Ce qui est couplé, c'est
    /// la matrice — lignes sur A, colonnes sur B — pas le vecteur : un résidu
    /// ne produit qu'un nombre par nœud.
    fn internal_force_contribution(&self) -> Vec<crate::models::ResidualContribution> {
        [
            (&self.side_a, &self.support_a),
            (&self.side_b, &self.support_b),
        ]
        .into_iter()
        .map(|(fes, support)| {
            crate::models::ResidualContribution::Computed(MatrixLayout {
                fespaces: vec![fes.clone()],
                support: support.clone(),
                dual_vars: self.dual_vars(),
                primal_vars: self.primal_vars(),
                ordering: DofOrdering::NodesThenVars,
                symmetric: true,
            })
        })
        .collect()
    }

    /// Le saut `a₁ − a₂` aux points, vu du côté `fespace` : positif sur A,
    /// négatif sur B. Les deux côtés étant **conformes** — même type d'élément,
    /// même nombre de mailles, la maille `i` de l'un face à la maille `i` de
    /// l'autre —, les deux interpolations s'alignent indice pour indice et le
    /// saut est une soustraction, pas une projection.
    fn residual_input(
        &self,
        fespace: &Handle<SubFiniteElementSpace>,
        solution: &crate::containers::node_field::NodeField,
    ) -> Result<Handle<SubElementField>> {
        let sur_a = Handle::same_object(fespace, &self.side_a);
        let a = Self::interp_side(&self.side_a, solution)?;
        let b = Self::interp_side(&self.side_b, solution)?;
        let (plus, moins) = if sur_a { (&a, &b) } else { (&b, &a) };
        let noms: Vec<String> = self.components.iter().map(|(p, _)| jump_name(p)).collect();
        let nj = noms.len();
        let mut out = SubElementField::new(fespace.clone(), noms)?;
        {
            let (p, m) = (plus.read(), moins.read());
            // Les deux champs interpolés portent **toutes** les composantes de
            // la solution, dans son ordre ; le saut n'en porte qu'une par
            // primale. Les indices se résolvent donc par nom, une fois pour la
            // zone, jamais par position — sitôt que la solution transporte un
            // multiplicateur ou une seconde physique, les deux dispositions
            // divergent.
            let primales: Vec<&str> = self.components.iter().map(|(v, _)| v.as_str()).collect();
            let ip = p.resolve_components(&primales, "solution")?;
            let im = m.resolve_components(&primales, "solution")?;
            let np = p.component_count();
            let nm = m.component_count();
            // Les deux côtés sont conformes par construction ; on le prouve
            // une fois ici plutôt que de laisser l'indexation le découvrir.
            let lignes = p.cell_count() * p.gauss_count();
            if m.cell_count() * m.gauss_count() != lignes {
                return Err(PyrucastError::Message(format!(
                    "InterfaceTransfer: the two sides carry {lignes} and {} integration \
                     point(s) — a jump needs them point for point",
                    m.cell_count() * m.gauss_count()
                )));
            }
            let (vp, vm) = (p.values(), m.values());
            let dst = out.values_mut();
            for row in 0..lignes {
                for v in 0..nj {
                    dst[row * nj + v] =
                        vp[row * np + ip[v] as usize] - vm[row * nm + im[v] as usize];
                }
            }
        }
        Ok(Handle::new(out))
    }

    /// Le **saut**, pas un flux : ce que le terme lit au point est
    /// `a₁ − a₂`, et c'est lui qu'il multiplie par le coefficient.
    fn internal_force_reads(&self) -> Vec<String> {
        self.components.iter().map(|(p, _)| jump_name(p)).collect()
    }

    /// `q_i = ∫ h·(a₁−a₂)·N_i dΓ` du côté considéré — le coefficient qui a
    /// bâti `∫h NᵀN` appliqué au saut, sans passer par une loi.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        jump: &SubElementField,
        lay: &[u32],
        material: &SubElementField,
        mat: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let n = lay.len();
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g);
            let w = geom.det_j_w(g);
            let saut = jump.row(geom.cell, g);
            let h = material.row(geom.cell, g);
            for v in 0..n {
                let hw = h[mat[v] as usize] * saut[lay[v] as usize] * w;
                if hw == 0.0 {
                    continue;
                }
                for i in 0..geom.n_nodes {
                    fe[i * n + v] += hw * shape[i];
                }
            }
        }
        Ok(())
    }

    fn label(&self) -> &'static str {
        "InterfaceTransfer"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let cells = self.side_a.read().cell_count();
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
    fn material_components(&self) -> Vec<String> {
        self.components
            .iter()
            .map(|(p, _)| coefficient_name(p))
            .collect()
    }

    /// A diagonal block: `+h ∫_Γ N_i N_j dΓ`.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material;
        exchange_matrix(geom, geom, mat, &lay.material, 1.0, ke)
    }

    /// An off-diagonal block: `−h ∫_Γ N_i^row N_j^col dΓ`. The sign lives here
    /// rather than in a factor, because the two kernels are already distinct.
    fn coupling_element(
        &self,
        _kind: MatrixKind,
        row_geoms: &[CellGeom],
        col_geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let mat = material;
        exchange_matrix(&row_geoms[0], &col_geoms[0], mat, &lay.material, -1.0, ke)
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
