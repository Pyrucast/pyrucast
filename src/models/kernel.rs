//! Per-physics parallel drivers — the bridge that lifts parallelism **above**
//! the constitutive/element kernels.
//!
//! A physics implementation never mentions rayon, the store, or a lock. It
//! supplies two kinds of **pure, sequential kernels**:
//!
//! - a **point kernel** at one Gauss point, driven either over an element-field
//!   input ([`element_pointwise`]) or a nodal-field input (`nodal_pointwise`).
//!   The constitutive law is one such kernel — read the deformation (+ `VAR0`)
//!   and material, write the flux/stress (+ `VAR1`) — but the same driver also
//!   powers point maps like the thermal strain;
//! - an **element kernel** ([`assemble_block`]): the local stiffness matrix of
//!   one cell from its geometry and material. It receives **one [`CellGeom`] per
//!   FE subspace** of the block — a single one for a plain volumetric physics, or
//!   several (sharing one mesh, differing by quadrature) for a multi-quadrature
//!   element such as a shear-deformable beam or a shell.
//!
//! These drivers own the fan-out (rayon), the zero-copy borrowing of store data
//! (read guards held across the parallel region, slices borrowed in place — no
//! per-cell snapshot copies), and the deterministic write-back / scatter. See
//! the book chapter *« Parallélisme »* and *« Ajouter une physique »*.
//!
//! # Integration measure
//!
//! Every kernel takes its quadrature weight from [`CellGeom::det_j_w`], which is
//! therefore the **single place** the geometric measure is decided. On an
//! [axisymmetric](crate::coords::Coords::axisymmetric) geometry it
//! returns `2πr |J| w` instead of `|J| w`, so stiffness, mass, conductivity,
//! distributed flux, volumes and internal forces all integrate over the full
//! ring with no per-physics change. What a physics *does* own is its operator:
//! only mechanics gains a term (the hoop strain `ε_θθ = u_r / r`), which is why
//! [`CellGeom::axisymmetric`] and [`CellGeom::radius`] are exposed.
//!
//! # Determinism
//!
//! [`element_pointwise`] writes each output slot exactly once
//! (`par_chunks_mut` over cells). [`assemble_block`] computes cell-local
//! matrices in parallel and scatters them into the COO **serially in cell
//! order**, so the assembled matrix is bit-for-bit identical to a sequential
//! run regardless of `RAYON_NUM_THREADS`.
//!
//! [`scatter_to_nodes`] (the shared nodal integrate-and-scatter driver behind the
//! internal forces, the weak divergence
//! [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence) and the
//! distributed flux load [`crate::ops::node_field::flux`](fn@crate::ops::node_field::flux))
//! instead builds each cell's local vector and scatters it in the **same parallel
//! pass**, by **cell colouring** (colours = node-disjoint cells): every node
//! accumulates in a fixed colour order, so the result is reproducible for any
//! `RAYON_NUM_THREADS` — though, summed in colour order rather than cell order, not
//! bit-for-bit with a sequential run (via [`crate::parallel::colored_scatter`]).

use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::{
    build_dn_dx, build_jacobian, jacobian_measure, SubFiniteElementSpace, MAX_JACOBIAN,
};
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::{NodeFieldView, SubNodeField};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::coloring;
use crate::parallel::*;
use nalgebra_sparse::CooMatrix;
use std::collections::HashMap;

/// Refuse a subspace that declares **no field basis** — the guard an operator
/// that *interpolates a field* owes its caller, stated once for the zone.
///
/// A `MODEL_EMBEDDED` subspace (a Timoshenko beam) leaves the interpolation to
/// its own formulation, so it has no shape values to lend and no curvature to
/// build; asking `CellGeom` for them would hand back the **geometric** basis,
/// which is a wrong answer rather than a refused one. Whether the question makes
/// sense is a property of the subspace, so it is settled here, before anything
/// is driven — the accessors themselves then only compute.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::{FiniteElementSpace, Interpolation};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let mut sm = SubMesh::new(coords, ElementType::SEG2);
/// # sm.add_cell(&[a.id(), b.id()])?;
/// # let maillage = Mesh::from_submesh(sm);
/// // Un espace de Lagrange interpole : la question a un sens.
/// let lagrange = FiniteElementSpace::lagrange1(&maillage)?;
/// assert!(kernel::require_field_basis(&lagrange.get(0)?, "shape values").is_ok());
/// // Un espace MODEL_EMBEDDED laisse l'interpolation à sa formulation.
/// let poutre = FiniteElementSpace::new(&maillage, Interpolation::ModelEmbedded)?;
/// let err = kernel::require_field_basis(&poutre.get(0)?, "shape values").unwrap_err();
/// assert!(format!("{err}").contains("MODEL_EMBEDDED"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn require_field_basis(fespace: &Handle<SubFiniteElementSpace>, what: &str) -> Result<()> {
    if fespace.read().interpolation().is_model_embedded() {
        return Err(PyrucastError::Message(format!(
            "this FE subspace is MODEL_EMBEDDED — it declares no field basis, so it has no \
             {what}. Its formulation owns the interpolation, so evaluating a field inside one \
             of its elements is that formulation's business."
        )));
    }
    Ok(())
}

/// Reference-element data of an FE subspace, snapshotted **once** before the
/// parallel loop (it would otherwise re-read the store on every call — through
/// `element_type()` — and serialise all threads on the global store lock).
/// Shape values, reference derivatives and weights at each Gauss point, plus the
/// fixed dimensions. Shared read-only across threads.
struct RefData {
    n_nodes: usize,
    n_gauss: usize,
    space_dim: usize,
    ref_dim: usize,
    /// Whether the geometry is a body of revolution — the `2πr` factor of
    /// [`CellGeom::det_j_w`].
    axisymmetric: bool,
    /// Reference coordinates `ξ_g` of each Gauss point.
    gauss_xi: Vec<Vec<f64>>,
    /// Shape values `N_i(ξ_g)` per Gauss point.
    n_ref: Vec<Vec<f64>>,
    /// Reference derivatives `∂N_i/∂ξ_k(ξ_g)` per Gauss point.
    dn_ref: Vec<Vec<f64>>,
    /// Gauss weights.
    weights: Vec<f64>,
    /// Number of **field** shape functions — `n_nodes` except under a C¹
    /// interpolation, where each node carries a value *and* a slope.
    shape_count: usize,
    /// Field `N_i(ξ_g)` and `∂²N_i/∂ξ²(ξ_g)` per Gauss point. Empty unless the
    /// field basis differs from the geometric one, so a Lagrange space
    /// snapshots exactly what it did before.
    field_n_ref: Vec<Vec<f64>>,
    field_d2n_ref: Vec<Vec<f64>>,
    /// Whether the space declares **no** field basis (the formulation owns it).
    /// Distinct from "the field tables are empty because they equal the
    /// geometric ones" — hence a flag rather than a length test.
    model_embedded: bool,
}

impl RefData {
    fn snapshot(fe: &SubFiniteElementSpace) -> Result<Self> {
        let n_gauss = fe.gauss_count();
        let mut gauss_xi = Vec::with_capacity(n_gauss);
        let mut n_ref = Vec::with_capacity(n_gauss);
        let mut dn_ref = Vec::with_capacity(n_gauss);
        let mut weights = Vec::with_capacity(n_gauss);
        for g in 0..n_gauss {
            gauss_xi.push(fe.gauss_xi(g)?.to_vec());
            n_ref.push(fe.n_at_g(g)?.to_vec());
            dn_ref.push(fe.dn_at_g(g)?.to_vec());
            weights.push(fe.gauss_weight(g)?);
        }
        // A C¹ space carries a second basis; a Lagrange one leaves these empty
        // and the accessors fall back to the geometric tables.
        let (mut field_n_ref, mut field_d2n_ref) = (Vec::new(), Vec::new());
        if fe.interpolation().is_hermite() {
            for g in 0..n_gauss {
                field_n_ref.push(fe.field_n_at_g(g)?.to_vec());
                field_d2n_ref.push(fe.field_d2n_at_g(g)?.to_vec());
            }
        }
        // Every kernel sizes its `dN/dx` scratch with `MAX_CELL_DOFS`. The bound is
        // a property of the *zone*, so it is proved here, once, and named: a wider
        // element would otherwise overflow a stack slice deep inside a kernel,
        // where the panic says nothing about which mesh caused it.
        let n_nodes = fe.nodes_per_cell()?;
        let space_dim = fe.space_dim();
        if n_nodes * space_dim > MAX_CELL_DOFS {
            return Err(PyrucastError::Message(format!(
                "this FE subspace has {n_nodes} nodes per cell in {space_dim}-D, so a cell needs \
                 {} values where a kernel's scratch holds {MAX_CELL_DOFS} — raise MAX_CELL_DOFS",
                n_nodes * space_dim
            )));
        }
        Ok(Self {
            n_nodes,
            n_gauss,
            space_dim,
            ref_dim: fe.ref_dim()?,
            axisymmetric: fe.is_axisymmetric(),
            gauss_xi,
            n_ref,
            dn_ref,
            weights,
            shape_count: fe.shape_count()?,
            field_n_ref,
            field_d2n_ref,
            model_embedded: fe.interpolation().is_model_embedded(),
        })
    }
}

/// Multiply the **slope** slots of a C¹ basis row by the Jacobian.
///
/// The basis alternates value, slope, value, slope — one pair per node — so the
/// odd indices are the ones whose reference degree of freedom is `∂w/∂ξ`.
fn scale_slope_slots_into(row: &[f64], j: f64, out: &mut [f64]) {
    for (i, (v, o)) in row.iter().zip(out.iter_mut()).enumerate() {
        *o = if i % 2 == 1 { v * j } else { *v };
    }
}

/// Geometry of one cell, computed **without touching the store**: coordinates
/// come from a held `Coords` guard, reference data is shared. Every accessor is
/// a pure local computation, so a kernel may call them while the driver runs the
/// cells in parallel — the kernel sees only this, never rayon or the store.
///
/// The cell's node coordinates are gathered **lazily** (only `dn_dx` / `det_j_w`
/// need them), so a point-local kernel that never asks for geometry (most
/// behaviour integrands) pays nothing. `CellGeom` is created and used within one
/// rayon task, so the interior-mutable cache is single-threaded.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::{assemble_block, CellGeom};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
/// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
/// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
/// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
/// # use pyrucast::containers::element_field::SubElementField;
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
/// #     assemble_block(
/// #         std::slice::from_ref(&zone), &support, &support,
/// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
/// #         &mat, None,
/// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
/// #     ).map(|_| ())
/// # };
/// // La géométrie d'une maille, sans jamais toucher au magasin : le noyau
/// // ne voit que ça — ni rayon, ni verrou.
/// noyau(&|geom| {
///     assert_eq!(geom.n_nodes, 3);
///     assert_eq!(geom.space_dim, 2);
///     assert!(!geom.axisymmetric);
///     Ok(())
/// })?;
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct CellGeom<'a> {
    rd: &'a RefData,
    coords: &'a Coords,
    conn: &'a [NodeId],
    cell_coords: std::cell::RefCell<Option<Vec<f64>>>,
    /// Index of this cell within the FE subspace.
    pub cell: usize,
    /// Nodes per cell.
    pub n_nodes: usize,
    /// Gauss points per cell.
    pub n_gauss: usize,
    /// Spatial dimension.
    pub space_dim: usize,
    /// Whether the geometry is the meridian plane of a body of revolution
    /// (`x = r`, `y = z`). [`det_j_w`](Self::det_j_w) already carries the `2πr`,
    /// so a kernel only reads this when its **operator** differs — mechanics,
    /// for the hoop strain `ε_θθ = u_r / r`.
    pub axisymmetric: bool,
}

impl<'a> CellGeom<'a> {
    fn new(rd: &'a RefData, coords: &'a Coords, conn: &'a [NodeId], cell: usize) -> Self {
        Self {
            rd,
            coords,
            conn,
            cell_coords: std::cell::RefCell::new(None),
            cell,
            n_nodes: rd.n_nodes,
            n_gauss: rd.n_gauss,
            space_dim: rd.space_dim,
            axisymmetric: rd.axisymmetric,
        }
    }

    /// Global node ids of this cell, in connectivity order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     assert_eq!(geom.node_ids().len(), 3); // dans l'ordre de connectivité
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn node_ids(&self) -> &'a [NodeId] {
        &self.conn[self.cell * self.n_nodes..(self.cell + 1) * self.n_nodes]
    }

    /// Coordinates of local node `local` (0-based within the cell). Read straight
    /// from the held `Coords` (no gather).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // Les coordonnées sont rassemblées **paresseusement** : un noyau
    ///     // purement local ne les paie pas.
    ///     assert_eq!(geom.node_coord(1), &[2.0, 0.0]);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn node_coord(&self, local: usize) -> &[f64] {
        self.coords.position_alive(self.node_ids()[local])
    }

    /// Fill the lazy `cell_coords` cache on first use (gather from the held
    /// `Coords`, no store access).
    fn ensure_cell_coords(&self) {
        let mut cc = self.cell_coords.borrow_mut();
        if cc.is_none() {
            let mut v = Vec::with_capacity(self.n_nodes * self.space_dim);
            for &id in self.node_ids() {
                // Live-ness was settled for the whole connectivity before the
                // parallel region (`Coords::ensure_all_alive`), so this reads
                // without asking again — once per node of every cell of every
                // call is a great many times to re-prove one fact.
                v.extend_from_slice(self.coords.position_alive(id));
            }
            *cc = Some(v);
        }
    }

    /// `∂N_i/∂x_a` at Gauss point `g`, flat layout `[i * space_dim + a]`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // La matrice B, écrite dans un tampon de l'appelant : au point de
    ///     // Gauss, une allocation coûte plus cher que l'algèbre qu'elle porte.
    ///     let mut b = [0.0_f64; 6];
    ///     geom.dn_dx(0, &mut b)?;
    ///     assert!((b[0] + b[2] + b[4]).abs() < 1e-12);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn dn_dx(&self, g: usize, out: &mut [f64]) -> Result<()> {
        self.ensure_cell_coords();
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let dn = &self.rd.dn_ref[g];
        let mut jac = [0.0_f64; MAX_JACOBIAN];
        build_jacobian(
            cc,
            dn,
            self.space_dim,
            self.rd.ref_dim,
            self.n_nodes,
            &mut jac,
        );
        build_dn_dx(&jac, dn, self.space_dim, self.rd.ref_dim, self.n_nodes, out)
    }

    /// Shape-function values `N_i(ξ_g)` at Gauss point `g`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     assert!((geom.n_at_g(0).iter().sum::<f64>() - 1.0).abs() < 1e-12);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn n_at_g(&self, g: usize) -> &[f64] {
        &self.rd.n_ref[g]
    }

    /// Reference coordinates `ξ_g` of Gauss point `g`, of length `ref_dim`.
    ///
    /// What a formulation carrying **its own** basis needs: the quadrature is
    /// the subspace's, but the functions to evaluate at it are the element's own
    /// — a discrete-Kirchhoff shell interpolates its rotations quadratically over
    /// a mesh the space declares linear.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     assert_eq!(geom.gauss_xi(0).len(), 2); // dans l'élément de référence
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gauss_xi(&self, g: usize) -> &[f64] {
        &self.rd.gauss_xi[g]
    }

    /// Number of **field** shape functions — `n_nodes` for a Lagrange space,
    /// twice that under a C¹ interpolation.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // Le nombre de formes du **champ** : il diffère de `n_nodes` dès que
    ///     // l'interpolation est d'Hermite.
    ///     assert_eq!(geom.shape_count(), 3);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn shape_count(&self) -> usize {
        self.rd.shape_count
    }

    /// Field shape values at Gauss point `g`, already scaled to act on
    /// **physical** degrees of freedom.
    ///
    /// Under a C¹ interpolation the odd slots are slope functions, whose
    /// reference degree of freedom is `∂w/∂ξ` while the physical one is
    /// `∂w/∂x`: they carry an extra factor `J`. For a Lagrange space this is
    /// [`n_at_g`](Self::n_at_g) unchanged.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // Le tampon ne sert qu'aux bases C¹ : ici la méthode prête
    ///     // directement la ligne déjà en mémoire.
    ///     let mut buf = [0.0_f64; 8];
    ///     assert_eq!(geom.field_n_at_g(0, &mut buf).len(), geom.shape_count());
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_n_at_g<'s>(&'s self, g: usize, scratch: &'s mut [f64]) -> &'s [f64] {
        // Whether the subspace *has* a field basis is a fact of the zone, and it
        // is the **operator** that must have settled it: see
        // [`require_field_basis`], which the interpolating operators call once
        // before driving anything.
        debug_assert!(
            !self.rd.model_embedded,
            "CellGeom::field_n_at_g on a MODEL_EMBEDDED subspace — call \
             kernel::require_field_basis once for the zone"
        );
        // The common case: the field basis **is** the geometric one, already in
        // reference data. Lending it beats copying it at every Gauss point.
        if self.rd.field_n_ref.is_empty() {
            return &self.rd.n_ref[g];
        }
        let j = self.segment_jacobian();
        let row = &self.rd.field_n_ref[g];
        scale_slope_slots_into(row, j, &mut scratch[..row.len()]);
        &scratch[..row.len()]
    }

    /// Field second derivatives `∂²N_i/∂x²` at Gauss point `g`, acting on
    /// physical degrees of freedom — the **curvature operator** of a C¹ element.
    ///
    /// Defined for a straight 1-D reference element, which is where a C¹ basis
    /// exists today. The chain rule is then simply `∂²/∂x² = J⁻² ∂²/∂ξ²`: the
    /// `∂J/∂ξ` term that would appear on a curved element vanishes, because a
    /// `SEG2` has a constant Jacobian by construction.
    ///
    /// # Errors
    ///
    /// The space is not C¹, or its reference element is not a segment.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::{FiniteElementSpace, Interpolation};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// // Les dérivées secondes **physiques**, ce qu'exige la courbure d'une
    /// // poutre. Réservées aux espaces C¹ sur un segment — `Bernoulli::new`
    /// // refuse tout autre espace, et c'est là que la question se tranche,
    /// // une fois, plutôt qu'à chaque point de Gauss.
    /// let hermite = FiniteElementSpace::new(&maillage, Interpolation::Hermite3)?;
    /// kernel::reduce_cells(&hermite.get(0)?, |geom| {
    ///     let mut buf = [0.0_f64; 8];
    ///     // Quatre fonctions de forme : deux valeurs, deux pentes.
    ///     assert_eq!(geom.field_d2n_dx2(0, &mut buf), 4);
    ///     Ok(0.0)
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_d2n_dx2(&self, g: usize, out: &mut [f64]) -> usize {
        // A curvature operator exists only under a C¹ interpolation, and that is
        // settled at construction: `Bernoulli::new` refuses a non-Hermite
        // subspace, which is the only physics asking for one.
        debug_assert!(
            !self.rd.field_d2n_ref.is_empty(),
            "CellGeom::field_d2n_dx2 on a subspace with no C¹ field basis"
        );
        let j = self.segment_jacobian();
        // `J` scales the slope columns to physical DOFs, `J⁻²` maps the
        // reference second derivative to a physical one.
        let row = &self.rd.field_d2n_ref[g];
        scale_slope_slots_into(row, j, &mut out[..row.len()]);
        for v in &mut out[..row.len()] {
            *v /= j * j;
        }
        row.len()
    }

    /// `J = ∂x/∂ξ` of a straight segment: **signed** in a 1-D space (where the
    /// slope degree of freedom is taken along the global axis, so a cell whose
    /// nodes run backwards must flip it), and the arc length `L/2` when the
    /// segment is embedded in a plane or in space — there the consumer has
    /// already rotated into a local axis running from node 0 to node 1.
    fn segment_jacobian(&self) -> f64 {
        // Une base C¹ vit sur un élément de référence 1-D. Ce n'est pas une
        // donnée à vérifier ici : `Bernoulli::new` refuse un espace non-Hermite,
        // et `RefData::snapshot` n'a de tables C¹ que pour un tel espace.
        debug_assert_eq!(
            self.rd.ref_dim, 1,
            "a C¹ basis needs a 1-D reference element"
        );
        // Deux emprunts immuables coexistent : la copie ne servait à rien, et
        // elle coûtait deux allocations **par point de Gauss** sur une poutre C¹.
        let (a, b) = (self.node_coord(0), self.node_coord(1));
        if self.space_dim == 1 {
            return (b[0] - a[0]) / 2.0;
        }
        let l2: f64 = (0..self.space_dim).map(|i| (b[i] - a[i]).powi(2)).sum();
        l2.sqrt() / 2.0
    }

    /// Physical coordinates of Gauss point `g`, `x_a = Σ_i N_i(ξ_g) · x_{i,a}`
    /// (length `space_dim`). Uses the same lazy coordinate gather as
    /// [`dn_dx`](Self::dn_dx), so a kernel that never asks pays nothing.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // Le point de Gauss en coordonnées **physiques** : Σ N_i x_i.
    ///     // La règle du TRI3 place ses points aux **milieux des arêtes** :
    ///     // le premier est donc sur le bord, en (1, 0).
    ///     let mut x = [0.0_f64; 2];
    ///     geom.x_at_g(0, &mut x);
    ///     assert_eq!(x, [1.0, 0.0]);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn x_at_g(&self, g: usize, out: &mut [f64]) {
        self.ensure_cell_coords();
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let n = &self.rd.n_ref[g];
        let d = self.space_dim;
        out[..d].fill(0.0);
        for i in 0..self.n_nodes {
            for (a, xa) in out[..d].iter_mut().enumerate() {
                *xa += n[i] * cc[i * d + a];
            }
        }
    }

    /// Radius `r` at Gauss point `g` — the first physical coordinate, on an
    /// **axisymmetric** geometry only (errors otherwise, since `x` is then just
    /// an abscissa and dividing by it would be meaningless).
    ///
    /// Gauss points are interior to the cell, so `r > 0` even for a cell touching
    /// the axis: the `N_i / r` of the hoop strain stays finite — the standard
    /// treatment of the axis in an axisymmetric formulation.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::assemble_block;
    /// // Le rayon n'a de sens que sur un repère de **révolution**.
    /// let coords = Handle::new(Coords::axisymmetric()?);
    /// # let n: Vec<Node> = [[1.0, 0.0], [3.0, 0.0], [1.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let zone = fes.get(0)?;
    /// # let support = zone.read().submesh().read().to_poi1()?;
    /// # let mat_bidon = Handle::new(
    /// #     pyrucast::containers::element_field::SubElementField::from_uniform_per_component(
    /// #         zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// assemble_block(
    ///     std::slice::from_ref(&zone), &support, &support,
    ///     vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    ///     &mat_bidon, None,
    ///     |geoms, _m, _s, _ke| {
    ///         let geom = &geoms[0];
    ///         assert!(geom.axisymmetric);
    ///         // `x = r` : le rayon est l'abscisse du point de Gauss.
    ///         let mut x = [0.0_f64; 2];
    ///         geom.x_at_g(0, &mut x);
    ///         assert_eq!(geom.radius(0), x[0]);
    ///         Ok(())
    ///     },
    /// )?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn radius(&self, g: usize) -> f64 {
        debug_assert!(
            self.axisymmetric,
            "CellGeom::radius on a Cartesian geometry: `axisymmetric` says whether \
             asking makes sense, and it is a fact of the zone"
        );
        self.ensure_cell_coords();
        let cc = self.cell_coords.borrow();
        Self::radius_from(
            cc.as_ref().unwrap(),
            &self.rd.n_ref[g],
            self.n_nodes,
            self.space_dim,
        )
    }

    /// `r = Σ_i N_i x_i` from already-borrowed cell coordinates — the scalar
    /// core of [`radius`](Self::radius), split out so the hot
    /// [`det_j_w`](Self::det_j_w) computes it under its **existing** borrow
    /// instead of allocating an `x_at_g` vector per Gauss point.
    fn radius_from(cell_coords: &[f64], n: &[f64], n_nodes: usize, space_dim: usize) -> f64 {
        let mut r = 0.0;
        for i in 0..n_nodes {
            r += n[i] * cell_coords[i * space_dim];
        }
        r
    }

    /// The cell's **tangent vectors** at Gauss point `g` — the columns of the
    /// Jacobian, `a_k = ∂x/∂ξ_k`, one per reference direction (`ref_dim` vectors
    /// of length `space_dim`).
    ///
    /// They are the raw material of surface kinematics: the deformed tangents
    /// are `ā_k = a_k + ∂u/∂ξ_k`, and everything about how a surface stretches
    /// and turns follows from them. A physics working on a **manifold** wants
    /// these rather than the tangential gradient `∇_s u`, because `∇_s u` has no
    /// component along the normal and so cannot be completed into a deformation
    /// gradient: `I + ∇_s u` goes singular under a quarter-turn, while the
    /// tangents never do.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // Une tangente par direction de l'élément de référence : deux pour
    ///     // une surface, une pour une ligne. Elles arrivent à plat,
    ///     // `out[k * space_dim + a]`.
    ///     let mut t = [0.0_f64; 4];
    ///     geom.tangents(0, &mut t);
    ///     assert!(t[..4].iter().any(|v| *v != 0.0));
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn tangents(&self, g: usize, out: &mut [f64]) {
        self.ensure_cell_coords();
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let (d, r) = (self.space_dim, self.rd.ref_dim);
        let mut jac = [0.0_f64; MAX_JACOBIAN];
        build_jacobian(cc, &self.rd.dn_ref[g], d, r, self.n_nodes, &mut jac);
        // `build_jacobian` lays the tangents out as `jac[a * r + k]`: column `k`
        // is the tangent along reference direction `k`. `out` receives them
        // tangent-major, `out[k * d + a]`.
        for k in 0..r {
            for a in 0..d {
                out[k * d + a] = jac[a * r + k];
            }
        }
    }

    /// The (unnormalised) normal of a boundary cell from its tangents — the
    /// cross product in 3-D, the tangent turned by −90° in 2-D. Its **norm** is
    /// the surface measure per unit reference parameter, which is what makes it
    /// usable for an area ratio as well as a direction.
    ///
    /// Shared by [`normal`](Self::normal) and by the surface kinematics of a
    /// follower load, which needs the same construction on the *deformed*
    /// tangents.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// // Une fonction libre : la normale déduite des tangentes — le produit
    /// // vectoriel en 3-D, la rotation d'un quart de tour en 2-D. Elle porte
    /// // la **norme** des tangentes, elle n'est pas normalisée ;
    /// // [`normal`](Self::normal) s'en charge.
    /// let mut nu = [0.0_f64; 2];
    /// CellGeom::normal_from_tangents(&[2.0, 0.0], 1, 2, &mut nu);
    /// assert_eq!(nu, [0.0, -2.0]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn normal_from_tangents(
        tangents: &[f64],
        n_tangents: usize,
        space_dim: usize,
        out: &mut [f64],
    ) {
        let t = |k: usize, a: usize| tangents[k * space_dim + a];
        // Une normale se prend sur un bord : une tangente en 2-D, deux en 3-D.
        // `CellGeom::normal` l'établit pour la zone avant d'appeler.
        debug_assert!(
            n_tangents == 1 || n_tangents == 2,
            "a normal needs 1 tangent (2-D) or 2 (3-D)"
        );
        if n_tangents == 1 {
            out[0] = t(0, 1);
            out[1] = -t(0, 0);
        } else {
            out[0] = t(0, 1) * t(1, 2) - t(0, 2) * t(1, 1);
            out[1] = t(0, 2) * t(1, 0) - t(0, 0) * t(1, 2);
            out[2] = t(0, 0) * t(1, 1) - t(0, 1) * t(1, 0);
        }
    }

    /// Unit **normal** of a boundary cell at Gauss point `g`, in the reference
    /// configuration.
    ///
    /// Defined only on a **manifold** (`ref_dim == space_dim − 1`): a boundary
    /// edge in 2-D, a boundary face in 3-D. A cell that fills its space has no
    /// normal, and asking for one is a modelling error, so it errors rather than
    /// returning a convention.
    ///
    /// The direction follows the cell's own **winding** — it is the mesh, not
    /// this accessor, that decides which side is « outside ». That is why the
    /// physics needing a normal (a pressure, a signed flux) are the ones that
    /// must care about their boundary mesh's orientation, while those that do
    /// not ([`boundary_transfer`](crate::models::boundary_transfer),
    /// [`radiation`](crate::models::radiation)) never call this: their direction
    /// is already consumed in writing `q·n`, and
    /// [`det_j_w`](Self::det_j_w) returns an orientation-invariant magnitude.
    ///
    /// - 2-D: the edge tangent `t` rotated by −90°, `n = (t_y, −t_x)/|t|`.
    /// - 3-D: `n = (a₁ × a₂)/|a₁ × a₂|`, the two Jacobian columns.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{self, assemble_block};
    /// # let coords = Handle::new(Coords::new(3).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// // La normale **unitaire** — celle que `normal_from_tangents` rend
    /// // brute, divisée par sa norme. Elle n'a de sens que sur une facette :
    /// // une dimension de référence de moins que l'espace.
    /// kernel::reduce_cells(&zone, |geom| {
    ///     let mut nu = [0.0_f64; 3];
    ///     geom.normal(0, &mut nu)?;
    ///     assert!((nu.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
    ///     assert!((nu[2].abs() - 1.0).abs() < 1e-12); // le triangle est dans z = 0
    ///     Ok(0.0)
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn normal(&self, g: usize, out: &mut [f64]) -> Result<()> {
        let (d, r) = (self.space_dim, self.rd.ref_dim);
        // Cette fonction rend un `Result` de toute façon — la normale nulle est
        // un fait du point — donc la précondition de forme, elle, y reste : elle
        // n'aurait nulle part ailleurs où vivre, et elle ne coûte rien ici.
        if d < 2 || r + 1 != d {
            return Err(PyrucastError::Message(format!(
                "CellGeom::normal: a {r}-D cell in a {d}-D space has no normal — a normal is \
                 defined on a boundary (a {}-D cell here)",
                d.saturating_sub(1)
            )));
        }
        let mut tan = [0.0_f64; MAX_JACOBIAN];
        self.tangents(g, &mut tan);
        Self::normal_from_tangents(&tan, r, d, out);
        let norm = out[..d].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "CellGeom::normal: cell {} is degenerate at Gauss point {g} (null normal)",
                self.cell
            )));
        }
        for v in &mut out[..d] {
            *v /= norm;
        }
        Ok(())
    }

    /// `|J|_g · w_g` — the integration weight of Gauss point `g`.
    ///
    /// On an **axisymmetric** geometry this is `2πr_g · |J|_g · w_g`: the
    /// circumferential measure is applied here, once, so *every* integral built
    /// on `CellGeom` — stiffness, mass, conductivity, distributed flux, volumes,
    /// internal forces, on the body and on its boundary alike — integrates over
    /// the full ring without its kernel knowing.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel::{assemble_block, CellGeom};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # // `CellGeom` n'existe qu'à l'intérieur d'un pilote : on en obtient un en
    /// # // passant un noyau d'élément à `assemble_block`, exactement comme le fait
    /// # // une physique. Ce noyau-ci ne lit aucun matériau, mais l'assembleur en
    /// # // veut un : on en donne un qui ne sert à rien plutôt qu'une `Option`.
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
    /// # let noyau = |verifier: &(dyn Fn(&CellGeom) -> pyrucast::Result<()> + Sync)| {
    /// #     assemble_block(
    /// #         std::slice::from_ref(&zone), &support, &support,
    /// #         vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
    /// #         &mat, None,
    /// #         |geoms, _m, _s, _ke| verifier(&geoms[0]),
    /// #     ).map(|_| ())
    /// # };
    /// noyau(&|geom| {
    ///     // **L'unique endroit** où se décide la mesure d'intégration : ici
    ///     // |J|·w, et 2πr·|J|·w en révolution. C'est ce qui permet à toute
    ///     // physique d'intégrer sur l'anneau complet sans rien changer.
    ///     let aire: f64 = (0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum();
    ///     assert!((aire - 2.0).abs() < 1e-12); // l'aire du triangle
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn det_j_w(&self, g: usize) -> f64 {
        self.ensure_cell_coords();
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let dn = &self.rd.dn_ref[g];
        let mut jac = [0.0_f64; MAX_JACOBIAN];
        build_jacobian(
            cc,
            dn,
            self.space_dim,
            self.rd.ref_dim,
            self.n_nodes,
            &mut jac,
        );
        let w = jacobian_measure(&jac, self.space_dim, self.rd.ref_dim) * self.rd.weights[g];
        // One predictable branch on a per-subspace constant; the Cartesian path
        // returns here having paid nothing else. The revolved path reuses the
        // borrow above, so it allocates nothing either.
        if !self.axisymmetric {
            return w;
        }
        let r = Self::radius_from(cc, &self.rd.n_ref[g], self.n_nodes, self.space_dim);
        w * std::f64::consts::TAU * r
    }
}

/// Integrate a point-local constitutive law over `fespace`, in parallel.
///
/// `point(geom, g, input, prev, material, out)` is a pure sequential kernel: for
/// the cell `geom.cell` at Gauss point `g` it receives **the row of that point**
/// in each input field — a borrowed slice of the field's own buffer, never a
/// copy — and writes the `out_components.len()` output values into `out`.
///
/// Rows, not fields, because a kernel that receives a field ends up searching it
/// by name at every point. The row is contiguous (the buffer is
/// cell-major/Gauss-major/component-minor), so slicing it is index arithmetic;
/// the caller resolves *which* index means what once per zone
/// ([`crate::models::ZoneLayout`]).
///
/// The three input fields are checked **here**, once, to span the same cells and
/// Gauss points as `fespace` — a field built by hand can disagree, and the
/// parallel loop below indexes without asking.
///
/// A producer with no previous state — a *geometric* one, such as
/// [`crate::ops::element_field::thermal_strain`](fn@crate::ops::element_field::thermal_strain)
/// — passes any field it has at hand: the row is sliced and never indexed. That
/// is cheaper to read than an `Option` whose `None` meant « this argument is
/// unused », which is not what an `Option` is for.
///
/// The **element-field-input** driver, mirrored by `nodal_pointwise` (which
/// reads a nodal field instead). Backs the constitutive integration
/// [`crate::ops::element_field::behavior::integrate`] and the thermal strain
/// [`crate::ops::element_field::thermal_strain`](fn@crate::ops::element_field::thermal_strain).
///
/// Returns the material-state field (flux/stress + `VAR1`) on `fespace`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat_bidon = Handle::new(
/// #     pyrucast::containers::element_field::SubElementField::from_uniform_per_component(
/// #         zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// # let mut entree = SubElementField::new(zone.clone(), vec!["eps".into()])?;
/// # entree.set_uniform("eps", 2.0)?;
/// # let entree = Handle::new(entree);
/// // Un noyau **au point de Gauss** : la loi de comportement en est un —
/// // lire la déformation et le matériau, écrire la contrainte. Ce noyau-ci
/// // ne lit ni état antérieur ni matériau : on lui en passe qui ne serviront
/// // pas, plutôt que des `Option` à déballer.
/// let sortie = kernel::element_pointwise(
///     &zone, &entree, &entree, &mat_bidon, vec!["sig".into()],
///     |_geom, _g, ligne, _prev, _mat, slot| {
///         slot[0] = 3.0 * ligne[0];
///         Ok(())
///     },
/// )?;
/// assert_eq!(sortie.value(0, 0, "sig")?, 6.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_pointwise(
    fespace: &Handle<SubFiniteElementSpace>,
    input: &Handle<SubElementField>,
    prev: &Handle<SubElementField>,
    material: &Handle<SubElementField>,
    out_components: Vec<String>,
    point: impl Fn(&CellGeom, usize, &[f64], &[f64], &[f64], &mut [f64]) -> Result<()> + Sync,
) -> Result<SubElementField> {
    let out_stride = out_components.len();
    let mut out = SubElementField::new(fespace.clone(), out_components)?;

    // Guards held for the whole parallel region — slices borrowed, not copied.
    // Reference data snapshotted once (no per-cell store reads inside the loop).
    let fe = fespace.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();
    let fin = input.read();
    let prev_guard = prev.read();
    let mat_guard = material.read();

    let rd = RefData::snapshot(&fe)?;
    let n_gauss = rd.n_gauss;
    let n_cells = fe.cell_count();
    let conn: &[NodeId] = sm.connectivity();
    // Every node of the connectivity, checked live **once** — the cell
    // geometry then reads coordinates without asking again.
    coords.ensure_all_alive(conn)?;
    let rd_ref: &RefData = &rd;
    let coords_ref: &Coords = &coords;

    // The shape of every input, settled **once**. Below this the loop slices
    // rows by arithmetic alone: a field that spans other cells or another
    // quadrature would silently read the wrong point, so it is refused here,
    // where the message can say which field and by how much.
    fn rows<'f>(
        f: &'f SubElementField,
        what: &str,
        n_cells: usize,
        n_gauss: usize,
    ) -> Result<(&'f [f64], usize)> {
        if f.cell_count() != n_cells || f.gauss_count() != n_gauss {
            return Err(PyrucastError::Message(format!(
                "{what}: {} cells × {} Gauss points, but this FE subspace has {} × {}",
                f.cell_count(),
                f.gauss_count(),
                n_cells,
                n_gauss
            )));
        }
        Ok((f.values(), f.component_count()))
    }
    let (in_vals, in_stride) = rows(&fin, "deformation", n_cells, n_gauss)?;
    let (prev_vals, prev_stride) = rows(&prev_guard, "previous state", n_cells, n_gauss)?;
    let (mat_vals, mat_stride) = rows(&mat_guard, "material", n_cells, n_gauss)?;

    // The row of `(cell, g)`: contiguous, so a start offset and a stride. A
    // stride of zero (an absent input) lands on the empty range of an empty
    // buffer, so the loop below needs no test for it.
    fn row(vals: &[f64], stride: usize, n_gauss: usize, cell: usize, g: usize) -> &[f64] {
        let start = (cell * n_gauss + g) * stride;
        &vals[start..start + stride]
    }

    out.values_mut()
        .par_chunks_mut(n_gauss * out_stride)
        .with_min_len((MIN_PARALLEL_LEN / n_gauss.max(1)).max(1))
        .enumerate()
        .try_for_each(|(cell, ochunk)| -> Result<()> {
            let geom = CellGeom::new(rd_ref, coords_ref, conn, cell);
            for g in 0..n_gauss {
                let slot = &mut ochunk[g * out_stride..(g + 1) * out_stride];
                point(
                    &geom,
                    g,
                    row(in_vals, in_stride, n_gauss, cell, g),
                    row(prev_vals, prev_stride, n_gauss, cell, g),
                    row(mat_vals, mat_stride, n_gauss, cell, g),
                    slot,
                )?;
            }
            Ok(())
        })?;
    Ok(out)
}

/// Produce a per-element (Gauss-point) field from a **nodal** field, in parallel.
///
/// The nodal counterpart of [`element_pointwise`]: where that reads an
/// element-field `input`, this reads a nodal-field view `field`. `point(geom,
/// field, g, out)` is a pure sequential kernel: for the cell `geom.cell` at Gauss
/// point `g`, it reads nodal values (`field.value(id, comp)` for
/// `id in geom.node_ids()`) and writes the `out_components.len()` output values
/// into `out`. It uses `geom.n_at_g(g)` to interpolate values and/or
/// `geom.dn_dx(g)` to differentiate them.
///
/// Same guarantees as [`element_pointwise`]: reference data snapshotted once,
/// guards held for the whole region (slices borrowed, not copied), each output
/// slot written exactly once (`par_chunks_mut` over cells) ⇒ **bit-for-bit
/// deterministic**. Backs the geometric field producers
/// [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient),
/// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation) and
/// [`crate::ops::element_field::interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss).
pub(crate) fn nodal_pointwise(
    fespace: &Handle<SubFiniteElementSpace>,
    field: &NodeFieldView,
    reads: &[String],
    out_components: Vec<String>,
    point: impl Fn(&CellGeom, usize, &[f64], &mut [f64]) -> Result<()> + Sync,
) -> Result<SubElementField> {
    let out_stride = out_components.len();
    let mut out = SubElementField::new(fespace.clone(), out_components)?;

    // Guards held for the whole parallel region — slices borrowed, not copied.
    let fe = fespace.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();

    let rd = RefData::snapshot(&fe)?;
    let n_gauss = rd.n_gauss;
    let conn: &[NodeId] = sm.connectivity();
    // Every node of the connectivity, checked live **once** — the cell
    // geometry then reads coordinates without asking again.
    coords.ensure_all_alive(conn)?;
    let rd_ref: &RefData = &rd;
    let coords_ref: &Coords = &coords;

    // The components each zone carries, resolved **once** — the gather below
    // compares no name.
    let reads_idx = field.resolve_reads(reads);

    // The gather buffer is a fixed-size array, so a cell costs no allocation.
    // Its width is `nodes × components read`, which `RefData::snapshot` cannot
    // know — it only proved `nodes × space_dim` — so the bound is checked here
    // too, once per zone, against the read list this call was given.
    let n_dofs = rd.n_nodes * reads.len();
    if n_dofs > MAX_CELL_DOFS {
        return Err(PyrucastError::Message(format!(
            "nodal_pointwise: {} nodes × {} components exceeds the {MAX_CELL_DOFS} \
             values a cell's gather buffer holds",
            rd.n_nodes,
            reads.len()
        )));
    }

    out.values_mut()
        .par_chunks_mut(n_gauss * out_stride)
        .with_min_len((MIN_PARALLEL_LEN / n_gauss.max(1)).max(1))
        .enumerate()
        .try_for_each(|(cell, ochunk)| -> Result<()> {
            let geom = CellGeom::new(rd_ref, coords_ref, conn, cell);
            // The cell's nodal values, read **once** — they do not change from
            // one Gauss point of the cell to the next.
            let mut dofs = [0.0_f64; MAX_CELL_DOFS];
            field.gather_cell(geom.node_ids(), &reads_idx, &mut dofs[..n_dofs]);
            for g in 0..n_gauss {
                let slot = &mut ochunk[g * out_stride..(g + 1) * out_stride];
                point(&geom, g, &dofs[..n_dofs], slot)?;
            }
            Ok(())
        })?;
    Ok(out)
}

/// The most values a per-cell buffer holds: twenty-seven nodes (HEX27) times
/// three components is 81 — the widest gather, and the widest `dN/dx` a cell can
/// have. Rounded up to leave room for the next element.
///
/// It sizes **stack** buffers, so a Gauss point costs no allocation. The bound is
/// enforced once per zone by [`RefData::snapshot`], where a wider element is
/// refused by name instead of overflowing a slice inside a kernel.
pub(crate) const MAX_CELL_DOFS: usize = 96;

/// Assemble one stiffness block by a per-cell element-matrix kernel, in
/// parallel.
///
/// `element(geoms, material, ke)` is a pure sequential kernel: it fills `ke` —
/// the cell's local dense matrix, row-major, **node-major / variable-minor**:
///   row `r = li * n_dual + di`, col `c = lj * n_primal + pj`
/// (with `li/lj` local node indices, `di/pj` indices into `dual_vars` /
/// `primal_vars`). `geoms` holds one [`CellGeom`] per FE subspace of `fespaces`
/// (same order); a single-space physics reads `geoms[0]`, a multi-quadrature one
/// reads each. The driver scatters `ke` into the block's COO serially in cell
/// order. `material` is `Some` iff the physics supplied one.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat_bidon = Handle::new(
/// #     pyrucast::containers::element_field::SubElementField::from_uniform_per_component(
/// #         zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // Le pilote appelle un noyau **pur et séquentiel** par maille ; la
/// // parallélisation, l'emprunt zéro-copie et le rangement en COO sont à
/// // lui. Ici, une matrice identité locale.
/// let bloc = kernel::assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
///     &mat_bidon, None,
///     |geoms, _m, _s, ke| {
///         let npc = geoms[0].n_nodes;
///         for i in 0..npc {
///             ke[i * npc + i] = 1.0;
///         }
///         Ok(())
///     },
/// )?;
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (3, 3));
/// assert_eq!(bloc.get(n[0].id(), "q", n[0].id(), "T"), 1.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn assemble_block(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    dual_vars: Vec<String>,
    primal_vars: Vec<String>,
    ordering: DofOrdering,
    symmetric: bool,
    material: &Handle<SubElementField>,
    state: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], &SubElementField, Option<&SubElementField>, &mut [f64]) -> Result<()>
        + Sync,
) -> Result<SubMatrix> {
    let (nrows, ncols, trips) = element_block_triplets(
        fespaces,
        row_support,
        col_support,
        dual_vars.len(),
        primal_vars.len(),
        ordering,
        material,
        state,
        element,
    )?;
    let mut rows = Vec::with_capacity(trips.len());
    let mut cols = Vec::with_capacity(trips.len());
    let mut vals = Vec::with_capacity(trips.len());
    for (r, c, v) in trips {
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }
    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals)
        .map_err(|e| PyrucastError::Message(format!("assemble_block: invalid COO: {e}")))?;
    SubMatrix::from_coo(
        row_support.clone(),
        col_support.clone(),
        dual_vars,
        primal_vars,
        ordering,
        symmetric,
        coo,
    )
}

/// `(nrows, ncols, local (row, col, value) triplets)` — the shape returned by
/// [`element_block_triplets`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // Ce noyau ne lit aucun matériau ; l'assembleur en veut un, on lui en
/// # // donne un qui ne sert à rien plutôt qu'une `Option` à déballer.
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // Ce que rend `element_block_triplets` : la taille du bloc et ses
/// // triplets, en numérotation **locale** au bloc.
/// let (nr, nc, trips): kernel::BlockTriplets = kernel::element_block_triplets(
///     std::slice::from_ref(&zone), &support, &support, 1, 1,
///     DofOrdering::NodesThenVars, &mat, None,
///     |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0); Ok(()) },
/// )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert!(!trips.is_empty());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type BlockTriplets = (usize, usize, Vec<(usize, usize, f64)>);

/// Compute one stiffness block's entries as **local** `(row, col, value)`
/// triplets (the block's own numbering, `(node_local, var)` via `ordering`),
/// driving the per-cell `element` kernel in parallel. This is the shared core of
/// both [`assemble_block`] (which wraps these in a [`SubMatrix`]) and the global
/// computed-block assembler (which remaps them to global indices).
///
/// Cells are computed in parallel and their triplets concatenated **in cell
/// order** (within a cell: `li, di, lj, pj`), so the stream is identical to a
/// sequential run — the assembled values are reproducible. Returns
/// `(nrows, ncols, triplets)`.
#[allow(clippy::too_many_arguments)]
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // Ce noyau ne lit aucun matériau ; l'assembleur en veut un, on lui en
/// # // donne un qui ne sert à rien plutôt qu'une `Option` à déballer.
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // Ce que rend `element_block_triplets` : la taille du bloc et ses
/// // triplets, en numérotation **locale** au bloc.
/// let (nr, nc, trips): kernel::BlockTriplets = kernel::element_block_triplets(
///     std::slice::from_ref(&zone), &support, &support, 1, 1,
///     DofOrdering::NodesThenVars, &mat, None,
///     |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0); Ok(()) },
/// )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert!(!trips.is_empty());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_block_triplets(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
    material: &Handle<SubElementField>,
    state: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], &SubElementField, Option<&SubElementField>, &mut [f64]) -> Result<()>
        + Sync,
) -> Result<BlockTriplets> {
    let (nrows, ncols, per_cell) = element_block_triplets_per_cell(
        fespaces,
        row_support,
        col_support,
        n_dual,
        n_primal,
        ordering,
        material,
        state,
        element,
    )?;
    let total: usize = per_cell.iter().map(|v| v.len()).sum();
    let mut trips = Vec::with_capacity(total);
    for cell_trips in per_cell {
        trips.extend(cell_trips);
    }
    Ok((nrows, ncols, trips))
}

/// `(nrows, ncols, per-cell local (row, col, value) triplets)` — the shape
/// returned by [`element_block_triplets_per_cell`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // Ce noyau ne lit aucun matériau ; l'assembleur en veut un, on lui en
/// # // donne un qui ne sert à rien plutôt qu'une `Option` à déballer.
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // La même chose, **groupée par maille** : ce que consomme l'assembleur
/// // global pour verser droit dans le CSR sans matérialiser de valeurs.
/// let (nr, nc, par_maille): kernel::BlockTripletsPerCell =
///     kernel::element_block_triplets_per_cell(
///         std::slice::from_ref(&zone), &support, &support, 1, 1,
///         DofOrdering::NodesThenVars, &mat, None,
///         |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0); Ok(()) },
///     )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(par_maille.len(), 1); // une maille
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type BlockTripletsPerCell = (usize, usize, Vec<Vec<(usize, usize, f64)>>);

/// Same as [`element_block_triplets`] but keeps each cell's triplets in its own
/// `Vec` **instead of concatenating** them. Cells are still computed in parallel
/// and their triplets ordered `li, di, lj, pj` within a cell, so the per-cell
/// lists are reproducible. The grouping lets a colour-driven scatter process one
/// colour's cells (which touch disjoint DOFs) in parallel without write
/// conflicts.
///
/// `fespaces` is the block's FE subspaces: usually one, but several for a
/// multi-quadrature element. They must **share one submesh** (same connectivity,
/// same coordinates, same nodes-per-cell), differing only by quadrature — the
/// primary (index 0) drives the cell loop and the scatter numbering, the others
/// only add their reference data. The kernel receives one [`CellGeom`] per
/// subspace, in order.
#[allow(clippy::too_many_arguments)]
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // Ce noyau ne lit aucun matériau ; l'assembleur en veut un, on lui en
/// # // donne un qui ne sert à rien plutôt qu'une `Option` à déballer.
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // La même chose, **groupée par maille** : ce que consomme l'assembleur
/// // global pour verser droit dans le CSR sans matérialiser de valeurs.
/// let (nr, nc, par_maille): kernel::BlockTripletsPerCell =
///     kernel::element_block_triplets_per_cell(
///         std::slice::from_ref(&zone), &support, &support, 1, 1,
///         DofOrdering::NodesThenVars, &mat, None,
///         |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0); Ok(()) },
///     )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(par_maille.len(), 1); // une maille
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_block_triplets_per_cell(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
    material: &Handle<SubElementField>,
    state: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], &SubElementField, Option<&SubElementField>, &mut [f64]) -> Result<()>
        + Sync,
) -> Result<BlockTripletsPerCell> {
    let primary = fespaces.first().ok_or_else(|| {
        PyrucastError::Message("element_block_triplets_per_cell: no FE subspace".into())
    })?;
    let fe = primary.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();
    let mat_guard = material.read();
    let state_guard = state.map(|h| h.read());

    // Reference data of every subspace, snapshotted once (they share the submesh
    // ⇒ one connectivity + coords drive every CellGeom; only quadrature differs).
    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = h.read();
        if !f.submesh().same_object(&submesh) {
            return Err(PyrucastError::Message(
                "element_block_triplets_per_cell: all FE subspaces of a block must share one submesh"
                    .into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count();
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    // Every node of the connectivity, checked live **once** — the cell
    // geometry then reads coordinates without asking again.
    coords.ensure_all_alive(conn)?;
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let mat_ref: &SubElementField = &mat_guard;
    let state_ref: Option<&SubElementField> = state_guard.as_deref();

    let n_cols_loc = n_nodes * n_primal;
    let ke_len = (n_nodes * n_dual) * n_cols_loc;

    // Support → local position maps (first occurrence wins), and the block's
    // local row/col dimensions — all known up front, so the whole loop runs in
    // parallel (no shared mutation).
    let row_nodes: Vec<NodeId> = row_support.read().connectivity().to_vec();
    let col_nodes: Vec<NodeId> = col_support.read().connectivity().to_vec();
    let n_row_nodes = row_nodes.len();
    let n_col_nodes = col_nodes.len();
    let nrows = n_row_nodes * n_dual;
    let ncols = n_col_nodes * n_primal;
    let pos_map = |nodes: &[NodeId]| -> HashMap<NodeId, u32> {
        let mut m = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            m.entry(n).or_insert(i as u32);
        }
        m
    };
    let row_pos = pos_map(&row_nodes);
    let col_pos = pos_map(&col_nodes);
    // La position de **chaque nœud de la connectivité**, d'un seul parcours,
    // avant la région parallèle. La hacher dans la boucle reposait la même
    // question à chaque maille et à chaque assemblage, pour une réponse qui est
    // un fait de la zone ; et l'erreur se nomme ici, où l'on sait encore quoi.
    let lookup = |pos: &HashMap<NodeId, u32>, side: &str| -> Result<Vec<u32>> {
        conn.iter()
            .map(|nid| {
                pos.get(nid).copied().ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_triplets: node {nid:?} not in {side} support"
                    ))
                })
            })
            .collect()
    };
    let row_slot = lookup(&row_pos, "row")?;
    let col_slot = lookup(&col_pos, "col")?;
    let (row_slot, col_slot): (&[u32], &[u32]) = (&row_slot, &col_slot);

    // Per cell, in parallel: compute the element matrix, then emit its triplets
    // in **local** indices. Order within a cell is li,di,lj,pj; cells are
    // concatenated in order below ⇒ identical triplet stream regardless of
    // thread count (bit-for-bit result).
    // Per-thread scratch: the geometry list, the element matrix and the two
    // position lists are rebuilt for every cell, and allocating them there cost
    // four `Vec` per cell.
    let per_cell: Vec<Vec<(usize, usize, f64)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map_init(
            || (Vec::with_capacity(rds_ref.len()), vec![0.0_f64; ke_len]),
            |(geoms, ke), cell| -> Result<Vec<(usize, usize, f64)>> {
                geoms.clear();
                geoms.extend(
                    rds_ref
                        .iter()
                        .map(|rd| CellGeom::new(rd, coords_ref, conn, cell)),
                );
                ke.fill(0.0);
                element(geoms, mat_ref, state_ref, ke)?;

                let base = cell * n_nodes;
                let rpos = &row_slot[base..base + n_nodes];
                let cpos = &col_slot[base..base + n_nodes];

                let mut trips = Vec::with_capacity(ke_len);
                for li in 0..n_nodes {
                    for di in 0..n_dual {
                        let r = li * n_dual + di;
                        let ri = ordering.to_index(rpos[li] as usize, di, n_row_nodes, n_dual);
                        for lj in 0..n_nodes {
                            for pj in 0..n_primal {
                                let c = lj * n_primal + pj;
                                let ci =
                                    ordering.to_index(cpos[lj] as usize, pj, n_col_nodes, n_primal);
                                trips.push((ri, ci, ke[r * n_cols_loc + c]));
                            }
                        }
                    }
                }
                Ok(trips)
            },
        )
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
}

/// The **symbolic** structure of a computed block, in compact form.
///
/// It used to be the `(row, col)` index pair of every entry of every cell —
/// five hundred and seventy-six pairs per `HEX8` in elasticity, ninety-two
/// gigabytes for ten million cells, built only to be walked twice and thrown
/// away. Every one of those pairs is a pure function of two things the cell
/// already carries: where each of its nodes sits in the row support, and where
/// it sits in the column support. So that is what is kept — eight numbers per
/// cell per side — and the pairs are **regenerated** where they are wanted.
///
/// Carries no geometry and evaluates no kernel: only connectivity and the DOF
/// `ordering`. An assembler builds the global CSR sparsity from it, caches the
/// result, and runs the numeric kernel only when values are needed.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let support = zone.read().submesh().read().to_poi1()?;
/// // Le **motif** seul, sans valeurs : c'est lui qui est mis en cache et
/// // réutilisé d'un assemblage à l'autre, la matière ne le changeant pas.
/// let motif = kernel::element_block_pattern(
///     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars,
/// )?;
/// assert_eq!((motif.nrows, motif.ncols), (3, 3));
/// assert_eq!(motif.entries_per_cell(), 3 * 3); // toutes les paires d'une maille
/// // La paire `(ligne, colonne)` d'une entrée se regénère à la demande.
/// assert_eq!(motif.row_index(0, 0, 0), 0);
/// assert_eq!(motif.col_index(0, 2, 0), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct BlockPattern {
    /// Number of block rows.
    pub nrows: usize,
    /// Number of block columns.
    pub ncols: usize,
    /// Number of cells the block walks.
    pub n_cells: usize,
    /// Row-support position of each entry of the row connectivity,
    /// `n_cells × row_nodes_per_cell`.
    pub row_slot: Vec<u32>,
    /// Column-support position of each entry of the column connectivity,
    /// `n_cells × col_nodes_per_cell`.
    pub col_slot: Vec<u32>,
    /// Nodes per cell on the row side.
    pub row_nodes_per_cell: usize,
    /// Nodes per cell on the column side (equal to the row count except on an
    /// inter-mesh coupling block).
    pub col_nodes_per_cell: usize,
    /// Nodes in the row support — the width the DOF ordering needs.
    pub n_row_support: usize,
    /// Nodes in the column support.
    pub n_col_support: usize,
    /// Dual variables per row node.
    pub n_dual: usize,
    /// Primal variables per column node.
    pub n_primal: usize,
    /// How `(node, variable)` maps to a flat index.
    pub ordering: DofOrdering,
}

impl BlockPattern {
    /// Block-local **row** index of the entry at local node `li`, dual variable
    /// `di` of `cell`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let zone = fes.get(0)?;
    /// # let support = zone.read().submesh().read().to_poi1()?;
    /// # let motif = kernel::element_block_pattern(
    /// #     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars)?;
    /// // La ligne d'une entrée, regénérée depuis la position du nœud.
    /// assert_eq!(motif.row_index(0, 0, 0), 0);
    /// assert_eq!(motif.row_index(0, 2, 0), 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn row_index(&self, cell: usize, li: usize, di: usize) -> usize {
        let p = self.row_slot[cell * self.row_nodes_per_cell + li] as usize;
        self.ordering
            .to_index(p, di, self.n_row_support, self.n_dual)
    }

    /// Block-local **column** index of the entry at local node `lj`, primal
    /// variable `pj` of `cell`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let zone = fes.get(0)?;
    /// # let support = zone.read().submesh().read().to_poi1()?;
    /// # let motif = kernel::element_block_pattern(
    /// #     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars)?;
    /// // La colonne, de même — le motif ne stocke aucune paire.
    /// assert_eq!(motif.col_index(0, 1, 0), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn col_index(&self, cell: usize, lj: usize, pj: usize) -> usize {
        let p = self.col_slot[cell * self.col_nodes_per_cell + lj] as usize;
        self.ordering
            .to_index(p, pj, self.n_col_support, self.n_primal)
    }

    /// Entries one cell contributes — the length of its `ke`, and the number of
    /// `(row, col)` pairs the loops above regenerate.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::kernel;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let zone = fes.get(0)?;
    /// # let support = zone.read().submesh().read().to_poi1()?;
    /// # let motif = kernel::element_block_pattern(
    /// #     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars)?;
    /// // Un TRI3 scalaire : trois nœuds au carré.
    /// assert_eq!(motif.entries_per_cell(), 3 * 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn entries_per_cell(&self) -> usize {
        (self.row_nodes_per_cell * self.n_dual) * (self.col_nodes_per_cell * self.n_primal)
    }
}

/// The [`BlockPattern`] of a computed stiffness block — where each cell's nodes
/// sit in the row and column supports, plus the DOF layout that turns those
/// positions into `(row, col)` pairs in the `(li, di, lj, pj)` order
/// [`element_block_triplets`] emits its values.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let support = zone.read().submesh().read().to_poi1()?;
/// // Le motif compact : la position de chaque nœud dans les deux supports.
/// let motif = kernel::element_block_pattern(
///     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars,
/// )?;
/// assert_eq!((motif.nrows, motif.ncols), (3, 3));
/// assert_eq!(motif.row_slot.len(), 3); // une maille, trois nœuds
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_block_pattern(
    fespace: &Handle<SubFiniteElementSpace>,
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
) -> Result<BlockPattern> {
    let fe = fespace.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let conn: &[NodeId] = sm.connectivity();
    let n_cells = fe.cell_count();
    let n_nodes = conn.len().checked_div(n_cells).unwrap_or(0);

    let row_nodes: Vec<NodeId> = row_support.read().connectivity().to_vec();
    let col_nodes: Vec<NodeId> = col_support.read().connectivity().to_vec();
    let (n_row_support, n_col_support) = (row_nodes.len(), col_nodes.len());

    Ok(BlockPattern {
        nrows: n_row_support * n_dual,
        ncols: n_col_support * n_primal,
        n_cells,
        row_slot: local_positions(conn, &position_map(&row_nodes), "row")?,
        col_slot: local_positions(conn, &position_map(&col_nodes), "column")?,
        row_nodes_per_cell: n_nodes,
        col_nodes_per_cell: n_nodes,
        n_row_support,
        n_col_support,
        n_dual,
        n_primal,
        ordering,
    })
}

// ─── Inter-mesh coupling blocks ─────────────────────────────────────────────

/// Check that two FE subspace groups face each other cell for cell, and return
/// the shared `(cell count, row nodes/cell, col nodes/cell)`.
///
/// Two conforming boundary meshes are what an interface law needs: cell `i` of
/// one against cell `i` of the other. Anything else is a meshing problem, and is
/// reported as one rather than silently paired.
fn check_conforming(
    row_fe: &SubFiniteElementSpace,
    col_fe: &SubFiniteElementSpace,
    row_conn: &[NodeId],
    col_conn: &[NodeId],
) -> Result<(usize, usize, usize)> {
    let (n_row_cells, n_col_cells) = (row_fe.cell_count(), col_fe.cell_count());
    if n_row_cells != n_col_cells {
        return Err(PyrucastError::Message(format!(
            "coupling block: the two sides of an interface must be conforming — \
             {n_row_cells} cell(s) facing {n_col_cells}"
        )));
    }
    let n_row_nodes = row_conn.len().checked_div(n_row_cells).unwrap_or(0);
    let n_col_nodes = col_conn.len().checked_div(n_col_cells).unwrap_or(0);
    Ok((n_row_cells, n_row_nodes, n_col_nodes))
}

/// Per-cell local triplets of an **inter-mesh** block: rows on one mesh, columns
/// on the facing one, paired cell by cell.
///
/// The same shape as [`element_block_triplets_per_cell`], and the same guarantees
/// (element matrices evaluated in parallel, triplets emitted in `li, di, lj, pj`
/// order so the stream is reproducible). What differs is that the cell loop
/// walks **two** connectivities: the row indices come from the row mesh, the
/// column indices from the facing one.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::{self, assemble_block};
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # // Ce noyau ne lit aucun matériau ; l'assembleur en veut un, on lui en
/// # // donne un qui ne sert à rien plutôt qu'une `Option` à déballer.
/// # use pyrucast::containers::element_field::SubElementField;
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0]).unwrap());
/// // Les valeurs du bloc de couplage, groupées par maille — le noyau y
/// // reçoit **deux** jeux de `CellGeom`, un par côté.
/// let (nr, nc, par_maille) = kernel::coupling_block_triplets_per_cell(
///     std::slice::from_ref(&zone), std::slice::from_ref(&zone),
///     &support, &support, 1, 1, DofOrdering::NodesThenVars, &mat,
///     |lignes, colonnes, _m, ke| {
///         assert_eq!(lignes.len(), colonnes.len());
///         ke[0] = lignes[0].det_j_w(0);
///         Ok(())
///     },
/// )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(par_maille.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn coupling_block_triplets_per_cell(
    row_fespaces: &[Handle<SubFiniteElementSpace>],
    col_fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
    material: &Handle<SubElementField>,
    element: impl Fn(&[CellGeom], &[CellGeom], &SubElementField, &mut [f64]) -> Result<()> + Sync,
) -> Result<BlockTripletsPerCell> {
    let row_primary = row_fespaces.first().ok_or_else(|| {
        PyrucastError::Message("coupling_block_triplets_per_cell: no row FE subspace".into())
    })?;
    let col_primary = col_fespaces.first().ok_or_else(|| {
        PyrucastError::Message("coupling_block_triplets_per_cell: no column FE subspace".into())
    })?;
    let (row_fe, col_fe) = (row_primary.read(), col_primary.read());
    let (row_sm_h, col_sm_h) = (row_fe.submesh(), col_fe.submesh());
    let (row_sm, col_sm) = (row_sm_h.read(), col_sm_h.read());
    let (row_coords_h, col_coords_h) = (row_sm.coords(), col_sm.coords());
    let (row_coords, col_coords) = (row_coords_h.read(), col_coords_h.read());
    let mat_guard = material.read();

    let row_conn: &[NodeId] = row_sm.connectivity();
    let col_conn: &[NodeId] = col_sm.connectivity();
    // Both connectivities, checked live **once** — the cell geometries then read
    // coordinates without asking again.
    row_coords.ensure_all_alive(row_conn)?;
    col_coords.ensure_all_alive(col_conn)?;
    let (n_cells, n_row_nodes_cell, n_col_nodes_cell) =
        check_conforming(&row_fe, &col_fe, row_conn, col_conn)?;

    let mut row_rds = vec![RefData::snapshot(&row_fe)?];
    for h in &row_fespaces[1..] {
        let f = h.read();
        row_rds.push(RefData::snapshot(&f)?);
    }
    let mut col_rds = vec![RefData::snapshot(&col_fe)?];
    for h in &col_fespaces[1..] {
        let f = h.read();
        col_rds.push(RefData::snapshot(&f)?);
    }

    let row_nodes: Vec<NodeId> = row_support.read().connectivity().to_vec();
    let col_nodes: Vec<NodeId> = col_support.read().connectivity().to_vec();
    let (n_row_support, n_col_support) = (row_nodes.len(), col_nodes.len());
    let (nrows, ncols) = (n_row_support * n_dual, n_col_support * n_primal);
    let row_pos = position_map(&row_nodes);
    let col_pos = position_map(&col_nodes);

    let n_cols_loc = n_col_nodes_cell * n_primal;
    let ke_len = (n_row_nodes_cell * n_dual) * n_cols_loc;
    let (row_rds_ref, col_rds_ref) = (&row_rds[..], &col_rds[..]);
    let (row_coords_ref, col_coords_ref): (&Coords, &Coords) = (&row_coords, &col_coords);
    let mat_ref: &SubElementField = &mat_guard;

    let per_cell: Vec<Vec<(usize, usize, f64)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_row_nodes_cell.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize, f64)>> {
            let row_geoms: Vec<CellGeom> = row_rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, row_coords_ref, row_conn, cell))
                .collect();
            let col_geoms: Vec<CellGeom> = col_rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, col_coords_ref, col_conn, cell))
                .collect();
            let mut ke = vec![0.0_f64; ke_len];
            element(&row_geoms, &col_geoms, mat_ref, &mut ke)?;

            let rpos = local_positions(
                &row_conn[cell * n_row_nodes_cell..(cell + 1) * n_row_nodes_cell],
                &row_pos,
                "row",
            )?;
            let cpos = local_positions(
                &col_conn[cell * n_col_nodes_cell..(cell + 1) * n_col_nodes_cell],
                &col_pos,
                "column",
            )?;

            let mut trips = Vec::with_capacity(ke_len);
            for (li, &rl) in rpos.iter().enumerate() {
                for di in 0..n_dual {
                    let r = li * n_dual + di;
                    let ri = ordering.to_index(rl as usize, di, n_row_support, n_dual);
                    for (lj, &cl) in cpos.iter().enumerate() {
                        for pj in 0..n_primal {
                            let c = lj * n_primal + pj;
                            let ci = ordering.to_index(cl as usize, pj, n_col_support, n_primal);
                            trips.push((ri, ci, ke[r * n_cols_loc + c]));
                        }
                    }
                }
            }
            Ok(trips)
        })
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
}

/// The element matrices of a block, **values only**, cell after cell — the form
/// the *computed* assembly path consumes.
///
/// Returns the concatenated `ke` of every cell (row-major, `ke_len` values each)
/// and that stride. It needs neither support, nor DOF ordering, nor index maps:
/// the triplet stream of [`element_block_triplets_per_cell`] emits `ke` in
/// exactly this order, and the computed path pairs it position-for-position with
/// the slots [`crate::ops::scatter`] resolved once. Asking for the `(row, col)`
/// of each entry only to drop them cost three times the memory — 13,8 ko per
/// cell on a 3-D mechanical HEX8 — plus a `to_index` and two hash lookups per
/// entry, at every assembly.
///
/// The *literal* path still wants triplets: it has no pattern to pair with.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0])?);
/// // Un noyau qui écrit l'identité : on retrouve `ke` en ligne-major,
/// // maille après maille, et rien d'autre — ni ligne, ni colonne.
/// let (valeurs, ke_len) = kernel::element_block_values_per_cell(
///     std::slice::from_ref(&zone), 1, 1, &mat, None,
///     |geoms, _m, _s, ke| {
///         let n = geoms[0].n_nodes;
///         for i in 0..n {
///             ke[i * n + i] = 1.0;
///         }
///         Ok(())
///     },
/// )?;
/// assert_eq!(ke_len, 3 * 3); // un TRI3, une variable de chaque côté
/// assert_eq!(valeurs.len(), ke_len); // une seule maille
/// assert_eq!(valeurs[0], 1.0);
/// assert_eq!(valeurs[1], 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_block_values_per_cell(
    fespaces: &[Handle<SubFiniteElementSpace>],
    n_dual: usize,
    n_primal: usize,
    material: &Handle<SubElementField>,
    state: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], &SubElementField, Option<&SubElementField>, &mut [f64]) -> Result<()>
        + Sync,
) -> Result<(Vec<f64>, usize)> {
    let primary = fespaces.first().ok_or_else(|| {
        PyrucastError::Message("element_block_values_per_cell: no FE subspace".into())
    })?;
    let fe = primary.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();
    let mat_guard = material.read();
    let state_guard = state.map(|h| h.read());

    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = h.read();
        if !f.submesh().same_object(&submesh) {
            return Err(PyrucastError::Message(
                "element_block_values_per_cell: all FE subspaces of a block must share one submesh"
                    .into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count();
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    coords.ensure_all_alive(conn)?;
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let mat_ref: &SubElementField = &mat_guard;
    let state_ref: Option<&SubElementField> = state_guard.as_deref();

    let ke_len = (n_nodes * n_dual) * (n_nodes * n_primal);
    // One flat buffer, written by disjoint chunks: each cell owns its own slice,
    // so the result is bit-for-bit independent of the thread count.
    let mut values = vec![0.0_f64; n_cells * ke_len];
    values
        .par_chunks_mut(ke_len.max(1))
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .enumerate()
        .try_for_each_init(
            || Vec::with_capacity(rds_ref.len()),
            |geoms, (cell, ke)| -> Result<()> {
                geoms.clear();
                geoms.extend(
                    rds_ref
                        .iter()
                        .map(|rd| CellGeom::new(rd, coords_ref, conn, cell)),
                );
                element(geoms, mat_ref, state_ref, ke)
            },
        )?;
    Ok((values, ke_len))
}

/// Drive a computed block's element kernel **cell by cell, colour by colour**,
/// handing each `ke` straight to `emit` instead of materialising them all.
///
/// [`element_block_values_per_cell`] returns every element matrix of the block
/// at once. On a solid mesh that is one buffer of `n_cells × ke_len` doubles —
/// forty-six gigabytes for ten million `HEX8` in elasticity — written in full,
/// then read back in full by the scatter that follows. The values have no life
/// of their own between the two: each is produced, consumed once, and never
/// looked at again.
///
/// So this form produces and consumes them in the same breath. Each task keeps
/// **one** `ke` scratch, reused from one cell to the next; `emit(cell, ke)`
/// receives it and scatters it wherever the caller keeps its accumulator.
///
/// The cells are visited in the colouring's order (cached on the primary FE
/// subspace, keyed on shared nodes): within a colour the cells touch
/// pairwise-disjoint nodes, so their scatters do not race, and the colours run
/// in sequence. `emit` therefore sees every cell of colour 0 before any cell of
/// colour 1 — the same order the two-phase form's caller imposed, hence the same
/// accumulation order and the same result.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["k".into()], &[1.0])?);
/// // Les mêmes valeurs que la forme en deux temps, mailles jamais toutes
/// // matérialisées : chaque `ke` est produite puis consommée sur-le-champ.
/// let noyau = |geoms: &[kernel::CellGeom], _m: &SubElementField,
///              _s: Option<&SubElementField>, ke: &mut [f64]| {
///     ke[0] = geoms[0].det_j_w(0);
///     Ok(())
/// };
/// let (attendu, ke_len) = kernel::element_block_values_per_cell(
///     std::slice::from_ref(&zone), 1, 1, &mat, None, noyau)?;
/// let vu = std::sync::Mutex::new(vec![0.0; attendu.len()]);
/// kernel::element_block_colored(
///     std::slice::from_ref(&zone), 1, 1, &mat, None, noyau,
///     |maille, ke| {
///         vu.lock().unwrap()[maille * ke_len..(maille + 1) * ke_len]
///             .copy_from_slice(ke);
///         Ok(())
///     },
/// )?;
/// assert_eq!(vu.into_inner().unwrap(), attendu);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_block_colored(
    fespaces: &[Handle<SubFiniteElementSpace>],
    n_dual: usize,
    n_primal: usize,
    material: &Handle<SubElementField>,
    state: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], &SubElementField, Option<&SubElementField>, &mut [f64]) -> Result<()>
        + Sync,
    emit: impl Fn(usize, &[f64]) -> Result<()> + Sync,
) -> Result<()> {
    // Same block prologue as `element_block_values_per_cell`: reference data,
    // coordinates and material resolved once, guards held for the whole drive so
    // the per-cell work stays lock-free.
    let primary = fespaces
        .first()
        .ok_or_else(|| PyrucastError::Message("element_block_colored: no FE subspace".into()))?;
    let fe = primary.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();
    let mat_guard = material.read();
    let state_guard = state.map(|h| h.read());

    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = h.read();
        if !f.submesh().same_object(&submesh) {
            return Err(PyrucastError::Message(
                "element_block_colored: all FE subspaces of a block must share one submesh".into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count();
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    coords.ensure_all_alive(conn)?;
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let mat_ref: &SubElementField = &mat_guard;
    let state_ref: Option<&SubElementField> = state_guard.as_deref();
    let ke_len = (n_nodes * n_dual) * (n_nodes * n_primal);

    let coloring =
        fe.coloring(|| coloring::greedy_color_nodes(n_cells, n_nodes, conn, coords.node_count()));

    for color in coloring {
        color
            .par_iter()
            .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
            .try_for_each_init(
                // One `ke` and one geometry list per task, reused across its
                // cells: no per-cell allocation, and no element set held whole.
                || (vec![0.0_f64; ke_len], Vec::with_capacity(rds_ref.len())),
                |(ke, geoms), &cell| -> Result<()> {
                    geoms.clear();
                    geoms.extend(
                        rds_ref
                            .iter()
                            .map(|rd| CellGeom::new(rd, coords_ref, conn, cell)),
                    );
                    ke.iter_mut().for_each(|v| *v = 0.0);
                    element(geoms, mat_ref, state_ref, ke)?;
                    emit(cell, ke)
                },
            )?;
    }
    Ok(())
}

/// [`element_block_values_per_cell`] for an **inter-mesh coupling** block.
///
/// Same contract: the concatenated `ke` of every facing cell pair, row-major,
/// and its stride.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["h".into()], &[1.0])?);
/// // Une interface conforme avec elle-même : deux SEG2 en vis-à-vis.
/// let (valeurs, ke_len) = kernel::coupling_block_values_per_cell(
///     std::slice::from_ref(&zone), std::slice::from_ref(&zone), 1, 1, &mat,
///     |row_geoms, _col_geoms, _m, ke| {
///         ke[0] = row_geoms[0].det_j_w(0);
///         Ok(())
///     },
/// )?;
/// assert_eq!(ke_len, 2 * 2); // deux nœuds de chaque côté
/// assert!(valeurs[0] > 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn coupling_block_values_per_cell(
    row_fespaces: &[Handle<SubFiniteElementSpace>],
    col_fespaces: &[Handle<SubFiniteElementSpace>],
    n_dual: usize,
    n_primal: usize,
    material: &Handle<SubElementField>,
    element: impl Fn(&[CellGeom], &[CellGeom], &SubElementField, &mut [f64]) -> Result<()> + Sync,
) -> Result<(Vec<f64>, usize)> {
    let row_primary = row_fespaces.first().ok_or_else(|| {
        PyrucastError::Message("coupling_block_values_per_cell: no row FE subspace".into())
    })?;
    let col_primary = col_fespaces.first().ok_or_else(|| {
        PyrucastError::Message("coupling_block_values_per_cell: no column FE subspace".into())
    })?;
    let (row_fe, col_fe) = (row_primary.read(), col_primary.read());
    let (row_sm_h, col_sm_h) = (row_fe.submesh(), col_fe.submesh());
    let (row_sm, col_sm) = (row_sm_h.read(), col_sm_h.read());
    let (row_coords_h, col_coords_h) = (row_sm.coords(), col_sm.coords());
    let (row_coords, col_coords) = (row_coords_h.read(), col_coords_h.read());
    let mat_guard = material.read();

    let row_conn: &[NodeId] = row_sm.connectivity();
    let col_conn: &[NodeId] = col_sm.connectivity();
    row_coords.ensure_all_alive(row_conn)?;
    col_coords.ensure_all_alive(col_conn)?;
    let (n_cells, n_row_nodes_cell, n_col_nodes_cell) =
        check_conforming(&row_fe, &col_fe, row_conn, col_conn)?;

    let mut row_rds = vec![RefData::snapshot(&row_fe)?];
    for h in &row_fespaces[1..] {
        row_rds.push(RefData::snapshot(&h.read())?);
    }
    let mut col_rds = vec![RefData::snapshot(&col_fe)?];
    for h in &col_fespaces[1..] {
        col_rds.push(RefData::snapshot(&h.read())?);
    }
    let (row_rds_ref, col_rds_ref) = (&row_rds[..], &col_rds[..]);
    let (row_coords_ref, col_coords_ref): (&Coords, &Coords) = (&row_coords, &col_coords);
    let mat_ref: &SubElementField = &mat_guard;

    let ke_len = (n_row_nodes_cell * n_dual) * (n_col_nodes_cell * n_primal);
    let mut values = vec![0.0_f64; n_cells * ke_len];
    values
        .par_chunks_mut(ke_len.max(1))
        .with_min_len((MIN_PARALLEL_LEN / n_row_nodes_cell.max(1)).max(1))
        .enumerate()
        .try_for_each_init(
            || {
                (
                    Vec::with_capacity(row_rds_ref.len()),
                    Vec::with_capacity(col_rds_ref.len()),
                )
            },
            |(row_geoms, col_geoms), (cell, ke)| -> Result<()> {
                row_geoms.clear();
                row_geoms.extend(
                    row_rds_ref
                        .iter()
                        .map(|rd| CellGeom::new(rd, row_coords_ref, row_conn, cell)),
                );
                col_geoms.clear();
                col_geoms.extend(
                    col_rds_ref
                        .iter()
                        .map(|rd| CellGeom::new(rd, col_coords_ref, col_conn, cell)),
                );
                element(row_geoms, col_geoms, mat_ref, ke)
            },
        )?;
    Ok((values, ke_len))
}

/// The [`BlockPattern`] of an inter-mesh block — the coupling counterpart of
/// [`element_block_pattern`], carrying no geometry and evaluating no kernel.
/// Its rows are read on one mesh and its columns on the facing one, so the two
/// sides carry their own node counts.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let zone = fes.get(0)?;
/// # let support = zone.read().submesh().read().to_poi1()?;
/// // Un bloc **hors diagonale** : ses lignes vivent sur un maillage, ses
/// // colonnes sur celui d'en face. C'est ce qu'exige une loi d'interface.
/// // Ici les deux côtés sont le même, ce qui suffit à montrer la forme.
/// let motif = kernel::coupling_block_pattern(
///     &zone, &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars)?;
/// assert_eq!((motif.nrows, motif.ncols), (3, 3));
/// assert_eq!(motif.entries_per_cell(), 3 * 3);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn coupling_block_pattern(
    row_fespace: &Handle<SubFiniteElementSpace>,
    col_fespace: &Handle<SubFiniteElementSpace>,
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
) -> Result<BlockPattern> {
    let (row_fe, col_fe) = (row_fespace.read(), col_fespace.read());
    let (row_sm_h, col_sm_h) = (row_fe.submesh(), col_fe.submesh());
    let (row_sm, col_sm) = (row_sm_h.read(), col_sm_h.read());
    let row_conn: &[NodeId] = row_sm.connectivity();
    let col_conn: &[NodeId] = col_sm.connectivity();
    let (n_cells, row_nodes_per_cell, col_nodes_per_cell) =
        check_conforming(&row_fe, &col_fe, row_conn, col_conn)?;

    let row_nodes: Vec<NodeId> = row_support.read().connectivity().to_vec();
    let col_nodes: Vec<NodeId> = col_support.read().connectivity().to_vec();
    let (n_row_support, n_col_support) = (row_nodes.len(), col_nodes.len());

    Ok(BlockPattern {
        nrows: n_row_support * n_dual,
        ncols: n_col_support * n_primal,
        n_cells,
        row_slot: local_positions(row_conn, &position_map(&row_nodes), "row")?,
        col_slot: local_positions(col_conn, &position_map(&col_nodes), "column")?,
        row_nodes_per_cell,
        col_nodes_per_cell,
        n_row_support,
        n_col_support,
        n_dual,
        n_primal,
        ordering,
    })
}

/// Node → position in a support (first occurrence wins).
fn position_map(nodes: &[NodeId]) -> HashMap<NodeId, u32> {
    let mut m = HashMap::with_capacity(nodes.len());
    for (i, &n) in nodes.iter().enumerate() {
        m.entry(n).or_insert(i as u32);
    }
    m
}

/// Positions of a cell's nodes in a support, erroring by name on a stray node.
fn local_positions(ids: &[NodeId], pos: &HashMap<NodeId, u32>, side: &str) -> Result<Vec<u32>> {
    ids.iter()
        .map(|nid| {
            pos.get(nid).copied().ok_or_else(|| {
                PyrucastError::Message(format!(
                    "coupling block: node {nid:?} is not in the {side} support"
                ))
            })
        })
        .collect()
}

/// Integrate a per-cell kernel over `fespaces` and **scatter the result to the
/// nodes** of `support`, in parallel — the shared nodal integrate-and-scatter
/// driver. It backs the `Bᵀ` operators (internal forces `∫ Bᵀ σ`, Cast3m `BSIG`;
/// the weak divergence
/// [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)) and the
/// distributed flux load `∫ φ N`
/// ([`crate::ops::node_field::flux`](fn@crate::ops::node_field::flux)) alike.
///
/// `element(geoms, fe)` is a pure sequential kernel: for one cell it fills `fe` —
/// the cell's local vector, **node-major / variable-minor**
/// (`fe[li * n_dual + di]`, `di` indexing `dual_vars`) — from the cell geometry
/// (one [`CellGeom`] per FE subspace of `fespaces`, same order). Its integrand
/// (a stress field, a flux density, …) is **captured by the closure**, borrowed
/// in place; the driver itself is agnostic to it. `element` never sees rayon, the
/// store, or a lock.
///
/// Each cell's vector is built and **scattered in the same parallel pass**, colour
/// by colour (the primary FE subspace's cached cell colouring): within a colour
/// the cells touch pairwise-disjoint nodes, so their accumulation into `support`'s
/// node slots never races, and colours run in sequence. Each node therefore sums
/// its cells in a fixed colour order — reproducible for any thread count, though
/// not bit-for-bit with a cell-order sum (see
/// [`crate::parallel::colored_scatter`]). The local vector lives on a per-thread
/// scratch buffer, so the whole element set is never materialised. Returns the
/// [`SubNodeField`] with one component per `dual_vars` on `support`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// // Le pilote des **producteurs de champ nodal** : forces internes, flux.
/// // L'écriture concurrente passe par un coloriage, d'où un résultat
/// // reproductible bit à bit.
/// let f = kernel::scatter_to_nodes(
///     std::slice::from_ref(&zone), &support, vec!["q".into()],
///     |geoms, fe| {
///         for (i, slot) in fe.iter_mut().enumerate() {
///             *slot = geoms[0].det_j_w(0) * (i as f64);
///         }
///         Ok(())
///     },
/// )?;
/// assert_eq!(f.node_count(), 3);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn scatter_to_nodes(
    fespaces: &[Handle<SubFiniteElementSpace>],
    support: &Handle<SubMesh>,
    dual_vars: Vec<String>,
    element: impl Fn(&[CellGeom], &mut [f64]) -> Result<()> + Sync,
) -> Result<SubNodeField> {
    let n_dual = dual_vars.len();
    let primary = fespaces
        .first()
        .ok_or_else(|| PyrucastError::Message("scatter_to_nodes: no FE subspace".into()))?;
    let fe = primary.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();

    // Reference data of every subspace, snapshotted once (they share the submesh
    // ⇒ one connectivity + coords drive every CellGeom; only quadrature differs).
    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = h.read();
        if !f.submesh().same_object(&submesh) {
            return Err(PyrucastError::Message(
                "scatter_to_nodes: all FE subspaces must share one submesh".into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count();
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    // Every node of the connectivity, checked live **once** — the cell
    // geometry then reads coordinates without asking again.
    coords.ensure_all_alive(conn)?;
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let fe_len = n_nodes * n_dual;

    // Support slots: the unique nodes of `support` and each node's flat base.
    // `support` is the POI1 of the submesh, so it covers every connectivity node
    // and the map is total.
    let unique: Vec<NodeId> = support.read().connectivity().to_vec();
    let slot_of: HashMap<NodeId, usize> = unique.iter().enumerate().map(|(k, &n)| (n, k)).collect();
    // La case de **chaque nœud de la connectivité**, d'un seul parcours, avant
    // la région parallèle. Le commentaire ci-dessus dit que la table est
    // totale : autant s'en servir une fois plutôt que la hacher par maille.
    let slots: Vec<usize> = conn
        .iter()
        .map(|nid| {
            slot_of.get(nid).copied().ok_or_else(|| {
                PyrucastError::Message(
                    "scatter_to_nodes: support does not cover a cell node".into(),
                )
            })
        })
        .collect::<Result<_>>()?;
    let slots: &[usize] = &slots;

    // Cell colouring (cached on the primary FE subspace): two cells sharing a
    // node get different colours, so within a colour the cells scatter to
    // pairwise-disjoint nodes.
    let coloring =
        fe.coloring(|| coloring::greedy_color_nodes(n_cells, n_nodes, conn, coords.node_count()));

    // Fused compute + scatter, colour by colour: each cell builds its local
    // vector on a per-thread scratch buffer (no per-cell heap alloc, no
    // materialisation of the whole element set) and scatters it straight into the
    // node slots.
    // The field first, so the scatter accumulates straight into its buffer:
    // one write lock held for the call, and no intermediate vector.
    let mut out_field = SubNodeField::from_poi1(support, dual_vars)?;
    colored_scatter(
        out_field.values_mut(),
        coloring,
        (MIN_PARALLEL_LEN / n_nodes.max(1)).max(1),
        // The scratch carries both the element vector and the geometry list:
        // rebuilding the latter per cell allocated a `Vec` per cell of every
        // call, for a list that is almost always one element long.
        || (vec![0.0_f64; fe_len], Vec::with_capacity(rds_ref.len())),
        |cell, (fe_cell, geoms), out| {
            geoms.clear();
            geoms.extend(
                rds_ref
                    .iter()
                    .map(|rd| CellGeom::new(rd, coords_ref, conn, cell)),
            );
            fe_cell.iter_mut().for_each(|v| *v = 0.0);
            element(geoms, fe_cell)?;
            let cell_slots = &slots[cell * n_nodes..(cell + 1) * n_nodes];
            for (li, &node_slot) in cell_slots.iter().enumerate() {
                let base = node_slot * n_dual;
                for di in 0..n_dual {
                    out.add(base + di, fe_cell[li * n_dual + di]);
                }
            }
            Ok(())
        },
    )?;

    // Nothing to write back: the scatter accumulated at
    // `slot_of[nid] * n_dual + di`, and `slot_of` is the position in the
    // support's own connectivity — exactly how the field indexes its rows. The
    // buffer it filled **is** the field's.
    Ok(out_field)
}

/// Parallel **scalar reduction over the cells** of `fespace` — the reduction
/// counterpart of [`scatter_to_nodes`]. `cell(geom)` returns one cell's scalar
/// contribution (e.g. `∫_cell f dΩ` by quadrature); the driver sums them over
/// every cell and returns `Σ_cell cell(geom)`.
///
/// The per-cell geometry is the same lock-free [`CellGeom`] (reference data
/// snapshotted once, coordinates/connectivity borrowed in place), so `cell`
/// never touches rayon or the store. Its integrand (a nodal field, an element
/// field, …) is **captured by the closure**.
///
/// Determinism: the sum is an adaptive parallel reduction, so — like
/// [`crate::containers::field::SubField::dot`] and the other value reductions —
/// floating-point non-associativity makes the total thread-count-dependent to
/// the last ULP (not bit-for-bit reproducible). Cell-local contributions and the
/// geometry are identical to a sequential run; only the summation grouping
/// varies.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::{ElementField, SubElementField};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// // Une réduction sur les mailles : l'aire du maillage, par exemple, se
/// // lit comme la somme des mesures d'intégration.
/// let aire = kernel::reduce_cells(&zone, |geom| {
///     Ok((0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum())
/// })?;
/// assert!((aire - 2.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn reduce_cells(
    fespace: &Handle<SubFiniteElementSpace>,
    cell: impl Fn(&CellGeom) -> Result<f64> + Sync,
) -> Result<f64> {
    let fe = fespace.read();
    let submesh = fe.submesh();
    let sm = submesh.read();
    let coords_h = sm.coords();
    let coords = coords_h.read();
    let rd = RefData::snapshot(&fe)?;

    let n_cells = fe.cell_count();
    let n_nodes = rd.n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    // Every node of the connectivity, checked live **once** — the cell
    // geometry then reads coordinates without asking again.
    coords.ensure_all_alive(conn)?;
    let rd_ref: &RefData = &rd;
    let coords_ref: &Coords = &coords;

    (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|c| -> Result<f64> {
            let geom = CellGeom::new(rd_ref, coords_ref, conn, c);
            cell(&geom)
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b))
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::handle::Handle;

    /// One QUA4 spanning `r ∈ [r0, r1]`, `z ∈ [0, 1]`, in the requested frame.
    fn one_quad(r0: f64, r1: f64, axisymmetric: bool) -> Handle<SubFiniteElementSpace> {
        let coords = Handle::new(if axisymmetric {
            Coords::axisymmetric().unwrap()
        } else {
            Coords::new(2).unwrap()
        });
        let a = Node::create_in(coords.clone(), &[r0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[r1, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[r1, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[r0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        FiniteElementSpace::lagrange1(&mesh)
            .unwrap()
            .get(0)
            .unwrap()
    }

    /// `Σ_g det_j_w` is the cell measure: the plane area in Cartesian, the
    /// **revolved volume** `π(r1² − r0²)·h` in axisymmetric — the single check
    /// that the 2πr factor lands in the quadrature weight.
    #[test]
    fn det_j_w_carries_the_revolution_measure() {
        let plane = reduce_cells(&one_quad(1.0, 2.0, false), |geom| {
            Ok((0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum())
        })
        .unwrap();
        assert!((plane - 1.0).abs() < 1e-12, "plane area {plane} ≠ 1");

        let revolved = reduce_cells(&one_quad(1.0, 2.0, true), |geom| {
            Ok((0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum())
        })
        .unwrap();
        let expected = std::f64::consts::PI * (2.0_f64.powi(2) - 1.0);
        assert!(
            (revolved - expected).abs() < 1e-10,
            "revolved volume {revolved} ≠ {expected}"
        );
    }

    /// Gauss points sit strictly inside the cell, so a cell **touching the axis**
    /// still has `r > 0` — what keeps the hoop term `N_i / r` finite there.
    #[test]
    fn radius_is_positive_even_on_a_cell_touching_the_axis() {
        reduce_cells(&one_quad(0.0, 1.0, true), |geom| {
            for g in 0..geom.n_gauss {
                let r = geom.radius(g);
                assert!(r > 0.0 && r < 1.0, "radius {r} outside (0, 1)");
                // radius is the first physical coordinate, and z stays in [0, 1].
                let mut x = [0.0_f64; 2];
                geom.x_at_g(g, &mut x);
                assert!((x[0] - r).abs() < 1e-15);
                assert!(x[1] > 0.0 && x[1] < 1.0);
            }
            Ok(0.0)
        })
        .unwrap();
    }

    /// `radius` is meaningless without the revolution hypothesis. Ce n'est pas
    /// une donnée à valider mais une faute de programmation : `axisymmetric` est
    /// public sur `CellGeom`, et dit si la question a un sens. Le `debug_assert`
    /// l'attrape en développement, sans peser sur la production ni forcer un
    /// `Result` que le noyau devrait dérouler à chaque point de Gauss.
    #[test]
    #[should_panic(expected = "Cartesian geometry")]
    fn radius_is_a_programming_error_on_a_cartesian_geometry() {
        reduce_cells(&one_quad(1.0, 2.0, false), |geom| Ok(geom.radius(0))).unwrap();
    }
}
