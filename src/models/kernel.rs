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
        Ok(Self {
            n_nodes: fe.nodes_per_cell()?,
            n_gauss,
            space_dim: fe.space_dim(),
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
    ///     assert_eq!(geom.node_coord(1)?, &[2.0, 0.0]);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn node_coord(&self, local: usize) -> Result<&[f64]> {
        self.coords.position(self.node_ids()[local])
    }

    /// Fill the lazy `cell_coords` cache on first use (gather from the held
    /// `Coords`, no store access).
    fn ensure_cell_coords(&self) -> Result<()> {
        let mut cc = self.cell_coords.borrow_mut();
        if cc.is_none() {
            let mut v = Vec::with_capacity(self.n_nodes * self.space_dim);
            for &id in self.node_ids() {
                v.extend_from_slice(self.coords.position(id)?);
            }
            *cc = Some(v);
        }
        Ok(())
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
        self.ensure_cell_coords()?;
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
    ///     assert_eq!(geom.gauss_xi(0)?.len(), 2); // dans l'élément de référence
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gauss_xi(&self, g: usize) -> Result<&[f64]> {
        self.rd
            .gauss_xi
            .get(g)
            .map(|v| v.as_slice())
            .ok_or_else(|| {
                PyrucastError::Message(format!(
                    "CellGeom: Gauss point {g} out of range ({} points)",
                    self.n_gauss
                ))
            })
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
    ///     assert_eq!(geom.field_n_at_g(0, &mut buf)?.len(), geom.shape_count());
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_n_at_g<'s>(&'s self, g: usize, scratch: &'s mut [f64]) -> Result<&'s [f64]> {
        self.reject_if_model_embedded("shape values")?;
        // The common case: the field basis **is** the geometric one, already in
        // reference data. Lending it beats copying it at every Gauss point.
        if self.rd.field_n_ref.is_empty() {
            return Ok(&self.rd.n_ref[g]);
        }
        let j = self.segment_jacobian()?;
        let row = &self.rd.field_n_ref[g];
        scale_slope_slots_into(row, j, &mut scratch[..row.len()]);
        Ok(&scratch[..row.len()])
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
    /// // Les dérivées secondes **physiques**, ce qu'exige la courbure d'une
    /// // poutre. Réservées aux espaces C¹ sur un segment : ailleurs, une
    /// // erreur nommée plutôt qu'un zéro.
    /// kernel::reduce_cells(&zone, |geom| {
    ///     let mut buf = [0.0_f64; 8];
    ///     assert!(geom.field_d2n_dx2(0, &mut buf).is_err()); // TRI3 Lagrange
    ///     Ok(0.0)
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_d2n_dx2(&self, g: usize, out: &mut [f64]) -> Result<usize> {
        self.reject_if_model_embedded("second derivatives")?;
        if self.rd.field_d2n_ref.is_empty() {
            return Err(PyrucastError::Message(
                "CellGeom: this subspace has no C¹ field basis, so no curvature operator".into(),
            ));
        }
        let j = self.segment_jacobian()?;
        // `J` scales the slope columns to physical DOFs, `J⁻²` maps the
        // reference second derivative to a physical one.
        let row = &self.rd.field_d2n_ref[g];
        scale_slope_slots_into(row, j, &mut out[..row.len()]);
        for v in &mut out[..row.len()] {
            *v /= j * j;
        }
        Ok(row.len())
    }

    /// The guard the field accessors share — see
    /// [`SubFiniteElementSpace`].
    fn reject_if_model_embedded(&self, what: &str) -> Result<()> {
        if self.rd.model_embedded {
            return Err(PyrucastError::Message(format!(
                "CellGeom: this subspace is MODEL_EMBEDDED — it declares no field basis, so it \
                 has no {what}. Its formulation owns the interpolation, so evaluating a field \
                 inside one of its elements is that formulation's business."
            )));
        }
        Ok(())
    }

    /// `J = ∂x/∂ξ` of a straight segment: **signed** in a 1-D space (where the
    /// slope degree of freedom is taken along the global axis, so a cell whose
    /// nodes run backwards must flip it), and the arc length `L/2` when the
    /// segment is embedded in a plane or in space — there the consumer has
    /// already rotated into a local axis running from node 0 to node 1.
    fn segment_jacobian(&self) -> Result<f64> {
        if self.rd.ref_dim != 1 {
            return Err(PyrucastError::Message(format!(
                "CellGeom: a C¹ basis needs a 1-D reference element, got ref_dim {}",
                self.rd.ref_dim
            )));
        }
        let (a, b) = (self.node_coord(0)?.to_vec(), self.node_coord(1)?.to_vec());
        if self.space_dim == 1 {
            return Ok((b[0] - a[0]) / 2.0);
        }
        let l2: f64 = (0..self.space_dim).map(|i| (b[i] - a[i]).powi(2)).sum();
        Ok(l2.sqrt() / 2.0)
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
    ///     geom.x_at_g(0, &mut x)?;
    ///     assert_eq!(x, [1.0, 0.0]);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn x_at_g(&self, g: usize, out: &mut [f64]) -> Result<()> {
        self.ensure_cell_coords()?;
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
        Ok(())
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
    ///         geom.x_at_g(0, &mut x)?;
    ///         assert_eq!(geom.radius(0)?, x[0]);
    ///         Ok(())
    ///     },
    /// )?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn radius(&self, g: usize) -> Result<f64> {
        if !self.axisymmetric {
            return Err(PyrucastError::Message(
                "CellGeom::radius: the geometry is not axisymmetric (build the Coords \
                 with Coords::axisymmetric)"
                    .into(),
            ));
        }
        self.ensure_cell_coords()?;
        let cc = self.cell_coords.borrow();
        Ok(Self::radius_from(
            cc.as_ref().unwrap(),
            &self.rd.n_ref[g],
            self.n_nodes,
            self.space_dim,
        ))
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
    ///     geom.tangents(0, &mut t)?;
    ///     assert!(t[..4].iter().any(|v| *v != 0.0));
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn tangents(&self, g: usize, out: &mut [f64]) -> Result<()> {
        self.ensure_cell_coords()?;
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
        Ok(())
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
    /// CellGeom::normal_from_tangents(&[2.0, 0.0], 1, 2, &mut nu)?;
    /// assert_eq!(nu, [0.0, -2.0]);
    /// // Une ou deux tangentes, pas davantage.
    /// assert!(CellGeom::normal_from_tangents(&[], 0, 2, &mut nu).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn normal_from_tangents(
        tangents: &[f64],
        n_tangents: usize,
        space_dim: usize,
        out: &mut [f64],
    ) -> Result<()> {
        let t = |k: usize, a: usize| tangents[k * space_dim + a];
        match n_tangents {
            1 => {
                out[0] = t(0, 1);
                out[1] = -t(0, 0);
            }
            2 => {
                out[0] = t(0, 1) * t(1, 2) - t(0, 2) * t(1, 1);
                out[1] = t(0, 2) * t(1, 0) - t(0, 0) * t(1, 2);
                out[2] = t(0, 0) * t(1, 1) - t(0, 1) * t(1, 0);
            }
            n => {
                return Err(PyrucastError::Message(format!(
                    "normal_from_tangents: a normal needs 1 tangent (2-D) or 2 (3-D), got {n}"
                )))
            }
        }
        Ok(())
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
        if r + 1 != d {
            return Err(PyrucastError::Message(format!(
                "CellGeom::normal: a {r}-D cell in a {d}-D space has no normal — a normal is \
                 defined on a boundary (a {}-D cell here)",
                d - 1
            )));
        }
        let mut tan = [0.0_f64; MAX_JACOBIAN];
        self.tangents(g, &mut tan)?;
        Self::normal_from_tangents(&tan, r, d, out)?;
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
    ///     let aire: f64 = (0..geom.n_gauss).map(|g| geom.det_j_w(g).unwrap()).sum();
    ///     assert!((aire - 2.0).abs() < 1e-12); // l'aire du triangle
    ///     Ok(())
    /// })?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn det_j_w(&self, g: usize) -> Result<f64> {
        self.ensure_cell_coords()?;
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
            return Ok(w);
        }
        let r = Self::radius_from(cc, &self.rd.n_ref[g], self.n_nodes, self.space_dim);
        Ok(w * std::f64::consts::TAU * r)
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
/// parallel loop below indexes without asking. An absent `prev`/`material`
/// yields an empty row, which a physics that declared none never indexes.
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
/// // lire la déformation et le matériau, écrire la contrainte.
/// let sortie = kernel::element_pointwise(
///     &zone, &entree, None, &mat_bidon, vec!["sig".into()],
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
    prev: Option<&Handle<SubElementField>>,
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
    let prev_guard = prev.map(|h| h.read());
    let mat_guard = material.read();

    let rd = RefData::snapshot(&fe)?;
    let n_gauss = rd.n_gauss;
    let n_cells = fe.cell_count()?;
    let conn: &[NodeId] = sm.connectivity();
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
    let (prev_vals, prev_stride) = match prev_guard.as_deref() {
        Some(p) => rows(p, "previous state", n_cells, n_gauss)?,
        None => (&[][..], 0),
    };
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
    let rd_ref: &RefData = &rd;
    let coords_ref: &Coords = &coords;

    // The components each zone carries, resolved **once** — the gather below
    // compares no name.
    let reads_idx = field.resolve_reads(reads);

    // The gather buffer is a fixed-size array, so a cell costs no allocation.
    // The bound is checked here, once: the largest element carries twenty nodes
    // (HEX20) and no producer reads more than three components per node.
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

/// The most values a per-cell buffer holds: twenty nodes (HEX20) times three
/// components — the widest gather, and the widest `dN/dx` a cell can have.
///
/// It sizes **stack** buffers, so a Gauss point costs no allocation.
pub(crate) const MAX_CELL_DOFS: usize = 64;

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
///     |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0)?; Ok(()) },
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
///     |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0)?; Ok(()) },
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
///         |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0)?; Ok(()) },
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
///         |geoms, _m, _s, ke| { ke[0] = geoms[0].det_j_w(0)?; Ok(()) },
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

    let n_cells = fe.cell_count()?;
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
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

    // Per cell, in parallel: compute the element matrix, then emit its triplets
    // in **local** indices. Order within a cell is li,di,lj,pj; cells are
    // concatenated in order below ⇒ identical triplet stream regardless of
    // thread count (bit-for-bit result).
    let per_cell: Vec<Vec<(usize, usize, f64)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize, f64)>> {
            let geoms: Vec<CellGeom> = rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, coords_ref, conn, cell))
                .collect();
            let mut ke = vec![0.0_f64; ke_len];
            element(&geoms, mat_ref, state_ref, &mut ke)?;

            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            let mut rpos = Vec::with_capacity(n_nodes);
            let mut cpos = Vec::with_capacity(n_nodes);
            for &nid in ids {
                rpos.push(*row_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_triplets: node {nid:?} not in row support"
                    ))
                })? as usize);
                cpos.push(*col_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_triplets: node {nid:?} not in col support"
                    ))
                })? as usize);
            }

            let mut trips = Vec::with_capacity(ke_len);
            for li in 0..n_nodes {
                for di in 0..n_dual {
                    let r = li * n_dual + di;
                    let ri = ordering.to_index(rpos[li], di, n_row_nodes, n_dual);
                    for lj in 0..n_nodes {
                        for pj in 0..n_primal {
                            let c = lj * n_primal + pj;
                            let ci = ordering.to_index(cpos[lj], pj, n_col_nodes, n_primal);
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

/// `(nrows, ncols, per-cell block-local (row, col) index pairs)` — the shape
/// returned by [`element_block_pattern`].
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
/// // Le **motif** seul, sans valeurs : c'est lui qui est mis en cache et
/// // réutilisé d'un assemblage à l'autre, la matière ne le changeant pas.
/// let (nr, nc, motif): kernel::BlockPattern = kernel::element_block_pattern(
///     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars,
/// )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(motif[0].len(), 3 * 3); // toutes les paires d'une maille
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type BlockPattern = (usize, usize, Vec<Vec<(usize, usize)>>);

/// The **symbolic** structure of a computed stiffness block: for each cell, the
/// block-**local** `(row, col)` index pairs it writes, in the exact order
/// [`element_block_triplets`] emits their values (`li, di, lj, pj`). Carries no
/// geometry and evaluates no kernel — only connectivity + the DOF `ordering` —
/// so an assembler can build the global CSR sparsity pattern (and, from it,
/// per-cell scatter slots) cheaply and cache it, then run the numeric kernel
/// only when values are needed. Returns `(nrows, ncols, per_cell_pairs)`.
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
/// // Le **motif** seul, sans valeurs : c'est lui qui est mis en cache et
/// // réutilisé d'un assemblage à l'autre, la matière ne le changeant pas.
/// let (nr, nc, motif): kernel::BlockPattern = kernel::element_block_pattern(
///     &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars,
/// )?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(motif[0].len(), 3 * 3); // toutes les paires d'une maille
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
    let n_cells = fe.cell_count()?;
    let n_nodes = conn.len().checked_div(n_cells).unwrap_or(0);

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

    let per_cell: Vec<Vec<(usize, usize)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize)>> {
            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            let mut rpos = Vec::with_capacity(n_nodes);
            let mut cpos = Vec::with_capacity(n_nodes);
            for &nid in ids {
                rpos.push(*row_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_pattern: node {nid:?} not in row support"
                    ))
                })? as usize);
                cpos.push(*col_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_pattern: node {nid:?} not in col support"
                    ))
                })? as usize);
            }
            let mut pairs = Vec::with_capacity(n_nodes * n_dual * n_nodes * n_primal);
            for li in 0..n_nodes {
                for di in 0..n_dual {
                    let ri = ordering.to_index(rpos[li], di, n_row_nodes, n_dual);
                    for lj in 0..n_nodes {
                        for pj in 0..n_primal {
                            let ci = ordering.to_index(cpos[lj], pj, n_col_nodes, n_primal);
                            pairs.push((ri, ci));
                        }
                    }
                }
            }
            Ok(pairs)
        })
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
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
    let (n_row_cells, n_col_cells) = (row_fe.cell_count()?, col_fe.cell_count()?);
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
///         ke[0] = lignes[0].det_j_w(0)?;
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
                    let ri = ordering.to_index(rl, di, n_row_support, n_dual);
                    for (lj, &cl) in cpos.iter().enumerate() {
                        for pj in 0..n_primal {
                            let c = lj * n_primal + pj;
                            let ci = ordering.to_index(cl, pj, n_col_support, n_primal);
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

/// The **symbolic** structure of an inter-mesh block — the coupling counterpart
/// of [`element_block_pattern`], carrying no geometry and evaluating no kernel.
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
/// // Un bloc **hors diagonale** : ses lignes vivent sur un maillage, ses
/// // colonnes sur celui d'en face. C'est ce qu'exige une loi d'interface.
/// // Ici les deux côtés sont le même, ce qui suffit à montrer la forme.
/// let (nr, nc, motif) = kernel::coupling_block_pattern(
///     &zone, &zone, &support, &support, 1, 1, DofOrdering::NodesThenVars)?;
/// assert_eq!((nr, nc), (3, 3));
/// assert_eq!(motif[0].len(), 3 * 3);
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
    let (n_cells, n_row_nodes_cell, n_col_nodes_cell) =
        check_conforming(&row_fe, &col_fe, row_conn, col_conn)?;

    let row_nodes: Vec<NodeId> = row_support.read().connectivity().to_vec();
    let col_nodes: Vec<NodeId> = col_support.read().connectivity().to_vec();
    let (n_row_support, n_col_support) = (row_nodes.len(), col_nodes.len());
    let (nrows, ncols) = (n_row_support * n_dual, n_col_support * n_primal);
    let row_pos = position_map(&row_nodes);
    let col_pos = position_map(&col_nodes);

    let per_cell: Vec<Vec<(usize, usize)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_row_nodes_cell.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize)>> {
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
            let mut pairs = Vec::with_capacity(rpos.len() * n_dual * cpos.len() * n_primal);
            for &rl in &rpos {
                for di in 0..n_dual {
                    let ri = ordering.to_index(rl, di, n_row_support, n_dual);
                    for &cl in &cpos {
                        for pj in 0..n_primal {
                            pairs.push((ri, ordering.to_index(cl, pj, n_col_support, n_primal)));
                        }
                    }
                }
            }
            Ok(pairs)
        })
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
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
fn local_positions(ids: &[NodeId], pos: &HashMap<NodeId, u32>, side: &str) -> Result<Vec<usize>> {
    ids.iter()
        .map(|nid| {
            pos.get(nid).map(|&p| p as usize).ok_or_else(|| {
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
///             *slot = geoms[0].det_j_w(0)? * (i as f64);
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

    let n_cells = fe.cell_count()?;
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let fe_len = n_nodes * n_dual;

    // Support slots: the unique nodes of `support` and each node's flat base.
    // `support` is the POI1 of the submesh, so it covers every connectivity node
    // and the map is total.
    let unique: Vec<NodeId> = support.read().connectivity().to_vec();
    let slot_of: HashMap<NodeId, usize> = unique.iter().enumerate().map(|(k, &n)| (n, k)).collect();

    // Cell colouring (cached on the primary FE subspace): two cells sharing a
    // node get different colours, so within a colour the cells scatter to
    // pairwise-disjoint nodes.
    let coloring = fe.coloring(|| coloring::greedy_color(n_cells, n_nodes, conn));

    // Fused compute + scatter, colour by colour: each cell builds its local
    // vector on a per-thread scratch buffer (no per-cell heap alloc, no
    // materialisation of the whole element set) and scatters it straight into the
    // node slots.
    let flat = colored_scatter(
        unique.len() * n_dual,
        coloring,
        (MIN_PARALLEL_LEN / n_nodes.max(1)).max(1),
        || vec![0.0_f64; fe_len],
        |cell, fe_cell, out| {
            let geoms: Vec<CellGeom> = rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, coords_ref, conn, cell))
                .collect();
            fe_cell.iter_mut().for_each(|v| *v = 0.0);
            element(&geoms, fe_cell)?;
            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            for (li, &nid) in ids.iter().enumerate() {
                let node_slot = *slot_of.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(
                        "scatter_to_nodes: support does not cover a cell node".into(),
                    )
                })?;
                let base = node_slot * n_dual;
                for di in 0..n_dual {
                    out.add(base + di, fe_cell[li * n_dual + di]);
                }
            }
            Ok(())
        },
    )?;

    let mut out = SubNodeField::from_poi1(support, dual_vars.clone())?;
    for (k, &nid) in unique.iter().enumerate() {
        for (di, name) in dual_vars.iter().enumerate() {
            out.set_value(nid, name, flat[k * n_dual + di])?;
        }
    }
    Ok(out)
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
///     (0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum()
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

    let n_cells = fe.cell_count()?;
    let n_nodes = rd.n_nodes;
    let conn: &[NodeId] = sm.connectivity();
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
            (0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum()
        })
        .unwrap();
        assert!((plane - 1.0).abs() < 1e-12, "plane area {plane} ≠ 1");

        let revolved = reduce_cells(&one_quad(1.0, 2.0, true), |geom| {
            (0..geom.n_gauss).map(|g| geom.det_j_w(g)).sum()
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
                let r = geom.radius(g)?;
                assert!(r > 0.0 && r < 1.0, "radius {r} outside (0, 1)");
                // radius is the first physical coordinate, and z stays in [0, 1].
                let mut x = [0.0_f64; 2];
                geom.x_at_g(g, &mut x)?;
                assert!((x[0] - r).abs() < 1e-15);
                assert!(x[1] > 0.0 && x[1] < 1.0);
            }
            Ok(0.0)
        })
        .unwrap();
    }

    /// `radius` is meaningless without the revolution hypothesis, and says so.
    #[test]
    fn radius_rejects_a_cartesian_geometry() {
        let err = reduce_cells(&one_quad(1.0, 2.0, false), |geom| geom.radius(0)).unwrap_err();
        assert!(format!("{err}").contains("not axisymmetric"));
    }
}
