//! `Element` — lightweight view on a single element of a
//! [`SubFiniteElementSpace`].
//!
//! It is to [`SubFiniteElementSpace`] what [`crate::atoms::Cell`]
//! is to [`crate::containers::mesh::SubMesh`] : a `(handle, cell_idx)`
//! pair that exposes the FE quantities (shape functions, Jacobian,
//! physical derivatives, …) for a single cell. Cloning an `Element` is
//! an `Arc` clone, so it is cheap to pass around and to create on the
//! fly inside an iterator.
//!
//! # Example
//!
//! ```
//! use pyrucast::coords::Coords;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::atoms::Node;
//! use pyrucast::handle::Handle;
//!
//! let coords = Handle::new(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//!
//! for el in fes.elements(0).unwrap() {
//!     // |J| of a SEG2 of length 2 in 1-D is constant = 1.
//!     for g in 0..el.gauss_count() {
//!         assert!((el.det_jacobian(g).unwrap() - 1.0).abs() < 1e-12);
//!     }
//! }
//! ```

use std::fmt;

use crate::atoms::Cell;
use crate::atoms::NodeId;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// Lightweight view on a single element of a [`SubFiniteElementSpace`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Element, ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
/// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let el = Element::new(zone.clone(), 0).unwrap();
/// // Une vue sur **une** maille de la zone : elle donne accès aux formes,
/// // au jacobien et à B sans jamais matérialiser de tableau par maille.
/// for el in fes.elements(0)? {
///     assert!((el.det_jacobian(0)? - 4.0).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone)]
pub struct Element {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    idx: usize,
}

impl Element {
    /// Build an element view. Errors if `idx ≥ cell_count()`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // Une vue légère : elle ne copie rien, elle tient la zone par son handle.
    /// let el = Element::new(zone.clone(), 0)?;
    /// assert_eq!(el.index(), 0);
    /// assert!(Element::new(zone.clone(), 7).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, idx: usize) -> Result<Self> {
        let n = fespace.read().cell_count()?;
        if idx >= n {
            return Err(PyrucastError::Message(format!(
                "element index {idx} out of range (cell_count={n})"
            )));
        }
        Ok(Self { fespace, idx })
    }

    /// Index of this element inside its parent [`SubFiniteElementSpace`].
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.index(), 0); // rang de la maille dans sa zone
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Handle to the parent [`SubFiniteElementSpace`] (internal clone).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// # use pyrucast::handle::Handle as H;
    /// assert!(H::same_object(&el.fespace(), &zone)); // partagé, pas copié
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// Underlying [`Cell`] view in the parent submesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // La maille géométrique sous-jacente — la connectivité, sans les
    /// // fonctions de forme.
    /// assert_eq!(el.cell()?.node_ids()?.len(), 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn cell(&self) -> Result<Cell> {
        let sm = self.fespace.read().submesh();
        Cell::new(sm, self.idx)
    }

    // ── Structural accessors ────────────────────────────────────────────

    /// Number of nodes per element (= element type's nodes_per_cell).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.nodes_per_cell()?, 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn nodes_per_cell(&self) -> Result<usize> {
        self.fespace.read().nodes_per_cell()
    }

    /// Geometric (physical) dimension of the underlying `Coords`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.space_dim()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn space_dim(&self) -> Result<usize> {
        Ok(self.fespace.read().space_dim())
    }

    /// Reference dimension (= topological dim of the element type).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // Un triangle est de dimension 2 dans son élément de référence ; il
    /// // pourrait vivre dans un espace 3-D (coque) sans que cela change.
    /// assert_eq!(el.ref_dim()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn ref_dim(&self) -> Result<usize> {
        self.fespace.read().ref_dim()
    }

    /// Number of Gauss points per element.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.gauss_count(), zone.read().gauss_count());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gauss_count(&self) -> usize {
        // Static across the fespace; cheap to look up under the guard.
        self.fespace.read().gauss_count()
    }

    /// Connectivity (node ids) of this element.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.node_ids()?, vec![n[0].id(), n[1].id(), n[2].id()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn node_ids(&self) -> Result<Vec<NodeId>> {
        self.cell()?.node_ids()
    }

    // ── Reference-space accessors (shared with the FE space) ────────────

    /// Reference coordinates of the `g`-th Gauss point.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.gauss_xi(0)?.len(), el.ref_dim()?);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gauss_xi(&self, g: usize) -> Result<Vec<f64>> {
        Ok(self.fespace.read().gauss_xi(g)?.to_vec())
    }

    /// Weight of the `g`-th Gauss point.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // Les poids somment à l'aire de référence — 1/2 pour un triangle.
    /// let total: f64 = (0..el.gauss_count()).map(|g| el.gauss_weight(g).unwrap()).sum();
    /// assert!((total - 0.5).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gauss_weight(&self, g: usize) -> Result<f64> {
        self.fespace.read().gauss_weight(g)
    }

    /// `N_i(ξ_g)` for all nodes `i` at the `g`-th Gauss point.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // Partition de l'unité.
    /// assert!((el.n_at_g(0)?.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn n_at_g(&self, g: usize) -> Result<Vec<f64>> {
        Ok(self.fespace.read().n_at_g(g)?.to_vec())
    }

    /// `∂N_i/∂ξ_j(ξ_g)` for all nodes at the `g`-th Gauss point
    /// (flat row-major `[i * ref_dim + j]`).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// assert_eq!(el.dn_at_g(0)?.len(), 3 * 2); // ∂N_i/∂ξ_k à plat
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn dn_at_g(&self, g: usize) -> Result<Vec<f64>> {
        Ok(self.fespace.read().dn_at_g(g)?.to_vec())
    }

    // ── Physical quantities (cell-specific, on-the-fly) ─────────────────

    /// Jacobian `J = ∂x/∂ξ` at the `g`-th Gauss point of this element.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // Le triangle (0,0), (2,0), (0,2) : J = 2·I, à plat en ligne-major.
    /// assert_eq!(el.jacobian(0)?, vec![2.0, 0.0, 0.0, 2.0]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn jacobian(&self, g: usize) -> Result<Vec<f64>> {
        self.fespace.read().jacobian(self.idx, g)
    }

    /// `|J|` (measure scaling factor) at the `g`-th Gauss point.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // |J| = 4 en tout point : le mapping est affine.
    /// assert!((el.det_jacobian(0)? - 4.0).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn det_jacobian(&self, g: usize) -> Result<f64> {
        self.fespace.read().det_jacobian(self.idx, g)
    }

    /// Physical derivatives `∂N_i/∂x_a` at the `g`-th Gauss point
    /// (flat row-major `[i * space_dim + a]`).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{Element, ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let el = Element::new(zone.clone(), 0).unwrap();
    /// // La matrice B, calculée à la volée. Ses lignes somment au vecteur nul :
    /// // la partition de l'unité, dérivée.
    /// let b = el.dn_dx(0)?;
    /// assert!((b[0] + b[2] + b[4]).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn dn_dx(&self, g: usize) -> Result<Vec<f64>> {
        self.fespace.read().dn_dx(self.idx, g)
    }
}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Element").field("idx", &self.idx).finish()
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cell() {
            Ok(cell) => {
                let et = cell.element_type().map(|e| e.name()).unwrap_or("?");
                write!(f, "Element<{}> #{}", et, self.idx)
            }
            Err(_) => write!(f, "Element #{}", self.idx),
        }
    }
}

impl crate::dump::Dump for Element {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        match self.cell() {
            Ok(cell) => format!(
                "Element #{} ({} Gauss point(s))\n{}",
                self.idx,
                self.gauss_count(),
                crate::dump::Dump::render(&cell, opts)
            ),
            Err(e) => format!("Element #{} <{e}>", self.idx),
        }
    }
}

/// Iterator over the elements of a single [`SubFiniteElementSpace`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Element, ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
/// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let el = Element::new(zone.clone(), 0).unwrap();
/// // Ce que rend `FiniteElementSpace::elements` : un parcours de la zone,
/// // maille par maille, sans allocation par élément.
/// let els: Vec<_> = fes.elements(0)?.collect();
/// assert_eq!(els.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone)]
pub struct ElementIter {
    fespace: Handle<SubFiniteElementSpace>,
    next: usize,
    end: usize,
}

impl ElementIter {
    pub(crate) fn new(fespace: Handle<SubFiniteElementSpace>, end: usize) -> Self {
        Self {
            fespace,
            next: 0,
            end,
        }
    }
}

impl Iterator for ElementIter {
    type Item = Element;
    fn next(&mut self) -> Option<Element> {
        if self.next < self.end {
            let el = Element {
                fespace: self.fespace.clone(),
                idx: self.next,
            };
            self.next += 1;
            Some(el)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ElementIter {}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn seg2_fes() -> (Handle<Coords>, Vec<Node>, FiniteElementSpace) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        (coords, vec![n0, n1, n2], fes)
    }

    #[test]
    fn element_exposes_cell_and_physical_quantities() {
        let (_cfg, nodes, fes) = seg2_fes();
        let el = fes.element(0, 0).unwrap();

        assert_eq!(el.index(), 0);
        assert_eq!(el.nodes_per_cell().unwrap(), 2);
        assert_eq!(el.node_ids().unwrap(), vec![nodes[0].id(), nodes[1].id()]);

        let cell = el.cell().unwrap();
        assert_eq!(cell.element_type().unwrap(), ElementType::SEG2);

        for g in 0..el.gauss_count() {
            // SEG2 of length 1 in 1-D → |J| = 0.5
            assert!((el.det_jacobian(g).unwrap() - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn elements_iterator_yields_all_elements_in_order() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let elements: Vec<Element> = fes.elements(0).unwrap().collect();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].index(), 0);
        assert_eq!(elements[1].index(), 1);
    }

    #[test]
    fn element_new_out_of_range_errors() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let sub = fes.get(0).unwrap();
        assert!(Element::new(sub, 5).is_err());
    }

    #[test]
    fn fespace_element_by_indices() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let el = fes.element(0, 1).unwrap();
        assert_eq!(el.index(), 1);
        // submesh out of bounds
        assert!(fes.element(1, 0).is_err());
        // cell out of bounds
        assert!(fes.element(0, 7).is_err());
    }
}
