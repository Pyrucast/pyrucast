//! Mesh — collection of homogeneous submeshes (one element type per
//! submesh).
//!
//! Hierarchy:
//!
//! - [`SubMesh`] — every cell of a single [`ElementType`]. Stores the
//!   connectivity flat (`Vec<NodeId>`, length `cell_count * nodes_per_cell`).
//!   RAII referencing: `add_cell` increments the node refcounts in the
//!   `Configuration`; the `SubMesh`'s `Drop` decrements every referenced
//!   node.
//! - [`Mesh`] — aggregate of SubMeshes attached to the same `Configuration`.
//!
//! The POI1 case is deliberately degenerate: a POI1 submesh is exactly a
//! list of nodes.
//!
//! # Example
//!
//! ```
//! use pyrucast::configuration::Configuration;
//! use pyrucast::element_type::ElementType;
//! use pyrucast::mesh::SubMesh;
//! use pyrucast::node::Node;
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//! assert_eq!(sm.cell_count(), 1);
//!
//! // The SubMesh holds refs on the 3 nodes, in addition to the `Node`s.
//! with(&cfg, |c| assert_eq!(c.refcount(a.id()), 2)).unwrap();
//! drop(sm);  // decrements the referenced nodes
//! with(&cfg, |c| assert_eq!(c.refcount(a.id()), 1)).unwrap();
//! ```

use crate::color::RgbColor;
use crate::configuration::{Configuration, NodeId};
use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::node::Node;
use crate::store::{insert, with, with_mut, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SubMesh ────────────────────────────────────────────────────────────────

/// Submesh: every cell of a single [`ElementType`].
///
/// The connectivity is stored flat; each cell occupies
/// `element_type.nodes_per_cell()` contiguous entries.
///
/// A [`RgbColor`] is attached as the **face colour** used by the
/// visualization layer (`viz` feature); it has no effect on numerics and
/// defaults to a light blue.
#[derive(Serialize, Deserialize)]
pub struct SubMesh {
    element_type: ElementType,
    config: Handle<Configuration>,
    /// Flat connectivity: cell `i` occupies `[i*npc, (i+1)*npc)`.
    connectivity: Vec<NodeId>,
    /// Face colour used by the viz layer. `serde(default)` keeps older
    /// snapshots (without the field) readable.
    #[serde(default)]
    face_color: RgbColor,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `config`.
    pub fn new(config: Handle<Configuration>, element_type: ElementType) -> Self {
        Self {
            element_type,
            config,
            connectivity: Vec::new(),
            face_color: RgbColor::default(),
        }
    }

    /// Face colour used when this submesh is drawn (no numerical effect).
    pub fn face_color(&self) -> RgbColor {
        self.face_color
    }

    /// Replace the face colour used when this submesh is drawn.
    pub fn set_face_color(&mut self, color: RgbColor) {
        self.face_color = color;
    }

    /// Add a cell. The length of `nodes` must equal
    /// `element_type.nodes_per_cell()`, and each node must be alive in the
    /// `Configuration`; each node is increfed. On increment failure
    /// (invalid / collected id), the increfs already performed for this
    /// cell are rolled back.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        let npc = self.element_type.nodes_per_cell();
        if nodes.len() != npc {
            return Err(PyrucastError::Message(format!(
                "add_cell({}): expected {} nodes, got {}",
                self.element_type,
                npc,
                nodes.len()
            )));
        }
        let result: Result<()> = with_mut(&self.config, |c| {
            let mut acquired = 0usize;
            for &n in nodes {
                if let Err(e) = c.incref(n) {
                    // Roll back the increfs already done for this cell.
                    for &m in &nodes[..acquired] {
                        let _ = c.decref(m);
                    }
                    return Err(e);
                }
                acquired += 1;
            }
            Ok(())
        })?;
        result?;
        let idx = self.connectivity.len() / npc;
        self.connectivity.extend_from_slice(nodes);
        Ok(idx)
    }

    /// Element type of the submesh.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Number of cells in the submesh.
    pub fn cell_count(&self) -> usize {
        self.connectivity.len() / self.element_type.nodes_per_cell()
    }

    /// Flat connectivity buffer (all cells concatenated).
    pub(crate) fn connectivity(&self) -> &[NodeId] {
        &self.connectivity
    }

    /// Handle to the owning `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.config.clone()
    }

    /// Visualize this submesh.
    ///
    /// - `view = None` ⇒ [`crate::viz::View::default`] (isometric).
    /// - `save = None` ⇒ open an interactive window (requires feature
    ///   `viz-interactive`).
    /// - `save = Some(path)` ⇒ write an image file; the format is inferred
    ///   from the extension (`.png` or `.svg`).
    ///
    /// Only `TRI3` submeshes are supported for now; calling this on
    /// another element type returns a clear error.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
    ) -> Result<()> {
        crate::viz::render(self, view, save)
    }
}

impl Drop for SubMesh {
    fn drop(&mut self) {
        // One lock acquisition for all decrefs.
        let _ = with_mut(&self.config, |c| {
            for &n in &self.connectivity {
                let _ = c.decref(n);
            }
        });
    }
}

impl fmt::Debug for SubMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubMesh")
            .field("element_type", &self.element_type)
            .field("cell_count", &self.cell_count())
            .finish()
    }
}

impl fmt::Display for SubMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubMesh<{}>: {} cell(s)",
            self.element_type,
            self.cell_count()
        )
    }
}

// ─── Mesh ───────────────────────────────────────────────────────────────────

/// Mesh: aggregate of submeshes attached to the same `Configuration`.
#[derive(Serialize, Deserialize)]
pub struct Mesh {
    config: Handle<Configuration>,
    submeshes: Vec<Handle<SubMesh>>,
}

impl Mesh {
    /// Create an empty mesh attached to `config`.
    pub fn new(config: Handle<Configuration>) -> Self {
        Self {
            config,
            submeshes: Vec::new(),
        }
    }

    /// Add a submesh. Requires that the submesh's `Configuration` matches
    /// the mesh's.
    pub fn add_submesh(&mut self, sm: Handle<SubMesh>) -> Result<()> {
        let sm_cfg = with(&sm, |s| s.configuration())?;
        if sm_cfg.index() != self.config.index() || sm_cfg.generation() != self.config.generation()
        {
            return Err(PyrucastError::Message(
                "add_submesh: submesh attached to a different Configuration".into(),
            ));
        }
        self.submeshes.push(sm);
        Ok(())
    }

    /// Number of submeshes.
    pub fn submesh_count(&self) -> usize {
        self.submeshes.len()
    }

    /// Return a clone of the handle to the submesh at index `idx`.
    pub fn submesh(&self, idx: usize) -> Result<Handle<SubMesh>> {
        self.submeshes
            .get(idx)
            .cloned()
            .ok_or_else(|| PyrucastError::Message(format!("submesh: index {} out of bounds", idx)))
    }

    /// Total cells in the mesh (sum across submeshes).
    pub fn cell_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for sm in &self.submeshes {
            total += with(sm, |s| s.cell_count())?;
        }
        Ok(total)
    }

    /// Handle to the `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.config.clone()
    }

    /// Create a mesh pre-loaded with one empty submesh of `element_type`.
    pub fn with_element_type(config: Handle<Configuration>, element_type: ElementType) -> Self {
        let sm = insert(SubMesh::new(config.clone(), element_type));
        let mut mesh = Self {
            config,
            submeshes: Vec::new(),
        };
        mesh.submeshes.push(sm);
        mesh
    }

    /// Add a cell directly when the mesh has exactly one submesh.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        if self.submeshes.len() != 1 {
            return Err(PyrucastError::Message(
                "add_cell: mesh must have exactly one submesh".into(),
            ));
        }
        with_mut(&self.submeshes[0], |s| s.add_cell(nodes))?
    }

    /// Element type of each submesh, in order.
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.submeshes
            .iter()
            .map(|sm| with(sm, |s| s.element_type()))
            .collect()
    }

    /// Cell count of each submesh, in order.
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.submeshes
            .iter()
            .map(|sm| with(sm, |s| s.cell_count()))
            .collect()
    }

    /// Node at position `node_idx` in cell `cell_idx` of submesh `submesh_idx`.
    pub fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> Result<Node> {
        let sm = self.submesh(submesh_idx)?;
        let nid: NodeId = with(&sm, |s| {
            let npc = s.element_type.nodes_per_cell();
            let n = s.cell_count();
            if cell_idx >= n {
                return Err(PyrucastError::Message(format!(
                    "node: cell index {} ≥ cell_count {}",
                    cell_idx, n
                )));
            }
            s.connectivity()
                .get(cell_idx * npc + node_idx)
                .copied()
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "node: node index {} ≥ nodes_per_cell {}",
                        node_idx, npc
                    ))
                })
        })??;
        Node::acquire(self.config.clone(), nid)
    }

    /// Create a POI1 mesh containing all live nodes of `config`.
    pub fn from_live_nodes(config: Handle<Configuration>) -> Result<Mesh> {
        let node_ids: Vec<NodeId> = with(&config, |c| c.iter_live().collect())?;
        let mut mesh = Mesh::with_element_type(config, ElementType::POI1);
        for nid in node_ids {
            mesh.add_cell(&[nid])?;
        }
        Ok(mesh)
    }

    /// Build a mesh of `n_elems` SEG2 elements along the straight line from
    /// node `a` to node `b`.
    ///
    /// Both nodes must belong to the same `Configuration` and have the same
    /// coordinate dimension. `n_elems` must be ≥ 1.
    ///
    /// The two endpoint nodes are re-used (their refcount is incremented);
    /// `n_elems - 1` intermediate nodes are created at evenly spaced positions.
    pub fn line_seg2(a: &Node, b: &Node, n_elems: usize) -> Result<Mesh> {
        if n_elems == 0 {
            return Err(PyrucastError::Message(
                "line_seg2: n_elems must be ≥ 1".into(),
            ));
        }
        let cfg = a.configuration();
        let cfg_b = b.configuration();
        if cfg.index() != cfg_b.index() || cfg.generation() != cfg_b.generation() {
            return Err(PyrucastError::Message(
                "line_seg2: nodes belong to different Configurations".into(),
            ));
        }
        let coords_a = a.coord()?;
        let coords_b = b.coord()?;
        if coords_a.len() != coords_b.len() {
            return Err(PyrucastError::Message(
                "line_seg2: nodes have incompatible dimensions".into(),
            ));
        }

        // n_elems+1 nodes: a, n_elems-1 intermediate, b.
        let mut nodes: Vec<Node> = Vec::with_capacity(n_elems + 1);
        nodes.push(Node::acquire(cfg.clone(), a.id())?);
        for i in 1..n_elems {
            let t = i as f64 / n_elems as f64;
            let coords: Vec<f64> = coords_a
                .iter()
                .zip(coords_b.iter())
                .map(|(&ca, &cb)| ca + t * (cb - ca))
                .collect();
            nodes.push(Node::create_in(cfg.clone(), &coords)?);
        }
        nodes.push(Node::acquire(cfg.clone(), b.id())?);

        let mut mesh = Mesh::with_element_type(cfg, ElementType::SEG2);
        for i in 0..n_elems {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
        }
        Ok(mesh)
    }

    /// Build a closed circle of `n_elems` SEG2 elements.
    ///
    /// The circle lies in the plane perpendicular to `normal`, centred on
    /// `center`, with the given `radius`. `normal` must be a 3-component
    /// vector (regardless of node dimension); for 2-D meshes the normal
    /// should point along z (`[0, 0, ±1]`).
    ///
    /// `n_elems` must be ≥ 3 and `radius` must be > 0. `n_elems` nodes are
    /// created at evenly spaced angles; the center node itself is **not**
    /// included in the mesh. The first and last elements share node 0,
    /// closing the loop.
    ///
    /// The in-plane basis `(u, v)` is built by Gram-Schmidt against the
    /// least-aligned coordinate axis so that `(u, v, n̂)` is right-handed.
    pub fn circle_seg2(
        center: &Node,
        normal: &[f64],
        radius: f64,
        n_elems: usize,
    ) -> Result<Mesh> {
        use std::f64::consts::PI;

        if n_elems < 3 {
            return Err(PyrucastError::Message(
                "circle_seg2: n_elems must be ≥ 3".into(),
            ));
        }
        if radius <= 0.0 {
            return Err(PyrucastError::Message(
                "circle_seg2: radius must be > 0".into(),
            ));
        }
        if normal.len() != 3 {
            return Err(PyrucastError::Message(
                "circle_seg2: normal must have exactly 3 components".into(),
            ));
        }

        let cfg = center.configuration();
        let center_coords = center.coord()?;
        let dim = center_coords.len();
        if !(2..=3).contains(&dim) {
            return Err(PyrucastError::Message(
                "circle_seg2: node dimension must be 2 or 3".into(),
            ));
        }

        use crate::triangulation::{in_plane_basis, Vector3};
        let n_vec = Vector3::new(normal[0], normal[1], normal[2]);
        if n_vec.norm() < 1e-15 {
            return Err(PyrucastError::Message(
                "circle_seg2: normal vector must not be zero".into(),
            ));
        }
        let n = n_vec.normalize();
        let (u, v) = in_plane_basis(n);

        // Center as 3-D point (zero-padded if the node lives in 2-D).
        let centre = Vector3::new(
            center_coords.first().copied().unwrap_or(0.0),
            center_coords.get(1).copied().unwrap_or(0.0),
            center_coords.get(2).copied().unwrap_or(0.0),
        );
        let mut nodes: Vec<Node> = Vec::with_capacity(n_elems);
        for i in 0..n_elems {
            let theta = 2.0 * PI * i as f64 / n_elems as f64;
            let p3 = centre + radius * (theta.cos() * u + theta.sin() * v);
            nodes.push(Node::create_in(cfg.clone(), &p3.as_slice()[..dim])?);
        }

        // Closed loop: element i connects node i to node (i+1) % n_elems.
        let mut mesh = Mesh::with_element_type(cfg, ElementType::SEG2);
        for i in 0..n_elems {
            mesh.add_cell(&[nodes[i].id(), nodes[(i + 1) % n_elems].id()])?;
        }
        Ok(mesh)
    }

    /// Sweep two SEG2 meshes into a QUA4 mesh by building `n_layers` layers.
    ///
    /// Both meshes must be single-submesh SEG2 meshes with the same number of
    /// elements, attached to the same `Configuration`. `n_layers` must be ≥ 1.
    ///
    /// Column `j` of `mesh_a` is linearly interpolated with column `j` of
    /// `mesh_b` to produce the intermediate layers. Endpoint nodes from both
    /// meshes are re-used (refcount incremented); intermediate nodes are
    /// created at evenly spaced positions.
    ///
    /// QUA4 node order per element (counterclockwise, `mesh_a` side first):
    /// `[k][j]`, `[k][j+1]`, `[k+1][j+1]`, `[k+1][j]`.
    pub fn sweep_qua4(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
        if n_layers == 0 {
            return Err(PyrucastError::Message(
                "sweep_qua4: n_layers must be ≥ 1".into(),
            ));
        }
        if mesh_a.submesh_count() != 1 {
            return Err(PyrucastError::Message(
                "sweep_qua4: mesh_a must have exactly one submesh".into(),
            ));
        }
        if mesh_b.submesh_count() != 1 {
            return Err(PyrucastError::Message(
                "sweep_qua4: mesh_b must have exactly one submesh".into(),
            ));
        }
        if mesh_a.config.index() != mesh_b.config.index()
            || mesh_a.config.generation() != mesh_b.config.generation()
        {
            return Err(PyrucastError::Message(
                "sweep_qua4: meshes are attached to different Configurations".into(),
            ));
        }

        let sm_a = mesh_a.submesh(0)?;
        let sm_b = mesh_b.submesh(0)?;

        let (et_a, n_elems, conn_a) =
            with(&sm_a, |s| (s.element_type(), s.cell_count(), s.connectivity().to_vec()))?;
        let (et_b, n_elems_b, conn_b) =
            with(&sm_b, |s| (s.element_type(), s.cell_count(), s.connectivity().to_vec()))?;

        if et_a != ElementType::SEG2 {
            return Err(PyrucastError::Message(
                "sweep_qua4: mesh_a must be a SEG2 mesh".into(),
            ));
        }
        if et_b != ElementType::SEG2 {
            return Err(PyrucastError::Message(
                "sweep_qua4: mesh_b must be a SEG2 mesh".into(),
            ));
        }
        if n_elems != n_elems_b {
            return Err(PyrucastError::Message(format!(
                "sweep_qua4: mesh_a has {} elements but mesh_b has {}",
                n_elems, n_elems_b
            )));
        }

        let cfg = mesh_a.config.clone();
        let n_cols = n_elems + 1;

        // Column j: first node of elem 0 (j=0), or second node of elem j-1 (j≥1).
        let col_ids_a: Vec<NodeId> = std::iter::once(conn_a[0])
            .chain((1..=n_elems).map(|j| conn_a[2 * j - 1]))
            .collect();
        let col_ids_b: Vec<NodeId> = std::iter::once(conn_b[0])
            .chain((1..=n_elems).map(|j| conn_b[2 * j - 1]))
            .collect();

        let coords_a: Vec<Vec<f64>> = col_ids_a
            .iter()
            .map(|&id| -> Result<Vec<f64>> {
                with(&cfg, |c| c.coord(id).map(|s| s.to_vec()))?
            })
            .collect::<Result<_>>()?;
        let coords_b: Vec<Vec<f64>> = col_ids_b
            .iter()
            .map(|&id| -> Result<Vec<f64>> {
                with(&cfg, |c| c.coord(id).map(|s| s.to_vec()))?
            })
            .collect::<Result<_>>()?;

        // layers[k][j] = Node at layer k, column j.
        // Layer 0 = re-acquired mesh_a nodes; layer n_layers = re-acquired mesh_b nodes.
        let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n_layers + 1);

        layers.push(
            col_ids_a
                .iter()
                .map(|&id| Node::acquire(cfg.clone(), id))
                .collect::<Result<Vec<_>>>()?,
        );
        for k in 1..n_layers {
            let t = k as f64 / n_layers as f64;
            let layer: Vec<Node> = (0..n_cols)
                .map(|j| {
                    let coords: Vec<f64> = coords_a[j]
                        .iter()
                        .zip(coords_b[j].iter())
                        .map(|(&ca, &cb)| ca + t * (cb - ca))
                        .collect();
                    Node::create_in(cfg.clone(), &coords)
                })
                .collect::<Result<_>>()?;
            layers.push(layer);
        }
        layers.push(
            col_ids_b
                .iter()
                .map(|&id| Node::acquire(cfg.clone(), id))
                .collect::<Result<Vec<_>>>()?,
        );

        let mut mesh = Mesh::with_element_type(cfg, ElementType::QUA4);
        for k in 0..n_layers {
            for j in 0..n_elems {
                mesh.add_cell(&[
                    layers[k][j].id(),
                    layers[k][j + 1].id(),
                    layers[k + 1][j + 1].id(),
                    layers[k + 1][j].id(),
                ])?;
            }
        }
        Ok(mesh)
    }

    /// Extrude a mesh by `n_layers` layers along `direction`.
    ///
    /// `direction` is the **total** displacement vector; each intermediate
    /// layer is placed at an evenly spaced fraction. Supported element types:
    /// SEG2 → QUA4, QUA4 → HEX8. Other types produce an error.
    ///
    /// Nodes shared between cells in the source mesh remain shared in the
    /// extruded mesh. Source nodes are re-used (refcount incremented);
    /// intermediate layer nodes are newly created.
    ///
    /// Node ordering:
    /// - QUA4: `bot[0], bot[1], top[1], top[0]`
    /// - HEX8: `bot[0..4], top[0..4]`
    pub fn extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Result<Mesh> {
        if n_layers == 0 {
            return Err(PyrucastError::Message(
                "extrude: n_layers must be ≥ 1".into(),
            ));
        }

        let cfg = mesh.config.clone();

        // Collect unique NodeIds across all submeshes, first-seen order.
        let mut col_map: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();
        let mut ordered_ids: Vec<NodeId> = Vec::new();
        for sm in &mesh.submeshes {
            for id in with(sm, |s| s.connectivity().to_vec())? {
                col_map.entry(id).or_insert_with(|| {
                    let col = ordered_ids.len();
                    ordered_ids.push(id);
                    col
                });
            }
        }

        if ordered_ids.is_empty() {
            return Err(PyrucastError::Message("extrude: mesh has no cells".into()));
        }

        let base_coords: Vec<Vec<f64>> = ordered_ids
            .iter()
            .map(|&id| -> Result<Vec<f64>> {
                with(&cfg, |c| c.coord(id).map(|s| s.to_vec()))?
            })
            .collect::<Result<_>>()?;

        let coord_dim = base_coords[0].len();
        if direction.len() != coord_dim {
            return Err(PyrucastError::Message(format!(
                "extrude: direction has {} components but node dimension is {}",
                direction.len(),
                coord_dim
            )));
        }

        let step: Vec<f64> = direction.iter().map(|&d| d / n_layers as f64).collect();
        let n_cols = ordered_ids.len();

        // layers[k][col] = Node at layer k, column col.
        let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n_layers + 1);

        layers.push(
            ordered_ids
                .iter()
                .map(|&id| Node::acquire(cfg.clone(), id))
                .collect::<Result<Vec<_>>>()?,
        );
        for k in 1..=n_layers {
            let layer: Vec<Node> = (0..n_cols)
                .map(|j| {
                    let coords: Vec<f64> = base_coords[j]
                        .iter()
                        .zip(step.iter())
                        .map(|(&c, &s)| c + k as f64 * s)
                        .collect();
                    Node::create_in(cfg.clone(), &coords)
                })
                .collect::<Result<_>>()?;
            layers.push(layer);
        }

        let col = |id: NodeId| *col_map.get(&id).unwrap();

        let mut result = Mesh::new(cfg.clone());

        for sm_handle in &mesh.submeshes {
            let (et, n_cells, conn) = with(sm_handle, |s| {
                (s.element_type(), s.cell_count(), s.connectivity().to_vec())
            })?;
            let npc = et.nodes_per_cell();

            let extruded_et = match et {
                ElementType::SEG2 => ElementType::QUA4,
                ElementType::QUA4 => ElementType::HEX8,
                _ => {
                    return Err(PyrucastError::Message(format!(
                        "extrude: cannot extrude {} elements (supported: SEG2, QUA4)",
                        et
                    )))
                }
            };

            let mut sm_out = SubMesh::new(cfg.clone(), extruded_et);

            for k in 0..n_layers {
                for ci in 0..n_cells {
                    let cell = &conn[ci * npc..(ci + 1) * npc];
                    let bot: Vec<NodeId> =
                        cell.iter().map(|&id| layers[k][col(id)].id()).collect();
                    let top: Vec<NodeId> =
                        cell.iter().map(|&id| layers[k + 1][col(id)].id()).collect();

                    match et {
                        ElementType::SEG2 => {
                            sm_out.add_cell(&[bot[0], bot[1], top[1], top[0]])?;
                        }
                        ElementType::QUA4 => {
                            sm_out.add_cell(&[
                                bot[0], bot[1], bot[2], bot[3],
                                top[0], top[1], top[2], top[3],
                            ])?;
                        }
                        _ => unreachable!(),
                    }
                }
            }

            result.add_submesh(insert(sm_out))?;
        }

        Ok(result)
    }

    /// Fill the interior of one or more closed SEG2 contours with 2-D elements.
    ///
    /// `contour` must be a [`Mesh`] with **one or more** SEG2 submeshes.
    /// Each submesh is treated as a single closed simple loop (each node
    /// appears once as the start of a segment and once as its end). The
    /// `Configuration` can be either:
    /// - **2-D** — points are used directly,
    /// - **3-D** — every loop must be (nearly) co-planar; an in-plane
    ///   basis is computed by Newell's method and the points are
    ///   projected onto the best-fit plane before triangulation. The
    ///   maximum signed distance from any node to that plane must not
    ///   exceed `1e-6 × diag`, where `diag` is the AABB diagonal of the
    ///   union of all loops; otherwise the call fails with a clear error.
    ///
    /// When more than one loop is provided, the **outer boundary** is
    /// detected automatically as the loop with the largest signed area
    /// (after 2-D projection if needed); the remaining loops are treated
    /// as **holes**. Orientation does not matter — every loop is
    /// internally re-oriented as needed.
    ///
    /// `element_type` selects the 2-D element to fill with. **Only**
    /// [`ElementType::TRI3`] is currently supported; passing any other
    /// type returns a clear error.
    ///
    /// The returned mesh has a single submesh of `element_type`, sharing
    /// the contour's `Configuration`: the existing contour nodes are
    /// re-used (their refcount is incremented). No new nodes (Steiner
    /// points) are created in this iteration.
    ///
    /// Algorithm:
    /// - **single loop, no holes** — fast path using plain ear clipping;
    ///   produces exactly `n - 2` triangles for `n` contour nodes.
    /// - **multiple loops (outer + holes)** — constrained Delaunay
    ///   triangulation (Bowyer-Watson + edge enforcement + parity
    ///   flood-fill across constrained edges).
    ///
    /// Triangles are oriented **CCW** in the projection plane regardless
    /// of the input contour's orientation.
    ///
    /// # Example
    /// ```
    /// use pyrucast::configuration::Configuration;
    /// use pyrucast::element_type::ElementType;
    /// use pyrucast::mesh::Mesh;
    /// use pyrucast::node::Node;
    /// use pyrucast::store::insert;
    ///
    /// // Unit square contour, CCW.
    /// let cfg = insert(Configuration::new(2).unwrap());
    /// let n = [
    ///     Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap(),
    ///     Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap(),
    ///     Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap(),
    ///     Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap(),
    /// ];
    /// let mut contour = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
    /// for i in 0..4 {
    ///     contour.add_cell(&[n[i].id(), n[(i + 1) % 4].id()]).unwrap();
    /// }
    ///
    /// let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
    /// assert_eq!(tri.cell_count().unwrap(), 2); // 4 - 2 = 2 triangles
    /// ```
    pub fn fill_surface(
        contour: &Mesh,
        element_type: ElementType,
        refinement: Option<crate::triangulation::RefinementOptions>,
    ) -> Result<Mesh> {
        if element_type != ElementType::TRI3 {
            return Err(PyrucastError::Message(format!(
                "fill_surface: only TRI3 is supported for now, got {}",
                element_type
            )));
        }
        let n_sub = contour.submesh_count();
        if n_sub == 0 {
            return Err(PyrucastError::Message(
                "fill_surface: contour must contain at least one SEG2 submesh".into(),
            ));
        }
        let cfg = contour.config.clone();
        let dim = with(&cfg, |c| c.dim())?;
        if dim != 2 && dim != 3 {
            return Err(PyrucastError::Message(format!(
                "fill_surface: contour configuration must be 2-D or 3-D, got dim={}",
                dim
            )));
        }

        // 1. Validate each submesh and extract its ordered closed chain of node ids.
        let mut chains: Vec<Vec<NodeId>> = Vec::with_capacity(n_sub);
        for sm_idx in 0..n_sub {
            let sm = contour.submesh(sm_idx)?;
            let (et, n_elems, conn) = with(&sm, |s| {
                (s.element_type(), s.cell_count(), s.connectivity().to_vec())
            })?;
            if et != ElementType::SEG2 {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: submesh #{} must be SEG2, got {}",
                    sm_idx, et
                )));
            }
            if n_elems < 3 {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: submesh #{} must have ≥ 3 segments, got {}",
                    sm_idx, n_elems
                )));
            }
            let mut next_node: std::collections::HashMap<NodeId, NodeId> =
                std::collections::HashMap::with_capacity(n_elems);
            for i in 0..n_elems {
                let a = conn[2 * i];
                let b = conn[2 * i + 1];
                if next_node.insert(a, b).is_some() {
                    return Err(PyrucastError::Message(format!(
                        "fill_surface: submesh #{}: node {} starts more than one segment",
                        sm_idx, a
                    )));
                }
            }
            let start = conn[0];
            let mut chain: Vec<NodeId> = Vec::with_capacity(n_elems);
            chain.push(start);
            let mut current = *next_node.get(&start).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "fill_surface: submesh #{}: node {} has no outgoing segment",
                    sm_idx, start
                ))
            })?;
            while current != start {
                if chain.len() > n_elems {
                    return Err(PyrucastError::Message(format!(
                        "fill_surface: submesh #{}: contour is not a closed simple loop",
                        sm_idx
                    )));
                }
                chain.push(current);
                current = *next_node.get(&current).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "fill_surface: submesh #{}: node {} has no outgoing segment",
                        sm_idx, current
                    ))
                })?;
            }
            if chain.len() != n_elems {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: submesh #{}: contour has multiple disjoint loops ({} nodes traced out of {})",
                    sm_idx, chain.len(), n_elems
                )));
            }
            chains.push(chain);
        }

        // 2. Flatten the chains into a single list with per-chain offsets.
        let mut chain_offsets: Vec<usize> = Vec::with_capacity(n_sub + 1);
        chain_offsets.push(0);
        let mut flat_nodes: Vec<NodeId> = Vec::new();
        for chain in &chains {
            flat_nodes.extend_from_slice(chain);
            chain_offsets.push(flat_nodes.len());
        }
        let n_total = flat_nodes.len();

        // 3. Collect 2-D points to triangulate. In 2-D direct (x, y);
        //    in 3-D project on the best-fit plane (Newell normal of the
        //    first non-degenerate loop + centroid origin), then verify
        //    planarity across **all** loops jointly.
        use crate::triangulation::{Point2, Point3, Vector3};
        // Projection state — only Some(...) when dim == 3. Carried past the
        // triangulation step so Steiner points born in the 2-D plane can be
        // anti-projected back into 3-D node coordinates.
        struct Projection3D {
            origin: Point3,
            u: Vector3,
            v: Vector3,
        }
        let mut projection: Option<Projection3D> = None;
        let points_2d: Vec<Point2> = if dim == 2 {
            let mut pts = Vec::with_capacity(n_total);
            with(&cfg, |c| -> Result<()> {
                for &id in &flat_nodes {
                    let s = c.coord(id)?;
                    pts.push(Point2::new(s[0], s[1]));
                }
                Ok(())
            })??;
            pts
        } else {
            let mut pts3: Vec<Point3> = Vec::with_capacity(n_total);
            with(&cfg, |c| -> Result<()> {
                for &id in &flat_nodes {
                    let s = c.coord(id)?;
                    pts3.push(Point3::new(s[0], s[1], s[2]));
                }
                Ok(())
            })??;

            // Pick the normal from the first chain that has a well-defined
            // Newell normal. Co-planarity of the other loops is checked
            // afterwards by the global planarity test.
            let normal: Vector3 = (0..n_sub)
                .find_map(|i| {
                    let pts_chain: Vec<Point3> = (chain_offsets[i]..chain_offsets[i + 1])
                        .map(|j| pts3[j])
                        .collect();
                    crate::triangulation::newell_normal(&pts_chain)
                })
                .ok_or_else(|| {
                    PyrucastError::Message(
                        "fill_surface: every 3-D loop is collinear or zero-area".into(),
                    )
                })?;

            // Centroid as the plane origin.
            let origin: Point3 = {
                let sum: Vector3 = pts3.iter().map(|p| p.coords).sum();
                Point3::from(sum / pts3.len() as f64)
            };

            // Planarity check + AABB diagonal in one pass.
            let mut bb_min = Vector3::repeat(f64::INFINITY);
            let mut bb_max = Vector3::repeat(f64::NEG_INFINITY);
            let mut max_dev = 0.0_f64;
            for p in &pts3 {
                let dev = (p - origin).dot(&normal).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
                bb_min = bb_min.zip_map(&p.coords, f64::min);
                bb_max = bb_max.zip_map(&p.coords, f64::max);
            }
            let diag = (bb_max - bb_min).norm();
            let tol = 1e-6 * diag;
            if max_dev > tol {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: contour is not planar — max deviation {:.3e} exceeds tolerance {:.3e} (1e-6 × diag={:.3e})",
                    max_dev, tol, diag
                )));
            }

            let (u, v) = crate::triangulation::in_plane_basis(normal);
            let pts_2d: Vec<Point2> = pts3
                .iter()
                .map(|p| {
                    let d = p - origin;
                    Point2::new(d.dot(&u), d.dot(&v))
                })
                .collect();
            projection = Some(Projection3D { origin, u, v });
            pts_2d
        };

        // 4. Triangulate. Fast path is plain ear clipping for a single loop
        //    without refinement; everything else goes through the CDT
        //    (constrained Delaunay + optional Ruppert refinement).
        let refine = refinement.filter(|o| o.is_active());
        let (triangles, mut flat_to_node, steiner_points_2d): (
            Vec<[usize; 3]>,
            Vec<NodeId>,
            Vec<Point2>,
        ) = if n_sub == 1 && refine.is_none() {
            let tris = crate::triangulation::ear_clip_2d(&points_2d)?;
            (tris, flat_nodes, Vec::new())
        } else {
            // Detect the outer loop (largest |signed area|), then build
            // the (outer, holes) input for the CDT façades.
            let mut areas: Vec<f64> = Vec::with_capacity(n_sub);
            for i in 0..n_sub {
                let slice = &points_2d[chain_offsets[i]..chain_offsets[i + 1]];
                areas.push(crate::triangulation::signed_area(slice).abs());
            }
            let outer_idx = (0..n_sub)
                .max_by(|&a, &b| {
                    areas[a]
                        .partial_cmp(&areas[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            let mut outer_pts: Vec<Point2> = Vec::new();
            let mut new_flat_nodes: Vec<NodeId> = Vec::new();
            for j in chain_offsets[outer_idx]..chain_offsets[outer_idx + 1] {
                outer_pts.push(points_2d[j]);
                new_flat_nodes.push(flat_nodes[j]);
            }
            let mut hole_pts_list: Vec<Vec<Point2>> = Vec::new();
            for i in 0..n_sub {
                if i == outer_idx {
                    continue;
                }
                let mut hole_pts = Vec::new();
                for j in chain_offsets[i]..chain_offsets[i + 1] {
                    hole_pts.push(points_2d[j]);
                    new_flat_nodes.push(flat_nodes[j]);
                }
                hole_pts_list.push(hole_pts);
            }

            let n_existing = new_flat_nodes.len();
            if let Some(opts) = refine {
                let (all_pts, tris) =
                    crate::triangulation::triangulate_polygon_with_holes_refined(
                        &outer_pts,
                        &hole_pts_list,
                        opts,
                    )?;
                // Steiner points are everything past the original contour count.
                let steiner = all_pts[n_existing..].to_vec();
                (tris, new_flat_nodes, steiner)
            } else {
                let tris = crate::triangulation::triangulate_polygon_with_holes(
                    &outer_pts,
                    &hole_pts_list,
                )?;
                (tris, new_flat_nodes, Vec::new())
            }
        };

        // 5. Create one Configuration node per Steiner point. In 2-D the
        //    (x, y) coordinates are used directly; in 3-D they are
        //    anti-projected back through the saved plane basis.
        //
        //    The handles are kept alive in `_steiner_nodes` until after
        //    `add_cell` has bumped the Configuration's per-node refcount
        //    — otherwise a Drop in between would let the GC reclaim the
        //    just-created node.
        let mut _steiner_nodes: Vec<Node> = Vec::with_capacity(steiner_points_2d.len());
        for p in &steiner_points_2d {
            let coords: Vec<f64> = match &projection {
                None => vec![p.x, p.y],
                Some(proj) => {
                    let p3 = proj.origin + proj.u * p.x + proj.v * p.y;
                    vec![p3.x, p3.y, p3.z]
                }
            };
            let node = Node::create_in(cfg.clone(), &coords)?;
            flat_to_node.push(node.id());
            _steiner_nodes.push(node);
        }

        // 6. Build the TRI3 mesh: each triangle is a triple of NodeIds.
        //    `add_cell` increfs every node it touches, so the Steiner
        //    nodes stay alive after `_steiner_nodes` drops at end of scope.
        let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
        for [i, j, k] in triangles {
            mesh.add_cell(&[flat_to_node[i], flat_to_node[j], flat_to_node[k]])?;
        }
        Ok(mesh)
    }

    /// Visualize this mesh — every TRI3 submesh is drawn, each in its own
    /// [`SubMesh::face_color`]. Other element types are silently skipped
    /// (support will be added incrementally). See [`SubMesh::plot`] for
    /// the meaning of `view` and `save`.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
    ) -> Result<()> {
        crate::viz::render(self, view, save)
    }

    /// Return a new mesh where submeshes of the same element type are fused
    /// into a single submesh, and duplicate cells (identical node sequences)
    /// are removed. Types appear in their first-seen order; the face colour of
    /// the first submesh of each type is kept.
    pub fn consolidate(&self) -> Result<Mesh> {
        use std::collections::HashSet;

        let mut result = Mesh::new(self.config.clone());

        // Collect types in first-seen order.
        let mut ordered_types: Vec<ElementType> = Vec::new();
        for sm_handle in &self.submeshes {
            let et = with(sm_handle, |s| s.element_type())?;
            if !ordered_types.contains(&et) {
                ordered_types.push(et);
            }
        }

        for et in ordered_types {
            let npc = et.nodes_per_cell();

            // Face colour from the first submesh of this type.
            let first_color = self
                .submeshes
                .iter()
                .find(|h| with(h, |s| s.element_type()).ok() == Some(et))
                .map(|h| with(h, |s| s.face_color()))
                .transpose()?
                .unwrap_or_default();

            let mut new_sm = SubMesh::new(self.config.clone(), et);
            new_sm.set_face_color(first_color);

            let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
            for sm_handle in &self.submeshes {
                let sm_et = with(sm_handle, |s| s.element_type())?;
                if sm_et != et {
                    continue;
                }
                let conn = with(sm_handle, |s| s.connectivity().to_vec())?;
                for chunk in conn.chunks(npc) {
                    if seen.insert(chunk.to_vec()) {
                        new_sm.add_cell(chunk)?;
                    }
                }
            }

            result.submeshes.push(insert(new_sm));
        }

        Ok(result)
    }

    /// Return a new mesh containing all submeshes of `self` followed by all
    /// submeshes of `other`. Both meshes must share the same `Configuration`.
    pub fn merge(&self, other: &Mesh) -> Result<Mesh> {
        if self.config.index() != other.config.index()
            || self.config.generation() != other.config.generation()
        {
            return Err(PyrucastError::Message(
                "merge: meshes are attached to different Configurations".into(),
            ));
        }
        let mut result = Mesh::new(self.config.clone());
        for sm in self.submeshes.iter().chain(other.submeshes.iter()) {
            result.submeshes.push(sm.clone());
        }
        Ok(result)
    }
}

impl std::ops::Add<&Mesh> for &Mesh {
    type Output = Result<Mesh>;
    fn add(self, rhs: &Mesh) -> Result<Mesh> {
        self.merge(rhs)
    }
}

impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("submesh_count", &self.submeshes.len())
            .finish()
    }
}

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.cell_count().unwrap_or(0);
        write!(
            f,
            "Mesh: {} submesh(es), {} cell(s) total",
            self.submeshes.len(),
            total
        )
    }
}

// ─── Python binding ─────────────────────────────────────────────────────────

#[cfg(feature = "extension-module")]
mod python {
    use super::*;
    use crate::configuration::PyConfiguration;
    use crate::node::PyNode;
    use crate::store::insert;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Python wrapper for [`SubMesh`].
    #[pyclass(name = "SubMesh")]
    pub struct PySubMesh {
        pub(crate) handle: Handle<SubMesh>,
    }

    #[pymethods]
    impl PySubMesh {
        #[new]
        fn py_new(config: PyRef<PyConfiguration>, element_type: &str) -> PyResult<Self> {
            let et = ElementType::from_name(element_type).ok_or_else(|| {
                PyValueError::new_err(format!("unknown element type: {element_type}"))
            })?;
            let cfg_handle = config.handle.clone();
            let sm = SubMesh::new(cfg_handle, et);
            Ok(Self { handle: insert(sm) })
        }

        #[getter]
        fn element_type(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| s.element_type().name().to_string())?)
        }

        fn add_cell(&self, nodes: Vec<u32>) -> PyResult<usize> {
            let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
            let idx = with_mut(&self.handle, move |s| s.add_cell(&nodes_typed))??;
            Ok(idx)
        }

        fn cell_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |s| s.cell_count())?)
        }

        /// Face colour as an `(r, g, b)` tuple of bytes.
        #[getter]
        fn face_color(&self) -> PyResult<(u8, u8, u8)> {
            let c = with(&self.handle, |s| s.face_color())?;
            Ok((c.r, c.g, c.b))
        }

        /// Set the face colour from an `(r, g, b)` tuple of bytes.
        #[setter]
        fn set_face_color(&self, rgb: (u8, u8, u8)) -> PyResult<()> {
            with_mut(&self.handle, |s| {
                s.set_face_color(crate::color::RgbColor::new(rgb.0, rgb.1, rgb.2))
            })?;
            Ok(())
        }

        /// Visualize this submesh.
        ///
        /// - `save=None`: interactive window (requires `viz-interactive`).
        /// - `save="<path>.png"` or `.svg`: image file.
        /// - `view`: optional `(yaw, pitch, scale)` triple; default is iso.
        #[cfg(feature = "viz")]
        #[pyo3(signature = (view=None, save=None))]
        fn plot(
            &self,
            view: Option<(f64, f64, f64)>,
            save: Option<std::path::PathBuf>,
        ) -> PyResult<()> {
            let view = view.map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
            });
            let save_ref = save.as_deref();
            with(&self.handle, |s| s.plot(view, save_ref))??;
            Ok(())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| format!("{:?}", s))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| format!("{}", s))?)
        }
    }

    /// Python wrapper for [`Mesh`].
    #[pyclass(name = "Mesh")]
    pub struct PyMesh {
        pub(crate) handle: Handle<Mesh>,
    }

    #[pymethods]
    impl PyMesh {
        /// `Mesh(config)` — empty mesh.
        /// `Mesh(config, element_type)` — mesh with one pre-created submesh.
        #[new]
        #[pyo3(signature = (config, element_type=None))]
        fn py_new(config: PyRef<PyConfiguration>, element_type: Option<&str>) -> PyResult<Self> {
            let cfg = config.handle.clone();
            let mesh = match element_type {
                Some(et_str) => {
                    let et = ElementType::from_name(et_str).ok_or_else(|| {
                        PyValueError::new_err(format!("unknown element type: {et_str}"))
                    })?;
                    Mesh::with_element_type(cfg, et)
                }
                None => Mesh::new(cfg),
            };
            Ok(Self { handle: insert(mesh) })
        }

        fn add_submesh(&self, sm: PyRef<PySubMesh>) -> PyResult<()> {
            let sm_handle = sm.handle.clone();
            with_mut(&self.handle, |m| m.add_submesh(sm_handle))??;
            Ok(())
        }

        fn add_cell(&self, nodes: Vec<u32>) -> PyResult<usize> {
            let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
            let idx = with_mut(&self.handle, move |m| m.add_cell(&nodes_typed))??;
            Ok(idx)
        }

        #[getter]
        fn element_type(&self) -> PyResult<Option<String>> {
            let maybe_sm = with(&self.handle, |m| -> Option<Handle<SubMesh>> {
                if m.submesh_count() == 1 {
                    m.submesh(0).ok()
                } else {
                    None
                }
            })?;
            match maybe_sm {
                Some(h) => Ok(Some(with(&h, |sm| sm.element_type().name().to_string())?)),
                None => Ok(None),
            }
        }

        fn element_types(&self) -> PyResult<Vec<String>> {
            let types = with(&self.handle, |m| m.element_types())??;
            Ok(types.into_iter().map(|et| et.name().to_string()).collect())
        }

        fn cell_counts(&self) -> PyResult<Vec<usize>> {
            Ok(with(&self.handle, |m| m.cell_counts())??)
        }

        fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> PyResult<PyNode> {
            let node = with(&self.handle, |m| m.node(submesh_idx, cell_idx, node_idx))??;
            Ok(PyNode::from_node(node))
        }

        #[classmethod]
        fn from_live_nodes(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            config: PyRef<PyConfiguration>,
        ) -> PyResult<Self> {
            let mesh = Mesh::from_live_nodes(config.handle.clone())?;
            Ok(Self { handle: insert(mesh) })
        }

        #[classmethod]
        fn line_seg2(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            a: PyRef<PyNode>,
            b: PyRef<PyNode>,
            n_elems: usize,
        ) -> PyResult<Self> {
            let mesh = Mesh::line_seg2(a.as_node(), b.as_node(), n_elems)?;
            Ok(Self { handle: insert(mesh) })
        }

        #[classmethod]
        fn circle_seg2(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            center: PyRef<PyNode>,
            normal: Vec<f64>,
            radius: f64,
            n_elems: usize,
        ) -> PyResult<Self> {
            let mesh = Mesh::circle_seg2(center.as_node(), &normal, radius, n_elems)?;
            Ok(Self { handle: insert(mesh) })
        }

        #[classmethod]
        fn sweep_qua4(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            mesh_a: PyRef<PyMesh>,
            mesh_b: PyRef<PyMesh>,
            n_layers: usize,
        ) -> PyResult<Self> {
            // Cannot nest `with` on two Handle<Mesh> values (same Mutex).
            // Snapshot each side into a local Mesh (just clones the
            // submesh handles + the configuration handle), then call the
            // pure-Rust API on the snapshots — outside any store lock.
            let handle_a = mesh_a.handle.clone();
            let handle_b = mesh_b.handle.clone();
            let snap_a = with(&handle_a, |a| -> Result<Mesh> {
                let mut copy = Mesh::new(a.configuration());
                for i in 0..a.submesh_count() {
                    copy.add_submesh(a.submesh(i)?)?;
                }
                Ok(copy)
            })??;
            let snap_b = with(&handle_b, |b| -> Result<Mesh> {
                let mut copy = Mesh::new(b.configuration());
                for i in 0..b.submesh_count() {
                    copy.add_submesh(b.submesh(i)?)?;
                }
                Ok(copy)
            })??;
            let mesh = Mesh::sweep_qua4(&snap_a, &snap_b, n_layers)?;
            Ok(Self { handle: insert(mesh) })
        }

        #[classmethod]
        fn extrude(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            mesh: PyRef<PyMesh>,
            direction: Vec<f64>,
            n_layers: usize,
        ) -> PyResult<Self> {
            let handle = mesh.handle.clone();
            let result = with(&handle, |m| Mesh::extrude(m, &direction, n_layers))??;
            Ok(Self { handle: insert(result) })
        }

        #[classmethod]
        #[pyo3(signature = (contour, element_type, max_edge_length=None, min_angle_deg=None))]
        fn fill_surface(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            contour: PyRef<PyMesh>,
            element_type: &str,
            max_edge_length: Option<f64>,
            min_angle_deg: Option<f64>,
        ) -> PyResult<Self> {
            let et = ElementType::from_name(element_type).ok_or_else(|| {
                PyValueError::new_err(format!("unknown element type: {element_type}"))
            })?;
            let refinement = if max_edge_length.is_some() || min_angle_deg.is_some() {
                Some(crate::triangulation::RefinementOptions {
                    max_edge_length,
                    min_angle_deg,
                })
            } else {
                None
            };
            let handle = contour.handle.clone();
            let mesh = with(&handle, |c| Mesh::fill_surface(c, et, refinement))??;
            Ok(Self { handle: insert(mesh) })
        }

        fn __add__(&self, other: PyRef<PyMesh>) -> PyResult<PyMesh> {
            // Cannot nest `with` on two Handle<Mesh> values — they share
            // the same per-type Mutex. Snapshot each side separately and
            // assemble outside the locks.
            let other_handle = other.handle.clone();
            let (other_cfg, other_subs): (Handle<Configuration>, Vec<Handle<SubMesh>>) =
                with(&other_handle, |b| -> Result<_> {
                    let mut subs = Vec::with_capacity(b.submesh_count());
                    for i in 0..b.submesh_count() {
                        subs.push(b.submesh(i)?);
                    }
                    Ok((b.configuration(), subs))
                })??;

            let mesh = with(&self.handle, |a| -> Result<Mesh> {
                let self_cfg = a.configuration();
                if self_cfg.index() != other_cfg.index()
                    || self_cfg.generation() != other_cfg.generation()
                {
                    return Err(PyrucastError::Message(
                        "merge: meshes are attached to different Configurations".into(),
                    ));
                }
                let mut result = Mesh::new(self_cfg);
                for i in 0..a.submesh_count() {
                    result.add_submesh(a.submesh(i)?)?;
                }
                for sm in &other_subs {
                    result.add_submesh(sm.clone())?;
                }
                Ok(result)
            })??;
            Ok(PyMesh { handle: insert(mesh) })
        }

        /// Fusionne les sous-maillages de même type et supprime les mailles en
        /// double. Retourne un nouveau maillage avec un sous-maillage par type.
        fn consolidate(&self) -> PyResult<PyMesh> {
            let mesh = with(&self.handle, |m| m.consolidate())??;
            Ok(PyMesh { handle: insert(mesh) })
        }

        fn submesh_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |m| m.submesh_count())?)
        }

        fn cell_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |m| m.cell_count())??)
        }

        /// Visualize this mesh (every TRI3 submesh, each in its own colour).
        /// See `SubMesh.plot` for the meaning of `view` and `save`.
        #[cfg(feature = "viz")]
        #[pyo3(signature = (view=None, save=None))]
        fn plot(
            &self,
            view: Option<(f64, f64, f64)>,
            save: Option<std::path::PathBuf>,
        ) -> PyResult<()> {
            let view = view.map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
            });
            let save_ref = save.as_deref();
            with(&self.handle, |m| m.plot(view, save_ref))??;
            Ok(())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |m| format!("{:?}", m))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |m| format!("{}", m))?)
        }
    }
}

#[cfg(feature = "extension-module")]
pub use python::{PyMesh, PySubMesh};

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::store::{insert, with};

    #[test]
    fn submesh_poi1_is_node_list() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.add_cell(&[b.id()]).unwrap();
        assert_eq!(sm.cell_count(), 2);
        assert_eq!(sm.connectivity()[0], a.id());
        assert_eq!(sm.connectivity()[1], b.id());
    }

    #[test]
    fn submesh_tri3_increfs_and_drop_decrefs() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // SubMesh increfed each of the 3 nodes, in addition to the Nodes.
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 2);
            assert_eq!(cf.refcount(b.id()), 2);
            assert_eq!(cf.refcount(c.id()), 2);
        })
        .unwrap();
        drop(sm);
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 1);
            assert_eq!(cf.refcount(b.id()), 1);
            assert_eq!(cf.refcount(c.id()), 1);
        })
        .unwrap();
    }

    #[test]
    fn submesh_add_cell_invalid_arity() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        let err = sm.add_cell(&[a.id()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        // No increment should have survived the failure.
        with(&cfg, |cf| assert_eq!(cf.refcount(a.id()), 1)).unwrap();
    }

    #[test]
    fn submesh_add_cell_collected_node_rollback() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let dead_id = with_mut(&cfg, |c| c.add_node(&[2.0])).unwrap().unwrap();
        // dead_id starts at refcount=1; decrement then collect.
        with_mut(&cfg, |c| {
            c.decref(dead_id).unwrap();
            assert_eq!(c.gc(), 1);
        })
        .unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        // a (live), b (live), dead_id (collected) → add_cell fails after
        // increfing a and b. The rollback must undo those increfs.
        let err = sm.add_cell(&[a.id(), b.id(), dead_id]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 1, "a must be rolled back");
            assert_eq!(cf.refcount(b.id()), 1, "b must be rolled back");
        })
        .unwrap();
        assert_eq!(sm.cell_count(), 0);
    }

    #[test]
    fn mesh_aggregates_submeshes_same_config() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let cc = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let sm_pts = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            insert(sm)
        };
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), cc.id()]).unwrap();
            insert(sm)
        };

        let mut mesh = Mesh::new(cfg.clone());
        mesh.add_submesh(sm_pts).unwrap();
        mesh.add_submesh(sm_tri).unwrap();
        assert_eq!(mesh.submesh_count(), 2);
        assert_eq!(mesh.cell_count().unwrap(), 3); // 2 points + 1 triangle
    }

    #[test]
    fn mesh_rejects_submesh_from_other_configuration() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());

        let sm = insert(SubMesh::new(cfg1.clone(), ElementType::POI1));
        let mut mesh = Mesh::new(cfg2);
        let err = mesh.add_submesh(sm).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn mesh_element_types_and_cell_counts() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[a.id()]).unwrap();
        m.add_cell(&[b.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        m.add_submesh(sm_tri).unwrap();

        assert_eq!(
            m.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::TRI3]
        );
        assert_eq!(m.cell_counts().unwrap(), vec![2, 1]);
    }

    #[test]
    fn mesh_node_access_by_indices() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::TRI3);
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let n = m.node(0, 0, 0).unwrap();
        assert_eq!(n.id(), a.id()); // noeud 0 de l'élément 0 = a
        assert!(m.node(1, 0, 0).is_err()); // sous-maillage hors bornes
        assert!(m.node(0, 1, 0).is_err()); // cellule hors bornes
        assert!(m.node(0, 0, 3).is_err()); // noeud hors bornes (TRI3 : indices 0..2)
    }

    #[test]
    fn mesh_from_live_nodes() {
        let cfg = insert(Configuration::new(1).unwrap());
        let _a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let _b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let _c = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        let m = Mesh::from_live_nodes(cfg.clone()).unwrap();
        assert_eq!(m.element_types().unwrap(), vec![ElementType::POI1]);
        assert_eq!(m.cell_count().unwrap(), 3);

        // from_live_nodes est un snapshot : le maillage m tient les refs,
        // un second appel sur la même configuration donne le même résultat.
        let m2 = Mesh::from_live_nodes(cfg).unwrap();
        assert_eq!(m2.cell_count().unwrap(), 3);
    }

    #[test]
    fn mesh_merge_combines_submeshes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m1 = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m1.add_cell(&[a.id()]).unwrap();
        m1.add_cell(&[b.id()]).unwrap();

        let mut m2 = Mesh::with_element_type(cfg.clone(), ElementType::TRI3);
        m2.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let merged = (&m1 + &m2).unwrap();
        assert_eq!(merged.submesh_count(), 2);
        assert_eq!(merged.cell_count().unwrap(), 3); // 2 POI1 + 1 TRI3
    }

    #[test]
    fn mesh_merge_rejects_different_configurations() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());
        let err = Mesh::new(cfg1).merge(&Mesh::new(cfg2)).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn line_seg2_basic() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[6.0]).unwrap();

        let mesh = Mesh::line_seg2(&a, &b, 3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 3);

        // noeud 0 de chaque cellule
        let n00 = mesh.node(0, 0, 0).unwrap();
        let n10 = mesh.node(0, 1, 0).unwrap();
        let n20 = mesh.node(0, 2, 0).unwrap();
        assert_eq!(n00.coord().unwrap(), vec![0.0]);
        assert!((n10.coord().unwrap()[0] - 2.0).abs() < 1e-12);
        assert!((n20.coord().unwrap()[0] - 4.0).abs() < 1e-12);

        // dernier nœud de la dernière cellule = nœud b
        let n21 = mesh.node(0, 2, 1).unwrap();
        assert_eq!(n21.id(), b.id());
    }

    #[test]
    fn line_seg2_one_element() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();

        let mesh = Mesh::line_seg2(&a, &b, 1).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        // aucun nœud intermédiaire créé : seuls a et b sont dans le maillage
        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(mesh.node(0, 0, 1).unwrap().id(), b.id());
    }

    #[test]
    fn line_seg2_zero_elems_is_error() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        assert!(Mesh::line_seg2(&a, &b, 0).is_err());
    }

    #[test]
    fn circle_seg2_basic_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mesh = Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 4).unwrap();

        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 4);

        // Nœud 0 : θ=0 → (1, 0)
        let n0 = mesh.node(0, 0, 0).unwrap();
        assert!((n0.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n0.coord().unwrap()[1]).abs() < 1e-12);
        // Nœud 1 : θ=π/2 → (0, 1)
        let n1 = mesh.node(0, 1, 0).unwrap();
        assert!((n1.coord().unwrap()[0]).abs() < 1e-12);
        assert!((n1.coord().unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn circle_seg2_closed_loop() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mesh = Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 6).unwrap();

        // Le 2e nœud du dernier élément = le 1er nœud du 1er élément
        let last_end = mesh.node(0, 5, 1).unwrap();
        let first_start = mesh.node(0, 0, 0).unwrap();
        assert_eq!(last_end.id(), first_start.id());
    }

    #[test]
    fn circle_seg2_radius_and_center_offset() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[1.0, 2.0]).unwrap();
        let mesh = Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], 3.0, 8).unwrap();

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().coord().unwrap();
            let dist = ((c[0] - 1.0).powi(2) + (c[1] - 2.0).powi(2)).sqrt();
            assert!((dist - 3.0).abs() < 1e-10, "élément {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_seg2_3d_xz_plane() {
        let cfg = insert(Configuration::new(3).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
        // Normale selon Y → cercle dans le plan XZ
        let mesh = Mesh::circle_seg2(&center, &[0.0, 1.0, 0.0], 2.0, 8).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 8);

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().coord().unwrap();
            assert!((c[1]).abs() < 1e-12, "élément {ei}: y={}", c[1]);
            let dist = (c[0].powi(2) + c[2].powi(2)).sqrt();
            assert!((dist - 2.0).abs() < 1e-10, "élément {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_seg2_rejects_too_few_elements() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 2).is_err());
    }

    #[test]
    fn circle_seg2_rejects_nonpositive_radius() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], 0.0, 4).is_err());
        assert!(Mesh::circle_seg2(&center, &[0.0, 0.0, 1.0], -1.0, 4).is_err());
    }

    #[test]
    fn circle_seg2_rejects_zero_normal() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(Mesh::circle_seg2(&center, &[0.0, 0.0, 0.0], 1.0, 4).is_err());
    }

    #[test]
    fn sweep_qua4_basic() {
        let cfg = insert(Configuration::new(2).unwrap());
        // mesh_a : y=0, 2 éléments SEG2 de (0,0) à (2,0)
        let a0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = Mesh::line_seg2(&a0, &a2, 2).unwrap();

        // mesh_b : y=1, 2 éléments SEG2 de (0,1) à (2,1)
        let b0 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(cfg.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = Mesh::line_seg2(&b0, &b2, 2).unwrap();

        let qua = Mesh::sweep_qua4(&mesh_a, &mesh_b, 2).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        // 2 éléments × 2 couches = 4 cellules QUA4
        assert_eq!(qua.cell_count().unwrap(), 4);

        // Cellule (k=0, j=0) : coin inférieur gauche doit être (0,0)
        let n00 = qua.node(0, 0, 0).unwrap();
        assert_eq!(n00.coord().unwrap(), vec![0.0, 0.0]);
        // coin inférieur droit : (1,0)
        let n01 = qua.node(0, 0, 1).unwrap();
        assert!((n01.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n01.coord().unwrap()[1]).abs() < 1e-12);
        // coin supérieur droit : (1,0.5) — couche intermédiaire
        let n02 = qua.node(0, 0, 2).unwrap();
        assert!((n02.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n02.coord().unwrap()[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sweep_qua4_one_layer_reuses_endpoints() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = Mesh::line_seg2(&a0, &a1, 1).unwrap();

        let b0 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = Mesh::line_seg2(&b0, &b1, 1).unwrap();

        let qua = Mesh::sweep_qua4(&mesh_a, &mesh_b, 1).unwrap();
        assert_eq!(qua.cell_count().unwrap(), 1);

        // Les nœuds originaux doivent être réutilisés
        let n0 = qua.node(0, 0, 0).unwrap();
        let n1 = qua.node(0, 0, 1).unwrap();
        let n2 = qua.node(0, 0, 2).unwrap();
        let n3 = qua.node(0, 0, 3).unwrap();
        assert_eq!(n0.id(), a0.id());
        assert_eq!(n1.id(), a1.id());
        assert_eq!(n2.id(), b1.id());
        assert_eq!(n3.id(), b0.id());
    }

    #[test]
    fn sweep_qua4_rejects_zero_layers() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let m = Mesh::line_seg2(&a, &b, 1).unwrap();
        assert!(Mesh::sweep_qua4(&m, &m, 0).is_err());
    }

    #[test]
    fn sweep_qua4_rejects_mismatched_elem_counts() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
        let m1 = Mesh::line_seg2(&a, &b, 1).unwrap();
        let m2 = Mesh::line_seg2(&a, &c, 2).unwrap();
        assert!(Mesh::sweep_qua4(&m1, &m2, 1).is_err());
    }

    #[test]
    fn sweep_qua4_rejects_non_seg2() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let seg = Mesh::line_seg2(&a, &b, 1).unwrap();

        // maillage TRI3 à la place d'un SEG2
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let mut tri_mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
        tri_mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        assert!(Mesh::sweep_qua4(&tri_mesh, &seg, 1).is_err());
        assert!(Mesh::sweep_qua4(&seg, &tri_mesh, 1).is_err());
    }

    #[test]
    fn extrude_seg2_to_qua4() {
        let cfg = insert(Configuration::new(2).unwrap());
        // SEG2 de (0,0) à (2,0) avec 2 éléments, 3 nœuds : x=0,1,2
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
        let seg = Mesh::line_seg2(&a, &b, 2).unwrap();

        // extrusion de 3 couches vers (0,3)
        let qua = Mesh::extrude(&seg, &[0.0, 3.0], 3).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        // 2 éléments × 3 couches = 6 cellules
        assert_eq!(qua.cell_count().unwrap(), 6);

        // Coin bas-gauche de la 1ère cellule = (0,0)
        let n = qua.node(0, 0, 0).unwrap();
        assert_eq!(n.coord().unwrap(), vec![0.0, 0.0]);
        // Coin haut-gauche de la 1ère cellule = (0,1) — couche 1 sur 3
        let n = qua.node(0, 0, 3).unwrap();
        assert!((n.coord().unwrap()[0]).abs() < 1e-12);
        assert!((n.coord().unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn extrude_seg2_shared_nodes_stay_shared() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
        let seg = Mesh::line_seg2(&a, &b, 2).unwrap();

        let qua = Mesh::extrude(&seg, &[0.0, 1.0], 1).unwrap();
        // 2 cellules QUA4, le nœud du milieu (bas) doit être commun
        let mid_cell0 = qua.node(0, 0, 1).unwrap(); // coin droit de la cellule 0
        let mid_cell1 = qua.node(0, 1, 0).unwrap(); // coin gauche de la cellule 1
        assert_eq!(mid_cell0.id(), mid_cell1.id());
    }

    #[test]
    fn extrude_qua4_to_hex8() {
        let cfg = insert(Configuration::new(3).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[1.0, 1.0, 0.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let mut qua_mesh = Mesh::with_element_type(cfg.clone(), ElementType::QUA4);
        qua_mesh.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()]).unwrap();

        let hex = Mesh::extrude(&qua_mesh, &[0.0, 0.0, 2.0], 1).unwrap();
        assert_eq!(hex.element_types().unwrap(), vec![ElementType::HEX8]);
        assert_eq!(hex.cell_count().unwrap(), 1);

        // Les 4 nœuds du bas sont réutilisés
        assert_eq!(hex.node(0, 0, 0).unwrap().id(), n0.id());
        assert_eq!(hex.node(0, 0, 1).unwrap().id(), n1.id());
        assert_eq!(hex.node(0, 0, 2).unwrap().id(), n2.id());
        assert_eq!(hex.node(0, 0, 3).unwrap().id(), n3.id());
        // Le 1er nœud du haut est à (0,0,2)
        let top0 = hex.node(0, 0, 4).unwrap();
        assert_eq!(top0.coord().unwrap(), vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn extrude_rejects_zero_layers() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let seg = Mesh::line_seg2(&a, &b, 1).unwrap();
        assert!(Mesh::extrude(&seg, &[0.0, 1.0], 0).is_err());
    }

    #[test]
    fn extrude_rejects_wrong_direction_dim() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let seg = Mesh::line_seg2(&a, &b, 1).unwrap();
        // direction à 3 composantes pour des nœuds en 2D
        assert!(Mesh::extrude(&seg, &[0.0, 1.0, 0.0], 1).is_err());
    }

    #[test]
    fn extrude_rejects_unsupported_element_type() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let mut tri = Mesh::with_element_type(cfg, ElementType::TRI3);
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(Mesh::extrude(&tri, &[0.0, 0.0], 1).is_err());
    }

    /// Build a closed SEG2 contour from a polyline of 2-D points.
    /// Returns the contour mesh and the (owned) node handles in input order.
    fn build_contour_2d(cfg: Handle<Configuration>, pts: &[(f64, f64)]) -> (Mesh, Vec<Node>) {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y)| Node::create_in(cfg.clone(), &[x, y]).unwrap())
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        (contour, nodes)
    }

    #[test]
    fn fill_surface_square_gives_two_triangles() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);

        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count().unwrap(), 2);

        // Every triangle node must be one of the four contour nodes
        // (no Steiner points in this first iteration).
        let node_ids: std::collections::HashSet<_> = nodes.iter().map(|n| n.id()).collect();
        for ci in 0..2 {
            for ni in 0..3 {
                let id = tri.node(0, ci, ni).unwrap().id();
                assert!(node_ids.contains(&id), "triangle node {} not in contour", id);
            }
        }
    }

    #[test]
    fn fill_surface_triangles_sum_to_polygon_area() {
        let cfg = insert(Configuration::new(2).unwrap());
        // Concave L-shape, CCW; expected area = 5.
        let l = [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ];
        let (contour, _nodes) = build_contour_2d(cfg.clone(), &l);

        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 4); // n - 2

        let mut total = 0.0;
        for ci in 0..4 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            // Signed area, all triangles must be CCW (positive).
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW (signed area {})", ci, a);
            total += a;
        }
        assert!((total - 5.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_increfs_contour_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let ids: Vec<_> = nodes.iter().map(|n| n.id()).collect();

        // Before filling: each node is referenced by its Node handle (+1)
        // and by the SEG2 contour (×2 because each node belongs to two
        // consecutive segments) ⇒ refcount = 3.
        with(&cfg, |c| {
            for &id in &ids {
                assert_eq!(c.refcount(id), 3);
            }
        })
        .unwrap();

        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();

        // After filling: each contour node is referenced once more per
        // incident triangle.
        let mut extra = [0u32; 4];
        for ci in 0..2 {
            for ni in 0..3 {
                let id = tri.node(0, ci, ni).unwrap().id();
                let k = ids.iter().position(|&x| x == id).unwrap();
                extra[k] += 1;
            }
        }
        with(&cfg, |c| {
            for k in 0..4 {
                assert_eq!(c.refcount(ids[k]), 3 + extra[k]);
            }
        })
        .unwrap();
    }

    #[test]
    fn fill_surface_rejects_non_tri3() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, _n) =
            build_contour_2d(cfg, &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert!(Mesh::fill_surface(&contour, ElementType::QUA4, None).is_err());
    }

    #[test]
    fn fill_surface_rejects_non_seg2_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let mut bogus = Mesh::with_element_type(cfg, ElementType::TRI3);
        bogus.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(Mesh::fill_surface(&bogus, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_rejects_dim_above_three() {
        // dim = 4 is not supported (no projection defined).
        let cfg = insert(Configuration::new(4).unwrap());
        let nodes: Vec<Node> = (0..4)
            .map(|i| {
                let t = i as f64;
                Node::create_in(cfg.clone(), &[t, 0.0, 0.0, 0.0]).unwrap()
            })
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        for i in 0..4 {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % 4].id()])
                .unwrap();
        }
        assert!(Mesh::fill_surface(&contour, ElementType::TRI3, None).is_err());
    }

    /// Build a closed SEG2 contour in 3-D from a polyline of 3-D points.
    fn build_contour_3d(cfg: Handle<Configuration>, pts: &[(f64, f64, f64)]) -> (Mesh, Vec<Node>) {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y, z)| Node::create_in(cfg.clone(), &[x, y, z]).unwrap())
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        (contour, nodes)
    }

    #[test]
    fn fill_surface_3d_square_in_z_plane() {
        // Unit square at z = 5: planar, CCW seen from +z.
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 5.0),
                (1.0, 0.0, 5.0),
                (1.0, 1.0, 5.0),
                (0.0, 1.0, 5.0),
            ],
        );

        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        // Every triangle vertex must lie exactly on z = 5 (nodes are
        // reused, no Steiner points).
        for ci in 0..2 {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 5.0).abs() < 1e-12);
            }
        }

        // Sum of 3-D triangle areas = 1 (unit square).
        let mut total = 0.0;
        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_3d_tilted_square() {
        // Square rotated 45° about the x axis; its plane has normal (0, -1, 1)/√2.
        // Vertices in CCW order (seen from +normal):
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, s, s),
                (0.0, s, s),
            ],
        );
        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        // Total 3-D area = 1 (unit square in the tilted plane).
        let mut total = 0.0;
        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_3d_rejects_non_planar_contour() {
        // Square corners with one vertex significantly out of plane.
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.5), // out of the z=0 plane by 0.5 — > 1e-6 × diag
                (0.0, 1.0, 0.0),
            ],
        );
        let err = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("not planar"), "unexpected message: {}", msg);
    }

    #[test]
    fn fill_surface_3d_accepts_tiny_numerical_noise() {
        // Same square but with sub-tolerance noise — must still triangulate.
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 1e-10), // diag ≈ √2 ⇒ tol ≈ 1.4e-6 ; 1e-10 ≪ tol
                (0.0, 1.0, 0.0),
            ],
        );
        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
    }

    #[test]
    fn fill_surface_rejects_empty_submesh() {
        // A submesh with zero cells is still rejected (< 3 segments).
        let cfg = insert(Configuration::new(2).unwrap());
        let (mut contour, _n) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let extra = insert(SubMesh::new(cfg, ElementType::SEG2));
        contour.add_submesh(extra).unwrap();
        assert!(Mesh::fill_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_with_one_hole_2d() {
        // 4×4 outer square with a 2×2 inner hole centred at (2, 2).
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _no) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _nh) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let combined = (&outer + &hole).unwrap();
        assert_eq!(combined.submesh_count(), 2);

        let tri = Mesh::fill_surface(&combined, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);

        // Triangulated area must equal the outer area minus the hole: 16 - 4 = 12.
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW (signed area {})", ci, a);
            total += a;
        }
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_outer_loop_is_autodetected() {
        // Same as above but the outer loop is given **second**.
        let cfg = insert(Configuration::new(2).unwrap());
        let (hole, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let combined = (&hole + &outer).unwrap();
        let tri = Mesh::fill_surface(&combined, ElementType::TRI3, None).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        // Area is invariant w.r.t. submesh order.
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_two_holes_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)],
        );
        let (h1, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
        );
        let (h2, _) = build_contour_2d(
            cfg.clone(),
            &[(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)],
        );
        let combined = (&(&outer + &h1).unwrap() + &h2).unwrap();
        assert_eq!(combined.submesh_count(), 3);
        let tri = Mesh::fill_surface(&combined, ElementType::TRI3, None).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        // 6×4 - 1 - 1 = 22.
        assert!((total - 22.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_one_hole_3d() {
        // Outer + hole both in z = 1 plane.
        let cfg = insert(Configuration::new(3).unwrap());
        let (outer, _) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 1.0),
                (4.0, 0.0, 1.0),
                (4.0, 4.0, 1.0),
                (0.0, 4.0, 1.0),
            ],
        );
        let (hole, _) = build_contour_3d(
            cfg.clone(),
            &[
                (1.0, 1.0, 1.0),
                (3.0, 1.0, 1.0),
                (3.0, 3.0, 1.0),
                (1.0, 3.0, 1.0),
            ],
        );
        let combined = (&outer + &hole).unwrap();

        let tri = Mesh::fill_surface(&combined, ElementType::TRI3, None).unwrap();

        // Every triangle vertex must sit exactly on z = 1.
        let n_cells = tri.cell_count().unwrap();
        for ci in 0..n_cells {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 1.0).abs() < 1e-12);
            }
        }
        // Sum of 3-D triangle areas must equal 16 - 4 = 12.
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_hole_rejects_different_configurations() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg1.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _) = build_contour_2d(
            cfg2,
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        // merge() rejects mismatched configurations, so this should fail
        // before fill_surface ever sees a mixed contour.
        assert!((&outer + &hole).is_err());
    }

    #[test]
    fn fill_surface_rejects_open_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        // Three segments but not closed: a→b, b→c, c→a is closed; here we leave it open.
        let mut open = Mesh::with_element_type(cfg, ElementType::SEG2);
        open.add_cell(&[a.id(), b.id()]).unwrap();
        open.add_cell(&[b.id(), c.id()]).unwrap();
        // Missing the closing c→a segment.
        assert!(Mesh::fill_surface(&open, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_refined_2d_square_creates_steiner_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, contour_nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
        let initial_node_count = with(&cfg, |c| c.node_count()).unwrap();

        let opts = crate::triangulation::RefinementOptions {
            max_edge_length: Some(1.5),
            min_angle_deg: None,
        };
        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, Some(opts)).unwrap();

        // Steiner nodes were created in the Configuration.
        let new_node_count = with(&cfg, |c| c.node_count()).unwrap();
        assert!(
            new_node_count > initial_node_count,
            "no Steiner nodes added: was {}, still {}",
            initial_node_count,
            new_node_count
        );

        // Conservation of area and CCW orientation.
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        let mut max_edge = 0.0_f64;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW", ci);
            total += a;
            for (u, v) in [(p0.as_slice(), p1.as_slice()), (p1.as_slice(), p2.as_slice()), (p2.as_slice(), p0.as_slice())] {
                let dx = v[0] - u[0];
                let dy = v[1] - u[1];
                max_edge = max_edge.max((dx * dx + dy * dy).sqrt());
            }
        }
        assert!((total - 16.0).abs() < 1e-9);
        assert!(max_edge <= 1.5 + 1e-9, "max edge length {} > 1.5", max_edge);

        // Make sure the contour nodes still exist (they are referenced by the
        // new TRI3 mesh, the contour mesh, and the user-held Node handles).
        for n in &contour_nodes {
            assert!(with(&cfg, |c| c.is_alive(n.id())).unwrap());
        }
    }

    #[test]
    fn fill_surface_refined_inactive_options_is_noop() {
        // Passing Some(RefinementOptions::default()) should behave just
        // like None — no Steiner points, fast ear-clipping path.
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, _nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let initial_count = with(&cfg, |c| c.node_count()).unwrap();
        let tri =
            Mesh::fill_surface(&contour, ElementType::TRI3, Some(Default::default())).unwrap();
        let final_count = with(&cfg, |c| c.node_count()).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert_eq!(initial_count, final_count, "no Steiner expected");
    }

    #[test]
    fn fill_surface_refined_3d_keeps_steiner_in_plane() {
        // 4×4 square in the plane z = 1, refined by size: every Steiner
        // node must land on z = 1 exactly (within float precision).
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 1.0),
                (4.0, 0.0, 1.0),
                (4.0, 4.0, 1.0),
                (0.0, 4.0, 1.0),
            ],
        );
        let opts = crate::triangulation::RefinementOptions {
            max_edge_length: Some(1.5),
            min_angle_deg: None,
        };
        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, Some(opts)).unwrap();
        let n_cells = tri.cell_count().unwrap();
        assert!(n_cells > 2, "no refinement happened: got only {} cells", n_cells);
        for ci in 0..n_cells {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 1.0).abs() < 1e-9, "Steiner node off plane: z={}", p[2]);
            }
        }
    }

    #[test]
    fn fill_surface_refined_with_hole_conserves_area() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let combined = (&outer + &hole).unwrap();
        let opts = crate::triangulation::RefinementOptions {
            max_edge_length: Some(1.0),
            min_angle_deg: None,
        };
        let tri = Mesh::fill_surface(&combined, ElementType::TRI3, Some(opts)).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        // 16 - 4 = 12.
        assert!((total - 12.0).abs() < 1e-9, "area drift: {}", total);
    }

    #[test]
    fn fill_surface_works_with_cw_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        // Same square but listed clockwise.
        let (contour, _n) =
            build_contour_2d(cfg, &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);
        let tri = Mesh::fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        // Resulting triangles must still be CCW (positive signed area).
        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW", ci);
        }
    }

    #[test]
    fn consolidate_merges_same_type_submeshes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        // Deux sous-maillages TRI3 séparés avec une cellule chacun.
        let sm1 = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[b.id(), c.id(), a.id()]).unwrap(); // nouvelle cellule
            insert(sm)
        };

        let mut mesh = Mesh::new(cfg.clone());
        mesh.add_submesh(sm1).unwrap();
        mesh.add_submesh(sm2).unwrap();
        assert_eq!(mesh.submesh_count(), 2);

        let c2 = mesh.consolidate().unwrap();
        assert_eq!(c2.submesh_count(), 1, "doit fusionner les deux TRI3");
        assert_eq!(c2.cell_count().unwrap(), 2, "deux cellules distinctes conservées");
        assert_eq!(c2.element_types().unwrap(), vec![ElementType::TRI3]);
    }

    #[test]
    fn consolidate_removes_duplicate_cells() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let sm1 = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap(); // doublon exact
            insert(sm)
        };

        let mut mesh = Mesh::new(cfg.clone());
        mesh.add_submesh(sm1).unwrap();
        mesh.add_submesh(sm2).unwrap();

        let c2 = mesh.consolidate().unwrap();
        assert_eq!(c2.submesh_count(), 1);
        assert_eq!(c2.cell_count().unwrap(), 1, "le doublon doit être supprimé");
    }

    #[test]
    fn consolidate_preserves_distinct_types() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm_poi = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            insert(sm)
        };
        // Deuxième TRI3 avec doublon.
        let sm_tri2 = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };

        let mut mesh = Mesh::new(cfg.clone());
        mesh.add_submesh(sm_tri).unwrap();
        mesh.add_submesh(sm_poi).unwrap();
        mesh.add_submesh(sm_tri2).unwrap();

        let c2 = mesh.consolidate().unwrap();
        assert_eq!(c2.submesh_count(), 2, "TRI3 + POI1");
        assert_eq!(
            c2.element_types().unwrap(),
            vec![ElementType::TRI3, ElementType::POI1],
            "ordre premier-rencontré"
        );
        assert_eq!(c2.cell_counts().unwrap(), vec![1, 1]);
    }

    #[test]
    fn debug_and_display_submesh_and_mesh() {
        let cfg = insert(Configuration::new(1).unwrap());
        let sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
        let d = format!("{:?}", sm);
        let s = format!("{}", sm);
        assert!(d.contains("SubMesh"));
        assert!(s.contains("SEG2"));

        let mesh = Mesh::new(cfg);
        assert!(format!("{:?}", mesh).contains("Mesh"));
        assert!(format!("{}", mesh).contains("submesh"));
    }
}
