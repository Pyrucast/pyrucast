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

        // Normalize the normal vector.
        let n_norm: f64 = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n_norm < 1e-15 {
            return Err(PyrucastError::Message(
                "circle_seg2: normal vector must not be zero".into(),
            ));
        }
        let n = [normal[0] / n_norm, normal[1] / n_norm, normal[2] / n_norm];

        // In-plane basis: pick e as the coordinate axis least aligned with n,
        // then orthogonalise (Gram-Schmidt) to get u, and v = n × u.
        let abs_n = [n[0].abs(), n[1].abs(), n[2].abs()];
        let e: [f64; 3] = if abs_n[0] <= abs_n[1] && abs_n[0] <= abs_n[2] {
            [1.0, 0.0, 0.0]
        } else if abs_n[1] <= abs_n[2] {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let e_dot_n = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
        let u_raw = [e[0] - e_dot_n * n[0], e[1] - e_dot_n * n[1], e[2] - e_dot_n * n[2]];
        let u_norm: f64 = u_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u = [u_raw[0] / u_norm, u_raw[1] / u_norm, u_raw[2] / u_norm];
        let v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];

        // Create n_elems evenly spaced nodes on the circle.
        let cx = center_coords.first().copied().unwrap_or(0.0);
        let cy = center_coords.get(1).copied().unwrap_or(0.0);
        let cz = center_coords.get(2).copied().unwrap_or(0.0);
        let mut nodes: Vec<Node> = Vec::with_capacity(n_elems);
        for i in 0..n_elems {
            let theta = 2.0 * PI * i as f64 / n_elems as f64;
            let (cos_t, sin_t) = (theta.cos(), theta.sin());
            let p3 = [
                cx + radius * (cos_t * u[0] + sin_t * v[0]),
                cy + radius * (cos_t * u[1] + sin_t * v[1]),
                cz + radius * (cos_t * u[2] + sin_t * v[2]),
            ];
            nodes.push(Node::create_in(cfg.clone(), &p3[..dim])?);
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
            let handle_a = mesh_a.handle.clone();
            let handle_b = mesh_b.handle.clone();
            let mesh = with(&handle_a, |a| {
                with(&handle_b, |b| Mesh::sweep_qua4(a, b, n_layers))
            })???;
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

        fn __add__(&self, other: PyRef<PyMesh>) -> PyResult<PyMesh> {
            let other_handle = other.handle.clone();
            let mesh = with(&self.handle, |a| {
                with(&other_handle, |b| a.merge(b))
            })???;
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
