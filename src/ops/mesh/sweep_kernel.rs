//! Sweep / extrusion mesh-construction operators.
//!
//! - [`qua4_between`] — sweep between two SEG2 contours, producing a
//!   strip of QUA4 elements (one per source segment per layer).
//! - [`extrude`] — translate every cell of a mesh along a vector by
//!   `n_layers` layers, bumping element dimension (SEG2 → QUA4,
//!   TRI3 → PENTA6, QUA4 → HEX8).
//! - [`revolve`] — the same layered sweep, but rotating about a centre
//!   (2-D) or an axis (3-D); a full turn closes the ring.
//! - [`solid_between`] — the 3-D companion of [`qua4_between`]: sweep
//!   between two matching surface meshes (TRI3 → PENTA6, QUA4 → HEX8).

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::Handle;

/// Sweep between two SEG2 contour meshes to produce a QUA4 strip.
///
/// `n_layers` ≥ 1 transverse layers are interpolated between `mesh_a`
/// and `mesh_b` to produce the intermediate node layers. Endpoint
/// nodes from both meshes are re-used (refcount incremented);
/// intermediate nodes are created at evenly spaced positions.
///
/// QUA4 node order per element (counterclockwise, `mesh_a` side first):
/// `[k][j]`, `[k][j+1]`, `[k+1][j+1]`, `[k+1][j]`.
pub fn qua4_between(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
    if n_layers == 0 {
        return Err(PyrucastError::Message("sweep: n_layers must be ≥ 1".into()));
    }
    if mesh_a.len() != 1 {
        return Err(PyrucastError::Message(
            "sweep: mesh_a must have exactly one submesh".into(),
        ));
    }
    if mesh_b.len() != 1 {
        return Err(PyrucastError::Message(
            "sweep: mesh_b must have exactly one submesh".into(),
        ));
    }
    let sm_a = mesh_a.get(0)?;
    let sm_b = mesh_b.get(0)?;
    let coords_a = sm_a.read().coords();
    let coords_b = sm_b.read().coords();
    if !coords_a.same_object(&coords_b) {
        return Err(PyrucastError::Message(
            "sweep: meshes are attached to different Coords".into(),
        ));
    }

    let (et_a, n_elems, conn_a) = {
        let s = sm_a.read();
        (s.element_type(), s.cell_count(), s.connectivity().to_vec())
    };
    let (et_b, n_elems_b, conn_b) = {
        let s = sm_b.read();
        (s.element_type(), s.cell_count(), s.connectivity().to_vec())
    };

    if et_a != ElementType::SEG2 {
        return Err(PyrucastError::Message(
            "sweep: mesh_a must be a SEG2 mesh".into(),
        ));
    }
    if et_b != ElementType::SEG2 {
        return Err(PyrucastError::Message(
            "sweep: mesh_b must be a SEG2 mesh".into(),
        ));
    }
    if n_elems != n_elems_b {
        return Err(PyrucastError::Message(format!(
            "sweep: mesh_a has {} elements but mesh_b has {}",
            n_elems, n_elems_b
        )));
    }

    let coords = coords_a;
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
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    let coords_b: Vec<Vec<f64>> = col_ids_b
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;

    // layers[k][j] = Node at layer k, column j.
    // Layer 0 = re-acquired mesh_a nodes; layer n_layers = re-acquired mesh_b nodes.
    let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n_layers + 1);

    layers.push(
        col_ids_a
            .iter()
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );
    for k in 1..n_layers {
        let t = k as f64 / n_layers as f64;
        let layer: Vec<Node> = (0..n_cols)
            .map(|j| {
                let coord: Vec<f64> = coords_a[j]
                    .iter()
                    .zip(coords_b[j].iter())
                    .map(|(&ca, &cb)| ca + t * (cb - ca))
                    .collect();
                Node::create_in(coords.clone(), &coord)
            })
            .collect::<Result<_>>()?;
        layers.push(layer);
    }
    layers.push(
        col_ids_b
            .iter()
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
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

// ---------------------------------------------------------------------------
// Layered sweeps (`extrude`, `revolve`)
// ---------------------------------------------------------------------------

/// The node columns of a mesh: every distinct node, once.
///
/// A layered sweep works column by column — each column is one node of the
/// source mesh, replicated once per layer — so nodes shared between cells of
/// the source stay shared in the swept mesh.
struct Columns {
    /// Distinct node ids, in first-seen connectivity order.
    ids: Vec<NodeId>,
    /// Column index of each id.
    index: std::collections::HashMap<NodeId, usize>,
    /// Current position of each column.
    positions: Vec<Vec<f64>>,
}

impl Columns {
    /// Column index of `id` (every id of the source mesh has one).
    fn of(&self, id: NodeId) -> usize {
        self.index[&id]
    }
}

/// Collect the node columns of `mesh`, erroring out (as `op`) if it is empty.
fn columns(mesh: &Mesh, op: &str) -> Result<Columns> {
    let coords = mesh.coords()?;
    let mut index: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    let mut ids: Vec<NodeId> = Vec::new();
    for sm in mesh {
        for id in sm.read().connectivity().to_vec() {
            index.entry(id).or_insert_with(|| {
                let col = ids.len();
                ids.push(id);
                col
            });
        }
    }
    if ids.is_empty() {
        return Err(PyrucastError::Message(format!("{op}: mesh has no cells")));
    }
    let positions: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    Ok(Columns {
        ids,
        index,
        positions,
    })
}

/// Sweep `mesh` into `n_layers` layers of cells, layer `k` sitting where
/// `place(base_position, k)` puts it.
///
/// Layer 0 re-uses the source nodes (refcount incremented); layers `1..` are
/// newly created — except when `closed` is set, where the last layer *is*
/// layer 0 again, so a sweep that comes back onto its start (a full turn)
/// closes on itself instead of duplicating a node layer.
///
/// Each submesh of `mesh` yields one submesh of the result, cells ordered
/// layer by layer. Element mapping and node ordering:
/// - SEG2 → QUA4: `bot[0], bot[1], top[1], top[0]`
/// - TRI3 → PENTA6: `bot[0..3], top[0..3]`
/// - QUA4 → HEX8: `bot[0..4], top[0..4]`
fn layered(
    mesh: &Mesh,
    cols: &Columns,
    n_layers: usize,
    closed: bool,
    op: &str,
    place: impl Fn(&[f64], usize) -> Vec<f64>,
) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let n_cols = cols.ids.len();

    // layers[k][col] = Node at layer k, column col.
    let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n_layers + 1);
    layers.push(
        cols.ids
            .iter()
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );
    let last_created = if closed { n_layers - 1 } else { n_layers };
    for k in 1..=last_created {
        let layer: Vec<Node> = (0..n_cols)
            .map(|c| Node::create_in(coords.clone(), &place(&cols.positions[c], k)))
            .collect::<Result<_>>()?;
        layers.push(layer);
    }
    if closed {
        layers.push(
            cols.ids
                .iter()
                .map(|&id| Node::acquire(coords.clone(), id))
                .collect::<Result<Vec<_>>>()?,
        );
    }

    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, n_cells, conn) = {
            let s = sm_handle.read();
            (s.element_type(), s.cell_count(), s.connectivity().to_vec())
        };
        let npc = et.nodes_per_cell();

        let swept_et = match et {
            ElementType::SEG2 => ElementType::QUA4,
            ElementType::TRI3 => ElementType::PENTA6,
            ElementType::QUA4 => ElementType::HEX8,
            _ => {
                return Err(PyrucastError::Message(format!(
                    "{op}: cannot sweep {et} elements (supported: SEG2, TRI3, QUA4)"
                )))
            }
        };
        let dim = coords.read().dim() as usize;
        if swept_et.topological_dim() > dim {
            return Err(PyrucastError::Message(format!(
                "{op}: sweeping {et} elements produces {swept_et}, which needs 3-D coordinates \
                 (the mesh is {dim}-D)"
            )));
        }

        let mut sm_out = SubMesh::new(coords.clone(), swept_et);
        for k in 0..n_layers {
            for ci in 0..n_cells {
                let cell = &conn[ci * npc..(ci + 1) * npc];
                let bot: Vec<NodeId> = cell.iter().map(|&id| layers[k][cols.of(id)].id()).collect();
                let top: Vec<NodeId> = cell
                    .iter()
                    .map(|&id| layers[k + 1][cols.of(id)].id())
                    .collect();

                match et {
                    ElementType::SEG2 => {
                        sm_out.add_cell(&[bot[0], bot[1], top[1], top[0]])?;
                    }
                    ElementType::TRI3 => {
                        sm_out.add_cell(&[bot[0], bot[1], bot[2], top[0], top[1], top[2]])?;
                    }
                    ElementType::QUA4 => {
                        sm_out.add_cell(&[
                            bot[0], bot[1], bot[2], bot[3], top[0], top[1], top[2], top[3],
                        ])?;
                    }
                    _ => unreachable!(),
                }
            }
        }

        result.add_sub(Handle::new(sm_out))?;
    }

    Ok(result)
}

/// Extrude a mesh by `n_layers` layers along `direction`.
///
/// `direction` is the **total** displacement vector; each intermediate
/// layer is placed at an evenly spaced fraction. Supported element types:
/// SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8. Other types produce an error.
///
/// Nodes shared between cells in the source mesh remain shared in the
/// extruded mesh. Source nodes are re-used (refcount incremented);
/// intermediate layer nodes are newly created.
///
/// Node ordering:
/// - QUA4: `bot[0], bot[1], top[1], top[0]`
/// - PENTA6: `bot[0..3], top[0..3]`
/// - HEX8: `bot[0..4], top[0..4]`
pub fn extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Result<Mesh> {
    if n_layers == 0 {
        return Err(PyrucastError::Message(
            "extrude: n_layers must be ≥ 1".into(),
        ));
    }
    let dim = mesh.coords()?.read().dim() as usize;
    if direction.len() != dim {
        return Err(PyrucastError::Message(format!(
            "extrude: direction has {} components but node dimension is {}",
            direction.len(),
            dim
        )));
    }

    let cols = columns(mesh, "extrude")?;
    let step: Vec<f64> = direction.iter().map(|&d| d / n_layers as f64).collect();
    layered(mesh, &cols, n_layers, false, "extrude", |base, k| {
        base.iter()
            .zip(step.iter())
            .map(|(&c, &s)| c + k as f64 * s)
            .collect()
    })
}

/// Revolve a mesh by `n_layers` layers over a total `angle` (radians) — the
/// rotational companion of [`extrude`].
///
/// Every node is swept along the circle it describes about the rotation
/// centre (2-D) or axis (3-D), and consecutive angular positions are linked
/// by one layer of cells: SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8.
///
/// - **2-D** (`center` has 2 components): revolution about the point
///   `center`, counterclockwise for a positive `angle`; `axis` is ignored.
///   Only SEG2 sources make sense there (a surface would sweep a 3-D solid,
///   which 2-D coordinates cannot hold).
/// - **3-D** (`center` has 3 components): revolution about the line through
///   `center` directed by `axis` (Rodrigues' formula, right-handed about
///   `axis`); `axis` is required and need not be normalized.
///
/// `angle` runs up to a full turn (`|angle| ≤ 2π`); a full turn **closes**
/// the sweep — the last node layer is the first one again, so the ring has
/// no seam and no duplicated nodes.
///
/// No node may sit on the axis: it would collapse one edge of every cell
/// touching it into a degenerate (zero-Jacobian) element.
///
/// Nodes shared between cells in the source mesh remain shared in the result.
/// Source nodes are re-used (refcount incremented); the other layers are
/// newly created. Node ordering per layer is [`extrude`]'s.
///
/// A negative `angle` sweeps the cells the other way round and turns them
/// inside out, just as an `extrude` against the surface normal does — call
/// [`orient`](fn@super::orient) on the result, or revolve by a positive
/// angle from the mirrored source.
pub fn revolve(
    mesh: &Mesh,
    angle: f64,
    n_layers: usize,
    center: &[f64],
    axis: Option<&[f64]>,
) -> Result<Mesh> {
    const TWO_PI: f64 = std::f64::consts::TAU;
    // Relative slack on the full-turn test: enough to catch a 360° written
    // as `2 * PI` or accumulated in degrees, tight enough not to swallow a
    // deliberately-not-quite-closed sweep.
    const TURN_TOL: f64 = 1e-9;

    if n_layers == 0 {
        return Err(PyrucastError::Message(
            "revolve: n_layers must be ≥ 1".into(),
        ));
    }
    if !angle.is_finite() || angle == 0.0 {
        return Err(PyrucastError::Message(
            "revolve: angle must be a non-zero, finite number of radians".into(),
        ));
    }
    if angle.abs() > TWO_PI * (1.0 + TURN_TOL) {
        return Err(PyrucastError::Message(format!(
            "revolve: angle {angle} rad exceeds a full turn (|angle| ≤ 2π); \
             revolve by 2π to close the ring"
        )));
    }
    let closed = (angle.abs() - TWO_PI).abs() <= TWO_PI * TURN_TOL;

    let dim = mesh.coords()?.read().dim() as usize;
    if center.len() != dim {
        return Err(PyrucastError::Message(format!(
            "revolve: center has {} components but the mesh is {}-D",
            center.len(),
            dim
        )));
    }

    // Unit axis (3-D only); in 2-D the rotation is fully set by `center`.
    let unit_axis: Option<[f64; 3]> = match dim {
        2 => None,
        3 => {
            let axis = axis.ok_or_else(|| {
                PyrucastError::Message("revolve: a 3-D mesh needs a rotation axis".into())
            })?;
            if axis.len() != 3 {
                return Err(PyrucastError::Message(format!(
                    "revolve: axis has {} components but must be 3-D",
                    axis.len()
                )));
            }
            let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if norm == 0.0 {
                return Err(PyrucastError::Message(
                    "revolve: axis must be non-zero".into(),
                ));
            }
            Some([axis[0] / norm, axis[1] / norm, axis[2] / norm])
        }
        other => {
            return Err(PyrucastError::Message(format!(
                "revolve: only 2-D and 3-D meshes are supported (got {other}-D)"
            )))
        }
    };

    let cols = columns(mesh, "revolve")?;

    // Every node must be off the axis, else the cells touching it collapse.
    // The test is relative to the widest radius, so it is scale-free.
    let radius = |p: &[f64]| -> f64 {
        match unit_axis {
            None => ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2)).sqrt(),
            Some(u) => {
                let v = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                let along = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                (v.iter().map(|x| x * x).sum::<f64>() - along * along)
                    .max(0.0)
                    .sqrt()
            }
        }
    };
    let radii: Vec<f64> = cols.positions.iter().map(|p| radius(p)).collect();
    let max_radius = radii.iter().cloned().fold(0.0, f64::max);
    if max_radius == 0.0 || radii.iter().any(|&r| r <= 1e-10 * max_radius) {
        return Err(PyrucastError::Message(
            "revolve: some node lies on the rotation axis, which would collapse \
             the cells touching it (move the source off the axis)"
                .into(),
        ));
    }

    let step = angle / n_layers as f64;
    layered(mesh, &cols, n_layers, closed, "revolve", |base, k| {
        let theta = step * k as f64;
        let (cos, sin) = (theta.cos(), theta.sin());
        match unit_axis {
            None => {
                let (x, y) = (base[0] - center[0], base[1] - center[1]);
                vec![center[0] + cos * x - sin * y, center[1] + sin * x + cos * y]
            }
            Some(u) => {
                let v = [
                    base[0] - center[0],
                    base[1] - center[1],
                    base[2] - center[2],
                ];
                // Rodrigues: v' = v cosθ + (u×v) sinθ + u (u·v)(1−cosθ).
                let cross = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
                let k = dot * (1.0 - cos);
                vec![
                    center[0] + v[0] * cos + cross[0] * sin + u[0] * k,
                    center[1] + v[1] * cos + cross[1] * sin + u[1] * k,
                    center[2] + v[2] * cos + cross[2] * sin + u[2] * k,
                ]
            }
        }
    })
}

/// Sweep between two matching surface meshes to produce a solid mesh —
/// the 3-D companion of [`qua4_between`].
///
/// Cell `i` of `mesh_a` is paired with cell `i` of `mesh_b`, node by node
/// (local node `j` of one matches local node `j` of the other), just as
/// [`qua4_between`] pairs two SEG2 contours. `n_layers` ≥ 1 layers of solid
/// cells are built between the two faces; intermediate node layers are
/// linearly interpolated. Endpoint nodes from both meshes are re-used
/// (refcount incremented); intermediate nodes are created.
///
/// Both meshes must be single-submesh meshes of the **same** surface
/// element type (TRI3 or QUA4), with the same number of cells, attached to
/// the same `Coords`. The node correspondence read off the connectivity
/// must be consistent — a node shared by two cells of `mesh_a` must map to
/// the same node of `mesh_b` in both.
///
/// Element mapping and node ordering (bottom face = `mesh_a` side):
/// - TRI3 → PENTA6: `bot[0..3], top[0..3]`
/// - QUA4 → HEX8: `bot[0..4], top[0..4]`
pub fn solid_between(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
    if n_layers == 0 {
        return Err(PyrucastError::Message(
            "sweep_solid: n_layers must be ≥ 1".into(),
        ));
    }
    if mesh_a.len() != 1 || mesh_b.len() != 1 {
        return Err(PyrucastError::Message(
            "sweep_solid: both meshes must have exactly one submesh".into(),
        ));
    }
    let sm_a = mesh_a.get(0)?;
    let sm_b = mesh_b.get(0)?;
    let coords = sm_a.read().coords();
    let coords_b = sm_b.read().coords();
    if !coords.same_object(&coords_b) {
        return Err(PyrucastError::Message(
            "sweep_solid: meshes are attached to different Coords".into(),
        ));
    }

    let (et_a, conn_a) = {
        let s = sm_a.read();
        (s.element_type(), s.connectivity().to_vec())
    };
    let (et_b, conn_b) = {
        let s = sm_b.read();
        (s.element_type(), s.connectivity().to_vec())
    };
    if et_a != et_b {
        return Err(PyrucastError::Message(format!(
            "sweep_solid: element types differ ({et_a} vs {et_b})"
        )));
    }
    let solid_et = match et_a {
        ElementType::TRI3 => ElementType::PENTA6,
        ElementType::QUA4 => ElementType::HEX8,
        other => {
            return Err(PyrucastError::Message(format!(
                "sweep_solid: unsupported surface type {other} (supported: TRI3, QUA4)"
            )))
        }
    };
    let npc = et_a.nodes_per_cell();
    let n_cells = conn_a.len() / npc;
    if conn_b.len() / npc != n_cells {
        return Err(PyrucastError::Message(format!(
            "sweep_solid: mesh_a has {} cells but mesh_b has {}",
            n_cells,
            conn_b.len() / npc
        )));
    }

    // Node correspondence: unique mesh_a node → paired mesh_b node, plus a
    // stable column order (first-seen). A shared node must map consistently.
    let mut pair: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let mut col_map: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    let mut cols_a: Vec<NodeId> = Vec::new();
    let mut cols_b: Vec<NodeId> = Vec::new();
    for (&a_id, &b_id) in conn_a.iter().zip(conn_b.iter()) {
        match pair.get(&a_id) {
            Some(&existing) if existing != b_id => {
                return Err(PyrucastError::Message(
                    "sweep_solid: inconsistent node correspondence between the two meshes".into(),
                ));
            }
            Some(_) => {}
            None => {
                pair.insert(a_id, b_id);
                col_map.insert(a_id, cols_a.len());
                cols_a.push(a_id);
                cols_b.push(b_id);
            }
        }
    }

    let n_cols = cols_a.len();
    let base_a: Vec<Vec<f64>> = cols_a
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    let base_b: Vec<Vec<f64>> = cols_b
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(coords.read().position(id)?.to_vec()) })
        .collect::<Result<_>>()?;

    // layers[k][col]: layer 0 re-acquires mesh_a nodes, layer n_layers
    // re-acquires mesh_b nodes, intermediate layers are interpolated.
    let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n_layers + 1);
    layers.push(
        cols_a
            .iter()
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );
    for k in 1..n_layers {
        let t = k as f64 / n_layers as f64;
        let layer: Vec<Node> = (0..n_cols)
            .map(|c| {
                let coord: Vec<f64> = base_a[c]
                    .iter()
                    .zip(base_b[c].iter())
                    .map(|(&ca, &cb)| ca + t * (cb - ca))
                    .collect();
                Node::create_in(coords.clone(), &coord)
            })
            .collect::<Result<_>>()?;
        layers.push(layer);
    }
    layers.push(
        cols_b
            .iter()
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );

    let col = |id: NodeId| *col_map.get(&id).unwrap();
    let mut sm_out = SubMesh::new(coords, solid_et);
    for k in 0..n_layers {
        for ci in 0..n_cells {
            let cell = &conn_a[ci * npc..(ci + 1) * npc];
            let bot: Vec<NodeId> = cell.iter().map(|&id| layers[k][col(id)].id()).collect();
            let top: Vec<NodeId> = cell.iter().map(|&id| layers[k + 1][col(id)].id()).collect();
            match et_a {
                ElementType::TRI3 => {
                    sm_out.add_cell(&[bot[0], bot[1], bot[2], top[0], top[1], top[2]])?;
                }
                ElementType::QUA4 => {
                    sm_out.add_cell(&[
                        bot[0], bot[1], bot[2], bot[3], top[0], top[1], top[2], top[3],
                    ])?;
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(Mesh::from_submesh(sm_out))
}
