//! Sweep / extrusion mesh-construction operators.
//!
//! - [`qua4_between`] — sweep between two SEG2 contours, producing a
//!   strip of QUA4 elements (one per source segment per layer).
//! - [`extrude`] — translate every cell of a mesh along a vector by
//!   `n_layers` layers, bumping element dimension (SEG2 → QUA4,
//!   TRI3 → PENTA6, QUA4 → HEX8).
//! - [`solid_between`] — the 3-D companion of [`qua4_between`]: sweep
//!   between two matching surface meshes (TRI3 → PENTA6, QUA4 → HEX8).

use crate::aggregate::Aggregate;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::Node;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};

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
    let coords_a = read(&sm_a)?.coords();
    let coords_b = read(&sm_b)?.coords();
    if coords_a.index() != coords_b.index() || coords_a.generation() != coords_b.generation() {
        return Err(PyrucastError::Message(
            "sweep: meshes are attached to different Coords".into(),
        ));
    }

    let (et_a, n_elems, conn_a) = {
        let s = read(&sm_a)?;
        (s.element_type(), s.cell_count(), s.connectivity().to_vec())
    };
    let (et_b, n_elems_b, conn_b) = {
        let s = read(&sm_b)?;
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
        .map(|&id| -> Result<Vec<f64>> { Ok(read(&coords)?.coord(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    let coords_b: Vec<Vec<f64>> = col_ids_b
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(read(&coords)?.coord(id)?.to_vec()) })
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

    let coords = mesh.coords()?;

    // Collect unique NodeIds across all submeshes, first-seen order.
    let mut col_map: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    let mut ordered_ids: Vec<NodeId> = Vec::new();
    for sm in mesh {
        for id in read(sm)?.connectivity().to_vec() {
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
        .map(|&id| -> Result<Vec<f64>> { Ok(read(&coords)?.coord(id)?.to_vec()) })
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
            .map(|&id| Node::acquire(coords.clone(), id))
            .collect::<Result<Vec<_>>>()?,
    );
    for k in 1..=n_layers {
        let layer: Vec<Node> = (0..n_cols)
            .map(|j| {
                let coord: Vec<f64> = base_coords[j]
                    .iter()
                    .zip(step.iter())
                    .map(|(&c, &s)| c + k as f64 * s)
                    .collect();
                Node::create_in(coords.clone(), &coord)
            })
            .collect::<Result<_>>()?;
        layers.push(layer);
    }

    let col = |id: NodeId| *col_map.get(&id).unwrap();

    let mut result = Mesh::empty();

    for sm_handle in mesh {
        let (et, n_cells, conn) = {
            let s = read(sm_handle)?;
            (s.element_type(), s.cell_count(), s.connectivity().to_vec())
        };
        let npc = et.nodes_per_cell();

        let extruded_et = match et {
            ElementType::SEG2 => ElementType::QUA4,
            ElementType::TRI3 => ElementType::PENTA6,
            ElementType::QUA4 => ElementType::HEX8,
            _ => {
                return Err(PyrucastError::Message(format!(
                    "extrude: cannot extrude {} elements (supported: SEG2, TRI3, QUA4)",
                    et
                )))
            }
        };

        let mut sm_out = SubMesh::new(coords.clone(), extruded_et);

        for k in 0..n_layers {
            for ci in 0..n_cells {
                let cell = &conn[ci * npc..(ci + 1) * npc];
                let bot: Vec<NodeId> = cell.iter().map(|&id| layers[k][col(id)].id()).collect();
                let top: Vec<NodeId> = cell.iter().map(|&id| layers[k + 1][col(id)].id()).collect();

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

        result.add_sub(insert(sm_out))?;
    }

    Ok(result)
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
    let coords = read(&sm_a)?.coords();
    let coords_b = read(&sm_b)?.coords();
    if coords.index() != coords_b.index() || coords.generation() != coords_b.generation() {
        return Err(PyrucastError::Message(
            "sweep_solid: meshes are attached to different Coords".into(),
        ));
    }

    let (et_a, conn_a) = {
        let s = read(&sm_a)?;
        (s.element_type(), s.connectivity().to_vec())
    };
    let (et_b, conn_b) = {
        let s = read(&sm_b)?;
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
        .map(|&id| -> Result<Vec<f64>> { Ok(read(&coords)?.coord(id)?.to_vec()) })
        .collect::<Result<_>>()?;
    let base_b: Vec<Vec<f64>> = cols_b
        .iter()
        .map(|&id| -> Result<Vec<f64>> { Ok(read(&coords)?.coord(id)?.to_vec()) })
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
