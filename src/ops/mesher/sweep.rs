//! Sweep / extrusion mesh-construction operators.
//!
//! - [`qua4_between`] — sweep between two SEG2 contours, producing a
//!   strip of QUA4 elements (one per source segment per layer).
//! - [`extrude`] — translate every cell of a mesh along a vector by
//!   `n_layers` layers, bumping element dimension (SEG2 → QUA4,
//!   QUA4 → HEX8).

use crate::mesh::configuration::NodeId;
use crate::error::{PyrucastError, Result};
use crate::mesh::element_type::ElementType;
use crate::mesh::node::Node;
use crate::mesh::{Mesh, SubMesh};
use crate::store::{insert, with};

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
    let sm_a = mesh_a.submesh(0)?;
    let sm_b = mesh_b.submesh(0)?;
    let cfg_a = with(&sm_a, |s| s.configuration())?;
    let cfg_b = with(&sm_b, |s| s.configuration())?;
    if cfg_a.index() != cfg_b.index() || cfg_a.generation() != cfg_b.generation() {
        return Err(PyrucastError::Message(
            "sweep_qua4: meshes are attached to different Configurations".into(),
        ));
    }

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

    let cfg = cfg_a;
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

    let cfg = mesh.configuration()?;

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

    let mut result = Mesh::empty();

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
