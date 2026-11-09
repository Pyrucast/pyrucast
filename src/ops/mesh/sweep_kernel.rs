//! Sweep / extrusion mesh-construction operators.
//!
//! - [`grid_between`] — sweep between two contours (SEG2 → QUA4, SEG3 →
//!   QUA8/QUA9), one cell per source segment per layer; two closed loops
//!   close the strip.
//! - [`extrude`] — translate every cell of a mesh along a vector by
//!   `n_layers` layers, bumping element dimension (SEG2 → QUA4,
//!   TRI3 → PENTA6, QUA4 → HEX8).
//! - [`revolve`] — the same layered sweep, but rotating about a centre
//!   (2-D) or an axis (3-D); a full turn closes the ring.
//! - [`solid_between`] — the 3-D companion of [`grid_between`]: sweep
//!   between two matching surface meshes (TRI3 → PENTA6, QUA4 → HEX8).

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// The node row of a line mesh, read in order along it.
///
/// A SEG2 line of `n` cells gives its `n + 1` corners. A SEG3 line gives
/// `2n + 1` nodes — corner, mid-node, corner, … — the mid-node of cell `j`
/// (local index 2) sitting between its two corners. The cells must follow
/// each other (the end of cell `j` is the start of cell `j + 1`), as
/// `line`, `arc`, `circle`, `border` and `consolidate` lay them out.
fn line_row(conn: &[NodeId], et: ElementType, n_elems: usize) -> Vec<NodeId> {
    let npc = et.nodes_per_cell();
    let mut row = Vec::with_capacity(if npc == 3 {
        2 * n_elems + 1
    } else {
        n_elems + 1
    });
    row.push(conn[0]);
    for cell in conn.chunks(npc) {
        if npc == 3 {
            row.push(cell[2]);
        }
        row.push(cell[1]);
    }
    row
}

/// Sweep between two line meshes — both SEG2, or both SEG3 — into a
/// structured strip of `element_type`: `QUA4` from SEG2 lines, `QUA8` or
/// `QUA9` from SEG3 lines.
///
/// The two lines are read as node rows ([`line_row`]) and interpolated
/// linearly, row node by row node, at evenly spaced levels between them:
/// `n_layers + 1` levels for QUA4, `2·n_layers + 1` for the quadratic types,
/// whose odd levels carry the mid-nodes of the edges running from one line
/// to the other. Level 0 is `mesh_a`'s own nodes and the last level
/// `mesh_b`'s, re-used (refcount incremented); the mid-nodes of SEG3 lines
/// are re-used too, so the strip follows a curved line exactly.
///
/// **Two closed loops close the strip**: when each line ends on its own first
/// node, the last column of every level *is* the first — the seam between
/// the two loops is shared, not doubled. A closed loop swept onto an open
/// line is an error.
///
/// Node order per cell (`mesh_a` side first), with `(level, column)`:
/// QUA4 `(k, j) (k, j+1) (k+1, j+1) (k+1, j)`; QUA8 the same corners on the
/// even levels and columns, then the mid-nodes of the edges in the order of
/// the QUA4 edges (`0-1`, `1-2`, `2-3`, `3-0`); QUA9 adds the centre.
pub fn grid_between(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    n_layers: usize,
    element_type: ElementType,
) -> Result<Mesh> {
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

    // Guards held for the read pass; the `Coords` write comes after.
    let (a, b) = (sm_a.read(), sm_b.read());
    let (et_a, n_elems, conn_a) = (a.element_type(), a.cell_count(), a.connectivity());
    let (et_b, n_elems_b, conn_b) = (b.element_type(), b.cell_count(), b.connectivity());

    let line_type = match element_type {
        ElementType::QUA4 => ElementType::SEG2,
        ElementType::QUA8 | ElementType::QUA9 => ElementType::SEG3,
        other => {
            return Err(PyrucastError::Message(format!(
                "sweep: cannot build {other} between two lines (expected QUA4, QUA8 or QUA9)"
            )))
        }
    };
    for (name, et) in [("mesh_a", et_a), ("mesh_b", et_b)] {
        if !matches!(et, ElementType::SEG2 | ElementType::SEG3) {
            return Err(PyrucastError::Message(format!(
                "sweep: {name} must be a SEG2 or SEG3 line, got {et}"
            )));
        }
    }
    if et_a != et_b {
        return Err(PyrucastError::Message(format!(
            "sweep: mesh_a is {et_a} but mesh_b is {et_b} — both lines must be of the same degree"
        )));
    }
    if et_a != line_type {
        return Err(PyrucastError::Message(format!(
            "sweep: {element_type} is built between {line_type} lines, got {et_a}"
        )));
    }
    if n_elems != n_elems_b {
        return Err(PyrucastError::Message(format!(
            "sweep: mesh_a has {} elements but mesh_b has {}",
            n_elems, n_elems_b
        )));
    }

    let coords = coords_a;
    let row_a = line_row(conn_a, et_a, n_elems);
    let row_b = line_row(conn_b, et_b, n_elems);
    let last = row_a.len() - 1;
    let closed_a = row_a[0] == row_a[last];
    let closed_b = row_b[0] == row_b[last];
    if closed_a != closed_b {
        let (closed, open) = if closed_a {
            ("mesh_a", "mesh_b")
        } else {
            ("mesh_b", "mesh_a")
        };
        return Err(PyrucastError::Message(format!(
            "sweep: {closed} is a closed loop but {open} is open"
        )));
    }
    let closed = closed_a;
    // Columns that own their nodes: the last one of a closed strip is the
    // first one again.
    let n_cols = if closed { last } else { last + 1 };

    let quadratic = line_type == ElementType::SEG3;
    let n_levels = if quadratic { 2 * n_layers } else { n_layers };
    let with_centre = element_type == ElementType::QUA9;

    // The two rows' positions, flat, `dim` numbers per column.
    let dim = coords.read().dim() as usize;
    let mut pos_a: Vec<f64> = Vec::with_capacity(n_cols * dim);
    let mut pos_b: Vec<f64> = Vec::with_capacity(n_cols * dim);
    {
        let c = coords.read();
        for col in 0..n_cols {
            pos_a.extend_from_slice(c.position(row_a[col])?);
            pos_b.extend_from_slice(c.position(row_b[col])?);
        }
    }

    // The fine grid of the strip, `width` columns per level: every node the
    // cells will read, decided once, so the cell loop below only indexes.
    // Levels 0 and `n_levels` are the two lines; the levels in between are
    // created here, laid out flat and added in one locked pass —
    // interpolating a million nodes must not mean a million write locks.
    // A QUA8 has no node where an odd level crosses an odd column (the middle
    // of a cell), a QUA9 has: odd levels of a QUA8 strip take every other
    // column.
    let width = last + 1;
    let mut grid: Vec<NodeId> = vec![NodeId(u32::MAX); (n_levels + 1) * width];
    grid[..n_cols].copy_from_slice(&row_a[..n_cols]);
    grid[n_levels * width..n_levels * width + n_cols].copy_from_slice(&row_b[..n_cols]);
    let mut flat: Vec<f64> = Vec::with_capacity(n_levels.saturating_sub(1) * n_cols * dim);
    let mut created: u32 = 0;
    for level in 1..n_levels {
        let t = level as f64 / n_levels as f64;
        let stride = if quadratic && level % 2 == 1 && !with_centre {
            2
        } else {
            1
        };
        for col in (0..n_cols).step_by(stride) {
            let (a, b) = (
                &pos_a[col * dim..(col + 1) * dim],
                &pos_b[col * dim..(col + 1) * dim],
            );
            flat.extend(a.iter().zip(b).map(|(&ca, &cb)| ca + t * (cb - ca)));
            // The rank among the created nodes, turned into an id below.
            grid[level * width + col] = NodeId(created);
            created += 1;
        }
    }
    let first = coords.write().add_nodes(&flat)?.start;
    for level in 1..n_levels {
        let stride = if quadratic && level % 2 == 1 && !with_centre {
            2
        } else {
            1
        };
        for col in (0..n_cols).step_by(stride) {
            grid[level * width + col].0 += first;
        }
    }
    // A closed strip's last column is its first.
    if closed {
        for level in 0..=n_levels {
            grid[level * width + last] = grid[level * width];
        }
    }
    let at = |level: usize, col: usize| grid[level * width + col];

    let npc = element_type.nodes_per_cell();
    let mut conn: Vec<NodeId> = Vec::with_capacity(n_layers * n_elems * npc);
    if quadratic {
        for k in 0..n_layers {
            let (l0, lm, l1) = (2 * k, 2 * k + 1, 2 * k + 2);
            for j in 0..n_elems {
                let (c0, cm, c1) = (2 * j, 2 * j + 1, 2 * j + 2);
                // The QUA9 order; a QUA8 keeps its first eight (its grid has
                // no centre: that slot holds a placeholder, dropped here).
                let cell = [
                    at(l0, c0),
                    at(l0, c1),
                    at(l1, c1),
                    at(l1, c0),
                    at(l0, cm),
                    at(lm, c1),
                    at(l1, cm),
                    at(lm, c0),
                    at(lm, cm),
                ];
                conn.extend_from_slice(&cell[..npc]);
            }
        }
    } else {
        for k in 0..n_layers {
            for j in 0..n_elems {
                conn.extend_from_slice(&[at(k, j), at(k, j + 1), at(k + 1, j + 1), at(k + 1, j)]);
            }
        }
    }
    let sm = SubMesh::from_connectivity(coords.clone(), element_type, conn)?;
    // The connectivity owns the fresh nodes now; hand back the unit
    // `add_nodes` handed us.
    let owned: Vec<NodeId> = (first..first + created).map(NodeId).collect();
    coords.write().decref_all(&owned)?;
    Ok(Mesh::from_submesh(sm))
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
        for &id in sm.read().connectivity() {
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

/// The cell type a layered sweep turns `et` into, or why it cannot.
fn swept_type(et: ElementType, op: &str) -> Result<ElementType> {
    Ok(match et {
        ElementType::SEG2 => ElementType::QUA4,
        ElementType::TRI3 => ElementType::PENTA6,
        ElementType::QUA4 => ElementType::HEX8,
        ElementType::SEG3 => ElementType::QUA8,
        ElementType::TRI6 => ElementType::PENTA15,
        ElementType::QUA8 => ElementType::HEX20,
        _ => {
            return Err(PyrucastError::Message(format!(
                "{op}: cannot sweep {et} elements \
                 (supported: SEG2, TRI3, QUA4 and their quadratic SEG3, TRI6, QUA8)"
            )))
        }
    })
}

/// Sweep `mesh` into `n_layers` layers of cells, the node of a column at
/// level `s` sitting where `place(base_position, s, flat)` puts it — the
/// placement appends the position to `flat`, the buffer the nodes are created
/// from, so no position is allocated on its own.
///
/// Layer 0 re-uses the source nodes (refcount incremented); levels `1..` are
/// newly created — except when `closed` is set, where the last level *is*
/// level 0 again, so a sweep that comes back onto its start (a full turn)
/// closes on itself instead of duplicating a node layer.
///
/// A **quadratic** source (SEG3, TRI6, QUA8) gives quadratic cells: every
/// node of the source — corners and mid-nodes — is copied at the whole
/// levels, and its corners alone also at the half levels `k + ½`, where the
/// mid-nodes of the edges running along the sweep sit. `place` decides where
/// a half level is: halfway along a translation, at the half angle of a
/// rotation — on the arc. Linear and quadratic zones cannot be mixed.
///
/// Each submesh of `mesh` yields one submesh of the result, cells ordered
/// layer by layer. Element mapping and node ordering (`bot`/`top` the two
/// whole levels, `v` the half level between them, at a corner):
/// - SEG2 → QUA4: `bot[0], bot[1], top[1], top[0]`
/// - TRI3 → PENTA6: `bot[0..3], top[0..3]`
/// - QUA4 → HEX8: `bot[0..4], top[0..4]`
/// - SEG3 → QUA8: `bot[0], bot[1], top[1], top[0], bot[2], v[1], top[2], v[0]`
/// - TRI6 → PENTA15: `bot[0..3], top[0..3], bot[3..6], top[3..6], v[0..3]`
/// - QUA8 → HEX20: `bot[0..4], top[0..4], bot[4..8], top[4..8], v[0..4]`
///
/// — the mid-nodes in the edge order of the linear cell (`EDGES` of
/// `qua4.rs`, `penta6.rs`, `hex8.rs`).
fn layered(
    mesh: &Mesh,
    cols: &Columns,
    n_layers: usize,
    closed: bool,
    op: &str,
    place: impl Fn(&[f64], f64, &mut Vec<f64>),
) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let n_cols = cols.ids.len();
    let dim = coords.read().dim() as usize;

    // Each zone's swept type, the degree of the sweep, and which columns are
    // corners: only those get a node at the half levels.
    let mut swept: Vec<(ElementType, ElementType)> = Vec::new();
    let mut quadratic: Option<bool> = None;
    let mut corner_rank: Vec<u32> = vec![u32::MAX; n_cols];
    let mut n_corner_cols: u32 = 0;
    for sm_handle in mesh {
        let s = sm_handle.read();
        let et = s.element_type();
        let swept_et = swept_type(et, op)?;
        if swept_et.topological_dim() > dim {
            return Err(PyrucastError::Message(format!(
                "{op}: sweeping {et} elements produces {swept_et}, which needs 3-D coordinates \
                 (the mesh is {dim}-D)"
            )));
        }
        let is_quadratic = et.as_kind().is_quadratic();
        if quadratic.is_some_and(|q| q != is_quadratic) {
            return Err(PyrucastError::Message(format!(
                "{op}: linear and quadratic zones cannot be swept together"
            )));
        }
        quadratic = Some(is_quadratic);
        swept.push((et, swept_et));
        let (npc, corners) = (et.nodes_per_cell(), et.as_kind().corner_count());
        for cell in s.connectivity().chunks(npc) {
            for &id in &cell[..corners] {
                let col = cols.of(id);
                if corner_rank[col] == u32::MAX {
                    corner_rank[col] = n_corner_cols;
                    n_corner_cols += 1;
                }
            }
        }
    }
    let quadratic = quadratic.unwrap_or(false);
    let n_half = if quadratic { n_layers } else { 0 };

    // Whole levels 1..=last_created for every column, then the half levels
    // for the corner columns — laid out flat and created in one locked pass:
    // a swept solid is where a node-at-a-time creation costs the most.
    let last_created = if closed { n_layers - 1 } else { n_layers };
    let n_created = last_created * n_cols + n_half * n_corner_cols as usize;
    let mut flat: Vec<f64> = Vec::with_capacity(n_created * dim);
    for k in 1..=last_created {
        for c in 0..n_cols {
            place(&cols.positions[c], k as f64, &mut flat);
        }
    }
    let mut corners_in_order: Vec<usize> = vec![0; n_corner_cols as usize];
    for (col, &rank) in corner_rank.iter().enumerate() {
        if rank != u32::MAX {
            corners_in_order[rank as usize] = col;
        }
    }
    for k in 0..n_half {
        for &c in &corners_in_order {
            place(&cols.positions[c], k as f64 + 0.5, &mut flat);
        }
    }
    let first = coords.write().add_nodes(&flat)?.start;

    // The node of every column at every level, decided once, so the cell
    // loop below only indexes: `whole` at the whole levels 0..=n_layers (a
    // closed sweep's last level is its first), `halves` at the half levels
    // k + ½, for the corner columns.
    let mut whole: Vec<NodeId> = Vec::with_capacity((n_layers + 1) * n_cols);
    whole.extend_from_slice(&cols.ids);
    whole.extend((0..last_created * n_cols).map(|i| NodeId(first + i as u32)));
    if closed {
        whole.extend_from_slice(&cols.ids);
    }
    let first_half = first + (last_created * n_cols) as u32;
    let mut halves: Vec<NodeId> = vec![NodeId(u32::MAX); n_half * n_cols];
    for k in 0..n_half {
        for (col, &rank) in corner_rank.iter().enumerate() {
            if rank != u32::MAX {
                halves[k * n_cols + col] = NodeId(first_half + k as u32 * n_corner_cols + rank);
            }
        }
    }

    let mut result = Mesh::empty();
    for (sm_handle, &(et, swept_et)) in mesh.into_iter().zip(&swept) {
        // Guard held for this zone's pass: what follows only reads.
        let s = sm_handle.read();
        let npc = et.nodes_per_cell();
        let pattern = sweep_pattern(et);
        // The zone's connectivity as column indices, looked up once rather
        // than once per layer.
        let conn_cols: Vec<usize> = s.connectivity().iter().map(|&id| cols.of(id)).collect();

        let mut out: Vec<NodeId> =
            Vec::with_capacity(n_layers * s.cell_count() * swept_et.nodes_per_cell());
        for k in 0..n_layers {
            let levels: [&[NodeId]; 3] = [
                &whole[k * n_cols..(k + 1) * n_cols],
                &whole[(k + 1) * n_cols..(k + 2) * n_cols],
                if quadratic {
                    &halves[k * n_cols..(k + 1) * n_cols]
                } else {
                    &[]
                },
            ];
            for cell in conn_cols.chunks(npc) {
                out.extend(pattern.iter().map(|&(level, i)| levels[level][cell[i]]));
            }
        }

        result.add_sub(Handle::new(SubMesh::from_connectivity(
            coords.clone(),
            swept_et,
            out,
        )?))?;
    }

    // The swept zones own the fresh nodes now; hand back the unit `add_nodes`
    // handed us.
    let owned: Vec<NodeId> = (first..first + n_created as u32).map(NodeId).collect();
    coords.write().decref_all(&owned)?;
    Ok(result)
}

/// The cell a layered sweep builds from a source cell, as `(level, local)`
/// pairs in the swept cell's node order: level 0 the layer's bottom, 1 its
/// top, 2 its half level; `local` the source cell's node.
fn sweep_pattern(et: ElementType) -> &'static [(usize, usize)] {
    const B: usize = 0;
    const T: usize = 1;
    const H: usize = 2;
    match et {
        ElementType::SEG2 => &[(B, 0), (B, 1), (T, 1), (T, 0)],
        ElementType::TRI3 => &[(B, 0), (B, 1), (B, 2), (T, 0), (T, 1), (T, 2)],
        ElementType::QUA4 => &[
            (B, 0),
            (B, 1),
            (B, 2),
            (B, 3),
            (T, 0),
            (T, 1),
            (T, 2),
            (T, 3),
        ],
        ElementType::SEG3 => &[
            (B, 0),
            (B, 1),
            (T, 1),
            (T, 0),
            (B, 2),
            (H, 1),
            (T, 2),
            (H, 0),
        ],
        ElementType::TRI6 => &[
            (B, 0),
            (B, 1),
            (B, 2),
            (T, 0),
            (T, 1),
            (T, 2),
            (B, 3),
            (B, 4),
            (B, 5),
            (T, 3),
            (T, 4),
            (T, 5),
            (H, 0),
            (H, 1),
            (H, 2),
        ],
        // QUA8, the last type `swept_type` admits.
        _ => &[
            (B, 0),
            (B, 1),
            (B, 2),
            (B, 3),
            (T, 0),
            (T, 1),
            (T, 2),
            (T, 3),
            (B, 4),
            (B, 5),
            (B, 6),
            (B, 7),
            (T, 4),
            (T, 5),
            (T, 6),
            (T, 7),
            (H, 0),
            (H, 1),
            (H, 2),
            (H, 3),
        ],
    }
}

/// Extrude a mesh by `n_layers` layers along `direction`.
///
/// `direction` is the **total** displacement vector; each intermediate
/// layer is placed at an evenly spaced fraction. Supported element types:
/// SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8, and their quadratic siblings
/// SEG3 → QUA8, TRI6 → PENTA15, QUA8 → HEX20 — the source's mid-nodes are
/// copied on every layer, and each edge along the extrusion gets its
/// mid-node halfway. Other types produce an error.
///
/// Nodes shared between cells in the source mesh remain shared in the
/// extruded mesh. Source nodes are re-used (refcount incremented);
/// intermediate layer nodes are newly created.
///
/// Node ordering:
/// - QUA4: `bot[0], bot[1], top[1], top[0]`
/// - PENTA6: `bot[0..3], top[0..3]`
/// - HEX8: `bot[0..4], top[0..4]`
/// - QUA8, PENTA15, HEX20: the corners as above, then the mid-nodes in the
///   edge order of the linear cell — those of the bottom face, of the top
///   face, then of the edges along the extrusion.
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
    layered(
        mesh,
        &cols,
        n_layers,
        false,
        "extrude",
        |base, level, out| {
            out.extend(base.iter().zip(step.iter()).map(|(&c, &s)| c + level * s));
        },
    )
}

/// Revolve a mesh by `n_layers` layers over a total `angle` (radians) — the
/// rotational companion of [`extrude`].
///
/// Every node is swept along the circle it describes about the rotation
/// centre (2-D) or axis (3-D), and consecutive angular positions are linked
/// by one layer of cells: SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8, and in
/// quadratic SEG3 → QUA8, TRI6 → PENTA15, QUA8 → HEX20 — the mid-node of an
/// edge along the sweep sits at the half angle, on the arc.
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
    // The rotation is chosen once, outside the per-node placement.
    match unit_axis {
        None => layered(
            mesh,
            &cols,
            n_layers,
            closed,
            "revolve",
            |base, level, out| {
                let (sin, cos) = (step * level).sin_cos();
                let (x, y) = (base[0] - center[0], base[1] - center[1]);
                out.extend_from_slice(&[
                    center[0] + cos * x - sin * y,
                    center[1] + sin * x + cos * y,
                ]);
            },
        ),
        Some(u) => layered(
            mesh,
            &cols,
            n_layers,
            closed,
            "revolve",
            |base, level, out| {
                let (sin, cos) = (step * level).sin_cos();
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
                let along = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) * (1.0 - cos);
                out.extend_from_slice(&[
                    center[0] + v[0] * cos + cross[0] * sin + u[0] * along,
                    center[1] + v[1] * cos + cross[1] * sin + u[1] * along,
                    center[2] + v[2] * cos + cross[2] * sin + u[2] * along,
                ]);
            },
        ),
    }
}

/// Sweep between two matching surface meshes to produce a solid mesh —
/// the 3-D companion of [`grid_between`].
///
/// Cell `i` of `mesh_a` is paired with cell `i` of `mesh_b`, node by node
/// (local node `j` of one matches local node `j` of the other), just as
/// [`grid_between`] pairs two SEG2 contours. `n_layers` ≥ 1 layers of solid
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

    // Guards held for the read pass; the `Coords` write comes after.
    let (a, b) = (sm_a.read(), sm_b.read());
    let (et_a, conn_a) = (a.element_type(), a.connectivity());
    let (et_b, conn_b) = (b.element_type(), b.connectivity());
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

    // Layer 0 is `mesh_a`'s own nodes and layer `n_layers` is `mesh_b`'s; the
    // interpolated layers in between are laid out flat and created in one
    // locked pass.
    let inner = n_layers.saturating_sub(1);
    let mut flat: Vec<f64> = Vec::new();
    for k in 1..n_layers {
        let t = k as f64 / n_layers as f64;
        for c in 0..n_cols {
            flat.extend(
                base_a[c]
                    .iter()
                    .zip(base_b[c].iter())
                    .map(|(&ca, &cb)| ca + t * (cb - ca)),
            );
        }
    }
    let first = coords.write().add_nodes(&flat)?.start;
    // Node at layer k, column col.
    let at = |k: usize, c: usize| -> NodeId {
        if k == 0 {
            cols_a[c]
        } else if k == n_layers {
            cols_b[c]
        } else {
            NodeId(first + ((k - 1) * n_cols + c) as u32)
        }
    };

    let col = |id: NodeId| *col_map.get(&id).unwrap();
    let mut out: Vec<NodeId> = Vec::with_capacity(n_layers * n_cells * solid_et.nodes_per_cell());
    for k in 0..n_layers {
        for ci in 0..n_cells {
            let cell = &conn_a[ci * npc..(ci + 1) * npc];
            let bot = |i: usize| at(k, col(cell[i]));
            let top = |i: usize| at(k + 1, col(cell[i]));
            match et_a {
                ElementType::TRI3 => {
                    out.extend_from_slice(&[bot(0), bot(1), bot(2), top(0), top(1), top(2)]);
                }
                ElementType::QUA4 => {
                    out.extend_from_slice(&[
                        bot(0),
                        bot(1),
                        bot(2),
                        bot(3),
                        top(0),
                        top(1),
                        top(2),
                        top(3),
                    ]);
                }
                _ => unreachable!(),
            }
        }
    }
    let sm_out = SubMesh::from_connectivity(coords.clone(), solid_et, out)?;
    // The solid owns the fresh nodes now; hand back the unit `add_nodes`
    // handed us.
    let owned: Vec<NodeId> = (first..first + (inner * n_cols) as u32)
        .map(NodeId)
        .collect();
    coords.write().decref_all(&owned)?;
    Ok(Mesh::from_submesh(sm_out))
}
