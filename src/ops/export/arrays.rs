//! Lay meshes and fields out as **flat arrays** — the one exit every exchange
//! format goes through (VTK, gmsh, MED).
//!
//! [`to_arrays`] is the mirror of
//! [`from_arrays`](crate::ops::mesh::arrays::from_arrays): its output has the
//! shape of that function's input — a node table, **blocks** of cells sharing
//! a type and a combination of groups (a MED family, a gmsh entity), then
//! field values keyed by the node and cell tags it hands out. Importing what
//! it exports gives the same groups, cells and values back.
//!
//! # Rules
//!
//! - All meshes share one `Coords`. The nodes they reference are numbered in
//!   order of **first appearance** (group after group, submesh after submesh,
//!   cell after cell), tags `first_tag..`; their coordinates keep the
//!   `Coords` dimension.
//! - A cell that sits in several groups — the same element type over the same
//!   nodes, as every gmsh or MED import produces — is exported **once**, in a
//!   block listing all its groups. The search only runs for the types present
//!   in two groups or more; the others go straight through.
//! - Cell tags are `first_tag..`, block after block; the connectivity is
//!   written in the requested [`NodeOrder`].
//! - A node field becomes one row per exported node (`0` where it defines
//!   nothing, the project's convention); the first zone defining a
//!   `(node, component)` wins.
//! - An element field becomes values on the cells its zones cover (zones on
//!   submeshes no exported mesh holds are left out): their **Gauss mean** per
//!   cell
//!   ([`ElementLayout::Cell`], what VTK shows), or the raw values per point
//!   ([`ElementLayout::Gauss`]) with the rule each type uses, declared in
//!   pyrucast's reference element renumbered in the external order — ready
//!   for [`gauss_to_external`](crate::ops::mesh::gauss_map::gauss_to_external)
//!   if the format has a reference element of its own.
//!
//! # Cost
//!
//! One pass numbers the nodes through a table indexed by `NodeId` (no
//! hashing), one locked gather fetches their positions, the connectivity is
//! permuted block by block; field values are copied zone by zone, one lock per
//! zone. Every array is allocated once, at its final size where it is known,
//! and handed over by move.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::element_field::ElementField;
use crate::containers::field::{Field, SubField};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::mesh::arrays::NodeOrder;
use std::collections::HashMap;

fn err(msg: impl Into<String>) -> PyrucastError {
    PyrucastError::Message(msg.into())
}

/// How an element field is laid out on export.
///
/// ```
/// # use pyrucast::ops::export::arrays::ElementLayout;
/// assert_ne!(ElementLayout::Cell, ElementLayout::Gauss);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementLayout {
    /// One value per cell and component: the mean over the cell's points.
    Cell,
    /// One value per Gauss point, component and cell, as stored.
    Gauss,
}

/// One block of exported cells: one type, one combination of groups.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let tri = Mesh::from_submesh(sm);
/// let out = to_arrays(&[("t".into(), &tri)], &[], &[], NodeOrder::Pyrucast, 1)?;
/// let b = &out.blocks[0];
/// assert_eq!((b.element_type, b.cell_tags.len()), (ElementType::TRI3, 1));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct ExportedBlock {
    pub element_type: ElementType,
    /// Flat connectivity in node tags, in the external order.
    pub node_tags: Vec<i64>,
    /// One tag per cell.
    pub cell_tags: Vec<i64>,
    pub groups: Vec<String>,
}

/// A node field: one row of `components.len()` values per exported node, in
/// the order of [`Exported::node_tags`].
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let tri = Mesh::from_submesh(sm);
/// let out = to_arrays(&[("t".into(), &tri)], &[], &[], NodeOrder::Pyrucast, 1)?;
/// assert!(out.node_fields.is_empty()); // none asked for
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct ExportedNodeField {
    pub components: Vec<String>,
    pub values: Vec<f64>,
}

/// The Gauss rule of one element type, in pyrucast's reference element with
/// its nodes listed in the external order.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let tri = Mesh::from_submesh(sm);
/// let out = to_arrays(&[("t".into(), &tri)], &[], &[], NodeOrder::Pyrucast, 1)?;
/// assert!(out.cell_fields.iter().all(|f| f.rules.is_empty()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct ExportedRule {
    pub element_type: ElementType,
    pub ref_nodes: Vec<f64>,
    pub xi: Vec<f64>,
    pub weights: Vec<f64>,
}

/// An element field on the cells `cell_tags`: per cell, `components.len()`
/// values (`Cell`), or `n_points × components.len()` in the order of its
/// type's rule (`Gauss`, one entry of `rules` per type met).
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let tri = Mesh::from_submesh(sm);
/// let out = to_arrays(&[("t".into(), &tri)], &[], &[], NodeOrder::Pyrucast, 1)?;
/// assert!(out.cell_fields.is_empty()); // none asked for
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct ExportedCellField {
    pub components: Vec<String>,
    pub cell_tags: Vec<i64>,
    pub values: Vec<f64>,
    pub rules: Vec<ExportedRule>,
}

/// What [`to_arrays`] lays out — the shape
/// [`from_arrays`](crate::ops::mesh::arrays::from_arrays) reads back.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let tri = Mesh::from_submesh(sm);
/// let out = to_arrays(&[("t".into(), &tri)], &[], &[], NodeOrder::Pyrucast, 1)?;
/// assert_eq!((out.dim, out.node_coords.len()), (2, 6));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct Exported {
    /// `first_tag..first_tag + n`.
    pub node_tags: Vec<i64>,
    /// `dim` coordinates per node.
    pub node_coords: Vec<f64>,
    pub dim: usize,
    pub blocks: Vec<ExportedBlock>,
    pub node_fields: Vec<ExportedNodeField>,
    pub cell_fields: Vec<ExportedCellField>,
}

/// One exported submesh: which group it came from, and the export cell each
/// of its cells became.
struct SubOut {
    handle: Handle<SubMesh>,
    cells: Vec<u32>,
}

/// Lay `groups` and the fields out as flat arrays — see the module
/// documentation for the rules.
///
/// Errors when the meshes do not share one `Coords`, when two zones of an
/// element field carry the same component on one submesh, or when two zones
/// of one element type use different Gauss rules.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::arrays::to_arrays;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # sm.add_cell(&[n[1].id(), n[3].id(), n[2].id()])?;
/// let plaque = Mesh::from_submesh(sm);
/// // The same mesh under two names: its cells are exported once, in a block
/// // naming both groups.
/// let out = to_arrays(
///     &[("plaque".into(), &plaque), ("tout".into(), &plaque)],
///     &[], &[], NodeOrder::Pyrucast, 1)?;
/// assert_eq!(out.node_tags, [1, 2, 3, 4]);
/// assert_eq!(out.blocks.len(), 1);
/// assert_eq!(out.blocks[0].groups, ["plaque", "tout"]);
/// assert_eq!(out.blocks[0].node_tags, [1, 2, 3, 2, 4, 3]);
/// assert_eq!(out.blocks[0].cell_tags, [1, 2]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn to_arrays(
    groups: &[(String, &Mesh)],
    node_fields: &[&NodeField],
    element_fields: &[(&ElementField, ElementLayout)],
    order: NodeOrder,
    first_tag: i64,
) -> Result<Exported> {
    // ── The shared Coords ───────────────────────────────────────────────────
    let mut coords = None;
    for (name, mesh) in groups {
        if mesh.is_empty() {
            continue;
        }
        let c = mesh.coords()?;
        match &coords {
            None => coords = Some(c),
            Some(first) if first.same_object(&c) => {}
            Some(_) => {
                return Err(err(format!(
                    "to_arrays: group {name} lives on another Coords than the first group"
                )));
            }
        }
    }
    let Some(coords) = coords else {
        return Ok(Exported {
            node_tags: Vec::new(),
            node_coords: Vec::new(),
            dim: 0,
            blocks: Vec::new(),
            node_fields: Vec::new(),
            cell_fields: Vec::new(),
        });
    };
    let c = coords.read();
    let dim = c.dim() as usize;

    // ── Submeshes, and the types that more than one group holds ─────────────
    let mut subs: Vec<(usize, Handle<SubMesh>)> = Vec::new();
    let mut groups_of_type: HashMap<ElementType, Vec<usize>> = HashMap::new();
    for (g, (_, mesh)) in groups.iter().enumerate() {
        for h in mesh.iter() {
            let et = h.read().element_type();
            let owners = groups_of_type.entry(et).or_default();
            if !owners.contains(&g) {
                owners.push(g);
            }
            subs.push((g, h.clone()));
        }
    }

    // ── Nodes, first seen first, through a table indexed by id ──────────────
    let mut row = vec![u32::MAX; c.capacity()];
    let mut ids: Vec<NodeId> = Vec::new();
    // ── Cells: export index, family (sorted group list) per cell ────────────
    // Families are interned: `family_of[f]` lists the groups of family `f`,
    // and adding group `g` to family `f` is memoised in `grown`.
    let mut families: Vec<Vec<usize>> = Vec::new();
    let mut family_id: HashMap<Vec<usize>, u32> = HashMap::new();
    let mut grown: HashMap<(u32, usize), u32> = HashMap::new();
    let mut intern = |list: Vec<usize>, families: &mut Vec<Vec<usize>>| -> u32 {
        *family_id.entry(list.clone()).or_insert_with(|| {
            families.push(list);
            (families.len() - 1) as u32
        })
    };
    let mut cell_type: Vec<ElementType> = Vec::new();
    let mut cell_family: Vec<u32> = Vec::new();
    let mut cell_nodes: Vec<(u32, u32)> = Vec::new(); // (sub, row in sub)
    let mut seen: HashMap<(ElementType, [u32; 27]), u32> = HashMap::new();
    let mut outs: Vec<SubOut> = Vec::with_capacity(subs.len());
    for (s, (g, h)) in subs.iter().enumerate() {
        let sm = h.read();
        let et = sm.element_type();
        let npc = et.nodes_per_cell();
        let shared = groups_of_type[&et].len() > 1;
        let own = intern(vec![*g], &mut families);
        let mut cells = Vec::with_capacity(sm.cell_count());
        for (k, cell) in sm.connectivity().chunks_exact(npc).enumerate() {
            for &id in cell {
                let r = &mut row[id.0 as usize];
                if *r == u32::MAX {
                    *r = ids.len() as u32;
                    ids.push(id);
                }
            }
            if shared {
                let mut key = [u32::MAX; 27];
                for (slot, id) in key.iter_mut().zip(cell) {
                    *slot = id.0;
                }
                key[..npc].sort_unstable();
                if let Some(&e) = seen.get(&(et, key)) {
                    let f = cell_family[e as usize];
                    if !families[f as usize].contains(g) {
                        let grown_f = match grown.get(&(f, *g)) {
                            Some(&x) => x,
                            None => {
                                let mut list = families[f as usize].clone();
                                list.push(*g);
                                list.sort_unstable();
                                let x = intern(list, &mut families);
                                grown.insert((f, *g), x);
                                x
                            }
                        };
                        cell_family[e as usize] = grown_f;
                    }
                    cells.push(e);
                    continue;
                }
                seen.insert((et, key), cell_type.len() as u32);
            }
            cells.push(cell_type.len() as u32);
            cell_type.push(et);
            cell_family.push(own);
            cell_nodes.push((s as u32, k as u32));
        }
        outs.push(SubOut {
            handle: h.clone(),
            cells,
        });
    }
    drop(seen);

    let n = ids.len();
    let node_tags: Vec<i64> = (0..n as i64).map(|k| first_tag + k).collect();
    let mut node_coords = vec![0.0; n * dim];
    c.positions_of(&ids, &mut node_coords);
    drop(c);

    // ── Blocks: cells grouped by (type, family), in order of first cell ─────
    let mut block_of: HashMap<(ElementType, u32), usize> = HashMap::new();
    let mut block_cells: Vec<Vec<u32>> = Vec::new();
    let mut block_key: Vec<(ElementType, u32)> = Vec::new();
    for (e, (&et, &f)) in cell_type.iter().zip(&cell_family).enumerate() {
        let b = *block_of.entry((et, f)).or_insert_with(|| {
            block_key.push((et, f));
            block_cells.push(Vec::new());
            block_key.len() - 1
        });
        block_cells[b].push(e as u32);
    }
    // Every submesh read-locked once for the whole layout.
    let sub_guards: Vec<_> = outs.iter().map(|o| o.handle.read()).collect();
    // Export cell → tag, block after block.
    let mut cell_tag = vec![0i64; cell_type.len()];
    let mut next_tag = first_tag;
    let mut blocks = Vec::with_capacity(block_cells.len());
    for ((et, f), cells) in block_key.iter().zip(&block_cells) {
        let npc = et.nodes_per_cell();
        let perm = order.permutation(*et);
        let mut conn = vec![0i64; cells.len() * npc];
        let mut tags = Vec::with_capacity(cells.len());
        for (dst, &e) in conn.chunks_exact_mut(npc).zip(cells) {
            let (s, k) = cell_nodes[e as usize];
            let sm = &sub_guards[s as usize];
            let cell = &sm.connectivity()[k as usize * npc..(k as usize + 1) * npc];
            // external[perm[i]] = pyrucast[i]
            for (i, &p) in perm.iter().enumerate() {
                dst[p] = first_tag + row[cell[i].0 as usize] as i64;
            }
            cell_tag[e as usize] = next_tag;
            tags.push(next_tag);
            next_tag += 1;
        }
        blocks.push(ExportedBlock {
            element_type: *et,
            node_tags: conn,
            cell_tags: tags,
            groups: families[*f as usize]
                .iter()
                .map(|&g| groups[g].0.clone())
                .collect(),
        });
    }

    drop(sub_guards);

    // ── Node fields ─────────────────────────────────────────────────────────
    let mut exported_nodes = Vec::with_capacity(node_fields.len());
    for field in node_fields {
        let components = field.components();
        let nc = components.len();
        let mut values = vec![0.0; n * nc];
        let mut written = vec![false; n * nc];
        for zone in field.iter() {
            let z = zone.read();
            let cols: Vec<usize> = z
                .components()
                .iter()
                .map(|name| {
                    components
                        .iter()
                        .position(|c| c == name)
                        .unwrap_or(usize::MAX)
                })
                .collect();
            let zc = cols.len();
            let support = z.support();
            let sup = support.read();
            for (id, vals) in sup.connectivity().iter().zip(z.values().chunks_exact(zc)) {
                let r = row.get(id.0 as usize).copied().unwrap_or(u32::MAX);
                if r == u32::MAX {
                    continue;
                }
                let base = r as usize * nc;
                for (&col, &v) in cols.iter().zip(vals) {
                    if !written[base + col] {
                        written[base + col] = true;
                        values[base + col] = v;
                    }
                }
            }
        }
        exported_nodes.push(ExportedNodeField { components, values });
    }

    // ── Element fields ──────────────────────────────────────────────────────
    let mut exported_cells = Vec::with_capacity(element_fields.len());
    for (field, layout) in element_fields {
        exported_cells.push(element_field(field, *layout, order, &outs, &cell_tag)?);
    }

    Ok(Exported {
        node_tags,
        node_coords,
        dim,
        blocks,
        node_fields: exported_nodes,
        cell_fields: exported_cells,
    })
}

/// One element field, zone by zone, onto the export cells.
fn element_field(
    field: &ElementField,
    layout: ElementLayout,
    order: NodeOrder,
    outs: &[SubOut],
    cell_tag: &[i64],
) -> Result<ExportedCellField> {
    let components = field.components();
    let nc = components.len();
    // Per export submesh: the zones on it, with their component columns.
    let mut per_sub: Vec<
        Vec<(
            Handle<crate::containers::element_field::SubElementField>,
            Vec<usize>,
        )>,
    > = vec![Vec::new(); outs.len()];
    for zone in field.iter() {
        let z = zone.read();
        let sm = z.support().read().submesh();
        // A zone on a submesh no exported mesh holds has no cell to land on.
        let Some(s) = outs.iter().position(|o| o.handle.same_object(&sm)) else {
            continue;
        };
        let cols: Vec<usize> = z
            .components()
            .iter()
            .map(|name| {
                components
                    .iter()
                    .position(|c| c == name)
                    .unwrap_or(usize::MAX)
            })
            .collect();
        for (_, other_cols) in &per_sub[s] {
            if cols.iter().any(|c| other_cols.contains(c)) {
                return Err(err(
                    "to_arrays: a component is carried by two zones on the same support — \
                     consolidate the field first (element_field::consolidate)",
                ));
            }
        }
        per_sub[s].push((zone.clone(), cols));
    }

    let mut cell_tags = Vec::new();
    let mut values = Vec::new();
    let mut rules: Vec<ExportedRule> = Vec::new();
    let mut done = vec![false; cell_tag.len()];
    for (s, zones) in per_sub.iter().enumerate() {
        let Some((first, _)) = zones.first() else {
            continue;
        };
        let (et, ng, rule) = {
            let z = first.read();
            let fes = z.support();
            let f = fes.read();
            let et = f.element_type();
            let ng = z.gauss_count();
            let rule = if layout == ElementLayout::Gauss {
                Some(f.quadrature().points(et)?)
            } else {
                None
            };
            (et, ng, rule)
        };
        if let Some((xi, w)) = rule {
            match rules.iter().find(|r| r.element_type == et) {
                Some(r) if r.weights != w || r.xi != xi => {
                    return Err(err(format!(
                        "to_arrays: two zones of type {et} use different Gauss rules"
                    )));
                }
                Some(_) => {}
                None => {
                    let kind = et.as_kind();
                    let d = kind.topological_dim();
                    let perm = order.permutation(et);
                    let pyr = kind.ref_nodes();
                    let mut ref_nodes = vec![0.0; pyr.len() * d];
                    for (i, &p) in perm.iter().enumerate() {
                        ref_nodes[p * d..(p + 1) * d].copy_from_slice(pyr[i]);
                    }
                    rules.push(ExportedRule {
                        element_type: et,
                        ref_nodes,
                        xi,
                        weights: w,
                    });
                }
            }
        }
        let per_cell = match layout {
            ElementLayout::Cell => nc,
            ElementLayout::Gauss => ng * nc,
        };
        let guards: Vec<_> = zones.iter().map(|(z, cols)| (z.read(), cols)).collect();
        for (k, &e) in outs[s].cells.iter().enumerate() {
            if std::mem::replace(&mut done[e as usize], true) {
                continue;
            }
            cell_tags.push(cell_tag[e as usize]);
            let at = values.len();
            values.resize(at + per_cell, 0.0);
            let dst = &mut values[at..];
            for (z, cols) in &guards {
                let zc = cols.len();
                let src = &z.values()[k * ng * zc..(k + 1) * ng * zc];
                match layout {
                    ElementLayout::Cell => {
                        for (j, &col) in cols.iter().enumerate() {
                            let mut acc = 0.0;
                            for g in 0..ng {
                                acc += src[g * zc + j];
                            }
                            dst[col] = if ng > 0 { acc / ng as f64 } else { 0.0 };
                        }
                    }
                    ElementLayout::Gauss => {
                        for g in 0..ng {
                            for (j, &col) in cols.iter().enumerate() {
                                dst[g * nc + col] = src[g * zc + j];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ExportedCellField {
        components,
        cell_tags,
        values,
        rules,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::coords::Coords;
    use crate::ops::mesh::arrays::{
        from_arrays, CellBlock, CellLayout, CellValues, GaussRule, NodeValues,
    };

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// A square of two TRI3 in "plate", one of which also in "corner", and a
    /// SEG2 in "bottom" — imported from arrays, as any exchange format would.
    fn imported() -> Vec<(String, Mesh)> {
        let tags = [1_i64, 2, 3, 4];
        let xyz = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (plate, both, bottom) = (
            names(&["plate"]),
            names(&["plate", "corner"]),
            names(&["bottom"]),
        );
        let blocks = [
            CellBlock {
                element_type: ElementType::TRI3,
                node_tags: &[1, 2, 3],
                cell_tags: &[],
                groups: &both,
            },
            CellBlock {
                element_type: ElementType::TRI3,
                node_tags: &[1, 3, 4],
                cell_tags: &[],
                groups: &plate,
            },
            CellBlock {
                element_type: ElementType::SEG2,
                node_tags: &[1, 2],
                cell_tags: &[],
                groups: &bottom,
            },
        ];
        let c = Handle::new(Coords::new(2).unwrap());
        from_arrays(c, &tags, &xyz, &blocks, &[], &[], NodeOrder::Pyrucast)
            .unwrap()
            .groups
    }

    /// Group → zones → cells as scaled coordinates.
    type Geometry = Vec<(String, Vec<(ElementType, Vec<Vec<[i64; 2]>>)>)>;

    fn view(groups: &[(String, Mesh)]) -> Vec<(&str, &Mesh)> {
        groups.iter().map(|(n, m)| (n.as_str(), m)).collect()
    }

    /// Group name → (type, sorted cells as sorted coordinate lists): what a
    /// mesh *is*, independently of any numbering.
    fn geometry(groups: &[(String, Mesh)]) -> Geometry {
        let mut out: Vec<_> = groups
            .iter()
            .map(|(name, mesh)| {
                let c = mesh.coords().unwrap();
                let c = c.read();
                let mut zones: Vec<_> = mesh
                    .iter()
                    .map(|h| {
                        let s = h.read();
                        let npc = s.element_type().nodes_per_cell();
                        let mut cells: Vec<Vec<[i64; 2]>> = s
                            .connectivity()
                            .chunks(npc)
                            .map(|cell| {
                                cell.iter()
                                    .map(|&id| {
                                        let p = c.position(id).unwrap();
                                        [(p[0] * 8.0) as i64, (p[1] * 8.0) as i64]
                                    })
                                    .collect()
                            })
                            .collect();
                        cells.sort();
                        (s.element_type(), cells)
                    })
                    .collect();
                zones.sort_by_key(|z| z.0.to_string());
                (name.clone(), zones)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn reimport(out: &Exported, order: NodeOrder) -> Vec<(String, Mesh)> {
        let blocks: Vec<CellBlock<'_, i64>> = out
            .blocks
            .iter()
            .map(|b| CellBlock {
                element_type: b.element_type,
                node_tags: &b.node_tags,
                cell_tags: &b.cell_tags,
                groups: &b.groups,
            })
            .collect();
        let c = Handle::new(Coords::new(out.dim as u8).unwrap());
        from_arrays(
            c,
            &out.node_tags,
            &out.node_coords,
            &blocks,
            &[],
            &[],
            order,
        )
        .unwrap()
        .groups
    }

    #[test]
    fn a_cell_of_two_groups_is_exported_once() {
        let groups = imported();
        let pairs: Vec<(String, &Mesh)> = groups.iter().map(|(n, m)| (n.clone(), m)).collect();
        let out = to_arrays(&pairs, &[], &[], NodeOrder::Pyrucast, 1).unwrap();
        let cells: usize = out.blocks.iter().map(|b| b.cell_tags.len()).sum();
        assert_eq!(cells, 3);
        let shared = out.blocks.iter().find(|b| b.groups.len() == 2).unwrap();
        assert_eq!(shared.groups, ["plate", "corner"]);
        assert_eq!(shared.cell_tags.len(), 1);
        let _ = view(&groups);
    }

    #[test]
    fn to_then_from_arrays_gives_the_mesh_back_in_every_order() {
        let groups = imported();
        let pairs: Vec<(String, &Mesh)> = groups.iter().map(|(n, m)| (n.clone(), m)).collect();
        for order in [NodeOrder::Pyrucast, NodeOrder::Gmsh, NodeOrder::Med] {
            let out = to_arrays(&pairs, &[], &[], order, 0).unwrap();
            assert_eq!(
                geometry(&reimport(&out, order)),
                geometry(&groups),
                "{order:?}"
            );
        }
    }

    #[test]
    fn a_med_tetrahedron_round_trips_through_the_permutation() {
        // Exported in MED order the base is reversed; read back, it is not.
        let tags = [1_i64, 2, 3, 4];
        let xyz = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let t = names(&["t"]);
        let blocks = [CellBlock {
            element_type: ElementType::TET4,
            node_tags: &tags,
            cell_tags: &[],
            groups: &t,
        }];
        let c = Handle::new(Coords::new(3).unwrap());
        let groups = from_arrays(c, &tags, &xyz, &blocks, &[], &[], NodeOrder::Pyrucast)
            .unwrap()
            .groups;
        let out = to_arrays(&[("t".into(), &groups[0].1)], &[], &[], NodeOrder::Med, 1).unwrap();
        assert_eq!(out.blocks[0].node_tags, [1, 3, 2, 4]);
    }

    #[test]
    fn fields_round_trip_with_absent_values_as_zero() {
        let tags = [1_i64, 2, 3, 4];
        let xyz = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let plate = names(&["plate"]);
        let blocks = [CellBlock {
            element_type: ElementType::TRI3,
            node_tags: &[1, 2, 3, 1, 3, 4],
            cell_tags: &[7, 8],
            groups: &plate,
        }];
        let t = names(&["T"]);
        // Node 4 left out: exported as 0.
        let nv = NodeValues {
            components: &t,
            node_tags: &[1, 2, 3],
            values: &[1.0, 2.0, 3.0],
        };
        let (xi, w) = ElementType::TRI3.as_kind().gauss();
        let rules = [GaussRule {
            element_type: ElementType::TRI3,
            ref_nodes: &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            xi: &xi,
            weights: &w,
        }];
        let s = names(&["s"]);
        let cv = CellValues {
            components: &s,
            cell_tags: &[7, 8],
            values: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            layout: CellLayout::Gauss(&rules),
        };
        let c = Handle::new(Coords::new(2).unwrap());
        let imp = from_arrays(c, &tags, &xyz, &blocks, &[nv], &[cv], NodeOrder::Pyrucast).unwrap();
        let pairs = [("plate".to_string(), &imp.groups[0].1)];
        let ef = &imp.element_fields[0];
        let out = to_arrays(
            &pairs,
            &[&imp.node_fields[0]],
            &[(ef, ElementLayout::Cell), (ef, ElementLayout::Gauss)],
            NodeOrder::Pyrucast,
            1,
        )
        .unwrap();
        assert_eq!(out.node_fields[0].values, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(out.cell_fields[0].values, [2.0, 5.0]); // Gauss means
        assert_eq!(out.cell_fields[1].values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out.cell_fields[1].rules[0].weights, w);
        assert_eq!(out.cell_fields[0].cell_tags, out.blocks[0].cell_tags);
    }

    #[test]
    fn zones_of_other_meshes_are_left_out() {
        let groups = imported();
        let fes = FiniteElementSpace::lagrange1(&groups[1].1).unwrap(); // "corner"
        let ef = ElementField::new(&fes, names(&["s"])).unwrap();
        // Export "bottom" only: the field has no cell there.
        let out = to_arrays(
            &[("bottom".into(), &groups[2].1)],
            &[],
            &[(&ef, ElementLayout::Cell)],
            NodeOrder::Pyrucast,
            1,
        )
        .unwrap();
        assert!(out.cell_fields[0].cell_tags.is_empty());
    }
}
