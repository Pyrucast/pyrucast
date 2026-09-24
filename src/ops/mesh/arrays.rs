//! Build meshes and fields from **flat arrays** — the one entry every
//! exchange format goes through.
//!
//! A mesh arrives as a node table (tags and coordinates) and **blocks** of
//! cells. A block holds one element type and one **combination of groups** —
//! exactly a MED family, or a gmsh entity with its physical groups — so the
//! group membership is resolved once per block, never per cell. Fields arrive
//! as values keyed by node tags or cell tags.
//!
//! [`from_arrays`] turns that into **one [`Mesh`] per group**, all sharing the
//! caller's [`Coords`], plus the requested [`NodeField`]s and
//! [`ElementField`]s. The external node order of each cell is realigned on
//! pyrucast's through the [`NodeOrder`] the caller names. The mirror image,
//! from pyrucast back to arrays, is
//! [`to_arrays`](crate::ops::export::arrays::to_arrays): its output is shaped
//! like this module's input.
//!
//! # Cost
//!
//! Built for large meshes. The arrays are **borrowed**, and the work goes in
//! passes over whole blocks: one pass translates the tags to rows and counts
//! every node's occurrences — which gives both "is this node used" and its
//! final refcount — then the used nodes are created in one go, with their
//! positions written straight into the store, and each block's connectivity
//! is rewritten **in place** into ids in pyrucast's order. No allocation and
//! no hashing per cell; the permutation is fetched once per block.
//!
//! # Rules
//!
//! - Only nodes referenced by a cell of some group are created.
//! - The dimension of `coords` decides how many coordinates are kept: the
//!   node table's stride (`len(coords) / len(tags)`, 1 to 3) is truncated or
//!   padded with zeros to it.
//! - Groups come out in order of first appearance over the blocks; inside a
//!   group, one [`SubMesh`] per element type, in order of first appearance.
//! - A node field lives on a POI1 support holding exactly the nodes it defines
//!   (those the mesh created). A cell field gets a zone on every group
//!   submesh **all** of whose cells it defines — `Cell` values on a one-point
//!   (`Reduced`) FE space, `Gauss` values on the full rule, once
//!   [`match_gauss`] has proven the
//!   external rule is pyrucast's. All the fields of one import share their
//!   FE spaces.

use crate::aggregate::Aggregate;
use crate::atoms::element_kind::identity_permutation;
use crate::atoms::{ElementType, NodeId, QuadratureRule};
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::mesh::gauss_map::match_gauss;
use std::collections::HashMap;

fn err(msg: impl Into<String>) -> PyrucastError {
    PyrucastError::Message(msg.into())
}

// ─── Node order ──────────────────────────────────────────────────────────────

/// The local node numbering an external format uses inside a cell. pyrucast's
/// own is VTK's; gmsh and MED differ on some types, and the permutation that
/// realigns them is a property of the element
/// ([`ElementKind::gmsh_permutation`](crate::atoms::ElementKind::gmsh_permutation),
/// [`ElementKind::med_permutation`](crate::atoms::ElementKind::med_permutation)).
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// // MED walks the base of a tetrahedron the other way round.
/// assert_eq!(NodeOrder::Med.permutation(ElementType::TET4), [0, 2, 1, 3]);
/// assert_eq!(NodeOrder::Gmsh.permutation(ElementType::TET4), [0, 1, 2, 3]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeOrder {
    /// pyrucast's (= VTK's) numbering: nothing to realign.
    Pyrucast,
    /// gmsh's numbering.
    Gmsh,
    /// MED's numbering (Salome, code_aster, medcoupling).
    Med,
}

impl NodeOrder {
    /// `pyrucast[i] = external[perm[i]]` for a cell of type `et` — the
    /// identity when the two numberings coincide, so a caller always gathers
    /// through it and never tests.
    ///
    /// ```
    /// # use pyrucast::atoms::ElementType;
    /// # use pyrucast::ops::mesh::arrays::NodeOrder;
    /// assert_eq!(NodeOrder::Pyrucast.permutation(ElementType::HEX8), [0, 1, 2, 3, 4, 5, 6, 7]);
    /// assert_eq!(NodeOrder::Med.permutation(ElementType::HEX8), [0, 3, 2, 1, 4, 7, 6, 5]);
    /// ```
    pub fn permutation(self, et: ElementType) -> &'static [usize] {
        let kind = et.as_kind();
        match self {
            Self::Pyrucast => identity_permutation(kind.nodes_per_cell()),
            Self::Gmsh => kind.gmsh_permutation(),
            Self::Med => kind.med_permutation(),
        }
    }
}

// ─── Tags ────────────────────────────────────────────────────────────────────

/// An integer usable as a node or cell tag: gmsh hands `u64`, medcoupling
/// `i64`. Generic so both are **borrowed** as they are, without a conversion
/// pass.
///
/// ```
/// # use pyrucast::ops::mesh::arrays::Tag;
/// assert!((-1_i64).is_negative());
/// assert_eq!(7_i64.key(), 7_u64.key());
/// ```
pub trait Tag: Copy + Ord + std::fmt::Display + Send + Sync + 'static {
    /// Whether the tag is below zero — refused, since no format numbers so.
    fn is_negative(self) -> bool;
    /// The tag as an unsigned key (only called on non-negative tags, or on
    /// tags whose lookup is allowed to miss).
    fn key(self) -> u64;
}

impl Tag for u64 {
    fn is_negative(self) -> bool {
        false
    }
    fn key(self) -> u64 {
        self
    }
}

impl Tag for i64 {
    fn is_negative(self) -> bool {
        self < 0
    }
    fn key(self) -> u64 {
        self as u64
    }
}

/// Tag → row. Formats number `1..=n`, so a dense table is both smaller and
/// faster than hashing; past a factor four of holes it hashes instead.
pub(crate) enum TagIndex {
    /// `rows[tag - min]`, `u32::MAX` where no tag sits.
    Dense {
        min: u64,
        rows: Vec<u32>,
    },
    Sparse(HashMap<u64, u32>),
}

impl TagIndex {
    /// Index the tags of `parts`, each `(tags, first row)`: tag `k` of a part
    /// gets row `first + k`. A negative or repeated tag is an error — a
    /// repeated one would silently merge two nodes (or cells).
    pub(crate) fn new<T: Tag>(parts: &[(&[T], usize)], what: &str) -> Result<Self> {
        let mut bounds: Option<(T, T)> = None;
        let mut count = 0usize;
        for (tags, _) in parts {
            count += tags.len();
            for &t in *tags {
                bounds = Some(match bounds {
                    None => (t, t),
                    Some((lo, hi)) => (lo.min(t), hi.max(t)),
                });
            }
        }
        let Some((lo, hi)) = bounds else {
            return Ok(Self::Sparse(HashMap::new()));
        };
        if lo.is_negative() {
            return Err(err(format!("negative {what} tag {lo}")));
        }
        let (min, max) = (lo.key(), hi.key());
        let repeated = |t: T| err(format!("{what} tag {t} appears twice"));
        // `checked_add`: the span of a garbage tag list can reach `u64::MAX`.
        if let Some(span) = (max - min).checked_add(1)
            && span <= 4 * count as u64
        {
            let mut rows = vec![u32::MAX; span as usize];
            for &(tags, first) in parts {
                for (k, &t) in tags.iter().enumerate() {
                    let slot = &mut rows[(t.key() - min) as usize];
                    if *slot != u32::MAX {
                        return Err(repeated(t));
                    }
                    *slot = (first + k) as u32;
                }
            }
            Ok(Self::Dense { min, rows })
        } else {
            let mut map = HashMap::with_capacity(count);
            for &(tags, first) in parts {
                for (k, &t) in tags.iter().enumerate() {
                    if map.insert(t.key(), (first + k) as u32).is_some() {
                        return Err(repeated(t));
                    }
                }
            }
            Ok(Self::Sparse(map))
        }
    }

    #[inline]
    pub(crate) fn row(&self, tag: u64) -> Option<usize> {
        match self {
            Self::Dense { min, rows } => match rows.get(tag.wrapping_sub(*min) as usize) {
                Some(&r) if r != u32::MAX => Some(r as usize),
                _ => None,
            },
            Self::Sparse(map) => map.get(&tag).map(|&r| r as usize),
        }
    }
}

// ─── Input descriptors ───────────────────────────────────────────────────────

/// One block of cells: one element type, one combination of groups.
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::ops::mesh::arrays::CellBlock;
/// let plaque = ["plaque".to_string()];
/// let bloc = CellBlock {
///     element_type: ElementType::TRI3,
///     node_tags: &[1_u64, 2, 3, 2, 4, 3],
///     cell_tags: &[10, 11],
///     groups: &plaque,
/// };
/// // The connectivity is flat: its length tells the cell count.
/// assert_eq!(bloc.node_tags.len() / ElementType::TRI3.nodes_per_cell(), 2);
/// ```
pub struct CellBlock<'a, T> {
    /// The pyrucast type of every cell of the block.
    pub element_type: ElementType,
    /// Flat connectivity, in the **external** node order: cell `i` occupies
    /// `[i*npc, (i+1)*npc)`.
    pub node_tags: &'a [T],
    /// One tag per cell, for the cell fields to refer to — or empty, for a
    /// block no cell field will name.
    pub cell_tags: &'a [T],
    /// Every group the block's cells belong to; they land in each of them. An
    /// empty list puts them nowhere.
    pub groups: &'a [String],
}

/// Values at nodes: `values[k*ncomp + c]` is component `c` at `node_tags[k]`.
///
/// ```
/// # use pyrucast::ops::mesh::arrays::NodeValues;
/// let comps = ["T".to_string()];
/// let t = NodeValues { components: &comps, node_tags: &[1_u64, 2], values: &[20.0, 25.0] };
/// assert_eq!(t.values.len(), t.node_tags.len() * t.components.len());
/// ```
pub struct NodeValues<'a, T> {
    pub components: &'a [String],
    pub node_tags: &'a [T],
    pub values: &'a [f64],
}

/// An external Gauss rule for one element type: its reference element's
/// nodes (external order, `dim` coordinates each), its points and weights.
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::ops::mesh::arrays::GaussRule;
/// let rule = GaussRule {
///     element_type: ElementType::TRI3,
///     ref_nodes: &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
///     xi: &[1.0 / 3.0, 1.0 / 3.0],
///     weights: &[0.5],
/// };
/// assert_eq!(rule.weights.len(), 1);
/// ```
pub struct GaussRule<'a> {
    pub element_type: ElementType,
    pub ref_nodes: &'a [f64],
    pub xi: &'a [f64],
    pub weights: &'a [f64],
}

/// Where a cell field's values sit inside each cell.
///
/// ```
/// # use pyrucast::ops::mesh::arrays::CellLayout;
/// // One value per cell and component.
/// let layout = CellLayout::Cell;
/// assert!(matches!(layout, CellLayout::Cell));
/// ```
pub enum CellLayout<'a> {
    /// One value per cell (and component).
    Cell,
    /// One value per Gauss point of the cell's type (and component), the
    /// points listed in the order of that type's rule.
    Gauss(&'a [GaussRule<'a>]),
}

/// Values on cells: for the `k`-th entry of `cell_tags`, its row of
/// `ncomp` values (`Cell`), or its `n_points × ncomp` values (`Gauss`).
///
/// ```
/// # use pyrucast::ops::mesh::arrays::{CellLayout, CellValues};
/// let comps = ["E".to_string()];
/// let e = CellValues {
///     components: &comps,
///     cell_tags: &[10_u64, 11],
///     values: &[210e3, 70e3],
///     layout: CellLayout::Cell,
/// };
/// assert_eq!(e.values.len(), 2);
/// ```
pub struct CellValues<'a, T> {
    pub components: &'a [String],
    pub cell_tags: &'a [T],
    pub values: &'a [f64],
    pub layout: CellLayout<'a>,
}

/// What [`from_arrays`] builds: one mesh per group, then the fields in the
/// order they were asked for.
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh::arrays::{from_arrays, CellBlock, Imported, NodeOrder};
/// let t = ["t".to_string()];
/// let blocks = [CellBlock {
///     element_type: ElementType::SEG2,
///     node_tags: &[1_u64, 2],
///     cell_tags: &[],
///     groups: &t,
/// }];
/// let coords = Handle::new(Coords::new(1)?);
/// let out: Imported =
///     from_arrays(coords, &[1_u64, 2], &[0.0, 1.0], &blocks, &[], &[], NodeOrder::Pyrucast)?;
/// assert_eq!(out.groups.len(), 1);
/// assert!(out.node_fields.is_empty() && out.element_fields.is_empty());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct Imported {
    pub groups: Vec<(String, Mesh)>,
    pub node_fields: Vec<NodeField>,
    pub element_fields: Vec<ElementField>,
}

// ─── The import ──────────────────────────────────────────────────────────────

/// One `(group, element type)` submesh to build: the blocks it is made of.
struct SubPlan {
    group: usize,
    element_type: ElementType,
    blocks: Vec<usize>,
    len: usize,
}

/// Build one [`Mesh`] per group, and the requested fields, from flat arrays —
/// see the module documentation for the rules.
///
/// Errors on input that cannot describe a mesh: a node table whose stride is
/// not 1 to 3, a block that is not a whole number of cells, a negative,
/// repeated or unknown tag, values of the wrong length, a Gauss rule that is
/// not pyrucast's, a cell field that covers no group entirely.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh::arrays::{from_arrays, CellBlock, CellLayout, CellValues, NodeOrder, NodeValues};
/// // Four nodes (three coordinates each), two triangles in "plaque".
/// let tags = [1_u64, 2, 3, 4];
/// let xyz = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
/// let plaque = ["plaque".to_string()];
/// let blocs = [CellBlock {
///     element_type: ElementType::TRI3,
///     node_tags: &[1, 2, 3, 2, 4, 3],
///     cell_tags: &[1, 2],
///     groups: &plaque,
/// }];
/// let t = ["T".to_string()];
/// let temperature = NodeValues { components: &t, node_tags: &tags, values: &[1.0, 2.0, 3.0, 4.0] };
/// let e = ["E".to_string()];
/// let module = CellValues { components: &e, cell_tags: &[1, 2], values: &[5.0, 6.0], layout: CellLayout::Cell };
///
/// // The `Coords` is 2-D: the third coordinate is dropped.
/// let coords = Handle::new(Coords::new(2)?);
/// let out = from_arrays(coords.clone(), &tags, &xyz, &blocs, &[temperature], &[module], NodeOrder::Pyrucast)?;
/// assert_eq!(out.groups[0].0, "plaque");
/// assert_eq!(out.groups[0].1.cell_count(), 2);
/// assert_eq!(coords.read().node_count(), 4);
/// let node = pyrucast::atoms::NodeId(3); // tag 4, created last
/// assert_eq!(out.node_fields[0].value(node, "T")?, 4.0);
/// assert_eq!(out.element_fields[0].get(0)?.read().value(1, 0, "E")?, 6.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn from_arrays<T: Tag>(
    coords: Handle<Coords>,
    node_tags: &[T],
    node_coords: &[f64],
    blocks: &[CellBlock<'_, T>],
    node_values: &[NodeValues<'_, T>],
    cell_values: &[CellValues<'_, T>],
    order: NodeOrder,
) -> Result<Imported> {
    let n = node_tags.len();
    let stride = node_coords.len().checked_div(n).unwrap_or(0);
    if n > 0 && (stride == 0 || stride > 3 || node_coords.len() != stride * n) {
        return Err(err(format!(
            "{n} node tags call for 1 to 3 coordinates each, got {} values",
            node_coords.len()
        )));
    }
    let index = TagIndex::new(&[(node_tags, 0)], "node")?;

    // ── Plan: groups, submeshes, checks that cost nothing per node ──────────
    let mut group_names: Vec<String> = Vec::new();
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    let mut block_groups: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    let mut subs: Vec<SubPlan> = Vec::new();
    let mut sub_of: HashMap<(usize, ElementType), usize> = HashMap::new();
    for (b, block) in blocks.iter().enumerate() {
        let et = block.element_type;
        let npc = et.nodes_per_cell();
        let len = block.node_tags.len();
        if !len.is_multiple_of(npc) {
            return Err(err(format!(
                "a {et} block holds {len} node tags, not a whole number of cells of {npc}"
            )));
        }
        if !block.cell_tags.is_empty() && block.cell_tags.len() != len / npc {
            return Err(err(format!(
                "a {et} block of {} cells carries {} cell tags",
                len / npc,
                block.cell_tags.len()
            )));
        }
        let mut gs = Vec::with_capacity(block.groups.len());
        for name in block.groups {
            let g = *group_of.entry(name.as_str()).or_insert_with(|| {
                group_names.push(name.clone());
                group_names.len() - 1
            });
            if gs.contains(&g) {
                continue;
            }
            gs.push(g);
            let s = *sub_of.entry((g, et)).or_insert_with(|| {
                subs.push(SubPlan {
                    group: g,
                    element_type: et,
                    blocks: Vec::new(),
                    len: 0,
                });
                subs.len() - 1
            });
            subs[s].blocks.push(b);
            subs[s].len += len;
        }
        block_groups.push(gs);
    }

    // ── Pass 1: tags → rows, occurrences counted ────────────────────────────
    // Each block's connectivity is written once, at its final size, holding
    // rows for now; pass 2 rewrites it in place.
    let mut counts = vec![0u32; n];
    let mut block_conn: Vec<Vec<NodeId>> = Vec::with_capacity(blocks.len());
    for (block, gs) in blocks.iter().zip(&block_groups) {
        let units = gs.len() as u32;
        let mut conn = Vec::with_capacity(block.node_tags.len());
        for &tag in block.node_tags {
            let row = index
                .row(tag.key())
                .ok_or_else(|| err(format!("a cell references unknown node {tag}")))?;
            counts[row] += units;
            conn.push(NodeId(row as u32));
        }
        block_conn.push(conn);
    }

    // ── Used nodes: ranks, then created at once, already counted ───────────
    let mut rank = vec![u32::MAX; n];
    let mut used_rows: Vec<u32> = Vec::new();
    let mut refcounts: Vec<u32> = Vec::new();
    for (row, &c) in counts.iter().enumerate() {
        if c > 0 {
            rank[row] = used_rows.len() as u32;
            used_rows.push(row as u32);
            refcounts.push(c);
        }
    }
    drop(counts);
    let dim = coords.read().dim() as usize;
    let keep = stride.min(dim);
    let first = coords
        .write()
        .add_counted_nodes(&refcounts, |buf| {
            for (dst, &row) in buf.chunks_exact_mut(dim).zip(&used_rows) {
                let at = row as usize * stride;
                dst[..keep].copy_from_slice(&node_coords[at..at + keep]);
            }
        })?
        .start;
    drop(refcounts);
    drop(used_rows);

    // ── Pass 2: rows → ids, in pyrucast's order, in place ───────────────────
    for (block, conn) in blocks.iter().zip(&mut block_conn) {
        let npc = block.element_type.nodes_per_cell();
        let perm = order.permutation(block.element_type);
        let mut cell = [NodeId(0); 27];
        for chunk in conn.chunks_exact_mut(npc) {
            cell[..npc].copy_from_slice(chunk);
            for (slot, &p) in chunk.iter_mut().zip(perm) {
                *slot = NodeId(first + rank[cell[p].0 as usize]);
            }
        }
    }

    // ── Submeshes and groups ────────────────────────────────────────────────
    // A block's buffer is moved into its last user when that user is made of
    // it alone; every other use copies it.
    let mut uses: Vec<usize> = block_groups.iter().map(Vec::len).collect();
    let mut sub_handles: Vec<Handle<SubMesh>> = Vec::with_capacity(subs.len());
    for sub in &subs {
        let conn = if let [b] = sub.blocks[..]
            && uses[b] == 1
        {
            std::mem::take(&mut block_conn[b])
        } else {
            let mut conn = Vec::with_capacity(sub.len);
            for &b in &sub.blocks {
                conn.extend_from_slice(&block_conn[b]);
            }
            conn
        };
        for &b in &sub.blocks {
            uses[b] -= 1;
            if uses[b] == 0 {
                block_conn[b] = Vec::new();
            }
        }
        sub_handles.push(Handle::new(SubMesh::from_counted_connectivity(
            coords.clone(),
            sub.element_type,
            conn,
        )));
    }
    let mut groups: Vec<(String, Mesh)> = group_names
        .into_iter()
        .map(|g| (g, Mesh::empty()))
        .collect();
    for (sub, h) in subs.iter().zip(&sub_handles) {
        groups[sub.group].1.add_sub(h.clone())?;
    }

    // ── Node fields ─────────────────────────────────────────────────────────
    // Fields defined on the same nodes share one support, so the steps of a
    // time series line up zone for zone.
    let mut node_fields = Vec::with_capacity(node_values.len());
    let mut supports: Vec<Handle<SubMesh>> = Vec::new();
    for (f, nv) in node_values.iter().enumerate() {
        node_fields.push(node_field(
            &coords,
            &index,
            &rank,
            first,
            n,
            f,
            nv,
            &mut supports,
        )?);
    }

    // ── Cell fields ─────────────────────────────────────────────────────────
    let mut element_fields = Vec::with_capacity(cell_values.len());
    if !cell_values.is_empty() {
        let cells = CellTable::new(blocks)?;
        let mut spaces = SpaceCache::new(subs.len());
        for (f, cv) in cell_values.iter().enumerate() {
            element_fields.push(cell_field(
                &cells,
                &subs,
                &sub_handles,
                &mut spaces,
                order,
                f,
                cv,
            )?);
        }
    }

    Ok(Imported {
        groups,
        node_fields,
        element_fields,
    })
}

/// A node field on a POI1 support made of the nodes it defines that the mesh
/// created, in the order of its tags — the support of an earlier field when
/// it holds the same nodes in the same order.
#[allow(clippy::too_many_arguments)]
fn node_field<T: Tag>(
    coords: &Handle<Coords>,
    index: &TagIndex,
    rank: &[u32],
    first: u32,
    n: usize,
    f: usize,
    nv: &NodeValues<'_, T>,
    supports: &mut Vec<Handle<SubMesh>>,
) -> Result<NodeField> {
    let nc = nv.components.len();
    if nv.values.len() != nv.node_tags.len() * nc {
        return Err(err(format!(
            "node field #{f}: {} values for {} nodes of {nc} components",
            nv.values.len(),
            nv.node_tags.len()
        )));
    }
    let mut seen = vec![false; n];
    let mut ids = Vec::with_capacity(nv.node_tags.len());
    let mut src = Vec::with_capacity(nv.node_tags.len());
    for (k, &tag) in nv.node_tags.iter().enumerate() {
        let row = index
            .row(tag.key())
            .ok_or_else(|| err(format!("node field #{f}: unknown node {tag}")))?;
        if std::mem::replace(&mut seen[row], true) {
            return Err(err(format!("node field #{f}: node {tag} appears twice")));
        }
        let r = rank[row];
        if r != u32::MAX {
            ids.push(NodeId(first + r));
            src.push(k);
        }
    }
    let known = supports
        .iter()
        .find(|h| h.read().connectivity() == ids.as_slice())
        .cloned();
    let support = match known {
        Some(h) => h,
        None => {
            let h = Handle::new(SubMesh::from_connectivity(
                coords.clone(),
                ElementType::POI1,
                ids,
            )?);
            supports.push(h.clone());
            h
        }
    };
    let mut sub = SubNodeField::from_poi1(&support, nv.components.to_vec())?;
    let dst = sub.values_mut();
    if src.len() == nv.node_tags.len() {
        dst.copy_from_slice(nv.values);
    } else {
        for (row, &k) in dst.chunks_exact_mut(nc).zip(&src) {
            row.copy_from_slice(&nv.values[k * nc..(k + 1) * nc]);
        }
    }
    Ok(NodeField::from_sub(sub))
}

/// Every cell of the import, numbered block after block: global cell `g`
/// is cell `g - offset[b]` of block `b`.
struct CellTable {
    index: TagIndex,
    offsets: Vec<usize>,
    types: Vec<ElementType>,
}

impl CellTable {
    fn new<T: Tag>(blocks: &[CellBlock<'_, T>]) -> Result<Self> {
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        let mut parts = Vec::with_capacity(blocks.len());
        let mut total = 0;
        for b in blocks {
            offsets.push(total);
            if !b.cell_tags.is_empty() {
                parts.push((b.cell_tags, total));
            }
            total += b.node_tags.len() / b.element_type.nodes_per_cell();
        }
        offsets.push(total);
        Ok(Self {
            index: TagIndex::new(&parts, "cell")?,
            offsets,
            types: blocks.iter().map(|b| b.element_type).collect(),
        })
    }

    fn total(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    fn type_of(&self, g: usize) -> ElementType {
        self.types[self.offsets.partition_point(|&o| o <= g) - 1]
    }

    fn range(&self, b: usize) -> std::ops::Range<usize> {
        self.offsets[b]..self.offsets[b + 1]
    }
}

/// The FE spaces the cell fields of one import share: per submesh, a
/// one-point space (`Cell` values) and the full-rule one (`Gauss` values),
/// each built the first time a field needs it.
struct SpaceCache {
    reduced: Vec<Option<Handle<SubFiniteElementSpace>>>,
    gauss: Vec<Option<Handle<SubFiniteElementSpace>>>,
}

impl SpaceCache {
    fn new(n: usize) -> Self {
        Self {
            reduced: vec![None; n],
            gauss: vec![None; n],
        }
    }

    fn get(
        &mut self,
        s: usize,
        sub: &Handle<SubMesh>,
        et: ElementType,
        quad: QuadratureRule,
    ) -> Result<Handle<SubFiniteElementSpace>> {
        let slot = match quad {
            QuadratureRule::Reduced => &mut self.reduced[s],
            QuadratureRule::Gauss => &mut self.gauss[s],
        };
        if let Some(h) = slot {
            return Ok(h.clone());
        }
        let interp = et
            .as_kind()
            .degree()
            .ok_or_else(|| err(format!("{et} carries no finite-element space")))?;
        let h = Handle::new(SubFiniteElementSpace::new(sub.clone(), interp, quad)?);
        *slot = Some(h.clone());
        Ok(h)
    }
}

/// A cell field with a zone on every non-POI1 group submesh it covers
/// entirely.
#[allow(clippy::too_many_arguments)]
fn cell_field<T: Tag>(
    cells: &CellTable,
    subs: &[SubPlan],
    sub_handles: &[Handle<SubMesh>],
    spaces: &mut SpaceCache,
    order: NodeOrder,
    f: usize,
    cv: &CellValues<'_, T>,
) -> Result<ElementField> {
    let nc = cv.components.len();
    // Per type: how many external points, and which one is pyrucast's `g`.
    let mut rules: HashMap<ElementType, (usize, Vec<usize>)> = HashMap::new();
    if let CellLayout::Gauss(list) = &cv.layout {
        for r in *list {
            let found = match_gauss(r.element_type, order, r.ref_nodes, r.xi, r.weights)?;
            rules.insert(r.element_type, (r.weights.len(), found));
        }
    }

    // Where each covered cell's values start.
    let mut start = vec![usize::MAX; cells.total()];
    let mut next = 0usize;
    for &tag in cv.cell_tags {
        let g = cells
            .index
            .row(tag.key())
            .ok_or_else(|| err(format!("cell field #{f}: unknown cell {tag}")))?;
        if start[g] != usize::MAX {
            return Err(err(format!("cell field #{f}: cell {tag} appears twice")));
        }
        start[g] = next;
        next += nc
            * match &cv.layout {
                CellLayout::Cell => 1,
                CellLayout::Gauss(_) => {
                    let et = cells.type_of(g);
                    rules
                        .get(&et)
                        .ok_or_else(|| {
                            err(format!("cell field #{f}: no Gauss rule given for {et}"))
                        })?
                        .0
                }
            };
    }
    if cv.values.len() != next {
        return Err(err(format!(
            "cell field #{f}: {} values where its cells call for {next}",
            cv.values.len()
        )));
    }

    let quad = match cv.layout {
        CellLayout::Cell => QuadratureRule::Reduced,
        CellLayout::Gauss(_) => QuadratureRule::Gauss,
    };
    let mut field = ElementField::default();
    for (s, (sub, h)) in subs.iter().zip(sub_handles).enumerate() {
        let et = sub.element_type;
        let covered = sub
            .blocks
            .iter()
            .all(|&b| cells.range(b).all(|g| start[g] != usize::MAX));
        if et == ElementType::POI1 || !covered {
            continue;
        }
        let fes = spaces.get(s, h, et, quad)?;
        let mut zone = SubElementField::new(fes, cv.components.to_vec())?;
        let ng = zone.gauss_count();
        let dst = zone.values_mut();
        let mut k = 0;
        match &cv.layout {
            CellLayout::Cell => {
                for &b in &sub.blocks {
                    for g in cells.range(b) {
                        let src = &cv.values[start[g]..start[g] + nc];
                        for row in dst[k * ng * nc..(k + 1) * ng * nc].chunks_exact_mut(nc) {
                            row.copy_from_slice(src);
                        }
                        k += 1;
                    }
                }
            }
            CellLayout::Gauss(_) => {
                let found = &rules[&et].1;
                for &b in &sub.blocks {
                    for g in cells.range(b) {
                        let at = start[g];
                        let rows = dst[k * ng * nc..(k + 1) * ng * nc].chunks_exact_mut(nc);
                        for (row, &e) in rows.zip(found) {
                            row.copy_from_slice(&cv.values[at + e * nc..at + (e + 1) * nc]);
                        }
                        k += 1;
                    }
                }
            }
        }
        field.add_sub(Handle::new(zone))?;
    }
    if field.is_empty() {
        return Err(err(format!(
            "cell field #{f} defines no group entirely (a zone needs all its cells)"
        )));
    }
    Ok(field)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coords(dim: u8) -> Handle<Coords> {
        Handle::new(Coords::new(dim).unwrap())
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// A unit square: four nodes, three coordinates each.
    fn square() -> (Vec<u64>, Vec<f64>) {
        #[rustfmt::skip]
        let xyz = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        (vec![1, 2, 3, 4], xyz)
    }

    fn tri<'a>(conn: &'a [u64], groups: &'a [String]) -> CellBlock<'a, u64> {
        CellBlock {
            element_type: ElementType::TRI3,
            node_tags: conn,
            cell_tags: &[],
            groups,
        }
    }

    fn import(
        c: Handle<Coords>,
        tags: &[u64],
        xyz: &[f64],
        blocks: &[CellBlock<'_, u64>],
    ) -> Result<Vec<(String, Mesh)>> {
        Ok(from_arrays(c, tags, xyz, blocks, &[], &[], NodeOrder::Pyrucast)?.groups)
    }

    fn message(e: PyrucastError) -> String {
        match e {
            PyrucastError::Message(m) => m,
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn share_one_coords_and_keep_only_used_nodes() {
        let (mut tags, mut xyz) = square();
        // A fifth node nothing references: it must not land in the `Coords`.
        tags.push(9);
        xyz.extend_from_slice(&[5.0, 5.0, 0.0]);
        let plate = names(&["plate"]);
        let c = coords(2);
        let groups = import(c.clone(), &tags, &xyz, &[tri(&[1, 2, 3], &plate)]).unwrap();
        assert_eq!(c.read().node_count(), 3);
        assert!(groups[0].1.coords().unwrap().same_object(&c));
    }

    #[test]
    fn refcounts_count_every_occurrence_in_every_group() {
        // Node 2 serves in two cells of a block that belongs to two groups:
        // four units, exactly what `from_connectivity` would have taken.
        let (tags, xyz) = square();
        let both = names(&["plate", "all"]);
        let c = coords(2);
        let groups = import(c.clone(), &tags, &xyz, &[tri(&[1, 2, 3, 2, 4, 3], &both)]).unwrap();
        let id = groups[0].1.get(0).unwrap().read().connectivity()[1];
        assert_eq!(c.read().refcount(id), 4);
        // Dropping the meshes hands every unit back.
        drop(groups);
        assert_eq!(c.write().gc(), 4);
    }

    #[test]
    fn apply_the_named_node_order() {
        // TET4 in MED order: the base is walked the other way round.
        let tags: Vec<u64> = (1..=4).collect();
        let xyz: Vec<f64> = (0..4).flat_map(|i| [i as f64, 0.0, 0.0]).collect();
        let bloc = names(&["bloc"]);
        let blocks = [CellBlock {
            element_type: ElementType::TET4,
            node_tags: &tags,
            cell_tags: &[],
            groups: &bloc,
        }];
        let out = from_arrays(coords(3), &tags, &xyz, &blocks, &[], &[], NodeOrder::Med).unwrap();
        let conn = out.groups[0]
            .1
            .get(0)
            .unwrap()
            .read()
            .connectivity()
            .to_vec();
        assert_eq!(conn, [NodeId(0), NodeId(2), NodeId(1), NodeId(3)]);
    }

    #[test]
    fn place_a_block_in_each_of_its_groups() {
        let (tags, xyz) = square();
        let both = names(&["plate", "everything"]);
        let groups = import(coords(2), &tags, &xyz, &[tri(&[1, 2, 3], &both)]).unwrap();
        let got: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(got, vec!["plate", "everything"]);
        assert_eq!(groups[0].1.cell_count(), 1);
        assert_eq!(groups[1].1.cell_count(), 1);
    }

    #[test]
    fn gather_the_blocks_of_a_group_into_one_submesh_per_type() {
        // Two families sharing "plate": one TRI3 submesh of both blocks.
        let (tags, xyz) = square();
        let (plate, both) = (names(&["plate"]), names(&["plate", "top"]));
        let groups = import(
            coords(2),
            &tags,
            &xyz,
            &[tri(&[1, 2, 3], &plate), tri(&[1, 3, 4], &both)],
        )
        .unwrap();
        assert_eq!(groups[0].0, "plate");
        assert_eq!(groups[0].1.len(), 1);
        assert_eq!(groups[0].1.cell_count(), 2);
        assert_eq!(groups[1].1.cell_count(), 1);
    }

    #[test]
    fn read_sparse_and_signed_tags() {
        // Tags far apart: `TagIndex` hashes instead of tabulating.
        let tags = vec![1_i64, 5_000_000, 9_000_000];
        let xyz = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]; // stride 2
        let plate = names(&["plate"]);
        let blocks = [CellBlock {
            element_type: ElementType::TRI3,
            node_tags: &[1_i64, 5_000_000, 9_000_000],
            cell_tags: &[],
            groups: &plate,
        }];
        let c = coords(3);
        let out = from_arrays(
            c.clone(),
            &tags,
            &xyz,
            &blocks,
            &[],
            &[],
            NodeOrder::Pyrucast,
        )
        .unwrap();
        assert_eq!(out.groups[0].1.cell_count(), 1);
        // Stride 2 into a 3-D `Coords`: z padded with zero.
        assert_eq!(c.read().position(NodeId(1)).unwrap(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn negative_and_repeated_tags_error() {
        let plate = names(&["plate"]);
        let blocks: [CellBlock<'_, i64>; 0] = [];
        let e = from_arrays(
            coords(2),
            &[-1_i64],
            &[0.0, 0.0],
            &blocks,
            &[],
            &[],
            NodeOrder::Pyrucast,
        );
        assert!(message(e.err().unwrap()).contains("negative node tag -1"));
        let (_, xyz) = square();
        let e = import(coords(2), &[1, 2, 2, 4], &xyz, &[tri(&[1, 2, 4], &plate)]).unwrap_err();
        assert!(message(e).contains("appears twice"));
    }

    #[test]
    fn unknown_node_errors() {
        let (tags, xyz) = square();
        let plate = names(&["plate"]);
        let e = import(coords(2), &tags, &xyz, &[tri(&[1, 2, 77], &plate)]).unwrap_err();
        assert!(message(e).contains("unknown node 77"));
    }

    #[test]
    fn ragged_block_errors() {
        let (tags, xyz) = square();
        let plate = names(&["plate"]);
        let e = import(coords(2), &tags, &xyz, &[tri(&[1, 2, 3, 4], &plate)]).unwrap_err();
        assert!(message(e).contains("whole number of cells"));
    }

    #[test]
    fn coordinate_count_mismatch_errors() {
        let (tags, _) = square();
        let plate = names(&["plate"]);
        // Seven values for four nodes.
        let flat = vec![0.0; 7];
        let e = import(coords(2), &tags, &flat, &[tri(&[1, 2, 3], &plate)]).unwrap_err();
        assert!(message(e).contains("1 to 3 coordinates"));
    }

    #[test]
    fn node_fields_keep_the_nodes_the_mesh_created() {
        let (mut tags, mut xyz) = square();
        tags.push(9);
        xyz.extend_from_slice(&[5.0, 5.0, 0.0]);
        let plate = names(&["plate"]);
        let blocks = [tri(&[4, 2, 3], &plate)];
        let comps = names(&["T", "p"]);
        // Values given for every tag, the unused node 9 included.
        let values: Vec<f64> = (0..5).flat_map(|i| [i as f64, 10.0 * i as f64]).collect();
        let nv = NodeValues {
            components: &comps,
            node_tags: &tags,
            values: &values,
        };
        let out = from_arrays(
            coords(2),
            &tags,
            &xyz,
            &blocks,
            &[nv],
            &[],
            NodeOrder::Pyrucast,
        )
        .unwrap();
        let field = &out.node_fields[0];
        // Nodes are created in tag order: tag 2 → id 0, 3 → 1, 4 → 2.
        assert_eq!(field.value(NodeId(0), "T").unwrap(), 1.0);
        assert_eq!(field.value(NodeId(2), "p").unwrap(), 30.0);
        assert_eq!(field.get(0).unwrap().read().node_count(), 3);
    }

    #[test]
    fn cell_fields_cover_whole_groups_only() {
        let (tags, xyz) = square();
        let (a, b) = (names(&["a"]), names(&["b"]));
        let blocks = [
            CellBlock {
                element_type: ElementType::TRI3,
                node_tags: &[1_u64, 2, 3],
                cell_tags: &[10],
                groups: &a,
            },
            CellBlock {
                element_type: ElementType::TRI3,
                node_tags: &[1, 3, 4, 1, 2, 4],
                cell_tags: &[20, 21],
                groups: &b,
            },
        ];
        let comps = names(&["E"]);
        // Group "b" only half covered: it gets no zone.
        let cv = CellValues {
            components: &comps,
            cell_tags: &[10_u64, 21],
            values: &[1.5, 2.5],
            layout: CellLayout::Cell,
        };
        let out = from_arrays(
            coords(2),
            &tags,
            &xyz,
            &blocks,
            &[],
            &[cv],
            NodeOrder::Pyrucast,
        )
        .unwrap();
        let field = &out.element_fields[0];
        assert_eq!(field.len(), 1);
        let zone = field.get(0).unwrap();
        assert_eq!(zone.read().gauss_count(), 1);
        assert_eq!(zone.read().value(0, 0, "E").unwrap(), 1.5);

        let none = CellValues {
            components: &comps,
            cell_tags: &[21_u64],
            values: &[2.5],
            layout: CellLayout::Cell,
        };
        let e = from_arrays(
            coords(2),
            &tags,
            &xyz,
            &blocks,
            &[],
            &[none],
            NodeOrder::Pyrucast,
        );
        assert!(message(e.err().unwrap()).contains("defines no group entirely"));
    }

    #[test]
    fn gauss_values_are_realigned_on_pyrucast_rule() {
        // The external rule is pyrucast's own, listed backwards: the values
        // must come back in pyrucast's point order.
        let (tags, xyz) = square();
        let plate = names(&["plate"]);
        let blocks = [CellBlock {
            element_type: ElementType::TRI3,
            node_tags: &[1_u64, 2, 4],
            cell_tags: &[1],
            groups: &plate,
        }];
        let refs = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let (xi, w) = ElementType::TRI3.as_kind().gauss();
        let xi_rev: Vec<f64> = xi.chunks(2).rev().flatten().copied().collect();
        let w_rev: Vec<f64> = w.iter().rev().copied().collect();
        let rules = [GaussRule {
            element_type: ElementType::TRI3,
            ref_nodes: &refs,
            xi: &xi_rev,
            weights: &w_rev,
        }];
        let comps = names(&["s"]);
        let cv = CellValues {
            components: &comps,
            cell_tags: &[1_u64],
            values: &[30.0, 20.0, 10.0],
            layout: CellLayout::Gauss(&rules),
        };
        let out = from_arrays(
            coords(2),
            &tags,
            &xyz,
            &blocks,
            &[],
            &[cv],
            NodeOrder::Pyrucast,
        )
        .unwrap();
        let zone = out.element_fields[0].get(0).unwrap();
        let got: Vec<f64> = (0..3)
            .map(|g| zone.read().value(0, g, "s").unwrap())
            .collect();
        assert_eq!(got, [10.0, 20.0, 30.0]);
    }
}
