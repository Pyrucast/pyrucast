//! A `Cell` is a lightweight view on a single cell of a [`SubMesh`].
//!
//! It carries a cloned `Handle<SubMesh>` plus the cell's index — cloning
//! a `Cell` is just an `Arc` clone, so it is cheap to pass around and to
//! create on the fly inside an iterator. The actual node coordinates live
//! in the `Coords` and are fetched on demand through
//! [`Cell::nodes`] / [`Cell::node_ids`].
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::insert;
//!
//! let coords = insert(Coords::new(2).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let cell = mesh.cell(0, 0).unwrap();
//! assert_eq!(cell.nodes().unwrap().len(), 3);
//! for node in cell.nodes().unwrap() {
//!     let _ = node.coord().unwrap();
//! }
//! ```

use std::fmt;

use crate::containers::mesh::ElementType;
use crate::containers::mesh::Node;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::SubMesh;
use crate::error::{PyrucastError, Result};
use crate::store::{read, Handle};

/// Lightweight view on a single cell of a `SubMesh`.
#[derive(Clone)]
pub struct Cell {
    pub(crate) sm: Handle<SubMesh>,
    idx: usize,
}

impl Cell {
    /// Build a cell view. Errors if `idx` is past the submesh's
    /// `cell_count`.
    pub fn new(sm: Handle<SubMesh>, idx: usize) -> Result<Self> {
        let n = read(&sm)?.cell_count();
        if idx >= n {
            return Err(PyrucastError::Message(format!(
                "cell index {idx} out of range (cell_count={n})"
            )));
        }
        Ok(Self { sm, idx })
    }

    /// Index of this cell inside its parent submesh.
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Element type of this cell (same as the parent submesh).
    pub fn element_type(&self) -> Result<ElementType> {
        Ok(read(&self.sm)?.element_type())
    }

    /// Number of nodes that make up this cell (= `element_type().nodes_per_cell()`).
    pub fn nodes_per_cell(&self) -> Result<usize> {
        Ok(read(&self.sm)?.element_type().nodes_per_cell())
    }

    /// Raw connectivity (node ids) of this cell, in submesh order.
    pub fn node_ids(&self) -> Result<Vec<NodeId>> {
        let s = read(&self.sm)?;
        let npc = s.element_type().nodes_per_cell();
        Ok(s.connectivity()[self.idx * npc..(self.idx + 1) * npc].to_vec())
    }

    /// Materialise the cell's nodes as a `Vec<Node>`. Each `Node`
    /// increments the node's refcount in the owning `Coords`,
    /// matching the behaviour of `Coords::add_node`.
    pub fn nodes(&self) -> Result<Vec<Node>> {
        let coords = read(&self.sm)?.coords();
        let ids = self.node_ids()?;
        ids.into_iter()
            .map(|id| Node::acquire(coords.clone(), id))
            .collect()
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cell").field("idx", &self.idx).finish()
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.element_type(), self.node_ids()) {
            (Ok(et), Ok(ids)) => {
                let raw: Vec<u32> = ids.into_iter().map(|n| n.0).collect();
                write!(f, "Cell<{}> #{}: {:?}", et, self.idx, raw)
            }
            _ => write!(f, "Cell #{}", self.idx),
        }
    }
}

impl crate::dump::Dump for Cell {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let header = match self.element_type() {
            Ok(et) => format!("Cell<{et}> #{}", self.idx),
            Err(_) => format!("Cell #{}", self.idx),
        };
        // Per-node coordinate table (one lock on the Coords).
        let body = (|| -> Result<String> {
            let coords = read(&self.sm)?.coords();
            let ids = self.node_ids()?;
            let c = read(&coords)?;
            let mut rows: Vec<Vec<String>> = Vec::with_capacity(ids.len());
            let mut dim = 0usize;
            for &id in &ids {
                let coord = c.coord(id)?;
                dim = dim.max(coord.len());
                let mut row = vec![id.to_string()];
                row.extend(coord.iter().map(|v| fmt_float(*v, opts.precision)));
                rows.push(row);
            }
            for row in &mut rows {
                row.resize(1 + dim, String::new());
            }
            const AXES: [&str; 3] = ["x", "y", "z"];
            let mut headers = vec!["node".to_string()];
            headers.extend((0..dim).map(|i| AXES.get(i).copied().unwrap_or("?").to_string()));
            Ok(table(&headers, &rows, opts))
        })();
        match body {
            Ok(t) => format!("{header}\n{t}"),
            Err(e) => format!("{header}\n<{e}>"),
        }
    }
}

/// Iterator over the cells of a single submesh.
#[derive(Clone)]
pub struct CellIter {
    sm: Handle<SubMesh>,
    next: usize,
    end: usize,
}

impl CellIter {
    pub(crate) fn new(sm: Handle<SubMesh>, end: usize) -> Self {
        Self { sm, next: 0, end }
    }
}

impl Iterator for CellIter {
    type Item = Cell;
    fn next(&mut self) -> Option<Cell> {
        if self.next < self.end {
            let c = Cell {
                sm: self.sm.clone(),
                idx: self.next,
            };
            self.next += 1;
            Some(c)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CellIter {}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::store::insert;

    #[test]
    fn cell_exposes_ids_and_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let h = insert(sm);

        let cell = Cell::new(h, 0).unwrap();
        assert_eq!(cell.element_type().unwrap(), ElementType::TRI3);
        assert_eq!(cell.nodes_per_cell().unwrap(), 3);
        assert_eq!(cell.node_ids().unwrap(), vec![a.id(), b.id(), c.id()]);
        let nodes = cell.nodes().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].id(), a.id());
    }

    #[test]
    fn cell_new_rejects_out_of_range() {
        let coords = insert(Coords::new(2).unwrap());
        let sm = insert(SubMesh::new(coords, ElementType::TRI3));
        assert!(Cell::new(sm, 0).is_err());
    }

    #[test]
    fn cell_dump_renders_coordinate_table() {
        use crate::dump::{Dump, DumpOptions};
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let cell = Cell::new(insert(sm), 0).unwrap();

        let s = cell.render(&DumpOptions::default());
        assert!(s.starts_with("Cell<TRI3> #0"), "header:\n{s}");
        assert!(s.contains("node"), "table header:\n{s}");
        assert!(s.contains('x') && s.contains('y'), "axis labels:\n{s}");
        // One row per node + coordinates at default precision (3).
        assert!(s.contains("0.000"), "coords:\n{s}");
        assert_eq!(s.lines().count(), 5, "header + table header + 3 rows:\n{s}");
    }

    #[test]
    fn cells_iterator_yields_all_cells() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[b.id(), c.id()]).unwrap();

        let cells: Vec<_> = mesh.cells(0).unwrap().collect();
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].index(), 0);
        assert_eq!(cells[1].index(), 1);
        assert_eq!(cells[0].node_ids().unwrap(), vec![a.id(), b.id()]);
        assert_eq!(cells[1].node_ids().unwrap(), vec![b.id(), c.id()]);
    }
}
