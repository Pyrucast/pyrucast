//! Greedy graph colouring of a cell set for conflict-free parallel scatters.
//!
//! Two cells **conflict** when they share a node. Colouring partitions the
//! cells so that no two cells of the same colour share a node: the cells of one
//! colour can then scatter their element contributions **in parallel** without
//! write conflicts, the colours being processed in sequence. The result is
//! deterministic (independent of the thread count), so the assembled values are
//! reproducible.
//!
//! A `NodeId` already indexes `Coords`, so the node → cells incidence is a flat
//! table indexed by node id: no map, no renumbering.

use crate::atoms::NodeId;

/// Greedily colour `n_cells`, where cell `c` touches the nodes
/// `conn[c*nodes_per_cell .. (c+1)*nodes_per_cell]`. Two cells sharing a node
/// are given different colours. Returns the cells grouped by colour (each inner
/// `Vec` lists one colour's cell indices, in increasing cell order).
///
/// `n_nodes` is the width of the node id space (`Coords::node_count`): the
/// incidence table has one entry per id, used by this cell set or not.
///
/// The incidence (node → the cells touching it) is a **CSR built by counting
/// sort**: one pass to count, a prefix sum, one pass to fill. One greedy pass
/// then gives each cell, in order, the smallest colour its neighbours do not
/// already use. `O(n_cells × nodes_per_cell × cells_per_node)`. Purely
/// topological, so a caller caches the result and reuses it across assemblies.
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::ops::coloring;
/// // Two cells sharing node 1 cannot be of the same colour: that is what makes
/// // the scatter parallel without a race.
/// let conn = [NodeId(0), NodeId(1), NodeId(1), NodeId(2)];
/// let couleurs = coloring::greedy_color_nodes(2, 2, &conn, 3);
/// assert_eq!(couleurs, vec![vec![0], vec![1]]);
/// // Without a shared node, one colour is enough.
/// let conn = [NodeId(0), NodeId(1), NodeId(2), NodeId(3)];
/// assert_eq!(coloring::greedy_color_nodes(2, 2, &conn, 4), vec![vec![0, 1]]);
/// ```
pub fn greedy_color_nodes(
    n_cells: usize,
    nodes_per_cell: usize,
    conn: &[NodeId],
    n_nodes: usize,
) -> Vec<Vec<usize>> {
    if n_cells == 0 {
        return Vec::new();
    }
    debug_assert_eq!(conn.len(), n_cells * nodes_per_cell);
    let n_ids = conn
        .iter()
        .map(|n| n.0 as usize + 1)
        .max()
        .unwrap_or(0)
        .max(n_nodes);
    let nodes_of = |c: usize| &conn[c * nodes_per_cell..(c + 1) * nodes_per_cell];

    let mut offsets = vec![0usize; n_ids + 1];
    for n in conn {
        offsets[n.0 as usize + 1] += 1;
    }
    for i in 0..n_ids {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut cells = vec![0u32; conn.len()];
    for c in 0..n_cells {
        for n in nodes_of(c) {
            let k = n.0 as usize;
            cells[cursor[k]] = c as u32;
            cursor[k] += 1;
        }
    }

    let mut color = vec![usize::MAX; n_cells];
    // `forbidden_by[col] == c` ⇔ colour `col` is taken by a neighbour of cell `c`.
    // The per-cell stamp avoids clearing this scratch on every cell.
    let mut forbidden_by: Vec<usize> = Vec::new();
    let mut n_colors = 0usize;

    for c in 0..n_cells {
        for n in nodes_of(c) {
            let k = n.0 as usize;
            for &other in &cells[offsets[k]..offsets[k + 1]] {
                let oc = color[other as usize];
                if oc != usize::MAX {
                    if oc >= forbidden_by.len() {
                        forbidden_by.resize(oc + 1, usize::MAX);
                    }
                    forbidden_by[oc] = c;
                }
            }
        }
        // Smallest colour this cell's neighbours do not already use.
        let mut chosen = 0;
        while chosen < forbidden_by.len() && forbidden_by[chosen] == c {
            chosen += 1;
        }
        color[c] = chosen;
        n_colors = n_colors.max(chosen + 1);
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_colors];
    for (c, &col) in color.iter().enumerate() {
        buckets[col].push(c);
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn ids(raw: &[u32]) -> Vec<NodeId> {
        raw.iter().map(|&i| NodeId(i)).collect()
    }

    /// A colouring is valid iff no two cells of the same colour share a node,
    /// and every cell appears exactly once.
    fn assert_valid(
        buckets: &[Vec<usize>],
        n_cells: usize,
        nodes_per_cell: usize,
        conn: &[NodeId],
    ) {
        let mut seen_cells = HashSet::new();
        for bucket in buckets {
            let mut seen_nodes = HashSet::new();
            for &c in bucket {
                assert!(seen_cells.insert(c), "cell {c} appears in two colours");
                for &n in &conn[c * nodes_per_cell..(c + 1) * nodes_per_cell] {
                    assert!(
                        seen_nodes.insert(n),
                        "node {} shared by two cells of the same colour",
                        n.0
                    );
                }
            }
        }
        assert_eq!(
            seen_cells.len(),
            n_cells,
            "not every cell was coloured once"
        );
    }

    #[test]
    fn empty_set_no_colors() {
        assert!(greedy_color_nodes(0, 2, &[], 0).is_empty());
    }

    #[test]
    fn disjoint_cells_share_one_color() {
        // Two cells with no shared node ⇒ a single colour suffices.
        let conn = ids(&[0, 1, 2, 3]);
        let buckets = greedy_color_nodes(2, 2, &conn, 4);
        assert_eq!(buckets.len(), 1);
        assert_valid(&buckets, 2, 2, &conn);
    }

    #[test]
    fn seg2_chain_alternates_two_colors() {
        // SEG2 chain 0-1-2-3-4: consecutive cells share an endpoint ⇒ 2 colours.
        let conn = ids(&[0, 1, 1, 2, 2, 3, 3, 4]);
        let buckets = greedy_color_nodes(4, 2, &conn, 5);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0], vec![0, 2]);
        assert_eq!(buckets[1], vec![1, 3]);
        assert_valid(&buckets, 4, 2, &conn);
    }

    #[test]
    fn quad_grid_is_validly_colored() {
        // n×n QUA4 grid: node id = j*(n+1)+i, cell (i,j) has the 4 corner nodes.
        let n = 8usize;
        let at = |i: usize, j: usize| NodeId((j * (n + 1) + i) as u32);
        let mut conn = Vec::with_capacity(n * n * 4);
        for j in 0..n {
            for i in 0..n {
                conn.extend_from_slice(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let buckets = greedy_color_nodes(n * n, 4, &conn, (n + 1) * (n + 1));
        assert_valid(&buckets, n * n, 4, &conn);
        // A structured quad mesh needs few colours; greedy stays well-bounded.
        assert!(buckets.len() <= 6, "too many colours: {}", buckets.len());
    }
}
