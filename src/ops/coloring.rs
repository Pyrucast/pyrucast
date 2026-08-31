//! Greedy graph colouring of a cell set for conflict-free parallel assembly.
//!
//! Two cells **conflict** when they share a DOF "key". Colouring partitions the
//! cells so that no two cells of the same colour share a key: the cells of one
//! colour can then scatter their element matrices into the global matrix **in
//! parallel** without write conflicts, the colours being processed in sequence.
//! The result is deterministic (independent of the thread count), so the
//! assembled values are reproducible.
//!
//! The key type `K` is **generic** on purpose: the same algorithm serves both
//! the per-fespace colouring used today (keys = shared **nodes**) and a future
//! global colouring (keys = shared global / master DOFs after MPC condensation).
//! Only the key builder changes — this function does not.
//!
//! Underneath, the colouring works on keys numbered `0..n_keys`, and the
//! incidence that drives it is a CSR built by counting sort. The node case —
//! every FE assembly — skips the numbering step entirely
//! ([`greedy_color_nodes`]): a `NodeId` already indexes `Coords`, so it *is* the
//! dense key. Both forms return the same partition on the same input.

use crate::atoms::NodeId;
use std::collections::HashMap;
use std::hash::Hash;

/// Greedily colour `n_cells`, where cell `c` touches the keys
/// `keys[c*keys_per_cell .. (c+1)*keys_per_cell]`. Two cells sharing any key are
/// given different colours. Returns the cells grouped by colour (each inner
/// `Vec` lists one colour's cell indices, in increasing cell order).
///
/// `O(n_cells × keys_per_cell × cells_per_key)`: one pass to build the
/// key→cells incidence, one greedy pass. Purely topological, so a caller caches
/// the result and reuses it across assemblies.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::ops::{element_field, matrix, mesh, scatter};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// # use pyrucast::ops::coloring;
/// # use pyrucast::ops::model;
/// // Deux mailles qui partagent une clé — un nœud — ne peuvent pas être de
/// // la même couleur : c'est ce qui rend le scatter parallèle sans course.
/// let couleurs = coloring::greedy_color(2, 2, &[0u32, 1, 1, 2]);
/// assert_eq!(couleurs.len(), 2); // elles partagent le nœud 1
/// // Sans clé commune, une seule couleur suffit.
/// assert_eq!(coloring::greedy_color(2, 2, &[0u32, 1, 2, 3]).len(), 1);
/// // Et chaque maille apparaît exactement une fois.
/// assert_eq!(couleurs.iter().map(|c| c.len()).sum::<usize>(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn greedy_color<K: Eq + Hash + Copy>(
    n_cells: usize,
    keys_per_cell: usize,
    keys: &[K],
) -> Vec<Vec<usize>> {
    if n_cells == 0 {
        return Vec::new();
    }
    debug_assert_eq!(keys.len(), n_cells * keys_per_cell);
    // Number the distinct keys, then colour on the numbering. One hash lookup
    // per key occurrence — no container allocated per key.
    let mut ids: HashMap<K, u32> = HashMap::new();
    let dense: Vec<u32> = keys
        .iter()
        .map(|&k| {
            let next = ids.len() as u32;
            *ids.entry(k).or_insert(next)
        })
        .collect();
    color_dense(n_cells, keys_per_cell, &dense, ids.len())
}

/// [`greedy_color`] where the keys are **nodes** — the case every FE assembly
/// hits, and the one worth not hashing at all.
///
/// A `NodeId` indexes `Coords` directly, so it *is* the dense key: `n_nodes` is
/// the width of that id space, and the incidence needs no map. Same partition as
/// the generic form on the same input — the greedy pass depends on which cells
/// share a key, never on how the keys are numbered.
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::ops::coloring;
/// // Deux mailles qui partagent le nœud 1 : deux couleurs, comme la forme
/// // générique — mais sans une seule table de hachage.
/// let conn = [NodeId(0), NodeId(1), NodeId(1), NodeId(2)];
/// let couleurs = coloring::greedy_color_nodes(2, 2, &conn, 3);
/// assert_eq!(couleurs, vec![vec![0], vec![1]]);
/// assert_eq!(couleurs, coloring::greedy_color(2, 2, &conn));
/// ```
pub fn greedy_color_nodes(
    n_cells: usize,
    keys_per_cell: usize,
    conn: &[NodeId],
    n_nodes: usize,
) -> Vec<Vec<usize>> {
    if n_cells == 0 {
        return Vec::new();
    }
    debug_assert_eq!(conn.len(), n_cells * keys_per_cell);
    let dense: Vec<u32> = conn.iter().map(|n| n.0).collect();
    let n_keys = dense.iter().map(|&k| k as usize + 1).max().unwrap_or(0);
    color_dense(n_cells, keys_per_cell, &dense, n_keys.max(n_nodes))
}

/// The colouring proper, over keys already numbered `0..n_keys`.
///
/// The incidence (key → the cells touching it) is a **CSR built by counting
/// sort**: one pass to count, a prefix sum, one pass to fill. The map of vectors
/// it replaces allocated a `Vec` per key — ten million of them on a solid mesh,
/// for lists of eight entries each.
fn color_dense(
    n_cells: usize,
    keys_per_cell: usize,
    keys: &[u32],
    n_keys: usize,
) -> Vec<Vec<usize>> {
    let mut offsets = vec![0usize; n_keys + 1];
    for &k in keys {
        offsets[k as usize + 1] += 1;
    }
    for i in 0..n_keys {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut cells = vec![0u32; keys.len()];
    for c in 0..n_cells {
        for &k in &keys[c * keys_per_cell..(c + 1) * keys_per_cell] {
            cells[cursor[k as usize]] = c as u32;
            cursor[k as usize] += 1;
        }
    }

    let mut color = vec![usize::MAX; n_cells];
    // `forbidden_by[col] == c` ⇔ colour `col` is taken by a neighbour of cell `c`.
    // The per-cell stamp avoids clearing this scratch on every cell.
    let mut forbidden_by: Vec<usize> = Vec::new();
    let mut n_colors = 0usize;

    for c in 0..n_cells {
        for &k in &keys[c * keys_per_cell..(c + 1) * keys_per_cell] {
            for &other in &cells[offsets[k as usize]..offsets[k as usize + 1]] {
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

    /// A colouring is valid iff no two cells of the same colour share a key, and
    /// every cell appears exactly once.
    fn assert_valid(buckets: &[Vec<usize>], n_cells: usize, keys_per_cell: usize, keys: &[u32]) {
        let mut seen_cells = HashSet::new();
        for bucket in buckets {
            let mut seen_keys = HashSet::new();
            for &c in bucket {
                assert!(seen_cells.insert(c), "cell {c} appears in two colours");
                for &k in &keys[c * keys_per_cell..(c + 1) * keys_per_cell] {
                    assert!(
                        seen_keys.insert(k),
                        "key {k} shared by two cells of the same colour"
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
        assert!(greedy_color::<u32>(0, 2, &[]).is_empty());
    }

    #[test]
    fn disjoint_cells_share_one_color() {
        // Two cells with no shared key ⇒ a single colour suffices.
        let keys = [0u32, 1, 2, 3];
        let buckets = greedy_color(2, 2, &keys);
        assert_eq!(buckets.len(), 1);
        assert_valid(&buckets, 2, 2, &keys);
    }

    #[test]
    fn seg2_chain_alternates_two_colors() {
        // SEG2 chain 0-1-2-3-4: consecutive cells share an endpoint ⇒ 2 colours.
        let keys = [0u32, 1, 1, 2, 2, 3, 3, 4];
        let buckets = greedy_color(4, 2, &keys);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0], vec![0, 2]);
        assert_eq!(buckets[1], vec![1, 3]);
        assert_valid(&buckets, 4, 2, &keys);
    }

    /// The dense form must give the **same partition** as the generic one: the
    /// scatter accumulates colour by colour, so a different grouping would
    /// change the summation order of the assembly.
    #[test]
    fn dense_and_generic_agree_on_a_quad_grid() {
        let n = 8usize;
        let at = |i: usize, j: usize| NodeId((j * (n + 1) + i) as u32);
        let mut conn = Vec::with_capacity(n * n * 4);
        for j in 0..n {
            for i in 0..n {
                conn.extend_from_slice(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let n_nodes = (n + 1) * (n + 1);
        assert_eq!(
            greedy_color_nodes(n * n, 4, &conn, n_nodes),
            greedy_color(n * n, 4, &conn)
        );
    }

    #[test]
    fn quad_grid_is_validly_colored() {
        // n×n QUA4 grid: node id = j*(n+1)+i, cell (i,j) has the 4 corner nodes.
        let n = 8usize;
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut keys = Vec::with_capacity(n * n * 4);
        for j in 0..n {
            for i in 0..n {
                keys.extend_from_slice(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let buckets = greedy_color(n * n, 4, &keys);
        assert_valid(&buckets, n * n, 4, &keys);
        // A structured quad mesh needs few colours; greedy stays well-bounded.
        assert!(buckets.len() <= 6, "too many colours: {}", buckets.len());
    }
}
