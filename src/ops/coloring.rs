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

    // key → cells touching it (the conflict graph, stored implicitly).
    let mut incidence: HashMap<K, Vec<usize>> = HashMap::new();
    for c in 0..n_cells {
        for &k in &keys[c * keys_per_cell..(c + 1) * keys_per_cell] {
            incidence.entry(k).or_default().push(c);
        }
    }

    let mut color = vec![usize::MAX; n_cells];
    // `forbidden_by[col] == c` ⇔ colour `col` is taken by a neighbour of cell `c`.
    // The per-cell stamp avoids clearing this scratch on every cell.
    let mut forbidden_by: Vec<usize> = Vec::new();
    let mut n_colors = 0usize;

    for c in 0..n_cells {
        for &k in &keys[c * keys_per_cell..(c + 1) * keys_per_cell] {
            for &other in &incidence[&k] {
                let oc = color[other];
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
