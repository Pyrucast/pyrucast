use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::handle::{Handle, ReadGuard};
use crate::parallel::*;
use std::collections::HashSet;

/// The cells of `a` that `b` holds too.
///
/// Two cells are the same with the same type and the same nodes in the same
/// order. Zone by zone of `a`, in order: each zone of the result keeps the cells of
/// that zone found in a zone of `b` of the same element type, in `a`'s order,
/// with `a`'s connectivity and face colour. A zone with no cell in common stays
/// in place, empty — zone `i` of the result is zone `i` of `a`. A zone kept
/// whole comes back **as it is**, shared: when `b` lists that very zone, or
/// holds every one of its cells. A field laid on that zone then still combines
/// with a field laid on the result.
///
/// A cell is the same only with the same nodes **in the same order**: the same
/// triangle numbered from another vertex, or turned over, is not. Meshes on
/// two different `Coords` share no cell. The cells of `b` that `a` lacks are
/// [`difference`]`(b, a)`; `difference(a, b) | difference(b, a)` is the
/// symmetric difference.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::Mesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: f64| Node::create_in(coords.clone(), &[x, 0.0]).unwrap();
/// // Three segments, and the two first taken back by the nodes they rest on.
/// let ligne = mesh::line(&p(0.0), &p(3.0), 3, ElementType::SEG2)?;
/// let debuts = (0..3)
///     .map(|c| ligne.node(0, c, 0))
///     .collect::<Result<Vec<Node>, pyrucast::PyrucastError>>()?;
/// let gauche = mesh::elements_on(&ligne, &mesh::poi1_from_nodes(&debuts)?, true)?;
/// assert_eq!(mesh::intersection(&ligne, &gauche).cell_count(), 2);
/// // A zone kept whole is the zone itself.
/// let tout = mesh::intersection(&ligne, &ligne);
/// assert!(tout.get(0)?.same_object(&ligne.get(0)?));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn intersection(a: &Mesh, b: &Mesh) -> Mesh {
    let mut out = Mesh::empty();
    for zone in a {
        out.add_sub(keep_cells(zone, b, true))
            .expect("the zones of `a` share its Coords");
    }
    out
}

/// The cells of `a` that `b` does not hold.
///
/// Two cells are the same with the same type and the same nodes in the same
/// order. Zone by zone of `a`, in order, as [`intersection`]: each zone of the result
/// keeps the cells of that zone that no zone of `b` of the same element type
/// holds, in `a`'s order, with `a`'s connectivity and face colour. A zone with
/// no cell left stays in place, empty; a zone kept whole — `b` holds none of
/// its cells — comes back as it is, shared. Meshes on two different `Coords`
/// share no cell, so `difference(a, b)` keeps all of `a`.
///
/// `intersection(a, b)` and `difference(a, b)` split each zone of `a` in two;
/// `difference(a, b) | difference(b, a)` is the symmetric difference.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::Mesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: f64| Node::create_in(coords.clone(), &[x, 0.0]).unwrap();
/// // Three segments, less the two first: the last one is left.
/// let ligne = mesh::line(&p(0.0), &p(3.0), 3, ElementType::SEG2)?;
/// let debuts = (0..3)
///     .map(|c| ligne.node(0, c, 0))
///     .collect::<Result<Vec<Node>, pyrucast::PyrucastError>>()?;
/// let gauche = mesh::elements_on(&ligne, &mesh::poi1_from_nodes(&debuts)?, true)?;
/// let reste = mesh::difference(&ligne, &gauche);
/// assert_eq!(reste.cell_count(), 1);
/// assert_eq!(reste.node(0, 0, 0)?.id(), debuts[2].id());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn difference(a: &Mesh, b: &Mesh) -> Mesh {
    let mut out = Mesh::empty();
    for zone in a {
        out.add_sub(keep_cells(zone, b, false))
            .expect("the zones of `a` share its Coords");
    }
    out
}

/// Keep the cells of `zone` that are in `b` (`in_b`), or those that are not.
///
/// A cell is in `b` when a zone of `b` of the same element type, on the same
/// `Coords`, lists the same nodes in the same order. When `b` lists `zone`
/// itself, nothing is looked up. A zone kept whole comes back as it is; any
/// other is a new zone, with `zone`'s face colour.
pub(crate) fn keep_cells(zone: &Handle<SubMesh>, b: &Mesh, in_b: bool) -> Handle<SubMesh> {
    let (coords, element_type, color, total) = {
        let z = zone.read();
        (
            z.coords(),
            z.element_type(),
            z.face_color(),
            z.connectivity().len(),
        )
    };
    let whole = b.iter().any(|h| h.same_object(zone));
    if whole && in_b {
        return zone.clone();
    }
    let kept: Vec<NodeId> = if whole {
        Vec::new()
    } else {
        let z = zone.read();
        let npc = element_type.nodes_per_cell();
        // The zones of `b` that can hold a cell of `zone`, read in place.
        let candidates: Vec<ReadGuard<SubMesh>> = b
            .iter()
            .map(|h| h.read())
            .filter(|s| s.element_type() == element_type && s.coords().same_object(&coords))
            .collect();
        let held: HashSet<&[NodeId]> = candidates
            .iter()
            .flat_map(|s| s.connectivity().chunks_exact(npc))
            .collect();
        z.connectivity()
            .par_chunks_exact(npc)
            .with_min_len(MIN_PARALLEL_LEN)
            .filter(|cell| held.contains(cell) == in_b)
            .flat_map_iter(|cell| cell.iter().copied())
            .collect()
    };
    if kept.len() == total {
        return zone.clone();
    }
    // The guards are released: building the zone write-locks the `Coords`.
    let mut part = SubMesh::from_connectivity(coords, element_type, kept)
        .expect("the kept cells are cells of `zone`, on its Coords");
    part.set_face_color(color);
    Handle::new(part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::coords::Coords;

    /// A 2×2 QUA4 grid, cells numbered row by row from the bottom left (0, 1
    /// below, 2, 3 above), and its left column `sel` (cells 0 and 2) selected
    /// through `elements_on`. Returns (coords, nodes, grid, sel).
    fn make_grid() -> (Handle<Coords>, Vec<Node>, Mesh, Mesh) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = (0..9)
            .map(|k| Node::create_in(coords.clone(), &[(k % 3) as f64, (k / 3) as f64]).unwrap())
            .collect();
        let mut grid = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let k = 3 * j + i;
            grid.add_cell(&[n[k].id(), n[k + 1].id(), n[k + 4].id(), n[k + 3].id()])
                .unwrap();
        }
        let left: Vec<Node> = [0, 1, 3, 4, 6, 7].iter().map(|&k| n[k].clone()).collect();
        let left = crate::ops::mesh::poi1_from_nodes(&left).unwrap();
        let sel = crate::ops::mesh::elements_on(&grid, &left, true).unwrap();
        (coords, n, grid, sel)
    }

    /// The cells of zone 0 of `mesh`, as node id lists.
    fn cells(mesh: &Mesh) -> Vec<Vec<NodeId>> {
        let s = mesh.get(0).unwrap().read();
        let npc = s.element_type().nodes_per_cell();
        s.connectivity()
            .chunks_exact(npc)
            .map(|c| c.to_vec())
            .collect()
    }

    #[test]
    fn the_grid_splits_into_its_two_columns() {
        let (_c, _n, grid, sel) = make_grid();
        let all = cells(&grid);
        assert_eq!(cells(&sel), vec![all[0].clone(), all[2].clone()]);
        assert_eq!(
            cells(&intersection(&grid, &sel)),
            vec![all[0].clone(), all[2].clone()]
        );
        assert_eq!(
            cells(&difference(&grid, &sel)),
            vec![all[1].clone(), all[3].clone()]
        );
        // The two parts of a zone share nothing and leave nothing out.
        let (inside, outside) = (intersection(&grid, &sel), difference(&grid, &sel));
        assert_eq!(
            inside.cell_count() + outside.cell_count(),
            grid.cell_count()
        );
        assert_eq!(intersection(&inside, &outside).cell_count(), 0);
    }

    #[test]
    fn a_zone_kept_whole_is_the_zone_itself() {
        let (_c, _n, grid, sel) = make_grid();
        let zone = grid.get(0).unwrap();
        // `b` lists the zone: nothing is looked up.
        assert!(intersection(&grid, &grid)
            .get(0)
            .unwrap()
            .same_object(&zone));
        assert_eq!(
            difference(&grid, &grid).get(0).unwrap().read().cell_count(),
            0
        );
        // Every cell found in another zone object.
        let copy = intersection(&sel, &grid);
        assert!(copy.get(0).unwrap().same_object(&sel.get(0).unwrap()));
        // No cell in common: the difference is the zone itself.
        let other = difference(&grid, &sel);
        assert!(difference(&other, &sel)
            .get(0)
            .unwrap()
            .same_object(&other.get(0).unwrap()));
    }

    #[test]
    fn a_cell_numbered_from_another_node_is_not_the_same() {
        let (coords, _n, grid, _sel) = make_grid();
        let first = cells(&grid)[0].clone();
        let mut turned = SubMesh::new(coords.clone(), ElementType::QUA4);
        turned
            .add_cell(&[first[1], first[2], first[3], first[0]])
            .unwrap();
        let turned = Mesh::from_submesh(turned);
        assert_eq!(intersection(&grid, &turned).cell_count(), 0);
        // The same nodes in the same order are the same cell.
        let mut same = SubMesh::new(coords, ElementType::QUA4);
        same.add_cell(&first).unwrap();
        assert_eq!(
            intersection(&grid, &Mesh::from_submesh(same)).cell_count(),
            1
        );
    }

    #[test]
    fn a_zone_without_common_cell_stays_empty_in_place() {
        let (coords, n, grid, _sel) = make_grid();
        // Two zones: the grid, and a segment no zone of `b` holds.
        let mut a = Mesh::empty();
        a.add_sub(grid.get(0).unwrap()).unwrap();
        let mut seg = SubMesh::new(coords, ElementType::SEG2);
        seg.add_cell(&[n[0].id(), n[1].id()]).unwrap();
        a.add_sub(Handle::new(seg)).unwrap();
        let r = intersection(&a, &grid);
        assert_eq!(r.len(), 2);
        assert_eq!(r.cell_counts().unwrap(), vec![4, 0]);
        assert_eq!(
            r.element_types().unwrap(),
            vec![ElementType::QUA4, ElementType::SEG2]
        );
    }

    #[test]
    fn another_coords_shares_no_cell() {
        let (_c, _n, grid, _sel) = make_grid();
        let (_c2, _n2, other, _sel2) = make_grid();
        assert_eq!(intersection(&grid, &other).cell_count(), 0);
        let rest = difference(&grid, &other);
        assert!(rest.get(0).unwrap().same_object(&grid.get(0).unwrap()));
    }

    #[test]
    fn a_new_zone_keeps_the_face_colour() {
        let (_c, _n, grid, sel) = make_grid();
        let color = crate::atoms::RgbColor::new(10, 20, 30);
        grid.get(0).unwrap().write().set_face_color(color);
        assert_eq!(
            difference(&grid, &sel).get(0).unwrap().read().face_color(),
            color
        );
    }
}
