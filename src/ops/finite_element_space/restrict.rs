use crate::aggregate::Aggregate;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::mesh::Mesh;
use crate::handle::Handle;
use crate::ops::mesh::intersection::keep_cells;

/// Restrict `fespace` to the cells of `mesh`, keeping each subspace's definition.
///
/// Each subspace keeps its interpolation and its quadrature. Zone by zone of `fespace`, in order: each subspace of the result lies on
/// the cells of its zone that `mesh` holds too — same type, same nodes in the
/// same order, as [`intersection`](fn@crate::ops::mesh::intersection) — with the
/// same interpolation and the same quadrature. A zone with no cell in common
/// gives an empty subspace, in place. A subspace whose zone is kept whole comes
/// back **as it is**, shared: a field laid on it still combines with one laid
/// on the result.
///
/// It is the way to a field at the Gauss points on a part of a mesh:
/// [`element_field::restrict`](fn@crate::ops::element_field::restrict)`(&f,
/// &restrict(&f.support(), &part))`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{finite_element_space, mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: f64| Node::create_in(coords.clone(), &[x, 0.0]).unwrap();
/// // Three segments, and the space on the two first.
/// let ligne = mesh::line(&p(0.0), &p(3.0), 3, ElementType::SEG2)?;
/// let fes = FiniteElementSpace::lagrange1(&ligne)?;
/// let debuts = (0..3)
///     .map(|c| ligne.node(0, c, 0))
///     .collect::<Result<Vec<Node>, pyrucast::PyrucastError>>()?;
/// let gauche = mesh::elements_on(&ligne, &mesh::poi1_from_nodes(&debuts)?, true)?;
/// let part = finite_element_space::restrict(&fes, &gauche);
/// assert_eq!(part.get(0)?.read().cell_count(), 2);
/// assert_eq!(part.get(0)?.read().quadrature(), fes.get(0)?.read().quadrature());
/// // Kept whole, the subspace is the subspace itself.
/// assert!(finite_element_space::restrict(&fes, &ligne).get(0)?.same_object(&fes.get(0)?));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn restrict(fespace: &FiniteElementSpace, mesh: &Mesh) -> FiniteElementSpace {
    let mut out = FiniteElementSpace::empty();
    for sub in fespace {
        let (zone, interpolation, quadrature) = {
            let s = sub.read();
            (s.submesh(), s.interpolation(), s.quadrature())
        };
        let kept = keep_cells(&zone, mesh, true);
        let part = if kept.same_object(&zone) {
            sub.clone()
        } else {
            Handle::new(
                SubFiniteElementSpace::new(kept, interpolation, quadrature)
                    .expect("a definition that holds on a zone holds on part of its cells"),
            )
        };
        out.add_sub(part)
            .expect("the subspaces of `fespace` share its Coords");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation, Node, QuadratureRule};
    use crate::coords::Coords;

    /// A 2×2 QUA4 grid (cells 0, 1 below, 2, 3 above) and its left column
    /// (cells 0 and 2), selected through `elements_on`.
    fn grid() -> (Mesh, Mesh) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = (0..9)
            .map(|k| Node::create_in(coords.clone(), &[(k % 3) as f64, (k / 3) as f64]).unwrap())
            .collect();
        let mut grid = Mesh::from_submesh(crate::containers::mesh::SubMesh::new(
            coords.clone(),
            ElementType::QUA4,
        ));
        for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let k = 3 * j + i;
            grid.add_cell(&[n[k].id(), n[k + 1].id(), n[k + 4].id(), n[k + 3].id()])
                .unwrap();
        }
        let left: Vec<Node> = [0, 1, 3, 4, 6, 7].iter().map(|&k| n[k].clone()).collect();
        let left = crate::ops::mesh::poi1_from_nodes(&left).unwrap();
        let sel = crate::ops::mesh::elements_on(&grid, &left, true).unwrap();
        (grid, sel)
    }

    #[test]
    fn a_selection_keeps_the_definition_on_its_cells() {
        let (grid, sel) = grid();
        let fes = FiniteElementSpace::new(&grid, Interpolation::Lagrange1).unwrap();
        let part = restrict(&fes, &sel);
        assert_eq!(part.len(), 1);
        let s = part.get(0).unwrap();
        let s = s.read();
        assert_eq!(s.cell_count(), 2);
        assert_eq!(s.interpolation(), Interpolation::Lagrange1);
        assert_eq!(s.quadrature(), QuadratureRule::Gauss);
    }

    #[test]
    fn a_subspace_kept_whole_is_the_subspace_itself() {
        let (grid, _sel) = grid();
        let fes = FiniteElementSpace::new(&grid, Interpolation::Lagrange1).unwrap();
        let whole = restrict(&fes, &grid);
        assert!(whole.get(0).unwrap().same_object(&fes.get(0).unwrap()));
        // No cell in common: an empty subspace, in place.
        let none = restrict(&fes, &crate::ops::mesh::difference(&grid, &grid));
        assert_eq!(none.len(), 1);
        assert_eq!(none.get(0).unwrap().read().cell_count(), 0);
    }
}
