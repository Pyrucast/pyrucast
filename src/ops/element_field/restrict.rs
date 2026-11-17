use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, ElementFieldView, SubElementField};
use crate::containers::field::{Field, SubField};
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::handle::Handle;

/// Fill `sub` (already on its target subspace) with the values of `source`,
/// component by component, cell by cell: a `(cell, component)` pair a zone of
/// `source` gives under the same finite-element definition is copied at every
/// Gauss point, one no zone gives is left at `0.0`. Shared by [`restrict`] and
/// [`restrict_like`].
fn fill_from(sub: &mut SubElementField, source: &ElementFieldView) {
    // The source's component positions, resolved **once** per zone; the gather
    // then writes straight into the target's rows, in cell order.
    let reads = source.resolve_reads(sub.components());
    let target = sub.support();
    source.gather_cells(&target, &reads, sub.values_mut());
}

/// Restrict `field` to the cells of `fespace`, with every component of `field`.
///
/// Same cells and same finite-element definition. Returns a new [`ElementField`] with one zone per subspace of `fespace`, on
/// that very subspace, each subspace once: one listed twice gives one zone.
/// Each zone carries the union of `field`'s components. A cell of the subspace
/// takes the values of `field` at its Gauss points when a zone of `field` holds
/// that cell — same nodes in the same order — under the **same finite-element
/// definition**: same element type, interpolation and quadrature, on the same
/// `Coords`. A zone of another definition is not read; a cell no zone gives
/// is assigned `0.0`; a cell of `field` outside `fespace` is dropped. For each
/// component, the first zone holding the cell and carrying it wins.
///
/// To bring a field onto **another** definition, interpolate it from the nodes
/// ([`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss)) or
/// integrate it cell by cell
/// ([`cell_integral`](fn@crate::ops::element_field::cell_integral)). To keep
/// the part of a field lying on a mesh, restrict its own space first:
/// `restrict(&f, &finite_element_space::restrict(&f.support(), &part))`.
///
/// An empty field carries no component: it gives an empty field, as with
/// [`restrict_like`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{element_field, finite_element_space, mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: f64| Node::create_in(coords.clone(), &[x, 0.0]).unwrap();
/// # let ligne = mesh::line(&p(0.0), &p(3.0), 3, ElementType::SEG2)?;
/// # let fes = FiniteElementSpace::lagrange1(&ligne)?;
/// # let debuts = (0..3)
/// #     .map(|c| ligne.node(0, c, 0))
/// #     .collect::<Result<Vec<Node>, pyrucast::PyrucastError>>()?;
/// # let gauche = mesh::elements_on(&ligne, &mesh::poi1_from_nodes(&debuts)?, true)?;
/// // On three segments, `k` holds the rank of its cell at each Gauss point.
/// let k = ElementField::new(&fes, vec!["k".into()])?;
/// for c in 0..3 {
///     k.get(0)?.write().set_cell_uniform(c, "k", c as f64)?;
/// }
/// // Restricted to the two first: their values, cell by cell.
/// let part = finite_element_space::restrict(&fes, &gauche);
/// let r = element_field::restrict(&k, &part);
/// assert_eq!(r.get(0)?.read().value(1, 0, "k")?, 1.0);
/// // Restricted to its own space, the field keeps every value.
/// assert_eq!(element_field::restrict(&k, &fes).get(0)?.read().value(2, 0, "k")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn restrict(field: &ElementField, fespace: &FiniteElementSpace) -> ElementField {
    if field.is_empty() {
        return ElementField::empty();
    }
    let components = Field::components(field);
    let view = field.view();
    let mut out = ElementField::empty();
    for subspace in &fespace.deduplicate() {
        let mut sub = SubElementField::new(subspace.clone(), components.clone())
            .expect("the components of a field are valid names");
        fill_from(&mut sub, &view);
        out.add_sub(Handle::new(sub))
            .expect("one zone per distinct subspace of `fespace`, all on its Coords");
    }
    out
}

/// Read `field` on the subspaces **and the components** of `target`.
///
/// Same cells and same finite-element definition. Zone by zone, it is [`restrict`]`(field, &target.support())` cut down to the
/// components of that zone of `target`: those `target` lacks are dropped, those
/// `field` lacks are created at `0.0`. [`restrict`] alone lands on the same
/// subspaces but keeps the components of `field`.
///
/// Taking the components of `target` too is what lets the result combine with
/// `target` by the arithmetic operators (`&target + &field.restrict_like(target)`),
/// whatever its zones. A cell takes the values of `field` only from a zone of
/// the same finite-element definition holding the same cell, as with
/// [`restrict`].
///
/// An empty `field` gives an empty field, as with [`restrict`]; so does an
/// empty `target`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::Field;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{element_field, mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: f64| Node::create_in(coords.clone(), &[x, 0.0]).unwrap();
/// # let ligne = mesh::line(&p(0.0), &p(2.0), 2, ElementType::SEG2)?;
/// // Two spaces on one mesh: their fields lie side by side under `+`.
/// let (fes_a, fes_b) = (FiniteElementSpace::lagrange1(&ligne)?, FiniteElementSpace::lagrange1(&ligne)?);
/// let t = ElementField::new(&fes_a, vec!["T".into()])?;
/// t.get(0)?.write().set_cell_uniform(1, "T", 5.0)?;
/// let cible = ElementField::new(&fes_b, vec!["T".into(), "q".into()])?;
/// // Read on `cible`'s subspace and components, it adds point by point.
/// let somme = (&cible + &element_field::restrict_like(&t, &cible))?;
/// assert_eq!(somme.len(), 1);
/// assert_eq!(somme.get(0)?.read().value(1, 0, "T")?, 5.0);
/// assert_eq!(Field::components(&element_field::restrict_like(&t, &cible)), vec!["T", "q"]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn restrict_like(field: &ElementField, target: &ElementField) -> ElementField {
    if field.is_empty() {
        return ElementField::empty();
    }
    let view = field.view();
    let mut out = ElementField::empty();
    for h in target.iter() {
        let (subspace, components) = {
            let zone = h.read();
            (zone.support(), zone.components().to_vec())
        };
        let mut sub = SubElementField::new(subspace, components)
            .expect("the components of a zone are valid names");
        fill_from(&mut sub, &view);
        out.add_sub(Handle::new(sub))
            .expect("the zones of `target` already hold each component once per subspace");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation, Node, QuadratureRule};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::NodeField;
    use crate::coords::Coords;
    use crate::ops::element_field::interp_to_gauss;

    /// A 2×2 QUA4 grid (cells 0, 1 below, 2, 3 above), its left column `sel`
    /// (cells 0 and 2), its Lagrange-1 space, and `k`: the rank of the cell at
    /// each Gauss point.
    struct Grid {
        nodes: Vec<Node>,
        grid: Mesh,
        sel: Mesh,
        fes: FiniteElementSpace,
        k: ElementField,
    }

    fn grid() -> Grid {
        let coords = Handle::new(Coords::new(2).unwrap());
        let nodes: Vec<Node> = (0..9)
            .map(|k| Node::create_in(coords.clone(), &[(k % 3) as f64, (k / 3) as f64]).unwrap())
            .collect();
        let mut grid = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let k = 3 * j + i;
            grid.add_cell(&[
                nodes[k].id(),
                nodes[k + 1].id(),
                nodes[k + 4].id(),
                nodes[k + 3].id(),
            ])
            .unwrap();
        }
        let left: Vec<Node> = [0, 1, 3, 4, 6, 7]
            .iter()
            .map(|&k| nodes[k].clone())
            .collect();
        let left = crate::ops::mesh::poi1_from_nodes(&left).unwrap();
        let sel = crate::ops::mesh::elements_on(&grid, &left, true).unwrap();
        let fes = FiniteElementSpace::new(&grid, Interpolation::Lagrange1).unwrap();
        let k = ElementField::new(&fes, vec!["k".into()]).unwrap();
        for c in 0..4 {
            k.get(0)
                .unwrap()
                .write()
                .set_cell_uniform(c, "k", c as f64)
                .unwrap();
        }
        Grid {
            nodes,
            grid,
            sel,
            fes,
            k,
        }
    }

    /// Every value of zone 0 of `f`, for `component`, cell by cell then point.
    fn values(f: &ElementField, component: &str) -> Vec<f64> {
        let z = f.get(0).unwrap();
        let z = z.read();
        (0..z.cell_count())
            .flat_map(|c| (0..z.gauss_count()).map(move |g| (c, g)))
            .map(|(c, g)| z.value(c, g, component).unwrap())
            .collect()
    }

    #[test]
    fn two_spaces_on_one_mesh_pass_every_value() {
        let g = grid();
        let other = FiniteElementSpace::new(&g.grid, Interpolation::Lagrange1).unwrap();
        let r = restrict(&g.k, &other);
        assert!(r
            .get(0)
            .unwrap()
            .read()
            .support()
            .same_object(&other.get(0).unwrap()));
        assert_eq!(values(&r, "k"), values(&g.k, "k"));
        // On the second space, it adds to a field of that space in one zone.
        let one = ElementField::new(&other, vec!["k".into()]).unwrap();
        assert_eq!((&one + &r).unwrap().len(), 1);
    }

    #[test]
    fn a_selection_takes_the_values_of_its_cells() {
        let g = grid();
        let part = crate::ops::finite_element_space::restrict(&g.fes, &g.sel);
        let r = restrict(&g.k, &part);
        let per_cell: Vec<f64> = values(&r, "k").chunks(4).map(|c| c[0]).collect();
        assert_eq!(per_cell, vec![0.0, 2.0]);
        // Measured another way: the Gauss interpolation of a nodal field on the
        // selection is the restriction of its interpolation on the grid.
        let support = crate::ops::mesh::poi1_from_nodes(&g.nodes).unwrap();
        let t = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
        for (i, n) in g.nodes.iter().enumerate() {
            t.get(0)
                .unwrap()
                .write()
                .set_value(n.id(), "T", (i * i) as f64)
                .unwrap();
        }
        let on_grid = interp_to_gauss(&t, &g.fes).unwrap();
        let on_part = interp_to_gauss(&t, &part).unwrap();
        assert_eq!(
            values(&restrict(&on_grid, &part), "T"),
            values(&on_part, "T")
        );
    }

    #[test]
    fn a_cell_the_field_lacks_reads_zero() {
        let g = grid();
        // The field on the left column alone, read on the whole grid.
        let part = crate::ops::finite_element_space::restrict(&g.fes, &g.sel);
        let left = restrict(&g.k, &part);
        let back = restrict(&left, &g.fes);
        let per_cell: Vec<f64> = values(&back, "k").chunks(4).map(|c| c[0]).collect();
        assert_eq!(per_cell, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn another_definition_is_not_read() {
        let g = grid();
        let reduced = FiniteElementSpace::with(
            &g.grid,
            &[(Interpolation::Lagrange1, QuadratureRule::Reduced)],
        )
        .unwrap();
        let r = restrict(&g.k, &reduced);
        assert_eq!(r.get(0).unwrap().read().gauss_count(), 1);
        assert!(values(&r, "k").iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_first_zone_carrying_a_component_wins() {
        let g = grid();
        // Two zones on two spaces of the grid: `k` in both, `m` in the second.
        let second = FiniteElementSpace::new(&g.grid, Interpolation::Lagrange1).unwrap();
        let mut other =
            SubElementField::new(second.get(0).unwrap(), vec!["k".into(), "m".into()]).unwrap();
        for c in 0..4 {
            other.set_cell_uniform(c, "k", 100.0).unwrap();
            other.set_cell_uniform(c, "m", 7.0).unwrap();
        }
        let mut both = ElementField::empty();
        both.add_sub(g.k.get(0).unwrap()).unwrap();
        both.add_sub(Handle::new(other)).unwrap();
        let r = restrict(&both, &g.fes);
        assert_eq!(values(&r, "k"), values(&g.k, "k"));
        assert!(values(&r, "m").iter().all(|&v| v == 7.0));
    }

    #[test]
    fn a_subspace_listed_twice_gives_one_zone() {
        let g = grid();
        let mut twice = FiniteElementSpace::empty();
        twice.add_sub(g.fes.get(0).unwrap()).unwrap();
        twice.add_sub(g.fes.get(0).unwrap()).unwrap();
        assert_eq!(restrict(&g.k, &twice).len(), 1);
    }

    #[test]
    fn restrict_like_takes_the_components_of_the_target() {
        let g = grid();
        let other = FiniteElementSpace::new(&g.grid, Interpolation::Lagrange1).unwrap();
        let target = ElementField::new(&other, vec!["q".into(), "k".into()]).unwrap();
        let r = restrict_like(&g.k, &target);
        assert_eq!(Field::components(&r), vec!["q", "k"]);
        assert_eq!(values(&r, "k"), values(&g.k, "k"));
        assert!(values(&r, "q").iter().all(|&v| v == 0.0));
        assert!(restrict_like(&ElementField::empty(), &target).is_empty());
    }
}
