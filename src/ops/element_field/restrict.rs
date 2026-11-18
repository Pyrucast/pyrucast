use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::{Field, SubField, ABSENT_COMPONENT};
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::handle::Handle;

/// Restrict `field` to the cells of `fespace`, keeping the components found there.
///
/// Same cells and same finite-element definition. Returns a new
/// [`ElementField`] with at most one zone per subspace of `fespace`, on that
/// very subspace, each subspace once: one listed twice gives one zone. A cell
/// of the subspace takes the values of `field` at its Gauss points when a zone
/// of `field` holds that cell — same nodes in the same order — under the
/// **same finite-element definition**: same element type, interpolation and
/// quadrature. A zone of another definition is not read; a cell of `field`
/// outside `fespace` is dropped. For each component, the first zone holding
/// the cell and carrying it wins.
///
/// A zone takes the components of the zones of `field` that hold at least one
/// of its cells, in the order of `field`: a zone of `field` sharing no cell
/// with the subspace — or of another definition — adds nothing to it, and a
/// subspace sharing no cell with `field` gives no zone. The test is on the
/// cells, not on the values: a component that is `0.0` everywhere stays. A
/// cell a zone of `field` does not hold is assigned `0.0` for its components.
///
/// To bring a field onto **another** definition, interpolate it from the nodes
/// ([`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss)) or
/// integrate it cell by cell
/// ([`cell_integral`](fn@crate::ops::element_field::cell_integral)). To keep
/// the part of a field lying on a mesh, restrict its own space first:
/// `restrict(&f, &finite_element_space::restrict(&f.support(), &part))`.
///
/// An empty field carries no component: it gives an empty field, as with
/// [`restrict_like`]. A `fespace` on another `Coords` shares no cell with
/// `field` — a node id names a node only in its own `Coords` — and gives an
/// empty field too.
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
    // Every zone of a field stands on one `Coords`.
    let coords = view.zones[0].support().read().coords();
    let all = view.resolve_reads(&components);
    let mut out = ElementField::empty();
    for subspace in &fespace.deduplicate() {
        // Another `Coords` first: no cell in common, nothing to search.
        if !subspace.read().coords().same_object(&coords) {
            continue;
        }
        let overlaps = view.overlaps(subspace, &all);
        // The components of the zones holding one of its cells, in the order
        // of `field`.
        let kept: Vec<String> = components
            .iter()
            .enumerate()
            .filter(|&(c, _)| {
                overlaps
                    .zones
                    .iter()
                    .any(|&(z, _)| all[z][c] != ABSENT_COMPONENT)
            })
            .map(|(_, name)| name.clone())
            .collect();
        if kept.is_empty() {
            continue;
        }
        let mut sub = SubElementField::new(subspace.clone(), kept)
            .expect("the components of a field are valid names");
        let reads = view.resolve_reads(sub.components());
        let points = sub.gauss_count();
        view.gather_rows(&overlaps, &reads, points, sub.values_mut());
        out.add_sub(Handle::new(sub))
            .expect("one zone per distinct subspace of `fespace`, all on the Coords of `field`");
    }
    out
}

/// Read `field` on the subspaces **and the components** of `target`.
///
/// Same cells and same finite-element definition. Zone by zone of `target`,
/// `field` is read as [`restrict`] reads it, on the zone's subspace and with
/// the zone's components: those `target` lacks are dropped, those no zone of
/// `field` gives at its cells are created at `0.0`. Every zone of `target` gets
/// its zone, where [`restrict`]`(field, &target.support())` keeps the
/// components of `field` and leaves out a subspace sharing no cell with it.
///
/// Taking the components of `target` too is what lets the result combine with
/// `target` by the arithmetic operators (`&target + &field.restrict_like(target)`),
/// whatever its zones. A cell takes the values of `field` only from a zone of
/// the same finite-element definition holding the same cell, as with
/// [`restrict`].
///
/// An empty `field` gives an empty field, as with [`restrict`]; so does an
/// empty `target`. A `target` on another `Coords` shares no cell with `field`:
/// its zones are all `0.0`.
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
    let coords = view.zones[0].support().read().coords();
    let mut out = ElementField::empty();
    for h in target.iter() {
        let (subspace, components) = {
            let zone = h.read();
            (zone.support(), zone.components().to_vec())
        };
        // Another `Coords` first: no cell in common, the zone stays at 0.
        let same_coords = subspace.read().coords().same_object(&coords);
        let mut sub = SubElementField::new(subspace.clone(), components)
            .expect("the components of a zone are valid names");
        if same_coords {
            let reads = view.resolve_reads(sub.components());
            let overlaps = view.overlaps(&subspace, &reads);
            let points = sub.gauss_count();
            view.gather_rows(&overlaps, &reads, points, sub.values_mut());
        }
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
        assert!(restrict(&g.k, &reduced).is_empty());
        // `restrict_like` keeps the zone of its target, at 0.
        let target = ElementField::new(&reduced, vec!["k".into()]).unwrap();
        let r = restrict_like(&g.k, &target);
        assert_eq!(r.get(0).unwrap().read().gauss_count(), 1);
        assert!(values(&r, "k").iter().all(|&v| v == 0.0));
    }

    /// `k` on the left column, `m` on the right one: a subspace takes the
    /// components of the zones holding one of its cells.
    #[test]
    fn restrict_takes_the_components_of_the_zones_sharing_a_cell() {
        let g = grid();
        let left = crate::ops::finite_element_space::restrict(&g.fes, &g.sel);
        let right_cells = crate::ops::mesh::difference(&g.grid, &g.sel);
        let right = crate::ops::finite_element_space::restrict(&g.fes, &right_cells);
        let m = ElementField::new(&right, vec!["m".into()]).unwrap();
        for c in 0..2 {
            m.get(0)
                .unwrap()
                .write()
                .set_cell_uniform(c, "m", 7.0)
                .unwrap();
        }
        let mut both = restrict(&g.k, &left);
        both.add_sub(m.get(0).unwrap()).unwrap();

        let r = restrict(&both, &left);
        assert_eq!(Field::components(&r), vec!["k"]);
        let r = restrict(&both, &g.fes);
        assert_eq!(Field::components(&r), vec!["k", "m"]);
        let per_cell = |c: &str| -> Vec<f64> { values(&r, c).chunks(4).map(|v| v[0]).collect() };
        assert_eq!(per_cell("k"), vec![0.0, 0.0, 2.0, 0.0]);
        assert_eq!(per_cell("m"), vec![0.0, 7.0, 0.0, 7.0]);
    }

    #[test]
    fn restrict_onto_another_coords_gives_nothing() {
        let g = grid();
        let other = grid();
        assert!(restrict(&g.k, &other.fes).is_empty());
        let target = ElementField::new(&other.fes, vec!["k".into()]).unwrap();
        assert!(values(&restrict_like(&g.k, &target), "k")
            .iter()
            .all(|&v| v == 0.0));
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
