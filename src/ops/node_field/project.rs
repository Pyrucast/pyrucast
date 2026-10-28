//! A nodal field carried onto the nodes of another mesh, through the shape
//! functions of the cells that interpolate it — Cast3M `PROI` onto a geometry.

use std::collections::HashMap;

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::field::{Field, SubField};
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::ops::geom::{locate_points, Location};
use crate::parallel::*;

/// `field` interpolated by the cells of `mesh` at the nodes of `target`: a node
/// of `target` that sits in a cell of `mesh` takes `Σᵢ Nᵢ(ξ) f(nodeᵢ)`, the
/// shape functions of that cell at the node's reference coordinates `ξ` —
/// quadratic cells interpolate quadratically. Every component of `field` is
/// carried.
///
/// A node of `target` outside `mesh` — farther than `tol` outside every
/// cell's reference domain — gets zero, as a nodal field reads a node it does
/// not define: projecting a patch onto a larger mesh is a normal thing. `tol`
/// is that reference-domain slack, without unit; the Python face defaults it
/// to `1e-6`, as for `embedded`. A point on a face shared by two cells is read in the first one,
/// which gives the same value for a continuous field.
///
/// `field` is read through its zones, first zone wins, and a node of `mesh`
/// it does not define reads zero. `mesh` and `target` may live in different
/// `Coords` — moving a field onto a remeshed part —, each read in its active
/// configuration. The result has one zone per submesh of `target`, like
/// [`positions`](fn@crate::ops::node_field::positions).
///
/// An empty `field` or `target` gives an empty field: nothing to carry, or
/// nowhere to carry it. An empty `mesh` holds no node of `target`: every one
/// gets zero.
///
/// Errors: `field` and `mesh` on different `Coords` (the field's nodes are the
/// ones `mesh`'s cells interpolate); `mesh` and `target` in different
/// dimensions.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{mesh, node_field};
/// // A temperature T = 10 + 20 x on a triangle…
/// let coords = Handle::new(Coords::new(2)?);
/// let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
///     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// let triangle = Mesh::from_submesh(sm);
/// let t = NodeField::new(&triangle, vec!["T".into()])?;
/// for (node, x) in n.iter().zip([0.0, 2.0, 0.0]) {
///     t.get(0)?.write().set_value(node.id(), "T", 10.0 + 20.0 * x)?;
/// }
/// // …carried onto two points of another `Coords`, the second outside.
/// let other = Handle::new(Coords::new(2)?);
/// let p: Vec<Node> = [[0.5, 0.5], [5.0, 5.0]]
///     .iter().map(|q| Node::create_in(other.clone(), q).unwrap()).collect();
/// let points = mesh::poi1_from_nodes(&p)?;
/// let carried = node_field::project(&t, &triangle, &points, 1e-6)?;
/// let z = carried.get(0)?.read();
/// assert!((z.value(p[0].id(), "T")? - 20.0).abs() < 1e-12);
/// assert_eq!(z.value(p[1].id(), "T")?, 0.0); // outside: zero, a patch
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn project(field: &NodeField, mesh: &Mesh, target: &Mesh, tol: f64) -> Result<NodeField> {
    let components = Field::components(field);
    if components.is_empty() || target.is_empty() {
        return Ok(NodeField::default());
    }
    let target_coords = target.coords()?;
    // A mesh without zones locates nothing: every node of `target` reads zero.
    if !mesh.is_empty() {
        let mesh_coords = mesh.coords()?;
        if !mesh_coords.same_object(&field.coords()?) {
            return Err(PyrucastError::Message(
                "project: `field` and `mesh` hang from different Coords — the cells of \
                 `mesh` interpolate the field from its own nodes"
                    .into(),
            ));
        }
        let (dim, target_dim) = (mesh_coords.read().dim(), target_coords.read().dim());
        if dim != target_dim {
            return Err(PyrucastError::Message(format!(
                "project: `mesh` is {dim}-D and `target` {target_dim}-D"
            )));
        }
    }

    // The distinct nodes of `target` and where they are, located once.
    let mut index: HashMap<NodeId, usize> = HashMap::new();
    let mut points: Vec<Vec<f64>> = Vec::new();
    {
        let c = target_coords.read();
        for sm in target {
            for &nid in sm.read().connectivity() {
                if let std::collections::hash_map::Entry::Vacant(e) = index.entry(nid) {
                    e.insert(points.len());
                    points.push(c.position(nid)?.to_vec());
                }
            }
        }
    }
    let locations = locate_points(mesh, &points, tol)?;

    // The values at every distinct node, node-major: each slot written once.
    let n_comp = components.len();
    let view = field.view()?;
    let reads = view.resolve_reads(&components);
    let mut values = vec![0.0; points.len() * n_comp];
    values
        .par_chunks_mut(n_comp)
        .zip(locations.par_iter())
        .with_min_len(MIN_PARALLEL_LEN)
        .for_each_init(Vec::new, |cell, (out, location)| {
            if let Some(Location { weights, nodes, .. }) = location {
                cell.resize(nodes.len() * n_comp, 0.0);
                view.gather_cell(nodes, &reads, cell);
                for (k, w) in weights.iter().enumerate() {
                    for (o, v) in out.iter_mut().zip(&cell[k * n_comp..(k + 1) * n_comp]) {
                        *o += w * v;
                    }
                }
            }
        });

    let mut out = NodeField::default();
    for sm in target {
        let mut sub = SubNodeField::from_support(sm, components.clone())?;
        let nodes = sub.node_ids();
        for (row, nid) in sub.values_mut().chunks_mut(n_comp).zip(&nodes) {
            let at = index[nid] * n_comp;
            row.copy_from_slice(&values[at..at + n_comp]);
        }
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::ops::mesh::poi1_from_nodes;

    /// An `n × n` grid of QUA4 on the unit square, and its nodes row by row.
    fn square(n: usize) -> (Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let nodes: Vec<Node> = (0..=n)
            .flat_map(|j| (0..=n).map(move |i| (i, j)))
            .map(|(i, j)| {
                Node::create_in(coords.clone(), &[i as f64 / n as f64, j as f64 / n as f64])
                    .unwrap()
            })
            .collect();
        let at = |i: usize, j: usize| nodes[j * (n + 1) + i].id();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        for j in 0..n {
            for i in 0..n {
                sm.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                    .unwrap();
            }
        }
        (Mesh::from_submesh(sm), nodes)
    }

    /// The nodal field `f(x, y)` on `nodes`, one zone on `mesh`.
    fn sampled(mesh: &Mesh, nodes: &[Node], f: impl Fn(f64, f64) -> f64) -> NodeField {
        let field = NodeField::new(mesh, vec!["a".into(), "b".into()]).unwrap();
        {
            let mut z = field.get(0).unwrap().write();
            for n in nodes {
                let p = n.position().unwrap();
                z.set_value(n.id(), "a", f(p[0], p[1])).unwrap();
                z.set_value(n.id(), "b", -f(p[0], p[1])).unwrap();
            }
        }
        field
    }

    /// Points of a fresh `Coords`, as one POI1 zone.
    fn points(at: &[[f64; 2]]) -> (Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let nodes: Vec<Node> = at
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        (poi1_from_nodes(&nodes).unwrap(), nodes)
    }

    /// A linear field is exact through bilinear cells, every component, on
    /// another `Coords`; a point off the square gets zero.
    #[test]
    fn a_linear_field_comes_back_exactly() {
        let (mesh, nodes) = square(3);
        let f = |x: f64, y: f64| 2.0 + 3.0 * x - 5.0 * y;
        let field = sampled(&mesh, &nodes, f);
        let at = [[0.1, 0.2], [0.5, 0.5], [0.97, 0.33], [1.0, 1.0], [1.5, 0.5]];
        let (target, targets) = points(&at);

        let carried = project(&field, &mesh, &target, 1e-6).unwrap();
        let z = carried.get(0).unwrap().read();
        for (node, p) in targets.iter().zip(&at[..4]) {
            assert!((z.value(node.id(), "a").unwrap() - f(p[0], p[1])).abs() < 1e-12);
            assert!((z.value(node.id(), "b").unwrap() + f(p[0], p[1])).abs() < 1e-12);
        }
        assert_eq!(z.value(targets[4].id(), "a").unwrap(), 0.0);
    }

    /// Through straight TRI6, a quadratic field is exact too.
    #[test]
    fn quadratic_cells_interpolate_quadratically() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let corners = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let mids = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let nodes: Vec<Node> = corners
            .iter()
            .chain(&mids)
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI6);
        let ids: Vec<NodeId> = nodes.iter().map(Node::id).collect();
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);
        let f = |x: f64, y: f64| x * x + x * y - 3.0 * y * y + 1.0;
        let field = sampled(&mesh, &nodes, f);
        let at = [[0.3, 0.4], [1.2, 0.5], [0.1, 1.7]];
        let (target, targets) = points(&at);

        let carried = project(&field, &mesh, &target, 1e-6).unwrap();
        let z = carried.get(0).unwrap().read();
        for (node, p) in targets.iter().zip(&at) {
            assert!((z.value(node.id(), "a").unwrap() - f(p[0], p[1])).abs() < 1e-12);
        }
    }

    /// `tol` is the slack beyond the reference element: a point just off the
    /// edge is caught with the usual 1e-6, left at zero with less.
    #[test]
    fn the_tolerance_catches_a_point_just_outside() {
        let (mesh, nodes) = square(1);
        let field = sampled(&mesh, &nodes, |x, _| 1.0 + x);
        // ξ = 2x − 1 = 1 + 4e-7 past the right edge.
        let (target, targets) = points(&[[1.0 + 2e-7, 0.5]]);
        let caught = project(&field, &mesh, &target, 1e-6).unwrap();
        let value = caught.get(0).unwrap().read().value(targets[0].id(), "a");
        assert!((value.unwrap() - 2.0).abs() < 1e-6);
        let missed = project(&field, &mesh, &target, 1e-8).unwrap();
        let value = missed.get(0).unwrap().read().value(targets[0].id(), "a");
        assert_eq!(value.unwrap(), 0.0);
    }

    /// One zone per submesh of `target`, a node they share carrying the same
    /// value in both; a field split in zones is read first zone first, and a
    /// node of `mesh` it does not define reads zero.
    #[test]
    fn zones_on_both_sides() {
        let (mesh, nodes) = square(1);
        // Zone 1 defines every node at 1; zone 0, read first, the left two at 3.
        let ones = NodeField::new(&mesh, vec!["a".into()]).unwrap();
        for n in &nodes {
            ones.get(0)
                .unwrap()
                .write()
                .set_value(n.id(), "a", 1.0)
                .unwrap();
        }
        let left = poi1_from_nodes(&[nodes[0].clone(), nodes[2].clone()]).unwrap();
        let threes = NodeField::new(&left, vec!["a".into()]).unwrap();
        for n in [&nodes[0], &nodes[2]] {
            threes
                .get(0)
                .unwrap()
                .write()
                .set_value(n.id(), "a", 3.0)
                .unwrap();
        }
        let mut field = NodeField::default();
        field.add_sub(threes.get(0).unwrap()).unwrap();
        field.add_sub(ones.get(0).unwrap()).unwrap();

        let coords = Handle::new(Coords::new(2).unwrap());
        let p: Vec<Node> = [[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]]
            .iter()
            .map(|q| Node::create_in(coords.clone(), q).unwrap())
            .collect();
        let mut target = Mesh::empty();
        for pair in [[0, 1], [1, 2]] {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[p[pair[0]].id(), p[pair[1]].id()]).unwrap();
            target.add_sub(Handle::new(sm)).unwrap();
        }

        let carried = project(&field, &mesh, &target, 1e-6).unwrap();
        assert_eq!(carried.len(), 2);
        for zone in &carried {
            let z = zone.read();
            assert!((z.value(p[1].id(), "a").unwrap() - 2.0).abs() < 1e-12);
        }

        // A node of `mesh` the field leaves out reads zero: the right side.
        let carried = project(&threes, &mesh, &target, 1e-6).unwrap();
        let z = carried.get(0).unwrap().read();
        assert!((z.value(p[1].id(), "a").unwrap() - 1.5).abs() < 1e-12);
    }

    /// Nothing to carry or nowhere to carry it: an empty field. No cell to
    /// carry it through: zero everywhere.
    #[test]
    fn empty_sides_are_no_error() {
        let (mesh, nodes) = square(1);
        let field = sampled(&mesh, &nodes, |x, _| 1.0 + x);
        let (target, targets) = points(&[[0.5, 0.5]]);
        assert!(project(&NodeField::default(), &mesh, &target, 1e-6)
            .unwrap()
            .is_empty());
        assert!(project(&field, &mesh, &Mesh::empty(), 1e-6)
            .unwrap()
            .is_empty());
        let carried = project(&field, &Mesh::empty(), &target, 1e-6).unwrap();
        let z = carried.get(0).unwrap().read();
        assert_eq!(z.value(targets[0].id(), "a").unwrap(), 0.0);
        assert_eq!(z.value(targets[0].id(), "b").unwrap(), 0.0);
    }

    /// The field must hang from the `Coords` of `mesh`, and `target` live in
    /// the same dimension.
    #[test]
    fn mismatches_are_refused() {
        let (mesh, nodes) = square(1);
        let field = sampled(&mesh, &nodes, |x, _| x);
        let (other, _) = square(1);
        let (target, _) = points(&[[0.5, 0.5]]);
        assert!(project(&field, &other, &target, 1e-6).is_err());

        let coords = Handle::new(Coords::new(3).unwrap());
        let n = Node::create_in(coords, &[0.5, 0.5, 0.0]).unwrap();
        let in_3d = poi1_from_nodes(&[n]).unwrap();
        assert!(project(&field, &mesh, &in_3d, 1e-6).is_err());
    }
}
