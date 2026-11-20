use crate::aggregate::Aggregate;
use crate::containers::field::{Field, SubField, ABSENT_COMPONENT};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;
use crate::handle::Handle;

/// Restrict `field` to the nodes of `mesh`, keeping the components found there.
///
/// Returns a new [`NodeField`] with at most one zone per submesh of `mesh`,
/// each submesh once: one listed twice gives one zone. Each zone is supported
/// on the submesh's **POI1 companion** — its distinct nodes, via
/// [`SubMesh::to_poi1`](crate::containers::mesh::SubMesh::to_poi1), which is
/// materialised once and cached. Two restrictions onto the **same** `mesh`
/// therefore land on the **same support slot** and combine directly by the
/// arithmetic operators: `restrict(a, mesh) - restrict(b, mesh)` is the
/// node-by-node difference. That cached support is also the one a stiffness
/// block over `mesh` uses, so `&K * &restrict(f, mesh)` and
/// `solve(K, f) - restrict(g, mesh)` line up too.
///
/// A zone takes the components of the zones of `field` that hold at least one
/// of its nodes, in the order of `field`: a zone of `field` sharing no node
/// with the submesh adds nothing to it, and a submesh sharing no node with
/// `field` gives no zone. The test is on the nodes, not on the values: a
/// component that is `0.0` everywhere stays. Values are read through the
/// aggregate (first zone of `field` defining the pair wins); a node of `mesh`
/// a zone of `field` does not cover is assigned `0.0` for its components; a
/// node of `field` absent from `mesh` is dropped.
///
/// Element operations on the region — [`gradient`](fn@crate::ops::element_field::gradient),
/// [`integral`](fn@crate::ops::measure::integral),
/// [`deformation`](fn@crate::ops::element_field::deformation),
/// [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss) — take the mesh /
/// finite-element space as a **separate argument** and read the field by node
/// id. To land on the supports of an existing field `a`, every one of them and
/// with its components, take [`restrict_like`]: `restrict(field, &a.support())`
/// keeps the components of `field`, and leaves out a support sharing no node
/// with it.
///
/// An empty field carries no component: it gives an empty field, as with
/// [`restrict_like`]. A `mesh` on another `Coords` shares no node with `field`
/// — a node id names a node only in its own `Coords` — and gives an empty field
/// too.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // The field brought back to a mesh's nodes — a border, a selection.
/// let bout = mesh::poi1_from_nodes(&n[..1])?;
/// let r = node_field::restrict(&temp, &bout)?;
/// assert_eq!(r.node_count(), 1);
/// assert_eq!(r.get(0)?.read().value(n[0].id(), "T")?, 10.0);
/// // Onto nodes it does not hold, a field gives nothing.
/// assert!(node_field::restrict(&r, &mesh::poi1_from_nodes(&n[1..])?)?.is_empty());
/// assert!(node_field::restrict(&NodeField::empty(), &bout)?.is_empty());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn restrict(field: &NodeField, mesh: &Mesh) -> Result<NodeField> {
    if field.is_empty() {
        return Ok(NodeField::default());
    }
    let coords = field.coords()?;
    let components = Field::components(field);
    let view = field.view();
    let all = view.resolve_reads(&components);
    let mut out = NodeField::default();
    for sm in &mesh.deduplicate() {
        // Another `Coords` first: no node in common, nothing to search.
        if !sm.read().coords().same_object(&coords) {
            continue;
        }
        // The support `SubNodeField::from_support` takes: the zone's POI1
        // companion.
        let support = SubMesh::to_poi1(sm);
        let overlaps = view.overlaps(&support, &all);
        // The components of the zones holding one of its nodes, in the order
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
        let mut sub = SubNodeField::from_poi1(&support, kept)?;
        let reads = view.resolve_reads(sub.components());
        view.gather_rows(&overlaps, &reads, 1, sub.values_mut());
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

/// Read `field` on the supports **and the components** of `target`.
///
/// Zone by zone of `target`, `field` is read as [`restrict`] reads it, on the
/// zone's support and with the zone's components: those `target` lacks are
/// dropped, those no zone of `field` gives at its nodes are created at `0.0`.
/// Every zone of `target` gets its zone, where [`restrict`]`(field,
/// &target.support())` keeps the components of `field` and leaves out a
/// support sharing no node with it.
///
/// Taking the components of `target` too is what lets the result combine with
/// `target` by the arithmetic operators (`&target + &field.restrict_like(target)`),
/// whatever its zones. A field from a solve has one zone per block, the zones
/// meeting at shared nodes with a part of the components each: with the
/// components of `field`, a component would land in a zone where `target` lacks
/// it, pass through unchanged there while another zone sums it at the same
/// node, and the operation would refuse the two values.
///
/// The typical use is folding a solver increment back into a running solution:
/// a `solve` result carries the primal *and* dual (Lagrange multiplier) DOFs on
/// a fresh support, whereas the running field carries only the primal DOFs —
/// `restrict_like` keeps exactly the latter, on the running field's own support.
///
/// An empty `field` gives an empty field, as with [`restrict`]; so does an
/// empty `target`. A `target` on another `Coords` shares no node with `field`:
/// its zones are all `0.0`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // The same thing targeting **another field's** support: the result pairs
/// // with it under `same_support`, which is what a term-by-term operation
/// // requires.
/// # let bout = mesh::poi1_from_nodes(&n[..2])?;
/// let cible = NodeField::from_submesh(&bout.get(0)?, vec!["q".into()])?;
/// let r = node_field::restrict_like(&temp, &cible)?;
/// assert!(r.get(0)?.read().same_support(&cible.get(0)?.read()));
/// // The components are those of `cible`: `q`, which `temp` lacks, is 0.
/// assert_eq!(Field::components(&r), vec!["q"]);
/// assert_eq!(r.value(n[0].id(), "q")?, 0.0);
/// // `restrict` onto the same supports keeps those of `temp`.
/// let sur = node_field::restrict(&temp, &cible.support())?;
/// assert!(sur.get(0)?.read().same_support(&cible.get(0)?.read()));
/// assert_eq!(Field::components(&sur), vec!["T"]);
/// // A field carrying none of the components is empty, and so is what it
/// // gives — on the arithmetic operators, an empty field changes nothing.
/// let rien = node_field::restrict_like(&Field::filter_components(&temp, "q"), &cible)?;
/// assert!(rien.is_empty());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn restrict_like(field: &NodeField, target: &NodeField) -> Result<NodeField> {
    if field.is_empty() {
        return Ok(NodeField::default());
    }
    let coords = field.coords()?;
    let view = field.view();
    let mut out = NodeField::default();
    for h in target.iter() {
        let zone = h.read();
        // `from_support` on the zone's own support handle shares its slot (POI1
        // supports are shared as-is), so the output pairs with `target` under
        // `same_support` — the precondition of the field operators.
        let mut sub = SubNodeField::from_support(&zone.support(), zone.components().to_vec())?;
        // Another `Coords` first: no node in common, the zone stays at 0.
        if sub.coords().same_object(&coords) {
            let reads = view.resolve_reads(sub.components());
            let overlaps = view.overlaps(&sub.support(), &reads);
            view.gather_rows(&overlaps, &reads, 1, sub.values_mut());
        }
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Build a single-zone POI1 field on `n` fresh 1-D nodes;
    /// returns (coords, nodes, field).
    fn poi1_field(
        n: usize,
        components: Vec<String>,
    ) -> (crate::handle::Handle<Coords>, Vec<Node>, NodeField) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in &nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        let field =
            NodeField::from_sub(SubNodeField::from_poi1(&Handle::new(sm), components).unwrap());
        (coords, nodes, field)
    }

    #[test]
    fn restrict_subset() {
        let (coords, nodes, f) = poi1_field(3, vec!["T".into(), "P".into()]);
        {
            let mut s = f.get(0).unwrap().write();
            s.set(0, 0, 1.0).unwrap();
            s.set(1, 0, 2.0).unwrap();
            s.set(2, 0, 3.0).unwrap();
        }

        // Mesh with only nodes[0] and nodes[2].
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nodes[2].id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.node_count(), 2);
        assert_eq!(Field::components(&r), vec!["T", "P"]);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 3.0);
        assert_eq!(r.value(nodes[0].id(), "P").unwrap(), 0.0); // absent → 0
        assert_eq!(r.value_opt(nodes[1].id(), "T"), None); // dropped
    }

    #[test]
    fn restrict_node_absent_from_field_gives_zero() {
        let (coords, nodes, f) = poi1_field(1, vec!["T".into()]);
        f.get(0).unwrap().write().set(0, 0, 7.0).unwrap();
        let nb = Node::create_in(coords.clone(), &[1.0]).unwrap();

        // Mesh contains nb which is NOT in the field.
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nb.id()]).unwrap();

        let r = restrict(&f, &m).unwrap();
        assert_eq!(r.node_count(), 2);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 7.0);
        assert_eq!(r.value(nb.id(), "T").unwrap(), 0.0);
    }

    /// Two zones meeting at node `b` with different components, as a solve's
    /// displacement and multiplier do: each component comes from the first zone
    /// that has it at `b`, the answer `value` gives.
    #[test]
    fn restrict_reads_each_component_from_the_first_zone_having_it() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let poi1 = |nodes: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for nd in nodes {
                sm.add_cell(&[nd.id()]).unwrap();
            }
            Handle::new(sm)
        };
        let mut u = SubNodeField::from_poi1(&poi1(&[&a, &b]), vec!["u_x".into()]).unwrap();
        u.set_value(a.id(), "u_x", 0.3).unwrap();
        u.set_value(b.id(), "u_x", 0.4).unwrap();
        let only_b = poi1(&[&b]);
        let mut lam =
            SubNodeField::from_poi1(&only_b, vec!["lambda".into(), "u_x".into()]).unwrap();
        lam.set_value(b.id(), "lambda", 5.0).unwrap();
        lam.set_value(b.id(), "u_x", 9.0).unwrap(); // behind the first zone's 0.4
        let mut field = NodeField::from_sub(u);
        field.add_sub(Handle::new(lam)).unwrap();

        let mut m = Mesh::empty();
        m.add_sub(only_b).unwrap();
        let r = restrict(&field, &m).unwrap();
        for component in ["lambda", "u_x"] {
            assert_eq!(
                r.value(b.id(), component).unwrap(),
                field.value(b.id(), component).unwrap(),
                "{component}"
            );
        }
        assert_eq!(r.value(b.id(), "lambda").unwrap(), 5.0);
        assert_eq!(r.value(b.id(), "u_x").unwrap(), 0.4);
    }

    #[test]
    fn restrict_onto_a_submesh_listed_twice_gives_one_zone() {
        let (coords, nodes, f) = poi1_field(2, vec!["T".into()]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[nodes[0].id()]).unwrap();
        let sm = Handle::new(sm);
        let mut twice = Mesh::empty();
        twice.add_sub(sm.clone()).unwrap();
        twice.add_sub(sm).unwrap();
        let r = restrict(&f, &twice).unwrap();
        assert_eq!(r.len(), 1, "each submesh once");
        assert_eq!(r.node_count(), 1);
    }

    #[test]
    fn restrict_one_zone_per_mesh_submesh() {
        let (coords, nodes, f) = poi1_field(2, vec!["T".into()]);
        let mut mesh = Mesh::empty();
        for nd in &nodes {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nd.id()]).unwrap();
            mesh.add_sub(Handle::new(sm)).unwrap();
        }
        let r = restrict(&f, &mesh).unwrap();
        assert_eq!(r.len(), 2, "one zone per submesh of the target mesh");
    }

    #[test]
    fn restrict_like_lands_on_target_support_and_components() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap(); // only in source

        // Target: POI1 on {n0, n1}, components [u_x, u_y].
        let mut sm_t = SubMesh::new(coords.clone(), ElementType::POI1);
        sm_t.add_cell(&[n0.id()]).unwrap();
        sm_t.add_cell(&[n1.id()]).unwrap();
        let target = NodeField::from_sub(
            SubNodeField::from_poi1(&Handle::new(sm_t), vec!["u_x".into(), "u_y".into()]).unwrap(),
        );

        // Source (`du`-like): superset nodes, extra `lambda` component.
        let mut sm_s = SubMesh::new(coords.clone(), ElementType::POI1);
        for n in [&n0, &n1, &n2] {
            sm_s.add_cell(&[n.id()]).unwrap();
        }
        let mut ss = SubNodeField::from_poi1(
            &Handle::new(sm_s),
            vec!["u_x".into(), "u_y".into(), "lambda".into()],
        )
        .unwrap();
        ss.set_value(n0.id(), "u_x", 1.0).unwrap();
        ss.set_value(n1.id(), "u_y", 2.0).unwrap();
        ss.set_value(n2.id(), "u_x", 9.0).unwrap(); // node dropped
        ss.set_value(n0.id(), "lambda", 7.0).unwrap(); // component dropped
        let source = NodeField::from_sub(ss);

        let out = restrict_like(&source, &target).unwrap();
        assert_eq!(Field::components(&out), vec!["u_x", "u_y"]);
        assert_eq!(out.node_count(), 2); // n2 dropped
        assert_eq!(out.value(n0.id(), "u_x").unwrap(), 1.0);
        assert_eq!(out.value(n1.id(), "u_y").unwrap(), 2.0);
        assert_eq!(out.value(n1.id(), "u_x").unwrap(), 0.0); // absent → 0

        // Lands on `target`'s own support ⇒ the arithmetic operator applies.
        let sum = (&target + &out).unwrap();
        assert_eq!(sum.value(n0.id(), "u_x").unwrap(), 1.0);
        assert_eq!(sum.value(n1.id(), "u_y").unwrap(), 2.0);
    }

    /// Node ids are numbered per `Coords`, so the first node of each `Coords`
    /// has the same id: comparing them would read 7 where nothing is shared.
    #[test]
    fn restrict_like_onto_another_coords_gives_zero() {
        let (_c1, _n1, f) = poi1_field(1, vec!["T".into()]);
        f.get(0).unwrap().write().set(0, 0, 7.0).unwrap();
        let (_c2, n2, target) = poi1_field(1, vec!["T".into()]);
        let r = restrict_like(&f, &target).unwrap();
        assert!(r
            .get(0)
            .unwrap()
            .read()
            .same_support(&target.get(0).unwrap().read()));
        assert_eq!(r.value(n2[0].id(), "T").unwrap(), 0.0);
    }

    #[test]
    fn restrict_onto_another_coords_gives_nothing() {
        let (_cfg1, nodes1, f) = poi1_field(1, vec!["T".into()]);
        f.get(0).unwrap().write().set(0, 0, 7.0).unwrap();
        let cfg2 = Handle::new(Coords::new(1).unwrap());
        let n2 = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        assert_eq!(n2.id(), nodes1[0].id(), "the same id, in two Coords");
        let mut m2 = Mesh::from_submesh(SubMesh::new(cfg2.clone(), ElementType::POI1));
        m2.add_cell(&[n2.id()]).unwrap();
        assert!(restrict(&f, &m2).unwrap().is_empty());
    }

    /// `T` on `a`, `p` on `b`, no node in common: a submesh of the target takes
    /// the components of the zones holding one of its nodes, and one holding
    /// none of the nodes of the field gives no zone.
    #[test]
    fn restrict_takes_the_components_of_the_zones_sharing_a_node() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n: Vec<Node> = (0..5)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let poi1 = |nodes: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for nd in nodes {
                sm.add_cell(&[nd.id()]).unwrap();
            }
            Handle::new(sm)
        };
        let (a, b) = (poi1(&[&n[0], &n[1]]), poi1(&[&n[2], &n[3]]));
        let mut t = SubNodeField::from_poi1(&a, vec!["T".into()]).unwrap();
        t.set_value(n[1].id(), "T", 3.0).unwrap();
        let mut p = SubNodeField::from_poi1(&b, vec!["p".into()]).unwrap();
        p.set_value(n[2].id(), "p", 4.0).unwrap();
        let mut field = NodeField::from_sub(t);
        field.add_sub(Handle::new(p)).unwrap();

        // `a`, `b`, one straddling both, and one on a node of neither.
        let mut m = Mesh::empty();
        for sm in [a, b, poi1(&[&n[1], &n[2]]), poi1(&[&n[4]])] {
            m.add_sub(sm).unwrap();
        }
        let r = restrict(&field, &m).unwrap();
        let components: Vec<Vec<String>> =
            r.iter().map(|z| z.read().components().to_vec()).collect();
        assert_eq!(components, [vec!["T"], vec!["p"], vec!["T", "p"]]);
        assert_eq!(r.get(2).unwrap().read().value(n[1].id(), "T").unwrap(), 3.0);
        assert_eq!(r.get(2).unwrap().read().value(n[2].id(), "p").unwrap(), 4.0);
        // Present, not read: `p` at a node of `a` is 0.
        assert_eq!(r.get(2).unwrap().read().value(n[1].id(), "p").unwrap(), 0.0);
    }

    /// Two zones on one support, with a component each: the table of that
    /// support is built once, and both read through it.
    #[test]
    fn two_zones_on_one_support_read_through_one_table() {
        let (coords, nodes, t) = poi1_field(3, vec!["T".into()]);
        let support = t.get(0).unwrap().read().support();
        let mut p = SubNodeField::from_poi1(&support, vec!["p".into()]).unwrap();
        for (i, nd) in nodes.iter().enumerate() {
            t.get(0).unwrap().write().set(i, 0, i as f64).unwrap();
            p.set_value(nd.id(), "p", 10.0 + i as f64).unwrap();
        }
        let mut field = t;
        field.add_sub(Handle::new(p)).unwrap();
        assert_eq!(field.len(), 2);

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[nodes[2].id()]).unwrap();
        m.add_cell(&[nodes[0].id()]).unwrap();
        let r = restrict(&field, &m).unwrap();
        assert_eq!(Field::components(&r), vec!["T", "p"]);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 2.0);
        assert_eq!(r.value(nodes[0].id(), "p").unwrap(), 10.0);
        assert_eq!(r.value(nodes[2].id(), "p").unwrap(), 12.0);
    }

    /// Two restrictions onto the **same element mesh** land on its cached POI1
    /// companion (one shared slot), so they subtract node-by-node instead of
    /// passing through as two disjoint zones.
    #[test]
    fn restrict_twice_to_element_mesh_shares_support_and_subtracts() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        // A genuine element mesh (TRI3), not a POI1 zone.
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Two source node fields carrying "v" on the same nodes.
        let mk = |va: f64, vb: f64, vc: f64| {
            let mut psm = SubMesh::new(coords.clone(), ElementType::POI1);
            for nd in [&a, &b, &c] {
                psm.add_cell(&[nd.id()]).unwrap();
            }
            let mut f = SubNodeField::from_poi1(&Handle::new(psm), vec!["v".into()]).unwrap();
            f.set_value(a.id(), "v", va).unwrap();
            f.set_value(b.id(), "v", vb).unwrap();
            f.set_value(c.id(), "v", vc).unwrap();
            NodeField::from_sub(f)
        };
        let a2 = restrict(&mk(1.0, 2.0, 3.0), &mesh).unwrap();
        let b2 = restrict(&mk(0.5, 0.5, 0.5), &mesh).unwrap();

        // Same canonical support slot ⇒ the operators pair the zones.
        let sa = a2.get(0).unwrap().read().support();
        let sb = b2.get(0).unwrap().read().support();
        assert!(
            sa.same_object(&sb),
            "both restricts share the cached POI1 support"
        );

        let diff = (&a2 - &b2).unwrap();
        assert_eq!(diff.len(), 1, "one fused zone, not two pass-through zones");
        assert_eq!(diff.value(a.id(), "v").unwrap(), 0.5);
        assert_eq!(diff.value(b.id(), "v").unwrap(), 1.5);
        assert_eq!(diff.value(c.id(), "v").unwrap(), 2.5);
    }

    /// Regression: restricting a field whose support **is** the mesh's cached
    /// POI1 companion must not deadlock. `restrict` holds a `view` (read guard)
    /// over that support across the loop, while `from_poi1` re-`seal`s it — and
    /// `seal`'s already-sealed read fast-path is what stops that from
    /// write-locking against the held reader. Before the fix this hung.
    #[test]
    fn restrict_onto_own_cached_companion_does_not_deadlock() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let mut psm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in [&a, &b, &c] {
            psm.add_cell(&[nd.id()]).unwrap();
        }
        let mut f = SubNodeField::from_poi1(&Handle::new(psm), vec!["v".into()]).unwrap();
        f.set_value(a.id(), "v", 5.0).unwrap();
        let base = NodeField::from_sub(f);

        // `on_companion` lives on the mesh's cached POI1 companion.
        let on_companion = restrict(&base, &mesh).unwrap();
        // Restricting it onto the same mesh reuses that very support slot.
        let again = restrict(&on_companion, &mesh).unwrap();
        assert_eq!(again.value(a.id(), "v").unwrap(), 5.0);

        let s1 = on_companion.get(0).unwrap().read().support();
        let s2 = again.get(0).unwrap().read().support();
        assert!(
            s1.same_object(&s2),
            "both land on the shared cached companion"
        );
    }
}
