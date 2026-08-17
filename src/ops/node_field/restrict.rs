use crate::aggregate::Aggregate;
use crate::containers::field::{Field, SubField};
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, NodeFieldView, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// Fill `sub` (already on its target support) with the values of `source`,
/// component by component, node by node: a `(node, component)` pair that
/// `source` covers is copied, one it does not is left at `0.0`. Shared by
/// [`restrict`] and [`restrict_like`].
fn fill_from(sub: &mut SubNodeField, source: &NodeFieldView) -> Result<()> {
    let components = sub.components().to_vec();
    for nid in sub.nodes().to_vec() {
        for comp in &components {
            if let Some(v) = source.value_opt(nid, comp) {
                sub.set_value(nid, comp, v)?;
            }
        }
    }
    Ok(())
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new [`NodeField`] with one zone per submesh of `mesh`. Each zone
/// is supported on the submesh's **canonical POI1 node cloud** — its distinct
/// nodes, via [`SubMesh::to_poi1`](crate::containers::mesh::SubMesh::to_poi1),
/// which is materialised once and cached. Two restrictions onto the **same**
/// `mesh` therefore land on the **same support slot** and combine directly by
/// the arithmetic operators: `restrict(a, mesh) - restrict(b, mesh)` is the
/// node-by-node difference. That cached support is also the one a stiffness
/// block over `mesh` uses, so `&K * &restrict(f, mesh)` and
/// `solve(K, f) - restrict(g, mesh)` line up too.
///
/// Each zone carries the union of `field`'s components. Values are read
/// through the aggregate (first zone of `field` defining the pair wins); a
/// node of `mesh` that `field` does not cover is assigned `0.0`; a node of
/// `field` absent from `mesh` is dropped.
///
/// Element operations on the region — [`gradient`](fn@crate::ops::element_field::gradient),
/// [`integral`](fn@crate::ops::measure::integral),
/// [`deformation`](fn@crate::ops::element_field::deformation),
/// [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss) — take the mesh /
/// finite-element space as a **separate argument** and read the field by node
/// id. To land instead on the *exact* support of an **existing field** (rather
/// than a mesh), use [`restrict_like`].
///
/// Errors if `mesh` is attached to a different `Coords` than
/// `field`.
pub fn restrict(field: &NodeField, mesh: &Mesh) -> Result<NodeField> {
    let mesh_coords = mesh.coords()?;
    let field_coords = field.coords()?;
    if !mesh_coords.same_object(&field_coords) {
        return Err(PyrucastError::Message(
            "restrict: mesh is not attached to the same Coords".into(),
        ));
    }

    let components = Field::components(field)?;
    let view = field.view()?;
    let mut out = NodeField::default();
    for sm in mesh {
        let mut sub = SubNodeField::from_support(sm, components.clone())?;
        fill_from(&mut sub, &view)?;
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

/// Reproject `field` onto the **exact support and components of `target`**,
/// zone by zone.
///
/// Unlike [`restrict`] (which lands on a *fresh* support materialised from a
/// mesh, carrying the union of `field`'s components), this reuses each zone of
/// `target` **as-is** — the same support object, the same component list — so the result is
/// on the very same support as `target` and can be combined with it directly by
/// the arithmetic operators (`&target + &field.restrict_like(target)`). Each
/// `(node, component)` pair is filled from `field` when it covers it, `0.0`
/// otherwise; nodes and components of `field` absent from `target` are dropped.
///
/// The typical use is folding a solver increment back into a running solution:
/// a `solve` result carries the primal *and* dual (Lagrange multiplier) DOFs on
/// a fresh support, whereas the running field carries only the primal DOFs —
/// `restrict_like` keeps exactly the latter, on the running field's own support.
///
/// Errors if `target` is attached to a different `Coords` than `field`.
pub fn restrict_like(field: &NodeField, target: &NodeField) -> Result<NodeField> {
    let target_coords = target.coords()?;
    let field_coords = field.coords()?;
    if !target_coords.same_object(&field_coords) {
        return Err(PyrucastError::Message(
            "restrict_like: target is not attached to the same Coords".into(),
        ));
    }

    let view = field.view()?;
    let mut out = NodeField::default();
    for h in target.iter() {
        let zone = h.read();
        // `from_support` on the zone's own support handle shares its slot (POI1
        // supports are shared as-is), so the output pairs with `target` under
        // `same_support` — the precondition of the field operators.
        let mut sub = SubNodeField::from_support(&zone.support(), zone.components().to_vec())?;
        fill_from(&mut sub, &view)?;
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
        assert_eq!(r.node_count().unwrap(), 2);
        assert_eq!(Field::components(&r).unwrap(), vec!["T", "P"]);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 3.0);
        assert_eq!(r.value(nodes[0].id(), "P").unwrap(), 0.0); // absent → 0
        assert_eq!(r.value_opt(nodes[1].id(), "T").unwrap(), None); // dropped
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
        assert_eq!(r.node_count().unwrap(), 2);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 7.0);
        assert_eq!(r.value(nb.id(), "T").unwrap(), 0.0);
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
        assert_eq!(Field::components(&out).unwrap(), vec!["u_x", "u_y"]);
        assert_eq!(out.node_count().unwrap(), 2); // n2 dropped
        assert_eq!(out.value(n0.id(), "u_x").unwrap(), 1.0);
        assert_eq!(out.value(n1.id(), "u_y").unwrap(), 2.0);
        assert_eq!(out.value(n1.id(), "u_x").unwrap(), 0.0); // absent → 0

        // Lands on `target`'s own support ⇒ the arithmetic operator applies.
        let sum = (&target + &out).unwrap();
        assert_eq!(sum.value(n0.id(), "u_x").unwrap(), 1.0);
        assert_eq!(sum.value(n1.id(), "u_y").unwrap(), 2.0);
    }

    #[test]
    fn restrict_like_incompatible_cfg_errors() {
        let (_c1, _n1, f) = poi1_field(1, vec!["T".into()]);
        let (_c2, _n2, target) = poi1_field(1, vec!["T".into()]); // different Coords
        assert!(restrict_like(&f, &target).is_err());
    }

    #[test]
    fn restrict_incompatible_cfg_errors() {
        let (_cfg1, _nodes1, f) = poi1_field(1, vec!["T".into()]);
        // A mesh attached to a *different* Coords.
        let cfg2 = Handle::new(Coords::new(1).unwrap());
        let n2 = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        let mut m2 = Mesh::from_submesh(SubMesh::new(cfg2.clone(), ElementType::POI1));
        m2.add_cell(&[n2.id()]).unwrap();
        assert!(restrict(&f, &m2).is_err());
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

        // A genuine element mesh (TRI3), not a POI1 cloud.
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
