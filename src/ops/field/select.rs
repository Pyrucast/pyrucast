//! Value-range selection — extract, zone by zone, the part of a field's
//! support whose values fall inside a `[lower, upper]` band (see
//! [`Band`](super::band::Band), built from `ge` / `gt` / `le` / `lt`).
//!
//! The two entry points mirror the two field flavours and the two kinds
//! of support they carry:
//!
//! - [`select_nodes`] over a [`NodeField`]: each zone yields a **POI1
//!   `SubMesh`** holding the nodes that pass the test;
//! - [`select_cells`] over an [`ElementField`]: each zone yields a
//!   `SubMesh` of the zone's **own element type** holding the cells that
//!   pass — a cell passes only when **every** Gauss point does (the band
//!   must hold all along the cell).
//!
//! Both return a [`Mesh`] with **one submesh per processed zone**
//! ([[feedback-viz-per-element]] — zones stay separate, nothing merged).
//! The structure is therefore parallel to the input field, minus the
//! zones skipped by the component filter.
//!
//! # Component filter
//!
//! - `components = None` ⇒ test **every** component of each zone;
//! - `components = Some(list)` ⇒ test **only** those components, and only
//!   on the zones that carry **all** of them. A zone missing any listed
//!   component is skipped entirely (it produces no submesh).
//!
//! Within a retained point/cell the bounds are combined with **AND**: the
//! point is kept only if *each* tested component lies in the band.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::mesh::{Mesh, NodeId, SubMesh};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::Result;
use crate::store::{insert, read};

use super::band::Band;

/// Indices, into a zone's component list, of the components to test.
///
/// `None` ⇒ the filter names a component the zone does not carry (callers
/// decide what that means: [`select`](self) skips the zone, [`mask`](super::mask)
/// leaves it identity). With no filter, every component is tested.
pub(crate) fn components_to_test(
    zone: &[String],
    requested: &Option<Vec<String>>,
) -> Option<Vec<usize>> {
    match requested {
        None => Some((0..zone.len()).collect()),
        Some(names) => {
            let mut idx = Vec::with_capacity(names.len());
            for name in names {
                idx.push(zone.iter().position(|c| c == name)?);
            }
            Some(idx)
        }
    }
}

// ─── Per-zone builders ──────────────────────────────────────────────────────

/// POI1 submesh of the nodes of `sub` passing the band on every tested
/// component. `None` when the component filter skips this zone.
fn sub_node_submesh(
    sub: &SubNodeField,
    band: &Band,
    components: &Option<Vec<String>>,
) -> Result<Option<SubMesh>> {
    let comps = SubField::components(sub);
    let test = match components_to_test(comps, components) {
        Some(t) => t,
        None => return Ok(None),
    };
    let ncomp = comps.len();
    let values = sub.values();
    let mut kept: Vec<NodeId> = Vec::new();
    for (ni, &nid) in sub.nodes().iter().enumerate() {
        let row = &values[ni * ncomp..(ni + 1) * ncomp];
        if test.iter().all(|&ci| band.contains(row[ci])) {
            kept.push(nid);
        }
    }
    Ok(Some(SubMesh::poi1_from_node_ids(sub.coords(), &kept)?))
}

/// Submesh (zone's own element type) of the cells of `sub` whose **every**
/// Gauss point passes the band on every tested component. `None` when the
/// component filter skips this zone.
fn sub_element_submesh(
    sub: &SubElementField,
    band: &Band,
    components: &Option<Vec<String>>,
) -> Result<Option<SubMesh>> {
    let comps = SubField::components(sub);
    let test = match components_to_test(comps, components) {
        Some(t) => t,
        None => return Ok(None),
    };
    let ncomp = comps.len();
    let ngauss = sub.gauss_count();
    let ncells = sub.cell_count();
    let values = sub.values();

    // Underlying mesh of the FE subspace — read its connectivity once.
    let smh = read(&SubField::support(sub))?.submesh();
    let smr = read(&smh)?;
    let et = smr.element_type();
    let npc = et.nodes_per_cell();
    let conn = smr.connectivity();
    let mut kept = SubMesh::new(smr.coords(), et);
    for cell in 0..ncells {
        let pass = (0..ngauss).all(|g| {
            let base = (cell * ngauss + g) * ncomp;
            test.iter().all(|&ci| band.contains(values[base + ci]))
        });
        if pass {
            kept.add_cell(&conn[cell * npc..(cell + 1) * npc])?;
        }
    }
    Ok(Some(kept))
}

// ─── Public operators ───────────────────────────────────────────────────────

/// Select the nodes of `field` passing `band`, zone by zone. Returns a
/// [`Mesh`] of POI1 submeshes — one per processed zone.
///
/// See the [module documentation](self) for the component-filter and
/// AND-across-components semantics.
pub fn select_nodes(
    field: &NodeField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<Mesh> {
    let mut out = Mesh::empty();
    for h in field.iter() {
        if let Some(sm) = sub_node_submesh(&*read(h)?, band, &components)? {
            out.add_sub(insert(sm))?;
        }
    }
    Ok(out)
}

/// Select the cells of `field` passing `band`, zone by zone. A cell is kept
/// only when **all** its Gauss points pass. Returns a [`Mesh`] of submeshes
/// (each of its zone's element type) — one per processed zone.
///
/// See the [module documentation](self) for the component-filter and
/// AND semantics.
pub fn select_cells(
    field: &ElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<Mesh> {
    let mut out = Mesh::empty();
    for h in field.iter() {
        if let Some(sm) = sub_element_submesh(&*read(h)?, band, &components)? {
            out.add_sub(insert(sm))?;
        }
    }
    Ok(out)
}

/// Single-zone [`select_nodes`] — a [`Mesh`] with one POI1 submesh, or an
/// empty mesh when the component filter skips the zone.
pub fn select_sub_nodes(
    sub: &SubNodeField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<Mesh> {
    let mut out = Mesh::empty();
    if let Some(sm) = sub_node_submesh(sub, band, &components)? {
        out.add_sub(insert(sm))?;
    }
    Ok(out)
}

/// Single-zone [`select_cells`] — a [`Mesh`] with one submesh, or an empty
/// mesh when the component filter skips the zone.
pub fn select_sub_cells(
    sub: &SubElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<Mesh> {
    let mut out = Mesh::empty();
    if let Some(sm) = sub_element_submesh(sub, band, &components)? {
        out.add_sub(insert(sm))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::store::{insert, read, write};

    /// Single-zone POI1 NodeField over `n` 1-D nodes; returns (nodes, field).
    fn poi1_field(n: usize, components: Vec<String>) -> (Vec<Node>, NodeField) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in &nodes {
            sm.add_cell(&[nd.id()]).unwrap();
        }
        let field = NodeField::from_sub(SubNodeField::from_poi1(&insert(sm), components).unwrap());
        (nodes, field)
    }

    /// Node ids of the (single) zone of a selection mesh, in order.
    fn picked(mesh: &Mesh, zone: usize) -> Vec<NodeId> {
        read(&mesh.get(zone).unwrap())
            .unwrap()
            .connectivity()
            .to_vec()
    }

    #[test]
    fn select_nodes_min_max_band() {
        let (nodes, f) = poi1_field(5, vec!["T".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            for i in 0..5 {
                s.set(i, 0, i as f64 * 10.0).unwrap(); // 0,10,20,30,40
            }
        }
        // 10 <= T <= 30 → nodes 1,2,3.
        let sel = select_nodes(
            &f,
            &Band::new(Some(10.0), None, Some(30.0), None).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(sel.len(), 1);
        let ids = picked(&sel, 0);
        assert_eq!(ids, vec![nodes[1].id(), nodes[2].id(), nodes[3].id()]);
    }

    #[test]
    fn select_nodes_open_bounds() {
        let (nodes, f) = poi1_field(3, vec!["T".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set(0, 0, -1.0).unwrap();
            s.set(1, 0, 5.0).unwrap();
            s.set(2, 0, 9.0).unwrap();
        }
        // Only an upper bound: T <= 5 → nodes 0,1.
        let sel = select_nodes(&f, &Band::new(None, None, Some(5.0), None).unwrap(), None).unwrap();
        assert_eq!(picked(&sel, 0), vec![nodes[0].id(), nodes[1].id()]);
        // Only a lower bound: T >= 5 → nodes 1,2.
        let sel = select_nodes(&f, &Band::new(Some(5.0), None, None, None).unwrap(), None).unwrap();
        assert_eq!(picked(&sel, 0), vec![nodes[1].id(), nodes[2].id()]);
    }

    #[test]
    fn select_nodes_all_components_and_semantics() {
        let (nodes, f) = poi1_field(3, vec!["U".into(), "V".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            // node0: U=1 V=1 ; node1: U=1 V=9 ; node2: U=1 V=2
            s.set(0, 0, 1.0).unwrap();
            s.set(0, 1, 1.0).unwrap();
            s.set(1, 0, 1.0).unwrap();
            s.set(1, 1, 9.0).unwrap();
            s.set(2, 0, 1.0).unwrap();
            s.set(2, 1, 2.0).unwrap();
        }
        // 0 <= * <= 5 on BOTH components → node1 dropped (V=9).
        let sel = select_nodes(
            &f,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(picked(&sel, 0), vec![nodes[0].id(), nodes[2].id()]);
    }

    #[test]
    fn select_nodes_component_filter_restricts_test() {
        let (nodes, f) = poi1_field(3, vec!["U".into(), "V".into()]);
        {
            let mut s = write(&f.get(0).unwrap()).unwrap();
            s.set(0, 0, 1.0).unwrap();
            s.set(0, 1, 1.0).unwrap();
            s.set(1, 0, 1.0).unwrap();
            s.set(1, 1, 9.0).unwrap();
            s.set(2, 0, 1.0).unwrap();
            s.set(2, 1, 2.0).unwrap();
        }
        // Test U only: U=1 everywhere ⇒ all kept despite V out of band.
        let sel = select_nodes(
            &f,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            Some(vec!["U".into()]),
        )
        .unwrap();
        assert_eq!(
            picked(&sel, 0),
            vec![nodes[0].id(), nodes[1].id(), nodes[2].id()]
        );
    }

    #[test]
    fn select_nodes_skips_zone_missing_requested_component() {
        // Two zones: zone0 has "T", zone1 has "P".
        let coords = insert(Coords::new(1).unwrap());
        let n: Vec<Node> = (0..2)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mk = |nid: NodeId, comp: &str| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nid]).unwrap();
            SubNodeField::from_poi1(&insert(sm), vec![comp.into()]).unwrap()
        };
        let f = NodeField::from_sub(mk(n[0].id(), "T"))
            .union(&NodeField::from_sub(mk(n[1].id(), "P")))
            .unwrap();
        assert_eq!(f.len(), 2);
        // Filter on "T": only zone0 is processed → one submesh out.
        let sel = select_nodes(
            &f,
            &Band::new(Some(-1e9), None, Some(1e9), None).unwrap(),
            Some(vec!["T".into()]),
        )
        .unwrap();
        assert_eq!(sel.len(), 1);
        assert_eq!(picked(&sel, 0), vec![n[0].id()]);
    }

    #[test]
    fn select_nodes_empty_selection_keeps_empty_zone() {
        let (_nodes, f) = poi1_field(3, vec!["T".into()]); // all zero
        let sel = select_nodes(
            &f,
            &Band::new(Some(100.0), None, Some(200.0), None).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(sel.len(), 1, "processed zone still yields a submesh");
        assert_eq!(read(&sel.get(0).unwrap()).unwrap().cell_count(), 0);
    }

    /// One TRI3 + one QUA4 zone, lagrange-1; returns the ElementField.
    fn two_zone_element_field() -> (Vec<Node>, ElementField) {
        let coords = insert(Coords::new(2).unwrap());
        #[rustfmt::skip]
        let n: Vec<Node> = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut mesh = Mesh::empty();
        let tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
            insert(sm)
        };
        let qua = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
            sm.add_cell(&[n[1].id(), n[3].id(), n[4].id(), n[2].id()])
                .unwrap();
            insert(sm)
        };
        mesh.add_sub(tri).unwrap();
        mesh.add_sub(qua).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = ElementField::new(&fes, vec!["s".into()]).unwrap();
        (n, ef)
    }

    #[test]
    fn select_cells_all_gauss_must_pass() {
        let (_n, ef) = two_zone_element_field();
        // Zone0 (TRI3): s = 2 everywhere → in band [0,5].
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("s", 2.0)
            .unwrap();
        // Zone1 (QUA4): one Gauss point at 100 → cell fails.
        {
            let mut s = write(&ef.get(1).unwrap()).unwrap();
            s.set_uniform("s", 2.0).unwrap();
            s.set_value(0, 0, "s", 100.0).unwrap();
        }
        let sel = select_cells(
            &ef,
            &Band::new(Some(0.0), None, Some(5.0), None).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(sel.len(), 2, "one submesh per zone");
        // TRI3 cell kept, QUA4 cell dropped.
        assert_eq!(read(&sel.get(0).unwrap()).unwrap().cell_count(), 1);
        assert_eq!(
            read(&sel.get(0).unwrap()).unwrap().element_type(),
            ElementType::TRI3
        );
        assert_eq!(read(&sel.get(1).unwrap()).unwrap().cell_count(), 0);
        assert_eq!(
            read(&sel.get(1).unwrap()).unwrap().element_type(),
            ElementType::QUA4
        );
    }
}
