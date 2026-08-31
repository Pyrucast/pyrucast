use crate::aggregate::Aggregate;
use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Copy `mesh`, either onto **fresh nodes** placed at the same spots, or onto
/// the very same nodes.
///
/// Both modes give back a mesh with the same submesh order, element types,
/// cell order and face colours as `mesh`, and **never sealed** — the copy is
/// editable even when the source's submeshes have been frozen by a consumer.
/// `mesh` itself is left untouched. What `new_nodes` decides is whether the
/// two meshes still share their geometry:
///
/// - `true` — one **fresh node per distinct source node**, created in the same
///   `Coords` at the same position. Nodes shared between cells of the source
///   stay shared in the copy. The two meshes are then fully independent: moving
///   a node of one leaves the other where it was. This is what
///   [`translate`](fn@super::translate) and the other rigid copies do, minus
///   the displacement.
/// - `false` — the **same** nodes, their refcount bumped once per occurrence;
///   only the connectivity is copied. Moving a node moves it in both meshes at
///   once. This is the free-function form of [`Mesh::duplicate`], the escape
///   hatch out of the seal.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// let barre = mesh::line(&p(&[0.0, 0.0]), &p(&[2.0, 0.0]), 2, ElementType::SEG2)?;
/// // Des nœuds **neufs**, aux mêmes endroits : les deux maillages ne se
/// // tiennent plus par la géométrie.
/// let neuve = mesh::copy(&barre, true)?;
/// assert_ne!(neuve.node(0, 0, 0)?.id(), barre.node(0, 0, 0)?.id());
/// assert_eq!(neuve.node(0, 0, 0)?.position()?, barre.node(0, 0, 0)?.position()?);
/// // La même connectivité sur les **mêmes** nœuds : un calque, pas un double.
/// let calque = mesh::copy(&barre, false)?;
/// assert_eq!(calque.node(0, 0, 0)?.id(), barre.node(0, 0, 0)?.id());
/// assert_eq!(calque.cell_count(), barre.cell_count());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn copy(mesh: &Mesh, new_nodes: bool) -> Result<Mesh> {
    if mesh.is_empty() {
        // Nothing to copy — and nothing to copy it *into* either: an empty
        // mesh has no `Coords` to create the fresh nodes in. Both modes agree
        // on the answer, so it is settled here rather than in one branch.
        return Ok(Mesh::empty());
    }
    if new_nodes {
        // The identity is not orientation-reversing: nothing to re-order.
        super::transform::map_coords(mesh, false, |position| Ok(position.to_vec()))
    } else {
        mesh.duplicate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node, RgbColor};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Two triangles sharing the edge (b, c) — 4 distinct nodes, 6 slots.
    fn two_triangles() -> (Handle<Coords>, Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        mesh.add_cell(&[n[1].id(), n[3].id(), n[2].id()]).unwrap();
        (coords, mesh, n)
    }

    #[test]
    fn fresh_nodes_land_at_the_same_places_and_keep_the_sharing() {
        let (_coords, mesh, n) = two_triangles();
        let copie = copy(&mesh, true).unwrap();

        assert_eq!(copie.cell_count(), 2);
        // Every node is new…
        for cell in 0..2 {
            for i in 0..3 {
                let old = mesh.node(0, cell, i).unwrap();
                let new = copie.node(0, cell, i).unwrap();
                assert_ne!(new.id(), old.id());
                assert_eq!(new.position().unwrap(), old.position().unwrap());
            }
        }
        // …but the two cells still share the edge (b, c): 4 distinct nodes,
        // not 6.
        assert_eq!(copie.to_poi1().unwrap().cell_count(), 4);
        // The source keeps its own nodes, unmoved.
        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), n[0].id());
    }

    #[test]
    fn shared_nodes_are_the_very_same_ones() {
        let (coords, mesh, n) = two_triangles();
        let before = coords.read().refcount(n[1].id());

        let copie = copy(&mesh, false).unwrap();
        for cell in 0..2 {
            for i in 0..3 {
                assert_eq!(
                    copie.node(0, cell, i).unwrap().id(),
                    mesh.node(0, cell, i).unwrap().id()
                );
            }
        }
        // `b` appears in both cells: the copy increfs it once per occurrence.
        assert_eq!(coords.read().refcount(n[1].id()), before + 2);
        drop(copie);
        assert_eq!(coords.read().refcount(n[1].id()), before);
    }

    #[test]
    fn both_modes_hand_back_an_unsealed_mesh() {
        let (_coords, mesh, n) = two_triangles();
        FiniteElementSpace::lagrange1(&mesh).unwrap(); // scelle la zone
        assert!(mesh.get(0).unwrap().read().is_sealed());

        for new_nodes in [true, false] {
            let mut copie = copy(&mesh, new_nodes).unwrap();
            assert!(!copie.get(0).unwrap().read().is_sealed());
            copie.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
            assert_eq!(copie.cell_count(), 3);
            assert_eq!(mesh.cell_count(), 2, "l'original n'a pas bougé");
        }
    }

    #[test]
    fn keeps_the_submesh_order_the_types_and_the_colours() {
        let (coords, mut mesh, n) = two_triangles();
        mesh.get(0)
            .unwrap()
            .write()
            .set_face_color(RgbColor::new(220, 60, 60));
        let seg = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
            sm.set_face_color(RgbColor::new(60, 60, 220));
            Handle::new(sm)
        };
        mesh.add_sub(seg).unwrap();

        for new_nodes in [true, false] {
            let copie = copy(&mesh, new_nodes).unwrap();
            assert_eq!(copie.len(), 2);
            assert_eq!(
                copie.element_types().unwrap(),
                vec![ElementType::TRI3, ElementType::SEG2]
            );
            assert_eq!(copie.cell_counts().unwrap(), vec![2, 1]);
            assert_eq!(
                copie.get(0).unwrap().read().face_color(),
                RgbColor::new(220, 60, 60)
            );
            assert_eq!(
                copie.get(1).unwrap().read().face_color(),
                RgbColor::new(60, 60, 220)
            );
        }
    }

    #[test]
    fn empty_mesh_copies_to_an_empty_mesh() {
        for new_nodes in [true, false] {
            assert_eq!(copy(&Mesh::empty(), new_nodes).unwrap().len(), 0);
        }
    }
}
