use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::Result;
use crate::handle::Handle;

/// Create a POI1 mesh containing all live nodes of `coords`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// // Tous les nœuds **vivants** d'un repère, en un POI1 — la porte de
/// // sortie quand on a construit de la géométrie sans maillage.
/// let _a = p(&[0.0, 0.0, 0.0]);
/// let _b = p(&[1.0, 0.0, 0.0]);
/// assert_eq!(mesh::from_live_nodes(coords.clone())?.cell_count()?, 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn from_live_nodes(coords: Handle<Coords>) -> Result<Mesh> {
    let node_ids: Vec<_> = coords.read().iter_live().collect();
    Ok(Mesh::from_submesh(SubMesh::poi1_from_node_ids(
        coords, &node_ids,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;

    #[test]
    fn mesh_from_live_nodes() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let _a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let _b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let _c = Node::create_in(coords.clone(), &[2.0]).unwrap();

        let m = from_live_nodes(coords.clone()).unwrap();
        assert_eq!(m.element_types().unwrap(), vec![ElementType::POI1]);
        assert_eq!(m.cell_count().unwrap(), 3);

        // from_live_nodes is a snapshot: mesh m holds the refs, so a
        // second call on the same configuration yields the same result.
        let m2 = from_live_nodes(coords).unwrap();
        assert_eq!(m2.cell_count().unwrap(), 3);
    }
}
