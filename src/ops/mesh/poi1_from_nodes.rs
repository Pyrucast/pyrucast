use crate::atoms::Node;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;

/// Build a POI1 mesh holding exactly `nodes`, **one cell per node**, in the
/// given order.
///
/// The [`Coords`](crate::coords::Coords) is taken from the nodes themselves
/// — every [`Node`] carries its own — so no store argument is needed. The
/// result has a single submesh. Errors if `nodes` is empty (no `Coords` to
/// attach to).
///
/// The companion of [`from_live_nodes`](super::from_live_nodes()), which takes
/// the whole store instead of a chosen list, and of
/// [`to_poi1`](super::to_poi1()), which takes an existing mesh.
///
/// ```
/// use pyrucast::atoms::Node;
/// use pyrucast::coords::Coords;
/// use pyrucast::handle::Handle;
/// use pyrucast::ops::mesh::poi1_from_nodes;
///
/// let coords = Handle::new(Coords::new(2).unwrap());
/// let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
/// let b = Node::create_in(coords, &[1.0, 0.0]).unwrap();
///
/// let cloud = poi1_from_nodes(&[a, b]).unwrap();
/// assert_eq!(cloud.cell_counts().unwrap(), vec![2]);
/// ```
pub fn poi1_from_nodes(nodes: &[Node]) -> Result<Mesh> {
    Ok(Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::coords::Coords;
    use crate::handle::Handle;

    #[test]
    fn builds_one_cell_per_node() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let m = poi1_from_nodes(&[a, b]).unwrap();

        assert_eq!(m.cell_counts().unwrap(), vec![2]);
        assert_eq!(m.element_types().unwrap(), vec![ElementType::POI1]);
    }

    #[test]
    fn empty_list_errors() {
        assert!(poi1_from_nodes(&[]).is_err());
    }
}
