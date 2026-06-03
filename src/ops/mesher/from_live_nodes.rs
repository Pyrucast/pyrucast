use crate::error::Result;
use crate::containers::mesh::Configuration;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{with, Handle};

/// Create a POI1 mesh containing all live nodes of `config`.
pub fn from_live_nodes(config: Handle<Configuration>) -> Result<Mesh> {
    let node_ids: Vec<_> = with(&config, |c| c.iter_live().collect())?;
    Ok(Mesh::from_submesh(SubMesh::poi1_from_nodes(config, &node_ids)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    #[test]
    fn mesh_from_live_nodes() {
        let cfg = insert(Configuration::new(1).unwrap());
        let _a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let _b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let _c = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        let m = from_live_nodes(cfg.clone()).unwrap();
        assert_eq!(m.element_types().unwrap(), vec![ElementType::POI1]);
        assert_eq!(m.cell_count().unwrap(), 3);

        // from_live_nodes is a snapshot: mesh m holds the refs, so a
        // second call on the same configuration yields the same result.
        let m2 = from_live_nodes(cfg).unwrap();
        assert_eq!(m2.cell_count().unwrap(), 3);
    }
}
