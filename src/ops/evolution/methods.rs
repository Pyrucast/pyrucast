//! Method forms of [`crate::ops::evolution`] on the containers they read.

use crate::containers::evolution::Evolution;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::Result;

impl NodeField {
    /// The values of this field along `line`, one curve per component — the
    /// method form of [`evolution::along`](fn@crate::ops::evolution::along).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// let coords = Handle::new(Coords::new(1)?);
    /// let n: Vec<Node> = [[0.0], [0.5], [2.0]]
    ///     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// let ligne = mesh::poi1_from_nodes(&n)?;
    /// let u = NodeField::new(&ligne, vec!["u_x".into()])?;
    /// u.get(0)?.write().set_value(n[2].id(), "u_x", 1.0)?;
    /// let curve = u.along(&ligne, &["u_x"])?.get(0)?.read().scalar_series()?;
    /// assert_eq!(curve, vec![(0.0, 0.0), (0.5, 0.0), (2.0, 1.0)]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn along(&self, line: &Mesh, components: &[&str]) -> Result<Evolution> {
        crate::ops::evolution::along(self, line, components)
    }
}
