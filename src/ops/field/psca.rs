//! Node-by-node scalar product of two fields — Cast3M `PSCA`.
//!
//! A **free function and not a method**, unlike most of what a field can do:
//! it takes two containers as peers, which the rule files under `ops`. It is
//! also symmetric — `psca(a, b)` and `psca(b, a)` are the same field — so it
//! carries no method form either, for the same reason `merge` does not.
//!
//! Generic over the four field flavours through
//! [`crate::containers::field::Pscal`], exactly as the element-wise
//! maths are generic through `MapValues`.

use crate::containers::field::Pscal;
use crate::error::Result;

/// Node-by-node (or point-by-point) scalar product: a **new field** of the
/// same flavour, carrying a single `"psca"` component whose value at each
/// node/point is `∑_c x_c·y_c` — a reduction over components only, the support
/// is kept.
///
/// `x` and `y` must sit on the same support/decomposition and carry the same
/// components, aligned by name. For the **global** scalar product (one float
/// over the whole field), see `xty`.
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
/// // Le produit scalaire **composante par composante**, sommé : un champ
/// // à une composante, nommée `psca`.
/// let a = champ(vec!["u_x".into(), "u_y".into()]);
/// a.get(0)?.write().add_to_component("u_x", 3.0)?;
/// a.get(0)?.write().add_to_component("u_y", 4.0)?;
/// let p = field::psca(&a, &a)?;
/// assert_eq!(p.get(0)?.read().value(n[0].id(), "psca")?, 25.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn psca<T: Pscal>(x: &T, y: &T) -> Result<T> {
    x.pscal_with(y)
}
