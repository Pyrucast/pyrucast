//! A one-value-per-cell field gathered at the nodes by **summation**: each
//! node receives the sum of the values of the cells around it (Cast3M `CHAN`
//! `'CHPO'` with `'SOMM'`).
//!
//! Not a smoothing: no average, no weight — a node shared by four cells gets
//! four contributions. Applied to the cell measures of
//! [`cell_integral`](fn@crate::ops::element_field::cell_integral), it gives the
//! volume each node stands for, the weight an incremental solver uses to put
//! residuals at comparable scales.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;

/// The nodal sum of a field carrying **one point per cell**.
///
/// `field` lives on a space with a single point per cell (the `Reduced`
/// quadrature, as produced by
/// [`cell_integral`](fn@crate::ops::element_field::cell_integral)); every node
/// receives, per component, the sum of the values of the cells it belongs to —
/// across zones too, since two submeshes meeting at a node both surround it.
/// The components keep their names.
///
/// Errors when a zone has more than one point per cell: which of them a node
/// should receive would be anybody's guess.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node, QuadratureRule};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::node_field;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # sm.add_cell(&[n[0].id(), n[2].id(), n[3].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// let centres = FiniteElementSpace::with(&maillage, &[(Interpolation::Lagrange1, QuadratureRule::Reduced)])?;
/// let v = ElementField::new(&centres, vec!["v".into()])?;
/// v.get(0)?.write().set_uniform("v", 1.0)?;
/// let somme = node_field::sum_to_nodes(&v)?;
/// // The diagonal's ends belong to both triangles, the other corners to one.
/// assert_eq!(somme.value(n[0].id(), "v")?, 2.0);
/// assert_eq!(somme.value(n[1].id(), "v")?, 1.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn sum_to_nodes(field: &ElementField) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for zone in field {
        let (fespace, components, n_gauss) = {
            let z = zone.read();
            (z.support(), z.components().to_vec(), z.gauss_count())
        };
        if n_gauss != 1 {
            return Err(PyrucastError::Message(format!(
                "sum_to_nodes: the field must carry one point per cell — this zone has {n_gauss}; \
                 build it on a space with the REDUCED quadrature (see cell_integral)"
            )));
        }
        let support = fespace.read().submesh().read().to_poi1()?;
        let n = components.len();
        let guard = zone.read();
        let values = guard.values();
        let summed = kernel::scatter_to_nodes(
            std::slice::from_ref(&fespace),
            &support,
            components,
            |geoms, fe| {
                let row = &values[geoms[0].cell * n..(geoms[0].cell + 1) * n];
                for node in fe.chunks_mut(n) {
                    node.copy_from_slice(row);
                }
                Ok(())
            },
        )?;
        drop(guard);
        out.add_sub(Handle::new(summed))?;
    }
    // Two zones meeting at a node both surround it: added, not side by side.
    out.summed()
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation, Node, NodeId, QuadratureRule};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::ops::element_field::cell_integral;

    /// An `nx × ny` grid of unit QUA4, split in two zones (left and right
    /// halves) to exercise the sum across zones. Returns the mesh and the grid.
    fn grid(nx: usize, ny: usize) -> (Mesh, Vec<NodeId>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let mut ids = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                ids.push(
                    Node::create_in(coords.clone(), &[i as f64, j as f64])
                        .unwrap()
                        .id(),
                );
            }
        }
        let at = |i: usize, j: usize| ids[j * (nx + 1) + i];
        let mut left = SubMesh::new(coords.clone(), ElementType::QUA4);
        let mut right = SubMesh::new(coords, ElementType::QUA4);
        for j in 0..ny {
            for i in 0..nx {
                let cell = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                if i < nx / 2 {
                    left.add_cell(&cell).unwrap();
                } else {
                    right.add_cell(&cell).unwrap();
                }
            }
        }
        let mut mesh = Mesh::from_submesh(left);
        mesh.add_sub(Handle::new(right)).unwrap();
        (mesh, ids)
    }

    /// On a regular grid of unit cells, an interior node — even one on the
    /// seam between the two zones — stands for four cells, an edge node for
    /// two, a corner for one.
    #[test]
    fn grid_nodes_count_their_cells() {
        let (mesh, ids) = grid(4, 3);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let centres = FiniteElementSpace::with(
            &mesh,
            &[
                (Interpolation::Lagrange1, QuadratureRule::Reduced),
                (Interpolation::Lagrange1, QuadratureRule::Reduced),
            ],
        )
        .unwrap();
        let un = ElementField::new(&fes, vec!["un".into()]).unwrap();
        for z in &un {
            z.write().set_uniform("un", 1.0).unwrap();
        }
        let volumes = sum_to_nodes(&cell_integral(&un, &centres).unwrap()).unwrap();
        let v = volumes.view().unwrap();
        let at = |i: usize, j: usize| ids[j * 5 + i];
        assert!((v.value(at(2, 1), "un").unwrap() - 4.0).abs() < 1e-12); // on the seam
        assert!((v.value(at(1, 2), "un").unwrap() - 4.0).abs() < 1e-12);
        assert!((v.value(at(2, 0), "un").unwrap() - 2.0).abs() < 1e-12);
        assert!((v.value(at(0, 0), "un").unwrap() - 1.0).abs() < 1e-12);
        assert!((v.value(at(4, 3), "un").unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn several_points_per_cell_is_refused() {
        let (mesh, _) = grid(2, 1);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let f = ElementField::new(&fes, vec!["f".into()]).unwrap();
        let err = sum_to_nodes(&f).unwrap_err();
        assert!(format!("{err}").contains("one point per cell"), "{err}");
    }
}
