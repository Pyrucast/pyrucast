//! A field at the Gauss points brought to the **nodes**: the inverse way of
//! [`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss).
//!
//! Each cell gives its own nodes the values its Gauss points fit there — the
//! least-squares fit local to the cell, the one a plot colours the cell with
//! (see [`project_element`](fn@crate::ops::node_field::project_element)). A
//! node shared by several cells then receives what they give it, combined by
//! a [`Reduction`]:
//!
//! - **average** — the value a reader expects at a node: the cells around it
//!   agree on it when the field is continuous, and their mean smooths the
//!   jumps between them otherwise;
//! - **sum** — the gathering of per-cell contributions: applied to the cell
//!   measures of [`cell_integral`](fn@crate::ops::element_field::cell_integral),
//!   the volume each node stands for.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::{add_partials, Field, SubField};
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::models::kernel;
use crate::ops::element_field::fit::CellFit;
use crate::ops::node_field::restrict_like;

/// How [`to_nodes`] combines, at a node, the values the cells around it give.
///
/// ```
/// use pyrucast::named::Named;
/// use pyrucast::ops::node_field::Reduction;
///
/// assert_eq!(Reduction::parse(" Average ")?, Reduction::Average);
/// assert_eq!(Reduction::names(), "average|sum");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// The mean over the cells around the node that carry the component.
    Average,
    /// The sum over the cells around the node.
    Sum,
}

impl crate::named::Named for Reduction {
    const LABEL: &'static str = "reduction";
    const VALUES: &'static [Self] = &[Self::Average, Self::Sum];

    fn name(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Sum => "sum",
        }
    }
}

/// `field` brought to the nodes of its cells, combined by `reduction`.
///
/// Each cell fits its Gauss values over its own nodes (least squares on the
/// cell's shape functions; a cell with fewer Gauss points than nodes gives
/// every node the mean of its points, so a field of one point per cell gives
/// each node the cell's value). A node then receives, per component:
///
/// - [`Reduction::Average`]: the mean of what the cells around it give,
///   counted over the cells **carrying that component** — a node no such cell
///   touches reads 0;
/// - [`Reduction::Sum`]: their sum.
///
/// Cells of several zones meeting at a node all count, and the components keep
/// their names. The Python face defaults `reduction` to `"average"`.
///
/// A field interpolated from the nodes of its space
/// ([`interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss)) comes
/// back exactly with `Average`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::node_field::{self, Reduction};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[4].id(), n[3].id()])?;
/// # sm.add_cell(&[n[1].id(), n[2].id(), n[5].id(), n[4].id()])?;
/// # let squares = Mesh::from_submesh(sm);
/// // Two unit squares side by side, 2×2 Gauss points: s = 1 in the left one,
/// // 5 in the right one.
/// let fes = FiniteElementSpace::lagrange1(&squares)?;
/// let s = ElementField::new(&fes, vec!["s".into()])?;
/// s.get(0)?.write().set_cell_uniform(0, "s", 1.0)?;
/// s.get(0)?.write().set_cell_uniform(1, "s", 5.0)?;
/// // The nodes x = 1 belong to both squares: their mean, then their sum.
/// let mean = node_field::to_nodes(&s, Reduction::Average)?;
/// assert!((mean.value(n[1].id(), "s")? - 3.0).abs() < 1e-12);
/// assert!((mean.value(n[0].id(), "s")? - 1.0).abs() < 1e-12);
/// assert!((mean.value(n[2].id(), "s")? - 5.0).abs() < 1e-12);
/// let sum = node_field::to_nodes(&s, Reduction::Sum)?;
/// assert!((sum.value(n[1].id(), "s")? - 6.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn to_nodes(field: &ElementField, reduction: Reduction) -> Result<NodeField> {
    let mut sums = NodeField::empty();
    let mut counts = NodeField::empty();
    for zone in field {
        let (fespace, components) = {
            let z = zone.read();
            (z.support(), z.components().to_vec())
        };
        let fit = CellFit::for_space(&fespace.read())?;
        let support = SubMesh::to_poi1(&fespace.read().submesh());
        let n_comp = components.len();
        let guard = zone.read();
        let block = guard.gauss_count() * n_comp;
        let values = guard.values();
        let partial = kernel::scatter_to_nodes(
            std::slice::from_ref(&fespace),
            &support,
            components.clone(),
            |geoms, fe| {
                let first = geoms[0].cell * block;
                fit.nodal_values_into(&values[first..first + block], n_comp, fe);
                Ok(())
            },
        )?;
        drop(guard);
        // Two zones on one cell list — a space listing it twice — add up
        // there and keep its support.
        sums = add_partials(&sums, &NodeField::from_sub(partial))?;
        if reduction == Reduction::Average {
            // Each cell counts once at each of its nodes, for the components
            // it carries.
            let ones = kernel::scatter_to_nodes(
                std::slice::from_ref(&fespace),
                &support,
                components,
                |_, fe| {
                    fe.fill(1.0);
                    Ok(())
                },
            )?;
            counts = add_partials(&counts, &NodeField::from_sub(ones))?;
        }
    }
    // Two zones meeting at a node both surround it: added, not side by side.
    let sums = sums.summed()?;
    match reduction {
        Reduction::Sum => Ok(sums),
        // A component no cell around a node carries is 0 there, counted 0
        // times: `max(n, 1)` keeps it 0.
        Reduction::Average => sums
            .merge_field(&restrict_like(&counts.summed()?, &sums)?, |s, n| {
                s / n.max(1.0)
            }),
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation, Node, NodeId, QuadratureRule};
    use crate::containers::element_field::SubElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::element_field::{cell_integral, interp_to_gauss};

    /// An `nx × ny` grid of unit QUA4, split in two zones (left and right
    /// halves) to exercise the gathering across zones. Returns the mesh and the
    /// grid's nodes, row by row.
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

    /// One point per cell, summed: on a regular grid of unit cells, an
    /// interior node — even one on the seam between the two zones — stands
    /// for four cells, an edge node for two, a corner for one.
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
        let volumes = to_nodes(&cell_integral(&un, &centres).unwrap(), Reduction::Sum).unwrap();
        let v = volumes.view();
        let at = |i: usize, j: usize| ids[j * 5 + i];
        assert!((v.value(at(2, 1), "un").unwrap() - 4.0).abs() < 1e-12); // on the seam
        assert!((v.value(at(1, 2), "un").unwrap() - 4.0).abs() < 1e-12);
        assert!((v.value(at(2, 0), "un").unwrap() - 2.0).abs() < 1e-12);
        assert!((v.value(at(0, 0), "un").unwrap() - 1.0).abs() < 1e-12);
        assert!((v.value(at(4, 3), "un").unwrap() - 1.0).abs() < 1e-12);
    }

    /// A bilinear field interpolated at 2×2 Gauss points comes back to its
    /// nodal values at every node, the seam between the two zones included:
    /// each cell fits it exactly, and the cells around a node agree there.
    #[test]
    fn a_bilinear_field_comes_back_exactly() {
        let (mesh, ids) = grid(4, 3);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let coords = mesh.get(0).unwrap().read().coords();
        let mut all = SubMesh::new(coords, ElementType::POI1);
        for &id in &ids {
            all.add_cell(&[id]).unwrap();
        }
        let all = Handle::new(all);
        let f = |i: usize, j: usize| i as f64 + 2.0 * j as f64 + 0.5 * (i * j) as f64;
        let nodal = NodeField::from_submesh(&all, vec!["f".into()]).unwrap();
        {
            let mut z = nodal.get(0).unwrap().write();
            for j in 0..=3 {
                for i in 0..=4 {
                    z.set_value(ids[j * 5 + i], "f", f(i, j)).unwrap();
                }
            }
        }
        let gauss = interp_to_gauss(&nodal, &fes).unwrap();
        let back = to_nodes(&gauss, Reduction::Average).unwrap();
        for j in 0..=3 {
            for i in 0..=4 {
                let v = back.value(ids[j * 5 + i], "f").unwrap();
                assert!((v - f(i, j)).abs() < 1e-12, "({i}, {j}): {v} ≠ {}", f(i, j));
            }
        }
    }

    /// A component only the right zone carries is averaged over the right
    /// cells alone: on the seam it keeps the right cells' value, not half of
    /// it, and a node no right cell touches reads 0, not NaN.
    #[test]
    fn a_component_one_zone_lacks_is_averaged_where_it_lives() {
        let (mesh, ids) = grid(2, 1);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut left = SubElementField::new(fes.get(0).unwrap(), vec!["a".into()]).unwrap();
        left.set_uniform("a", 1.0).unwrap();
        let mut right =
            SubElementField::new(fes.get(1).unwrap(), vec!["a".into(), "b".into()]).unwrap();
        right.set_uniform("a", 3.0).unwrap();
        right.set_uniform("b", 7.0).unwrap();
        let mut field = ElementField::empty();
        field.add_sub(Handle::new(left)).unwrap();
        field.add_sub(Handle::new(right)).unwrap();
        let mean = to_nodes(&field, Reduction::Average).unwrap();
        let (corner, seam, far) = (ids[0], ids[1], ids[2]);
        assert!((mean.value(seam, "a").unwrap() - 2.0).abs() < 1e-12);
        assert!((mean.value(seam, "b").unwrap() - 7.0).abs() < 1e-12);
        assert!((mean.value(far, "b").unwrap() - 7.0).abs() < 1e-12);
        assert_eq!(mean.value(corner, "b").unwrap(), 0.0);
    }
}
