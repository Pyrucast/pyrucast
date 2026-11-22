//! The values a cell's own Gauss points give at its nodes — a least-squares
//! fit **local to the cell**, never averaged with the neighbouring cells.
//!
//! A field at the Gauss points has no value at a node: each cell holds its own
//! values, and two neighbours may disagree on the face they share. Drawing such
//! a field, or reading it at an arbitrary point, needs a value inside the cell.
//! The fit gives one, from that cell's Gauss values alone:
//!
//! ```text
//! min Σ_g w_g (Σ_i N_i(ξ_g) v_i − f_g)²   ⇒   v = M f,   M = (AᵀWA)⁻¹ AᵀW
//! ```
//!
//! with `A[g, i] = N_i(ξ_g)` and `W = diag(w_g)`. `M` depends on the FE space
//! only, so it is computed once for a zone and applied to every cell. A field
//! that is the Gauss interpolation of a nodal field of the space comes back
//! exactly; jumps between cells stay, since each cell is fitted alone.
//!
//! The basis is the **geometric** one: what is drawn of a cell is its
//! geometry, and a point is located in a cell by it, so the value has to vary
//! along that same shape.

use nalgebra::DMatrix;

use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::error::Result;

/// The fit of one FE space: `M`, `nodes_per_cell × gauss_count`. When the
/// fit is under-determined (fewer Gauss points than nodes) or singular, `M`
/// is the Gauss mean — every row `1 / gauss_count`, a constant over the cell —
/// which is a linear map too: the cell loop applies `M` and never branches.
pub(crate) struct CellFit {
    map: DMatrix<f64>,
}

impl CellFit {
    /// The fit of `fespace`, its `M` computed once.
    pub(crate) fn for_space(fespace: &SubFiniteElementSpace) -> Result<Self> {
        let nodes_per_cell = fespace.nodes_per_cell();
        let n_gauss = fespace.gauss_count();
        if n_gauss < nodes_per_cell {
            return Ok(Self::mean(nodes_per_cell, n_gauss));
        }
        let mut a = DMatrix::zeros(n_gauss, nodes_per_cell);
        let mut atw = DMatrix::zeros(nodes_per_cell, n_gauss);
        for g in 0..n_gauss {
            let row = fespace.n_at_g(g)?;
            let w = fespace.gauss_weight(g)?;
            for i in 0..nodes_per_cell {
                a[(g, i)] = row[i];
                atw[(i, g)] = row[i] * w;
            }
        }
        Ok(match (&atw * &a).lu().solve(&atw) {
            Some(map) => Self { map },
            None => Self::mean(nodes_per_cell, n_gauss),
        })
    }

    /// The Gauss mean at every node; no Gauss point at all reads 0.
    fn mean(nodes_per_cell: usize, n_gauss: usize) -> Self {
        let weight = if n_gauss == 0 {
            0.0
        } else {
            1.0 / n_gauss as f64
        };
        Self {
            map: DMatrix::from_element(nodes_per_cell, n_gauss, weight),
        }
    }

    /// Nodes per cell of the space: per component, the number of values
    /// [`nodal_values_into`] writes.
    ///
    /// [`nodal_values_into`]: Self::nodal_values_into
    pub(crate) fn nodes_per_cell(&self) -> usize {
        self.map.nrows()
    }

    /// The cell's nodal values from its Gauss values, `M F`, for `n_comp`
    /// components at once: `cell` holds them Gauss point by Gauss point
    /// (`cell[g·n_comp + c]`, the layout of a zone's values), `out` receives
    /// them node by node (`out[i·n_comp + c]`, length
    /// [`nodes_per_cell`](Self::nodes_per_cell)` · n_comp`) — no allocation,
    /// no branch.
    pub(crate) fn nodal_values_into(&self, cell: &[f64], n_comp: usize, out: &mut [f64]) {
        for (i, node) in out.chunks_exact_mut(n_comp).enumerate() {
            node.fill(0.0);
            for (g, point) in cell.chunks_exact(n_comp).enumerate() {
                let m = self.map[(i, g)];
                for (slot, f) in node.iter_mut().zip(point) {
                    *slot += m * f;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn tri3_space() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    /// Gauss values sampled from nodal values come back to them exactly — two
    /// components at once, interleaved Gauss point by Gauss point in, node by
    /// node out.
    #[test]
    fn an_interpolated_field_comes_back_exactly() {
        let fes = tri3_space();
        let zone = fes.get(0).unwrap();
        let s = zone.read();
        let fit = CellFit::for_space(&s).unwrap();
        let nodal = [[1.0, -3.0], [2.0, 5.0], [4.0, 0.5]];
        let mut cell = Vec::new();
        for g in 0..s.gauss_count() {
            let n = s.n_at_g(g).unwrap();
            for c in 0..2 {
                cell.push((0..3).map(|i| n[i] * nodal[i][c]).sum::<f64>());
            }
        }
        let mut out = vec![0.0; fit.nodes_per_cell() * 2];
        fit.nodal_values_into(&cell, 2, &mut out);
        for (i, node) in nodal.iter().enumerate() {
            for (c, t) in node.iter().enumerate() {
                let v = out[i * 2 + c];
                assert!((v - t).abs() < 1e-10, "{out:?} ≠ {nodal:?}");
            }
        }
    }

    /// Fewer Gauss points than nodes: the Gauss mean, constant over the cell.
    #[test]
    fn an_under_determined_fit_falls_back_to_the_mean() {
        let fit = CellFit::mean(3, 2);
        let mut out = [0.0; 3];
        fit.nodal_values_into(&[2.0, 4.0], 1, &mut out);
        assert_eq!(out, [3.0, 3.0, 3.0]);
        // No Gauss point at all: 0.
        CellFit::mean(3, 0).nodal_values_into(&[], 1, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }
}
