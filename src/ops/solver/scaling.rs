//! Balancing the relation rows of a saddle-point before it is factorized.
//!
//! A Lagrange system `[[K, Cᵀ], [C, 0]]` mixes two scales: `K` carries the
//! stiffness of the structure (`E·h`, some 10⁹ N/m for a steel part), `C` the
//! coefficients of the relations (1 for a clamp). Factorized as assembled, the
//! pivots of the relation rows are a billion times smaller than the others,
//! and the round-off of the elimination shows where it matters most: a clamped
//! node of a steel bar keeps a displacement of a few millionths of the tip's,
//! where it should keep none.
//!
//! The cure is to bring each relation to the scale of the unknowns it ties:
//! relation `r` (its row, dual `imposed`, and its multiplier column, primal
//! `lambda`) is multiplied by
//!
//! ```text
//! s_r = 1.5 · max_k |K|_k / max_k |a_rk|
//! ```
//!
//! where `k` runs over the unknowns of the relation, `a_rk` its coefficients,
//! and `|K|_k` the largest entry, in absolute value, of unknown `k`'s column
//! in the rest of the matrix — the scale of that unknown's own stiffness. An
//! unknown with nothing to measure there falls back on the largest entry of
//! the whole matrix. The system solved is `R A C · y = R b`, with `R` and `C`
//! diagonal (`s_r` on the relations, `1` elsewhere), and the solution is
//! `x = C y`: the same `x`, the relations now holding to round-off.
//!
//! One factor per relation, not one for all: a relation on a soft part, next
//! to a part a thousand times stiffer, is brought to its own part's scale, not
//! to the stiff one's.

use crate::containers::matrix::{dof_node, dof_var, DofKey};
use crate::models::{IMPOSED, MULTIPLIER};

/// The diagonal scalings of the rows and of the columns of a saddle-point.
pub(super) struct RelationScaling {
    /// One factor per row: `s` on a relation row, `1` elsewhere.
    rows: Vec<f64>,
    /// One factor per column: `s` on a multiplier column, `1` elsewhere.
    cols: Vec<f64>,
}

impl RelationScaling {
    /// The scaling of the square CSR `(offsets, cols, vals)`, its rows and
    /// columns named by `row_keys` / `col_keys` against `vars` — or `None` when
    /// the matrix holds no relation: it is then factorized as assembled.
    pub(super) fn of(
        vars: &[String],
        row_keys: &[DofKey],
        col_keys: &[DofKey],
        offsets: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> Option<Self> {
        let imposed = vars.iter().position(|v| v == IMPOSED)? as u32;
        let lambda = vars.iter().position(|v| v == MULTIPLIER)? as u32;
        let is_relation_row: Vec<bool> = row_keys.iter().map(|&k| dof_var(k) == imposed).collect();
        let is_multiplier_col: Vec<bool> = col_keys.iter().map(|&k| dof_var(k) == lambda).collect();
        if !is_relation_row.iter().any(|&r| r) {
            return None;
        }

        // The scale of each unknown: its column's largest entry outside the
        // relations, and the largest of all for the fallback — one pass.
        let mut column_scale = vec![0.0_f64; col_keys.len()];
        for (row, &relation) in is_relation_row.iter().enumerate() {
            if relation {
                continue;
            }
            for k in offsets[row]..offsets[row + 1] {
                let c = cols[k];
                column_scale[c] = column_scale[c].max(vals[k].abs());
            }
        }
        let largest = column_scale.iter().copied().fold(0.0_f64, f64::max);
        // Nothing at all to measure against: the relations keep their scale.
        let fallback = if largest.is_finite() && largest > f64::MIN_POSITIVE {
            largest
        } else {
            1.0
        };

        // Each relation's multiplier column, found by its multiplier node.
        let lambda_col_of: std::collections::HashMap<u32, usize> = col_keys
            .iter()
            .enumerate()
            .filter(|&(c, _)| is_multiplier_col[c])
            .map(|(c, &key)| (dof_node(key).0, c))
            .collect();

        let mut rows = vec![1.0; row_keys.len()];
        let mut col_factors = vec![1.0; col_keys.len()];
        for (row, &relation) in is_relation_row.iter().enumerate() {
            if !relation {
                continue;
            }
            // A relation row holds its coefficients on primal columns only.
            let (mut stiffness, mut coefficient) = (0.0_f64, 0.0_f64);
            for k in offsets[row]..offsets[row + 1] {
                stiffness = stiffness.max(column_scale[cols[k]]);
                coefficient = coefficient.max(vals[k].abs());
            }
            if stiffness <= f64::EPSILON * fallback {
                stiffness = fallback;
            }
            let factor = if coefficient > 0.0 {
                1.5 * stiffness / coefficient
            } else {
                1.0
            };
            rows[row] = factor;
            if let Some(&c) = lambda_col_of.get(&dof_node(row_keys[row]).0) {
                col_factors[c] = factor;
            }
        }
        Some(Self {
            rows,
            cols: col_factors,
        })
    }

    /// Whether the rows and the columns are scaled alike — what keeps a
    /// symmetric matrix symmetric once scaled.
    pub(super) fn is_symmetric(&self) -> bool {
        self.rows == self.cols
    }

    /// `R A C`: the values of the CSR `(offsets, cols, vals)`, scaled.
    pub(super) fn scale_csr(&self, offsets: &[usize], cols: &[usize], vals: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(vals.len());
        for row in 0..offsets.len() - 1 {
            let r = self.rows[row];
            for k in offsets[row]..offsets[row + 1] {
                out.push(r * vals[k] * self.cols[cols[k]]);
            }
        }
        out
    }

    /// `R A C` on CSC values: column `j` scaled by its factor, each entry by
    /// its row's.
    pub(super) fn scale_csc(&self, col_offsets: &[usize], row_indices: &[usize], vals: &mut [f64]) {
        for col in 0..col_offsets.len() - 1 {
            let c = self.cols[col];
            for k in col_offsets[col]..col_offsets[col + 1] {
                vals[k] *= self.rows[row_indices[k]] * c;
            }
        }
    }

    /// The right-hand side of the scaled system: `R b`, in place.
    pub(super) fn scale_rhs(&self, b: &mut [f64]) {
        for (v, r) in b.iter_mut().zip(&self.rows) {
            *v *= r;
        }
    }

    /// The solution of the original system from the scaled one's: `x = C y`,
    /// in place.
    pub(super) fn unscale_solution(&self, y: &mut [f64]) {
        for (v, c) in y.iter_mut().zip(&self.cols) {
            *v *= c;
        }
    }

    /// The factor of row `row`.
    pub(super) fn row(&self, row: usize) -> f64 {
        self.rows[row]
    }

    /// The factor of column `col`.
    pub(super) fn col(&self, col: usize) -> f64 {
        self.cols[col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::NodeId;
    use crate::containers::matrix::dof_key;

    /// Two unknowns of stiffness 1 and 1000, one relation on each, of
    /// coefficients 1 and 2: each relation takes its own unknown's scale,
    /// `1.5 · |K| / |a|` — 1.5 and 750 — on its row and its multiplier column.
    #[test]
    fn each_relation_takes_its_own_unknowns_scale() {
        let vars: Vec<String> = ["u", "f", IMPOSED, MULTIPLIER]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (u, f, imposed, lambda) = (0, 1, 2, 3);
        // Rows: f(0), f(1), imposed(m0), imposed(m1); columns: u(0), u(1),
        // lambda(m0), lambda(m1).
        let (n0, n1, m0, m1) = (NodeId(0), NodeId(1), NodeId(2), NodeId(3));
        let row_keys = [
            dof_key(n0, f),
            dof_key(n1, f),
            dof_key(m0, imposed),
            dof_key(m1, imposed),
        ];
        let col_keys = [
            dof_key(n0, u),
            dof_key(n1, u),
            dof_key(m0, lambda),
            dof_key(m1, lambda),
        ];
        // [[1, 0, 1, 0], [0, 1000, 0, 2], [1, 0, 0, 0], [0, 2, 0, 0]]
        let offsets = [0, 2, 4, 5, 6];
        let cols = [0, 2, 1, 3, 0, 1];
        let vals = [1.0, 1.0, 1000.0, 2.0, 1.0, 2.0];
        let s = RelationScaling::of(&vars, &row_keys, &col_keys, &offsets, &cols, &vals)
            .expect("relations to scale");
        assert_eq!((s.row(0), s.row(1)), (1.0, 1.0));
        assert_eq!((s.row(2), s.row(3)), (1.5, 750.0));
        assert_eq!((s.col(0), s.col(1)), (1.0, 1.0));
        assert_eq!((s.col(2), s.col(3)), (1.5, 750.0));
        assert!(s.is_symmetric());
    }
}
