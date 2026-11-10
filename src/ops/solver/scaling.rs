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
//! The cure is to bring the relations to the scale of the structure: the
//! relation rows (dual `imposed`) and the multiplier columns (primal `lambda`)
//! are multiplied by `s`, the largest diagonal entry of the rest of the matrix.
//! The system solved is `R A C · y = R b`, with `R` and `C` diagonal (`s` on the
//! relations, `1` elsewhere), and the solution is `x = C y`: the same `x`, the
//! relations now holding to round-off.

use crate::containers::matrix::{dof_var, DofKey};
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
    /// the matrix holds no relation, or nothing to scale it against: it is then
    /// factorized as assembled.
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

        // The scale of the structure: its largest diagonal entry.
        let mut scale = 0.0_f64;
        for (row, relation) in is_relation_row.iter().enumerate() {
            if *relation {
                continue;
            }
            for k in offsets[row]..offsets[row + 1] {
                if cols[k] == row {
                    scale = scale.max(vals[k].abs());
                }
            }
        }
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        let factor = |flag: &bool| if *flag { scale } else { 1.0 };
        Some(Self {
            rows: is_relation_row.iter().map(factor).collect(),
            cols: is_multiplier_col.iter().map(factor).collect(),
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
