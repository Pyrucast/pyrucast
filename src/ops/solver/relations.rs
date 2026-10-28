//! The constraint relations of an assembled [`Matrix`], read from the matrix
//! itself — what both relation-aware paths (the active set and the
//! elimination) start from.
//!
//! A relation is a row named [`IMPOSED`]; its coefficients are that row, its
//! multiplier is the column [`MULTIPLIER`] of the same node, and its sense is
//! declared by the block that built it ([`Matrix::inequality_senses`]). No
//! model is consulted: the matrix is what is solved, so it is what is read.

use crate::atoms::NodeId;
use crate::containers::matrix::{dof_node, dof_var, Matrix};
use crate::error::{PyrucastError, Result};
use crate::models::{RelationSense, IMPOSED, MULTIPLIER};
use std::collections::HashMap;

/// One constraint relation, as the assembled matrix holds it.
pub(super) struct RelationRow {
    /// Index of the row `(multiplier, imposed)` — the relation itself.
    pub(super) row: usize,
    /// The relation's multiplier node.
    pub(super) multiplier: NodeId,
    /// Index of the column `(multiplier, lambda)` — its multiplier.
    pub(super) lambda_col: usize,
    /// `(column, coefficient)` of every entry the row stores: `Σ aₖ·uₖ`.
    pub(super) terms: Vec<(usize, f64)>,
    /// `Equality`, or the sense of an inequality.
    pub(super) sense: RelationSense,
}

/// Every relation of `matrix`, in row order. Empty for a matrix without
/// relation rows.
///
/// # Errors
///
/// A relation row whose multiplier column is missing: the matrix was not
/// assembled from a constraint (each one writes both).
pub(super) fn relation_rows(matrix: &Matrix) -> Result<Vec<RelationRow>> {
    let vars = matrix.dof_vars();
    let Some(imposed) = vars.iter().position(|v| v == IMPOSED) else {
        return Ok(Vec::new());
    };
    let lambda = vars.iter().position(|v| v == MULTIPLIER);
    let row_keys = matrix.row_dof_keys()?;
    let col_keys = matrix.col_dof_keys()?;
    let lambda_col_of: HashMap<NodeId, usize> = col_keys
        .iter()
        .enumerate()
        .filter(|(_, k)| Some(dof_var(**k) as usize) == lambda)
        .map(|(j, &k)| (dof_node(k), j))
        .collect();
    let senses = matrix.inequality_senses();
    let (offsets, cols, values) = matrix.csr_arrays()?;

    let mut rows = Vec::new();
    for (row, &key) in row_keys.iter().enumerate() {
        if dof_var(key) as usize != imposed {
            continue;
        }
        let multiplier = dof_node(key);
        let lambda_col = *lambda_col_of.get(&multiplier).ok_or_else(|| {
            PyrucastError::Message(format!(
                "solve: the relation row ({multiplier:?}, '{IMPOSED}') has no multiplier \
                 column ({multiplier:?}, '{MULTIPLIER}') — a constraint writes both"
            ))
        })?;
        let terms = (offsets[row]..offsets[row + 1])
            .map(|k| (cols[k], values[k]))
            .collect();
        rows.push(RelationRow {
            row,
            multiplier,
            lambda_col,
            terms,
            sense: senses
                .get(&multiplier)
                .copied()
                .unwrap_or(RelationSense::Equality),
        });
    }
    Ok(rows)
}
