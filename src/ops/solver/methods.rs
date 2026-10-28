//! Delegation methods — the "subject" face of the solver.
//!
//! See `CONVENTIONS.md` § "The verb also exposed as a method": the matrix is
//! the subject of a solve, and the free function puts it first.

use crate::containers::matrix::Matrix;
use crate::containers::node_field::NodeField;
use crate::error::Result;

use super::SolveOptions;

impl Matrix {
    /// See [`solver::solve`](fn@crate::ops::solver::solve).
    pub fn solve(&self, rhs: &NodeField) -> Result<NodeField> {
        crate::ops::solver::solve(self, rhs)
    }

    /// See [`solver::solve_with_options`](fn@crate::ops::solver::solve_with_options).
    pub fn solve_with_options(&self, rhs: &NodeField, options: &SolveOptions) -> Result<NodeField> {
        crate::ops::solver::solve_with_options(self, rhs, options)
    }
}
