//! Delegation methods — the "subject" face of the solvers.
//!
//! See `CONVENTIONS.md` § "The verb also exposed as a method". The three
//! variants file as a family behind `Matrix::solve`, which the order
//! of the free functions' arguments has mirrored since it puts the matrix —
//! le sujet — en tête.

use crate::containers::matrix::Matrix;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::Result;

impl Matrix {
    /// Voir [`solver::lu::solve`](fn@crate::ops::solver::lu::solve).
    pub fn solve(&self, rhs: &NodeField) -> Result<NodeField> {
        crate::ops::solver::lu::solve(self, rhs)
    }

    /// Voir [`solver::eliminate::solve`](fn@crate::ops::solver::eliminate::solve).
    pub fn solve_eliminate(&self, model: &Model, rhs: &NodeField) -> Result<NodeField> {
        crate::ops::solver::eliminate::solve(self, model, rhs)
    }

    /// Voir [`solver::unilateral::solve`](fn@crate::ops::solver::unilateral::solve).
    pub fn solve_unilateral(&self, model: &Model, rhs: &NodeField) -> Result<NodeField> {
        crate::ops::solver::unilateral::solve(self, model, rhs)
    }
}
