//! `PENTA15` — the 15-node serendipity prism.

use super::penta6::{contains_prism, Penta6, EDGES};
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 15-node serendipity prism (Lagrange-2 `PENTA6`). Corners 0..5 as `PENTA6`,
/// then mid-edge nodes 6..14: bottom triangle `(0,1)`, `(1,2)`, `(2,0)`, top
/// triangle `(3,4)`, `(4,5)`, `(5,3)`, vertical `(0,3)`, `(1,4)`, `(2,5)`.
pub struct Penta15;

/// Two `TRI6` caps then three `QUA8` sides. Mid nodes follow the edge order:
/// `mid(0,1) = 6`, `mid(1,2) = 7`, `mid(2,0) = 8`, `mid(3,4) = 9`,
/// `mid(4,5) = 10`, `mid(5,3) = 11`, `mid(0,3) = 12`, `mid(1,4) = 13`,
/// `mid(2,5) = 14`.
const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[0, 2, 1, 8, 7, 6],
    },
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[3, 4, 5, 9, 10, 11],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[0, 1, 4, 3, 6, 13, 9, 12],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[1, 2, 5, 4, 7, 14, 10, 13],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[2, 0, 3, 5, 8, 12, 11, 14],
    },
];

impl ElementKind for Penta15 {
    fn element_type(&self) -> ElementType {
        ElementType::PENTA15
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[0.5, 0.0, 0.0],
            &[0.5, 0.5, 0.0],
            &[0.0, 0.5, 0.0],
            &[0.5, 0.0, 1.0],
            &[0.5, 0.5, 1.0],
            &[0.0, 0.5, 1.0],
            &[0.0, 0.0, 0.5],
            &[1.0, 0.0, 0.5],
            &[0.0, 1.0, 0.5],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13]
    }

    fn corner_count(&self) -> usize {
        6
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        Penta6.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Penta6.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_prism(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
}
