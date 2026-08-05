//! Per-element behaviour — one file per [`ElementType`], one trait to bind them.
//!
//! This module is to elements what [`crate::models`] is to physics. The
//! [`ElementType`] enum carries **storage and serialisation only**; everything
//! an element *knows about itself* — its reference nodes, its facets, its
//! reference domain — lives in a struct implementing [`ElementKind`], and a
//! single `match`, [`ElementType::as_kind`], binds the two.
//!
//! ```text
//! ElementType  (enum : stockage + sérialisation + nom cast3m)
//! ├── POI1, SEG2, TRI3, …, HEX27
//! └── as_kind(&self) -> &'static dyn ElementKind   ← l'unique match
//!
//! ElementKind  (trait : le comportement d'un type d'élément)
//! ├── ref_nodes / reversal_permutation / corner_count   ── l'identité
//! ├── facets / edges                                    ── la topologie
//! └── ref_centroid / ref_measure / contains_ref / clamp_ref  ── le domaine
//! ```
//!
//! Adding an element type means adding `atoms/element_kind/<name>.rs`, one variant
//! to [`ElementType`], one entry in [`ElementType::ALL`] and one arm to
//! `as_kind`. Every generic consumer — `skin`, `orient`, `border`, the
//! locators, the renderer — is written against the trait and does not change.
//!
//! # The reference-node convention
//!
//! One rule holds across every quadratic type and the rest of the crate leans
//! on it: **corners first, then one mid-side node per edge, in `edges()`
//! order**. So the mid node of edge `k` is always local index
//! `corner_count() + k`. `QUA9`/`HEX27` append their face and body centres
//! after that. The unit test `mid_edge_nodes_follow_the_convention` proves it
//! against the reference coordinates, for every type.

use crate::atoms::ElementType;

pub use interpolation::Interpolation;

mod hex20;
mod hex27;
mod hex8;
mod interpolation;
mod penta15;
mod penta6;
mod poi1;
mod pyra5;
mod qua4;
mod qua8;
mod qua9;
mod seg2;
mod seg3;
mod tet10;
mod tet4;
mod tri3;
mod tri6;

/// One facet of an element: a lower-dimensional element in its own right.
///
/// `nodes` lists the facet's local node indices **in the local order of
/// `element_type`**, oriented so the facet's normal points out of the parent
/// element. A facet of a quadratic element is itself quadratic — a `TET10`
/// face is a `TRI6`, a `HEX27` face is a `QUA9` — so `nodes` carries the
/// mid-side (and centre) nodes, not only the corners.
///
/// [`corners`](Self::corners) narrows it to the corner indices, which is what
/// adjacency keying wants: two neighbouring cells agree on the corners of the
/// facet they share whatever their degree.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Facet {
    /// The facet seen as an element (`SEG2` for a surface's side, `TRI6` for a
    /// `TET10` face, …).
    pub element_type: ElementType,
    /// Local node indices in the parent, in `element_type`'s local order,
    /// outward-oriented.
    pub nodes: &'static [usize],
}

impl Facet {
    /// The facet's corner indices — the first `element_type.corner_count()`
    /// entries of [`nodes`](Self::nodes), by the corners-first convention.
    pub fn corners(&self) -> &'static [usize] {
        &self.nodes[..self.element_type.as_kind().corner_count()]
    }
}

/// Everything a single element type knows about itself.
///
/// Split, like [`crate::models::SubModelKind`], between **required** methods
/// (the compiler forces every element to answer) and **provided** ones derived
/// from a more primitive datum. A new element writes the required set; the
/// rest follows.
pub trait ElementKind: Sync {
    // ── Identity (required) ─────────────────────────────────────────────

    /// The enum variant this kind implements — the way back from behaviour to
    /// storage.
    fn element_type(&self) -> ElementType;

    /// Reference coordinates of each node, in local order: the points where
    /// the matching shape function equals 1. **The root datum** — the node
    /// count and the topological dimension are read off it, and it is the
    /// ground truth every geometric table is tested against.
    fn ref_nodes(&self) -> &'static [&'static [f64]];

    /// Local-node permutation that reverses the element's orientation
    /// (output node `i` = input node `reversal_permutation()[i]`).
    ///
    /// Geometrically the orientation-reversing reflection swapping the first
    /// two reference axes, `(ξ, η, ζ) ↦ (η, ξ, ζ)` — for `SEG*`, the flip
    /// `ξ ↦ -ξ`. It carries **every** node, mid-side and centre included.
    fn reversal_permutation(&self) -> &'static [usize];

    /// Number of corner (vertex) nodes — the node count of the linear parent.
    /// Equals [`nodes_per_cell`](Self::nodes_per_cell) for a linear type.
    fn corner_count(&self) -> usize;

    // ── Reference topology (provided: nothing, for a point) ─────────────

    /// The element's facets — its edges if it is a surface, its faces if it is
    /// a volume — each oriented outwards. Empty for `POI1` and for `SEG*`,
    /// whose ends carry no orientation of their own.
    fn facets(&self) -> &'static [Facet] {
        &[]
    }

    /// The element's edges, as **corner** index pairs. For a quadratic type
    /// the mid node of edge `k` is `corner_count() + k` (see the module
    /// documentation), so this one table also describes the mid-side layout.
    fn edges(&self) -> &'static [[usize; 2]] {
        &[]
    }

    // ── Reference domain (required) ─────────────────────────────────────

    /// An interior point of the reference domain — the centroid. Doubles as
    /// the point of the one-point (`Reduced`) quadrature and as the Newton
    /// starting guess when inverting the geometric mapping.
    fn ref_centroid(&self) -> &'static [f64];

    /// Measure (length / area / volume) of the reference domain: 2 for `SEG2`,
    /// 1/2 for `TRI3`, 4/3 for `PYRA5`, 8 for `HEX8`. Quadrature weights sum
    /// to it.
    fn ref_measure(&self) -> f64;

    /// Is `ξ` inside the reference domain, allowing `tol` of slack?
    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool;

    /// Clamp `ξ` into the reference domain, in place. Used when projecting a
    /// point onto a cell: a Newton step that leaves the element is pulled back
    /// to its closest admissible reference point.
    fn clamp_ref(&self, xi: &mut [f64]);

    // ── Interpolation (required) ────────────────────────────────────────

    /// The Lagrange degree this type carries, or `None` for `POI1`, which has
    /// no reference frame to interpolate over. This is what
    /// [`Interpolation::is_compatible_with`] answers from: the degree is a
    /// property **of the element**, not a free choice.
    fn degree(&self) -> Option<Interpolation>;

    /// Shape functions `N_i(ξ)`, written into `out` (length
    /// [`nodes_per_cell`](Self::nodes_per_cell)).
    ///
    /// The allocation-free form, and the one to implement: inverting the
    /// geometric mapping ([`crate::ops::geom::locate_points`],
    /// [`crate::ops::geom::project_points`]) evaluates it dozens of times per
    /// point inside a Newton loop, where a `Vec` per call would dominate the
    /// cost. [`shape`](Self::shape) is the allocating convenience over it.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]);

    /// Reference derivatives `∂N_i/∂ξ_j`, written into `out` row-major:
    /// `out[i * topological_dim() + j]`, length
    /// `nodes_per_cell() × topological_dim()`.
    fn dshape_into(&self, xi: &[f64], out: &mut [f64]);

    // ── Provided: derived from ref_nodes ────────────────────────────────

    /// Number of nodes per cell.
    fn nodes_per_cell(&self) -> usize {
        self.ref_nodes().len()
    }

    /// Topological dimension (0 = point, 1 = segment, 2 = surface, 3 = volume).
    fn topological_dim(&self) -> usize {
        self.ref_nodes()[0].len()
    }

    /// Whether this type carries mid-side nodes, i.e. has more nodes than
    /// corners.
    fn is_quadratic(&self) -> bool {
        self.nodes_per_cell() > self.corner_count()
    }

    /// Shape functions `N_i(ξ)` as a fresh `Vec`. Convenience over
    /// [`shape_into`](Self::shape_into) for the callers that are not in a hot
    /// loop.
    fn shape(&self, xi: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nodes_per_cell()];
        self.shape_into(xi, &mut out);
        out
    }

    /// Reference derivatives as a fresh `Vec`. Convenience over
    /// [`dshape_into`](Self::dshape_into).
    fn dshape(&self, xi: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nodes_per_cell() * self.topological_dim()];
        self.dshape_into(xi, &mut out);
        out
    }
}

impl ElementType {
    /// Borrow the variant as its [`ElementKind`] behaviour.
    ///
    /// This is the **only** per-variant `match` in the element layer; every
    /// generic consumer (facets, reference domain, rendering) dispatches
    /// through it. Mirrors
    /// [`SubModel::as_kind`](crate::containers::model::SubModel::as_kind).
    pub fn as_kind(self) -> &'static dyn ElementKind {
        match self {
            ElementType::POI1 => &poi1::Poi1,
            ElementType::SEG2 => &seg2::Seg2,
            ElementType::TRI3 => &tri3::Tri3,
            ElementType::QUA4 => &qua4::Qua4,
            ElementType::TET4 => &tet4::Tet4,
            ElementType::PENTA6 => &penta6::Penta6,
            ElementType::PYRA5 => &pyra5::Pyra5,
            ElementType::HEX8 => &hex8::Hex8,
            ElementType::SEG3 => &seg3::Seg3,
            ElementType::TRI6 => &tri6::Tri6,
            ElementType::QUA8 => &qua8::Qua8,
            ElementType::TET10 => &tet10::Tet10,
            ElementType::PENTA15 => &penta15::Penta15,
            ElementType::HEX20 => &hex20::Hex20,
            ElementType::QUA9 => &qua9::Qua9,
            ElementType::HEX27 => &hex27::Hex27,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every type with a reference frame (`POI1` has none).
    fn fe_types() -> impl Iterator<Item = ElementType> {
        ElementType::ALL
            .iter()
            .copied()
            .filter(|et| et.topological_dim() > 0)
    }

    #[test]
    fn as_kind_round_trips_and_agrees_with_the_enum_metadata() {
        for &et in ElementType::ALL {
            let k = et.as_kind();
            assert_eq!(k.element_type(), et);
            assert_eq!(k.nodes_per_cell(), et.nodes_per_cell(), "{et}: node count");
            assert_eq!(
                k.topological_dim(),
                et.topological_dim(),
                "{et}: topological dim"
            );
            assert_eq!(
                k.reversal_permutation(),
                et.reversal_permutation(),
                "{et}: reversal"
            );
            assert_eq!(k.ref_nodes().len(), et.nodes_per_cell(), "{et}: ref_nodes");
            for (i, n) in k.ref_nodes().iter().enumerate() {
                assert_eq!(n.len(), et.topological_dim(), "{et}: node {i} dimension");
            }
        }
    }

    /// The convention the whole crate leans on: corners first, then the mid
    /// node of edge `k` at local index `corner_count() + k`, sitting exactly
    /// at that edge's midpoint in the reference frame.
    #[test]
    fn mid_edge_nodes_follow_the_convention() {
        for et in fe_types() {
            let k = et.as_kind();
            if !k.is_quadratic() {
                continue;
            }
            let nodes = k.ref_nodes();
            let nc = k.corner_count();
            for (e, [a, b]) in k.edges().iter().enumerate() {
                let mid = nc + e;
                assert!(
                    mid < nodes.len(),
                    "{et}: edge {e} has no mid node at slot {mid}"
                );
                for d in 0..k.topological_dim() {
                    let want = 0.5 * (nodes[*a][d] + nodes[*b][d]);
                    assert!(
                        (nodes[mid][d] - want).abs() < 1e-12,
                        "{et}: node {mid} is not the midpoint of edge {e} = ({a}, {b})"
                    );
                }
            }
        }
    }

    /// Edges are corner pairs, distinct, and every corner is on some edge.
    #[test]
    fn edges_are_well_formed() {
        for et in fe_types() {
            let k = et.as_kind();
            let nc = k.corner_count();
            let mut seen = std::collections::HashSet::new();
            for &[a, b] in k.edges() {
                assert!(
                    a < nc && b < nc,
                    "{et}: edge ({a}, {b}) is not a corner pair"
                );
                assert_ne!(a, b, "{et}: degenerate edge");
                let key = if a < b { (a, b) } else { (b, a) };
                assert!(seen.insert(key), "{et}: edge ({a}, {b}) listed twice");
            }
            if k.topological_dim() > 0 {
                for c in 0..nc {
                    assert!(
                        k.edges().iter().any(|&[a, b]| a == c || b == c),
                        "{et}: corner {c} lies on no edge"
                    );
                }
            }
        }
    }

    /// A facet's nodes are valid indices, its degree matches its parent's, and
    /// its geometry is consistent: the facet's own reference nodes map onto
    /// the parent's, so the hand-written tables cannot drift.
    #[test]
    fn facets_are_consistent_with_the_parent_geometry() {
        for et in fe_types() {
            let k = et.as_kind();
            let npc = k.nodes_per_cell();
            for (f, facet) in k.facets().iter().enumerate() {
                let fk = facet.element_type.as_kind();
                assert_eq!(
                    facet.nodes.len(),
                    fk.nodes_per_cell(),
                    "{et}: facet {f} has {} nodes, {} wants {}",
                    facet.nodes.len(),
                    facet.element_type,
                    fk.nodes_per_cell()
                );
                assert_eq!(
                    fk.topological_dim(),
                    k.topological_dim() - 1,
                    "{et}: facet {f} is not of codimension 1"
                );
                assert_eq!(
                    fk.is_quadratic(),
                    k.is_quadratic(),
                    "{et}: facet {f} does not share its parent's degree"
                );
                for &n in facet.nodes {
                    assert!(n < npc, "{et}: facet {f} references node {n} ≥ {npc}");
                }
                let mut seen = std::collections::HashSet::new();
                for &n in facet.nodes {
                    assert!(seen.insert(n), "{et}: facet {f} repeats node {n}");
                }
                // A mid-side node of the facet must be the midpoint of the
                // matching corner pair, measured in the *parent's* frame.
                // Linear facets have no mid node to check.
                if !fk.is_quadratic() {
                    continue;
                }
                let parent_nodes = k.ref_nodes();
                let fnc = fk.corner_count();
                for (e, [a, b]) in fk.edges().iter().enumerate() {
                    let mid = facet.nodes[fnc + e];
                    let (pa, pb) = (facet.nodes[*a], facet.nodes[*b]);
                    for d in 0..k.topological_dim() {
                        let want = 0.5 * (parent_nodes[pa][d] + parent_nodes[pb][d]);
                        assert!(
                            (parent_nodes[mid][d] - want).abs() < 1e-12,
                            "{et}: facet {f} node {mid} is not the midpoint of ({pa}, {pb})"
                        );
                    }
                }
            }
        }
    }

    /// Every facet corner set appears exactly once: a volume's faces partition
    /// its boundary, so no face is listed twice or omitted by symmetry.
    #[test]
    fn facet_corner_sets_are_distinct() {
        for et in fe_types() {
            let k = et.as_kind();
            let mut seen = std::collections::HashSet::new();
            for (f, facet) in k.facets().iter().enumerate() {
                let mut key = facet.corners().to_vec();
                key.sort_unstable();
                assert!(seen.insert(key), "{et}: facet {f} duplicates another");
            }
        }
    }

    #[test]
    fn the_centroid_is_inside_and_survives_clamping() {
        for et in fe_types() {
            let k = et.as_kind();
            let c = k.ref_centroid();
            assert_eq!(c.len(), k.topological_dim(), "{et}: centroid dimension");
            assert!(k.contains_ref(c, 0.0), "{et}: centroid is outside");
            let mut xi = c.to_vec();
            k.clamp_ref(&mut xi);
            for (a, b) in xi.iter().zip(c.iter()) {
                assert!((a - b).abs() < 1e-12, "{et}: clamping moved the centroid");
            }
            assert!(
                k.ref_measure() > 0.0,
                "{et}: non-positive reference measure"
            );
        }
    }

    /// Every reference node is in the (closed) reference domain, and a point
    /// pushed well outside comes back inside once clamped — for the types
    /// that define a clamp (surfaces and lines; volumes are never projected
    /// onto, so their clamp is a no-op).
    #[test]
    fn reference_nodes_lie_in_the_domain() {
        for et in fe_types() {
            let k = et.as_kind();
            for (i, n) in k.ref_nodes().iter().enumerate() {
                assert!(k.contains_ref(n, 1e-12), "{et}: node {i} is outside");
            }
        }
    }

    #[test]
    fn clamping_brings_an_outside_point_back_in() {
        for et in fe_types().filter(|et| et.topological_dim() < 3) {
            let k = et.as_kind();
            let mut xi = vec![7.5; k.topological_dim()];
            k.clamp_ref(&mut xi);
            assert!(k.contains_ref(&xi, 1e-9), "{et}: clamp left {xi:?} outside");
            let mut xi = vec![-7.5; k.topological_dim()];
            k.clamp_ref(&mut xi);
            assert!(k.contains_ref(&xi, 1e-9), "{et}: clamp left {xi:?} outside");
        }
    }
}
