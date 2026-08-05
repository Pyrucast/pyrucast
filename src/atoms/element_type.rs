//! Finite-element types supported by pyrucast.
//!
//! The list is deliberately small; adding a new element type means adding
//! a variant to [`ElementType`] and completing the metadata functions
//! ([`nodes_per_cell`](ElementType::nodes_per_cell),
//! [`topological_dim`](ElementType::topological_dim),
//! [`name`](ElementType::name)).
//!
//! # Reference frame and local node numbering
//!
//! These conventions define the reference element `ξ` of each
//! [`ElementType`] and the order in which a cell's nodes must be supplied
//! to [`crate::containers::mesh::SubMesh::add_cell`] / [`crate::containers::mesh::Mesh::add_cell`].
//! They are the contract between the geometry layer (mesh) and any
//! interpolation layer ([`crate::containers::finite_element_space::Interpolation`]) built on
//! top.
//!
//! | Variant | Reference domain | Reference node coordinates (in order) |
//! |---|---|---|
//! | `POI1` | `{ξ = 0}` (no reference frame) | none — POI1 is just a node list |
//! | `SEG2` | `ξ ∈ [-1, +1]` | node 0 at `ξ = -1`, node 1 at `ξ = +1` |
//! | `TRI3` | `ξ, η ∈ [0, 1]` with `ξ + η ≤ 1` | `(0, 0)`, `(1, 0)`, `(0, 1)` — CCW |
//! | `QUA4` | `ξ, η ∈ [-1, +1]` | `(-1, -1)`, `(1, -1)`, `(1, 1)`, `(-1, 1)` — CCW |
//! | `TET4` | `ξ, η, ζ ∈ [0, 1]` with `ξ + η + ζ ≤ 1` | `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)` — face 0-1-2 CCW seen from node 3 |
//! | `PENTA6` | `ξ, η ∈ [0, 1]` with `ξ + η ≤ 1`, `ζ ∈ [0, 1]` | bottom triangle CCW then top triangle CCW: `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, `(1,0,1)`, `(0,1,1)` — the extrusion of a TRI3 along `ζ` |
//! | `PYRA5` | `ζ ∈ [0, 1]`, `ξ, η ∈ [-(1-ζ), +(1-ζ)]` | square base CCW seen from the apex, then the apex: `(-1,-1,0)`, `(1,-1,0)`, `(1,1,0)`, `(-1,1,0)`, `(0,0,1)` |
//! | `HEX8` | `ξ, η, ζ ∈ [-1, +1]` | bottom face CCW then top face CCW: `(-1,-1,-1)`, `(1,-1,-1)`, `(1,1,-1)`, `(-1,1,-1)`, `(-1,-1,1)`, `(1,-1,1)`, `(1,1,1)`, `(-1,1,1)` |
//!
//! ## Quadratic (Lagrange-2) variants
//!
//! Each quadratic type shares the reference frame and corner numbering of
//! its linear parent, then adds the **mid-edge** nodes in a fixed edge
//! order (the VTK convention, so [`crate::ops::export`] writes them
//! verbatim). Mid-edge node `k` sits at the midpoint of the edge listed
//! for it.
//!
//! | Variant | Parent | Mid-edge nodes (in order), each on edge `(a, b)` |
//! |---|---|---|
//! | `SEG3` | `SEG2` | node 2 on `(0, 1)` — i.e. `ξ = 0` |
//! | `TRI6` | `TRI3` | 3 on `(0,1)`, `(1,2)`, `(2,0)` |
//! | `QUA8` | `QUA4` | 4 on `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)` |
//! | `QUA9` | `QUA4` | 4 on `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)`, then a **center** node 8 at `(0, 0)` |
//! | `HEX27` | `HEX8` | 12 edges (as `HEX20`), then 6 **face-center** nodes 20..25 (`x∓`, `y∓`, `z∓`), then a **body-center** node 26 |
//! | `TET10` | `TET4` | 6 on `(0,1)`, `(1,2)`, `(2,0)`, `(0,3)`, `(1,3)`, `(2,3)` |
//! | `PENTA15` | `PENTA6` | 9: `(0,1)`,`(1,2)`,`(2,0)` (bottom), `(3,4)`,`(4,5)`,`(5,3)` (top), `(0,3)`,`(1,4)`,`(2,5)` (vertical) |
//! | `HEX20` | `HEX8` | 12: `(0,1)`,`(1,2)`,`(2,3)`,`(3,0)` (bottom), `(4,5)`,`(5,6)`,`(6,7)`,`(7,4)` (top), `(0,4)`,`(1,5)`,`(2,6)`,`(3,7)` (vertical) |
//!
//! `QUA8`, `HEX20` and `PENTA15` are **serendipity** (edge nodes only, no
//! face or interior nodes); `SEG3`, `TRI6`, `TET10`, `QUA9` and `HEX27` are
//! complete Lagrange elements (`QUA9`/`HEX27` are the bi-/tri-quadratic quad
//! and hex, carrying face and interior nodes).
//!
//! These conventions are compatible with the orientations already enforced
//! elsewhere in the codebase (CCW filling in
//! [`crate::ops::mesh::triangulate_surface()`], HEX8 node ordering in
//! [`crate::ops::mesh::extrude()`]).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported element types. Names follow the cast3m convention.
///
/// See the module-level documentation for the reference frame `ξ` and the
/// local node numbering attached to each variant.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ElementType {
    /// 1 node. A POI1 submesh is effectively a list of nodes; no reference
    /// frame is attached.
    POI1,
    /// 2-node segment. Reference: `ξ ∈ [-1, +1]`. Local order: node 0 at
    /// `ξ = -1`, node 1 at `ξ = +1`.
    SEG2,
    /// 3-node triangle. Reference: `ξ, η ∈ [0, 1]`, `ξ + η ≤ 1`. Local
    /// order (CCW): `(0, 0)`, `(1, 0)`, `(0, 1)`.
    TRI3,
    /// 4-node quadrangle. Reference: `ξ, η ∈ [-1, +1]`. Local order (CCW):
    /// `(-1, -1)`, `(1, -1)`, `(1, 1)`, `(-1, 1)`.
    QUA4,
    /// 4-node tetrahedron. Reference: `ξ, η, ζ ∈ [0, 1]`, `ξ + η + ζ ≤ 1`.
    /// Local order: `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)` — face
    /// 0-1-2 CCW seen from node 3.
    TET4,
    /// 6-node prism (pentahedron), the extrusion of a TRI3 along `ζ`.
    /// Reference: `ξ, η ∈ [0, 1]`, `ξ + η ≤ 1`, `ζ ∈ [0, 1]`. Local order:
    /// bottom triangle CCW (nodes 0..2 at `ζ = 0`), then top triangle CCW
    /// (nodes 3..5 at `ζ = 1`), i.e. `(0,0,0)`, `(1,0,0)`, `(0,1,0)`,
    /// `(0,0,1)`, `(1,0,1)`, `(0,1,1)`.
    PENTA6,
    /// 5-node pyramid: a square base and an apex. Reference: `ζ ∈ [0, 1]`
    /// with `ξ, η ∈ [-(1-ζ), +(1-ζ)]`, so the square shrinks to a point at
    /// the apex. Local order: base CCW seen from the apex (nodes 0..3 at
    /// `ζ = 0`), then the apex, i.e. `(-1,-1,0)`, `(1,-1,0)`, `(1,1,0)`,
    /// `(-1,1,0)`, `(0,0,1)`.
    ///
    /// This is the element that makes a hexahedron and a tetrahedron meet:
    /// its square face matches a `HEX8` face and its four triangles match
    /// `TET4` faces, so a hexahedral layer can be closed onto a tetrahedral
    /// core without a hanging node.
    PYRA5,
    /// 8-node hexahedron. Reference: `ξ, η, ζ ∈ [-1, +1]`. Local order:
    /// bottom face CCW (nodes 0..3), then top face CCW (nodes 4..7),
    /// i.e. `(-1,-1,-1)`, `(1,-1,-1)`, `(1,1,-1)`, `(-1,1,-1)`,
    /// `(-1,-1,1)`, `(1,-1,1)`, `(1,1,1)`, `(-1,1,1)`.
    HEX8,
    /// 3-node quadratic segment (Lagrange-2 `SEG2`). Corners 0, 1 at
    /// `ξ = ∓1`, mid node 2 at `ξ = 0`.
    SEG3,
    /// 6-node quadratic triangle (Lagrange-2 `TRI3`). Corners 0..2 as
    /// `TRI3`, then mid-edge nodes 3, 4, 5 on edges `(0,1)`, `(1,2)`,
    /// `(2,0)`.
    TRI6,
    /// 8-node serendipity quadrangle (Lagrange-2 `QUA4`). Corners 0..3 as
    /// `QUA4`, then mid-edge nodes 4..7 on edges `(0,1)`, `(1,2)`, `(2,3)`,
    /// `(3,0)`.
    QUA8,
    /// 10-node quadratic tetrahedron (Lagrange-2 `TET4`). Corners 0..3 as
    /// `TET4`, then mid-edge nodes 4..9 on edges `(0,1)`, `(1,2)`, `(2,0)`,
    /// `(0,3)`, `(1,3)`, `(2,3)`.
    TET10,
    /// 15-node serendipity prism (Lagrange-2 `PENTA6`). Corners 0..5 as
    /// `PENTA6`, then mid-edge nodes 6..14: bottom triangle `(0,1)`,
    /// `(1,2)`, `(2,0)`, top triangle `(3,4)`, `(4,5)`, `(5,3)`, vertical
    /// `(0,3)`, `(1,4)`, `(2,5)`.
    PENTA15,
    /// 20-node serendipity hexahedron (Lagrange-2 `HEX8`). Corners 0..7 as
    /// `HEX8`, then mid-edge nodes 8..19: bottom `(0,1)`, `(1,2)`, `(2,3)`,
    /// `(3,0)`, top `(4,5)`, `(5,6)`, `(6,7)`, `(7,4)`, vertical `(0,4)`,
    /// `(1,5)`, `(2,6)`, `(3,7)`.
    HEX20,
    /// 9-node biquadratic quadrangle (full Lagrange-2 `QUA4`). Corners 0..3
    /// as `QUA4`, mid-edge nodes 4..7 on edges `(0,1)`, `(1,2)`, `(2,3)`,
    /// `(3,0)`, then a **center** node 8 at `(0, 0)`. Unlike the serendipity
    /// `QUA8`, it carries the central node (complete `Q2` tensor product).
    QUA9,
    /// 27-node tri-quadratic hexahedron (full Lagrange-2 `HEX8`). Corners
    /// 0..7 and mid-edge nodes 8..19 as `HEX20`, then 6 face-center nodes
    /// 20..25 (faces `x-`, `x+`, `y-`, `y+`, `z-`, `z+`), then a body-center
    /// node 26 at `(0, 0, 0)`. The complete `Q2` tensor product on the hex.
    HEX27,
}

impl ElementType {
    /// Every variant, in declaration order.
    ///
    /// The single list the rest of the crate iterates over: parsing
    /// ([`from_name`](Self::from_name)) and the exhaustiveness tests read it
    /// rather than repeating a literal array that a new variant would silently
    /// miss. Adding a variant means extending **this** slice — nothing else
    /// enumerates the element types.
    ///
    /// It includes `POI1`, which has no reference frame; a consumer that wants
    /// only the types carrying one filters on
    /// [`topological_dim`](Self::topological_dim) `> 0`.
    pub const ALL: &'static [ElementType] = &[
        Self::POI1,
        Self::SEG2,
        Self::TRI3,
        Self::QUA4,
        Self::TET4,
        Self::PENTA6,
        Self::PYRA5,
        Self::HEX8,
        Self::SEG3,
        Self::TRI6,
        Self::QUA8,
        Self::TET10,
        Self::PENTA15,
        Self::HEX20,
        Self::QUA9,
        Self::HEX27,
    ];

    /// Number of nodes per cell for this element type.
    pub fn nodes_per_cell(self) -> usize {
        match self {
            Self::POI1 => 1,
            Self::SEG2 => 2,
            Self::TRI3 => 3,
            Self::QUA4 | Self::TET4 => 4,
            Self::PENTA6 => 6,
            Self::PYRA5 => 5,
            Self::HEX8 => 8,
            Self::SEG3 => 3,
            Self::TRI6 => 6,
            Self::QUA8 => 8,
            Self::TET10 => 10,
            Self::PENTA15 => 15,
            Self::HEX20 => 20,
            Self::QUA9 => 9,
            Self::HEX27 => 27,
        }
    }

    /// Topological dimension (0 = point, 1 = segment, 2 = surface, 3 = volume).
    pub fn topological_dim(self) -> usize {
        match self {
            Self::POI1 => 0,
            Self::SEG2 => 1,
            Self::TRI3 | Self::QUA4 => 2,
            Self::TET4 | Self::PYRA5 | Self::PENTA6 | Self::HEX8 => 3,
            Self::SEG3 => 1,
            Self::TRI6 | Self::QUA8 => 2,
            Self::TET10 | Self::PENTA15 | Self::HEX20 => 3,
            Self::QUA9 => 2,
            Self::HEX27 => 3,
        }
    }

    /// Short name (cast3m-style).
    pub fn name(self) -> &'static str {
        match self {
            Self::POI1 => "POI1",
            Self::SEG2 => "SEG2",
            Self::TRI3 => "TRI3",
            Self::QUA4 => "QUA4",
            Self::TET4 => "TET4",
            Self::PENTA6 => "PENTA6",
            Self::PYRA5 => "PYRA5",
            Self::HEX8 => "HEX8",
            Self::SEG3 => "SEG3",
            Self::TRI6 => "TRI6",
            Self::QUA8 => "QUA8",
            Self::TET10 => "TET10",
            Self::PENTA15 => "PENTA15",
            Self::HEX20 => "HEX20",
            Self::QUA9 => "QUA9",
            Self::HEX27 => "HEX27",
        }
    }

    /// Local-node permutation that **reverses the element's orientation**.
    ///
    /// Applying it to a cell's node list (output node `i` = input node
    /// `reversal_permutation()[i]`) yields the same element with the opposite
    /// orientation: a flipped winding for a surface cell, a reversed traversal
    /// for a segment, a mirrored (negative-Jacobian) volume cell. The result is
    /// always a **valid** cell of the same type — mid-edge, face-center and
    /// body-center nodes of the quadratic variants are carried to their correct
    /// slots, not just the corners.
    ///
    /// Geometrically the permutation is the orientation-reversing reflection
    /// that swaps the first two reference axes, `(ξ, η, ζ) ↦ (η, ξ, ζ)` (for
    /// `SEG*`, the single-axis flip `ξ ↦ -ξ`). `POI1` has no orientation, so its
    /// permutation is the identity (reversal is a no-op).
    ///
    /// Shared building block of [`crate::ops::mesh::invert()`] (applied to
    /// every cell) and [`crate::ops::mesh::orient()`] (applied to the cells a
    /// consistency pass decides to flip).
    pub fn reversal_permutation(self) -> &'static [usize] {
        match self {
            Self::POI1 => &[0],
            Self::SEG2 => &[1, 0],
            Self::SEG3 => &[1, 0, 2],
            Self::TRI3 => &[0, 2, 1],
            Self::TRI6 => &[0, 2, 1, 5, 4, 3],
            Self::QUA4 => &[0, 3, 2, 1],
            Self::QUA8 => &[0, 3, 2, 1, 7, 6, 5, 4],
            Self::QUA9 => &[0, 3, 2, 1, 7, 6, 5, 4, 8],
            Self::TET4 => &[0, 2, 1, 3],
            Self::TET10 => &[0, 2, 1, 3, 6, 5, 4, 7, 9, 8],
            Self::PENTA6 => &[0, 2, 1, 3, 5, 4],
            Self::PENTA15 => &[0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13],
            Self::PYRA5 => &[0, 3, 2, 1, 4],
            Self::HEX8 => &[0, 3, 2, 1, 4, 7, 6, 5],
            Self::HEX20 => &[
                0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
            ],
            Self::HEX27 => &[
                0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17, 22, 23, 20,
                21, 24, 25, 26,
            ],
        }
    }

    /// Parse from a short name (case-insensitive).
    ///
    /// Derived from [`ALL`](Self::ALL) and [`name`](Self::name), so a new
    /// variant is parsable as soon as it is declared — there is no separate
    /// string table to keep in step.
    pub fn from_name(s: &str) -> Option<Self> {
        let upper = s.to_ascii_uppercase();
        Self::ALL.iter().copied().find(|et| et.name() == upper)
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl crate::dump::Dump for ElementType {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        format!(
            "{}: {} node(s)/cell, topo dim {}",
            self.name(),
            self.nodes_per_cell(),
            self.topological_dim()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_metadata() {
        assert_eq!(ElementType::POI1.nodes_per_cell(), 1);
        assert_eq!(ElementType::POI1.topological_dim(), 0);
        assert_eq!(ElementType::SEG2.nodes_per_cell(), 2);
        assert_eq!(ElementType::SEG2.topological_dim(), 1);
        assert_eq!(ElementType::TRI3.topological_dim(), 2);
        assert_eq!(ElementType::QUA4.nodes_per_cell(), 4);
        assert_eq!(ElementType::TET4.topological_dim(), 3);
        assert_eq!(ElementType::PENTA6.nodes_per_cell(), 6);
        assert_eq!(ElementType::PENTA6.topological_dim(), 3);
        assert_eq!(ElementType::HEX8.nodes_per_cell(), 8);
        assert_eq!(ElementType::SEG3.nodes_per_cell(), 3);
        assert_eq!(ElementType::SEG3.topological_dim(), 1);
        assert_eq!(ElementType::TRI6.nodes_per_cell(), 6);
        assert_eq!(ElementType::TRI6.topological_dim(), 2);
        assert_eq!(ElementType::QUA8.nodes_per_cell(), 8);
        assert_eq!(ElementType::TET10.nodes_per_cell(), 10);
        assert_eq!(ElementType::TET10.topological_dim(), 3);
        assert_eq!(ElementType::PENTA15.nodes_per_cell(), 15);
        assert_eq!(ElementType::HEX20.nodes_per_cell(), 20);
        assert_eq!(ElementType::HEX20.topological_dim(), 3);
        assert_eq!(ElementType::QUA9.nodes_per_cell(), 9);
        assert_eq!(ElementType::QUA9.topological_dim(), 2);
        assert_eq!(ElementType::HEX27.nodes_per_cell(), 27);
        assert_eq!(ElementType::HEX27.topological_dim(), 3);
    }

    /// `ALL` is the list everything else iterates over, so it has to be
    /// complete. The exhaustive `match` breaks compilation on a new variant,
    /// and the length assert then forces a look at `ALL` itself.
    #[test]
    fn all_lists_every_variant_exactly_once() {
        for &et in ElementType::ALL {
            match et {
                ElementType::POI1
                | ElementType::SEG2
                | ElementType::TRI3
                | ElementType::QUA4
                | ElementType::TET4
                | ElementType::PENTA6
                | ElementType::PYRA5
                | ElementType::HEX8
                | ElementType::SEG3
                | ElementType::TRI6
                | ElementType::QUA8
                | ElementType::TET10
                | ElementType::PENTA15
                | ElementType::HEX20
                | ElementType::QUA9
                | ElementType::HEX27 => {}
            }
        }
        assert_eq!(ElementType::ALL.len(), 16);
        let mut seen = std::collections::HashSet::new();
        for &et in ElementType::ALL {
            assert!(seen.insert(et), "{et} listed twice in ALL");
        }
    }

    /// Every variant round-trips through its own name, which is what makes
    /// `from_name` derivable rather than a table of its own.
    #[test]
    fn every_variant_parses_back_from_its_name() {
        for &et in ElementType::ALL {
            assert_eq!(ElementType::from_name(et.name()), Some(et), "{et}");
            assert_eq!(
                ElementType::from_name(&et.name().to_ascii_lowercase()),
                Some(et),
                "{et} (lowercase)"
            );
        }
    }

    #[test]
    fn reversal_permutation_is_an_involution_bijection() {
        for &et in ElementType::ALL {
            let p = et.reversal_permutation();
            let npc = et.nodes_per_cell();
            assert_eq!(p.len(), npc, "{et}: permutation length");
            // A bijection of 0..npc (every slot hit exactly once).
            let mut seen = vec![false; npc];
            for &i in p {
                assert!(i < npc, "{et}: index {i} out of range");
                assert!(!seen[i], "{et}: index {i} repeated");
                seen[i] = true;
            }
            // Reflection is its own inverse: reversing twice is the identity.
            for i in 0..npc {
                assert_eq!(p[p[i]], i, "{et}: not an involution at {i}");
            }
        }
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", ElementType::QUA4), "QUA4");
        assert_eq!(ElementType::from_name("tri3"), Some(ElementType::TRI3));
        assert_eq!(ElementType::from_name("HEX8"), Some(ElementType::HEX8));
        assert_eq!(ElementType::from_name("tri6"), Some(ElementType::TRI6));
        assert_eq!(
            ElementType::from_name("PENTA15"),
            Some(ElementType::PENTA15)
        );
        assert_eq!(format!("{}", ElementType::HEX20), "HEX20");
        assert_eq!(ElementType::from_name("unknown"), None);
    }
}
