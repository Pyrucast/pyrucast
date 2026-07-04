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
//! | `HEX8` | `ξ, η, ζ ∈ [-1, +1]` | bottom face CCW then top face CCW: `(-1,-1,-1)`, `(1,-1,-1)`, `(1,1,-1)`, `(-1,1,-1)`, `(-1,-1,1)`, `(1,-1,1)`, `(1,1,1)`, `(-1,1,1)` |
//!
//! These conventions are compatible with the orientations already enforced
//! elsewhere in the codebase (CCW filling in
//! [`crate::ops::mesher::fill_surface()`], HEX8 node ordering in
//! [`crate::ops::mesher::extrude()`]).

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
    /// 8-node hexahedron. Reference: `ξ, η, ζ ∈ [-1, +1]`. Local order:
    /// bottom face CCW (nodes 0..3), then top face CCW (nodes 4..7),
    /// i.e. `(-1,-1,-1)`, `(1,-1,-1)`, `(1,1,-1)`, `(-1,1,-1)`,
    /// `(-1,-1,1)`, `(1,-1,1)`, `(1,1,1)`, `(-1,1,1)`.
    HEX8,
}

impl ElementType {
    /// Number of nodes per cell for this element type.
    pub fn nodes_per_cell(self) -> usize {
        match self {
            Self::POI1 => 1,
            Self::SEG2 => 2,
            Self::TRI3 => 3,
            Self::QUA4 | Self::TET4 => 4,
            Self::PENTA6 => 6,
            Self::HEX8 => 8,
        }
    }

    /// Topological dimension (0 = point, 1 = segment, 2 = surface, 3 = volume).
    pub fn topological_dim(self) -> usize {
        match self {
            Self::POI1 => 0,
            Self::SEG2 => 1,
            Self::TRI3 | Self::QUA4 => 2,
            Self::TET4 | Self::PENTA6 | Self::HEX8 => 3,
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
            Self::HEX8 => "HEX8",
        }
    }

    /// Parse from a short name (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "POI1" => Some(Self::POI1),
            "SEG2" => Some(Self::SEG2),
            "TRI3" => Some(Self::TRI3),
            "QUA4" => Some(Self::QUA4),
            "TET4" => Some(Self::TET4),
            "PENTA6" => Some(Self::PENTA6),
            "HEX8" => Some(Self::HEX8),
            _ => None,
        }
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
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", ElementType::QUA4), "QUA4");
        assert_eq!(ElementType::from_name("tri3"), Some(ElementType::TRI3));
        assert_eq!(ElementType::from_name("HEX8"), Some(ElementType::HEX8));
        assert_eq!(ElementType::from_name("unknown"), None);
    }
}
