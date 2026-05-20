//! Finite-element types supported by pyrucast.
//!
//! The list is deliberately small; adding a new element type means adding
//! a variant to [`ElementType`] and completing the metadata functions
//! ([`nodes_per_cell`](ElementType::nodes_per_cell),
//! [`topological_dim`](ElementType::topological_dim),
//! [`name`](ElementType::name)).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported element types. Names follow the cast3m convention.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ElementType {
    /// 1 node. A POI1 submesh is effectively a list of nodes.
    POI1,
    /// 2-node segment (1D).
    SEG2,
    /// 3-node triangle (2D).
    TRI3,
    /// 4-node quadrangle (2D).
    QUA4,
    /// 4-node tetrahedron (3D).
    TET4,
    /// 8-node hexahedron (3D).
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
            Self::HEX8 => 8,
        }
    }

    /// Topological dimension (0 = point, 1 = segment, 2 = surface, 3 = volume).
    pub fn topological_dim(self) -> usize {
        match self {
            Self::POI1 => 0,
            Self::SEG2 => 1,
            Self::TRI3 | Self::QUA4 => 2,
            Self::TET4 | Self::HEX8 => 3,
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
