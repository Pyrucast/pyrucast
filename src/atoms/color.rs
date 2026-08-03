//! Small RGB colour type shared across pyrucast (mesh appearance, fields…).
//!
//! Kept deliberately minimal: 8-bit per channel, no alpha. The visualization
//! layer applies a fixed face opacity when rendering; the colour itself
//! belongs to the data and stays opaque.
//!
//! # Example
//!
//! ```
//! use pyrucast::atoms::RgbColor;
//!
//! let red = RgbColor::new(220, 60, 60);
//! assert_eq!((red.r, red.g, red.b), (220, 60, 60));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// 8-bit RGB colour (no alpha).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RgbColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl RgbColor {
    /// Build a colour from explicit RGB components.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Default face colour (a light blue).
    pub const fn default_face() -> Self {
        Self {
            r: 180,
            g: 200,
            b: 230,
        }
    }
}

impl Default for RgbColor {
    fn default() -> Self {
        Self::default_face()
    }
}

impl fmt::Debug for RgbColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RgbColor({}, {}, {})", self.r, self.g, self.b)
    }
}

impl fmt::Display for RgbColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

impl crate::dump::Dump for RgbColor {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        format!("{self} (r={}, g={}, b={})", self.r, self.g, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_light_blue() {
        let d = RgbColor::default();
        assert_eq!(d, RgbColor::new(180, 200, 230));
    }

    #[test]
    fn display_uses_hex_uppercase() {
        let c = RgbColor::new(255, 0, 16);
        assert_eq!(format!("{}", c), "#FF0010");
    }

    #[test]
    fn debug_components() {
        let c = RgbColor::new(1, 2, 3);
        assert_eq!(format!("{:?}", c), "RgbColor(1, 2, 3)");
    }

    #[test]
    fn serde_roundtrip() {
        use crate::persist::Persist;
        let c = RgbColor::new(10, 20, 30);
        let bytes = c.to_bytes().unwrap();
        let back = RgbColor::from_bytes(&bytes).unwrap();
        assert_eq!(c, back);
    }
}
