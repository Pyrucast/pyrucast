//! Growing a volume mesh inward from a closed surface — the machinery behind
//! [`pave_volume`](super::pave_volume).
//!
//! - [`shell`] reads and validates the closed envelope the front starts from;
//! - [`front`] is the advancing front itself: local step, seams, and the cells
//!   left behind it;
//! - [`smooth`] relaxes what the front laid down, under a validity guard.

pub mod front;
pub mod shell;
pub mod smooth;
