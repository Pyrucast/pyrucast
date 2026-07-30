//! The advancing front: a set of closed, oriented loops of slots.
//!
//! A **slot** is a position on the front; a **vertex** is a point of the mesh
//! being built. They are deliberately different things, because a single
//! vertex can sit on the front twice — that is exactly what a pinch point is,
//! and it is how a loop splits in two. Keeping the two notions apart is what
//! makes [`Front::merge`] a single operation instead of a pile of cases.
//!
//! Every loop is oriented so that **material lies to the left** of the walk
//! `prev → cur → next`. A domain's outer boundary is counter-clockwise and its
//! holes are clockwise, which already satisfies this, so loops enter the front
//! exactly as the contour gives them.
//!
//! ## The one interesting operation
//!
//! [`merge`](Front::merge) identifies two non-adjacent slots `a` and `b` with a
//! single vertex. Writing the ring as `… a⁻ a a⁺ … b⁻ b b⁺ …`, it drops `a` and
//! `b` and inserts two fresh slots on the same new vertex: `m₁` between `b⁻`
//! and `a⁺`, and `m₂` between `a⁻` and `b⁺`. Then:
//!
//! - if `a` and `b` were on the **same** loop, the loop **splits** into
//!   `a⁺ … b⁻ m₁` and `b⁺ … a⁻ m₂` — this is how a concave region divides;
//! - if they were on **different** loops, the two loops **join** into one —
//!   this is how a hole is absorbed, with no special case for holes anywhere
//!   in the paver.
//!
//! Same relinking, opposite effects, decided entirely by where the slots were.

use std::collections::HashSet;

#[derive(Clone, Copy)]
struct Slot {
    v: u32,
    prev: u32,
    next: u32,
    alive: bool,
}

/// The advancing front. Slots are addressed by stable indices and recycled
/// through a free list, so a slot index never shifts under the caller.
#[derive(Default)]
pub struct Front {
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl Front {
    pub fn new() -> Self {
        Front::default()
    }

    fn alloc(&mut self, v: u32) -> u32 {
        if let Some(i) = self.free.pop() {
            self.slots[i as usize] = Slot {
                v,
                prev: i,
                next: i,
                alive: true,
            };
            i
        } else {
            let i = self.slots.len() as u32;
            self.slots.push(Slot {
                v,
                prev: i,
                next: i,
                alive: true,
            });
            i
        }
    }

    #[inline]
    fn link(&mut self, a: u32, b: u32) {
        self.slots[a as usize].next = b;
        self.slots[b as usize].prev = a;
    }

    /// Add a closed loop over `verts`, in front order. Returns a slot of it.
    pub fn add_loop(&mut self, verts: &[u32]) -> u32 {
        debug_assert!(verts.len() >= 3);
        let slots: Vec<u32> = verts.iter().map(|&v| self.alloc(v)).collect();
        let n = slots.len();
        for i in 0..n {
            self.link(slots[i], slots[(i + 1) % n]);
        }
        slots[0]
    }

    #[inline]
    pub fn vertex(&self, s: u32) -> u32 {
        self.slots[s as usize].v
    }

    #[inline]
    pub fn next(&self, s: u32) -> u32 {
        self.slots[s as usize].next
    }

    #[inline]
    pub fn prev(&self, s: u32) -> u32 {
        self.slots[s as usize].prev
    }

    #[inline]
    pub fn is_alive(&self, s: u32) -> bool {
        self.slots[s as usize].alive
    }

    /// Walk the loop containing `rep`, starting there.
    pub fn loop_slots(&self, rep: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut s = rep;
        loop {
            out.push(s);
            s = self.next(s);
            if s == rep || out.len() > self.slots.len() {
                break;
            }
        }
        out
    }

    /// Number of slots in the loop containing `rep`.
    pub fn loop_len(&self, rep: u32) -> usize {
        let mut n = 0usize;
        let mut s = rep;
        loop {
            n += 1;
            s = self.next(s);
            if s == rep || n > self.slots.len() {
                break;
            }
        }
        n
    }

    /// Every live slot, in index order — the deterministic iteration order
    /// used to rebuild spatial indices.
    pub fn live_slots(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.slots.len() as u32).filter(|&i| self.slots[i as usize].alive)
    }

    /// Retire a whole loop without producing anything (a degenerate ring).
    pub fn kill_loop(&mut self, rep: u32) {
        for s in self.loop_slots(rep) {
            self.slots[s as usize].alive = false;
            self.free.push(s);
        }
    }

    /// Replace the loop containing `rep` by a fresh ring over `verts`, in
    /// order. Used by the row advance, which rebuilds a whole loop at once.
    pub fn relink_loop(&mut self, rep: u32, verts: &[u32]) -> u32 {
        self.kill_loop(rep);
        self.add_loop(verts)
    }

    /// Collapse the front edge `(s, s.next)` onto the single vertex `v`.
    /// The loop loses one slot. Returns the surviving slot.
    pub fn collapse_edge(&mut self, s: u32, v: u32) -> u32 {
        let b = self.next(s);
        debug_assert_ne!(b, s);
        let after = self.next(b);
        self.slots[b as usize].alive = false;
        self.free.push(b);
        self.slots[s as usize].v = v;
        self.link(s, after);
        s
    }

    /// Identify the two **non-adjacent** slots `a` and `b` with vertex `v`.
    ///
    /// Splits one loop in two when `a` and `b` share a loop, and joins two
    /// loops into one when they do not. Returns the two fresh slots carrying
    /// `v`, one per resulting ring — in the split case they represent the two
    /// new loops, in the join case both lie on the single result.
    pub fn merge(&mut self, a: u32, b: u32, v: u32) -> (u32, u32) {
        debug_assert_ne!(a, b);
        debug_assert_ne!(self.next(a), b);
        debug_assert_ne!(self.next(b), a);
        let (ap, an) = (self.prev(a), self.next(a));
        let (bp, bn) = (self.prev(b), self.next(b));

        self.slots[a as usize].alive = false;
        self.slots[b as usize].alive = false;
        self.free.push(a);
        self.free.push(b);

        let m1 = self.alloc(v);
        let m2 = self.alloc(v);
        self.link(bp, m1);
        self.link(m1, an);
        self.link(ap, m2);
        self.link(m2, bn);
        (m1, m2)
    }

    /// Are `a` and `b` on the same loop?
    pub fn same_loop(&self, a: u32, b: u32) -> bool {
        let mut s = a;
        loop {
            if s == b {
                return true;
            }
            s = self.next(s);
            if s == a {
                return false;
            }
        }
    }

    /// One representative slot per live loop, in ascending slot order.
    pub fn loop_representatives(&self) -> Vec<u32> {
        let mut seen: HashSet<u32> = HashSet::new();
        let mut reps = Vec::new();
        for s in self.live_slots() {
            if seen.contains(&s) {
                continue;
            }
            reps.push(s);
            for t in self.loop_slots(s) {
                seen.insert(t);
            }
        }
        reps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verts(f: &Front, rep: u32) -> Vec<u32> {
        f.loop_slots(rep).iter().map(|&s| f.vertex(s)).collect()
    }

    #[test]
    fn a_loop_walks_in_order_and_closes() {
        let mut f = Front::new();
        let r = f.add_loop(&[10, 11, 12, 13]);
        assert_eq!(f.loop_len(r), 4);
        assert_eq!(verts(&f, r), vec![10, 11, 12, 13]);
        assert_eq!(f.vertex(f.prev(r)), 13);
    }

    #[test]
    fn merging_within_one_loop_splits_it() {
        let mut f = Front::new();
        // 0 1 2 3 4 5, merge slots at vertices 1 and 4.
        let r = f.add_loop(&[0, 1, 2, 3, 4, 5]);
        let slots = f.loop_slots(r);
        let (a, b) = (slots[1], slots[4]);
        assert!(f.same_loop(a, b));
        let (m1, m2) = f.merge(a, b, 99);

        // b⁻=3 → m₁ → a⁺=2 gives the ring 2 3 99; a⁻=0 → m₂ → b⁺=5 gives 5 0 99.
        let l1 = verts(&f, m1);
        let l2 = verts(&f, m2);
        assert_eq!(l1.len() + l2.len(), 6 + 2 - 2);
        assert!(l1.contains(&99) && l2.contains(&99));
        assert!(!f.same_loop(m1, m2), "the loop must have split in two");
        let mut all: Vec<u32> = l1.iter().chain(l2.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 2, 3, 5, 99, 99]);
    }

    #[test]
    fn merging_across_two_loops_joins_them() {
        let mut f = Front::new();
        let r1 = f.add_loop(&[0, 1, 2, 3]);
        let r2 = f.add_loop(&[10, 11, 12, 13]);
        assert_eq!(f.loop_representatives().len(), 2);
        let a = f.loop_slots(r1)[0];
        let b = f.loop_slots(r2)[0];
        assert!(!f.same_loop(a, b));

        let (m1, m2) = f.merge(a, b, 99);
        assert!(f.same_loop(m1, m2), "the two loops must have joined");
        assert_eq!(f.loop_len(m1), 8);
        assert_eq!(f.loop_representatives().len(), 1);
    }

    #[test]
    fn collapsing_an_edge_shortens_the_loop() {
        let mut f = Front::new();
        let r = f.add_loop(&[0, 1, 2, 3, 4]);
        let s = f.loop_slots(r)[1];
        let kept = f.collapse_edge(s, 99);
        assert_eq!(f.loop_len(kept), 4);
        assert_eq!(verts(&f, kept), vec![99, 3, 4, 0]);
    }

    #[test]
    fn slots_are_recycled_but_indices_stay_stable() {
        let mut f = Front::new();
        let r = f.add_loop(&[0, 1, 2, 3]);
        let kept = f.loop_slots(r)[0];
        f.collapse_edge(kept, 42);
        // The recycled slot comes back for the next loop; the surviving slot
        // is untouched.
        let r2 = f.add_loop(&[7, 8, 9]);
        assert_eq!(f.vertex(kept), 42);
        assert_eq!(f.loop_len(r2), 3);
    }
}
