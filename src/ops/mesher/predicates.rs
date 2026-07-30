//! Exact geometric predicates: [`orient2d`] in 2-D, [`orient3d`] and
//! [`insphere`] in 3-D.
//!
//! A mesher is a program that asks the same two questions millions of times
//! — *on which side of this line/plane does that point lie?* and *is that
//! point inside this circumcircle/sphere?* — and builds a data structure out
//! of the answers. The answers must be **mutually
//! consistent**: if `orient3d(a, b, c, d)` says "above" then
//! `orient3d(b, a, c, d)` must say "below", and a point cannot be at once
//! inside a tetrahedron and outside all four of its faces. Plain `f64`
//! determinants break that consistency near degeneracies, and a single
//! broken answer corrupts the adjacency graph — which is how a Delaunay
//! kernel ends up looping forever or emitting overlapping cells.
//!
//! Neither a tolerance nor a coordinate jitter fixes this. They make the
//! predicate *usually* right, not *self-consistent*; the failure just moves
//! somewhere less predictable. What does fix it is computing the sign
//! exactly, which is what this module does, following Shewchuk's
//! floating-point expansion technique:
//!
//! 1. Evaluate the determinant in `f64` and compare it against a rigorous
//!    forward error bound. If it is larger, the sign is already certain —
//!    this is the path taken by essentially every call on generic input.
//! 2. Otherwise re-evaluate with **exact** arithmetic, representing each
//!    intermediate value as a non-overlapping sum of `f64` components (an
//!    *expansion*). Every operation used here — `two_sum`, `two_product`
//!    and the expansion sum/scale — is error-free, so the final sign is the
//!    true sign, including an exact `0` for genuinely degenerate input.
//!
//! Exactly-degenerate configurations (a cube's cospherical corners, three
//! collinear nodes) always take the slow path, but such inputs are small and
//! structured; large meshes born of curved geometry take it almost never.
//!
//! The predicates take plain `[f64; 3]` arrays, so they stay independent of
//! the container layer.
//!
//! # Example
//!
//! ```
//! use pyrucast::ops::mesher::tetrahedralization::predicates::{insphere, orient3d};
//!
//! let a = [0.0, 0.0, 0.0];
//! let b = [1.0, 0.0, 0.0];
//! let c = [1.0, 1.0, 0.0];
//! let d = [0.0, 0.0, 1.0];
//! assert!(orient3d(&a, &b, &c, &d) > 0.0);
//!
//! // (0, 1, 0) is another corner of the same unit cube: exactly cospherical.
//! assert_eq!(insphere(&a, &b, &c, &d, &[0.0, 1.0, 0.0]), 0.0);
//! // The cube's centre is the circumcentre, so strictly inside.
//! assert!(insphere(&a, &b, &c, &d, &[0.5, 0.5, 0.5]) > 0.0);
//! ```

// `f64` has a 53-bit significand, so the unit roundoff is 2^-53.
const EPSILON: f64 = 1.110_223_024_625_156_5e-16;
// 2^27 + 1 — splits a f64 into two halves of at most 26 significant bits,
// whose pairwise products are then exact.
const SPLITTER: f64 = 134_217_729.0;
// Relative error bounds of the `f64` estimates, as derived by Shewchuk for
// exactly the two expressions evaluated below. Multiplied by the
// "permanent" (the same determinant with every term made positive), they
// bound the absolute error of the estimate.
const ORIENT2D_ERRBOUND: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const ORIENT3D_ERRBOUND: f64 = (7.0 + 56.0 * EPSILON) * EPSILON;
const INSPHERE_ERRBOUND: f64 = (16.0 + 224.0 * EPSILON) * EPSILON;

// ─── Error-free transformations ─────────────────────────────────────────
//
// Each returns `(hi, lo)` with `hi` the rounded result and `lo` the exact
// rounding error, so that `hi + lo` is the exact value of the operation.

/// Exact sum, valid only when `|a| >= |b|`.
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    (x, b - b_virtual)
}

/// Exact sum, valid for any operands.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    let a_virtual = x - b_virtual;
    (x, (a - a_virtual) + (b - b_virtual))
}

/// Exact difference, valid for any operands.
#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let b_virtual = a - x;
    let a_virtual = x + b_virtual;
    (x, (a - a_virtual) + (b_virtual - b))
}

/// Split a `f64` into two halves of at most 26 significant bits.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let a_big = c - a;
    let a_hi = c - a_big;
    (a_hi, a - a_hi)
}

/// Exact product `a · b`, where `a` has already been [`split`] into
/// `(a_hi, a_lo)`.
#[inline]
fn two_product_presplit(a: f64, b: f64, a_hi: f64, a_lo: f64) -> (f64, f64) {
    let x = a * b;
    let (b_hi, b_lo) = split(b);
    let err1 = x - (a_hi * b_hi);
    let err2 = err1 - (a_lo * b_hi);
    let err3 = err2 - (a_hi * b_lo);
    (x, (a_lo * b_lo) - err3)
}

// ─── Expansion arithmetic ───────────────────────────────────────────────
//
// An *expansion* is a `Vec<f64>` whose components are non-overlapping and
// sorted by increasing magnitude; the value it represents is their exact
// sum. Zero components are eliminated as they appear, which is what keeps
// the representations short in practice — the worst-case lengths implied by
// the formulas below are never approached on real input.

/// The exact difference `a - b`, as an expansion.
#[inline]
fn diff(a: f64, b: f64) -> Vec<f64> {
    let (hi, lo) = two_diff(a, b);
    if lo == 0.0 {
        vec![hi]
    } else {
        vec![lo, hi]
    }
}

/// Exact sum of two expansions (Shewchuk's `fast_expansion_sum_zeroelim`):
/// a merge of the two component sequences that accumulates a running carry.
fn expansion_sum(e: &[f64], f: &[f64]) -> Vec<f64> {
    if e.is_empty() {
        return f.to_vec();
    }
    if f.is_empty() {
        return e.to_vec();
    }
    let mut h: Vec<f64> = Vec::with_capacity(e.len() + f.len());
    let (mut ei, mut fi) = (0usize, 0usize);

    // `(f > e) == (f > -e)` is true exactly when `|f| > |e|`; the branch
    // therefore always consumes the component smaller in magnitude, so the
    // running carry `q` absorbs the terms in increasing order.
    let mut q = {
        let (enow, fnow) = (e[0], f[0]);
        if (fnow > enow) == (fnow > -enow) {
            ei += 1;
            enow
        } else {
            fi += 1;
            fnow
        }
    };

    if ei < e.len() && fi < f.len() {
        let (enow, fnow) = (e[ei], f[fi]);
        // Only the first merge step comes with the `|a| >= |b|` guarantee,
        // hence the cheaper `fast_two_sum` here and `two_sum` below.
        let (sum, err) = if (fnow > enow) == (fnow > -enow) {
            ei += 1;
            fast_two_sum(enow, q)
        } else {
            fi += 1;
            fast_two_sum(fnow, q)
        };
        q = sum;
        if err != 0.0 {
            h.push(err);
        }

        while ei < e.len() && fi < f.len() {
            let (enow, fnow) = (e[ei], f[fi]);
            let (sum, err) = if (fnow > enow) == (fnow > -enow) {
                ei += 1;
                two_sum(q, enow)
            } else {
                fi += 1;
                two_sum(q, fnow)
            };
            q = sum;
            if err != 0.0 {
                h.push(err);
            }
        }
    }

    for &enow in &e[ei..] {
        let (sum, err) = two_sum(q, enow);
        q = sum;
        if err != 0.0 {
            h.push(err);
        }
    }
    for &fnow in &f[fi..] {
        let (sum, err) = two_sum(q, fnow);
        q = sum;
        if err != 0.0 {
            h.push(err);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

/// Exact product of an expansion by a single `f64`
/// (Shewchuk's `scale_expansion_zeroelim`).
fn expansion_scale(e: &[f64], b: f64) -> Vec<f64> {
    if e.is_empty() || b == 0.0 {
        return vec![0.0];
    }
    let mut h: Vec<f64> = Vec::with_capacity(2 * e.len());
    let (b_hi, b_lo) = split(b);

    let (mut q, err) = two_product_presplit(b, e[0], b_hi, b_lo);
    if err != 0.0 {
        h.push(err);
    }
    for &ei in &e[1..] {
        let (product_hi, product_lo) = two_product_presplit(b, ei, b_hi, b_lo);
        let (sum, err_lo) = two_sum(q, product_lo);
        if err_lo != 0.0 {
            h.push(err_lo);
        }
        let (carry, err_hi) = fast_two_sum(product_hi, sum);
        q = carry;
        if err_hi != 0.0 {
            h.push(err_hi);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

/// Exact product of two expansions: distribute over the right operand's
/// components, then sum the partial products exactly.
fn expansion_mul(e: &[f64], f: &[f64]) -> Vec<f64> {
    let mut acc: Vec<f64> = Vec::new();
    for &fi in f {
        let partial = expansion_scale(e, fi);
        acc = if acc.is_empty() {
            partial
        } else {
            expansion_sum(&acc, &partial)
        };
    }
    if acc.is_empty() {
        acc.push(0.0);
    }
    acc
}

/// Exact negation.
fn expansion_neg(e: &[f64]) -> Vec<f64> {
    e.iter().map(|x| -x).collect()
}

/// Exact difference of two expansions.
fn expansion_diff(e: &[f64], f: &[f64]) -> Vec<f64> {
    expansion_sum(e, &expansion_neg(f))
}

/// A `f64` carrying the sign of the expansion.
///
/// Components are non-overlapping and increasing in magnitude, so the last
/// one dominates the sum and fixes its sign; the plain sum is therefore a
/// sign-preserving summary of the exact value.
fn expansion_estimate(e: &[f64]) -> f64 {
    e.iter().sum()
}

/// `a·b - c·d` for expansions — the 2×2 minor both predicates are built on.
fn minor2(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    expansion_diff(&expansion_mul(a, b), &expansion_mul(c, d))
}

// ─── orient2d ───────────────────────────────────────────────────────────

/// Planar orientation test: `> 0` when `pa`, `pb`, `pc` run
/// counter-clockwise, `< 0` clockwise, and exactly `0.0` when they are
/// collinear.
///
/// The value is twice the signed area of the triangle, up to rounding; only
/// the **sign** is guaranteed exact. This is the usual convention, the one
/// [`crate::ops::mesher::triangulation::signed_area`] already follows.
///
/// # Example
///
/// ```
/// use pyrucast::ops::mesher::tetrahedralization::predicates::orient2d;
///
/// assert!(orient2d(&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]) > 0.0);
/// assert!(orient2d(&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0]) < 0.0);
/// assert_eq!(orient2d(&[0.0, 0.0], &[1.0, 2.0], &[2.0, 4.0]), 0.0);
/// ```
pub fn orient2d(pa: &[f64; 2], pb: &[f64; 2], pc: &[f64; 2]) -> f64 {
    let detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    let detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    let det = detleft - detright;

    // Only a difference of like-signed products can cancel; anything else
    // already has its sign settled by the estimate.
    let detsum = if detleft > 0.0 {
        if detright <= 0.0 {
            return det;
        }
        detleft + detright
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return det;
        }
        -detleft - detright
    } else {
        return det;
    };
    let errbound = ORIENT2D_ERRBOUND * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }
    orient2d_exact(pa, pb, pc)
}

/// The exact fallback of [`orient2d`], in expansion arithmetic.
fn orient2d_exact(pa: &[f64; 2], pb: &[f64; 2], pc: &[f64; 2]) -> f64 {
    let acx = diff(pa[0], pc[0]);
    let acy = diff(pa[1], pc[1]);
    let bcx = diff(pb[0], pc[0]);
    let bcy = diff(pb[1], pc[1]);
    expansion_estimate(&minor2(&acx, &bcy, &acy, &bcx))
}

/// Whether `pa`, `pb`, `pc` are exactly collinear in 3-D.
///
/// Three points are collinear when the cross product of the two edge
/// vectors vanishes, i.e. when all three of its components — each a planar
/// orientation test on one coordinate projection — are exactly zero.
///
/// # Example
///
/// ```
/// use pyrucast::ops::mesher::tetrahedralization::predicates::collinear3d;
///
/// assert!(collinear3d(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]));
/// assert!(!collinear3d(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], &[2.0, 4.0, 7.0]));
/// // A repeated point degenerates to a segment, hence collinear.
/// assert!(collinear3d(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], &[9.0, 0.0, 0.0]));
/// ```
pub fn collinear3d(pa: &[f64; 3], pb: &[f64; 3], pc: &[f64; 3]) -> bool {
    const PROJECTIONS: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];
    PROJECTIONS
        .iter()
        .all(|&(i, j)| orient2d(&[pa[i], pa[j]], &[pb[i], pb[j]], &[pc[i], pc[j]]) == 0.0)
}

// ─── orient3d ───────────────────────────────────────────────────────────

/// Signed volume test: `> 0` when `pd` lies **above** the plane of `pa`,
/// `pb`, `pc` — "above" meaning the side from which `pa`, `pb`, `pc` appear
/// counter-clockwise. Returns exactly `0.0` if and only if the four points
/// are coplanar.
///
/// The value is six times the signed volume of the tetrahedron, up to
/// rounding; only the **sign** is guaranteed exact.
///
/// This is the sign convention of
/// [`ElementType::TET4`](crate::containers::mesh::ElementType::TET4), whose
/// face `0-1-2` is counter-clockwise seen from node 3: a well-formed `TET4`
/// has `orient3d(n0, n1, n2, n3) > 0`. Note that it is the *opposite* of
/// Shewchuk's published convention, whose formula is evaluated internally
/// and negated.
///
/// # Example
///
/// ```
/// use pyrucast::ops::mesher::tetrahedralization::predicates::orient3d;
///
/// let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
/// assert!(orient3d(&a, &b, &c, &[0.0, 0.0, 1.0]) > 0.0);
/// assert!(orient3d(&a, &b, &c, &[0.0, 0.0, -1.0]) < 0.0);
/// assert_eq!(orient3d(&a, &b, &c, &[3.0, 5.0, 0.0]), 0.0);
/// ```
pub fn orient3d(pa: &[f64; 3], pb: &[f64; 3], pc: &[f64; 3], pd: &[f64; 3]) -> f64 {
    let (adx, ady, adz) = (pa[0] - pd[0], pa[1] - pd[1], pa[2] - pd[2]);
    let (bdx, bdy, bdz) = (pb[0] - pd[0], pb[1] - pd[1], pb[2] - pd[2]);
    let (cdx, cdy, cdz) = (pc[0] - pd[0], pc[1] - pd[1], pc[2] - pd[2]);

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * adz.abs()
        + (cdxady.abs() + adxcdy.abs()) * bdz.abs()
        + (adxbdy.abs() + bdxady.abs()) * cdz.abs();
    let errbound = ORIENT3D_ERRBOUND * permanent;
    if det > errbound || -det > errbound {
        // `det` is the determinant of the rows (pa−pd, pb−pd, pc−pd), which
        // is the negative of the signed volume of (pa, pb, pc, pd).
        return -det;
    }
    -orient3d_exact(pa, pb, pc, pd)
}

/// The exact fallback of [`orient3d`], in expansion arithmetic and in
/// Shewchuk's sign convention (the caller negates).
fn orient3d_exact(pa: &[f64; 3], pb: &[f64; 3], pc: &[f64; 3], pd: &[f64; 3]) -> f64 {
    let (adx, ady, adz) = (diff(pa[0], pd[0]), diff(pa[1], pd[1]), diff(pa[2], pd[2]));
    let (bdx, bdy, bdz) = (diff(pb[0], pd[0]), diff(pb[1], pd[1]), diff(pb[2], pd[2]));
    let (cdx, cdy, cdz) = (diff(pc[0], pd[0]), diff(pc[1], pd[1]), diff(pc[2], pd[2]));

    // The same expression as the estimate above, term for term.
    let t0 = expansion_mul(&adz, &minor2(&bdx, &cdy, &cdx, &bdy));
    let t1 = expansion_mul(&bdz, &minor2(&cdx, &ady, &adx, &cdy));
    let t2 = expansion_mul(&cdz, &minor2(&adx, &bdy, &bdx, &ady));
    expansion_estimate(&expansion_sum(&expansion_sum(&t0, &t1), &t2))
}

// ─── insphere ───────────────────────────────────────────────────────────

/// Circumsphere test: `> 0` when `pe` lies strictly **inside** the sphere
/// through `pa`, `pb`, `pc`, `pd`, `< 0` when outside, and exactly `0.0`
/// when the five points are cospherical.
///
/// `pa`, `pb`, `pc`, `pd` must be positively oriented in the sense of
/// [`orient3d`] — the sign flips otherwise. Every tetrahedron the mesher
/// builds satisfies that by construction.
///
/// # Example
///
/// ```
/// use pyrucast::ops::mesher::tetrahedralization::predicates::{insphere, orient3d};
///
/// let a = [0.0, 0.0, 0.0];
/// let b = [1.0, 0.0, 0.0];
/// let c = [0.0, 1.0, 0.0];
/// let d = [0.0, 0.0, 1.0];
/// assert!(orient3d(&a, &b, &c, &d) > 0.0);
/// assert!(insphere(&a, &b, &c, &d, &[0.2, 0.2, 0.2]) > 0.0);
/// assert!(insphere(&a, &b, &c, &d, &[9.0, 9.0, 9.0]) < 0.0);
/// ```
pub fn insphere(pa: &[f64; 3], pb: &[f64; 3], pc: &[f64; 3], pd: &[f64; 3], pe: &[f64; 3]) -> f64 {
    let (aex, aey, aez) = (pa[0] - pe[0], pa[1] - pe[1], pa[2] - pe[2]);
    let (bex, bey, bez) = (pb[0] - pe[0], pb[1] - pe[1], pb[2] - pe[2]);
    let (cex, cey, cez) = (pc[0] - pe[0], pc[1] - pe[1], pc[2] - pe[2]);
    let (dex, dey, dez) = (pd[0] - pe[0], pd[1] - pe[1], pd[2] - pe[2]);

    let aexbey = aex * bey;
    let bexaey = bex * aey;
    let ab = aexbey - bexaey;
    let bexcey = bex * cey;
    let cexbey = cex * bey;
    let bc = bexcey - cexbey;
    let cexdey = cex * dey;
    let dexcey = dex * cey;
    let cd = cexdey - dexcey;
    let dexaey = dex * aey;
    let aexdey = aex * dey;
    let da = dexaey - aexdey;
    let aexcey = aex * cey;
    let cexaey = cex * aey;
    let ac = aexcey - cexaey;
    let bexdey = bex * dey;
    let dexbey = dex * bey;
    let bd = bexdey - dexbey;

    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;

    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    let aezplus = aez.abs();
    let bezplus = bez.abs();
    let cezplus = cez.abs();
    let dezplus = dez.abs();
    let aexbeyplus = aexbey.abs();
    let bexaeyplus = bexaey.abs();
    let bexceyplus = bexcey.abs();
    let cexbeyplus = cexbey.abs();
    let cexdeyplus = cexdey.abs();
    let dexceyplus = dexcey.abs();
    let dexaeyplus = dexaey.abs();
    let aexdeyplus = aexdey.abs();
    let aexceyplus = aexcey.abs();
    let cexaeyplus = cexaey.abs();
    let bexdeyplus = bexdey.abs();
    let dexbeyplus = dexbey.abs();
    let permanent = ((cexdeyplus + dexceyplus) * bezplus
        + (dexbeyplus + bexdeyplus) * cezplus
        + (bexceyplus + cexbeyplus) * dezplus)
        * alift
        + ((dexaeyplus + aexdeyplus) * cezplus
            + (aexceyplus + cexaeyplus) * dezplus
            + (cexdeyplus + dexceyplus) * aezplus)
            * blift
        + ((aexbeyplus + bexaeyplus) * dezplus
            + (bexdeyplus + dexbeyplus) * aezplus
            + (dexaeyplus + aexdeyplus) * bezplus)
            * clift
        + ((bexceyplus + cexbeyplus) * aezplus
            + (cexaeyplus + aexceyplus) * bezplus
            + (aexbeyplus + bexaeyplus) * cezplus)
            * dlift;
    let errbound = INSPHERE_ERRBOUND * permanent;
    if det > errbound || -det > errbound {
        // Shewchuk's determinant pairs with his orientation convention,
        // which is the negative of ours; negate to match `orient3d`.
        return -det;
    }
    -insphere_exact(pa, pb, pc, pd, pe)
}

/// The exact fallback of [`insphere`], in expansion arithmetic and in
/// Shewchuk's sign convention (the caller negates).
fn insphere_exact(
    pa: &[f64; 3],
    pb: &[f64; 3],
    pc: &[f64; 3],
    pd: &[f64; 3],
    pe: &[f64; 3],
) -> f64 {
    let (aex, aey, aez) = (diff(pa[0], pe[0]), diff(pa[1], pe[1]), diff(pa[2], pe[2]));
    let (bex, bey, bez) = (diff(pb[0], pe[0]), diff(pb[1], pe[1]), diff(pb[2], pe[2]));
    let (cex, cey, cez) = (diff(pc[0], pe[0]), diff(pc[1], pe[1]), diff(pc[2], pe[2]));
    let (dex, dey, dez) = (diff(pd[0], pe[0]), diff(pd[1], pe[1]), diff(pd[2], pe[2]));

    // The same expression as the estimate above, term for term.
    let ab = minor2(&aex, &bey, &bex, &aey);
    let bc = minor2(&bex, &cey, &cex, &bey);
    let cd = minor2(&cex, &dey, &dex, &cey);
    let da = minor2(&dex, &aey, &aex, &dey);
    let ac = minor2(&aex, &cey, &cex, &aey);
    let bd = minor2(&bex, &dey, &dex, &bey);

    let abc = expansion_sum(
        &expansion_diff(&expansion_mul(&aez, &bc), &expansion_mul(&bez, &ac)),
        &expansion_mul(&cez, &ab),
    );
    let bcd = expansion_sum(
        &expansion_diff(&expansion_mul(&bez, &cd), &expansion_mul(&cez, &bd)),
        &expansion_mul(&dez, &bc),
    );
    let cda = expansion_sum(
        &expansion_sum(&expansion_mul(&cez, &da), &expansion_mul(&dez, &ac)),
        &expansion_mul(&aez, &cd),
    );
    let dab = expansion_sum(
        &expansion_sum(&expansion_mul(&dez, &ab), &expansion_mul(&aez, &bd)),
        &expansion_mul(&bez, &da),
    );

    let lift = |x: &[f64], y: &[f64], z: &[f64]| -> Vec<f64> {
        expansion_sum(
            &expansion_sum(&expansion_mul(x, x), &expansion_mul(y, y)),
            &expansion_mul(z, z),
        )
    };
    let alift = lift(&aex, &aey, &aez);
    let blift = lift(&bex, &bey, &bez);
    let clift = lift(&cex, &cey, &cez);
    let dlift = lift(&dex, &dey, &dez);

    let det = expansion_sum(
        &expansion_diff(&expansion_mul(&dlift, &abc), &expansion_mul(&clift, &dab)),
        &expansion_diff(&expansion_mul(&blift, &cda), &expansion_mul(&alift, &bcd)),
    );
    expansion_estimate(&det)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Expansion arithmetic ───────────────────────────────────────────

    #[test]
    fn expansion_sum_keeps_what_f64_addition_drops() {
        // 1 + 2^-60 is not representable: the naive sum loses the tail.
        let tiny = 2f64.powi(-60);
        assert_eq!(1.0 + tiny, 1.0);
        let e = expansion_sum(&[1.0], &[tiny]);
        assert_eq!(e, vec![tiny, 1.0]);
        // Subtracting 1 back recovers the tail exactly.
        assert_eq!(expansion_estimate(&expansion_diff(&e, &[1.0])), tiny);
    }

    #[test]
    fn expansion_mul_keeps_what_f64_multiplication_drops() {
        // (2^30 + 1)^2 = 2^60 + 2^31 + 1 needs 61 bits, so f64 rounds the 1
        // away.
        let a = 2f64.powi(30) + 1.0;
        let naive = a * a;
        let exact = expansion_mul(&[a], &[a]);
        assert_eq!(expansion_estimate(&expansion_diff(&exact, &[naive])), 1.0);
    }

    #[test]
    fn expansion_ops_cancel_to_exact_zero() {
        let e = expansion_mul(&[0.1, 3.0], &[7.0, 1e17]);
        assert_eq!(expansion_estimate(&expansion_diff(&e, &e)), 0.0);
    }

    /// Only the sign of the predicates is contractual — the magnitude is
    /// whatever the `f64` estimate produced, and reordering the arguments
    /// reorders the roundings.
    fn sign(x: f64) -> i32 {
        match x.partial_cmp(&0.0).expect("predicates never return NaN") {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    // ─── orient2d ───────────────────────────────────────────────────────

    #[test]
    fn orient2d_follows_permutation_parity() {
        let (a, b, c) = ([0.3, 0.1], [1.2, 0.4], [0.5, 1.7]);
        let base = sign(orient2d(&a, &b, &c));
        assert_eq!(base, 1);
        assert_eq!(sign(orient2d(&b, &a, &c)), -base);
        assert_eq!(sign(orient2d(&b, &c, &a)), base);
        assert_eq!(orient2d(&a, &b, &a), 0.0);
    }

    #[test]
    fn orient2d_resolves_a_sign_the_estimate_cannot() {
        // The classic near-collinear stress: c one ulp off the line (a, b),
        // with coordinates that make the two products cancel almost fully.
        let a = [0.5, 0.5];
        let b = [12.0, 12.0];
        let ulp = 2f64.powi(-48); // ulp(24.0)
        assert_eq!(orient2d(&a, &b, &[24.0, 24.0]), 0.0);
        assert!(orient2d(&a, &b, &[24.0, 24.0 + ulp]) > 0.0);
        assert!(orient2d(&a, &b, &[24.0, 24.0 - ulp]) < 0.0);
    }

    #[test]
    fn collinear3d_detects_exact_degeneracy_only() {
        assert!(collinear3d(
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0]
        ));
        assert!(collinear3d(
            &[7.0, -1.0, 0.5],
            &[7.0, -1.0, 0.5],
            &[0.0, 0.0, 0.0]
        ));
        // One ulp off the line is not collinear, however thin the triangle.
        assert!(!collinear3d(
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0 + 2f64.powi(-50)]
        ));
    }

    // ─── orient3d ───────────────────────────────────────────────────────

    #[test]
    fn orient3d_matches_tet4_numbering() {
        // The TET4 reference element: face 0-1-2 CCW seen from node 3.
        let (a, b, c, d) = (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        // Six times the volume of a tetrahedron of volume 1/6.
        assert_eq!(orient3d(&a, &b, &c, &d), 1.0);
    }

    #[test]
    fn orient3d_follows_permutation_parity() {
        let (a, b, c, d) = (
            [0.3, 0.1, 0.7],
            [1.2, 0.4, 0.9],
            [0.5, 1.7, 0.2],
            [0.8, 0.6, 2.3],
        );
        let base = sign(orient3d(&a, &b, &c, &d));
        assert_ne!(base, 0);
        // Odd permutations negate the sign, even ones preserve it.
        assert_eq!(sign(orient3d(&b, &a, &c, &d)), -base);
        assert_eq!(sign(orient3d(&a, &c, &b, &d)), -base);
        assert_eq!(sign(orient3d(&a, &b, &d, &c)), -base);
        assert_eq!(sign(orient3d(&b, &c, &a, &d)), base);
        assert_eq!(sign(orient3d(&c, &a, &b, &d)), base);
        assert_eq!(sign(orient3d(&b, &a, &d, &c)), base);
    }

    #[test]
    fn orient3d_is_exactly_zero_on_collinear_points() {
        // a, b, c are exactly collinear in f64, so every d is coplanar with
        // them. The estimate returns a non-zero rounding artefact here, so
        // reaching 0.0 proves the exact path ran.
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let c = [2.0, 4.0, 6.0];
        for d in [[0.7, 0.2, 0.5], [-3.25, 11.0, 0.125], [1e17, 3.0, -2.0]] {
            assert_eq!(orient3d(&a, &b, &c, &d), 0.0, "d = {d:?}");
        }
    }

    #[test]
    fn orient3d_is_exactly_zero_on_a_common_plane() {
        // Four points of the plane z = 0 whose coordinates are not exactly
        // representable.
        let a = [0.1, 0.7, 0.0];
        let b = [1.3, 0.2, 0.0];
        let c = [-0.9, 2.1, 0.0];
        let d = [7.7, -3.3, 0.0];
        assert_eq!(orient3d(&a, &b, &c, &d), 0.0);
    }

    #[test]
    fn orient3d_resolves_a_sign_the_estimate_cannot() {
        // Nudging c off the line (a, b) by a single ulp makes the true
        // determinant 1.2·2^-50 ≈ 1e-15 while the intermediate products are
        // of order 1 — right at the noise floor of the f64 estimate.
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let d = [0.7, 0.2, 0.5];
        let ulp = 2f64.powi(-50); // ulp(6.0)
        assert_eq!(orient3d(&a, &b, &[2.0, 4.0, 6.0], &d), 0.0);
        assert!(orient3d(&a, &b, &[2.0, 4.0, 6.0 + ulp], &d) > 0.0);
        assert!(orient3d(&a, &b, &[2.0, 4.0, 6.0 - ulp], &d) < 0.0);
    }

    // ─── insphere ───────────────────────────────────────────────────────

    /// Four positively-oriented corners of the unit cube.
    fn cube_tet() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let t = (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        assert!(orient3d(&t.0, &t.1, &t.2, &t.3) > 0.0);
        t
    }

    #[test]
    fn insphere_is_exactly_zero_on_cospherical_points() {
        let (a, b, c, d) = cube_tet();
        // Every remaining corner of the cube lies on the same sphere.
        for e in [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ] {
            assert_eq!(insphere(&a, &b, &c, &d, &e), 0.0, "e = {e:?}");
        }
    }

    #[test]
    fn insphere_separates_inside_from_outside() {
        let (a, b, c, d) = cube_tet();
        // The cube's centre is the circumcentre: as inside as it gets.
        assert!(insphere(&a, &b, &c, &d, &[0.5, 0.5, 0.5]) > 0.0);
        assert!(insphere(&a, &b, &c, &d, &[0.5, 0.5, 0.4]) > 0.0);
        assert!(insphere(&a, &b, &c, &d, &[-1.0, -1.0, -1.0]) < 0.0);
        assert!(insphere(&a, &b, &c, &d, &[2.0, 0.0, 0.0]) < 0.0);
    }

    #[test]
    fn insphere_flips_with_the_orientation_of_the_base() {
        let (a, b, c, d) = cube_tet();
        let inside = [0.5, 0.5, 0.5];
        let base = sign(insphere(&a, &b, &c, &d, &inside));
        assert_eq!(base, 1);
        // Swapping two base points reverses orient3d, hence insphere.
        assert_eq!(sign(insphere(&b, &a, &c, &d, &inside)), -base);
        // An even permutation leaves both invariant.
        assert_eq!(sign(insphere(&b, &c, &a, &d, &inside)), base);
    }

    #[test]
    fn insphere_resolves_a_point_one_ulp_off_the_sphere() {
        let (a, b, c, d) = cube_tet();
        let on = [1.0, 1.0, 1.0];
        assert_eq!(insphere(&a, &b, &c, &d, &on), 0.0);
        assert!(insphere(&a, &b, &c, &d, &[1.0 - f64::EPSILON, 1.0, 1.0]) > 0.0);
        assert!(insphere(&a, &b, &c, &d, &[1.0 + f64::EPSILON, 1.0, 1.0]) < 0.0);
    }

    // ─── Self-consistency ───────────────────────────────────────────────

    /// Deterministic index-seeded pseudo-random stream — no RNG dependency,
    /// and the same sequence on every run and every platform.
    fn pseudo_random(i: u64) -> f64 {
        let x = i
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(31)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn predicates_stay_self_consistent_on_degenerate_clouds() {
        // Coordinates drawn from a tiny integer grid so that coplanar and
        // cospherical configurations occur constantly — precisely the input
        // on which a tolerance-based or jittered predicate contradicts
        // itself, and precisely what a mesher sees on structured geometry.
        let coord = |i: u64| (pseudo_random(i) * 4.0).floor() * 0.5;
        let point = |i: u64| [coord(3 * i), coord(3 * i + 1), coord(3 * i + 2)];

        let (mut oriented, mut cospherical) = (0usize, 0usize);
        for k in 0..4000u64 {
            let (a, b, c, d, e) = (
                point(5 * k),
                point(5 * k + 1),
                point(5 * k + 2),
                point(5 * k + 3),
                point(5 * k + 4),
            );

            // orient3d: antisymmetric under every transposition, and a
            // repeated argument always makes it exactly degenerate.
            let o = sign(orient3d(&a, &b, &c, &d));
            assert_eq!(sign(orient3d(&b, &a, &c, &d)), -o);
            assert_eq!(sign(orient3d(&a, &b, &d, &c)), -o);
            assert_eq!(sign(orient3d(&c, &d, &a, &b)), o);
            assert_eq!(orient3d(&a, &b, &c, &a), 0.0);
            if o == 0 {
                continue;
            }
            oriented += 1;

            // insphere: relative to a positively-oriented base, and every
            // base point is on its own circumsphere.
            let (p, q) = if o > 0 { (b, c) } else { (c, b) };
            assert!(orient3d(&a, &p, &q, &d) > 0.0);
            for v in [a, p, q, d] {
                assert_eq!(insphere(&a, &p, &q, &d, &v), 0.0);
            }
            let s = sign(insphere(&a, &p, &q, &d, &e));
            // A transposition of the base flips the orientation, hence the
            // predicate; a rotation leaves both alone.
            assert_eq!(sign(insphere(&p, &a, &q, &d, &e)), -s);
            assert_eq!(sign(insphere(&p, &q, &a, &d, &e)), s);
            if s == 0 {
                cospherical += 1;
            }
        }

        // The point of the fixture is that it is degenerate on purpose: if
        // these ever drop to zero the test has stopped exercising the exact
        // path and must be rebuilt.
        assert!(oriented > 1000, "only {oriented} non-degenerate bases");
        assert!(cospherical > 0, "no cospherical case was generated");
    }

    #[test]
    fn insphere_agrees_with_the_circumradius_on_generic_input() {
        // A tetrahedron with no symmetry: cross-check the predicate against
        // an independent (inexact, but far from any degeneracy) computation
        // of the circumsphere.
        let a = [0.1, 0.2, 0.3];
        let b = [1.7, 0.1, 0.4];
        let c = [0.3, 2.1, 0.5];
        let d = [0.2, 0.4, 1.9];
        assert!(orient3d(&a, &b, &c, &d) > 0.0);

        // The circumcentre solves 2·(p − a)·x = |p|² − |a|² for p in b, c, d.
        let det3 = |m: &[[f64; 3]; 3]| -> f64 {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let mut m = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        for (i, p) in [b, c, d].iter().enumerate() {
            for k in 0..3 {
                m[i][k] = 2.0 * (p[k] - a[k]);
                rhs[i] += p[k] * p[k] - a[k] * a[k];
            }
        }
        let base = det3(&m);
        let mut centre = [0.0f64; 3];
        for col in 0..3 {
            let mut mc = m;
            for row in 0..3 {
                mc[row][col] = rhs[row];
            }
            centre[col] = det3(&mc) / base;
        }
        let dist2 = |p: &[f64; 3]| (0..3).map(|k| (p[k] - centre[k]).powi(2)).sum::<f64>();
        let r2 = dist2(&a);

        for e in [
            [0.4, 0.5, 0.6],
            [1.5, 1.5, 1.5],
            [-1.0, 0.0, 0.0],
            [0.6, 0.7, 0.2],
            [2.5, 2.5, 2.5],
        ] {
            assert_eq!(
                insphere(&a, &b, &c, &d, &e) > 0.0,
                dist2(&e) < r2,
                "e = {e:?}"
            );
        }
    }
}
