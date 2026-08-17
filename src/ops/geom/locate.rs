//! Point location in a host mesh — the inverse iso-parametric mapping.
//!
//! The forward map of a cell sends reference coordinates `ξ` to a physical
//! point `x = Σᵢ Nᵢ(ξ)·Xᵢ`. [`locate_points`] inverts it: given a physical
//! point, it finds the host cell that **contains** it and the reference
//! coordinates `ξ` there, then returns the shape-function weights `Nᵢ(ξ)` and
//! the cell's node ids. It is the geometric primitive under an *embedded* /
//! *immersed* constraint: the weights are exactly the coefficients tying an
//! immersed node's field to the host nodes' fields (`u(p) = Σᵢ Nᵢ·u(hostᵢ)`).
//!
//! The inverse map is solved cell-by-cell by a small Newton iteration on the
//! residual `r(ξ) = x − Σᵢ Nᵢ(ξ)·Xᵢ`, using the element Jacobian
//! `J = ∂x/∂ξ` (built from [`Interpolation::dshape_dxi`]). A cheap per-cell
//! bounding-box test rejects the obvious misses first (broad phase); the
//! reference-domain containment test on the converged `ξ` is the final word.
//! For a point on a shared face the first containing cell (in submesh, then
//! cell order) wins.

use crate::atoms::Interpolation;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::Mesh;
use crate::error::Result;
use crate::parallel::*;

/// Where one physical point sits inside a host mesh.
#[derive(Clone, Debug)]
pub struct Location {
    /// Index of the host submesh the containing cell belongs to.
    pub submesh: usize,
    /// Index of the containing cell within that submesh.
    pub cell: usize,
    /// Reference coordinates `ξ` of the point in that cell.
    pub xi: Vec<f64>,
    /// Shape-function weights `Nᵢ(ξ)`, ordered like the cell's nodes
    /// (sums to 1; length = `element_type.nodes_per_cell()`).
    pub weights: Vec<f64>,
    /// The containing cell's node ids, ordered like `weights`.
    pub nodes: Vec<NodeId>,
}

/// Locate each physical point of `points` inside `host`.
///
/// Returns one entry per input point, `None` when the point lies in no host
/// cell (within `tol` of a cell's reference domain). Each point coordinate
/// slice must have the host `Coords` spatial dimension.
///
/// `tol` is the reference-domain slack for the containment test (a point up to
/// `tol` outside the reference element is still accepted, so points exactly on
/// a face are captured); `1e-6` is a sensible default.
pub fn locate_points(host: &Mesh, points: &[Vec<f64>], tol: f64) -> Result<Vec<Option<Location>>> {
    // Snapshot every host cell once: (submesh, cell, element type, node ids,
    // node coordinates, bbox). One read lock per submesh; coordinates are copied
    // out so no Coords guard is held during the Newton solves.
    struct Cell {
        submesh: usize,
        cell: usize,
        element_type: ElementType,
        nodes: Vec<NodeId>,
        coords: Vec<Vec<f64>>,
        lo: Vec<f64>,
        hi: Vec<f64>,
    }

    let mut cells: Vec<Cell> = Vec::new();
    for (s_idx, sm_handle) in host.into_iter().enumerate() {
        let (element_type, conn, coords_handle) = {
            let sm = sm_handle.read();
            (sm.element_type(), sm.connectivity().to_vec(), sm.coords())
        };
        if element_type == ElementType::POI1 {
            continue; // A node has no interior to contain anything.
        }
        let npc = element_type.nodes_per_cell();
        let c = coords_handle.read();
        for cell in 0..conn.len() / npc {
            let ids = &conn[cell * npc..(cell + 1) * npc];
            let coords: Vec<Vec<f64>> = ids
                .iter()
                .map(|&n| Ok(c.position(n)?.to_vec()))
                .collect::<Result<_>>()?;
            let dim = coords[0].len();
            let mut lo = coords[0].clone();
            let mut hi = coords[0].clone();
            for p in &coords[1..] {
                for a in 0..dim {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            cells.push(Cell {
                submesh: s_idx,
                cell,
                element_type,
                nodes: ids.to_vec(),
                coords,
                lo,
                hi,
            });
        }
    }

    if cells.is_empty() {
        return Ok(vec![None; points.len()]);
    }

    // Spatial index: a uniform grid bucketing cells by the grid cells their
    // (margin-expanded) bbox overlaps. A query then tests only the cells in the
    // point's own bucket instead of scanning all of them — any cell whose
    // broad-phase bbox contains the point was inserted into that bucket.
    let lo: Vec<Vec<f64>> = cells.iter().map(|c| c.lo.clone()).collect();
    let hi: Vec<Vec<f64>> = cells.iter().map(|c| c.hi.clone()).collect();
    let grid = Grid::build(&lo, &hi);

    // One independent location per point, in parallel (each writes its own output
    // slot exactly once ⇒ result is independent of the thread count). Within a
    // point, candidates are visited in ascending cell order, so the "first
    // containing cell wins" tie-break is deterministic and matches a full scan.
    points
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|x| -> Result<Option<Location>> {
            for &ci in grid.candidates(x) {
                let cell = &cells[ci];
                if !in_bbox(x, &cell.lo, &cell.hi) {
                    continue;
                }
                if let Some((xi, weights)) = invert_cell(cell.element_type, &cell.coords, x, tol)? {
                    return Ok(Some(Location {
                        submesh: cell.submesh,
                        cell: cell.cell,
                        xi,
                        weights,
                        nodes: cell.nodes.clone(),
                    }));
                }
            }
            Ok(None)
        })
        .collect()
}

/// A uniform grid over the host bounding box, mapping each grid cell to the host
/// cells whose (margin-expanded) bbox overlaps it — the broad-phase accelerator
/// of [`locate_points`].
struct Grid {
    lo: Vec<f64>,
    inv: Vec<f64>,   // 1 / grid-cell size per axis (0 on a degenerate axis)
    res: Vec<usize>, // grid cells per axis
    stride: Vec<usize>,
    buckets: Vec<Vec<usize>>,
}

impl Grid {
    /// Build the grid from per-cell bounding boxes (`lo[i]`, `hi[i]`), aiming for
    /// roughly one host cell per grid cell.
    fn build(lo: &[Vec<f64>], hi: &[Vec<f64>]) -> Grid {
        let n = lo.len();
        let dim = lo[0].len();

        // Global bbox.
        let mut glo = lo[0].clone();
        let mut ghi = hi[0].clone();
        for i in 1..n {
            for a in 0..dim {
                glo[a] = glo[a].min(lo[i][a]);
                ghi[a] = ghi[a].max(hi[i][a]);
            }
        }

        // ~n buckets total ⇒ res ≈ n^(1/dim) per non-degenerate axis.
        let target = (n as f64).powf(1.0 / dim as f64).round().max(1.0) as usize;
        let target = target.clamp(1, 128);
        let mut res = vec![1usize; dim];
        let mut inv = vec![0.0f64; dim];
        for a in 0..dim {
            let extent = ghi[a] - glo[a];
            if extent > 1e-12 {
                res[a] = target;
                inv[a] = res[a] as f64 / extent;
            }
        }
        let mut stride = vec![1usize; dim];
        for a in 1..dim {
            stride[a] = stride[a - 1] * res[a - 1];
        }
        let total: usize = res.iter().product();
        let mut buckets = vec![Vec::new(); total];

        // Insert each cell into every bucket its margin-expanded bbox overlaps
        // (the same margin as `in_bbox`, so a point that passes the broad phase
        // for a cell always lands in a bucket holding that cell).
        for i in 0..n {
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(dim);
            for a in 0..dim {
                let margin = 0.1 * (hi[i][a] - lo[i][a]).abs() + 1e-12;
                let la = axis_index(lo[i][a] - margin, glo[a], inv[a], res[a]);
                let ha = axis_index(hi[i][a] + margin, glo[a], inv[a], res[a]);
                ranges.push((la, ha));
            }
            insert_ranges(&mut buckets, &ranges, &stride, i);
        }

        Grid {
            lo: glo,
            inv,
            res,
            stride,
            buckets,
        }
    }

    /// The host-cell indices to test for a query point (its bucket's contents),
    /// in ascending cell order.
    fn candidates(&self, x: &[f64]) -> &[usize] {
        let mut linear = 0;
        for a in 0..self.lo.len() {
            linear += axis_index(x[a], self.lo[a], self.inv[a], self.res[a]) * self.stride[a];
        }
        &self.buckets[linear]
    }
}

/// Grid index of coordinate `v` on one axis, clamped to `[0, res-1]`.
fn axis_index(v: f64, origin: f64, inv: f64, res: usize) -> usize {
    if inv == 0.0 {
        return 0;
    }
    let idx = ((v - origin) * inv).floor();
    if idx < 0.0 {
        0
    } else if idx as usize >= res {
        res - 1
    } else {
        idx as usize
    }
}

/// Push cell index `i` into every bucket of the axis-aligned index box `ranges`
/// (inclusive per axis), enumerating the cartesian product.
fn insert_ranges(
    buckets: &mut [Vec<usize>],
    ranges: &[(usize, usize)],
    stride: &[usize],
    i: usize,
) {
    let dim = ranges.len();
    let mut idx = vec![0usize; dim];
    for (a, &(l, _)) in ranges.iter().enumerate() {
        idx[a] = l;
    }
    loop {
        let mut linear = 0;
        for a in 0..dim {
            linear += idx[a] * stride[a];
        }
        buckets[linear].push(i);

        // Increment the multi-index (row-major over the ranges).
        let mut a = 0;
        loop {
            if a == dim {
                return;
            }
            if idx[a] < ranges[a].1 {
                idx[a] += 1;
                break;
            }
            idx[a] = ranges[a].0;
            a += 1;
        }
    }
}

/// Is `x` within the axis-aligned box `[lo, hi]` expanded by 10 % of its size?
fn in_bbox(x: &[f64], lo: &[f64], hi: &[f64]) -> bool {
    for a in 0..x.len() {
        let margin = 0.1 * (hi[a] - lo[a]).abs() + 1e-12;
        if x[a] < lo[a] - margin || x[a] > hi[a] + margin {
            return false;
        }
    }
    true
}

/// Newton inverse map for one cell. Returns `Some((ξ, N))` when `x` converges
/// to a reference point inside the element (within `tol`), `None` otherwise.
fn invert_cell(
    element_type: ElementType,
    coords: &[Vec<f64>],
    x: &[f64],
    tol: f64,
) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    let kind = element_type.as_kind();
    let tdim = element_type.topological_dim();
    let sdim = x.len();
    let npc = element_type.nodes_per_cell();
    let mut xi = reference_centroid(element_type);

    // Scratch buffers, reused across the up-to-40 Newton steps: `shape_into` /
    // `dshape_into` write in place, so the loop allocates nothing.
    let mut n = vec![0.0; npc];
    let mut dn = vec![0.0; npc * tdim];
    let mut jac = vec![0.0; sdim * tdim];

    for _ in 0..40 {
        kind.shape_into(&xi, &mut n);
        // Residual r = x − Σᵢ Nᵢ·Xᵢ  (length sdim).
        let mut r = x.to_vec();
        for (i, &ni) in n.iter().enumerate() {
            for a in 0..sdim {
                r[a] -= ni * coords[i][a];
            }
        }
        // Jacobian J[a][j] = ∂x_a/∂ξ_j = Σᵢ (∂Nᵢ/∂ξ_j)·X[i][a]  (sdim × tdim).
        kind.dshape_into(&xi, &mut dn); // [i*tdim + j]
        jac.fill(0.0);
        for (i, coord) in coords.iter().enumerate() {
            for a in 0..sdim {
                for j in 0..tdim {
                    jac[a * tdim + j] += dn[i * tdim + j] * coord[a];
                }
            }
        }
        // Solve J·δξ = r in the least-squares sense (normal equations, tdim ≤ 3).
        let dxi = solve_normal(&jac, &r, sdim, tdim)?;
        let mut step = 0.0;
        for j in 0..tdim {
            xi[j] += dxi[j];
            step += dxi[j] * dxi[j];
        }
        if step.sqrt() <= 1e-12 {
            break;
        }
    }

    // Accept iff the converged point maps back onto x and lies in the reference
    // domain (a diverged / out-of-cell solve fails one of the two).
    kind.shape_into(&xi, &mut n);
    let mut r2 = 0.0;
    for a in 0..sdim {
        let mut xa = 0.0;
        for (i, &ni) in n.iter().enumerate() {
            xa += ni * coords[i][a];
        }
        r2 += (x[a] - xa).powi(2);
    }
    let scale = bbox_scale(coords);
    if r2.sqrt() <= 1e-9 * scale.max(1.0) && contains_reference(element_type, &xi, tol) {
        Ok(Some((xi, n)))
    } else {
        Ok(None)
    }
}

/// Solve the (possibly rectangular) system `J·δ = r` via normal equations
/// `(JᵀJ)·δ = Jᵀr`, with `J` stored row-major `sdim × tdim`. `tdim ≤ 3`.
pub(super) fn solve_normal(jac: &[f64], r: &[f64], sdim: usize, tdim: usize) -> Result<Vec<f64>> {
    // A = JᵀJ (tdim × tdim), b = Jᵀr (tdim).
    let mut a = vec![0.0; tdim * tdim];
    let mut b = vec![0.0; tdim];
    for i in 0..tdim {
        for k in 0..tdim {
            let mut s = 0.0;
            for row in 0..sdim {
                s += jac[row * tdim + i] * jac[row * tdim + k];
            }
            a[i * tdim + k] = s;
        }
        let mut s = 0.0;
        for row in 0..sdim {
            s += jac[row * tdim + i] * r[row];
        }
        b[i] = s;
    }
    solve_small(&mut a, &mut b, tdim)
}

/// In-place Gaussian elimination with partial pivoting for `n ≤ 3`. A singular
/// system yields a zero step (the Newton iteration then simply stalls and the
/// cell is rejected by the residual test).
fn solve_small(a: &mut [f64], b: &mut [f64], n: usize) -> Result<Vec<f64>> {
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        for row in (col + 1)..n {
            if a[row * n + col].abs() > a[piv * n + col].abs() {
                piv = row;
            }
        }
        if a[piv * n + col].abs() < 1e-300 {
            return Ok(vec![0.0; n]);
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        for row in (col + 1)..n {
            let f = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for k in (col + 1)..n {
            s -= a[col * n + k] * x[k];
        }
        x[col] = s / a[col * n + col];
    }
    Ok(x)
}

/// A characteristic length of a cell (its bbox diagonal) for scaling residuals.
fn bbox_scale(coords: &[Vec<f64>]) -> f64 {
    let dim = coords[0].len();
    let mut lo = coords[0].clone();
    let mut hi = coords[0].clone();
    for p in &coords[1..] {
        for a in 0..dim {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let mut d2 = 0.0;
    for a in 0..dim {
        d2 += (hi[a] - lo[a]).powi(2);
    }
    d2.sqrt()
}

/// The interpolation whose degree matches `element_type` (linear ↔ Lagrange-1,
/// quadratic ↔ Lagrange-2).
pub(super) fn interpolation_for(element_type: ElementType) -> Interpolation {
    if Interpolation::Lagrange2.is_compatible_with(element_type) {
        Interpolation::Lagrange2
    } else {
        Interpolation::Lagrange1
    }
}

/// A reference-domain interior point used as the Newton starting guess.
///
/// The element's own centroid — the same point the one-point (`Reduced`)
/// quadrature integrates at.
pub(super) fn reference_centroid(element_type: ElementType) -> Vec<f64> {
    element_type.as_kind().ref_centroid().to_vec()
}

/// Is `ξ` inside `element_type`'s reference domain, allowing `tol` of slack?
fn contains_reference(element_type: ElementType, xi: &[f64], tol: f64) -> bool {
    element_type.as_kind().contains_ref(xi, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A unit HEX8 at the origin; locate its centre and a corner.
    #[test]
    fn locate_in_hex8() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let ids: Vec<_> = corners
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::HEX8);
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let pts = vec![vec![0.5, 0.5, 0.5], vec![2.0, 2.0, 2.0]];
        let loc = locate_points(&mesh, &pts, 1e-6).unwrap();

        // Centre: found, all weights 1/8.
        let c = loc[0].as_ref().expect("centre is inside");
        assert_eq!(c.nodes.len(), 8);
        for w in &c.weights {
            assert!((w - 0.125).abs() < 1e-9);
        }
        // Weights partition unity, and interpolate the coordinate back.
        let sum: f64 = c.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Outside point: not found.
        assert!(loc[1].is_none());
    }

    /// A single TET4; the shape weights at a known interior point match the
    /// barycentric coordinates.
    #[test]
    fn locate_in_tet4_weights_are_barycentric() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let ids: Vec<_> = v
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Point x reconstructs from barycentric weights (0.1,0.2,0.3,0.4).
        let x = vec![0.2, 0.3, 0.4];
        let loc = locate_points(&mesh, std::slice::from_ref(&x), 1e-6).unwrap();
        let l = loc[0].as_ref().expect("inside tet");
        let expected = [0.1, 0.2, 0.3, 0.4];
        for (w, e) in l.weights.iter().zip(expected.iter()) {
            assert!((w - e).abs() < 1e-9, "weight {w} vs {e}");
        }
    }

    /// A block of `n³` HEX8 cells: every cell centre locates in **its** cell
    /// (weights 1/8), and a point outside the block is rejected — exercising the
    /// uniform-grid spatial index over many cells.
    #[test]
    fn locate_in_hex8_block() {
        let n = 3usize;
        let coords = Handle::new(Coords::new(3).unwrap());
        let node =
            |i: usize, j: usize, k: usize| -> Vec<f64> { vec![i as f64, j as f64, k as f64] };
        // Grid of (n+1)³ nodes.
        let mut id = std::collections::HashMap::new();
        for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    id.insert(
                        (i, j, k),
                        Node::create_in(coords.clone(), &node(i, j, k))
                            .unwrap()
                            .id(),
                    );
                }
            }
        }
        let mut sm = SubMesh::new(coords, ElementType::HEX8);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    sm.add_cell(&[
                        id[&(i, j, k)],
                        id[&(i + 1, j, k)],
                        id[&(i + 1, j + 1, k)],
                        id[&(i, j + 1, k)],
                        id[&(i, j, k + 1)],
                        id[&(i + 1, j, k + 1)],
                        id[&(i + 1, j + 1, k + 1)],
                        id[&(i, j + 1, k + 1)],
                    ])
                    .unwrap();
                }
            }
        }
        let mesh = Mesh::from_submesh(sm);

        // One point at each cell centre, plus one outside the block.
        let mut pts = Vec::new();
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    pts.push(vec![i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5]);
                }
            }
        }
        pts.push(vec![100.0, 100.0, 100.0]);
        let loc = locate_points(&mesh, &pts, 1e-6).unwrap();

        for (c, l) in loc.iter().take(n * n * n).enumerate() {
            let l = l
                .as_ref()
                .unwrap_or_else(|| panic!("cell centre {c} not located"));
            assert_eq!(l.cell, c, "point {c} should map to cell {c}");
            for w in &l.weights {
                assert!((w - 0.125).abs() < 1e-9);
            }
        }
        assert!(loc[n * n * n].is_none(), "outside point rejected");
    }

    /// A point just outside a QUA4 (2D) is rejected; one inside is accepted.
    #[test]
    fn locate_in_qua4_2d() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let v = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let ids: Vec<_> = v
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let pts = vec![vec![1.0, 0.5], vec![1.0, 2.0]];
        let loc = locate_points(&mesh, &pts, 1e-6).unwrap();
        assert!(loc[0].is_some());
        assert!(loc[1].is_none());
    }
}
