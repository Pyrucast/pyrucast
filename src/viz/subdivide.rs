//! Reference-space subdivision of elements for interpolated rendering.
//!
//! The backend only fills flat-coloured polygons, so a colour that varies
//! inside an element is approximated by splitting the element into small
//! sub-triangles: each sub-vertex carries the shape-function weights
//! `N_i(ξ)` evaluated at its reference position, so both its geometry
//! (`x = Σ N_i·x_i`) and its value (`v = Σ N_i·v_i`) follow the element's
//! interpolation — including the bilinear warp of QUA4 / HEX8 faces.
//!
//! The subdivision is computed **once per element type** (the weights do
//! not depend on the cell) and reused for every cell of a submesh. It is
//! purely graphical and internal to each element: sub-vertices on a
//! shared edge are evaluated separately by each side, so inter-element
//! discontinuities of a Gauss-point field stay visible.

use crate::containers::finite_element_space::Interpolation;
use crate::containers::mesh::ElementType;
use crate::error::Result;
use crate::viz::mesh_draw::{HEX8_FACES, PENTA6_FACES, TET4_FACES};

/// Subdivision of one face of the reference element.
pub(crate) struct FaceSubdivision {
    /// One row of shape-function weights (length = nodes-per-cell) per
    /// sub-vertex.
    pub weights: Vec<Vec<f64>>,
    /// Sub-triangles, as indices into `weights`. CCW in the face's own
    /// orientation (consistent with the flat renderer's face tables).
    pub triangles: Vec<[usize; 3]>,
    /// Weights of the face's original corners, in order — used to draw
    /// the element boundary as a wire on top of the sub-faces.
    pub outline: Vec<Vec<f64>>,
}

/// Reference-space subdivision of one element type at level `n`.
pub(crate) enum CellSubdivision {
    /// POI1 — nothing to subdivide; one point per cell.
    Points,
    /// SEG2 — `n` sub-segments.
    Segments {
        weights: Vec<Vec<f64>>,
        segments: Vec<[usize; 2]>,
    },
    /// TRI3 / QUA4 (one face), TET4 (4 faces), HEX8 (6 faces).
    Faces(Vec<FaceSubdivision>),
}

/// Reference coordinates of each node, per element type (the points
/// where the matching shape function equals 1).
fn ref_nodes(et: ElementType) -> Vec<Vec<f64>> {
    match et {
        ElementType::POI1 => vec![vec![]],
        ElementType::SEG2 => vec![vec![-1.0], vec![1.0]],
        ElementType::TRI3 => vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        ElementType::QUA4 => vec![
            vec![-1.0, -1.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
            vec![-1.0, 1.0],
        ],
        ElementType::TET4 => vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        ElementType::PENTA6 => vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ],
        ElementType::HEX8 => [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        .iter()
        .map(|&(a, b, c)| vec![a, b, c])
        .collect(),
    }
}

/// Subdivide `et` at level `n` (`n ≥ 1`): TRI3 → n² sub-triangles,
/// QUA4 → 2n², SEG2 → n segments, TET4 / HEX8 → their outer faces
/// subdivided likewise.
pub(crate) fn subdivide(
    et: ElementType,
    interp: Interpolation,
    n: usize,
) -> Result<CellSubdivision> {
    let n = n.max(1);
    let nodes = ref_nodes(et);
    match et {
        ElementType::POI1 => Ok(CellSubdivision::Points),
        ElementType::SEG2 => {
            let mut weights = Vec::with_capacity(n + 1);
            for k in 0..=n {
                let xi = -1.0 + 2.0 * k as f64 / n as f64;
                weights.push(interp.shape(et, &[xi])?);
            }
            let segments = (0..n).map(|k| [k, k + 1]).collect();
            Ok(CellSubdivision::Segments { weights, segments })
        }
        ElementType::TRI3 => Ok(CellSubdivision::Faces(vec![tri_face(
            et,
            interp,
            n,
            [&nodes[0], &nodes[1], &nodes[2]],
        )?])),
        ElementType::QUA4 => Ok(CellSubdivision::Faces(vec![quad_face(
            et,
            interp,
            n,
            [&nodes[0], &nodes[1], &nodes[2], &nodes[3]],
        )?])),
        ElementType::TET4 => {
            let mut faces = Vec::with_capacity(4);
            for f in &TET4_FACES {
                faces.push(tri_face(
                    et,
                    interp,
                    n,
                    [&nodes[f[0]], &nodes[f[1]], &nodes[f[2]]],
                )?);
            }
            Ok(CellSubdivision::Faces(faces))
        }
        ElementType::HEX8 => {
            let mut faces = Vec::with_capacity(6);
            for f in &HEX8_FACES {
                faces.push(quad_face(
                    et,
                    interp,
                    n,
                    [&nodes[f[0]], &nodes[f[1]], &nodes[f[2]], &nodes[f[3]]],
                )?);
            }
            Ok(CellSubdivision::Faces(faces))
        }
        ElementType::PENTA6 => {
            // Two triangular caps + three quadrilateral sides.
            let mut faces = Vec::with_capacity(5);
            for f in PENTA6_FACES.iter().filter(|f| f.len() == 3) {
                faces.push(tri_face(
                    et,
                    interp,
                    n,
                    [&nodes[f[0]], &nodes[f[1]], &nodes[f[2]]],
                )?);
            }
            for f in PENTA6_FACES.iter().filter(|f| f.len() == 4) {
                faces.push(quad_face(
                    et,
                    interp,
                    n,
                    [&nodes[f[0]], &nodes[f[1]], &nodes[f[2]], &nodes[f[3]]],
                )?);
            }
            Ok(CellSubdivision::Faces(faces))
        }
    }
}

/// Affine combination of reference points: `Σ c_k · p_k`.
fn combine(points: &[&Vec<f64>], coeffs: &[f64]) -> Vec<f64> {
    let dim = points[0].len();
    let mut out = vec![0.0; dim];
    for (p, &c) in points.iter().zip(coeffs) {
        for d in 0..dim {
            out[d] += c * p[d];
        }
    }
    out
}

/// Triangular face spanned by corners `[p, q, r]` (reference coords of
/// the element): barycentric lattice of side `n`.
fn tri_face(
    et: ElementType,
    interp: Interpolation,
    n: usize,
    corners: [&Vec<f64>; 3],
) -> Result<FaceSubdivision> {
    let mut weights = Vec::new();
    // idx(i, j) with i + j ≤ n, rows of decreasing length.
    let mut index = vec![vec![0usize; n + 1]; n + 1];
    for i in 0..=n {
        for j in 0..=(n - i) {
            let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
            let xi = combine(&corners, &[1.0 - u - v, u, v]);
            index[i][j] = weights.len();
            weights.push(interp.shape(et, &xi)?);
        }
    }
    let mut triangles = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..(n - i) {
            triangles.push([index[i][j], index[i + 1][j], index[i][j + 1]]);
            if i + j < n - 1 {
                triangles.push([index[i + 1][j], index[i + 1][j + 1], index[i][j + 1]]);
            }
        }
    }
    let outline = corners
        .iter()
        .map(|c| interp.shape(et, c))
        .collect::<Result<_>>()?;
    Ok(FaceSubdivision {
        weights,
        triangles,
        outline,
    })
}

/// Quadrangular face spanned by corners `[c0, c1, c2, c3]` (CCW):
/// bilinear lattice of `(n+1)²` points, two sub-triangles per grid quad.
fn quad_face(
    et: ElementType,
    interp: Interpolation,
    n: usize,
    corners: [&Vec<f64>; 4],
) -> Result<FaceSubdivision> {
    let mut weights = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            let s = -1.0 + 2.0 * i as f64 / n as f64;
            let t = -1.0 + 2.0 * j as f64 / n as f64;
            let coeffs = [
                0.25 * (1.0 - s) * (1.0 - t),
                0.25 * (1.0 + s) * (1.0 - t),
                0.25 * (1.0 + s) * (1.0 + t),
                0.25 * (1.0 - s) * (1.0 + t),
            ];
            let xi = combine(&corners, &coeffs);
            weights.push(interp.shape(et, &xi)?);
        }
    }
    let idx = |i: usize, j: usize| j * (n + 1) + i;
    let mut triangles = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            triangles.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            triangles.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    let outline = corners
        .iter()
        .map(|c| interp.shape(et, c))
        .collect::<Result<_>>()?;
    Ok(FaceSubdivision {
        weights,
        triangles,
        outline,
    })
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every sub-vertex's weights must sum to 1 (partition of unity).
    fn assert_partition_of_unity(weights: &[Vec<f64>]) {
        for w in weights {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "Σ N_i = {s} ≠ 1");
        }
    }

    #[test]
    fn tri3_counts_and_unity() {
        let n = 4;
        let CellSubdivision::Faces(faces) =
            subdivide(ElementType::TRI3, Interpolation::Lagrange1, n).unwrap()
        else {
            panic!("expected faces");
        };
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].triangles.len(), n * n);
        assert_eq!(faces[0].weights.len(), (n + 1) * (n + 2) / 2);
        assert_partition_of_unity(&faces[0].weights);
        assert_eq!(faces[0].outline.len(), 3);
    }

    #[test]
    fn qua4_counts_and_unity() {
        let n = 3;
        let CellSubdivision::Faces(faces) =
            subdivide(ElementType::QUA4, Interpolation::Lagrange1, n).unwrap()
        else {
            panic!("expected faces");
        };
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].triangles.len(), 2 * n * n);
        assert_eq!(faces[0].weights.len(), (n + 1) * (n + 1));
        assert_partition_of_unity(&faces[0].weights);
        assert_eq!(faces[0].outline.len(), 4);
    }

    #[test]
    fn tet4_and_hex8_face_counts() {
        let n = 2;
        let CellSubdivision::Faces(tet) =
            subdivide(ElementType::TET4, Interpolation::Lagrange1, n).unwrap()
        else {
            panic!("expected faces");
        };
        assert_eq!(tet.len(), 4);
        for f in &tet {
            assert_eq!(f.triangles.len(), n * n);
            assert_partition_of_unity(&f.weights);
        }
        let CellSubdivision::Faces(hex) =
            subdivide(ElementType::HEX8, Interpolation::Lagrange1, n).unwrap()
        else {
            panic!("expected faces");
        };
        assert_eq!(hex.len(), 6);
        for f in &hex {
            assert_eq!(f.triangles.len(), 2 * n * n);
            assert_partition_of_unity(&f.weights);
        }
    }

    #[test]
    fn seg2_segments() {
        let CellSubdivision::Segments { weights, segments } =
            subdivide(ElementType::SEG2, Interpolation::Lagrange1, 5).unwrap()
        else {
            panic!("expected segments");
        };
        assert_eq!(weights.len(), 6);
        assert_eq!(segments.len(), 5);
        assert_partition_of_unity(&weights);
    }

    /// At a corner of the lattice, the weights are the Kronecker delta of
    /// the matching node: geometry and values pass exactly through the
    /// element's nodes.
    #[test]
    fn corners_are_kronecker() {
        let CellSubdivision::Faces(faces) =
            subdivide(ElementType::TRI3, Interpolation::Lagrange1, 2).unwrap()
        else {
            panic!("expected faces");
        };
        let w0 = &faces[0].weights[0]; // lattice (0,0) = node 0
        assert!((w0[0] - 1.0).abs() < 1e-12);
        assert!(w0[1].abs() < 1e-12 && w0[2].abs() < 1e-12);
    }
}
