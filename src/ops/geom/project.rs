//! Closest-point projection onto a surface mesh — the geometric primitive of
//! node-to-surface **contact**.
//!
//! Where [`locate_points`](super::locate_points) inverts the iso-parametric map
//! for a point *inside* a host cell, [`project_points`] handles the nominal
//! contact situation: the point lies **off** the surface, and we want the
//! closest point *on* it. For each query point `x` and each candidate facet, a
//! projected Gauss–Newton iteration minimises `‖x − F(ξ)‖²` with `ξ` clamped to
//! the reference domain (so edges and corners are handled by the clamping, not
//! by special cases); the facet with the smallest distance wins.
//!
//! The result carries the shape weights `Nᵢ(ξ)` at the projection — exactly the
//! master coefficients of a contact relation — plus the facet **normal** and the
//! **signed gap** `g = (x − p)·n`:
//!
//! - in 2D (`SEG2`/`SEG3` facets), `n` is the unit tangent rotated by −90°
//!   (`t = ∂x/∂ξ` ⇒ `n = (t_y, −t_x)/‖t‖`) — outward for a counter-clockwise
//!   contour;
//! - in 3D (`TRI*`/`QUA*` facets), `n = (∂x/∂ξ₁ × ∂x/∂ξ₂)` normalised — the
//!   right-hand rule on the facet's node ordering.
//!
//! The surface must therefore be **consistently oriented** by the user (normal
//! pointing toward the approaching body): the gap is then positive for a
//! separated point and negative for a penetrated one, which is the sign
//! convention the contact constraint consumes.

use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::parallel::*;

use super::locate::{interpolation_for, reference_centroid, solve_normal};

/// The closest point of a surface mesh to one query point.
#[derive(Clone, Debug)]
pub struct Projection {
    /// Index of the surface submesh the closest facet belongs to.
    pub submesh: usize,
    /// Index of the closest facet within that submesh.
    pub cell: usize,
    /// Reference coordinates `ξ` of the projection in that facet (clamped to
    /// the reference domain — an off-edge point projects onto the edge).
    pub xi: Vec<f64>,
    /// Shape-function weights `Nᵢ(ξ)` at the projection, ordered like the
    /// facet's nodes (sums to 1).
    pub weights: Vec<f64>,
    /// The closest facet's node ids, ordered like `weights`.
    pub nodes: Vec<NodeId>,
    /// The projected point `p = Σᵢ Nᵢ(ξ)·Xᵢ` on the surface.
    pub point: Vec<f64>,
    /// Unit facet normal at the projection (orientation from the facet's node
    /// ordering — see the module documentation).
    pub normal: Vec<f64>,
    /// Signed distance `(x − p)·normal`: positive on the normal's side of the
    /// surface, negative behind it (penetration).
    pub gap: f64,
}

/// Project each point of `points` onto its closest facet of `surface`.
///
/// `surface` must be a **surface mesh** of the ambient space: facets of
/// topological dimension `sdim − 1` (`SEG2`/`SEG3` in 2D, `TRI*`/`QUA*` in 3D).
/// Every point receives a projection (the closest point always exists); an
/// empty surface is an error. Ties between facets (a point facing a shared
/// edge) resolve to the first facet in (submesh, cell) order — deterministic.
pub fn project_points(surface: &Mesh, points: &[Vec<f64>]) -> Result<Vec<Projection>> {
    // Snapshot every facet once (same pattern as `locate_points`): no Coords
    // guard is held during the Gauss–Newton solves.
    struct Facet {
        submesh: usize,
        cell: usize,
        element_type: ElementType,
        nodes: Vec<NodeId>,
        coords: Vec<Vec<f64>>,
        lo: Vec<f64>,
        hi: Vec<f64>,
    }

    let sdim = surface.coords()?.read().dim() as usize;
    let mut facets: Vec<Facet> = Vec::new();
    for (s_idx, sm_handle) in surface.into_iter().enumerate() {
        let (element_type, conn, coords_handle) = {
            let sm = sm_handle.read();
            (sm.element_type(), sm.connectivity().to_vec(), sm.coords())
        };
        if element_type.topological_dim() + 1 != sdim {
            return Err(PyrucastError::Message(format!(
                "project_points: submesh {s_idx} is {element_type} (topological dim {}) \
                 but the surface of a {sdim}-D space must have dim {}",
                element_type.topological_dim(),
                sdim - 1
            )));
        }
        let npc = element_type.nodes_per_cell();
        let c = coords_handle.read();
        for cell in 0..conn.len() / npc {
            let ids = &conn[cell * npc..(cell + 1) * npc];
            let coords: Vec<Vec<f64>> = ids
                .iter()
                .map(|&n| Ok(c.position(n)?.to_vec()))
                .collect::<Result<_>>()?;
            let mut lo = coords[0].clone();
            let mut hi = coords[0].clone();
            for p in &coords[1..] {
                for a in 0..sdim {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            facets.push(Facet {
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
    if facets.is_empty() {
        return Err(PyrucastError::Message(
            "project_points: the surface mesh carries no facet".into(),
        ));
    }

    // One independent projection per point, in parallel. Facets are scanned in
    // (submesh, cell) order with a cheap bbox-distance lower bound pruning the
    // ones that cannot beat the best candidate — result identical to a full
    // scan, so the tie-break is deterministic.
    points
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|x| -> Result<Projection> {
            if x.len() != sdim {
                return Err(PyrucastError::Message(format!(
                    "project_points: point has dim {} but the surface lives in {sdim}-D",
                    x.len()
                )));
            }
            let mut best: Option<(f64, &Facet, Vec<f64>)> = None;
            for facet in &facets {
                if let Some((d2_min, _, _)) = &best {
                    if bbox_dist2(x, &facet.lo, &facet.hi) >= *d2_min {
                        continue; // cannot beat the current best
                    }
                }
                let xi = closest_on_facet(facet.element_type, &facet.coords, x)?;
                let interp = interpolation_for(facet.element_type);
                let n = interp.shape(facet.element_type, &xi)?;
                let mut d2 = 0.0;
                for a in 0..sdim {
                    let mut pa = 0.0;
                    for (i, &ni) in n.iter().enumerate() {
                        pa += ni * facet.coords[i][a];
                    }
                    d2 += (x[a] - pa).powi(2);
                }
                if best.as_ref().is_none_or(|(bd2, _, _)| d2 < *bd2) {
                    best = Some((d2, facet, xi));
                }
            }
            let (_, facet, xi) = best.expect("facets is non-empty");

            // Rebuild the projection data at the winning ξ.
            let interp = interpolation_for(facet.element_type);
            let weights = interp.shape(facet.element_type, &xi)?;
            let mut point = vec![0.0; sdim];
            for (i, &ni) in weights.iter().enumerate() {
                for a in 0..sdim {
                    point[a] += ni * facet.coords[i][a];
                }
            }
            let normal = facet_normal(facet.element_type, &facet.coords, &xi, sdim)?;
            let gap: f64 = (0..sdim).map(|a| (x[a] - point[a]) * normal[a]).sum();

            Ok(Projection {
                submesh: facet.submesh,
                cell: facet.cell,
                xi,
                weights,
                nodes: facet.nodes.clone(),
                point,
                normal,
                gap,
            })
        })
        .collect()
}

/// Squared distance from `x` to the axis-aligned box `[lo, hi]` (0 inside) — the
/// lower bound of the distance to anything inside the box.
fn bbox_dist2(x: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut d2 = 0.0;
    for a in 0..x.len() {
        let d = if x[a] < lo[a] {
            lo[a] - x[a]
        } else if x[a] > hi[a] {
            x[a] - hi[a]
        } else {
            0.0
        };
        d2 += d * d;
    }
    d2
}

/// Projected Gauss–Newton: the `ξ` (clamped to the reference domain) minimising
/// `‖x − F(ξ)‖²` over one facet. Clamping handles edge/corner projections.
fn closest_on_facet(element_type: ElementType, coords: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>> {
    let kind = element_type.as_kind();
    let tdim = element_type.topological_dim();
    let sdim = x.len();
    let npc = element_type.nodes_per_cell();
    let mut xi = reference_centroid(element_type);

    // Scratch buffers reused across the Gauss–Newton steps (see `invert_cell`).
    let mut n = vec![0.0; npc];
    let mut dn = vec![0.0; npc * tdim];
    let mut jac = vec![0.0; sdim * tdim];

    for _ in 0..40 {
        kind.shape_into(&xi, &mut n);
        // Residual r = x − F(ξ)  (length sdim).
        let mut r = x.to_vec();
        for (i, &ni) in n.iter().enumerate() {
            for a in 0..sdim {
                r[a] -= ni * coords[i][a];
            }
        }
        // Jacobian J[a][j] = ∂x_a/∂ξ_j  (sdim × tdim).
        kind.dshape_into(&xi, &mut dn);
        jac.fill(0.0);
        for (i, coord) in coords.iter().enumerate() {
            for a in 0..sdim {
                for j in 0..tdim {
                    jac[a * tdim + j] += dn[i * tdim + j] * coord[a];
                }
            }
        }
        // Gauss–Newton step (least squares on the tangent), then clamp.
        let dxi = solve_normal(&jac, &r, sdim, tdim)?;
        let mut step = 0.0;
        for j in 0..tdim {
            xi[j] += dxi[j];
        }
        clamp_reference(element_type, &mut xi);
        for j in 0..tdim {
            step += dxi[j] * dxi[j];
        }
        if step.sqrt() <= 1e-12 {
            break;
        }
    }
    Ok(xi)
}

/// Clamp `ξ` into `element_type`'s reference domain (in place).
fn clamp_reference(element_type: ElementType, xi: &mut [f64]) {
    element_type.as_kind().clamp_ref(xi);
}

/// Unit facet normal at `ξ`, from the facet's node ordering: the −90°-rotated
/// tangent in 2D, the cross product of the tangents in 3D.
fn facet_normal(
    element_type: ElementType,
    coords: &[Vec<f64>],
    xi: &[f64],
    sdim: usize,
) -> Result<Vec<f64>> {
    let interp = interpolation_for(element_type);
    let tdim = element_type.topological_dim();
    let dn = interp.dshape_dxi(element_type, xi)?;
    // Tangents t_j[a] = Σᵢ ∂Nᵢ/∂ξ_j · X[i][a].
    let mut t = vec![vec![0.0; sdim]; tdim];
    for (i, coord) in coords.iter().enumerate() {
        for (j, tj) in t.iter_mut().enumerate() {
            for a in 0..sdim {
                tj[a] += dn[i * tdim + j] * coord[a];
            }
        }
    }
    let mut n = match (sdim, tdim) {
        (2, 1) => vec![t[0][1], -t[0][0]],
        (3, 2) => vec![
            t[0][1] * t[1][2] - t[0][2] * t[1][1],
            t[0][2] * t[1][0] - t[0][0] * t[1][2],
            t[0][0] * t[1][1] - t[0][1] * t[1][0],
        ],
        _ => unreachable!("facet dimensions validated at snapshot time"),
    };
    let norm = n.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-300 {
        return Err(PyrucastError::Message(
            "project_points: degenerate facet (zero normal)".into(),
        ));
    }
    for v in &mut n {
        *v /= norm;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn mesh_2d_segment() -> Mesh {
        // One SEG2 from (0,0) to (2,0): tangent +x ⇒ normal (0,−1)… wait,
        // n = (t_y, −t_x)/‖t‖ = (0, −1). Points above have negative gap.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        Mesh::from_submesh(sm)
    }

    /// Projection onto a 2-D segment: foot, weights, normal and signed gap.
    #[test]
    fn project_onto_seg2() {
        let mesh = mesh_2d_segment();
        // Above the middle, off the right end, and on the segment.
        let pts = vec![vec![0.5, 1.0], vec![3.0, 1.0], vec![1.0, 0.0]];
        let proj = project_points(&mesh, &pts).unwrap();

        // Interior projection: p = (0.5, 0), weights (0.75, 0.25).
        let p = &proj[0];
        assert!((p.point[0] - 0.5).abs() < 1e-9 && p.point[1].abs() < 1e-9);
        assert!((p.weights[0] - 0.75).abs() < 1e-9);
        assert!((p.weights[1] - 0.25).abs() < 1e-9);
        // Normal (0, −1) from the +x tangent; the point is behind it: gap −1.
        assert!((p.normal[0]).abs() < 1e-9 && (p.normal[1] + 1.0).abs() < 1e-9);
        assert!((p.gap + 1.0).abs() < 1e-9);

        // Off the end: clamped to the corner (2, 0).
        let p = &proj[1];
        assert!((p.point[0] - 2.0).abs() < 1e-9 && p.point[1].abs() < 1e-9);
        assert!((p.weights[1] - 1.0).abs() < 1e-9);

        // On the segment: zero gap.
        assert!(proj[2].gap.abs() < 1e-9);
    }

    /// Projection onto a TRI3 in 3D: interior foot, oriented normal, edge clamp.
    #[test]
    fn project_onto_tri3_3d() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let ids: Vec<_> = v
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Above the interior: normal +z (right-hand rule), gap = height.
        let pts = vec![
            vec![0.25, 0.25, 0.7],
            vec![0.25, 0.25, -0.3],
            vec![2.0, 2.0, 0.0],
        ];
        let proj = project_points(&mesh, &pts).unwrap();

        let p = &proj[0];
        assert!((p.normal[2] - 1.0).abs() < 1e-9);
        assert!((p.gap - 0.7).abs() < 1e-9);
        assert!((p.point[0] - 0.25).abs() < 1e-9 && (p.point[1] - 0.25).abs() < 1e-9);
        // Weights are the barycentric coordinates (0.5, 0.25, 0.25).
        assert!((p.weights[0] - 0.5).abs() < 1e-9);

        // Below: same foot, negative gap (penetration side).
        assert!((proj[1].gap + 0.3).abs() < 1e-9);

        // Far off the hypotenuse (in-plane): clamped onto its midpoint
        // (0.5, 0.5, 0); the offset is tangential, so the normal gap is 0.
        let p = &proj[2];
        assert!((p.point[0] - 0.5).abs() < 1e-9 && (p.point[1] - 0.5).abs() < 1e-9);
        assert!(p.gap.abs() < 1e-9);
    }

    /// Several facets: each point picks its closest one (bbox pruning exact).
    #[test]
    fn picks_the_closest_facet() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        // A polyline y = 0 for x ∈ [0, 4], four segments.
        let ids: Vec<_> = (0..=4)
            .map(|i| {
                Node::create_in(coords.clone(), &[i as f64, 0.0])
                    .unwrap()
                    .id()
            })
            .collect();
        for i in 0..4 {
            sm.add_cell(&[ids[i], ids[i + 1]]).unwrap();
        }
        let mesh = Mesh::from_submesh(sm);

        let pts = vec![vec![0.5, 0.2], vec![3.5, -0.2]];
        let proj = project_points(&mesh, &pts).unwrap();
        assert_eq!(proj[0].cell, 0);
        assert_eq!(proj[1].cell, 3);
        assert!((proj[0].gap + 0.2).abs() < 1e-9); // above ⇒ behind (0,−1)
        assert!((proj[1].gap - 0.2).abs() < 1e-9);
    }

    /// A volume element is not a surface: clear error.
    #[test]
    fn rejects_non_surface_mesh() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let v = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ids: Vec<_> = v
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);
        // TRI3 in 2-D is a *domain*, not a surface of the 2-D space.
        assert!(project_points(&mesh, &[vec![0.0, 0.0]]).is_err());
    }
}
