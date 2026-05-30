use crate::error::{PyrucastError, Result};
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::Node;
use crate::containers::mesh::Mesh;
use crate::store::with;

/// Fill the interior of one or more closed SEG2 contours with 2-D elements.
///
/// `contour` must be a [`Mesh`] with **one or more** SEG2 submeshes.
/// Each submesh is treated as a single closed simple loop (each node
/// appears once as the start of a segment and once as its end). The
/// `Configuration` can be either:
/// - **2-D** — points are used directly,
/// - **3-D** — every loop must be (nearly) co-planar; an in-plane
///   basis is computed by Newell's method and the points are
///   projected onto the best-fit plane before triangulation. The
///   maximum signed distance from any node to that plane must not
///   exceed `1e-6 × diag`, where `diag` is the AABB diagonal of the
///   union of all loops; otherwise the call fails with a clear error.
///
/// When more than one loop is provided, the **outer boundary** is
/// detected automatically as the loop with the largest signed area
/// (after 2-D projection if needed); the remaining loops are treated
/// as **holes**. Orientation does not matter — every loop is
/// internally re-oriented as needed.
///
/// `element_type` selects the 2-D element to fill with. **Only**
/// [`ElementType::TRI3`] is currently supported; passing any other
/// type returns a clear error.
///
/// Algorithm:
/// - **single loop, no holes** — fast path using plain ear clipping;
///   produces exactly `n - 2` triangles for `n` contour nodes.
/// - **multiple loops (outer + holes)** — constrained Delaunay
///   triangulation (Bowyer-Watson + edge enforcement + parity
///   flood-fill across constrained edges).
///
/// Triangles are oriented **CCW** in the projection plane regardless
/// of the input contour's orientation.
pub fn fill_surface(
    contour: &Mesh,
    element_type: ElementType,
    refinement: Option<crate::ops::mesher::triangulation::RefinementOptions>,
) -> Result<Mesh> {
    if element_type != ElementType::TRI3 {
        return Err(PyrucastError::Message(format!(
            "fill_surface: only TRI3 is supported for now, got {}",
            element_type
        )));
    }
    let n_sub = contour.submesh_count();
    if n_sub == 0 {
        return Err(PyrucastError::Message(
            "fill_surface: contour must contain at least one SEG2 submesh".into(),
        ));
    }
    let cfg = contour.configuration()?;
    let dim = with(&cfg, |c| c.dim())?;
    if dim != 2 && dim != 3 {
        return Err(PyrucastError::Message(format!(
            "fill_surface: contour configuration must be 2-D or 3-D, got dim={}",
            dim
        )));
    }

    // 1. Validate each submesh and extract its ordered closed chain of node ids.
    let mut chains: Vec<Vec<NodeId>> = Vec::with_capacity(n_sub);
    for sm_idx in 0..n_sub {
        let sm = contour.submesh(sm_idx)?;
        let (et, n_elems, conn) = with(&sm, |s| {
            (s.element_type(), s.cell_count(), s.connectivity().to_vec())
        })?;
        if et != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "fill_surface: submesh #{} must be SEG2, got {}",
                sm_idx, et
            )));
        }
        if n_elems < 3 {
            return Err(PyrucastError::Message(format!(
                "fill_surface: submesh #{} must have ≥ 3 segments, got {}",
                sm_idx, n_elems
            )));
        }
        let mut next_node: std::collections::HashMap<NodeId, NodeId> =
            std::collections::HashMap::with_capacity(n_elems);
        for i in 0..n_elems {
            let a = conn[2 * i];
            let b = conn[2 * i + 1];
            if next_node.insert(a, b).is_some() {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: submesh #{}: node {} starts more than one segment",
                    sm_idx, a
                )));
            }
        }
        let start = conn[0];
        let mut chain: Vec<NodeId> = Vec::with_capacity(n_elems);
        chain.push(start);
        let mut current = *next_node.get(&start).ok_or_else(|| {
            PyrucastError::Message(format!(
                "fill_surface: submesh #{}: node {} has no outgoing segment",
                sm_idx, start
            ))
        })?;
        while current != start {
            if chain.len() > n_elems {
                return Err(PyrucastError::Message(format!(
                    "fill_surface: submesh #{}: contour is not a closed simple loop",
                    sm_idx
                )));
            }
            chain.push(current);
            current = *next_node.get(&current).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "fill_surface: submesh #{}: node {} has no outgoing segment",
                    sm_idx, current
                ))
            })?;
        }
        if chain.len() != n_elems {
            return Err(PyrucastError::Message(format!(
                "fill_surface: submesh #{}: contour has multiple disjoint loops ({} nodes traced out of {})",
                sm_idx, chain.len(), n_elems
            )));
        }
        chains.push(chain);
    }

    // 2. Flatten the chains into a single list with per-chain offsets.
    let mut chain_offsets: Vec<usize> = Vec::with_capacity(n_sub + 1);
    chain_offsets.push(0);
    let mut flat_nodes: Vec<NodeId> = Vec::new();
    for chain in &chains {
        flat_nodes.extend_from_slice(chain);
        chain_offsets.push(flat_nodes.len());
    }
    let n_total = flat_nodes.len();

    // 3. Collect 2-D points to triangulate. In 2-D direct (x, y);
    //    in 3-D project on the best-fit plane (Newell normal + centroid origin).
    use crate::containers::mesh::{Point2, Point3, Vector3};
    struct Projection3D {
        origin: Point3,
        u: Vector3,
        v: Vector3,
    }
    let mut projection: Option<Projection3D> = None;
    let points_2d: Vec<Point2> = if dim == 2 {
        let mut pts = Vec::with_capacity(n_total);
        with(&cfg, |c| -> Result<()> {
            for &id in &flat_nodes {
                let s = c.coord(id)?;
                pts.push(Point2::new(s[0], s[1]));
            }
            Ok(())
        })??;
        pts
    } else {
        let mut pts3: Vec<Point3> = Vec::with_capacity(n_total);
        with(&cfg, |c| -> Result<()> {
            for &id in &flat_nodes {
                let s = c.coord(id)?;
                pts3.push(Point3::new(s[0], s[1], s[2]));
            }
            Ok(())
        })??;

        let normal: Vector3 = (0..n_sub)
            .find_map(|i| {
                let pts_chain: Vec<Point3> = (chain_offsets[i]..chain_offsets[i + 1])
                    .map(|j| pts3[j])
                    .collect();
                crate::ops::mesher::triangulation::newell_normal(&pts_chain)
            })
            .ok_or_else(|| {
                PyrucastError::Message(
                    "fill_surface: every 3-D loop is collinear or zero-area".into(),
                )
            })?;

        let origin: Point3 = {
            let sum: Vector3 = pts3.iter().map(|p| p.coords).sum();
            Point3::from(sum / pts3.len() as f64)
        };

        let mut bb_min = Vector3::repeat(f64::INFINITY);
        let mut bb_max = Vector3::repeat(f64::NEG_INFINITY);
        let mut max_dev = 0.0_f64;
        for p in &pts3 {
            let dev = (p - origin).dot(&normal).abs();
            if dev > max_dev {
                max_dev = dev;
            }
            bb_min = bb_min.zip_map(&p.coords, f64::min);
            bb_max = bb_max.zip_map(&p.coords, f64::max);
        }
        let diag = (bb_max - bb_min).norm();
        let tol = 1e-6 * diag;
        if max_dev > tol {
            return Err(PyrucastError::Message(format!(
                "fill_surface: contour is not planar — max deviation {:.3e} exceeds tolerance {:.3e} (1e-6 × diag={:.3e})",
                max_dev, tol, diag
            )));
        }

        let (u, v) = crate::ops::mesher::triangulation::in_plane_basis(normal);
        let pts_2d: Vec<Point2> = pts3
            .iter()
            .map(|p| {
                let d = p - origin;
                Point2::new(d.dot(&u), d.dot(&v))
            })
            .collect();
        projection = Some(Projection3D { origin, u, v });
        pts_2d
    };

    // 4. Triangulate.
    let refine = refinement.filter(|o| o.is_active());
    let (triangles, mut flat_to_node, steiner_points_2d): (
        Vec<[usize; 3]>,
        Vec<NodeId>,
        Vec<Point2>,
    ) = if n_sub == 1 && refine.is_none() {
        let tris = crate::ops::mesher::triangulation::ear_clip_2d(&points_2d)?;
        (tris, flat_nodes, Vec::new())
    } else {
        let mut areas: Vec<f64> = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            let slice = &points_2d[chain_offsets[i]..chain_offsets[i + 1]];
            areas.push(crate::ops::mesher::triangulation::signed_area(slice).abs());
        }
        let outer_idx = (0..n_sub)
            .max_by(|&a, &b| {
                areas[a]
                    .partial_cmp(&areas[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let mut outer_pts: Vec<Point2> = Vec::new();
        let mut new_flat_nodes: Vec<NodeId> = Vec::new();
        for j in chain_offsets[outer_idx]..chain_offsets[outer_idx + 1] {
            outer_pts.push(points_2d[j]);
            new_flat_nodes.push(flat_nodes[j]);
        }
        let mut hole_pts_list: Vec<Vec<Point2>> = Vec::new();
        for i in 0..n_sub {
            if i == outer_idx {
                continue;
            }
            let mut hole_pts = Vec::new();
            for j in chain_offsets[i]..chain_offsets[i + 1] {
                hole_pts.push(points_2d[j]);
                new_flat_nodes.push(flat_nodes[j]);
            }
            hole_pts_list.push(hole_pts);
        }

        let n_existing = new_flat_nodes.len();
        if let Some(opts) = refine {
            let (all_pts, tris) =
                crate::ops::mesher::triangulation::triangulate_polygon_with_holes_refined(
                    &outer_pts,
                    &hole_pts_list,
                    opts,
                )?;
            let steiner = all_pts[n_existing..].to_vec();
            (tris, new_flat_nodes, steiner)
        } else {
            let tris = crate::ops::mesher::triangulation::triangulate_polygon_with_holes(
                &outer_pts,
                &hole_pts_list,
            )?;
            (tris, new_flat_nodes, Vec::new())
        }
    };

    // 5. Create one Configuration node per Steiner point.
    let mut _steiner_nodes: Vec<Node> = Vec::with_capacity(steiner_points_2d.len());
    for p in &steiner_points_2d {
        let coords: Vec<f64> = match &projection {
            None => vec![p.x, p.y],
            Some(proj) => {
                let p3 = proj.origin + proj.u * p.x + proj.v * p.y;
                vec![p3.x, p3.y, p3.z]
            }
        };
        let node = Node::create_in(cfg.clone(), &coords)?;
        flat_to_node.push(node.id());
        _steiner_nodes.push(node);
    }

    // 6. Build the TRI3 mesh.
    let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
    for [i, j, k] in triangles {
        mesh.add_cell(&[flat_to_node[i], flat_to_node[j], flat_to_node[k]])?;
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::store::{insert, with, Handle};

    fn build_contour_2d(cfg: Handle<Configuration>, pts: &[(f64, f64)]) -> (Mesh, Vec<Node>) {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y)| Node::create_in(cfg.clone(), &[x, y]).unwrap())
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        (contour, nodes)
    }

    fn build_contour_3d(cfg: Handle<Configuration>, pts: &[(f64, f64, f64)]) -> (Mesh, Vec<Node>) {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y, z)| Node::create_in(cfg.clone(), &[x, y, z]).unwrap())
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        (contour, nodes)
    }

    #[test]
    fn fill_surface_square_gives_two_triangles() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);

        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count().unwrap(), 2);

        let node_ids: std::collections::HashSet<_> = nodes.iter().map(|n| n.id()).collect();
        for ci in 0..2 {
            for ni in 0..3 {
                let id = tri.node(0, ci, ni).unwrap().id();
                assert!(node_ids.contains(&id), "triangle node {} not in contour", id);
            }
        }
    }

    #[test]
    fn fill_surface_triangles_sum_to_polygon_area() {
        let cfg = insert(Configuration::new(2).unwrap());
        let l = [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ];
        let (contour, _nodes) = build_contour_2d(cfg.clone(), &l);

        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 4);

        let mut total = 0.0;
        for ci in 0..4 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW (signed area {})", ci, a);
            total += a;
        }
        assert!((total - 5.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_increfs_contour_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let ids: Vec<_> = nodes.iter().map(|n| n.id()).collect();

        with(&cfg, |c| {
            for &id in &ids {
                assert_eq!(c.refcount(id), 3);
            }
        })
        .unwrap();

        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();

        let mut extra = [0u32; 4];
        for ci in 0..2 {
            for ni in 0..3 {
                let id = tri.node(0, ci, ni).unwrap().id();
                let k = ids.iter().position(|&x| x == id).unwrap();
                extra[k] += 1;
            }
        }
        with(&cfg, |c| {
            for k in 0..4 {
                assert_eq!(c.refcount(ids[k]), 3 + extra[k]);
            }
        })
        .unwrap();
    }

    #[test]
    fn fill_surface_rejects_non_tri3() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, _n) =
            build_contour_2d(cfg, &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert!(fill_surface(&contour, ElementType::QUA4, None).is_err());
    }

    #[test]
    fn fill_surface_rejects_non_seg2_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let mut bogus = Mesh::with_element_type(cfg, ElementType::TRI3);
        bogus.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(fill_surface(&bogus, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_rejects_dim_above_three() {
        let cfg = insert(Configuration::new(4).unwrap());
        let nodes: Vec<Node> = (0..4)
            .map(|i| {
                let t = i as f64;
                Node::create_in(cfg.clone(), &[t, 0.0, 0.0, 0.0]).unwrap()
            })
            .collect();
        let mut contour = Mesh::with_element_type(cfg, ElementType::SEG2);
        for i in 0..4 {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % 4].id()])
                .unwrap();
        }
        assert!(fill_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_3d_square_in_z_plane() {
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 5.0),
                (1.0, 0.0, 5.0),
                (1.0, 1.0, 5.0),
                (0.0, 1.0, 5.0),
            ],
        );

        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        for ci in 0..2 {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 5.0).abs() < 1e-12);
            }
        }

        let mut total = 0.0;
        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_3d_tilted_square() {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, s, s), (0.0, s, s)],
        );
        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        let mut total = 0.0;
        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fill_surface_3d_rejects_non_planar_contour() {
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.5),
                (0.0, 1.0, 0.0),
            ],
        );
        let err = fill_surface(&contour, ElementType::TRI3, None).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("not planar"), "unexpected message: {}", msg);
    }

    #[test]
    fn fill_surface_3d_accepts_tiny_numerical_noise() {
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _nodes) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 1e-10),
                (0.0, 1.0, 0.0),
            ],
        );
        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
    }

    #[test]
    fn fill_surface_rejects_empty_submesh() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (mut contour, _n) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let extra = insert(SubMesh::new(cfg, ElementType::SEG2));
        contour.add_sub(extra).unwrap();
        assert!(fill_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_with_one_hole_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _no) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _nh) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let combined = (&outer + &hole).unwrap();
        assert_eq!(combined.submesh_count(), 2);

        let tri = fill_surface(&combined, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);

        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW (signed area {})", ci, a);
            total += a;
        }
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_outer_loop_is_autodetected() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (hole, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let combined = (&hole + &outer).unwrap();
        let tri = fill_surface(&combined, ElementType::TRI3, None).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_two_holes_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)],
        );
        let (h1, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
        );
        let (h2, _) = build_contour_2d(
            cfg.clone(),
            &[(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)],
        );
        let combined = (&(&outer + &h1).unwrap() + &h2).unwrap();
        assert_eq!(combined.submesh_count(), 3);
        let tri = fill_surface(&combined, ElementType::TRI3, None).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        assert!((total - 22.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_one_hole_3d() {
        let cfg = insert(Configuration::new(3).unwrap());
        let (outer, _) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 1.0),
                (4.0, 0.0, 1.0),
                (4.0, 4.0, 1.0),
                (0.0, 4.0, 1.0),
            ],
        );
        let (hole, _) = build_contour_3d(
            cfg.clone(),
            &[
                (1.0, 1.0, 1.0),
                (3.0, 1.0, 1.0),
                (3.0, 3.0, 1.0),
                (1.0, 3.0, 1.0),
            ],
        );
        let combined = (&outer + &hole).unwrap();

        let tri = fill_surface(&combined, ElementType::TRI3, None).unwrap();

        let n_cells = tri.cell_count().unwrap();
        for ci in 0..n_cells {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 1.0).abs() < 1e-12);
            }
        }
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            total += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        assert!((total - 12.0).abs() < 1e-9, "total area = {}", total);
    }

    #[test]
    fn fill_surface_with_hole_rejects_different_configurations() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg1.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _) = build_contour_2d(
            cfg2,
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        assert!((&outer + &hole).is_err());
    }

    #[test]
    fn fill_surface_rejects_open_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mut open = Mesh::with_element_type(cfg, ElementType::SEG2);
        open.add_cell(&[a.id(), b.id()]).unwrap();
        open.add_cell(&[b.id(), c.id()]).unwrap();
        assert!(fill_surface(&open, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn fill_surface_refined_2d_square_creates_steiner_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, contour_nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
        let initial_node_count = with(&cfg, |c| c.node_count()).unwrap();

        let opts = crate::ops::mesher::triangulation::RefinementOptions {
            max_edge_length: Some(1.5),
            min_angle_deg: None,
        };
        let tri = fill_surface(&contour, ElementType::TRI3, Some(opts)).unwrap();

        let new_node_count = with(&cfg, |c| c.node_count()).unwrap();
        assert!(
            new_node_count > initial_node_count,
            "no Steiner nodes added: was {}, still {}",
            initial_node_count,
            new_node_count
        );

        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        let mut max_edge = 0.0_f64;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW", ci);
            total += a;
            for (u, v) in [
                (p0.as_slice(), p1.as_slice()),
                (p1.as_slice(), p2.as_slice()),
                (p2.as_slice(), p0.as_slice()),
            ] {
                let dx = v[0] - u[0];
                let dy = v[1] - u[1];
                max_edge = max_edge.max((dx * dx + dy * dy).sqrt());
            }
        }
        assert!((total - 16.0).abs() < 1e-9);
        assert!(max_edge <= 1.5 + 1e-9, "max edge length {} > 1.5", max_edge);

        for n in &contour_nodes {
            assert!(with(&cfg, |c| c.is_alive(n.id())).unwrap());
        }
    }

    #[test]
    fn fill_surface_refined_inactive_options_is_noop() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, _nodes) =
            build_contour_2d(cfg.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let initial_count = with(&cfg, |c| c.node_count()).unwrap();
        let tri = fill_surface(&contour, ElementType::TRI3, Some(Default::default())).unwrap();
        let final_count = with(&cfg, |c| c.node_count()).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert_eq!(initial_count, final_count, "no Steiner expected");
    }

    #[test]
    fn fill_surface_refined_3d_keeps_steiner_in_plane() {
        let cfg = insert(Configuration::new(3).unwrap());
        let (contour, _) = build_contour_3d(
            cfg.clone(),
            &[
                (0.0, 0.0, 1.0),
                (4.0, 0.0, 1.0),
                (4.0, 4.0, 1.0),
                (0.0, 4.0, 1.0),
            ],
        );
        let opts = crate::ops::mesher::triangulation::RefinementOptions {
            max_edge_length: Some(1.5),
            min_angle_deg: None,
        };
        let tri = fill_surface(&contour, ElementType::TRI3, Some(opts)).unwrap();
        let n_cells = tri.cell_count().unwrap();
        assert!(n_cells > 2, "no refinement happened: got only {} cells", n_cells);
        for ci in 0..n_cells {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 1.0).abs() < 1e-9, "Steiner node off plane: z={}", p[2]);
            }
        }
    }

    #[test]
    fn fill_surface_refined_with_hole_conserves_area() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (outer, _) = build_contour_2d(
            cfg.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let (hole, _) = build_contour_2d(
            cfg.clone(),
            &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        );
        let combined = (&outer + &hole).unwrap();
        let opts = crate::ops::mesher::triangulation::RefinementOptions {
            max_edge_length: Some(1.0),
            min_angle_deg: None,
        };
        let tri = fill_surface(&combined, ElementType::TRI3, Some(opts)).unwrap();
        let n_cells = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n_cells {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            total += 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        }
        assert!((total - 12.0).abs() < 1e-9, "area drift: {}", total);
    }

    #[test]
    fn fill_surface_works_with_cw_contour() {
        let cfg = insert(Configuration::new(2).unwrap());
        let (contour, _n) =
            build_contour_2d(cfg, &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);
        let tri = fill_surface(&contour, ElementType::TRI3, None).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);

        for ci in 0..2 {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW", ci);
        }
    }
}
