//! Projection of a mesh onto a surface: [`project_on_sphere`].
//!
//! Like the rigid copies of [`super::transform`], it builds a **brand-new
//! mesh with its own fresh nodes**, the source left untouched: same zones,
//! element types, face colours and connectivity structure, nodes shared
//! between cells of the source shared in the copy.

use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};

/// Project `mesh` **radially** onto the sphere of centre `center` and radius
/// `radius`: every node — corners and mid-nodes alike — is pushed along the
/// ray from `center` through it, to the distance `radius`.
///
/// The result is a fresh copy with its own nodes (one per distinct node of
/// `mesh`, so a closed loop stays closed), in the same zones, element types
/// and face colours. In a plane through `center`, the sphere is a circle: a
/// line of that plane lands on the circle, each node at its own angle. That
/// is how one loop is laid onto another one node for node — a hole's
/// boundary built from the outer contour around it, ready for
/// [`sweep`](fn@super::sweep).
///
/// `center` matches the mesh dimension; `radius` must be positive and
/// finite. A node sitting on `center` has no direction to be pushed along:
/// it is an error, and no node is created.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// // A straight side, projected onto the circle of radius 1 around the
/// // origin: its far end (2, 2) lands at 45°.
/// let side = mesh::line(&p(&[2.0, 0.0]), &p(&[2.0, 2.0]), 2, ElementType::SEG2)?;
/// let on_circle = mesh::project_on_sphere(&side, &[0.0, 0.0], 1.0)?;
/// let end = on_circle.node(0, 1, 1)?.position()?;
/// assert!((end[0] - 0.5f64.sqrt()).abs() < 1e-12 && (end[1] - 0.5f64.sqrt()).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn project_on_sphere(mesh: &Mesh, center: &[f64], radius: f64) -> Result<Mesh> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(PyrucastError::Message(format!(
            "project_on_sphere: radius must be positive and finite, got {radius}"
        )));
    }
    let dim = mesh.coords()?.read().dim() as usize;
    if center.len() != dim {
        return Err(PyrucastError::Message(format!(
            "project_on_sphere: center has {} components but the mesh is {dim}-D",
            center.len()
        )));
    }
    super::transform::map_coords(mesh, false, |p| {
        let distance = p
            .iter()
            .zip(center)
            .map(|(x, c)| (x - c) * (x - c))
            .sum::<f64>()
            .sqrt();
        if distance == 0.0 {
            return Err(PyrucastError::Message(format!(
                "project_on_sphere: the node at {p:?} sits on the centre, it has no direction \
                 to be projected along"
            )));
        }
        let scale = radius / distance;
        Ok(p.iter()
            .zip(center)
            .map(|(x, c)| c + scale * (x - c))
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::mesh::{circle, line, to_quadratic};
    use std::collections::HashSet;

    fn distinct(mesh: &Mesh) -> HashSet<NodeId> {
        mesh.get(0)
            .unwrap()
            .read()
            .connectivity()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn every_node_lands_at_the_radius_and_sharing_is_kept() {
        // A quadratic polyline off-centre in 3-D: corners and mid-nodes all
        // end up at the radius, one fresh node per distinct node.
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
        let source = to_quadratic(
            &line(
                &p(&[3.0, -1.0, 0.5]),
                &p(&[3.0, 4.0, 2.0]),
                5,
                ElementType::SEG2,
            )
            .unwrap(),
        )
        .unwrap();
        let c = [1.0, 1.0, 1.0];
        let projected = project_on_sphere(&source, &c, 0.75).unwrap();
        assert_eq!(projected.element_types().unwrap(), vec![ElementType::SEG3]);
        assert_eq!(projected.cell_count(), source.cell_count());
        assert_eq!(distinct(&projected).len(), distinct(&source).len());
        assert!(distinct(&projected).is_disjoint(&distinct(&source)));
        for cell in 0..5 {
            for k in 0..3 {
                let x = projected.node(0, cell, k).unwrap().position().unwrap();
                let r = x
                    .iter()
                    .zip(&c)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                assert!((r - 0.75).abs() < 1e-12, "cell {cell}, node {k}: r = {r}");
            }
        }
    }

    #[test]
    fn a_closed_loop_stays_closed_and_a_node_on_the_sphere_stays_put() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let o = Node::create_in(coords.clone(), &[0.5, -0.5]).unwrap();
        let ring = circle(&o, &[0.0, 0.0, 1.0], 2.0, 8, ElementType::SEG2).unwrap();
        let same = project_on_sphere(&ring, &[0.5, -0.5], 2.0).unwrap();
        assert_eq!(distinct(&same).len(), 8);
        assert_eq!(
            same.node(0, 7, 1).unwrap().id(),
            same.node(0, 0, 0).unwrap().id()
        );
        for cell in 0..8 {
            let (a, b) = (
                ring.node(0, cell, 0).unwrap().position().unwrap(),
                same.node(0, cell, 0).unwrap().position().unwrap(),
            );
            assert!((a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15);
        }
    }

    #[test]
    fn bad_radius_wrong_centre_and_a_node_at_the_centre_are_refused() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
        let through_origin = line(&p(&[0.0, 0.0]), &p(&[1.0, 0.0]), 1, ElementType::SEG2).unwrap();
        assert!(project_on_sphere(&through_origin, &[0.0, 5.0], 0.0).is_err());
        assert!(project_on_sphere(&through_origin, &[0.0, 5.0], f64::NAN).is_err());
        assert!(project_on_sphere(&through_origin, &[0.0, 5.0, 0.0], 1.0).is_err());
        let before = coords.read().capacity();
        assert!(project_on_sphere(&through_origin, &[0.0, 0.0], 1.0).is_err());
        assert_eq!(
            coords.read().capacity(),
            before,
            "a refused projection creates no node"
        );
    }
}
