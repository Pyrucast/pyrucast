//! Smoothing an existing surface mesh: moving its nodes, and nothing else.

use super::Surface;
use crate::containers::mesh::Mesh;
use crate::error::{PyrucastError, Result};
use crate::ops::mesh::paving::smooth::{self, Incidence, Patch, Rule};

/// Move the interior nodes of `mesh` to improve its cells, leaving the
/// connectivity and the boundary exactly as they are.
///
/// `sweeps` is how many passes to run. `angular` picks the rule: `true` for
/// angle-based smoothing, which aims at the right angles a quadrangle wants;
/// `false` for the plain Laplacian, which aims at the one-ring's barycentre and
/// knows nothing about angles. `in_place` writes the new positions onto the
/// caller's own nodes and hands the same mesh back; otherwise the moved nodes
/// are duplicated and a fresh mesh comes out, the boundary's nodes being shared
/// since they never moved.
///
/// Two things are guaranteed whatever the rule, because the sweep is the
/// pavers' own: **no node on the boundary ever moves**, and a candidate
/// position is taken only when every incident cell stays valid *and* the worst
/// incident quality does not get worse. A plain Laplacian pass has neither
/// guarantee and will happily turn a cell inside out against a concave
/// boundary.
///
/// Smoothing cannot change who is next to whom, so it cannot fix a node with
/// the wrong number of cells around it — the angles around a node sum to 2π
/// whatever the positions. That is [`cleanup`](fn@super::cleanup::cleanup)'s
/// job, and running it first is usually what unlocks the smoothing.
///
/// Only `TRI3` and `QUA4` cells, in 2-D. `POI1` and `SEG2` submeshes are
/// ignored, anything else is an error.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // La **géométrie** : où sont les nœuds. Tout nœud de bord est épinglé,
/// // de sorte que le maillage garde exactement le contour reçu — et qu'un
/// // maillage d'une seule maille, sans intérieur, est refusé en le disant.
/// assert!(mesh::regularize(&maillage, 3, false, false).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn regularize(mesh: &Mesh, sweeps: usize, angular: bool, in_place: bool) -> Result<Mesh> {
    let mut surf = Surface::read(mesh, "regularize")?;
    if surf.pinned() == surf.pts.len() {
        return Err(PyrucastError::Message(
            "regularize: every node is on the boundary — there is nothing to move. A mesh one \
             cell deep has no interior; refine it first."
                .into(),
        ));
    }

    let patch = Patch {
        quads: &surf.quads,
        tris: &surf.tris,
        movable: &surf.movable,
    };
    let inc = Incidence::build(&patch, surf.pts.len());
    let rule = if angular {
        Rule::Angular
    } else {
        Rule::Laplacian
    };
    smooth::smooth_with(&mut surf.pts, &patch, &inc, None, sweeps, rule);

    if in_place {
        surf.write_positions(mesh)
    } else {
        surf.to_mesh("regularize")
    }
}
