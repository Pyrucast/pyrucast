//! Topological cleanup of an existing surface mesh: changing who is next to
//! whom, and moving a node only where the change forces it.

use super::Surface;
use crate::containers::mesh::Mesh;
use crate::error::Result;
use crate::ops::mesh::paving::cleanup as pass;

/// Fix the connectivity of `mesh`: remove its doublets, give up the nodes that
/// have only three cells around them, and switch the diagonals that lower the
/// valence error.
///
/// **No node of the contour ever moves**, and none is ever given up: the mesh
/// keeps exactly the boundary it came with. That is the guarantee to hold on
/// to. Inside it, one move does relax the ring it has just sewn — see below.
///
/// Three defects are worth naming, and none is geometry:
///
/// - a **doublet** — an interior node with only two quadrangles around it,
///   which therefore share *two* edges. The node sits in a wedge and both
///   cells stay pinched however they are smoothed; merging the two into one
///   removes node and wedge together.
///
/// - an interior node with only **three cells** around it. Its corners average
///   120° and no smoothing will square them, because the angles around a node
///   sum to 2π whatever the positions. But the star of such a node can always
///   be re-cut with a cell fewer, and the node given up with it: round a node
///   carrying `q` quadrangles and `t` triangles, each quadrangle lays two edges
///   that do not touch it and each triangle lays one, so the star is bounded by
///   a polygon of `n = 2q + t` sides — and a decomposition of an `n`-gon with
///   no interior node satisfies `2q' + t' = n - 2`.
///
///   | `q, t` | boundary | before | after | cells |
///   |---|---|---|---|---|
///   | 3, 0 | hexagon | 3 quadrangles | 2 quadrangles | 3 → 2 |
///   | 2, 1 | pentagon | 2 quadrangles, 1 triangle | 1 of each | 3 → 2 |
///   | 1, 2 | quadrangle | 1 quadrangle, 2 triangles | 1 quadrangle | 3 → 1 |
///   | 0, 3 | triangle | 3 triangles | 1 triangle | 3 → 1 |
///
///   It is the only pass here that removes a node, the only one that changes
///   how many cells there are, and the only one that moves anything. Removing
///   a node leaves the ring round it stretched, so the move is judged **after**
///   relaxing that ring — and the relaxation is kept along with the move,
///   since a verdict on positions one then discards is a verdict on a mesh
///   nobody receives. A move that does not improve the worst cell of the
///   neighbourhood is undone whole, ring positions included.
///
/// - **two** interior nodes sharing an edge, at least one of them short of a
///   cell. Neither can be given up alone: dropping either takes two of its
///   neighbours from four to three, trading one irregular node for two, so the
///   move above refuses it. Together they are a different proposition — their
///   stars overlap along the shared edge, so the two of them carry
///   `val(a) + val(b) - 2` cells, and the same `2q' + t' = n - 2` re-cuts what
///   bounds them:
///
///   | `val(a), val(b)` | boundary | before | after | cells |
///   |---|---|---|---|---|
///   | 3, 3 | hexagon | 4 quadrangles | 2 quadrangles | 4 → 2 |
///   | 3, 4 | heptagon | 4 quadrangles, 1 triangle | 2 of one, 1 of the other | 5 → 3 |
///
///   Two nodes and two cells go at once, and a triangle in the star is carried
///   across rather than created — the parity forbids conjuring one. The pair is
///   looked at before the single node, being both the more specific pattern and
///   the better bargain.
///
/// - a node with the **wrong valence** otherwise. The move is the diagonal
///   switch: two quadrangles sharing an edge form a hexagon, which splits
///   across any of its three diagonals, and switching moves one unit of
///   valence from the two nodes on the old diagonal to the two on the new. It
///   changes no node and no boundary, so it cannot make the mesh
///   non-conforming, and it is applied only when it strictly lowers the total
///   valence error and leaves both cells convex.
///
/// The bottom two rows above shed a triangle each — two of them, so the parity
/// tying their number to the boundary's edge count is untouched. Turning a
/// lone triangle into a quadrangle is a different problem and
/// [`merge_triangles`](fn@super::merge_triangles::merge_triangles)'s job.
///
/// The result is a fresh mesh sharing every node the caller had, save those on
/// a ring a collapse relaxed, which are duplicated so the mesh handed in is
/// left exactly as it was. It comes back wound the way it went in, clockwise
/// or not.
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
/// // La **topologie** : qui est voisin de qui. Le bord n'est jamais touché,
/// // donc un maillage à une seule maille en ressort intact.
/// assert_eq!(mesh::cleanup(&maillage)?.cell_count()?, 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn cleanup(mesh: &Mesh) -> Result<Mesh> {
    let mut surf = Surface::read(mesh, "cleanup")?;
    pass::run(
        &mut surf.pts,
        &surf.movable,
        &mut surf.quads,
        &mut surf.tris,
    );
    surf.to_mesh("cleanup")
}
