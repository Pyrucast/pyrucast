//! Between the physical unknowns and the coordinates of a list of modes
//! ([`Mode`]): [`project_on_basis`] brings a field of forces onto the modes,
//! [`recombine`] brings modal coordinates back to the physical unknowns. Their
//! matrix twin is [`crate::ops::matrix::project_on_basis`].

use crate::aggregate::Aggregate;
use crate::containers::field::{Field, SubField};
use crate::containers::matrix::dof_key;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::ops::solver::{real_support_zone, Mode, MODAL_COORDINATE, MODAL_FORCE};

/// `field`, a field of forces, projected on `modes`: `F*ᵢ = φᵢᵀ F`, at the
/// node of each mode, component [`MODAL_FORCE`], on the modes' support.
///
/// The product is that of `xty`: each component of a shape multiplies the
/// component of `field` its pair names (`u_x` × `f_x`), zone by zone on the
/// same support. Components of `field` no pair names (`imposed`…) do not
/// count. The result lies on the modes' own support, as a matrix projected on
/// them does: `solve` takes them together.
///
/// An empty list gives an empty field.
///
/// # Errors
///
/// - `field` carries none of the modes' duals on the supports of their shapes
///   (a displacement given for a force, or a field on other supports:
///   `restrict_like` brings it onto theirs);
/// - the modes do not share one support whose nodes are exactly theirs, or
///   one is complex ([`Mode`]).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::containers::node_field::{NodeField, SubNodeField};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{node_field, solver};
/// # use pyrucast::ops::solver::Normalization;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let p = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[p.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let point = |value: f64| -> pyrucast::Result<Matrix> {
/// #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
/// #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
/// #     z.add_entry(p.id(), "f", p.id(), "u", value)?;
/// #     let mut m = Matrix::empty();
/// #     m.add_sub(Handle::new(z))?;
/// #     m.assemble()?;
/// #     Ok(m)
/// # };
/// let modes = solver::eigen(&point(4e5)?, &point(40.0)?, 1, 0.0, Normalization::Max)?;
/// let mut force = SubNodeField::from_poi1(&support, vec!["f".into()])?;
/// force.set_value(p.id(), "f", 2.0)?;
/// // φ = 1 at the point: the modal force is the force.
/// let projected = node_field::project_on_basis(&NodeField::from_sub(force), &modes)?;
/// assert_eq!(projected.value(modes[0].node.id(), "f_alpha")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn project_on_basis(field: &NodeField, modes: &[Mode]) -> Result<NodeField> {
    if modes.is_empty() {
        return Ok(NodeField::default());
    }
    let (zone, _) = real_support_zone(modes, "project_on_basis")?;
    // A zone of `field` that carries a dual on the support of a shape's zone.
    let meets = |mode: &Mode| {
        field.iter().any(|f| {
            let f = f.read();
            mode.shape
                .iter()
                .any(|s| s.read().support().same_object(&f.support()))
                && mode
                    .pairs
                    .iter()
                    .any(|(_, dual)| f.components().contains(dual))
        })
    };
    if !modes.iter().any(meets) {
        return Err(PyrucastError::Message(
            "project_on_basis: the field carries none of the modes' duals on the supports of \
             their shapes; bring it onto them with restrict_like"
                .into(),
        ));
    }
    let mut projected = SubNodeField::from_poi1(&zone, vec![MODAL_FORCE.into()])?;
    for mode in modes {
        let pairs: Vec<(&str, &str)> = mode
            .pairs
            .iter()
            .map(|(p, d)| (p.as_str(), d.as_str()))
            .collect();
        let value = mode.shape.dot_field(field, Some(&pairs));
        projected.set_value(mode.node.id(), MODAL_FORCE, value)?;
    }
    Ok(NodeField::from_sub(projected))
}

/// The physical field of the modal coordinates `coordinates`:
/// `u = Σ qᵢ φᵢ`, `qᵢ` read at the node of mode `i`, component
/// [`MODAL_COORDINATE`].
///
/// The coordinates are read node by node, as `solve` returns them on the modes'
/// support; a node that does not carry them reads `0`. The sum keeps the
/// supports of the shapes, those of the matrix the modes come from: a static
/// solution and a recombined field combine.
///
/// An empty list gives an empty field.
///
/// # Errors
///
/// - `coordinates` does not carry [`MODAL_COORDINATE`];
/// - the modes do not share one support whose nodes are exactly theirs, or
///   one is complex ([`Mode`]);
/// - two shapes lie on supports their sum cannot combine.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::containers::node_field::{NodeField, SubNodeField};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{node_field, solver};
/// # use pyrucast::ops::solver::Normalization;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let p = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[p.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let point = |value: f64| -> pyrucast::Result<Matrix> {
/// #     let mut z = SubMatrix::new(support.clone(), support.clone(), vec!["f".into()],
/// #         vec!["u".into()], DofOrdering::NodesThenVars, Symmetry::Full)?;
/// #     z.add_entry(p.id(), "f", p.id(), "u", value)?;
/// #     let mut m = Matrix::empty();
/// #     m.add_sub(Handle::new(z))?;
/// #     m.assemble()?;
/// #     Ok(m)
/// # };
/// let modes = solver::eigen(&point(4e5)?, &point(40.0)?, 1, 0.0, Normalization::Max)?;
/// let mut q = SubNodeField::from_poi1(&modes[0].support.unit()?, vec!["alpha".into()])?;
/// q.set_value(modes[0].node.id(), "alpha", 3.0)?;
/// // u = 3·φ, and φ = 1 at the point.
/// let u = node_field::recombine(&NodeField::from_sub(q), &modes)?;
/// assert_eq!(u.value(p.id(), "u")?, 3.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn recombine(coordinates: &NodeField, modes: &[Mode]) -> Result<NodeField> {
    if modes.is_empty() {
        return Ok(NodeField::default());
    }
    real_support_zone(modes, "recombine")?;
    if !coordinates
        .components()
        .iter()
        .any(|c| c == MODAL_COORDINATE)
    {
        return Err(PyrucastError::Message(format!(
            "recombine: the coordinates carry no `{MODAL_COORDINATE}`"
        )));
    }
    let keys: Vec<_> = modes
        .iter()
        .map(|mode| dof_key(mode.node.id(), 0))
        .collect();
    let q = coordinates.gather_keys(&keys, &[MODAL_COORDINATE.to_string()]);
    let mut u = modes[0].shape.combine_scalar(|v, s| v * s, q[0])?;
    for (mode, &qi) in modes.iter().zip(&q).skip(1) {
        u = u.merge_field(&mode.shape.combine_scalar(|v, s| v * s, qi)?, |a, b| a + b)?;
    }
    Ok(u)
}

// The tests of the three operators between physical unknowns and modal
// coordinates live together, the matrix projection's included: they share
// their chain of springs and its modes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::models::RelationSense;
    use crate::ops::solver::{eigen, quadratic_eigen, Normalization};
    use crate::ops::{element_field, matrix, mesh, model};

    /// A chain of three unit springs (heat conduction, k = 1) held at its
    /// first node by a relation, a unit mass on each other node, and its three
    /// modes normalized by the mass.
    fn chain() -> (Matrix, Matrix, Vec<Node>, Vec<Mode>) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n: Vec<Node> = (0..4)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        for i in 0..3 {
            sm.add_cell(&[n[i].id(), n[i + 1].id()]).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
        let springs = model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
        let held = mesh::poi1_from_nodes(&n[..1]).unwrap();
        let held = model::dirichlet(&held, &springs, "T", RelationSense::Equality).unwrap();
        let both = springs.union(&held).unwrap();
        let materials = element_field::material_field(&both, &[("k", 1.0)]).unwrap();
        let k = matrix::stiffness(&both, &materials).unwrap();
        let support = mesh::poi1_from_nodes(&n[1..]).unwrap().unit().unwrap();
        let mut z = SubMatrix::new(
            support.clone(),
            support,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        )
        .unwrap();
        for node in &n[1..] {
            z.add_entry(node.id(), "q", node.id(), "T", 1.0).unwrap();
        }
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(z)).unwrap();
        m.assemble().unwrap();
        let modes = eigen(&k, &m, 3, 0.0, Normalization::Mass).unwrap();
        (k, m, n, modes)
    }

    /// `mode` again, on the support `zone`: its handles shared.
    fn moved(mode: &Mode, zone: &Handle<SubMesh>) -> Mode {
        let mut support = Mesh::default();
        support.add_sub(zone.clone()).unwrap();
        Mode {
            node: mode.node.clone(),
            support,
            eigenvalue: mode.eigenvalue,
            eigenvalue_imag: mode.eigenvalue_imag,
            shape: mode.shape.union(&NodeField::default()).unwrap(),
            shape_imag: None,
            pairs: mode.pairs.clone(),
            generalized_mass: mode.generalized_mass,
        }
    }

    #[test]
    fn project_on_basis_gives_identity_mass_and_diagonal_stiffness() {
        let (k, m, _, modes) = chain();
        // `k` carries its relation: its row and its multiplier pair nothing.
        let k_star = matrix::project_on_basis(&k, &modes).unwrap();
        let m_star = matrix::project_on_basis(&m, &modes).unwrap();
        for (i, a) in modes.iter().enumerate() {
            for (j, b) in modes.iter().enumerate() {
                let (ka, kb) = (a.node.id(), b.node.id());
                let kij = k_star.get(ka, MODAL_FORCE, kb, MODAL_COORDINATE).unwrap();
                let mij = m_star.get(ka, MODAL_FORCE, kb, MODAL_COORDINATE).unwrap();
                let delta = f64::from(u8::from(i == j));
                assert!((mij - delta).abs() < 1e-12, "M*[{i}][{j}] = {mij}");
                assert!(
                    (kij - delta * a.eigenvalue).abs() < 1e-12,
                    "K*[{i}][{j}] = {kij}"
                );
                assert_eq!(
                    kij,
                    k_star.get(kb, MODAL_FORCE, ka, MODAL_COORDINATE).unwrap()
                );
            }
        }
        let block = k_star.iter().next().unwrap().read().row_support().clone();
        assert!(block.same_object(&modes[0].support.unit().unwrap()));
    }

    #[test]
    fn project_then_recombine_reproduces_a_field_of_the_span() {
        let (_, m, n, modes) = chain();
        // u = Σ cᵢφᵢ, read node by node: on modes normalized by the mass,
        // F = M u projects to c, and the coordinates c recombine to u.
        let c = [0.3, -1.2, 2.0];
        let expected: Vec<f64> = n
            .iter()
            .map(|node| {
                modes
                    .iter()
                    .zip(c)
                    .map(|(mode, ci)| ci * mode.shape.value(node.id(), "T").unwrap())
                    .sum()
            })
            .collect();
        let support = m.iter().next().unwrap().read().col_support().clone();
        let mut u = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        for (node, value) in n[1..].iter().zip(&expected[1..]) {
            u.set_value(node.id(), "T", *value).unwrap();
        }
        let force = m.mul_field(&NodeField::from_sub(u)).unwrap();
        let f_star = project_on_basis(&force, &modes).unwrap();
        let m_star = matrix::project_on_basis(&m, &modes).unwrap();
        let q = crate::ops::solver::solve(&m_star, &f_star).unwrap();
        for (mode, ci) in modes.iter().zip(c) {
            assert!((q.value(mode.node.id(), MODAL_COORDINATE).unwrap() - ci).abs() < 1e-12);
        }
        // The projected force and the projected matrix share their support.
        let residual = f_star
            .merge_field(&m_star.mul_field(&q).unwrap(), |a, b| a - b)
            .unwrap();
        assert!(residual.norm(0, None) < 1e-12);
        let back = recombine(&q, &modes).unwrap();
        for (node, value) in n.iter().zip(&expected) {
            assert!((back.value(node.id(), "T").unwrap() - value).abs() < 1e-12);
        }
    }

    #[test]
    fn a_slice_with_its_own_support_projects_by_node() {
        let (k, _, _, modes) = chain();
        let full = matrix::project_on_basis(&k, &modes).unwrap();
        // Modes 2 and 1, in that order, on a support of their own.
        let nodes = [modes[2].node.clone(), modes[1].node.clone()];
        let zone = mesh::poi1_from_nodes(&nodes).unwrap().unit().unwrap();
        let part = [moved(&modes[2], &zone), moved(&modes[1], &zone)];
        let k_part = matrix::project_on_basis(&k, &part).unwrap();
        for a in &nodes {
            for b in &nodes {
                let get = |m: &Matrix| {
                    m.get(a.id(), MODAL_FORCE, b.id(), MODAL_COORDINATE)
                        .unwrap()
                };
                // The symmetric half is computed the other way round: to rounding.
                assert!((get(&k_part) - get(&full)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn modal_consumers_refuse_inconsistent_bases() {
        let (k, m, _, modes) = chain();
        let err = |r: Result<Matrix>| r.unwrap_err().to_string();
        // A slice that kept the full support: the error gives the way out.
        let e = err(matrix::project_on_basis(&k, &modes[..2]));
        assert!(
            e.contains("holds 3 nodes for 2 modes") && e.contains("poi1_from_nodes"),
            "{e}"
        );
        // Two modes on two supports.
        let other = mesh::poi1_from_nodes(&[modes[0].node.clone()])
            .unwrap()
            .unit()
            .unwrap();
        let split = [
            moved(&modes[0], &other),
            moved(&modes[1], &modes[1].support.unit().unwrap()),
        ];
        assert!(err(matrix::project_on_basis(&k, &split)).contains("do not share one support"));
        // A complex mode, from the projected problem.
        let k_star = matrix::project_on_basis(&k, &modes).unwrap();
        let m_star = matrix::project_on_basis(&m, &modes).unwrap();
        let complex =
            quadratic_eigen(&m_star, &k_star, &Matrix::default(), Normalization::Mass).unwrap();
        assert!(err(matrix::project_on_basis(&k, &complex)).contains("complex"));
        // A displacement given for a force.
        let e = project_on_basis(&modes[0].shape, &modes)
            .unwrap_err()
            .to_string();
        assert!(e.contains("none of the modes' duals"), "{e}");
        // Coordinates without alpha.
        let e = recombine(&modes[0].shape, &modes).unwrap_err().to_string();
        assert!(e.contains("carry no `alpha`"), "{e}");
        // No mode: nothing to project on, an empty result.
        assert!(matrix::project_on_basis(&k, &[]).unwrap().is_empty());
        assert!(recombine(&modes[0].shape, &[]).unwrap().is_empty());
    }
}
