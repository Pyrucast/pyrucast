//! Distributed-flux load operator — the analogue of Cast3m `FLUX` / `PRES`.
//!
//! [`flux`] integrates a scalar flux density `φ` against the shape functions of
//! an FE subspace and returns the **consistent nodal loads**
//!
//! ```text
//! f_i = ∫_Γ φ N_i dΓ  ≈  Σ_cell Σ_g φ(cell,g) · N_i(ξ_g) · |J|_g · w_g
//! ```
//!
//! accumulated per node into a [`SubNodeField`] whose single component is the
//! model's dual variable (e.g. `"q"` for heat conduction). This is the proper
//! way to turn a *distributed* edge or body flux into a right-hand-side
//! contribution, instead of splitting it onto the nodes by hand. The density is
//! either a uniform constant or the single component of a per-element field
//! (see [`FluxDensity`]).
//!
//! The element measure `|J|` comes from the FE subspace, so a **boundary** mesh
//! works directly: a `SEG2` edge embedded in a 2-D `Coords` integrates as
//! a line (manifold Jacobian), a surface mesh as an area.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::node_field::SubNodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::kernel::MAX_CELL_DOFS;

/// Per-Gauss flux density consumed by [`flux`].
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
/// # use pyrucast::ops::node_field::flux::FluxDensity;
/// // Une densité **uniforme**, ou lue point par point dans un champ par
/// // éléments — le même opérateur sert les deux.
/// let f = node_field::flux(&zone, FluxDensity::Uniform(3.0), "q")?;
/// // ∫ Nᵢ dΩ somme à l'aire × densité : ici 2 × 3.
/// let total: f64 = (0..3).map(|i| f.value(n[i].id(), "q").unwrap()).sum();
/// assert!((total - 6.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub enum FluxDensity<'a> {
    /// Spatially uniform density (same value at every Gauss point).
    Uniform(f64),
    /// Read from the **single** component of a per-element field at each
    /// `(cell, Gauss)` point. The field must live on the same subspace as the
    /// one passed to [`flux`].
    Field(&'a Handle<SubElementField>),
}

/// Consistent nodal loads of a distributed flux over `fespace` (see the module
/// docs). Returns a [`SubNodeField`] on the subspace's unique nodes carrying
/// the single component `component` (the dual variable, e.g. `"q"`).
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
/// # use pyrucast::ops::node_field::flux::FluxDensity;
/// // Le chargement nodal **cohérent** d'un flux réparti : il répartit par
/// // les fonctions de forme, non à parts égales.
/// let f = node_field::flux(&zone, FluxDensity::Uniform(3.0), "q")?;
/// assert_eq!(f.components(), &["q".to_string()]);
/// let total: f64 = (0..3).map(|i| f.value(n[i].id(), "q").unwrap()).sum();
/// assert!((total - 6.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn flux(
    fespace: &Handle<SubFiniteElementSpace>,
    density: FluxDensity,
    component: &str,
) -> Result<SubNodeField> {
    let submesh = fespace.read().submesh();

    // Flux density: a bare constant, or the field's single component snapshotted
    // once (its guard cannot cross the parallel region). The component count is
    // checked here, for the zone; the copy below then reads **by index**, where
    // it used to search the name again at every (cell, point).
    let (uniform, densities): (f64, Option<Vec<f64>>) = match density {
        FluxDensity::Uniform(v) => (v, None),
        FluxDensity::Field(field) => {
            let f = field.read();
            let comps = f.components();
            if comps.len() != 1 {
                return Err(PyrucastError::Message(format!(
                    "flux: la densité par champ doit avoir exactement une composante (en a {})",
                    comps.len()
                )));
            }
            // Une seule composante : le tampon du champ **est** la densité,
            // ligne pour ligne. Plus de `Vec` de `Vec`, plus de recherche.
            (0.0, Some(f.values().to_vec()))
        }
    };

    // `f_i = Σ_g φ · N_i · |J| w`, scattered to the nodes: the density is captured
    // by the element closure (a plain snapshot, borrowed in place) and the shared
    // driver owns the colour-parallel scatter — this is a mass-like `N`-weighted
    // instance of the same nodal integrate-and-scatter as the `Bᵀ` operators.
    let support = submesh.read().to_poi1()?;
    kernel::scatter_to_nodes(
        std::slice::from_ref(fespace),
        &support,
        vec![component.to_string()],
        |geoms, fe| {
            let geom = &geoms[0];
            let cell = geom.cell;
            for g in 0..geom.n_gauss {
                let mut n_buf = [0.0_f64; MAX_CELL_DOFS];
                let shape = geom.field_n_at_g(g, &mut n_buf)?;
                let phi = densities
                    .as_deref()
                    .map_or(uniform, |d| d[cell * geom.n_gauss + g]);
                let w = geom.det_j_w(g) * phi;
                for i in 0..geom.n_nodes {
                    fe[i] += shape[i] * w;
                }
            }
            Ok(())
        },
    )
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node, NodeId};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Lagrange-1 FE subspace over a fresh SEG2 line of `n` equal elements from
    /// `a` to `b` (built on the given coordinates).
    fn seg2_line(points: &[Vec<f64>]) -> Handle<SubFiniteElementSpace> {
        let coords = Handle::new(Coords::new(points[0].len() as u8).unwrap());
        let nodes: Vec<Node> = points
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        for w in nodes.windows(2) {
            mesh.add_cell(&[w[0].id(), w[1].id()]).unwrap();
        }
        FiniteElementSpace::lagrange1(&mesh)
            .unwrap()
            .get(0)
            .unwrap()
    }

    /// Uniform flux on a 1-D SEG2 line: interior nodes receive `φ·h`, the two
    /// ends `φ·h/2`, and the total is `φ·L`.
    #[test]
    fn uniform_flux_consistent_loads_on_seg2_line() {
        // Two elements of length h = 0.5 on [0, 1].
        let fes = seg2_line(&[vec![0.0], vec![0.5], vec![1.0]]);
        let nodes: Vec<NodeId> = fes.read().submesh().read().connectivity().to_vec();
        // connectivity = [n0, n1, n1, n2] → unique [n0, n1, n2].
        let (n0, n1, n2) = (nodes[0], nodes[1], nodes[3]);

        let phi = 3.0;
        let load = flux(&fes, FluxDensity::Uniform(phi), "q").unwrap();
        let tol = 1e-12;
        let h = 0.5;
        assert!((load.value(n0, "q").unwrap() - phi * h / 2.0).abs() < tol);
        assert!((load.value(n1, "q").unwrap() - phi * h).abs() < tol);
        assert!((load.value(n2, "q").unwrap() - phi * h / 2.0).abs() < tol);
        let total = load.value(n0, "q").unwrap()
            + load.value(n1, "q").unwrap()
            + load.value(n2, "q").unwrap();
        assert!((total - phi * 1.0).abs() < tol, "total {total} ≠ {}", phi);
    }

    /// A SEG2 edge embedded in a 2-D Coords is integrated with the
    /// **line** measure (manifold Jacobian): a unit edge of uniform flux `φ`
    /// gives nodal loads summing to `φ·length`.
    #[test]
    fn uniform_flux_on_2d_edge_uses_line_measure() {
        // Single vertical edge from (0,0) to (0,1), length 1.
        let fes = seg2_line(&[vec![0.0, 0.0], vec![0.0, 1.0]]);
        let nodes: Vec<NodeId> = fes.read().submesh().read().connectivity().to_vec();
        let (a, b) = (nodes[0], nodes[1]);

        let phi = 10.0;
        let load = flux(&fes, FluxDensity::Uniform(phi), "q").unwrap();
        let tol = 1e-12;
        // One element ⇒ each end gets φ·L/2.
        assert!((load.value(a, "q").unwrap() - phi * 0.5).abs() < tol);
        assert!((load.value(b, "q").unwrap() - phi * 0.5).abs() < tol);
        let total = load.value(a, "q").unwrap() + load.value(b, "q").unwrap();
        assert!((total - phi).abs() < tol);
    }

    /// A single-component per-element field gives the same loads as the uniform
    /// density it carries.
    #[test]
    fn field_density_matches_uniform() {
        let fes = seg2_line(&[vec![0.0], vec![0.5], vec![1.0]]);
        let phi = 7.5;
        let mut field = SubElementField::new(fes.clone(), vec!["phi".into()]).unwrap();
        field.set_uniform("phi", phi).unwrap();
        let field = Handle::new(field);

        let nodes: Vec<NodeId> = fes.read().submesh().read().connectivity().to_vec();
        let n1 = nodes[1];
        let from_field = flux(&fes, FluxDensity::Field(&field), "q").unwrap();
        let from_uniform = flux(&fes, FluxDensity::Uniform(phi), "q").unwrap();
        assert!(
            (from_field.value(n1, "q").unwrap() - from_uniform.value(n1, "q").unwrap()).abs()
                < 1e-12
        );
    }

    /// A multi-component density field is rejected.
    #[test]
    fn multi_component_field_rejected() {
        let fes = seg2_line(&[vec![0.0], vec![1.0]]);
        let field =
            Handle::new(SubElementField::new(fes.clone(), vec!["a".into(), "b".into()]).unwrap());
        assert!(flux(&fes, FluxDensity::Field(&field), "q").is_err());
    }
}
