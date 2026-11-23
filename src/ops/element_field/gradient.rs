//! Gradient of a nodal field at the Gauss points — purely geometric, no
//! model/physics involved (it depends only on the FE space and the field).
//!
//! `∇f = Σ_i f_i ∇N_i` evaluated cell-by-cell, Gauss-point by Gauss-point. A
//! component is a scalar unless the field carries it as one axis of a vector
//! (`u_x, u_y(, u_z)`): a vector's gradient is its tensor, row by row, with the
//! hoop term `u_r / r` on an axisymmetric space — the `∇u` of
//! [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)'s
//! gradient measure `F = I + ∇u`.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::Field;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::error::Result;
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::kernel::MAX_CELL_DOFS;

/// Axis suffixes for spatial directions (`x`, `y`, `z`).
pub(crate) const AXES: [&str; 3] = ["x", "y", "z"];

/// Gradient `∇f` of a node `field` at the Gauss points of every subspace of
/// `fespace`.
///
/// Geometric and physics-agnostic. The rank of each component is read off the
/// names:
///
/// - a **vector** `p`, when `field` carries `p_x, p_y` (and `p_z` in 3-D), in
///   any order, gives its gradient tensor `grad_p_ij = ∂p_i/∂x_j`, row by row
///   (`grad_p_xx, grad_p_xy, grad_p_yx, grad_p_yy` in 2-D); on an
///   axisymmetric space, where `x` is the radius, it also gives the
///   circumferential (hoop) term `grad_p_zz = p_x / r`, listed after every
///   derivative;
/// - any other component `c` is a **scalar**, with one component
///   `grad_c_<axis>` per axis (`grad_T_x, grad_T_y`).
///
/// A scalar gradient is what a physics whose behaviour consumes a gradient
/// (heat conduction) reads, through
/// [`crate::ops::element_field::behavior::integrate`]. A vector gradient is a
/// tensor that [`tensor_dot`](fn@crate::ops::element_field::tensor_dot),
/// [`tensor_ddot`](fn@crate::ops::element_field::tensor_ddot) and
/// [`divergence`](fn@crate::ops::node_field::divergence) read under the
/// prefix `grad_p`.
///
/// `∇f = Σ_i f_i ∇N_i` evaluated cell-by-cell, Gauss point by Gauss point, via
/// the shared parallel driver `models::kernel::nodal_pointwise`.
///
/// Errors: the hoop term interpolates the field, which a `MODEL_EMBEDDED`
/// subspace does not allow
/// ([`require_field_basis`](crate::models::kernel::require_field_basis)).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let sol = champ(vec!["T".into(), "u_x".into(), "u_y".into()]);
/// # {
/// #     let mut z = sol.get(0).unwrap().write();
/// #     // T = 10 + 20x + 40y; u_x = 2x + 0.5y, u_y = 0.1x + 3y.
/// #     for (k, [x, y]) in [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]].into_iter().enumerate() {
/// #         z.set_value(n[k].id(), "T", 10.0 + 20.0 * x + 40.0 * y).unwrap();
/// #         z.set_value(n[k].id(), "u_x", 2.0 * x + 0.5 * y).unwrap();
/// #         z.set_value(n[k].id(), "u_y", 0.1 * x + 3.0 * y).unwrap();
/// #     }
/// # }
/// // A temperature T = 10 + 20x + 40y and a displacement u_x = 2x + 0.5y,
/// // u_y = 0.1x + 3y, in one field.
/// let g = element_field::gradient(&sol, &fes)?;
/// let z = g.get(0)?.read();
/// // T is a scalar: one component per axis.
/// assert!((z.value(0, 0, "grad_T_x")? - 20.0).abs() < 1e-9);
/// assert!((z.value(0, 0, "grad_T_y")? - 40.0).abs() < 1e-9);
/// // u is a vector: its gradient tensor, row = component, column = derivative.
/// assert_eq!(
///     z.components(),
///     ["grad_T_x", "grad_T_y", "grad_u_xx", "grad_u_xy", "grad_u_yx", "grad_u_yy"]
/// );
/// assert!((z.value(0, 0, "grad_u_xy")? - 0.5).abs() < 1e-9); // ∂u_x/∂y
/// assert!((z.value(0, 0, "grad_u_yx")? - 0.1).abs() < 1e-9); // ∂u_y/∂x
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn gradient(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(field);
    let view = field.view();
    let mut out = ElementField::empty();
    for sub in fespace {
        out.add_sub(Handle::new(gradient_zone(sub, &view, &components)?))?;
    }
    Ok(out)
}

/// [`gradient`] of the named `field_components` on one zone — the per-zone body
/// shared with [`strain`](fn@crate::ops::element_field::strain).
pub(crate) fn gradient_zone(
    subfespace: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    field_components: &[String],
) -> Result<SubElementField> {
    let (space_dim, axisymmetric) = {
        let s = subfespace.read();
        (s.space_dim(), s.is_axisymmetric())
    };
    let axes = &AXES[..space_dim];
    // The vector a component is one axis of: `p` for `p_<a>`, when the field
    // carries `p` along every axis.
    let vector_of = |c: &str| -> Option<String> {
        axes.iter()
            .find_map(|a| c.strip_suffix(&format!("_{a}")))
            .filter(|p| {
                axes.iter()
                    .all(|a| field_components.contains(&format!("{p}_{a}")))
            })
            .map(str::to_string)
    };
    // Settled once for the zone: the components the kernel reads, in its order
    // (a vector along the axes), the names it writes, and, for each vector on
    // an axisymmetric space, where its radial component sits.
    let mut components: Vec<String> = Vec::with_capacity(field_components.len());
    let mut names = Vec::with_capacity(field_components.len() * space_dim + 1);
    let mut radial_positions = Vec::new();
    let mut radial_names = Vec::new();
    for c in field_components {
        if components.contains(c) {
            continue; // an axis of a vector already laid out
        }
        match vector_of(c) {
            Some(p) => {
                if axisymmetric {
                    radial_positions.push(components.len());
                    radial_names.push(format!("grad_{p}_zz"));
                }
                for ai in axes {
                    components.push(format!("{p}_{ai}"));
                    for aj in axes {
                        names.push(format!("grad_{p}_{ai}{aj}"));
                    }
                }
            }
            None => {
                components.push(c.clone());
                for a in axes {
                    names.push(format!("grad_{c}_{a}"));
                }
            }
        }
    }
    names.extend(radial_names);
    if !radial_positions.is_empty() {
        kernel::require_field_basis(subfespace, "shape values")?;
    }
    // Point kernel: ∂(component r)/∂x_a = Σ_i f_i · ∂N_i/∂x_a at r·d + a —
    // within a vector, its entry (i, j) row by row — then the hoop terms.
    let nc = components.len();
    kernel::nodal_pointwise(
        subfespace,
        view,
        &components,
        names,
        |geom, g, dofs, out| {
            let sd = geom.space_dim;
            let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
            let dn_dx = &mut dn_buf[..geom.n_nodes * sd];
            geom.dn_dx(g, dn_dx)?;
            let mut oc = 0;
            for c in 0..nc {
                for a in 0..sd {
                    let mut grad = 0.0;
                    for i in 0..geom.n_nodes {
                        grad += dofs[i * nc + c] * dn_dx[i * sd + a];
                    }
                    out[oc] = grad;
                    oc += 1;
                }
            }
            // The hoop term u_r / r of each vector: the list is empty off an
            // axisymmetric space, and nothing below runs.
            for (h, &r) in radial_positions.iter().enumerate() {
                let mut n_buf = [0.0_f64; MAX_CELL_DOFS];
                let n = geom.field_n_at_g(g, &mut n_buf);
                let mut u_r = 0.0;
                for k in 0..geom.n_nodes {
                    u_r += dofs[k * nc + r] * n[k];
                }
                out[nc * sd + h] = u_r / geom.radius(g);
            }
            Ok(())
        },
    )
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// 1-D linear field `T(x) = x` on a single SEG2 ⇒ `∇T = 1`.
    #[test]
    fn gradient_of_linear_field_on_seg2() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut t = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        t.set_value(a.id(), "T", 0.0).unwrap();
        t.set_value(b.id(), "T", 2.0).unwrap(); // T = x
        let t = NodeField::from_sub(t);

        let grad = gradient(&t, &fes).unwrap();
        assert_eq!(grad.len(), 1);
        {
            let s = grad.get(0).unwrap().read();
            assert_eq!(s.components(), &["grad_T_x".to_string()]);
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_T_x").unwrap() - 1.0).abs() < 1e-12);
            }
        }
    }

    /// 2-D linear field `f = 2x + 3y` on a TRI3 ⇒ `∇f = (2, 3)`.
    #[test]
    fn gradient_2d_linear_field_on_tri3() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        f.set_value(a.id(), "f", 0.0).unwrap(); // 2·0 + 3·0
        f.set_value(b.id(), "f", 2.0).unwrap(); // 2·1 + 3·0
        f.set_value(c.id(), "f", 3.0).unwrap(); // 2·0 + 3·1
        let f = NodeField::from_sub(f);

        let grad = gradient(&f, &fes).unwrap();
        {
            let s = grad.get(0).unwrap().read();
            assert_eq!(
                s.components(),
                &["grad_f_x".to_string(), "grad_f_y".to_string()]
            );
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_f_x").unwrap() - 2.0).abs() < 1e-12);
                assert!((s.value(0, g, "grad_f_y").unwrap() - 3.0).abs() < 1e-12);
            }
        }
    }

    /// A nodal field on the nodes of `mesh`, its components set node by node
    /// from their position.
    fn nodal(nodes: &[Node], names: &[&str], value: impl Fn(&str, [f64; 2]) -> f64) -> NodeField {
        let support = Handle::new(SubMesh::poi1_from_nodes(nodes).unwrap());
        let mut f =
            SubNodeField::from_poi1(&support, names.iter().map(|s| s.to_string()).collect())
                .unwrap();
        for n in nodes {
            let p = n.position().unwrap();
            for name in names {
                f.set_value(n.id(), name, value(name, [p[0], p[1]]))
                    .unwrap();
            }
        }
        NodeField::from_sub(f)
    }

    /// The rank is read off the names, not their order: `u_y … u_x` is the
    /// vector `u`, laid out along the axes; `T` and a lone `q_x` are scalars.
    #[test]
    fn the_rank_is_read_off_the_names() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        // u_x = 2x + 0.5y, u_y = 0.1x + 3y, T = x, q_x = y.
        let field = nodal(&n, &["u_y", "T", "u_x", "q_x"], |name, [x, y]| match name {
            "u_x" => 2.0 * x + 0.5 * y,
            "u_y" => 0.1 * x + 3.0 * y,
            "T" => x,
            _ => y,
        });

        let grad = gradient(&field, &fes).unwrap();
        let s = grad.get(0).unwrap().read();
        assert_eq!(
            s.components(),
            [
                "grad_u_xx",
                "grad_u_xy",
                "grad_u_yx",
                "grad_u_yy",
                "grad_T_x",
                "grad_T_y",
                "grad_q_x_x",
                "grad_q_x_y"
            ]
        );
        let expected = [
            ("grad_u_xx", 2.0),
            ("grad_u_xy", 0.5),
            ("grad_u_yx", 0.1),
            ("grad_u_yy", 3.0),
            ("grad_T_x", 1.0),
            ("grad_T_y", 0.0),
            ("grad_q_x_x", 0.0),
            ("grad_q_x_y", 1.0),
        ];
        for (name, value) in expected {
            assert!(
                (s.value(0, 0, name).unwrap() - value).abs() < 1e-12,
                "{name}"
            );
        }
    }

    /// On an axisymmetric space, the gradient of a vector is `F − I` of the
    /// deformation gradient, its hoop term `u_r / r` included, at every Gauss
    /// point; a scalar beside it gets no hoop term.
    #[test]
    fn the_gradient_matches_deformation_on_an_axisymmetric_space() {
        use crate::ops::element_field::{deformation, StrainMeasure};
        let coords = Handle::new(Coords::axisymmetric());
        let n: Vec<Node> = [[1.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.9, 1.3]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        let ids: Vec<_> = n.iter().map(Node::id).collect();
        mesh.add_cell(&ids).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        // A displacement that is not linear, and a temperature.
        let value = |name: &str, [r, z]: [f64; 2]| match name {
            "u_x" => 0.01 * r * r + 0.02 * z,
            "u_y" => -0.03 * r * z,
            _ => 20.0 + r,
        };
        let field = nodal(&n, &["u_x", "u_y", "T"], value);
        let u = nodal(&n, &["u_x", "u_y"], value);

        let grad = gradient(&field, &fes).unwrap();
        let f = deformation(&u, &fes, StrainMeasure::Gradient).unwrap();
        let (g, f) = (grad.get(0).unwrap().read(), f.get(0).unwrap().read());
        assert!(g.components().contains(&"grad_u_zz".to_string()));
        assert!(!g.components().contains(&"grad_T_zz".to_string()));
        for p in 0..g.gauss_count() {
            for (ij, delta) in [
                ("xx", 1.0),
                ("xy", 0.0),
                ("yx", 0.0),
                ("yy", 1.0),
                ("zz", 1.0),
            ] {
                let got = g.value(0, p, &format!("grad_u_{ij}")).unwrap();
                let want = f.value(0, p, &format!("F_{ij}")).unwrap() - delta;
                assert!((got - want).abs() < 1e-12, "grad_u_{ij} at point {p}");
            }
        }
    }
}
