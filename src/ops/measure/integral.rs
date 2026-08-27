//! Integral of a field over its support, using the finite-element quadrature —
//! `∫_Ω f dΩ`, one component, summed over the whole support.
//!
//! Two entry points, for the two field flavours (the answer to
//! [`crate::ops::element_field::behavior::integrate`]'s dual: it turns a distributed field back
//! into a single total, e.g. the **resultant** of a distributed force density):
//!
//! - [`integral`] on a **nodal** field interpolates with the shape functions,
//!   `∫ Σ_i f_i N_i dΩ` — the values live at the nodes and are lifted to the
//!   Gauss points by `N_i`;
//! - [`integral_element`] on a **per-element** field integrates the Gauss-point
//!   values directly, `∫ f dΩ = Σ_cell Σ_g f(cell,g) |J|_g w_g` — no `N_i`, the
//!   values already sit at the quadrature points.
//!
//! Both delegate the parallel per-cell reduction to
//! [`crate::models::kernel::reduce_cells`]. A **nodal resultant** (summing
//! already-integrated nodal forces, `Σ_nodes f`) is a plain value reduction
//! instead — see [`crate::containers::field::Field::sum`].

use crate::atoms::NodeId;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::models::kernel::MAX_CELL_DOFS;
use std::collections::HashMap;

/// `∫_Ω f dΩ` of a **nodal** `field`, interpolated with the FE shape functions
/// of `fespace`: `Σ_cell Σ_g (Σ_i f_i N_i(ξ_g)) |J|_g w_g`, for the single
/// `component`. Summed over every subspace of `fespace`.
///
/// `field` must define `component` at every node of `fespace` (else an error).
/// Returns the total; see the module docs for the per-element counterpart.
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
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // ∫ T dΩ sur le triangle (0,0), (2,0), (0,2), d'aire 2. La moyenne
/// // des trois valeurs nodales étant 50, l'intégrale vaut 100.
/// assert!((measure::integral(&temp, &fes, "T")? - 100.0).abs() < 1e-9);
/// // Une composante absente est une erreur.
/// assert!(measure::integral(&temp, &fes, "q").is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn integral(field: &NodeField, fespace: &FiniteElementSpace, component: &str) -> Result<f64> {
    let view = field.view()?;
    let mut total = 0.0;
    for sub_h in fespace {
        let submesh = sub_h.read().submesh();
        let conn: Vec<NodeId> = submesh.read().connectivity().to_vec();

        // Gather this subspace's nodal values for `component` once (serial), so
        // the parallel per-cell kernel does O(1) look-ups and never re-reads the
        // store.
        let mut vals: HashMap<NodeId, f64> = HashMap::new();
        for &nid in &conn {
            if let std::collections::hash_map::Entry::Vacant(e) = vals.entry(nid) {
                e.insert(view.value(nid, component)?);
            }
        }

        total += kernel::reduce_cells(sub_h, |geom| {
            let ids = geom.node_ids();
            let mut acc = 0.0;
            for g in 0..geom.n_gauss {
                let mut n_buf = [0.0_f64; MAX_CELL_DOFS];
                let n = geom.field_n_at_g(g, &mut n_buf)?; // field shape values N_i(ξ_g)
                let mut fg = 0.0;
                for i in 0..geom.n_nodes {
                    fg += vals[&ids[i]] * n[i];
                }
                acc += fg * geom.det_j_w(g);
            }
            Ok(acc)
        })?;
    }
    Ok(total)
}

/// `∫_Ω f dΩ` of a **per-element** `field` (values at the Gauss points), by
/// direct quadrature: `Σ_cell Σ_g f(cell,g) |J|_g w_g`, for the single
/// `component`. Summed over every subspace defining `component`.
///
/// Errors if no subspace defines `component`. Returns the total; see the module
/// docs for the nodal (shape-function) counterpart.
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
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // Le pendant pour un champ **par éléments** : il n'a besoin d'aucun
/// // espace EF, le champ portant déjà son support.
/// # let mut f = ElementField::new(&fes, vec!["q".into()])?;
/// # f.get(0)?.write().set_uniform("q", 3.0)?;
/// assert!((measure::integral_element(&f, "q")? - 3.0 * 2.0).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn integral_element(field: &ElementField, component: &str) -> Result<f64> {
    let mut total = 0.0;
    let mut found = false;
    for sub_h in field {
        let s = sub_h.read();
        if s.component_index(component).is_none() {
            continue; // this zone does not carry the component — skip it
        }
        found = true;
        let fespace = s.support();
        total += kernel::reduce_cells(&fespace, |geom| {
            let mut acc = 0.0;
            for g in 0..geom.n_gauss {
                acc += s.value(geom.cell, g, component)? * geom.det_j_w(g);
            }
            Ok(acc)
        })?;
    }
    if !found {
        return Err(PyrucastError::Message(format!(
            "integral: no sub-field defines component {}",
            component
        )));
    }
    Ok(total)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::element_field::SubElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Nodal integral on SEG2 `[0, 1]` of the linear field `f(x) = x`
    /// (`f_0 = 0`, `f_1 = 1`): `∫₀¹ x dx = 1/2`. Exercises the `N_i` lift.
    #[test]
    fn nodal_integral_of_linear_field_is_half() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        f.set_value(a.id(), "f", 0.0).unwrap();
        f.set_value(b.id(), "f", 1.0).unwrap();
        let f = NodeField::from_sub(f);

        let got = integral(&f, &fes, "f").unwrap();
        assert!((got - 0.5).abs() < 1e-12, "got {got}");
    }

    /// Nodal integral of a **constant** field equals `constant × measure`: on a
    /// unit right TRI3 (area 1/2), `f ≡ 4` ⇒ `∫ = 2`.
    #[test]
    fn nodal_integral_of_constant_is_value_times_area() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        for n in [&a, &b, &c] {
            f.set_value(n.id(), "f", 4.0).unwrap();
        }
        let f = NodeField::from_sub(f);

        let got = integral(&f, &fes, "f").unwrap();
        assert!((got - 2.0).abs() < 1e-12, "got {got}");
    }

    /// Per-element integral by direct quadrature: a constant element field
    /// `c ≡ 3` on the unit right TRI3 (area 1/2) integrates to `3 × 1/2 = 3/2`.
    #[test]
    fn element_integral_of_constant_is_value_times_area() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut ef = SubElementField::new(fes.get(0).unwrap(), vec!["c".into()]).unwrap();
        ef.set_uniform("c", 3.0).unwrap();
        let mut field = ElementField::empty();
        field.add_sub(Handle::new(ef)).unwrap();

        let got = integral_element(&field, "c").unwrap();
        assert!((got - 1.5).abs() < 1e-12, "got {got}");
    }

    /// The two integrals agree on a constant field: interpolating a constant
    /// nodal field (all `N_i` sum to 1) matches integrating the same constant at
    /// the Gauss points.
    #[test]
    fn nodal_and_element_agree_on_constant() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 3.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut nf = SubNodeField::from_poi1(&support, vec!["v".into()]).unwrap();
        for n in [&a, &b, &c] {
            nf.set_value(n.id(), "v", 1.5).unwrap();
        }
        let nf = NodeField::from_sub(nf);

        let mut ef = SubElementField::new(fes.get(0).unwrap(), vec!["v".into()]).unwrap();
        ef.set_uniform("v", 1.5).unwrap();
        let mut ef_agg = ElementField::empty();
        ef_agg.add_sub(Handle::new(ef)).unwrap();

        let via_nodes = integral(&nf, &fes, "v").unwrap();
        let via_gauss = integral_element(&ef_agg, "v").unwrap();
        assert!(
            (via_nodes - via_gauss).abs() < 1e-12,
            "{via_nodes} ≠ {via_gauss}"
        );
    }

    /// An unknown component on the element integral is rejected.
    #[test]
    fn element_integral_unknown_component_errors() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let ef = SubElementField::new(fes.get(0).unwrap(), vec!["c".into()]).unwrap();
        let mut field = ElementField::empty();
        field.add_sub(Handle::new(ef)).unwrap();
        assert!(integral_element(&field, "nope").is_err());
    }
}
