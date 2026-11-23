//! Contractions of two tensors at every Gauss point: the single contraction
//! `A·B` ([`tensor_dot`](fn@tensor_dot)) and the double contraction `A:B`
//! ([`tensor_ddot`](fn@tensor_ddot)).
//!
//! Both read their tensors off the component names, the way
//! [`divergence`](fn@crate::ops::node_field::divergence) does: `A_xx, A_xy,
//! A_yy` is a symmetric tensor, read whole when the field also carries `A_yx`.
//! In 2-D the out-of-plane term `A_zz` is required on an axisymmetric space
//! (the hoop term) and read on a plane one when the field carries it.
//!
//! The two fields are paired by FE subspace: the result has one zone per
//! subspace both carry, in the order of `a`, as
//! [`psca`](fn@crate::ops::field::psca) does.

use nalgebra::Matrix3;

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::continuum::internal_force::{full_matrix_reads, voigt_matrix_reads};
use crate::models::kernel;

/// One tensor of a product, on one FE subspace: the zone that carries it, and
/// the entry of a 3×3 matrix each position of its layout fills.
struct Operand {
    zone: Handle<SubElementField>,
    /// `(i, j, c)`: the entry `(i, j)` of the matrix takes the value at
    /// position `c` of a Gauss point's row.
    entries: Vec<(usize, usize, u32)>,
    /// Whether the tensor carries its out-of-plane term `zz` (2-D only).
    with_zz: bool,
}

impl Operand {
    /// The tensor `prefix` read in the zone of `field` on `fespace`.
    ///
    /// Settled once per subspace: which zone, which names, where each one
    /// sits in the row, which entry of the matrix it fills.
    fn new(
        field: &ElementField,
        fespace: &Handle<SubFiniteElementSpace>,
        prefix: &str,
    ) -> Result<Operand> {
        let (space_dim, axisymmetric) = {
            let s = fespace.read();
            (s.space_dim(), s.is_axisymmetric())
        };
        let zone = field.sub_for_fespace_reading(fespace, &[format!("{prefix}_xx")])?;
        let components = zone.read().components().to_vec();
        let zz = format!("{prefix}_zz");
        let with_zz = space_dim == 2 && (axisymmetric || components.contains(&zz));
        let names = |mut names: Vec<String>| {
            if with_zz {
                names.push(zz.clone());
            }
            names
        };
        let positions = |names: &[String]| -> Option<Vec<u32>> {
            names
                .iter()
                .map(|n| components.iter().position(|c| c == n).map(|i| i as u32))
                .collect()
        };
        // Read whole when every entry is there — in 1-D both lists are the
        // single `A_xx`, read as symmetric.
        let whole = if space_dim > 1 {
            positions(&names(full_matrix_reads(prefix, space_dim)))
        } else {
            None
        };
        let symmetric = names(voigt_matrix_reads(prefix, space_dim));
        let (layout, is_whole) = match whole {
            Some(layout) => (layout, true),
            None => (
                positions(&symmetric).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "the zone on the FE space {fespace} carries no tensor `{prefix}`: \
                         expected {symmetric:?}, it carries {components:?}"
                    ))
                })?,
                false,
            ),
        };

        // The (i, j) of each position, in the order of the names: every entry
        // of a whole tensor, the upper triangle of a symmetric one, whose
        // off-diagonal terms fill both of their entries.
        let mut pairs: Vec<(usize, usize)> = (0..space_dim)
            .flat_map(|i| (if is_whole { 0 } else { i }..space_dim).map(move |j| (i, j)))
            .collect();
        if with_zz {
            pairs.push((2, 2));
        }
        let mut entries = Vec::with_capacity(9);
        for (&(i, j), &c) in pairs.iter().zip(&layout) {
            entries.push((i, j, c));
            if !is_whole && i != j {
                entries.push((j, i, c));
            }
        }
        Ok(Operand {
            zone,
            entries,
            with_zz,
        })
    }

    /// The tensor of one Gauss point, from its row: a matrix on the stack,
    /// zero where the tensor has no entry.
    fn matrix(&self, row: &[f64]) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for &(i, j, c) in &self.entries {
            m[(i, j)] = row[c as usize];
        }
        m
    }
}

/// The FE subspaces both fields carry, in the order of `a`, each with its two
/// operands. A subspace only one of them carries has no counterpart and is
/// left out.
fn operand_pairs(
    a: &ElementField,
    b: &ElementField,
    a_prefix: &str,
    b_prefix: &str,
) -> Result<Vec<(Handle<SubFiniteElementSpace>, Operand, Operand)>> {
    let mut pairs: Vec<(Handle<SubFiniteElementSpace>, Operand, Operand)> = Vec::new();
    for zone in a {
        let fespace = zone.read().support();
        let paired = pairs.iter().any(|(s, _, _)| s.same_object(&fespace));
        if paired || b.subs_for_fespace(&fespace).is_empty() {
            continue;
        }
        let a_operand = Operand::new(a, &fespace, a_prefix)?;
        let b_operand = Operand::new(b, &fespace, b_prefix)?;
        pairs.push((fespace, a_operand, b_operand));
    }
    Ok(pairs)
}

/// Single contraction `A·B` of two tensors, at every Gauss point.
///
/// `(A·B)_ij = Σ_k A_ik B_kj`, with `A` the tensor `a_prefix` of `a` and `B`
/// the tensor `b_prefix` of `b`. Each tensor is read off its names: `A_xx, A_xy, A_yy` is symmetric, read
/// whole when `A_yx` is there too. In 2-D, `A_zz` is required on an
/// axisymmetric space and read on a plane one when the field carries it; an
/// absent `A_zz` counts as 0. The fields are paired by FE subspace: one zone
/// per subspace both carry, in the order of `a`.
///
/// The result is the whole tensor `prefix`, row by row (`p_xx, p_xy, p_yx,
/// p_yy` in 2-D, nine components in 3-D), followed by `p_zz` when both tensors
/// carry their `zz` term. The order of the operands counts.
///
/// Errors: on a subspace both fields carry, a zone that does not carry its
/// tensor, or several zones that could.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// use pyrucast::ops::element_field::tensor_dot;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// // One field, two tensors: A = [[1, 2], [2, 3]], stored symmetric, and
/// // B = [[4, 5], [6, 7]], stored whole since it carries B_yx.
/// let names = ["A_xx", "A_xy", "A_yy", "B_xx", "B_xy", "B_yx", "B_yy"];
/// let t = ElementField::new(&fes, names.iter().map(|s| s.to_string()).collect())?;
/// for (name, value) in names.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) {
///     t.get(0)?.write().set_uniform(name, value)?;
/// }
/// // A·B = [[16, 19], [26, 31]]: a whole tensor, not symmetric.
/// let p = tensor_dot(&t, &t, "A", "B", "p")?;
/// let z = p.get(0)?.read();
/// assert_eq!(z.components(), ["p_xx", "p_xy", "p_yx", "p_yy"]);
/// assert_eq!(z.value(0, 0, "p_xy")?, 19.0);
/// assert_eq!(z.value(0, 0, "p_yx")?, 26.0);
/// // The order counts: B·A = [[14, 23], [20, 33]].
/// let q = tensor_dot(&t, &t, "B", "A", "q")?;
/// assert_eq!(q.get(0)?.read().value(0, 0, "q_xy")?, 23.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn tensor_dot(
    a: &ElementField,
    b: &ElementField,
    a_prefix: &str,
    b_prefix: &str,
    prefix: &str,
) -> Result<ElementField> {
    let mut result = ElementField::empty();
    for (fespace, a_operand, b_operand) in operand_pairs(a, b, a_prefix, b_prefix)? {
        let space_dim = fespace.read().space_dim();
        let mut names = full_matrix_reads(prefix, space_dim);
        let mut outputs: Vec<(usize, usize)> = (0..space_dim)
            .flat_map(|i| (0..space_dim).map(move |j| (i, j)))
            .collect();
        if a_operand.with_zz && b_operand.with_zz {
            names.push(format!("{prefix}_zz"));
            outputs.push((2, 2));
        }
        let zone = kernel::element_pointwise(
            &fespace,
            &a_operand.zone,
            &b_operand.zone,
            &b_operand.zone,
            names,
            |_, _, a_row, b_row, _, out| {
                let product = a_operand.matrix(a_row) * b_operand.matrix(b_row);
                for (slot, &(i, j)) in out.iter_mut().zip(&outputs) {
                    *slot = product[(i, j)];
                }
                Ok(())
            },
        )?;
        result.add_sub(Handle::new(zone))?;
    }
    Ok(result)
}

/// Double contraction `A:B` of two tensors, at every Gauss point.
///
/// `A:B = Σ_ij A_ij B_ij`, with `A` the tensor `a_prefix` of `a` and `B` the
/// tensor `b_prefix` of `b`. The tensors are read and the fields paired as in
/// [`tensor_dot`](fn@tensor_dot). An off-diagonal term stored once in a
/// symmetric tensor stands for two entries, so it counts twice. The `zz`
/// terms count when both tensors carry them. The trace of `A` is `A:I`, with
/// `I` a uniform field carrying `I_xx = I_yy = I_zz = 1` and `I_xy = 0`.
///
/// The result carries the single component `"tensor_ddot"`. `A:B = B:A`: the
/// operands may come in either order.
///
/// Errors: as [`tensor_dot`](fn@tensor_dot).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// use pyrucast::ops::element_field::tensor_ddot;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let uniform = |pairs: &[(&str, f64)]| -> pyrucast::Result<ElementField> {
/// #     let f = ElementField::new(&fes, pairs.iter().map(|(c, _)| c.to_string()).collect())?;
/// #     for (c, v) in pairs {
/// #         f.get(0)?.write().set_uniform(c, *v)?;
/// #     }
/// #     Ok(f)
/// # };
/// // A = [[1, 2], [2, 3]], symmetric; B = [[4, 5], [6, 7]], whole.
/// let t = uniform(&[("A_xx", 1.0), ("A_xy", 2.0), ("A_yy", 3.0),
///                   ("B_xx", 4.0), ("B_xy", 5.0), ("B_yx", 6.0), ("B_yy", 7.0)])?;
/// let value = |f: &ElementField| f.get(0).unwrap().read().value(0, 0, "tensor_ddot").unwrap();
/// // A:B = 1·4 + 2·5 + 2·6 + 3·7.
/// assert_eq!(value(&tensor_ddot(&t, &t, "A", "B")?), 47.0);
/// // A:A = 1 + 2·(2·2) + 9: A_xy, stored once, counts twice.
/// assert_eq!(value(&tensor_ddot(&t, &t, "A", "A")?), 18.0);
/// // The trace of B is B:I.
/// let identity = uniform(&[("I_xx", 1.0), ("I_xy", 0.0), ("I_yy", 1.0), ("I_zz", 1.0)])?;
/// assert_eq!(value(&tensor_ddot(&t, &identity, "B", "I")?), 11.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn tensor_ddot(
    a: &ElementField,
    b: &ElementField,
    a_prefix: &str,
    b_prefix: &str,
) -> Result<ElementField> {
    let mut result = ElementField::empty();
    for (fespace, a_operand, b_operand) in operand_pairs(a, b, a_prefix, b_prefix)? {
        let zone = kernel::element_pointwise(
            &fespace,
            &a_operand.zone,
            &b_operand.zone,
            &b_operand.zone,
            vec!["tensor_ddot".into()],
            |_, _, a_row, b_row, _, out| {
                out[0] = a_operand.matrix(a_row).dot(&b_operand.matrix(b_row));
                Ok(())
            },
        )?;
        result.add_sub(Handle::new(zone))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;

    /// One TRI3 on `coords`, away from the axis, and the FE space on it.
    fn triangle(coords: &Handle<Coords>) -> FiniteElementSpace {
        let n: Vec<Node> = [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap()
    }

    /// A field on `fes` holding the uniform `(component, value)` pairs.
    fn uniform(fes: &FiniteElementSpace, pairs: &[(&str, f64)]) -> ElementField {
        let names = pairs.iter().map(|(c, _)| c.to_string()).collect();
        let field = ElementField::new(fes, names).unwrap();
        for (c, v) in pairs {
            field.get(0).unwrap().write().set_uniform(c, *v).unwrap();
        }
        field
    }

    /// The value of `component` at the first Gauss point of the first zone.
    fn read(field: &ElementField, component: &str) -> f64 {
        field.get(0).unwrap().read().value(0, 0, component).unwrap()
    }

    const A: [(&str, f64); 3] = [("A_xx", 1.0), ("A_xy", 2.0), ("A_yy", 3.0)];
    const B: [(&str, f64); 4] = [("B_xx", 4.0), ("B_xy", 5.0), ("B_yx", 6.0), ("B_yy", 7.0)];

    /// On an axisymmetric space the hoop terms are part of both tensors: they
    /// multiply into `p_zz`, last, and add to `A:B`. A tensor without one is
    /// refused.
    #[test]
    fn the_zz_terms_are_required_on_an_axisymmetric_space() {
        let fes = triangle(&Handle::new(Coords::axisymmetric()));
        let a = uniform(&fes, &[A.as_slice(), &[("A_zz", 0.5)]].concat());
        let b = uniform(&fes, &[B.as_slice(), &[("B_zz", 4.0)]].concat());

        let p = tensor_dot(&a, &b, "A", "B", "p").unwrap();
        assert_eq!(
            p.get(0).unwrap().read().components(),
            ["p_xx", "p_xy", "p_yx", "p_yy", "p_zz"]
        );
        assert_eq!(read(&p, "p_zz"), 2.0);
        assert_eq!(read(&p, "p_xy"), 19.0);
        assert_eq!(
            read(&tensor_ddot(&a, &b, "A", "B").unwrap(), "tensor_ddot"),
            49.0
        );

        let no_hoop = uniform(&fes, &A);
        assert!(tensor_dot(&no_hoop, &b, "A", "B", "p").is_err());
    }

    /// On a plane space the `zz` term counts when both tensors carry it; when
    /// only one does, it multiplies a 0 and gives no `p_zz`.
    #[test]
    fn a_plane_zz_term_counts_when_both_fields_carry_it() {
        let fes = triangle(&Handle::new(Coords::new(2).unwrap()));
        let a = uniform(&fes, &[A.as_slice(), &[("A_zz", 0.5)]].concat());
        let b_zz = uniform(&fes, &[B.as_slice(), &[("B_zz", 4.0)]].concat());
        let b = uniform(&fes, &B);

        let both = tensor_dot(&a, &b_zz, "A", "B", "p").unwrap();
        assert_eq!(read(&both, "p_zz"), 2.0);
        assert_eq!(
            read(&tensor_ddot(&a, &b_zz, "A", "B").unwrap(), "tensor_ddot"),
            49.0
        );

        let one = tensor_dot(&a, &b, "A", "B", "p").unwrap();
        assert_eq!(
            one.get(0).unwrap().read().components(),
            ["p_xx", "p_xy", "p_yx", "p_yy"]
        );
        assert_eq!(
            read(&tensor_ddot(&a, &b, "A", "B").unwrap(), "tensor_ddot"),
            47.0
        );
    }

    /// In 3-D, a symmetric `A` by a whole `B`, with arbitrary values: every
    /// entry of `A·B` and `A:B` is the matrix algebra's, the off-diagonal `xz`
    /// and `yz` terms included.
    #[test]
    fn a_3d_contraction_matches_the_matrix_algebra() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<Node> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        let ids: Vec<_> = n.iter().map(Node::id).collect();
        sm.add_cell(&ids).unwrap();
        let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();

        let sym = Matrix3::new(1.5, -0.7, 2.1, -0.7, 0.3, 0.9, 2.1, 0.9, -1.2);
        let whole = Matrix3::new(0.4, 1.1, -2.3, 3.7, -0.6, 0.8, 1.9, 2.6, -0.5);
        let axes = ["x", "y", "z"];
        let mut pairs: Vec<(String, f64)> = Vec::new();
        for i in 0..3 {
            for j in i..3 {
                pairs.push((format!("A_{}{}", axes[i], axes[j]), sym[(i, j)]));
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                pairs.push((format!("B_{}{}", axes[i], axes[j]), whole[(i, j)]));
            }
        }
        let refs: Vec<(&str, f64)> = pairs.iter().map(|(c, v)| (c.as_str(), *v)).collect();
        let t = uniform(&fes, &refs);

        let p = tensor_dot(&t, &t, "A", "B", "p").unwrap();
        let expected = sym * whole;
        for i in 0..3 {
            for j in 0..3 {
                let got = read(&p, &format!("p_{}{}", axes[i], axes[j]));
                assert!((got - expected[(i, j)]).abs() < 1e-12, "p_{i}{j}");
            }
        }
        let ddot = read(&tensor_ddot(&t, &t, "A", "B").unwrap(), "tensor_ddot");
        assert!((ddot - sym.dot(&whole)).abs() < 1e-12);
    }

    /// Two subspaces, `b` on one only: one zone, on the subspace both carry,
    /// whatever the order of the operands. A zone that does not carry its
    /// tensor is refused.
    #[test]
    fn operands_pair_by_shared_subspace() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let first = triangle(&coords);
        let second = triangle(&coords);
        let mut a = ElementField::empty();
        a.add_sub(uniform(&first, &A).get(0).unwrap()).unwrap();
        a.add_sub(uniform(&second, &A).get(0).unwrap()).unwrap();
        let b = uniform(&first, &B);

        let ab = tensor_ddot(&a, &b, "A", "B").unwrap();
        let ba = tensor_ddot(&b, &a, "B", "A").unwrap();
        assert_eq!(ab.len(), 1);
        assert_eq!(ba.len(), 1);
        assert!(ab
            .get(0)
            .unwrap()
            .read()
            .support()
            .same_object(&first.get(0).unwrap()));
        assert_eq!(read(&ab, "tensor_ddot"), read(&ba, "tensor_ddot"));

        let partial = uniform(&first, &[("A_xx", 1.0)]);
        assert!(tensor_ddot(&partial, &b, "A", "B").is_err());
    }
}
