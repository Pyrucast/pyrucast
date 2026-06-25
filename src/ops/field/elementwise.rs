//! Element-wise unary maths on fields (`cos`, `exp`, `sqrt`, …).
//!
//! Each function returns a **new** field with the chosen `f64` function
//! applied to every value — all nodes/points × all components × all zones.
//! They work uniformly on a single zone (`SubNodeField` / `SubElementField`)
//! and on an aggregate (`NodeField` / `ElementField`), through the
//! [`MapValues`] trait.
//!
//! There is **no new core logic**: this is a thin naming layer over the
//! existing primitives — [`crate::containers::field::SubField::map_all`] for
//! one zone and [`crate::containers::field::Field::map_subs`] to fan over
//! zones. As with the field arithmetic, results are **unguarded**
//! (numpy-like): `log` of a non-positive value yields `-inf` / `nan`,
//! `sqrt` of a negative yields `nan`.
//!
//! Rust has no `cos(x)`-style operator syntax, so these are plain named
//! functions here (e.g. `ops::field::cos(&f)`); the Python binding exposes
//! them flat as `pyrucast.cos(f)`, …, mirroring numpy.
//!
//! The functions are generic over [`MapValues`] (defined next to the field
//! traits), which unifies the zone (`map_all`) and aggregate (`map_all`)
//! element-wise maps so a single definition serves all four field types.

use crate::containers::field::MapValues;
use crate::error::Result;

macro_rules! field_unary {
    ($(#[$doc:meta])* $name:ident, $f:expr) => {
        $(#[$doc])*
        pub fn $name<T: MapValues>(field: &T) -> Result<T> {
            field.map_values($f)
        }
    };
}

field_unary!(
    /// Element-wise absolute value.
    abs, f64::abs
);
field_unary!(
    /// Element-wise square root (`nan` for negative values).
    sqrt, f64::sqrt
);
field_unary!(
    /// Element-wise exponential `eˣ`.
    exp, f64::exp
);
field_unary!(
    /// Element-wise natural logarithm (`-inf` / `nan` for values ≤ 0).
    log, f64::ln
);
field_unary!(
    /// Element-wise base-10 logarithm.
    log10, f64::log10
);
field_unary!(
    /// Element-wise cosine (argument in radians).
    cos, f64::cos
);
field_unary!(
    /// Element-wise sine (argument in radians).
    sin, f64::sin
);
field_unary!(
    /// Element-wise tangent (argument in radians).
    tan, f64::tan
);
field_unary!(
    /// Element-wise hyperbolic sine.
    sinh, f64::sinh
);
field_unary!(
    /// Element-wise hyperbolic cosine.
    cosh, f64::cosh
);
field_unary!(
    /// Element-wise hyperbolic tangent.
    tanh, f64::tanh
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, ElementType, Node, NodeId, SubMesh};
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::store::insert;

    /// Build a single-zone `SubNodeField` named "T" carrying `values`,
    /// returning it together with the node ids (to read values back).
    fn make_field(values: &[f64]) -> (SubNodeField, Vec<NodeId>) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..values.len())
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id()).collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            for n in &nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        for (i, &v) in values.iter().enumerate() {
            f.set(i, 0, v).unwrap();
        }
        (f, ids)
    }

    #[test]
    fn cos_on_subfield_maps_every_value() {
        let (f, ids) = make_field(&[0.0, std::f64::consts::PI]);
        let g = cos(&f).unwrap();
        assert!((g.value(ids[0], "T").unwrap() - 1.0).abs() < 1e-12);
        assert!((g.value(ids[1], "T").unwrap() + 1.0).abs() < 1e-12);
        // Original untouched.
        assert_eq!(f.value(ids[0], "T").unwrap(), 0.0);
    }

    #[test]
    fn sqrt_and_exp_and_log() {
        let (f, ids) = make_field(&[4.0, 1.0]);
        assert!((sqrt(&f).unwrap().value(ids[0], "T").unwrap() - 2.0).abs() < 1e-12);
        assert!((exp(&f).unwrap().value(ids[1], "T").unwrap() - std::f64::consts::E).abs() < 1e-12);
        assert!(log(&f).unwrap().value(ids[1], "T").unwrap().abs() < 1e-12); // ln(1) = 0
    }

    #[test]
    fn unguarded_log_of_negative_is_nan() {
        let (f, ids) = make_field(&[-1.0]);
        assert!(log(&f).unwrap().value(ids[0], "T").unwrap().is_nan());
    }

    #[test]
    fn works_on_aggregate() {
        let (sub, ids) = make_field(&[0.0]);
        let agg = NodeField::from_sub(sub);
        let g = cos(&agg).unwrap();
        assert!((g.value(ids[0], "T").unwrap() - 1.0).abs() < 1e-12);
    }
}
