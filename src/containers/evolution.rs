//! Evolution — a value tabulated against a variable, with linear
//! interpolation.
//!
//! An *evolution* associates a series of **values** to a **variable** (often
//! time, but not necessarily) and interpolates linearly between the tabulated
//! samples. The value at each sample may be a **scalar**, a **node sub-field**
//! ([`SubNodeField`]) or an **element sub-field** ([`SubElementField`]).
//!
//! Hierarchy mirroring [`crate::containers::node_field`] /
//! [`crate::containers::element_field`]:
//!
//! - [`SubEvolution`] — one **table**: a sorted list of abscissas plus the
//!   matching list of [`SubValue`]s (all of the same kind, fields on the same
//!   support). Interpolating it at `x` yields a single [`SubValue`].
//! - [`Evolution`] — aggregate of [`SubEvolution`], one per zone (just as a
//!   [`NodeField`] aggregates [`SubNodeField`]s, one per zone). Interpolating
//!   it regroups the per-zone results into a [`NodeField`] / [`ElementField`]
//!   (or, for scalars, a plain `Vec<f64>` — there is no aggregate of floats).
//!
//! The blend between two bracketing field samples reuses the field arithmetic
//! of [`crate::containers::field`] (`map_all` + `combine`), so no numerics are
//! duplicated here.
//!
//! # Out-of-range policy
//!
//! Each evolution carries an [`OutOfRange`] policy (default
//! [`OutOfRange::Error`]) that decides what happens when the requested
//! abscissa falls outside the tabulated range. The interpolation call may
//! **override** it for a single query.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::evolution::{SubEvolution, SubValue, OutOfRange};
//!
//! // A scalar X→Y curve: 0→10, 1→20.
//! let se = SubEvolution::new(
//!     vec![(0.0, SubValue::Scalar(10.0)), (1.0, SubValue::Scalar(20.0))],
//!     OutOfRange::Error,
//! )
//! .unwrap();
//! match se.interpolate(0.5, None).unwrap() {
//!     SubValue::Scalar(v) => assert_eq!(v, 15.0),
//!     _ => unreachable!(),
//! }
//! // Out of range with the default Error policy → error; Clamp → endpoint.
//! assert!(se.interpolate(2.0, None).is_err());
//! match se.interpolate(2.0, Some(OutOfRange::Clamp)).unwrap() {
//!     SubValue::Scalar(v) => assert_eq!(v, 20.0),
//!     _ => unreachable!(),
//! }
//! ```

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── OutOfRange policy ──────────────────────────────────────────────────────

/// What an interpolation does when the requested abscissa falls **outside**
/// the tabulated `[x_min, x_max]` range.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OutOfRange {
    /// Raise an error (the default).
    #[default]
    Error,
    /// Return the value of the nearest endpoint (no extrapolation).
    Clamp,
    /// Extend linearly using the extreme segment.
    Extrapolate,
}

impl OutOfRange {
    /// Parse a policy from its lowercase name (`"error"`, `"clamp"`,
    /// `"extrapolate"`). Used by the Python layer.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "error" => Ok(OutOfRange::Error),
            "clamp" => Ok(OutOfRange::Clamp),
            "extrapolate" => Ok(OutOfRange::Extrapolate),
            other => Err(PyrucastError::Message(format!(
                "unknown out_of_range policy '{other}' (expected 'error', 'clamp' or 'extrapolate')"
            ))),
        }
    }

    /// The policy's canonical name.
    pub fn name(self) -> &'static str {
        match self {
            OutOfRange::Error => "error",
            OutOfRange::Clamp => "clamp",
            OutOfRange::Extrapolate => "extrapolate",
        }
    }
}

// ─── SubValue ───────────────────────────────────────────────────────────────

/// The kind of value carried by a [`SubValue`] — used for homogeneity checks
/// and messages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueKind {
    Scalar,
    Node,
    Element,
}

impl ValueKind {
    fn label(self) -> &'static str {
        match self {
            ValueKind::Scalar => "scalar",
            ValueKind::Node => "node",
            ValueKind::Element => "element",
        }
    }
}

/// A single tabulated value: a scalar, a node sub-field, or an element
/// sub-field. Stored **inline** (owned) inside a [`SubEvolution`], like the
/// physics structs inside `SubModel`.
#[derive(Serialize, Deserialize, Clone)]
pub enum SubValue {
    /// A plain scalar.
    Scalar(f64),
    /// Values on the nodes of one zone (a POI1 support).
    Node(SubNodeField),
    /// Values per `(cell, Gauss point)` of one FE subspace.
    Element(SubElementField),
}

impl SubValue {
    /// This value's [`ValueKind`].
    pub fn kind(&self) -> ValueKind {
        match self {
            SubValue::Scalar(_) => ValueKind::Scalar,
            SubValue::Node(_) => ValueKind::Node,
            SubValue::Element(_) => ValueKind::Element,
        }
    }
}

/// Linear blend `lo*(1-t) + hi*t` of two values of the **same** kind. For
/// fields the per-value arithmetic of [`SubField`] (`map_all` + `combine`)
/// is reused; `combine` enforces same support and components.
fn lerp(lo: &SubValue, hi: &SubValue, t: f64) -> Result<SubValue> {
    match (lo, hi) {
        (SubValue::Scalar(a), SubValue::Scalar(b)) => {
            Ok(SubValue::Scalar(a * (1.0 - t) + b * t))
        }
        (SubValue::Node(a), SubValue::Node(b)) => {
            let la = a.map_all(|v| v * (1.0 - t));
            let lb = b.map_all(|v| v * t);
            Ok(SubValue::Node(la.combine(&lb, |x, y| x + y)?))
        }
        (SubValue::Element(a), SubValue::Element(b)) => {
            let la = a.map_all(|v| v * (1.0 - t));
            let lb = b.map_all(|v| v * t);
            Ok(SubValue::Element(la.combine(&lb, |x, y| x + y)?))
        }
        _ => Err(PyrucastError::Message(
            "evolution: cannot interpolate between values of different kinds".into(),
        )),
    }
}

// ─── SubEvolution ───────────────────────────────────────────────────────────

/// One tabulated curve: abscissas (sorted, strictly increasing) and the
/// matching values, all of the same [`ValueKind`] (and, for fields, on the
/// same support).
#[derive(Serialize, Deserialize, Clone)]
pub struct SubEvolution {
    /// Strictly increasing abscissas of the samples.
    abscissas: Vec<f64>,
    /// Tabulated values, aligned with `abscissas`.
    values: Vec<SubValue>,
    /// Out-of-range policy used when this curve is interpolated directly.
    out_of_range: OutOfRange,
}

impl SubEvolution {
    /// Build a curve from `(abscissa, value)` samples. The samples are sorted
    /// by abscissa; the values must be homogeneous in kind and, for fields, on
    /// the same support with the same component count.
    ///
    /// Errors:
    /// - no sample, a `NaN` abscissa, or duplicate abscissas;
    /// - mixed value kinds, or field values on different supports.
    pub fn new(mut samples: Vec<(f64, SubValue)>, out_of_range: OutOfRange) -> Result<Self> {
        if samples.is_empty() {
            return Err(PyrucastError::Message(
                "SubEvolution requires at least one sample".into(),
            ));
        }
        for (x, _) in &samples {
            if x.is_nan() {
                return Err(PyrucastError::Message(
                    "SubEvolution: NaN abscissa".into(),
                ));
            }
        }
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN abscissas"));
        for w in samples.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(PyrucastError::Message(format!(
                    "SubEvolution: duplicate abscissa {}",
                    w[0].0
                )));
            }
        }

        let kind = samples[0].1.kind();
        for (_, v) in &samples[1..] {
            if v.kind() != kind {
                return Err(PyrucastError::Message(format!(
                    "SubEvolution: mixed value kinds ({} and {})",
                    kind.label(),
                    v.kind().label()
                )));
            }
        }
        // For field values, every sample must share the first's support and
        // component count, so the blend (which uses `combine`) is well-defined.
        check_same_support(&samples)?;

        let (abscissas, values): (Vec<f64>, Vec<SubValue>) = samples.into_iter().unzip();
        Ok(SubEvolution {
            abscissas,
            values,
            out_of_range,
        })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.abscissas.len()
    }

    /// Whether the curve holds no sample (never true for a constructed one).
    pub fn is_empty(&self) -> bool {
        self.abscissas.is_empty()
    }

    /// The sorted abscissas.
    pub fn abscissas(&self) -> &[f64] {
        &self.abscissas
    }

    /// This curve's value kind.
    pub fn kind(&self) -> ValueKind {
        self.values[0].kind()
    }

    /// This curve's stored out-of-range policy.
    pub fn out_of_range(&self) -> OutOfRange {
        self.out_of_range
    }

    /// Interpolate at `x`. `policy` overrides the stored [`OutOfRange`] for
    /// this single query when `Some`.
    pub fn interpolate(&self, x: f64, policy: Option<OutOfRange>) -> Result<SubValue> {
        let policy = policy.unwrap_or(self.out_of_range);
        let xs = &self.abscissas;
        let n = xs.len();
        // `new` guarantees n >= 1.

        if x < xs[0] {
            return self.below_range(x, policy);
        }
        if x > xs[n - 1] {
            return self.above_range(x, policy);
        }
        // In range: locate the bracketing segment.
        match xs.binary_search_by(|v| v.partial_cmp(&x).expect("non-NaN abscissas")) {
            Ok(i) => Ok(self.values[i].clone()), // exact hit
            Err(i) => {
                // xs[i-1] < x < xs[i], with 1 <= i <= n-1.
                let (lo, hi) = (i - 1, i);
                let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
                lerp(&self.values[lo], &self.values[hi], t)
            }
        }
    }

    fn below_range(&self, x: f64, policy: OutOfRange) -> Result<SubValue> {
        let xs = &self.abscissas;
        match policy {
            OutOfRange::Error => Err(self.range_error(x)),
            OutOfRange::Clamp => Ok(self.values[0].clone()),
            OutOfRange::Extrapolate => {
                if xs.len() == 1 {
                    Ok(self.values[0].clone())
                } else {
                    let t = (x - xs[0]) / (xs[1] - xs[0]);
                    lerp(&self.values[0], &self.values[1], t)
                }
            }
        }
    }

    fn above_range(&self, x: f64, policy: OutOfRange) -> Result<SubValue> {
        let xs = &self.abscissas;
        let n = xs.len();
        match policy {
            OutOfRange::Error => Err(self.range_error(x)),
            OutOfRange::Clamp => Ok(self.values[n - 1].clone()),
            OutOfRange::Extrapolate => {
                if n == 1 {
                    Ok(self.values[n - 1].clone())
                } else {
                    let t = (x - xs[n - 2]) / (xs[n - 1] - xs[n - 2]);
                    lerp(&self.values[n - 2], &self.values[n - 1], t)
                }
            }
        }
    }

    fn range_error(&self, x: f64) -> PyrucastError {
        let xs = &self.abscissas;
        PyrucastError::Message(format!(
            "evolution: abscissa {x} is outside the tabulated range [{}, {}]",
            xs[0],
            xs[xs.len() - 1]
        ))
    }
}

/// Verify that all field samples share the first sample's support and
/// component count (scalars are unconstrained).
fn check_same_support(samples: &[(f64, SubValue)]) -> Result<()> {
    match &samples[0].1 {
        SubValue::Scalar(_) => Ok(()),
        SubValue::Node(f0) => {
            for (_, v) in &samples[1..] {
                if let SubValue::Node(f) = v {
                    if !f.same_support(f0) || f.component_count() != f0.component_count() {
                        return Err(PyrucastError::Message(
                            "SubEvolution: node values must share the same support and components"
                                .into(),
                        ));
                    }
                }
            }
            Ok(())
        }
        SubValue::Element(f0) => {
            for (_, v) in &samples[1..] {
                if let SubValue::Element(f) = v {
                    if !f.same_support(f0) || f.component_count() != f0.component_count() {
                        return Err(PyrucastError::Message(
                            "SubEvolution: element values must share the same support and components"
                                .into(),
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

impl fmt::Debug for SubEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubEvolution")
            .field("samples", &self.abscissas.len())
            .field("kind", &self.kind().label())
            .field("out_of_range", &self.out_of_range.name())
            .finish()
    }
}

impl fmt::Display for SubEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubEvolution: {} sample(s), {} value(s), out_of_range={}",
            self.abscissas.len(),
            self.kind().label(),
            self.out_of_range.name()
        )
    }
}

impl crate::dump::Dump for SubEvolution {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let headers = vec!["abscissa".to_string(), "value".to_string()];
        let rows: Vec<Vec<String>> = self
            .abscissas
            .iter()
            .zip(&self.values)
            .map(|(x, v)| {
                let cell = match v {
                    SubValue::Scalar(s) => fmt_float(*s, opts.precision),
                    SubValue::Node(f) => {
                        format!("SubNodeField({} node(s))", f.node_count())
                    }
                    SubValue::Element(f) => format!(
                        "SubElementField({} cell(s))",
                        f.cell_count()
                    ),
                };
                vec![fmt_float(*x, opts.precision), cell]
            })
            .collect();
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── Evolution (aggregate) ──────────────────────────────────────────────────

/// Aggregate of [`SubEvolution`] — one curve per zone.
///
/// Mirrors the [`NodeField`] / [`ElementField`] hierarchies: a list of
/// sub-evolution handles with the uniform [`Aggregate`] grammar (`len`,
/// indexing, iteration, `|`). All sub-evolutions must carry the same value
/// kind. Interpolating the aggregate interpolates every curve and regroups
/// the results (see [`Evolution::interpolate`]).
///
/// The aggregate carries its own [`OutOfRange`] policy, applied to every
/// curve at interpolation time. Note that the structural union (`a | b`) and
/// slicing reset the policy to the default [`OutOfRange::Error`].
#[derive(Serialize, Deserialize, Default)]
pub struct Evolution {
    subs: Vec<Handle<SubEvolution>>,
    out_of_range: OutOfRange,
}

crate::impl_aggregate!(Evolution, SubEvolution, sub_evolution, "sub-evolution(s)", {
    /// Reject a sub-evolution whose value kind differs from the others'.
    fn check_push(&self, h: &Handle<SubEvolution>) -> Result<()> {
        if self.subs.is_empty() {
            return Ok(());
        }
        let k0 = read(&self.subs[0])?.kind();
        let k = read(h)?.kind();
        if k != k0 {
            return Err(PyrucastError::Message(format!(
                "Evolution: cannot mix {} and {} sub-evolutions",
                k0.label(),
                k.label()
            )));
        }
        Ok(())
    }
});
crate::impl_aggregate_dump!(Evolution);

/// The result of interpolating an [`Evolution`]: an aggregate of the per-zone
/// results, except for scalars where a plain `Vec<f64>` is returned (there is
/// no aggregate of floats).
pub enum Interpolated {
    /// One scalar per sub-evolution.
    Scalars(Vec<f64>),
    /// One node sub-field per sub-evolution, regrouped into a [`NodeField`].
    Node(NodeField),
    /// One element sub-field per sub-evolution, regrouped into an [`ElementField`].
    Element(ElementField),
}

impl Evolution {
    /// The aggregate's stored out-of-range policy.
    pub fn out_of_range(&self) -> OutOfRange {
        self.out_of_range
    }

    /// Interpolate every sub-evolution at `x` and regroup the results.
    ///
    /// `policy` overrides the aggregate's [`OutOfRange`] for this query when
    /// `Some`; the effective policy is applied to every curve.
    pub fn interpolate(&self, x: f64, policy: Option<OutOfRange>) -> Result<Interpolated> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution: cannot interpolate an empty evolution".into(),
            ));
        }
        let policy = Some(policy.unwrap_or(self.out_of_range));
        match read(&self.subs[0])?.kind() {
            ValueKind::Scalar => {
                let mut out = Vec::with_capacity(self.subs.len());
                for h in &self.subs {
                    match read(h)?.interpolate(x, policy)? {
                        SubValue::Scalar(v) => out.push(v),
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Scalars(out))
            }
            ValueKind::Node => {
                let mut field = NodeField::default();
                for h in &self.subs {
                    match read(h)?.interpolate(x, policy)? {
                        SubValue::Node(sf) => field.add_sub(insert(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Node(field))
            }
            ValueKind::Element => {
                let mut field = ElementField::default();
                for h in &self.subs {
                    match read(h)?.interpolate(x, policy)? {
                        SubValue::Element(sf) => field.add_sub(insert(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Element(field))
            }
        }
    }

    /// Build a single-curve scalar `Evolution` from `(abscissa, scalar)`
    /// samples — the classic X→Y curve (one sub-evolution).
    pub fn from_scalars(samples: Vec<(f64, f64)>, out_of_range: OutOfRange) -> Result<Self> {
        let pairs = samples
            .into_iter()
            .map(|(x, v)| (x, SubValue::Scalar(v)))
            .collect();
        let sub = SubEvolution::new(pairs, out_of_range)?;
        Ok(Evolution {
            subs: vec![insert(sub)],
            out_of_range,
        })
    }

    /// Build an `Evolution` from whole node fields tabulated at each abscissa
    /// (time-major). The fields are **transposed** into one curve per zone:
    /// zones are identified by their support (matched across steps), so every
    /// step must carry the same set of supports as the first.
    pub fn from_node_fields(
        samples: &[(f64, &NodeField)],
        out_of_range: OutOfRange,
    ) -> Result<Self> {
        if samples.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution::from_node_fields: no sample".into(),
            ));
        }
        let first = samples[0].1;
        if first.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution::from_node_fields: the first field has no zone".into(),
            ));
        }
        let mut evolution = Evolution {
            subs: Vec::with_capacity(first.len()),
            out_of_range,
        };
        for z in 0..first.len() {
            let support = read(&first.get(z)?)?.support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((*x, SubValue::Node(node_sub_on(field, &support)?)));
            }
            evolution
                .subs
                .push(insert(SubEvolution::new(curve, out_of_range)?));
        }
        Ok(evolution)
    }

    /// Build an `Evolution` from whole element fields tabulated at each
    /// abscissa (time-major). Transposed into one curve per FE subspace zone,
    /// matched across steps by support — see [`Evolution::from_node_fields`].
    pub fn from_element_fields(
        samples: &[(f64, &ElementField)],
        out_of_range: OutOfRange,
    ) -> Result<Self> {
        if samples.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution::from_element_fields: no sample".into(),
            ));
        }
        let first = samples[0].1;
        if first.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution::from_element_fields: the first field has no zone".into(),
            ));
        }
        let mut evolution = Evolution {
            subs: Vec::with_capacity(first.len()),
            out_of_range,
        };
        for z in 0..first.len() {
            let support = read(&first.get(z)?)?.support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((*x, SubValue::Element(element_sub_on(field, &support)?)));
            }
            evolution
                .subs
                .push(insert(SubEvolution::new(curve, out_of_range)?));
        }
        Ok(evolution)
    }
}

fn mixed_kind_err() -> PyrucastError {
    PyrucastError::Message("Evolution: inconsistent value kinds across sub-evolutions".into())
}

/// Clone of the `field`'s sub-field on the given POI1 `support` (matched by
/// store slot). Errors if no zone carries that support.
fn node_sub_on(field: &NodeField, support: &Handle<SubMesh>) -> Result<SubNodeField> {
    for h in field.iter() {
        let s = read(h)?;
        if s.support().same_slot(support) {
            return Ok((*s).clone());
        }
    }
    Err(PyrucastError::Message(
        "Evolution::from_node_fields: a step is missing a zone present in the first step".into(),
    ))
}

/// Clone of the `field`'s sub-field on the given FE-subspace `support`.
fn element_sub_on(
    field: &ElementField,
    support: &Handle<SubFiniteElementSpace>,
) -> Result<SubElementField> {
    for h in field.iter() {
        let s = read(h)?;
        if s.support().same_slot(support) {
            return Ok((*s).clone());
        }
    }
    Err(PyrucastError::Message(
        "Evolution::from_element_fields: a step is missing a zone present in the first step".into(),
    ))
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, ElementType, Node, SubMesh};
    use crate::store::{insert, Handle};

    fn scalar_curve(samples: &[(f64, f64)], oor: OutOfRange) -> SubEvolution {
        SubEvolution::new(
            samples.iter().map(|&(x, v)| (x, SubValue::Scalar(v))).collect(),
            oor,
        )
        .unwrap()
    }

    /// A POI1 support over `n` nodes on a fresh 1-D Coords.
    fn poi1(n: usize) -> Handle<SubMesh> {
        let coords = insert(Coords::new(1).unwrap());
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for i in 0..n {
            let nd = Node::create_in(coords.clone(), &[i as f64]).unwrap();
            sm.add_cell(&[nd.id()]).unwrap();
        }
        insert(sm)
    }

    fn node_field(sm: &Handle<SubMesh>, vals: &[f64]) -> SubNodeField {
        let mut f = SubNodeField::from_poi1(sm, vec!["T".into()]).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            f.set(i, 0, v).unwrap();
        }
        f
    }

    fn as_scalar(v: SubValue) -> f64 {
        match v {
            SubValue::Scalar(s) => s,
            _ => panic!("expected a scalar"),
        }
    }

    #[test]
    fn scalar_interpolation_midpoint_and_exact() {
        let se = scalar_curve(&[(0.0, 10.0), (1.0, 20.0)], OutOfRange::Error);
        assert_eq!(as_scalar(se.interpolate(0.5, None).unwrap()), 15.0);
        assert_eq!(as_scalar(se.interpolate(0.0, None).unwrap()), 10.0);
        assert_eq!(as_scalar(se.interpolate(1.0, None).unwrap()), 20.0);
        assert_eq!(as_scalar(se.interpolate(0.25, None).unwrap()), 12.5);
    }

    #[test]
    fn samples_are_sorted_and_duplicates_rejected() {
        // Unsorted input is sorted on construction.
        let se = scalar_curve(&[(2.0, 30.0), (0.0, 10.0), (1.0, 20.0)], OutOfRange::Error);
        assert_eq!(se.abscissas(), &[0.0, 1.0, 2.0]);
        assert_eq!(as_scalar(se.interpolate(1.5, None).unwrap()), 25.0);
        // Duplicate abscissa is rejected.
        assert!(SubEvolution::new(
            vec![(0.0, SubValue::Scalar(1.0)), (0.0, SubValue::Scalar(2.0))],
            OutOfRange::Error
        )
        .is_err());
    }

    #[test]
    fn out_of_range_policies() {
        let se = scalar_curve(&[(0.0, 10.0), (1.0, 20.0)], OutOfRange::Error);
        // Default Error policy.
        assert!(se.interpolate(2.0, None).is_err());
        assert!(se.interpolate(-1.0, None).is_err());
        // Clamp → endpoints.
        assert_eq!(as_scalar(se.interpolate(2.0, Some(OutOfRange::Clamp)).unwrap()), 20.0);
        assert_eq!(as_scalar(se.interpolate(-1.0, Some(OutOfRange::Clamp)).unwrap()), 10.0);
        // Extrapolate → linear extension (slope 10/unit).
        assert_eq!(as_scalar(se.interpolate(2.0, Some(OutOfRange::Extrapolate)).unwrap()), 30.0);
        assert_eq!(as_scalar(se.interpolate(-1.0, Some(OutOfRange::Extrapolate)).unwrap()), 0.0);
    }

    #[test]
    fn stored_policy_used_and_overridable() {
        let se = scalar_curve(&[(0.0, 10.0), (1.0, 20.0)], OutOfRange::Clamp);
        // Stored Clamp used when no override.
        assert_eq!(as_scalar(se.interpolate(2.0, None).unwrap()), 20.0);
        // Call-time Error overrides the stored Clamp.
        assert!(se.interpolate(2.0, Some(OutOfRange::Error)).is_err());
    }

    #[test]
    fn node_field_interpolation_blends_per_node() {
        let sm = poi1(2);
        let a = node_field(&sm, &[0.0, 10.0]);
        let b = node_field(&sm, &[2.0, 20.0]);
        let se = SubEvolution::new(
            vec![(0.0, SubValue::Node(a)), (1.0, SubValue::Node(b))],
            OutOfRange::Error,
        )
        .unwrap();
        match se.interpolate(0.5, None).unwrap() {
            SubValue::Node(f) => {
                assert_eq!(f.get(0, 0).unwrap(), 1.0); // (0 + 2)/2
                assert_eq!(f.get(1, 0).unwrap(), 15.0); // (10 + 20)/2
            }
            _ => panic!("expected a node value"),
        }
    }

    #[test]
    fn node_values_on_different_supports_rejected() {
        let a = node_field(&poi1(2), &[0.0, 0.0]);
        let b = node_field(&poi1(2), &[1.0, 1.0]); // different support
        assert!(SubEvolution::new(
            vec![(0.0, SubValue::Node(a)), (1.0, SubValue::Node(b))],
            OutOfRange::Error
        )
        .is_err());
    }

    #[test]
    fn aggregate_scalar_interpolation_returns_list() {
        let e = Evolution::from_scalars(vec![(0.0, 10.0), (1.0, 20.0)], OutOfRange::Error).unwrap();
        match e.interpolate(0.5, None).unwrap() {
            Interpolated::Scalars(v) => assert_eq!(v, vec![15.0]),
            _ => panic!("expected scalars"),
        }
    }

    #[test]
    fn aggregate_rejects_mixed_kinds() {
        let scalar = insert(scalar_curve(&[(0.0, 1.0)], OutOfRange::Error));
        let node = insert(
            SubEvolution::new(
                vec![(0.0, SubValue::Node(node_field(&poi1(1), &[1.0])))],
                OutOfRange::Error,
            )
            .unwrap(),
        );
        let mut e = Evolution::default();
        e.add_sub(scalar).unwrap();
        assert!(e.add_sub(node).is_err());
    }

    #[test]
    fn union_collects_sub_evolutions() {
        let e0 = Evolution::from_scalars(vec![(0.0, 1.0), (1.0, 2.0)], OutOfRange::Error).unwrap();
        let e1 = Evolution::from_scalars(vec![(0.0, 3.0), (1.0, 4.0)], OutOfRange::Error).unwrap();
        let u = e0.union(&e1).unwrap();
        assert_eq!(u.len(), 2);
        match u.interpolate(0.5, None).unwrap() {
            Interpolated::Scalars(v) => assert_eq!(v, vec![1.5, 3.5]),
            _ => panic!("expected scalars"),
        }
    }

    #[test]
    fn from_node_fields_transposes_per_zone() {
        let sm = poi1(2);
        let f0 = NodeField::from_sub(node_field(&sm, &[0.0, 0.0]));
        let f1 = NodeField::from_sub(node_field(&sm, &[10.0, 20.0]));
        let e = Evolution::from_node_fields(
            &[(0.0, &f0), (1.0, &f1)],
            OutOfRange::Error,
        )
        .unwrap();
        assert_eq!(e.len(), 1);
        match e.interpolate(0.5, None).unwrap() {
            Interpolated::Node(field) => {
                let sub = read(&field.get(0).unwrap()).unwrap();
                assert_eq!(sub.get(0, 0).unwrap(), 5.0);
                assert_eq!(sub.get(1, 0).unwrap(), 10.0);
            }
            _ => panic!("expected a node field"),
        }
    }
}
