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
use crate::containers::field::{Field, SubField};
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::parallel::*;
use crate::store::{insert, read, write, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

/// One labelled `(abscissa, value)` curve per zone — the input of a scalar
/// X-Y plot ([`Evolution::scalar_series_set`]).
pub type ScalarSeriesSet = Vec<(String, Vec<(f64, f64)>)>;

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
        (SubValue::Scalar(a), SubValue::Scalar(b)) => Ok(SubValue::Scalar(a * (1.0 - t) + b * t)),
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
    /// Physical **type** of the abscissa (e.g. `"T"`, `"time"`). Optional; used
    /// to label plots and — when a field is mapped through a scalar curve — to
    /// select which field component to look up (see
    /// [`SubEvolution::interpolate_field`]).
    #[serde(default)]
    abscissa_type: Option<String>,
    /// Physical **type** of the ordinate, for **scalar** curves only (e.g.
    /// `"young"`). Optional; labels plots and names the component produced when
    /// a field is mapped through the curve.
    #[serde(default)]
    ordinate_type: Option<String>,
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
                return Err(PyrucastError::Message("SubEvolution: NaN abscissa".into()));
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
            abscissa_type: None,
            ordinate_type: None,
        })
    }

    /// The abscissa's physical type, if set.
    pub fn abscissa_type(&self) -> Option<&str> {
        self.abscissa_type.as_deref()
    }

    /// The ordinate's physical type (scalar curves), if set.
    pub fn ordinate_type(&self) -> Option<&str> {
        self.ordinate_type.as_deref()
    }

    /// Set (or clear) the abscissa's physical type.
    pub fn set_abscissa_type(&mut self, t: Option<String>) {
        self.abscissa_type = t;
    }

    /// Set (or clear) the ordinate's physical type. Errors if a type is given
    /// for a curve that carries fields rather than scalars — only a scalar
    /// curve has an ordinate to type.
    pub fn set_ordinate_type(&mut self, t: Option<String>) -> Result<()> {
        if t.is_some() && self.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(format!(
                "ordinate type applies to scalar evolutions only (this one is {})",
                self.kind().label()
            )));
        }
        self.ordinate_type = t;
        Ok(())
    }

    /// Builder form of [`SubEvolution::set_abscissa_type`].
    pub fn with_abscissa_type(mut self, t: Option<String>) -> Self {
        self.abscissa_type = t;
        self
    }

    /// Builder form of [`SubEvolution::set_ordinate_type`] (validated).
    pub fn with_ordinate_type(mut self, t: Option<String>) -> Result<Self> {
        self.set_ordinate_type(t)?;
        Ok(self)
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

    /// The `(abscissa, value)` points of a **scalar** curve, in abscissa
    /// order. Errors if the curve carries fields rather than scalars.
    pub fn scalar_series(&self) -> Result<Vec<(f64, f64)>> {
        if self.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "scalar_series: this evolution carries fields, not scalars".into(),
            ));
        }
        Ok(self
            .abscissas
            .iter()
            .zip(&self.values)
            .map(|(x, v)| match v {
                SubValue::Scalar(s) => (*x, *s),
                _ => unreachable!("kind checked above"),
            })
            .collect())
    }

    /// The `k`-th tabulated value (a clone). Errors if `k` is out of range.
    pub fn value_at(&self, k: usize) -> Result<SubValue> {
        self.values.get(k).cloned().ok_or_else(|| {
            PyrucastError::Message(format!(
                "SubEvolution: frame {} out of range (len={})",
                k,
                self.values.len()
            ))
        })
    }

    /// Plot this single curve — see [`Evolution::plot`]. Delegates to a
    /// one-zone [`Evolution`] wrapping a clone of `self`.
    #[cfg(feature = "viz")]
    #[allow(clippy::too_many_arguments)]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        mesh: Option<&crate::containers::mesh::Mesh>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        smooth: usize,
        frame: Option<usize>,
        x_label: Option<&str>,
        y_label: Option<&str>,
        title: Option<&str>,
    ) -> Result<()> {
        let evo = Evolution {
            subs: vec![insert(self.clone())],
            out_of_range: self.out_of_range,
        };
        evo.plot(
            view, save, mesh, component, scale, smooth, frame, x_label, y_label, title,
        )
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

    /// Interpolate a **scalar** curve at `x`, returning the value directly.
    /// Errors if this curve carries fields rather than scalars.
    pub fn eval_scalar(&self, x: f64, policy: Option<OutOfRange>) -> Result<f64> {
        match self.interpolate(x, policy)? {
            SubValue::Scalar(v) => Ok(v),
            _ => Err(PyrucastError::Message(
                "evolution: a field can only be mapped through a scalar-valued evolution".into(),
            )),
        }
    }

    /// Map every value of `field`'s abscissa-typed component through this
    /// **scalar** curve, node by node (or Gauss point by Gauss point).
    ///
    /// The curve acts as a transfer function `y = f(x)`: the field component
    /// **named like the abscissa type** supplies the `x` values; the result is
    /// a fresh single-component field on the **same support**, its component
    /// named after the ordinate type (or `"value"` when none is set).
    ///
    /// Errors if the curve is not scalar-valued, has no abscissa type set, or
    /// the field has no component matching that type (the requested
    /// **type correspondence**).
    pub fn interpolate_field<F>(&self, field: &F, policy: Option<OutOfRange>) -> Result<F>
    where
        F: SubField + Clone,
    {
        if self.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "evolution: a field can only be mapped through a scalar-valued evolution".into(),
            ));
        }
        let abscissa = self.abscissa_type.as_deref().ok_or_else(|| {
            PyrucastError::Message(
                "evolution: set the abscissa type to map a field \
                 (it selects which field component to look up)"
                    .into(),
            )
        })?;
        let ci = field.component_index(abscissa).ok_or_else(|| {
            PyrucastError::Message(format!(
                "evolution: the field has no component '{abscissa}' matching the abscissa type"
            ))
        })?;
        let out_name = self
            .ordinate_type
            .clone()
            .unwrap_or_else(|| "value".to_string());
        let nc = field.component_count();
        // Per-row map in abscissa order (each output written once ⇒
        // thread-count-independent); short-circuits on the first out-of-range.
        let out_vals: Vec<f64> = field
            .values()
            .par_chunks(nc)
            .with_min_len((MIN_PARALLEL_LEN / nc.max(1)).max(1))
            .map(|row| self.eval_scalar(row[ci], policy))
            .collect::<Result<Vec<f64>>>()?;
        let mut out = field.same_support_with(vec![out_name])?;
        out.values_mut().copy_from_slice(&out_vals);
        Ok(out)
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
                    SubValue::Element(f) => format!("SubElementField({} cell(s))", f.cell_count()),
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

crate::impl_aggregate!(
    Evolution,
    SubEvolution,
    sub_evolution,
    "sub-evolution(s)",
    {
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
    }
);
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

    /// The abscissa's physical type (taken from the first sub-evolution), if
    /// set. `None` for an empty evolution.
    pub fn abscissa_type(&self) -> Result<Option<String>> {
        match self.subs.first() {
            Some(h) => Ok(read(h)?.abscissa_type().map(str::to_string)),
            None => Ok(None),
        }
    }

    /// The ordinate's physical type (scalar evolutions), if set. `None` for an
    /// empty evolution.
    pub fn ordinate_type(&self) -> Result<Option<String>> {
        match self.subs.first() {
            Some(h) => Ok(read(h)?.ordinate_type().map(str::to_string)),
            None => Ok(None),
        }
    }

    /// Set (or clear) the abscissa type on **every** sub-evolution.
    pub fn set_abscissa_type(&mut self, t: Option<String>) -> Result<()> {
        for h in &self.subs {
            write(h)?.set_abscissa_type(t.clone());
        }
        Ok(())
    }

    /// Set (or clear) the ordinate type on **every** sub-evolution. Errors on a
    /// field-valued evolution (only scalar curves have an ordinate to type).
    pub fn set_ordinate_type(&mut self, t: Option<String>) -> Result<()> {
        for h in &self.subs {
            write(h)?.set_ordinate_type(t.clone())?;
        }
        Ok(())
    }

    /// The single scalar curve of a one-curve, scalar-valued evolution — the
    /// transfer function used to map a field ([`Evolution::interpolate_node_field`]
    /// / [`Evolution::interpolate_element_field`]). Returns a clone. Errors
    /// unless the evolution holds exactly one scalar sub-evolution.
    fn single_scalar_curve(&self) -> Result<SubEvolution> {
        if self.subs.len() != 1 {
            return Err(PyrucastError::Message(format!(
                "evolution: mapping a field needs a single-curve evolution, but this one has {} curve(s)",
                self.subs.len()
            )));
        }
        let sub = read(&self.subs[0])?;
        if sub.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "evolution: a field can only be mapped through a scalar-valued evolution".into(),
            ));
        }
        Ok((*sub).clone())
    }

    /// Map a whole [`NodeField`] through the (single, scalar) curve — see
    /// [`SubEvolution::interpolate_field`]. Each zone's abscissa-typed
    /// component is looked up; the result is a node field of one component
    /// (named after the ordinate type) on the same decomposition.
    pub fn interpolate_node_field(
        &self,
        field: &NodeField,
        policy: Option<OutOfRange>,
    ) -> Result<NodeField> {
        let curve = self.single_scalar_curve()?;
        let policy = Some(policy.unwrap_or(self.out_of_range));
        field.map_subs(move |s| curve.interpolate_field(s, policy))
    }

    /// Map a whole [`ElementField`] through the (single, scalar) curve — see
    /// [`Evolution::interpolate_node_field`].
    pub fn interpolate_element_field(
        &self,
        field: &ElementField,
        policy: Option<OutOfRange>,
    ) -> Result<ElementField> {
        let curve = self.single_scalar_curve()?;
        let policy = Some(policy.unwrap_or(self.out_of_range));
        field.map_subs(move |s| curve.interpolate_field(s, policy))
    }

    /// The aggregate's value kind (taken from the first sub-evolution).
    /// Errors if the evolution is empty.
    pub fn kind(&self) -> Result<ValueKind> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message("Evolution: empty evolution".into()));
        }
        Ok(read(&self.subs[0])?.kind())
    }

    /// The abscissa grid **shared** by every sub-evolution, validated to be
    /// identical across zones (a global frame slider requires it). Errors on
    /// an empty evolution or mismatched grids.
    pub fn shared_abscissas(&self) -> Result<Vec<f64>> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message("Evolution: empty evolution".into()));
        }
        let first = read(&self.subs[0])?.abscissas().to_vec();
        for h in &self.subs[1..] {
            if read(h)?.abscissas() != first.as_slice() {
                return Err(PyrucastError::Message(
                    "Evolution: sub-evolutions have different abscissa grids — \
                     a frame index is ambiguous"
                        .into(),
                ));
            }
        }
        Ok(first)
    }

    /// Number of tabulated frames (length of the shared abscissa grid).
    pub fn frame_count(&self) -> Result<usize> {
        Ok(self.shared_abscissas()?.len())
    }

    /// Regroup the `k`-th tabulated node sub-field of every zone into a
    /// [`NodeField`]. Errors if the values are not node fields.
    pub fn node_frame(&self, k: usize) -> Result<NodeField> {
        let mut field = NodeField::default();
        for h in &self.subs {
            match read(h)?.value_at(k)? {
                SubValue::Node(sf) => field.add_sub(insert(sf))?,
                _ => return Err(not_node_err()),
            }
        }
        Ok(field)
    }

    /// Regroup the `k`-th tabulated element sub-field of every zone into an
    /// [`ElementField`]. Errors if the values are not element fields.
    pub fn element_frame(&self, k: usize) -> Result<ElementField> {
        let mut field = ElementField::default();
        for h in &self.subs {
            match read(h)?.value_at(k)? {
                SubValue::Element(sf) => field.add_sub(insert(sf))?,
                _ => return Err(not_element_err()),
            }
        }
        Ok(field)
    }

    /// One labelled `(abscissa, value)` series per sub-evolution — for a
    /// scalar X-Y plot. Errors if the values are not scalars.
    pub fn scalar_series_set(&self) -> Result<ScalarSeriesSet> {
        let mut out = Vec::with_capacity(self.subs.len());
        for (i, h) in self.subs.iter().enumerate() {
            let label = if self.subs.len() == 1 {
                "value".to_string()
            } else {
                format!("zone {i}")
            };
            out.push((label, read(h)?.scalar_series()?));
        }
        Ok(out)
    }

    /// Plot the evolution.
    ///
    /// - **scalar** evolution → an X-Y curve (one line per zone) ;
    /// - **field** evolution → the field rendered like [`crate::containers::mesh::Mesh::plot`],
    ///   with a frame slider (interactive) picking the tabulated value.
    ///
    /// `save=Some(path)` writes a PNG/SVG (a single `frame`, default = last for
    /// fields); `save=None` opens the interactive window. `mesh` supplies the
    /// surface for field evolutions (node frames default to a point cloud;
    /// element frames reconstruct their FE support when no mesh is given).
    #[cfg(feature = "viz")]
    #[allow(clippy::too_many_arguments)]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        mesh: Option<&crate::containers::mesh::Mesh>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        smooth: usize,
        frame: Option<usize>,
        x_label: Option<&str>,
        y_label: Option<&str>,
        title: Option<&str>,
    ) -> Result<()> {
        // Axis / slider labels fall back to the abscissa & ordinate **types**
        // (then to generic defaults) when the caller passes none.
        let abscissa_label = match x_label {
            Some(s) => s.to_string(),
            None => self.abscissa_type()?.unwrap_or_else(|| "variable".into()),
        };
        match self.kind()? {
            ValueKind::Scalar => {
                let y = match y_label {
                    Some(s) => s.to_string(),
                    None => self.ordinate_type()?.unwrap_or_else(|| "value".into()),
                };
                crate::viz::render_curve(
                    self.scalar_series_set()?,
                    &abscissa_label,
                    &y,
                    title.unwrap_or(""),
                    view,
                    save,
                )
            }
            ValueKind::Node => {
                let abscissas = self.shared_abscissas()?;
                let frames: Vec<crate::viz::FrameField> = (0..abscissas.len())
                    .map(|k| Ok(crate::viz::FrameField::Node(self.node_frame(k)?)))
                    .collect::<Result<_>>()?;
                crate::viz::render_evolution_field(
                    mesh,
                    &frames,
                    &abscissas,
                    &abscissa_label,
                    component,
                    scale,
                    smooth,
                    frame,
                    view,
                    save,
                )
            }
            ValueKind::Element => {
                let abscissas = self.shared_abscissas()?;
                let frames: Vec<crate::viz::FrameField> = (0..abscissas.len())
                    .map(|k| Ok(crate::viz::FrameField::Element(self.element_frame(k)?)))
                    .collect::<Result<_>>()?;
                // No mesh given → draw on the element field's own FE support.
                let reconstructed = match (mesh, frames.first()) {
                    (None, Some(crate::viz::FrameField::Element(ef))) => {
                        Some(element_support_mesh(ef)?)
                    }
                    _ => None,
                };
                let geom = mesh.or(reconstructed.as_ref());
                crate::viz::render_evolution_field(
                    geom,
                    &frames,
                    &abscissas,
                    &abscissa_label,
                    component,
                    scale,
                    smooth,
                    frame,
                    view,
                    save,
                )
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

fn not_node_err() -> PyrucastError {
    PyrucastError::Message("Evolution: expected node-field values".into())
}

fn not_element_err() -> PyrucastError {
    PyrucastError::Message("Evolution: expected element-field values".into())
}

/// Reconstruct the surface [`crate::containers::mesh::Mesh`] backing an
/// element field, from each zone's FE support sub-mesh — so an element-field
/// evolution can be plotted without the user re-supplying the geometry.
#[cfg(feature = "viz")]
fn element_support_mesh(field: &ElementField) -> Result<crate::containers::mesh::Mesh> {
    use crate::containers::field::SubField;
    let mut mesh = crate::containers::mesh::Mesh::empty();
    for h in field.iter() {
        let sub = read(h)?;
        let fes = read(&sub.support())?;
        mesh.add_sub(fes.submesh())?;
    }
    Ok(mesh)
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
            samples
                .iter()
                .map(|&(x, v)| (x, SubValue::Scalar(v)))
                .collect(),
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
        assert_eq!(
            as_scalar(se.interpolate(2.0, Some(OutOfRange::Clamp)).unwrap()),
            20.0
        );
        assert_eq!(
            as_scalar(se.interpolate(-1.0, Some(OutOfRange::Clamp)).unwrap()),
            10.0
        );
        // Extrapolate → linear extension (slope 10/unit).
        assert_eq!(
            as_scalar(se.interpolate(2.0, Some(OutOfRange::Extrapolate)).unwrap()),
            30.0
        );
        assert_eq!(
            as_scalar(se.interpolate(-1.0, Some(OutOfRange::Extrapolate)).unwrap()),
            0.0
        );
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
    fn scalar_series_set_labels_zones() {
        let e0 = Evolution::from_scalars(vec![(0.0, 1.0), (1.0, 2.0)], OutOfRange::Error).unwrap();
        let series = e0.scalar_series_set().unwrap();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].0, "value");
        assert_eq!(series[0].1, vec![(0.0, 1.0), (1.0, 2.0)]);
        // Two zones → labelled "zone 0" / "zone 1".
        let e1 = Evolution::from_scalars(vec![(0.0, 3.0), (1.0, 4.0)], OutOfRange::Error).unwrap();
        let u = e0.union(&e1).unwrap();
        let s = u.scalar_series_set().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].0, "zone 0");
        assert_eq!(s[1].0, "zone 1");
    }

    #[test]
    fn shared_abscissas_validates_grid() {
        let sm = poi1(2);
        let a = node_field(&sm, &[0.0, 0.0]);
        let b = node_field(&sm, &[10.0, 20.0]);
        let f0 = NodeField::from_sub(a);
        let f1 = NodeField::from_sub(b);
        let e = Evolution::from_node_fields(&[(0.0, &f0), (2.0, &f1)], OutOfRange::Error).unwrap();
        assert_eq!(e.shared_abscissas().unwrap(), vec![0.0, 2.0]);
        assert_eq!(e.frame_count().unwrap(), 2);
        // frame 1 = the second tabulated NodeField.
        let frame = e.node_frame(1).unwrap();
        let sub = read(&frame.get(0).unwrap()).unwrap();
        assert_eq!(sub.get(0, 0).unwrap(), 10.0);
        assert_eq!(sub.get(1, 0).unwrap(), 20.0);
    }

    #[test]
    fn from_node_fields_transposes_per_zone() {
        let sm = poi1(2);
        let f0 = NodeField::from_sub(node_field(&sm, &[0.0, 0.0]));
        let f1 = NodeField::from_sub(node_field(&sm, &[10.0, 20.0]));
        let e = Evolution::from_node_fields(&[(0.0, &f0), (1.0, &f1)], OutOfRange::Error).unwrap();
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

    // ─── Types & field-through-curve interpolation ───────────────────────────

    /// A scalar curve `T → E` (double the abscissa), typed on both axes.
    fn typed_curve() -> SubEvolution {
        scalar_curve(&[(0.0, 0.0), (10.0, 20.0)], OutOfRange::Error)
            .with_abscissa_type(Some("T".into()))
            .with_ordinate_type(Some("E".into()))
            .unwrap()
    }

    #[test]
    fn ordinate_type_rejected_on_field_curve() {
        // A node-valued curve has no ordinate to type.
        let mut se = SubEvolution::new(
            vec![(0.0, SubValue::Node(node_field(&poi1(1), &[1.0])))],
            OutOfRange::Error,
        )
        .unwrap();
        assert!(se.set_ordinate_type(Some("E".into())).is_err());
        // Clearing it (None) is always fine.
        assert!(se.set_ordinate_type(None).is_ok());
    }

    #[test]
    fn interpolate_field_maps_component_pointwise() {
        let se = typed_curve();
        // Input field carries the abscissa-typed component "T".
        let sm = poi1(3);
        let input = node_field(&sm, &[0.0, 5.0, 10.0]);
        let out = se.interpolate_field(&input, None).unwrap();
        // Output is single-component, named after the ordinate type.
        assert_eq!(out.components(), &["E".to_string()]);
        assert_eq!(out.get(0, 0).unwrap(), 0.0); // 2·0
        assert_eq!(out.get(1, 0).unwrap(), 10.0); // 2·5
        assert_eq!(out.get(2, 0).unwrap(), 20.0); // 2·10
    }

    #[test]
    fn interpolate_field_checks_type_correspondence() {
        // Abscissa type "P" has no counterpart in a field whose component is "T".
        let se = scalar_curve(&[(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error)
            .with_abscissa_type(Some("P".into()));
        let input = node_field(&poi1(1), &[0.5]);
        assert!(se.interpolate_field(&input, None).is_err());
    }

    #[test]
    fn interpolate_field_requires_abscissa_type() {
        // No abscissa type set → cannot pick a field component.
        let se = scalar_curve(&[(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error);
        let input = node_field(&poi1(1), &[0.5]);
        assert!(se.interpolate_field(&input, None).is_err());
    }

    #[test]
    fn interpolate_field_out_of_range_honours_policy() {
        let se = typed_curve(); // range [0, 10], Error
        let input = node_field(&poi1(1), &[15.0]); // beyond the range
        assert!(se.interpolate_field(&input, None).is_err());
        // Clamp → the endpoint value (20).
        assert_eq!(
            se.interpolate_field(&input, Some(OutOfRange::Clamp))
                .unwrap()
                .get(0, 0)
                .unwrap(),
            20.0
        );
    }

    #[test]
    fn aggregate_interpolate_node_field_end_to_end() {
        // A single-curve typed scalar Evolution used as a transfer function.
        let mut e =
            Evolution::from_scalars(vec![(0.0, 0.0), (10.0, 20.0)], OutOfRange::Error).unwrap();
        e.set_abscissa_type(Some("T".into())).unwrap();
        e.set_ordinate_type(Some("E".into())).unwrap();
        let sm = poi1(2);
        let input = NodeField::from_sub(node_field(&sm, &[0.0, 10.0]));
        let out = e.interpolate_node_field(&input, None).unwrap();
        let sub = read(&out.get(0).unwrap()).unwrap();
        assert_eq!(sub.components(), &["E".to_string()]);
        assert_eq!(sub.get(0, 0).unwrap(), 0.0);
        assert_eq!(sub.get(1, 0).unwrap(), 20.0);
    }

    #[test]
    fn aggregate_field_map_requires_single_scalar_curve() {
        // Two curves → ambiguous, rejected.
        let e0 = Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        let e1 = Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        let mut two = e0.union(&e1).unwrap();
        two.set_abscissa_type(Some("T".into())).unwrap();
        let input = NodeField::from_sub(node_field(&poi1(1), &[0.5]));
        assert!(two.interpolate_node_field(&input, None).is_err());
    }

    #[test]
    fn types_round_trip_on_aggregate() {
        let mut e =
            Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        e.set_abscissa_type(Some("time".into())).unwrap();
        e.set_ordinate_type(Some("force".into())).unwrap();
        assert_eq!(e.abscissa_type().unwrap().as_deref(), Some("time"));
        assert_eq!(e.ordinate_type().unwrap().as_deref(), Some("force"));
    }
}
