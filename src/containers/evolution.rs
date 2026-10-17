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
//! of [`crate::containers::field`] (`map_all` + `merge_components`, guarded by
//! `check_same_components`), so no numerics are duplicated here.
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
//! use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue};
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
use crate::handle::Handle;
use crate::parallel::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// One labelled `(abscissa, value)` curve per zone — the input of a scalar
/// X-Y plot ([`Evolution::scalar_series_set`]).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// # use pyrucast::containers::evolution::ScalarSeriesSet;
/// // What an X-Y plot consumes: one labelled series per zone.
/// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
/// let series: ScalarSeriesSet = e.scalar_series_set()?;
/// assert_eq!(series.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type ScalarSeriesSet = Vec<(String, Vec<(f64, f64)>)>;

// ─── OutOfRange policy ──────────────────────────────────────────────────────

/// What an interpolation does when the requested abscissa falls **outside**
/// the tabulated `[x_min, x_max]` range.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// # let c = SubEvolution::new(
/// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
/// #     OutOfRange::Error).unwrap();
/// // Three possible answers to "and beyond the last tabulated point?"
/// assert!(c.eval_scalar(2.0, Some(OutOfRange::Error)).is_err());
/// assert_eq!(c.eval_scalar(2.0, Some(OutOfRange::Clamp))?, 120.0);
/// assert_eq!(c.eval_scalar(2.0, Some(OutOfRange::Extrapolate))?, 220.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// The policy's canonical name.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
    /// assert_eq!(OutOfRange::Extrapolate.name(), "extrapolate");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            OutOfRange::Error => "error",
            OutOfRange::Clamp => "clamp",
            OutOfRange::Extrapolate => "extrapolate",
        }
    }
}

impl crate::named::Named for OutOfRange {
    const LABEL: &'static str = "out_of_range policy";
    const VALUES: &'static [Self] = &[Self::Error, Self::Clamp, Self::Extrapolate];

    fn name(self) -> &'static str {
        OutOfRange::name(self)
    }
}

// ─── SubValue ───────────────────────────────────────────────────────────────

/// The kind of value carried by a [`SubValue`] — used for homogeneity checks
/// and messages.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// # let c = SubEvolution::new(vec![(0.0, SubValue::Scalar(20.0))], OutOfRange::Clamp).unwrap();
/// // A curve is homogeneous: scalars, nodal fields or element fields, never
/// // a mixture.
/// assert_eq!(c.kind(), ValueKind::Scalar);
/// ```
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// # let c = SubEvolution::new(
/// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
/// #     OutOfRange::Clamp).unwrap();
/// // A tabulated value is read by variant matching — or, for the scalar
/// // case, through the `eval_scalar` shortcut.
/// match c.value_at(0)? {
///     SubValue::Scalar(v) => assert_eq!(v, 20.0),
///     SubValue::Node(_) | SubValue::Element(_) => unreachable!(),
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
    /// assert_eq!(SubValue::Scalar(1.0).kind(), ValueKind::Scalar);
    /// ```
    pub fn kind(&self) -> ValueKind {
        match self {
            SubValue::Scalar(_) => ValueKind::Scalar,
            SubValue::Node(_) => ValueKind::Node,
            SubValue::Element(_) => ValueKind::Element,
        }
    }
}

/// An abscissa: a number for a curve, a tuple of numbers for a table of
/// several variables — what the constructors take, so that one of each serves
/// both.
///
/// ```
/// # use pyrucast::containers::evolution::Abscissa;
/// assert_eq!(2.0.coordinates(), vec![2.0]);
/// assert_eq!([20.0, 0.5].coordinates(), vec![20.0, 0.5]);
/// ```
pub trait Abscissa {
    /// The coordinates of this abscissa, one per variable.
    fn coordinates(self) -> Vec<f64>;
}

impl Abscissa for f64 {
    fn coordinates(self) -> Vec<f64> {
        vec![self]
    }
}

impl Abscissa for Vec<f64> {
    fn coordinates(self) -> Vec<f64> {
        self
    }
}

impl<const N: usize> Abscissa for [f64; N] {
    fn coordinates(self) -> Vec<f64> {
        self.to_vec()
    }
}

/// The most variables a table may tabulate along. Bounds the samples one
/// query weighs in (`2^MAX_ABSCISSAS`), which is what lets the weights live on
/// the stack at every Gauss point.
///
/// ```
/// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, MAX_ABSCISSAS};
/// // One variable too many is refused at build time.
/// let trop = vec![0.0; MAX_ABSCISSAS + 1];
/// assert!(SubEvolution::new(vec![(trop, SubValue::Scalar(1.0))], OutOfRange::Error).is_err());
/// ```
pub const MAX_ABSCISSAS: usize = 4;

/// The samples one query weighs in, with their weights — at most two per
/// variable, so at most `2^MAX_ABSCISSAS`, on the stack.
struct Weights {
    index: [usize; 1 << MAX_ABSCISSAS],
    weight: [f64; 1 << MAX_ABSCISSAS],
    len: usize,
}

impl Weights {
    fn new() -> Self {
        Self {
            index: [0; 1 << MAX_ABSCISSAS],
            weight: [0.0; 1 << MAX_ABSCISSAS],
            len: 0,
        }
    }

    fn push(&mut self, index: usize, weight: f64) {
        self.index[self.len] = index;
        self.weight[self.len] = weight;
        self.len += 1;
    }

    fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.index[..self.len]
            .iter()
            .copied()
            .zip(self.weight[..self.len].iter().copied())
    }
}

/// `Σ wᵢ·vᵢ` over the weighed samples, all of one kind. For fields the
/// per-value arithmetic of [`SubField`] (`map_all` + `merge_components`) is
/// reused; the samples were checked at construction to share support and
/// components.
fn blend(values: &[SubValue], weights: &Weights) -> Result<SubValue> {
    let mut acc: Option<SubValue> = None;
    for (i, w) in weights.iter() {
        let term = match &values[i] {
            SubValue::Scalar(v) => SubValue::Scalar(v * w),
            SubValue::Node(f) => SubValue::Node(f.map_all(|v| v * w)),
            SubValue::Element(f) => SubValue::Element(f.map_all(|v| v * w)),
        };
        acc = Some(match (acc, term) {
            (None, t) => t,
            (Some(SubValue::Scalar(a)), SubValue::Scalar(b)) => SubValue::Scalar(a + b),
            // Two samples of one field: a mismatched component set is a
            // genuine error, caught before the merge would union it.
            (Some(SubValue::Node(a)), SubValue::Node(b)) => {
                a.check_same_components(&b)?;
                SubValue::Node(a.merge_components(&b, |x, y| x + y)?)
            }
            (Some(SubValue::Element(a)), SubValue::Element(b)) => {
                a.check_same_components(&b)?;
                SubValue::Element(a.merge_components(&b, |x, y| x + y)?)
            }
            _ => {
                return Err(PyrucastError::Message(
                    "evolution: cannot interpolate between values of different kinds".into(),
                ))
            }
        });
    }
    Ok(acc.expect("a query weighs in at least one sample"))
}

// ─── SubEvolution ───────────────────────────────────────────────────────────

/// One tabulated function: abscissas and the matching values, all of the same
/// [`ValueKind`] (and, for fields, on the same support).
///
/// A tabulated map `f : ℝᵐ → ℝⁿ`: a **curve** when each abscissa is a number
/// (`m = 1`, sorted, strictly increasing), a **table** when it is a tuple of
/// `m` numbers. A table is read **hierarchically**: the samples are grouped by
/// their first coordinate, each group by the second, and so on, every
/// innermost group being a curve. The groups need not share their abscissas —
/// an irregular table, as material data usually comes — and a full grid gives
/// back the multilinear interpolation. The first variable is the outer level,
/// which matters on an irregular table: there, the order of the variables
/// changes the result.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// // A temperature rising from 20 to 120 in one time unit.
/// let c = SubEvolution::new(
///     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
///     OutOfRange::Clamp)?;
/// assert_eq!(c.len(), 2);
/// assert_eq!(c.eval_scalar(0.5, None)?, 70.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize, Clone)]
pub struct SubEvolution {
    /// The samples' abscissas, `dims` per sample, the samples in lexicographic
    /// order of their abscissa tuples — for a curve, its abscissas, strictly
    /// increasing.
    abscissas: Vec<f64>,
    /// How many variables each abscissa has: 1 for a curve, more for a table.
    dims: usize,
    /// Tabulated values, one per sample.
    values: Vec<SubValue>,
    /// Out-of-range policy used when this curve is interpolated directly.
    out_of_range: OutOfRange,
    /// Physical **type** of each abscissa (e.g. `["T"]`, `["T", "phi"]`),
    /// empty when unset. Labels plots and — when a field is mapped through a
    /// scalar curve — names the field components each variable is read from
    /// (see [`SubEvolution::interpolate_field`]).
    abscissa_type: Vec<String>,
    /// Physical **type** of the ordinate, for **scalar** curves only (e.g.
    /// `"young"`). Optional; labels plots and names the component produced when
    /// a field is mapped through the curve.
    #[serde(default)]
    ordinate_type: Option<String>,
}

impl SubEvolution {
    /// Build a curve or a table from `(abscissa, value)` samples — the abscissa
    /// a number for a curve, a tuple (`[f64; m]` or `Vec<f64>`) for a table of
    /// `m` variables. The samples are sorted; the values must be homogeneous in
    /// kind and, for fields, on the same support with the same component count.
    ///
    /// Errors:
    /// - no sample, a `NaN` abscissa, or the same abscissa twice;
    /// - abscissas of different lengths, or more than [`MAX_ABSCISSAS`] variables;
    /// - mixed value kinds, or field values on different supports.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
    /// // The samples are **sorted** at build time: giving them out of order is
    /// // legal, giving them twice at the same abscissa is not.
    /// let c = SubEvolution::new(
    ///     vec![(1.0, SubValue::Scalar(120.0)), (0.0, SubValue::Scalar(20.0))],
    ///     OutOfRange::Clamp)?;
    /// assert_eq!(c.abscissas(), &[0.0, 1.0]);
    /// assert!(SubEvolution::new(
    ///     vec![(0.0, SubValue::Scalar(1.0)), (0.0, SubValue::Scalar(2.0))],
    ///     OutOfRange::Error).is_err());
    /// // A table of two variables, tabulated irregularly: two values of the
    /// // second at the first's 20, three at its 500.
    /// let t = SubEvolution::new(
    ///     vec![([20.0, 0.0], SubValue::Scalar(300.0)), ([20.0, 1.0], SubValue::Scalar(450.0)),
    ///          ([500.0, 0.0], SubValue::Scalar(200.0)), ([500.0, 0.5], SubValue::Scalar(260.0)),
    ///          ([500.0, 1.0], SubValue::Scalar(320.0))],
    ///     OutOfRange::Error)?;
    /// assert_eq!((t.dims(), t.len()), (2, 5));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new<A: Abscissa>(samples: Vec<(A, SubValue)>, out_of_range: OutOfRange) -> Result<Self> {
        if samples.is_empty() {
            return Err(PyrucastError::Message(
                "SubEvolution requires at least one sample".into(),
            ));
        }
        let mut samples: Vec<(Vec<f64>, SubValue)> = samples
            .into_iter()
            .map(|(x, v)| (x.coordinates(), v))
            .collect();
        let dims = samples[0].0.len();
        if dims == 0 || dims > MAX_ABSCISSAS {
            return Err(PyrucastError::Message(format!(
                "SubEvolution: an abscissa has between 1 and {MAX_ABSCISSAS} variables, \
                 not {dims}"
            )));
        }
        for (x, _) in &samples {
            if x.len() != dims {
                return Err(PyrucastError::Message(format!(
                    "SubEvolution: every abscissa must have {dims} variable(s) like the first, \
                     one has {}",
                    x.len()
                )));
            }
            if x.iter().any(|c| c.is_nan()) {
                return Err(PyrucastError::Message("SubEvolution: NaN abscissa".into()));
            }
        }
        // Lexicographic order: what groups a table by its first variable, then
        // by the second within each group, and so on.
        samples.sort_by(|a, b| {
            a.0.iter()
                .zip(&b.0)
                .map(|(x, y)| x.partial_cmp(y).expect("non-NaN abscissas"))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for w in samples.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(PyrucastError::Message(format!(
                    "SubEvolution: duplicate abscissa {:?}",
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
        let (abscissas, values): (Vec<Vec<f64>>, Vec<SubValue>) = samples.into_iter().unzip();
        // For field values, every sample must share the first's support and
        // component count, so the blend is well-defined.
        check_same_support(&values)?;

        Ok(SubEvolution {
            abscissas: abscissas.concat(),
            dims,
            values,
            out_of_range,
            abscissa_type: Vec::new(),
            ordinate_type: None,
        })
    }

    /// The physical type of each abscissa variable — empty when unset, else
    /// one per variable.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let c = courbe.with_abscissa_types(vec!["T".into()])?;
    /// assert_eq!(c.abscissa_types(), ["T"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn abscissa_types(&self) -> &[String] {
        &self.abscissa_type
    }

    /// How many variables each abscissa has: 1 for a curve, more for a table.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.dims(), 1);
    /// ```
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The ordinate's physical type (scalar curves), if set.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let c = courbe.with_ordinate_type(Some("young".into()))?;
    /// assert_eq!(c.ordinate_type(), Some("young"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn ordinate_type(&self) -> Option<&str> {
        self.ordinate_type.as_deref()
    }

    /// Set (or clear, with an empty list) the physical type of each abscissa
    /// variable. Errors unless one type is given per variable.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let mut c = courbe;
    /// c.set_abscissa_types(vec!["time".into()])?;
    /// assert_eq!(c.abscissa_types(), ["time"]);
    /// c.set_abscissa_types(Vec::new())?; // et l'on peut le retirer
    /// assert!(c.abscissa_types().is_empty());
    /// // One type per variable: a curve has one.
    /// assert!(c.set_abscissa_types(vec!["T".into(), "phi".into()]).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_abscissa_types(&mut self, types: Vec<String>) -> Result<()> {
        if !types.is_empty() && types.len() != self.dims {
            return Err(PyrucastError::Message(format!(
                "abscissa types: {} given for an abscissa of {} variable(s) — one per variable",
                types.len(),
                self.dims
            )));
        }
        self.abscissa_type = types;
        Ok(())
    }

    /// Set (or clear) the ordinate's physical type. Errors if a type is given
    /// for a curve that carries fields rather than scalars — only a scalar
    /// curve has an ordinate to type.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let mut c = courbe;
    /// c.set_ordinate_type(Some("young".into()))?;
    /// assert_eq!(c.ordinate_type(), Some("young"));
    /// // An ordinate has a type only on a **scalar** curve: on a curve of
    /// // fields, setting it fails.
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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

    /// Builder form of [`SubEvolution::set_abscissa_types`].
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Chained form, to set the type at build time.
    /// let c = SubEvolution::new(
    ///     vec![(20.0, SubValue::Scalar(210_000.0)), (300.0, SubValue::Scalar(180_000.0))],
    ///     OutOfRange::Clamp)?
    ///     .with_abscissa_types(vec!["T".into()])?;
    /// assert_eq!(c.abscissa_types(), ["T"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_abscissa_types(mut self, types: Vec<String>) -> Result<Self> {
        self.set_abscissa_types(types)?;
        Ok(self)
    }

    /// Builder form of [`SubEvolution::set_ordinate_type`] (validated).
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // A Young's modulus that depends on temperature: the two types set
    /// // together make the curve a named transfer function.
    /// let c = courbe
    ///     .with_abscissa_types(vec!["T".into()])?
    ///     .with_ordinate_type(Some("young".into()))?;
    /// assert_eq!((c.abscissa_types(), c.ordinate_type()), (&["T".to_string()][..], Some("young")));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_ordinate_type(mut self, t: Option<String>) -> Result<Self> {
        self.set_ordinate_type(t)?;
        Ok(self)
    }

    /// Number of samples.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.len(), 2); // deux points tabulés
    /// ```
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the curve holds no sample (never true for a constructed one).
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert!(!courbe.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The abscissas, [`dims`](Self::dims) per sample in sample order — for a
    /// curve, its abscissas, sorted strictly increasing.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.abscissas(), &[0.0, 1.0]); // toujours croissantes
    /// ```
    pub fn abscissas(&self) -> &[f64] {
        &self.abscissas
    }

    /// This curve's value kind.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.kind(), ValueKind::Scalar);
    /// ```
    pub fn kind(&self) -> ValueKind {
        self.values[0].kind()
    }

    /// This curve's stored out-of-range policy.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.out_of_range(), OutOfRange::Clamp);
    /// ```
    pub fn out_of_range(&self) -> OutOfRange {
        self.out_of_range
    }

    /// The `(abscissa, value)` points of a **scalar** curve, in abscissa
    /// order. Errors if the curve carries fields rather than scalars.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// assert_eq!(courbe.scalar_series()?, vec![(0.0, 20.0), (1.0, 120.0)]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn scalar_series(&self) -> Result<Vec<(f64, f64)>> {
        if self.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "scalar_series: this evolution carries fields, not scalars".into(),
            ));
        }
        if self.dims != 1 {
            return Err(PyrucastError::Message(format!(
                "scalar_series: a table of {} variables is not an X-Y curve",
                self.dims
            )));
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
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // The **tabulated** value of index k — without interpolation.
    /// match courbe.value_at(1)? {
    ///     SubValue::Scalar(v) => assert_eq!(v, 120.0),
    ///     _ => unreachable!(),
    /// }
    /// assert!(courbe.value_at(2).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
        view: crate::viz::View,
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
            subs: vec![Handle::new(self.clone())],
            out_of_range: self.out_of_range,
        };
        evo.plot(
            view, save, mesh, component, scale, smooth, frame, x_label, y_label, title,
        )
    }

    /// Interpolate a **curve** at `x`. `policy` overrides the stored
    /// [`OutOfRange`] for this single query when `Some`. Errors on a table:
    /// it takes a tuple, see [`interpolate_at`](Self::interpolate_at).
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Linear interpolation between two tabulated points…
    /// match courbe.interpolate(0.25, None)? {
    ///     SubValue::Scalar(v) => assert_eq!(v, 45.0),
    ///     _ => unreachable!(),
    /// }
    /// // …and out of range, the curve's policy applies — unless the call
    /// // imposes another one.
    /// assert!(courbe.interpolate(2.0, Some(OutOfRange::Error)).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate(&self, x: f64, policy: Option<OutOfRange>) -> Result<SubValue> {
        self.interpolate_at(&[x], policy)
    }

    /// Interpolate at the abscissa `x`, one number per variable.
    ///
    /// A table is read level by level: the two groups of the first variable
    /// that bracket `x[0]` are found and weighed linearly, each is read the same
    /// way along the second variable, and so on down to the samples. Out of a
    /// level's range, the policy applies (`Error`, `Clamp` to the nearest group,
    /// `Extrapolate` along the two nearest); a group whose next variable takes
    /// a **single** value is constant along it — a property measured at one
    /// value only there does not vary with it.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue};
    /// let sy = SubEvolution::new(
    ///     vec![([20.0, 0.0], SubValue::Scalar(300.0)), ([20.0, 1.0], SubValue::Scalar(450.0)),
    ///          ([500.0, 0.0], SubValue::Scalar(200.0)), ([500.0, 0.5], SubValue::Scalar(260.0)),
    ///          ([500.0, 1.0], SubValue::Scalar(320.0))],
    ///     OutOfRange::Error)?;
    /// // At T = 20 the curve in φ gives 360, at T = 500 its own curve gives
    /// // 248; between them, linearly in T.
    /// let attendu = 360.0 + (248.0 - 360.0) * (300.0 - 20.0) / 480.0;
    /// assert!((sy.eval_scalar_at(&[300.0, 0.4], None)? - attendu).abs() < 1e-9);
    /// // A tuple of the wrong length is refused.
    /// assert!(sy.interpolate_at(&[300.0], None).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate_at(&self, x: &[f64], policy: Option<OutOfRange>) -> Result<SubValue> {
        self.check_arity(x)?;
        let weights = self.weights(x, policy.unwrap_or(self.out_of_range))?;
        blend(&self.values, &weights)
    }

    /// Refuse an abscissa with the wrong number of variables.
    fn check_arity(&self, x: &[f64]) -> Result<()> {
        if x.len() != self.dims {
            return Err(PyrucastError::Message(format!(
                "evolution: this {} takes {} variable(s), {} given",
                if self.dims == 1 { "curve" } else { "table" },
                self.dims,
                x.len()
            )));
        }
        Ok(())
    }

    /// The samples `x` weighs in, and their weights.
    fn weights(&self, x: &[f64], policy: OutOfRange) -> Result<Weights> {
        let mut out = Weights::new();
        self.gather(0, 0, self.values.len(), x, policy, 1.0, &mut out)?;
        Ok(out)
    }

    /// The abscissa of sample `i` along variable `level`.
    fn coordinate(&self, i: usize, level: usize) -> f64 {
        self.abscissas[i * self.dims + level]
    }

    /// First sample of `[lo, hi)` whose coordinate along `level` fails `keep`
    /// — the samples of a group are sorted along the next variable, so a
    /// binary search finds it.
    fn partition(&self, level: usize, lo: usize, hi: usize, keep: impl Fn(f64) -> bool) -> usize {
        let (mut lo, mut hi) = (lo, hi);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if keep(self.coordinate(mid, level)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Weigh the samples `[s, e)` — one group, all sharing their first `level`
    /// coordinates — at `x`, along variable `level` and the ones after it,
    /// scaling every weight by `w`.
    #[allow(clippy::too_many_arguments)]
    fn gather(
        &self,
        level: usize,
        s: usize,
        e: usize,
        x: &[f64],
        policy: OutOfRange,
        w: f64,
        out: &mut Weights,
    ) -> Result<()> {
        let at = |i: usize| self.coordinate(i, level);
        // The group of samples sharing sample `i`'s value along `level`.
        let group_end = |i: usize| self.partition(level, i, e, |v| v <= at(i));
        let group_start = |i: usize| self.partition(level, s, i, |v| v < at(i));
        let first = (s, group_end(s));
        let last = (group_start(e - 1), e);
        let xl = x[level];
        // A single value along this variable: constant along it, past the first
        // variable (whose range is the table's own).
        if first.1 == e && level > 0 {
            return self.descend(level, first, x, policy, w, out);
        }
        let k = self.partition(level, s, e, |v| v < xl);
        // The two groups to weigh, and the second's weight `t`.
        let (lo, hi, t) = if k < e && at(k) == xl {
            return self.descend(level, (k, group_end(k)), x, policy, w, out);
        } else if k == s || k == e {
            let below = k == s;
            match policy {
                OutOfRange::Error => {
                    return Err(PyrucastError::Message(format!(
                        "evolution: abscissa {xl}{} is outside the tabulated range [{}, {}]",
                        match self.abscissa_type.get(level) {
                            Some(name) if self.dims > 1 => format!(" ({name})"),
                            _ => String::new(),
                        },
                        at(s),
                        at(e - 1)
                    )))
                }
                OutOfRange::Clamp => {
                    let g = if below { first } else { last };
                    return self.descend(level, g, x, policy, w, out);
                }
                OutOfRange::Extrapolate => {
                    if first.1 == e {
                        return self.descend(level, first, x, policy, w, out);
                    }
                    let (a, b) = if below {
                        (first, (first.1, group_end(first.1)))
                    } else {
                        ((group_start(last.0 - 1), last.0), last)
                    };
                    (a, b, (xl - at(a.0)) / (at(b.0) - at(a.0)))
                }
            }
        } else {
            let lo = (group_start(k - 1), k);
            let hi = (k, group_end(k));
            (lo, hi, (xl - at(lo.0)) / (at(hi.0) - at(lo.0)))
        };
        self.descend(level, lo, x, policy, w * (1.0 - t), out)?;
        self.descend(level, hi, x, policy, w * t, out)
    }

    /// Carry on with the group `g` along the next variable, or weigh its
    /// single sample when `level` was the last.
    fn descend(
        &self,
        level: usize,
        g: (usize, usize),
        x: &[f64],
        policy: OutOfRange,
        w: f64,
        out: &mut Weights,
    ) -> Result<()> {
        if level + 1 == self.dims {
            out.push(g.0, w);
            Ok(())
        } else {
            self.gather(level + 1, g.0, g.1, x, policy, w, out)
        }
    }

    /// Interpolate a **scalar** curve at `x`, returning the value directly.
    /// Errors if this curve carries fields rather than scalars.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // The shortcut without variant matching, for a scalar curve.
    /// assert_eq!(courbe.eval_scalar(0.5, None)?, 70.0);
    /// assert_eq!(courbe.eval_scalar(9.0, None)?, 120.0); // Clamp
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn eval_scalar(&self, x: f64, policy: Option<OutOfRange>) -> Result<f64> {
        self.eval_scalar_at(&[x], policy)
    }

    /// Interpolate a **scalar** curve or table at the abscissa `x`, returning
    /// the value directly — see [`interpolate_at`](Self::interpolate_at). Errors
    /// if it carries fields rather than scalars.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue};
    /// // A full grid: the hierarchical reading is the bilinear interpolation.
    /// let f = SubEvolution::new(
    ///     vec![([0.0, 0.0], SubValue::Scalar(1.0)), ([0.0, 1.0], SubValue::Scalar(2.0)),
    ///          ([1.0, 0.0], SubValue::Scalar(3.0)), ([1.0, 1.0], SubValue::Scalar(5.0))],
    ///     OutOfRange::Error)?;
    /// assert!((f.eval_scalar_at(&[0.5, 0.5], None)? - 2.75).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn eval_scalar_at(&self, x: &[f64], policy: Option<OutOfRange>) -> Result<f64> {
        if self.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "evolution: a field can only be mapped through a scalar-valued evolution".into(),
            ));
        }
        self.check_arity(x)?;
        self.scalar_at(x, policy.unwrap_or(self.out_of_range))
    }

    /// The scalar value at `x` — kind and arity already checked by the caller,
    /// once, rather than at every point.
    fn scalar_at(&self, x: &[f64], policy: OutOfRange) -> Result<f64> {
        let weights = self.weights(x, policy)?;
        Ok(weights
            .iter()
            .map(|(i, w)| match self.values[i] {
                SubValue::Scalar(v) => v * w,
                _ => unreachable!("a scalar curve, checked by the caller"),
            })
            .sum())
    }

    /// Map `field` through this **scalar** curve or table, node by node (or
    /// Gauss point by Gauss point).
    ///
    /// The function is applied like `cos` to a field: its variables are read
    /// from the field components **named like the abscissa types**; the result
    /// is a fresh single-component field on the **same support**, its component
    /// named after the ordinate type (or `"value"` when none is set). See
    /// [`Evolution::interpolate_element_field`] to apply several at once.
    ///
    /// Errors if the curve is not scalar-valued, has no abscissa type set, or
    /// the field has no component matching one of them (the requested
    /// **type correspondence**).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let temp = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # temp.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// // A conductivity that depends on temperature: the curve is a transfer
    /// // function, and it is the **types** that hook it onto the field — the
    /// // abscissa names the component read, the ordinate the one produced.
    /// let loi = SubEvolution::new(
    ///     vec![(20.0, SubValue::Scalar(50.0)), (120.0, SubValue::Scalar(30.0))],
    ///     OutOfRange::Clamp)?
    ///     .with_abscissa_types(vec!["T".into()])?
    ///     .with_ordinate_type(Some("k".into()))?;
    ///
    /// let k = loi.interpolate_field(&*temp.get(0)?.read(), None)?;
    /// assert_eq!(k.components(), &["k".to_string()]);
    /// assert_eq!(k.value(n[0].id(), "k")?, 30.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate_field<F>(&self, field: &F, policy: Option<OutOfRange>) -> Result<F>
    where
        F: SubField + Clone,
    {
        map_through(std::slice::from_ref(self), field, policy)
    }
}

/// Map `field` through every scalar curve of `curves`, one output component per
/// curve, named after its ordinate type (`"value"` when unset).
///
/// What each curve reads — its kind, its variables' components in the field,
/// its out-of-range policy — is settled **once**, before the rows are walked;
/// a row then gathers its abscissas on the stack and weighs the samples without
/// allocating. `policy`, when given, overrides every curve's own.
fn map_through<F>(curves: &[SubEvolution], field: &F, policy: Option<OutOfRange>) -> Result<F>
where
    F: SubField + Clone,
{
    let mut names: Vec<String> = Vec::with_capacity(curves.len());
    let mut reads: Vec<(Vec<usize>, OutOfRange)> = Vec::with_capacity(curves.len());
    for curve in curves {
        if curve.kind() != ValueKind::Scalar {
            return Err(PyrucastError::Message(
                "evolution: a field can only be mapped through a scalar-valued evolution".into(),
            ));
        }
        if curve.abscissa_type.is_empty() {
            return Err(PyrucastError::Message(
                "evolution: set the abscissa type to map a field \
                 (it selects which field component to look up)"
                    .into(),
            ));
        }
        let mut idx = Vec::with_capacity(curve.dims);
        for name in &curve.abscissa_type {
            idx.push(field.component_index(name).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "evolution: the field has no component '{name}' matching the abscissa type"
                ))
            })?);
        }
        let name = curve
            .ordinate_type
            .clone()
            .unwrap_or_else(|| "value".to_string());
        if names.contains(&name) {
            return Err(PyrucastError::Message(format!(
                "evolution: two curves produce the component '{name}' — give each its own \
                 ordinate type"
            )));
        }
        names.push(name);
        reads.push((idx, policy.unwrap_or(curve.out_of_range)));
    }
    let nc = field.component_count();
    let k = curves.len();
    let mut out = field.same_support_with(names)?;
    // Each output row written once, from its own input row ⇒ the result does
    // not depend on the thread count; the first out-of-range point aborts.
    field
        .values()
        .par_chunks(nc)
        .zip(out.values_mut().par_chunks_mut(k))
        .with_min_len((MIN_PARALLEL_LEN / nc.max(1)).max(1))
        .try_for_each(|(row, slot)| -> Result<()> {
            let mut x = [0.0_f64; MAX_ABSCISSAS];
            for (j, (curve, (idx, policy))) in curves.iter().zip(&reads).enumerate() {
                for (v, &i) in idx.iter().enumerate() {
                    x[v] = row[i];
                }
                slot[j] = curve.scalar_at(&x[..idx.len()], *policy)?;
            }
            Ok(())
        })?;
    Ok(out)
}

/// Verify that all field samples share the first sample's support and
/// component count (scalars are unconstrained).
fn check_same_support(samples: &[SubValue]) -> Result<()> {
    match &samples[0] {
        SubValue::Scalar(_) => Ok(()),
        SubValue::Node(f0) => {
            for v in &samples[1..] {
                if let SubValue::Node(f) = v
                    && (!f.same_support(f0) || f.component_count() != f0.component_count())
                {
                    return Err(PyrucastError::Message(
                        "SubEvolution: node values must share the same support and components"
                            .into(),
                    ));
                }
            }
            Ok(())
        }
        SubValue::Element(f0) => {
            for v in &samples[1..] {
                if let SubValue::Element(f) = v
                    && (!f.same_support(f0) || f.component_count() != f0.component_count())
                {
                    return Err(PyrucastError::Message(
                        "SubEvolution: element values must share the same support and components"
                            .into(),
                    ));
                }
            }
            Ok(())
        }
    }
}

impl fmt::Debug for SubEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubEvolution")
            .field("samples", &self.values.len())
            .field("dims", &self.dims)
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
            self.values.len(),
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
            .chunks(self.dims)
            .zip(&self.values)
            .map(|(x, v)| {
                let cell = match v {
                    SubValue::Scalar(s) => fmt_float(*s, opts.precision),
                    SubValue::Node(f) => {
                        format!("SubNodeField({} node(s))", f.node_count())
                    }
                    SubValue::Element(f) => format!("SubElementField({} cell(s))", f.cell_count()),
                };
                let abscissa = match x {
                    [one] => fmt_float(*one, opts.precision),
                    tuple => format!(
                        "({})",
                        tuple
                            .iter()
                            .map(|c| fmt_float(*c, opts.precision))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                };
                vec![abscissa, cell]
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
/// assert_eq!(e.len(), 1); // one curve per zone; here a single one
/// match e.interpolate(0.5, None)? {
///     Interpolated::Scalars(v) => assert_eq!(v, vec![70.0]),
///     _ => unreachable!(),
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
            let k0 = self.subs[0].read().kind();
            let k = h.read().kind();
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// // Interpolating an aggregate returns an aggregate — except for scalars,
/// // of which no aggregate exists: a `Vec<f64>`, one value per zone.
/// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
/// match e.interpolate(0.25, None)? {
///     Interpolated::Scalars(v) => assert_eq!(v, vec![45.0]),
///     Interpolated::Node(_) | Interpolated::Element(_) => unreachable!(),
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.out_of_range(), OutOfRange::Clamp);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn out_of_range(&self) -> OutOfRange {
        self.out_of_range
    }

    /// Interpolate every sub-evolution at `x` and regroup the results.
    ///
    /// `policy` overrides the aggregate's [`OutOfRange`] for this query when
    /// `Some`; the effective policy is applied to every curve.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// // A load at mid-course: the whole field, zone by zone.
    /// match e.interpolate(0.5, None)? {
    ///     Interpolated::Node(f) => assert_eq!(f.get(0)?.read().value(n[0].id(), "T")?, 70.0),
    ///     _ => unreachable!(),
    /// }
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate(&self, x: f64, policy: Option<OutOfRange>) -> Result<Interpolated> {
        self.interpolate_at(&[x], policy)
    }

    /// Interpolate every sub-evolution at the abscissa `x` — one number per
    /// variable — and regroup the results; [`interpolate`](Self::interpolate)
    /// is its one-variable form.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// let e = Evolution::from_scalars(
    ///     vec![([0.0, 0.0], 1.0), ([0.0, 1.0], 2.0), ([1.0, 0.0], 3.0), ([1.0, 1.0], 5.0)],
    ///     OutOfRange::Error)?;
    /// match e.interpolate_at(&[0.5, 0.5], None)? {
    ///     Interpolated::Scalars(v) => assert!((v[0] - 2.75).abs() < 1e-12),
    ///     _ => unreachable!(),
    /// }
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate_at(&self, x: &[f64], policy: Option<OutOfRange>) -> Result<Interpolated> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message(
                "Evolution: cannot interpolate an empty evolution".into(),
            ));
        }
        let policy = Some(policy.unwrap_or(self.out_of_range));
        match self.subs[0].read().kind() {
            ValueKind::Scalar => {
                let mut out = Vec::with_capacity(self.subs.len());
                for h in &self.subs {
                    match h.read().interpolate_at(x, policy)? {
                        SubValue::Scalar(v) => out.push(v),
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Scalars(out))
            }
            ValueKind::Node => {
                let mut field = NodeField::default();
                for h in &self.subs {
                    match h.read().interpolate_at(x, policy)? {
                        SubValue::Node(sf) => field.add_sub(Handle::new(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Node(field))
            }
            ValueKind::Element => {
                let mut field = ElementField::default();
                for h in &self.subs {
                    match h.read().interpolate_at(x, policy)? {
                        SubValue::Element(sf) => field.add_sub(Handle::new(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Element(field))
            }
        }
    }

    /// The abscissa types of the **first** sub-evolution — one per variable,
    /// empty when unset or for an empty evolution. The curves of one evolution
    /// may read different variables (`E(T)` beside `σ_y(T, φ)`): each keeps its
    /// own, see [`SubEvolution::abscissa_types`].
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let mut e = Evolution::from_scalars(vec![(0.0, 1.0), (1.0, 2.0)], OutOfRange::Clamp)?;
    /// assert!(e.abscissa_types().is_empty());
    /// e.set_abscissa_types(vec!["time".into()])?;
    /// assert_eq!(e.abscissa_types(), ["time"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn abscissa_types(&self) -> Vec<String> {
        match self.subs.first() {
            Some(h) => h.read().abscissa_types().to_vec(),
            None => Vec::new(),
        }
    }

    /// The ordinate's physical type (scalar evolutions), if set. `None` for an
    /// empty evolution.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let mut e = Evolution::from_scalars(vec![(20.0, 210e3), (300.0, 180e3)], OutOfRange::Clamp)?;
    /// e.set_ordinate_type(Some("young".into()))?;
    /// assert_eq!(e.ordinate_type()?, Some("young".into()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn ordinate_type(&self) -> Result<Option<String>> {
        match self.subs.first() {
            Some(h) => Ok(h.read().ordinate_type().map(str::to_string)),
            None => Ok(None),
        }
    }

    /// Set (or clear, with an empty list) the abscissa types on **every**
    /// sub-evolution — one per variable. Errors if a curve has another number
    /// of variables.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// // The type is set on **all** the zones at once.
    /// let mut e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// e.set_abscissa_types(vec!["time".into()])?;
    /// assert_eq!(e.abscissa_types(), ["time"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_abscissa_types(&mut self, types: Vec<String>) -> Result<()> {
        for h in &self.subs {
            h.write().set_abscissa_types(types.clone())?;
        }
        Ok(())
    }

    /// Set (or clear) the ordinate type on **every** sub-evolution. Errors on a
    /// field-valued evolution (only scalar curves have an ordinate to type).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let mut e = Evolution::from_scalars(vec![(0.0, 1.0), (1.0, 2.0)], OutOfRange::Clamp)?;
    /// e.set_ordinate_type(Some("young".into()))?;
    /// // On an evolution of fields, the ordinate has no type to carry.
    /// let mut champs = Evolution::from_node_fields(
    ///     &[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert!(champs.set_ordinate_type(Some("young".into())).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_ordinate_type(&mut self, t: Option<String>) -> Result<()> {
        for h in &self.subs {
            h.write().set_ordinate_type(t.clone())?;
        }
        Ok(())
    }

    /// The curves of a scalar-valued evolution, cloned — what maps a field
    /// ([`Evolution::interpolate_node_field`] /
    /// [`Evolution::interpolate_element_field`]). Errors on an empty or a
    /// field-valued evolution.
    fn scalar_curves(&self) -> Result<Vec<SubEvolution>> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message(
                "evolution: an empty evolution maps no field".into(),
            ));
        }
        self.subs
            .iter()
            .map(|h| {
                let sub = h.read();
                if sub.kind() != ValueKind::Scalar {
                    return Err(PyrucastError::Message(
                        "evolution: a field can only be mapped through a scalar-valued evolution"
                            .into(),
                    ));
                }
                Ok((*sub).clone())
            })
            .collect()
    }

    /// Map a whole [`NodeField`] through every **scalar** curve of this
    /// evolution, like `cos` through a field: each curve reads its variables
    /// from the components named like its abscissa types and writes one
    /// component, named after its ordinate type — see
    /// [`SubEvolution::interpolate_field`]. Same decomposition as the input.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut temp = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # temp.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// // The curve as a **transfer function**: it is the types that hook it
    /// // onto the field — the abscissa names the component read, the ordinate
    /// // the one produced.
    /// let mut loi = Evolution::from_scalars(
    ///     vec![(20.0, 50.0), (120.0, 30.0)], OutOfRange::Clamp)?;
    /// loi.set_abscissa_types(vec!["T".into()])?;
    /// loi.set_ordinate_type(Some("k".into()))?;
    /// let k = loi.interpolate_node_field(&temp, None)?;
    /// assert_eq!(k.get(0)?.read().value(n[0].id(), "k")?, 30.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate_node_field(
        &self,
        field: &NodeField,
        policy: Option<OutOfRange>,
    ) -> Result<NodeField> {
        let curves = self.scalar_curves()?;
        let policy = Some(policy.unwrap_or(self.out_of_range));
        field.map_subs(move |s| map_through(&curves, s, policy))
    }

    /// Map a whole [`ElementField`] through every **scalar** curve of this
    /// evolution — see [`Evolution::interpolate_node_field`]. Several curves give
    /// several components: material laws `E(T)`, `σ_y(T, φ)` applied at once to
    /// the state at the Gauss points.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::ElementField;
    /// # use pyrucast::containers::evolution::{Evolution, OutOfRange, SubEvolution, SubValue};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let etat = ElementField::new(&fes, vec!["T".into(), "phi".into()])?;
    /// etat.get(0)?.write().set_uniform("T", 260.0)?;
    /// etat.get(0)?.write().set_uniform("phi", 0.5)?;
    /// // E depends on T alone, σ_y on (T, φ): two curves, two components.
    /// let e_t = SubEvolution::new(vec![(20.0, SubValue::Scalar(210.0)), (500.0, SubValue::Scalar(170.0))],
    ///     OutOfRange::Error)?.with_abscissa_types(vec!["T".into()])?.with_ordinate_type(Some("E".into()))?;
    /// let sy = SubEvolution::new(
    ///     vec![([20.0, 0.0], SubValue::Scalar(300.0)), ([20.0, 1.0], SubValue::Scalar(400.0)),
    ///          ([500.0, 0.0], SubValue::Scalar(200.0)), ([500.0, 1.0], SubValue::Scalar(300.0))],
    ///     OutOfRange::Error)?.with_abscissa_types(vec!["T".into(), "phi".into()])?
    ///     .with_ordinate_type(Some("sigma_y".into()))?;
    /// let mut lois = Evolution::default();
    /// lois.add_sub(Handle::new(e_t))?;
    /// lois.add_sub(Handle::new(sy))?;
    /// let mat = lois.interpolate_element_field(&etat, None)?;
    /// let z = mat.get(0)?;
    /// assert!((z.read().value(0, 0, "E")? - 190.0).abs() < 1e-9);
    /// assert!((z.read().value(0, 0, "sigma_y")? - 300.0).abs() < 1e-9);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate_element_field(
        &self,
        field: &ElementField,
        policy: Option<OutOfRange>,
    ) -> Result<ElementField> {
        let curves = self.scalar_curves()?;
        let policy = Some(policy.unwrap_or(self.out_of_range));
        field.map_subs(move |s| map_through(&curves, s, policy))
    }

    /// The aggregate's value kind (taken from the first sub-evolution).
    /// Errors if the evolution is empty.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// # use pyrucast::containers::evolution::ValueKind;
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.kind()?, ValueKind::Node);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn kind(&self) -> Result<ValueKind> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message("Evolution: empty evolution".into()));
        }
        Ok(self.subs[0].read().kind())
    }

    /// The abscissa grid **shared** by every sub-evolution, validated to be
    /// identical across zones (a global frame slider requires it). Errors on
    /// an empty evolution or mismatched grids.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// // A grid common to every zone — what a frame slider requires.
    /// assert_eq!(e.shared_abscissas()?, vec![0.0, 1.0]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn shared_abscissas(&self) -> Result<Vec<f64>> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message("Evolution: empty evolution".into()));
        }
        if self.subs.iter().any(|h| h.read().dims() != 1) {
            return Err(PyrucastError::Message(
                "Evolution: a table of several variables has no frame sequence".into(),
            ));
        }
        let first = self.subs[0].read().abscissas().to_vec();
        for h in &self.subs[1..] {
            if h.read().abscissas() != first.as_slice() {
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.frame_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn frame_count(&self) -> Result<usize> {
        Ok(self.shared_abscissas()?.len())
    }

    /// Regroup the `k`-th tabulated node sub-field of every zone into a
    /// [`NodeField`]. Errors if the values are not node fields.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// // L'image tabulée k, recomposée en champ multi-zones.
    /// let f = e.node_frame(1)?;
    /// assert_eq!(f.get(0)?.read().value(n[0].id(), "T")?, 120.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn node_frame(&self, k: usize) -> Result<NodeField> {
        let mut field = NodeField::default();
        for h in &self.subs {
            match h.read().value_at(k)? {
                SubValue::Node(sf) => field.add_sub(Handle::new(sf))?,
                _ => return Err(not_node_err()),
            }
        }
        Ok(field)
    }

    /// Regroup the `k`-th tabulated element sub-field of every zone into an
    /// [`ElementField`]. Errors if the values are not element fields.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::{ElementField, SubElementField};
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let faire = |v: f64| {
    /// #     let f = ElementField::new(&fes, vec!["T".into()]).unwrap();
    /// #     f.get(0).unwrap().write().set_uniform("T", v).unwrap();
    /// #     f
    /// # };
    /// # let (froid, chaud) = (faire(20.0), faire(120.0));
    /// let e = Evolution::from_element_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// let f = e.element_frame(1)?;
    /// assert_eq!(f.get(0)?.read().value(0, 0, "T")?, 120.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_frame(&self, k: usize) -> Result<ElementField> {
        let mut field = ElementField::default();
        for h in &self.subs {
            match h.read().value_at(k)? {
                SubValue::Element(sf) => field.add_sub(Handle::new(sf))?,
                _ => return Err(not_element_err()),
            }
        }
        Ok(field)
    }

    /// One labelled `(abscissa, value)` series per sub-evolution — for a
    /// scalar X-Y plot. Errors if the values are not scalars.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
    /// // One labelled series per zone, ready for an X-Y plot.
    /// let series = e.scalar_series_set()?;
    /// assert_eq!(series[0].0, "value"); // "zone i" as soon as there are several
    /// assert_eq!(series[0].1, vec![(0.0, 20.0), (1.0, 120.0)]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn scalar_series_set(&self) -> Result<ScalarSeriesSet> {
        let mut out = Vec::with_capacity(self.subs.len());
        for (i, h) in self.subs.iter().enumerate() {
            let label = if self.subs.len() == 1 {
                "value".to_string()
            } else {
                format!("zone {i}")
            };
            out.push((label, h.read().scalar_series()?));
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
        view: crate::viz::View,
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
            None => self
                .abscissa_types()
                .first()
                .cloned()
                .unwrap_or_else(|| "variable".into()),
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
    /// samples — the classic X→Y curve (one sub-evolution), or a table when the
    /// abscissas are tuples (see [`SubEvolution::new`]).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
    /// assert_eq!(e.len(), 1); // a single curve
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_scalars<A: Abscissa>(
        samples: Vec<(A, f64)>,
        out_of_range: OutOfRange,
    ) -> Result<Self> {
        let pairs = samples
            .into_iter()
            .map(|(x, v)| (x, SubValue::Scalar(v)))
            .collect();
        let sub = SubEvolution::new(pairs, out_of_range)?;
        Ok(Evolution {
            subs: vec![Handle::new(sub)],
            out_of_range,
        })
    }

    /// Build an `Evolution` from whole node fields tabulated at each abscissa
    /// (time-major). The fields are **transposed** into one curve per zone:
    /// zones are identified by their support (matched across steps), so every
    /// step must carry the same set of supports as the first.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let mut froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # froid.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
    /// # let mut chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
    /// # chaud.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// // The fields are given **by date**; the evolution transposes them into
    /// // one curve per zone, the zones being matched by their support.
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.len(), froid.len());
    /// assert_eq!(e.frame_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_node_fields<A: Abscissa + Clone>(
        samples: &[(A, &NodeField)],
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
            let support = first.get(z)?.read().support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((x.clone(), SubValue::Node(node_sub_on(field, &support)?)));
            }
            evolution
                .subs
                .push(Handle::new(SubEvolution::new(curve, out_of_range)?));
        }
        Ok(evolution)
    }

    /// Build an `Evolution` from whole element fields tabulated at each
    /// abscissa (time-major). Transposed into one curve per FE subspace zone,
    /// matched across steps by support — see [`Evolution::from_node_fields`].
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::{ElementField, SubElementField};
    /// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let faire = |v: f64| {
    /// #     let f = ElementField::new(&fes, vec!["T".into()]).unwrap();
    /// #     f.get(0).unwrap().write().set_uniform("T", v).unwrap();
    /// #     f
    /// # };
    /// # let (froid, chaud) = (faire(20.0), faire(120.0));
    /// // Element fields transpose like nodal fields: one curve per FE
    /// // subspace, matched by its support.
    /// let e = Evolution::from_element_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.frame_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_element_fields<A: Abscissa + Clone>(
        samples: &[(A, &ElementField)],
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
            let support = first.get(z)?.read().support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((
                    x.clone(),
                    SubValue::Element(element_sub_on(field, &support)?),
                ));
            }
            evolution
                .subs
                .push(Handle::new(SubEvolution::new(curve, out_of_range)?));
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
        let sub = h.read();
        let fes = sub.support().read();
        mesh.add_sub(fes.submesh())?;
    }
    Ok(mesh)
}

/// Clone of the `field`'s sub-field on the given POI1 `support` (matched by
/// store slot). Errors if no zone carries that support.
fn node_sub_on(field: &NodeField, support: &Handle<SubMesh>) -> Result<SubNodeField> {
    for h in field.iter() {
        let s = h.read();
        if s.support().same_object(support) {
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
        let s = h.read();
        if s.support().same_object(support) {
            return Ok((*s).clone());
        }
    }
    Err(PyrucastError::Message(
        "Evolution::from_element_fields: a step is missing a zone present in the first step".into(),
    ))
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

// ─── Archive ────────────────────────────────────────────────────────────────

impl crate::archive::Archivable for SubEvolution {
    const TAG: &'static str = "SubEvolution";
}

impl crate::archive::Archivable for Evolution {
    const TAG: &'static str = "Evolution";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

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
        let coords = Handle::new(Coords::new(1).unwrap());
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for i in 0..n {
            let nd = Node::create_in(coords.clone(), &[i as f64]).unwrap();
            sm.add_cell(&[nd.id()]).unwrap();
        }
        Handle::new(sm)
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
        let scalar = Handle::new(scalar_curve(&[(0.0, 1.0)], OutOfRange::Error));
        let node = Handle::new(
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
        let sub = frame.get(0).unwrap().read();
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
                let sub = field.get(0).unwrap().read();
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
            .with_abscissa_types(vec!["T".into()])
            .unwrap()
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
            .with_abscissa_types(vec!["P".into()])
            .unwrap();
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
        e.set_abscissa_types(vec!["T".into()]).unwrap();
        e.set_ordinate_type(Some("E".into())).unwrap();
        let sm = poi1(2);
        let input = NodeField::from_sub(node_field(&sm, &[0.0, 10.0]));
        let out = e.interpolate_node_field(&input, None).unwrap();
        let sub = out.get(0).unwrap().read();
        assert_eq!(sub.components(), &["E".to_string()]);
        assert_eq!(sub.get(0, 0).unwrap(), 0.0);
        assert_eq!(sub.get(1, 0).unwrap(), 20.0);
    }

    #[test]
    fn two_curves_writing_one_component_are_refused() {
        // Two curves, no ordinate type: both would write `value`.
        let e0 = Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        let e1 = Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        let mut two = e0.union(&e1).unwrap();
        two.set_abscissa_types(vec!["T".into()]).unwrap();
        let input = NodeField::from_sub(node_field(&poi1(1), &[0.5]));
        assert!(two.interpolate_node_field(&input, None).is_err());
    }

    /// Several curves map a field into several components, each curve reading
    /// its own variables — one of them, here, a table of two.
    #[test]
    fn curves_of_different_variables_map_one_field() {
        let e = scalar_curve(&[(0.0, 1.0), (10.0, 3.0)], OutOfRange::Error)
            .with_abscissa_types(vec!["T".into()])
            .unwrap()
            .with_ordinate_type(Some("E".into()))
            .unwrap();
        let table = SubEvolution::new(
            vec![
                ([0.0, 0.0], SubValue::Scalar(0.0)),
                ([0.0, 1.0], SubValue::Scalar(1.0)),
                ([10.0, 0.0], SubValue::Scalar(10.0)),
                ([10.0, 1.0], SubValue::Scalar(11.0)),
            ],
            OutOfRange::Error,
        )
        .unwrap()
        .with_abscissa_types(vec!["T".into(), "phi".into()])
        .unwrap()
        .with_ordinate_type(Some("s".into()))
        .unwrap();
        let mut laws = Evolution::default();
        laws.add_sub(Handle::new(e)).unwrap();
        laws.add_sub(Handle::new(table)).unwrap();
        let sm = poi1(2);
        let mut input = SubNodeField::from_poi1(&sm, vec!["phi".into(), "T".into()]).unwrap();
        input.set(0, 0, 0.5).unwrap();
        input.set(0, 1, 5.0).unwrap();
        input.set(1, 0, 1.0).unwrap();
        input.set(1, 1, 10.0).unwrap();
        let out = laws
            .interpolate_node_field(&NodeField::from_sub(input), None)
            .unwrap();
        let z = out.get(0).unwrap().read();
        assert_eq!(z.components(), &["E".to_string(), "s".to_string()]);
        assert!((z.get(0, 0).unwrap() - 2.0).abs() < 1e-12); // E(5)
        assert!((z.get(0, 1).unwrap() - 5.5).abs() < 1e-12); // T + φ
        assert!((z.get(1, 1).unwrap() - 11.0).abs() < 1e-12);
    }

    /// `f(x, y) = 1 + 2x + 3y + 4xy` is linear in each variable alone, so the
    /// hierarchical reading reproduces it exactly — on a full grid, and on an
    /// irregular table whose groups do not share their second abscissas.
    #[test]
    fn a_table_reproduces_a_bilinear_function() {
        let f = |x: f64, y: f64| 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * y;
        let grid: Vec<([f64; 2], SubValue)> = [0.0, 1.0, 3.0]
            .iter()
            .flat_map(|&x| [0.0, 2.0].map(move |y| ([x, y], SubValue::Scalar(f(x, y)))))
            .collect();
        let irregular: Vec<([f64; 2], SubValue)> = [
            [0.0, 0.0],
            [0.0, 0.7],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [3.0, 0.0],
            [3.0, 1.2],
            [3.0, 1.5],
            [3.0, 2.0],
        ]
        .iter()
        .rev() // given out of order: sorting is the constructor's business
        .map(|&[x, y]| ([x, y], SubValue::Scalar(f(x, y))))
        .collect();
        for samples in [grid, irregular] {
            let t = SubEvolution::new(samples, OutOfRange::Error).unwrap();
            for (x, y) in [(0.3, 0.4), (1.0, 1.9), (2.2, 0.05), (3.0, 2.0), (0.0, 0.0)] {
                let got = t.eval_scalar_at(&[x, y], None).unwrap();
                assert!((got - f(x, y)).abs() < 1e-12, "f({x}, {y}): {got}");
            }
        }
    }

    /// Out of a level's range the policy applies; a variable measured at a
    /// single value in a group is constant along it there.
    #[test]
    fn a_table_honours_ranges_and_single_values() {
        let t = SubEvolution::new(
            vec![
                ([0.0, 0.0], SubValue::Scalar(1.0)),
                ([0.0, 1.0], SubValue::Scalar(2.0)),
                ([1.0, 0.5], SubValue::Scalar(7.0)), // one value of the second variable
            ],
            OutOfRange::Error,
        )
        .unwrap();
        // At the first variable's 1, the second does not vary.
        assert_eq!(t.eval_scalar_at(&[1.0, 0.9], None).unwrap(), 7.0);
        // Halfway: (1.5 + 7) / 2 at y = 0.5.
        assert!((t.eval_scalar_at(&[0.5, 0.5], None).unwrap() - 4.25).abs() < 1e-12);
        // Out of the second variable's range in the first group: an error…
        assert!(t.eval_scalar_at(&[0.0, 1.5], None).is_err());
        // …or the clamped value, if asked for.
        assert_eq!(
            t.eval_scalar_at(&[0.0, 1.5], Some(OutOfRange::Clamp))
                .unwrap(),
            2.0
        );
        // Extrapolated along the two nearest values of the first variable.
        assert!(
            (t.eval_scalar_at(&[2.0, 0.5], Some(OutOfRange::Extrapolate))
                .unwrap()
                - 12.5)
                .abs()
                < 1e-12
        );
        // The wrong number of variables is refused.
        assert!(t.eval_scalar_at(&[0.5], None).is_err());
    }

    /// Tuples of mixed lengths, and repeated tuples, are refused at build time.
    #[test]
    fn a_table_validates_its_abscissas() {
        assert!(SubEvolution::new(
            vec![
                (vec![0.0, 0.0], SubValue::Scalar(1.0)),
                (vec![1.0], SubValue::Scalar(2.0)),
            ],
            OutOfRange::Error,
        )
        .is_err());
        assert!(SubEvolution::new(
            vec![
                ([0.0, 1.0], SubValue::Scalar(1.0)),
                ([0.0, 1.0], SubValue::Scalar(2.0)),
            ],
            OutOfRange::Error,
        )
        .is_err());
    }

    #[test]
    fn types_round_trip_on_aggregate() {
        let mut e =
            Evolution::from_scalars(vec![(0.0, 0.0), (1.0, 1.0)], OutOfRange::Error).unwrap();
        e.set_abscissa_types(vec!["time".into()]).unwrap();
        e.set_ordinate_type(Some("force".into())).unwrap();
        assert_eq!(e.abscissa_types(), ["time"]);
        assert_eq!(e.ordinate_type().unwrap().as_deref(), Some("force"));
    }
}
