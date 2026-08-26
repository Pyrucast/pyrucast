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
/// // Ce que consomme un tracé X-Y : une série étiquetée par zone.
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
/// // Trois réponses possibles à « et au-delà du dernier point tabulé ? »
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
/// // Une courbe est homogène : scalaires, champs nodaux ou champs par
/// // éléments, jamais un mélange.
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
/// // Une valeur tabulée se lit par filtrage de variante — ou, pour le cas
/// // scalaire, par le raccourci `eval_scalar`.
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

/// Linear blend `lo*(1-t) + hi*t` of two values of the **same** kind. For
/// fields the per-value arithmetic of [`SubField`] (`map_all` +
/// `merge_components`) is reused; interpolating between two tabulated values of
/// the *same* field, a mismatched support or component set is a genuine error,
/// so [`SubField::check_same_components`] guards the merge up front.
fn lerp(lo: &SubValue, hi: &SubValue, t: f64) -> Result<SubValue> {
    match (lo, hi) {
        (SubValue::Scalar(a), SubValue::Scalar(b)) => Ok(SubValue::Scalar(a * (1.0 - t) + b * t)),
        (SubValue::Node(a), SubValue::Node(b)) => {
            a.check_same_components(b)?;
            let la = a.map_all(|v| v * (1.0 - t));
            let lb = b.map_all(|v| v * t);
            Ok(SubValue::Node(la.merge_components(&lb, |x, y| x + y)?))
        }
        (SubValue::Element(a), SubValue::Element(b)) => {
            a.check_same_components(b)?;
            let la = a.map_all(|v| v * (1.0 - t));
            let lb = b.map_all(|v| v * t);
            Ok(SubValue::Element(la.merge_components(&lb, |x, y| x + y)?))
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// // Une température qui monte de 20 à 120 en une unité de temps.
/// let c = SubEvolution::new(
///     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
///     OutOfRange::Clamp)?;
/// assert_eq!(c.len(), 2);
/// assert_eq!(c.eval_scalar(0.5, None)?, 70.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Les échantillons sont **triés** à la construction : les donner dans le
    /// // désordre est licite, les donner deux fois à la même abscisse ne l'est pas.
    /// let c = SubEvolution::new(
    ///     vec![(1.0, SubValue::Scalar(120.0)), (0.0, SubValue::Scalar(20.0))],
    ///     OutOfRange::Clamp)?;
    /// assert_eq!(c.abscissas(), &[0.0, 1.0]);
    /// assert!(SubEvolution::new(
    ///     vec![(0.0, SubValue::Scalar(1.0)), (0.0, SubValue::Scalar(2.0))],
    ///     OutOfRange::Error).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
        // component count, so the blend (guarded by `check_same_components`)
        // is well-defined.
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
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let c = courbe.with_abscissa_type(Some("T".into()));
    /// assert_eq!(c.abscissa_type(), Some("T"));
    /// ```
    pub fn abscissa_type(&self) -> Option<&str> {
        self.abscissa_type.as_deref()
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

    /// Set (or clear) the abscissa's physical type.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// let mut c = courbe;
    /// c.set_abscissa_type(Some("time".into()));
    /// assert_eq!(c.abscissa_type(), Some("time"));
    /// c.set_abscissa_type(None); // et l'on peut le retirer
    /// assert!(c.abscissa_type().is_none());
    /// ```
    pub fn set_abscissa_type(&mut self, t: Option<String>) {
        self.abscissa_type = t;
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
    /// // Une ordonnée n'a de type que sur une courbe **scalaire** : sur une
    /// // courbe de champs, la pose échoue.
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

    /// Builder form of [`SubEvolution::set_abscissa_type`].
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Forme chaînée, pour poser le type à la construction.
    /// let c = SubEvolution::new(
    ///     vec![(20.0, SubValue::Scalar(210_000.0)), (300.0, SubValue::Scalar(180_000.0))],
    ///     OutOfRange::Clamp)?
    ///     .with_abscissa_type(Some("T".into()));
    /// assert_eq!(c.abscissa_type(), Some("T"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_abscissa_type(mut self, t: Option<String>) -> Self {
        self.abscissa_type = t;
        self
    }

    /// Builder form of [`SubEvolution::set_ordinate_type`] (validated).
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Un module d'Young qui dépend de la température : les deux types posés
    /// // ensemble font de la courbe une fonction de transfert nommée.
    /// let c = courbe
    ///     .with_abscissa_type(Some("T".into()))
    ///     .with_ordinate_type(Some("young".into()))?;
    /// assert_eq!((c.abscissa_type(), c.ordinate_type()), (Some("T"), Some("young")));
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
        self.abscissas.len()
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
        self.abscissas.is_empty()
    }

    /// The sorted abscissas.
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
    /// // La valeur **tabulée** d'indice k — sans interpolation.
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
            subs: vec![Handle::new(self.clone())],
            out_of_range: self.out_of_range,
        };
        evo.plot(
            view, save, mesh, component, scale, smooth, frame, x_label, y_label, title,
        )
    }

    /// Interpolate at `x`. `policy` overrides the stored [`OutOfRange`] for
    /// this single query when `Some`.
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Interpolation linéaire entre deux points tabulés…
    /// match courbe.interpolate(0.25, None)? {
    ///     SubValue::Scalar(v) => assert_eq!(v, 45.0),
    ///     _ => unreachable!(),
    /// }
    /// // …et hors domaine, la politique de la courbe s'applique — sauf si
    /// // l'appel en impose une autre.
    /// assert!(courbe.interpolate(2.0, Some(OutOfRange::Error)).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::containers::evolution::{OutOfRange, SubEvolution, SubValue, ValueKind};
    /// # let courbe = SubEvolution::new(
    /// #     vec![(0.0, SubValue::Scalar(20.0)), (1.0, SubValue::Scalar(120.0))],
    /// #     OutOfRange::Clamp).unwrap();
    /// // Le raccourci sans filtrage de variante, pour une courbe scalaire.
    /// assert_eq!(courbe.eval_scalar(0.5, None)?, 70.0);
    /// assert_eq!(courbe.eval_scalar(9.0, None)?, 120.0); // Clamp
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    /// // Une conductivité qui dépend de la température : la courbe est une
    /// // fonction de transfert, et ce sont les **types** qui l'accrochent au
    /// // champ — l'abscisse nomme la composante lue, l'ordonnée celle produite.
    /// let loi = SubEvolution::new(
    ///     vec![(20.0, SubValue::Scalar(50.0)), (120.0, SubValue::Scalar(30.0))],
    ///     OutOfRange::Clamp)?
    ///     .with_abscissa_type(Some("T".into()))
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
            for (_, v) in &samples[1..] {
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::containers::evolution::{Evolution, Interpolated, OutOfRange, SubEvolution, SubValue, ValueKind};
/// let e = Evolution::from_scalars(vec![(0.0, 20.0), (1.0, 120.0)], OutOfRange::Clamp)?;
/// assert_eq!(e.len(), 1); // une courbe par zone ; ici une seule
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
/// // Interpoler un agrégat rend un agrégat — sauf pour les scalaires, dont
/// // il n'existe pas d'agrégat : un `Vec<f64>`, une valeur par zone.
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
    /// // Un chargement à mi-course : le champ entier, zone par zone.
    /// match e.interpolate(0.5, None)? {
    ///     Interpolated::Node(f) => assert_eq!(f.get(0)?.read().value(n[0].id(), "T")?, 70.0),
    ///     _ => unreachable!(),
    /// }
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn interpolate(&self, x: f64, policy: Option<OutOfRange>) -> Result<Interpolated> {
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
                    match h.read().interpolate(x, policy)? {
                        SubValue::Scalar(v) => out.push(v),
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Scalars(out))
            }
            ValueKind::Node => {
                let mut field = NodeField::default();
                for h in &self.subs {
                    match h.read().interpolate(x, policy)? {
                        SubValue::Node(sf) => field.add_sub(Handle::new(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Node(field))
            }
            ValueKind::Element => {
                let mut field = ElementField::default();
                for h in &self.subs {
                    match h.read().interpolate(x, policy)? {
                        SubValue::Element(sf) => field.add_sub(Handle::new(sf))?,
                        _ => return Err(mixed_kind_err()),
                    }
                }
                Ok(Interpolated::Element(field))
            }
        }
    }

    /// The abscissa's physical type (taken from the first sub-evolution), if
    /// set. `None` for an empty evolution.
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
    /// assert_eq!(e.abscissa_type()?, None);
    /// e.set_abscissa_type(Some("time".into()))?;
    /// assert_eq!(e.abscissa_type()?, Some("time".into()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn abscissa_type(&self) -> Result<Option<String>> {
        match self.subs.first() {
            Some(h) => Ok(h.read().abscissa_type().map(str::to_string)),
            None => Ok(None),
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

    /// Set (or clear) the abscissa type on **every** sub-evolution.
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
    /// // Le type est posé sur **toutes** les zones à la fois.
    /// let mut e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// e.set_abscissa_type(Some("time".into()))?;
    /// assert_eq!(e.abscissa_type()?, Some("time".into()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_abscissa_type(&mut self, t: Option<String>) -> Result<()> {
        for h in &self.subs {
            h.write().set_abscissa_type(t.clone());
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
    /// // Sur une évolution de champs, l'ordonnée n'a pas de type à porter.
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
        let sub = self.subs[0].read();
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::evolution::{Evolution, OutOfRange};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
    /// #                                    vec!["T".into()]).unwrap();
    /// # temp.get(0).unwrap().write().add_to_component("T", 120.0).unwrap();
    /// // La courbe en **fonction de transfert** : ce sont les types qui
    /// // l'accrochent au champ — l'abscisse nomme la composante lue,
    /// // l'ordonnée celle produite.
    /// let mut loi = Evolution::from_scalars(
    ///     vec![(20.0, 50.0), (120.0, 30.0)], OutOfRange::Clamp)?;
    /// loi.set_abscissa_type(Some("T".into()))?;
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
        let curve = self.single_scalar_curve()?;
        let policy = Some(policy.unwrap_or(self.out_of_range));
        field.map_subs(move |s| curve.interpolate_field(s, policy))
    }

    /// Map a whole [`ElementField`] through the (single, scalar) curve — see
    /// [`Evolution::interpolate_node_field`].
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
    /// // La courbe joue ici le rôle de **fonction de transfert** : `T` fournit
    /// // les abscisses, `k` est la composante produite.
    /// let mut loi =
    ///     Evolution::from_scalars(vec![(20.0, 50.0), (120.0, 30.0)], OutOfRange::Clamp)?;
    /// loi.set_abscissa_type(Some("T".into()))?;
    /// loi.set_ordinate_type(Some("k".into()))?;
    /// let k = loi.interpolate_element_field(&chaud, None)?;
    /// assert_eq!(k.get(0)?.read().value(0, 0, "k")?, 30.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    /// // Une grille commune à toutes les zones — ce qu'exige un curseur d'image.
    /// assert_eq!(e.shared_abscissas()?, vec![0.0, 1.0]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn shared_abscissas(&self) -> Result<Vec<f64>> {
        if self.subs.is_empty() {
            return Err(PyrucastError::Message("Evolution: empty evolution".into()));
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
    /// // Une série étiquetée par zone, prête pour un tracé X-Y.
    /// let series = e.scalar_series_set()?;
    /// assert_eq!(series[0].0, "value"); // « zone i » dès qu'il y en a plusieurs
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
    /// assert_eq!(e.len(), 1); // une seule courbe
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_scalars(samples: Vec<(f64, f64)>, out_of_range: OutOfRange) -> Result<Self> {
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
    /// // Les champs sont donnés **par date** ; l'évolution les transpose en une
    /// // courbe par zone, les zones étant appariées par leur support.
    /// let e = Evolution::from_node_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.len(), froid.len());
    /// assert_eq!(e.frame_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
            let support = first.get(z)?.read().support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((*x, SubValue::Node(node_sub_on(field, &support)?)));
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
    /// // Les champs par éléments se transposent comme les champs nodaux : une
    /// // courbe par sous-espace EF, apparié par son support.
    /// let e = Evolution::from_element_fields(&[(0.0, &froid), (1.0, &chaud)], OutOfRange::Clamp)?;
    /// assert_eq!(e.frame_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
            let support = first.get(z)?.read().support();
            let mut curve = Vec::with_capacity(samples.len());
            for (x, field) in samples {
                curve.push((*x, SubValue::Element(element_sub_on(field, &support)?)));
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
        let sub = out.get(0).unwrap().read();
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
