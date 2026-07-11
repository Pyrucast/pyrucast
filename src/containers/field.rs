//! Common traits for field containers — the shared grammar between node
//! fields and element fields.
//!
//! - [`SubField`] — one homogeneous block of field values: named
//!   components plus a flat buffer in which the component index varies
//!   fastest. Name lookup and per-component `min`/`max` are derived from
//!   that contract alone.
//! - [`Field`] — the aggregate-level view, blanket-implemented for every
//!   [`Aggregate`] whose sub-object is a `SubField`: union of the subs'
//!   components, `min`/`max` folded across the subs.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::field::SubField;
//! use pyrucast::containers::mesh::{Coords, ElementType, Node, SubMesh};
//! use pyrucast::containers::node_field::SubNodeField;
//! use pyrucast::store::insert;
//!
//! let coords = insert(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let sm = {
//!     let mut sm = SubMesh::new(coords, ElementType::POI1);
//!     sm.add_cell(&[a.id()]).unwrap();
//!     sm.add_cell(&[b.id()]).unwrap();
//!     insert(sm)
//! };
//! let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
//! f.set(0, 0, -3.0).unwrap();
//! f.set(1, 0, 7.5).unwrap();
//! assert_eq!(SubField::min(&f, "T").unwrap(), -3.0);
//! assert_eq!(SubField::max(&f, "T").unwrap(), 7.5);
//! ```

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::parallel::*;
use crate::persist::Persist;
use crate::store::{insert, read, write, Handle, ReadGuard};
use std::any::Any;

/// Validate a component-name list: non-empty, no duplicate names.
/// `kind` names the container in the error message.
pub(crate) fn check_components(kind: &str, components: &[String]) -> Result<()> {
    if components.is_empty() {
        return Err(PyrucastError::Message(format!(
            "{kind} requires at least one component"
        )));
    }
    for i in 0..components.len() {
        for j in (i + 1)..components.len() {
            if components[i] == components[j] {
                return Err(PyrucastError::Message(format!(
                    "duplicate component name: {}",
                    components[i]
                )));
            }
        }
    }
    Ok(())
}

/// Zero-copy view of a field aggregate's zones: one owned read guard
/// per sub-field plus the union of the component names, built by
/// [`Field::view`]. Holding the view keeps a shared lock on every sub:
/// concurrent reads are free, writes wait until the view is dropped.
///
/// The kind-specific reading methods live next to each concrete sub
/// type, on the `NodeFieldView` and `ElementFieldView` aliases.
pub struct FieldView<S: Persist + Any + Send + Sync> {
    pub(crate) zones: Vec<ReadGuard<S>>,
    components: Vec<String>,
}

impl<S: Persist + Any + Send + Sync> FieldView<S> {
    /// Union of the zones' component names, first-seen order.
    pub fn components(&self) -> &[String] {
        &self.components
    }
}

// ─── SubField ───────────────────────────────────────────────────────────────

/// One homogeneous block of field values.
///
/// The contract is purely structural: named components plus a flat value
/// buffer in which the component index varies fastest (stride =
/// `component_count()`). Both `SubNodeField` (node-major) and
/// `SubElementField` (cell → gauss major) satisfy it.
pub trait SubField {
    /// Type of the object backing this sub-field's support — a `SubMesh` for
    /// node fields, a `SubFiniteElementSpace` for element fields. Its store
    /// slot identity defines [`SubField::same_support`].
    type Support: Persist + Any + Send + Sync;

    /// Handle to the support backing this sub-field.
    fn support(&self) -> Handle<Self::Support>;

    /// Whether `self` and `other` are backed by the **same** support slot
    /// ([`Handle::same_slot`]) — the precondition for combining them.
    fn same_support(&self, other: &Self) -> bool {
        self.support().same_slot(&other.support())
    }

    /// Component names, in order.
    fn components(&self) -> &[String];

    /// Flat value buffer; the component index varies fastest.
    fn values(&self) -> &[f64];

    /// Number of components.
    fn component_count(&self) -> usize {
        self.components().len()
    }

    /// Index of a named component, or `None` if absent.
    fn component_index(&self, name: &str) -> Option<usize> {
        self.components().iter().position(|c| c == name)
    }

    /// Index of a named component, or an error naming it.
    fn component_index_or_err(&self, name: &str) -> Result<usize> {
        self.component_index(name)
            .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", name)))
    }

    /// Flat value buffer, mutable (same layout as [`SubField::values`]).
    fn values_mut(&mut self) -> &mut [f64];

    /// A fresh, zero-initialised field on the **same support** as `self`, but
    /// carrying `components` instead of `self`'s. Its rows are in the same
    /// order as `self`'s (both derive from the same support), so values line
    /// up positionally — the primitive behind reductions that change the
    /// component set, such as [`SubField::pscal`].
    fn same_support_with(&self, components: Vec<String>) -> Result<Self>
    where
        Self: Sized;

    /// Add `scalar` to every entry of the named component.
    fn add_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, |v| v + scalar)
    }

    /// Subtract `scalar` from every entry of the named component.
    fn sub_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, |v| v - scalar)
    }

    /// Multiply every entry of the named component by `scalar`.
    fn mul_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, |v| v * scalar)
    }

    /// Divide every entry of the named component by `scalar`.
    ///
    /// Returns an error if `scalar` is zero.
    fn div_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        if scalar == 0.0 {
            return Err(PyrucastError::Message(
                "div_to_component: division by zero".into(),
            ));
        }
        self.map_component(component, |v| v / scalar)
    }

    /// Set every entry of the named component to `value`.
    fn set_uniform(&mut self, component: &str, value: f64) -> Result<()> {
        self.map_component(component, |_| value)
    }

    /// Apply `f` to every entry of the named component (stride =
    /// component count, offset = component index). Parallel; each touched slot
    /// is written once ⇒ thread-count-independent.
    fn map_component(
        &mut self,
        component: &str,
        f: impl Fn(f64) -> f64 + Sync + Send,
    ) -> Result<()> {
        let ci = self.component_index_or_err(component)?;
        let ncomp = self.component_count();
        crate::parallel::map_component_inplace(self.values_mut(), ncomp, ci, f);
        Ok(())
    }

    /// A clone of `self` with `f` applied to **every** value (all
    /// nodes/points × all components) — the scalar-broadcast primitive.
    /// Parallel; each value written once ⇒ thread-count-independent.
    fn map_all(&self, f: impl Fn(f64) -> f64 + Sync + Send) -> Self
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        crate::parallel::map_inplace(out.values_mut(), f);
        out
    }

    /// Check that `self` and `other` share the **same support** and the **same
    /// set of components** (aligned by name, order may differ) — otherwise an
    /// error. A standalone coherence guard: call it *before*
    /// [`SubField::merge_components`] where a mismatched component set is a
    /// genuine error rather than a passthrough (e.g. interpolating between two
    /// tabulated values of the *same* field). When it returns `Ok`, the union
    /// `merge_components` produces equals the strict element-by-element
    /// combination (identical sets ⇒ no passthrough branch fires).
    fn check_same_components(&self, other: &Self) -> Result<()> {
        if !self.same_support(other) {
            return Err(PyrucastError::Message(
                "check_same_components: operands are not on the same support".into(),
            ));
        }
        if self.component_count() != other.component_count()
            || self
                .components()
                .iter()
                .any(|c| other.component_index(c).is_none())
        {
            return Err(PyrucastError::Message(format!(
                "check_same_components: mismatched components ({:?} vs {:?})",
                self.components(),
                other.components()
            )));
        }
        Ok(())
    }

    /// Component-wise **union** combination with another same-support
    /// sub-field — the primitive behind the zone-level operators
    /// (`&a + &b`, …). Unlike a strict combination (identical component sets,
    /// guarded by [`SubField::check_same_components`]), the output carries the
    /// **union** of the two component sets
    /// (`self`'s first, then `other`'s extras, first-seen order). For a
    /// component defined on **both** sides, the value is `op(self, other)`; for
    /// a component defined on **only one** side, that side's value **passes
    /// through unchanged** (raw passthrough — for every operator, so
    /// `a - b`'s `b`-only component is `b`, not `-b`). Both must be on the same
    /// support ([`SubField::same_support`]); same support ⇒ same rows in the
    /// same order, so values line up positionally.
    fn merge_components(&self, other: &Self, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized + Clone,
    {
        if !self.same_support(other) {
            return Err(PyrucastError::Message(
                "merge_components: operands are not on the same support".into(),
            ));
        }
        // Union of components, self's first then other's extras.
        let mut components: Vec<String> = self.components().to_vec();
        for c in other.components() {
            if !components.contains(c) {
                components.push(c.clone());
            }
        }
        let mut out = self.same_support_with(components.clone())?;
        let out_nc = out.component_count();
        let self_nc = self.component_count();
        let other_nc = other.component_count();
        let sv = self.values();
        let ov = other.values();
        // Same support ⇒ identical row count on every operand.
        let rows = out.values().len().checked_div(out_nc).unwrap_or(0);
        // Precompute, per output component, the source column on each side.
        let src: Vec<(usize, Option<usize>, Option<usize>)> = components
            .iter()
            .enumerate()
            .map(|(oc, name)| (oc, self.component_index(name), other.component_index(name)))
            .collect();
        let outv = out.values_mut();
        for row in 0..rows {
            for &(oc, si, oi) in &src {
                let v = match (si, oi) {
                    (Some(si), Some(oi)) => op(sv[row * self_nc + si], ov[row * other_nc + oi]),
                    (Some(si), None) => sv[row * self_nc + si],
                    (None, Some(oi)) => ov[row * other_nc + oi],
                    (None, None) => unreachable!("output component comes from at least one side"),
                };
                outv[row * out_nc + oc] = v;
            }
        }
        Ok(out)
    }

    /// Scalar product `∑ selfᵢ · otherᵢ` over the components **shared** by both
    /// sub-fields. Mirrors [`SubField::merge_components`]'s union/passthrough
    /// spirit at the inner-product level: a component present on only one side
    /// has no counterpart to multiply, so it contributes nothing. Both must sit
    /// on the same support ([`SubField::same_support`]); components are aligned
    /// by name. This is the field inner product behind Cast3M's `XTY`/`PSCA`
    /// (energy `F·u`, residual norms, …).
    fn dot(&self, other: &Self) -> Result<f64> {
        if !self.same_support(other) {
            return Err(PyrucastError::Message(
                "dot: operands are not on the same support".into(),
            ));
        }
        let self_nc = self.component_count();
        let other_nc = other.component_count();
        // Shared components: (self column, other column) for each name in both.
        let pairs: Vec<(usize, usize)> = self
            .components()
            .iter()
            .enumerate()
            .filter_map(|(si, name)| other.component_index(name).map(|oi| (si, oi)))
            .collect();
        if pairs.is_empty() {
            return Ok(0.0);
        }
        let sv = self.values();
        let ov = other.values();
        let rows = sv.len().checked_div(self_nc).unwrap_or(0);
        // Per-row product-sum in parallel, then an associative reduction.
        // Same support ⇒ identical row layout on both sides. Floating-point
        // `+` is not associative, so the total is thread-count-dependent to
        // the last ULP — like the solver, not bit-for-bit reproducible.
        let sum = (0..rows)
            .into_par_iter()
            .with_min_len((MIN_PARALLEL_LEN / self_nc.max(1)).max(1))
            .map(|row| {
                let mut acc = 0.0;
                for &(si, oi) in &pairs {
                    acc += sv[row * self_nc + si] * ov[row * other_nc + oi];
                }
                acc
            })
            .sum();
        Ok(sum)
    }

    /// Per-row scalar product `∑_c selfᵣ,c · otherᵣ,c`, one value per row —
    /// Cast3M's `PSCA`. Unlike [`SubField::dot`] (which reduces the whole field
    /// to a single number), this reduces **over components only**, keeping the
    /// support: the result is a fresh single-component field (component
    /// `"psca"`) on the same support, holding the node-by-node (or
    /// point-by-point) scalar product.
    ///
    /// Same union spirit as [`SubField::dot`]: the sum runs over the components
    /// **shared** by both sides (aligned by name); a component present on only
    /// one side has no counterpart and contributes nothing (rows with no shared
    /// component are `0`). Both must be on the same support
    /// ([`SubField::same_support`]).
    fn pscal(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        if !self.same_support(other) {
            return Err(PyrucastError::Message(
                "pscal: operands are not on the same support".into(),
            ));
        }
        let self_nc = self.component_count();
        let other_nc = other.component_count();
        // Shared components: (self column, other column) for each name in both.
        let pairs: Vec<(usize, usize)> = self
            .components()
            .iter()
            .enumerate()
            .filter_map(|(si, name)| other.component_index(name).map(|oi| (si, oi)))
            .collect();
        let sv = self.values();
        let ov = other.values();
        // Same support ⇒ identical row layout on both operands and on the
        // fresh output, so rows line up positionally. Each output slot is
        // written once (per-row, no shared accumulation) ⇒ the result is
        // independent of the thread count.
        let mut out = self.same_support_with(vec!["psca".to_string()])?;
        out.values_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, o)| {
                let mut acc = 0.0;
                for &(si, oi) in &pairs {
                    acc += sv[row * self_nc + si] * ov[row * other_nc + oi];
                }
                *o = acc;
            });
        Ok(out)
    }

    /// Smallest value of the named component.
    ///
    /// Errors if the component is unknown or the field holds no value.
    fn min(&self, component: &str) -> Result<f64> {
        fold_component(self, component, "min", f64::min)
    }

    /// Largest value of the named component.
    ///
    /// Errors if the component is unknown or the field holds no value.
    fn max(&self, component: &str) -> Result<f64> {
        fold_component(self, component, "max", f64::max)
    }

    /// Sum of the named component over the support (`Σ` over nodes / points).
    ///
    /// An empty support sums to `0.0`; errors only if the component is unknown.
    /// Unlike [`SubField::min`] / [`SubField::max`] (exact regardless of order),
    /// the sum groups adaptively in parallel, so — like [`SubField::dot`] — it is
    /// thread-count-dependent to the last ULP.
    fn sum(&self, component: &str) -> Result<f64> {
        let ci = self
            .component_index(component)
            .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", component)))?;
        let nc = self.component_count();
        Ok(self
            .values()
            .par_chunks(nc.max(1))
            .with_min_len((MIN_PARALLEL_LEN / nc.max(1)).max(1))
            .map(|row| row[ci])
            .sum())
    }

    /// Squared Euclidean norm `xᵀx = Σ v²` over **every** value (all components)
    /// — Cast3M `XTX`, i.e. [`SubField::dot`] of the field with itself. Like
    /// `dot`, thread-count-dependent to the last ULP.
    fn xtx(&self) -> f64 {
        self.values()
            .par_iter()
            .with_min_len(MIN_PARALLEL_LEN)
            .map(|v| v * v)
            .sum()
    }

    /// Squared Euclidean norm restricted to the named `components` — `Σ v²`
    /// over the selected components only, the rest ignored. Components this
    /// sub-field does not carry are silently skipped (an aggregate may spread
    /// them across zones); errors only if **none** of them is present here.
    /// Like [`SubField::xtx`], thread-count-dependent to the last ULP.
    fn xtx_components(&self, components: &[&str]) -> Result<f64> {
        let nc = self.component_count();
        let indices: Vec<usize> = components
            .iter()
            .filter_map(|c| self.component_index(c))
            .collect();
        if indices.is_empty() {
            return Err(PyrucastError::Message(format!(
                "xtx_components: none of {:?} present in this sub-field",
                components
            )));
        }
        // Fast path: every component selected ⇒ the whole flat buffer.
        if indices.len() == nc {
            return Ok(self.xtx());
        }
        Ok(self
            .values()
            .par_chunks(nc.max(1))
            .with_min_len((MIN_PARALLEL_LEN / nc.max(1)).max(1))
            .map(|row| indices.iter().map(|&ci| row[ci] * row[ci]).sum::<f64>())
            .sum())
    }
}

/// Fold `op` over every value of one component of a [`SubField`].
fn fold_component<S: SubField + ?Sized>(
    field: &S,
    component: &str,
    op_name: &str,
    op: fn(f64, f64) -> f64,
) -> Result<f64> {
    let ci = field
        .component_index(component)
        .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", component)))?;
    let n_comp = field.component_count();
    // Parallel reduction over rows: `op` (min/max) is associative & commutative
    // ⇒ the result is identical to the sequential left-fold for any thread count.
    field
        .values()
        .par_chunks(n_comp)
        .with_min_len((MIN_PARALLEL_LEN / n_comp).max(1))
        .map(|row| row[ci])
        .reduce_with(op)
        .ok_or_else(|| {
            PyrucastError::Message(format!(
                "{}: no value for component {} (empty support)",
                op_name, component
            ))
        })
}

// ─── Scalar-operator macro ──────────────────────────────────────────────────

/// Generate `Add/Sub/Mul/Div<f64>` for a [`SubField`] container: the
/// consuming versions mutate every value in place and return `self`
/// (zero-copy); the reference versions clone first.
///
/// # Usage
/// ```ignore
/// impl_subfield_scalar_ops!(SubNodeField);
/// ```
#[macro_export]
macro_rules! impl_subfield_scalar_ops {
    ($T:ty) => {
        $crate::__subfield_scalar_op!($T, Add, add, +);
        $crate::__subfield_scalar_op!($T, Sub, sub, -);
        $crate::__subfield_scalar_op!($T, Mul, mul, *);
        $crate::__subfield_scalar_op!($T, Div, div, /);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __subfield_scalar_op {
    ($T:ty, $Trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$Trait<f64> for $T {
            type Output = $T;
            fn $method(mut self, rhs: f64) -> $T {
                // Parallel in-place broadcast; each value written once.
                $crate::parallel::map_inplace(
                    $crate::containers::field::SubField::values_mut(&mut self),
                    move |v| v $op rhs,
                );
                self
            }
        }
        impl std::ops::$Trait<f64> for &$T {
            type Output = $T;
            fn $method(self, rhs: f64) -> $T {
                std::ops::$Trait::$method(self.clone(), rhs)
            }
        }
    };
}

// ─── Field-by-field operator macros ─────────────────────────────────────────

/// Generate `Add/Sub/Mul/Div` between two [`SubField`] containers of the same
/// type, delegating to [`SubField::merge_components`] (per-component union with
/// passthrough on a shared support). Combining can fail (different support), so
/// — like the crate's other fallible operators (e.g. `&Matrix * &NodeField`) —
/// the output is a [`Result`]: `(&a + &b)?`. Division does **not** guard against
/// zero (numpy-like `inf`/`nan`).
///
/// # Usage
/// ```ignore
/// impl_subfield_field_ops!(SubNodeField);
/// ```
#[macro_export]
macro_rules! impl_subfield_field_ops {
    ($T:ty) => {
        $crate::__subfield_field_op!($T, Add, add, +);
        $crate::__subfield_field_op!($T, Sub, sub, -);
        $crate::__subfield_field_op!($T, Mul, mul, *);
        $crate::__subfield_field_op!($T, Div, div, /);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __subfield_field_op {
    ($T:ty, $Trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$Trait<&$T> for &$T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: &$T) -> Self::Output {
                $crate::containers::field::SubField::merge_components(self, rhs, |a, b| a $op b)
            }
        }
        impl std::ops::$Trait<$T> for $T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: $T) -> Self::Output {
                $crate::containers::field::SubField::merge_components(&self, &rhs, |a, b| a $op b)
            }
        }
    };
}

/// Generate `Add/Sub/Mul/Div` for an aggregate [`Field`], both **field ∘ field**
/// (via [`Field::merge_field`], per `(support, component)` with passthrough) and
/// **field ∘ scalar** (via [`Field::combine_scalar`], broadcast over every
/// value). Both go through the store (fallible reads) so the output is a
/// [`Result`]: `(&a + &b)?`, `(&a * 2.0)?`.
///
/// # Usage
/// ```ignore
/// impl_field_ops!(NodeField);
/// ```
#[macro_export]
macro_rules! impl_field_ops {
    ($T:ty) => {
        $crate::__field_op!($T, Add, add, +);
        $crate::__field_op!($T, Sub, sub, -);
        $crate::__field_op!($T, Mul, mul, *);
        $crate::__field_op!($T, Div, div, /);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __field_op {
    ($T:ty, $Trait:ident, $method:ident, $op:tt) => {
        // field ∘ field
        impl std::ops::$Trait<&$T> for &$T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: &$T) -> Self::Output {
                $crate::containers::field::Field::merge_field(self, rhs, |a, b| a $op b)
            }
        }
        impl std::ops::$Trait<$T> for $T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: $T) -> Self::Output {
                $crate::containers::field::Field::merge_field(&self, &rhs, |a, b| a $op b)
            }
        }
        // field ∘ scalar
        impl std::ops::$Trait<f64> for &$T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: f64) -> Self::Output {
                $crate::containers::field::Field::combine_scalar(self, |a, b| a $op b, rhs)
            }
        }
        impl std::ops::$Trait<f64> for $T {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: f64) -> Self::Output {
                $crate::containers::field::Field::combine_scalar(&self, |a, b| a $op b, rhs)
            }
        }
    };
}

// ─── Field (aggregate level) ────────────────────────────────────────────────

/// Aggregate-level view of a field.
///
/// Blanket-implemented for every [`Aggregate`] whose sub-object is a
/// [`SubField`] — `ElementField` today, `NodeField` once it becomes an
/// aggregate. A component may exist on some subs only; `min`/`max` fold
/// over the subs that define it and error if none does.
pub trait Field: Aggregate
where
    Self::Sub: SubField,
{
    /// Union of the subs' component names, first-seen order.
    fn components(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for h in self.iter() {
            let s = read(h)?;
            for c in s.components() {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
        }
        Ok(out)
    }

    /// Smallest value of `component` across the subs defining it.
    ///
    /// Errors if no sub defines the component.
    fn min(&self, component: &str) -> Result<f64> {
        fold_subs(self, component, "min", f64::min)
    }

    /// Largest value of `component` across the subs defining it.
    ///
    /// Errors if no sub defines the component.
    fn max(&self, component: &str) -> Result<f64> {
        fold_subs(self, component, "max", f64::max)
    }

    /// Sum of `component` across the subs defining it (`Σ` over the whole field
    /// — e.g. the resultant of a nodal force field, one component at a time).
    ///
    /// Errors if no sub defines the component; a defining sub with an empty
    /// support contributes `0.0`. Thread-count-dependent to the last ULP.
    fn sum(&self, component: &str) -> Result<f64> {
        let handles: Vec<&Handle<Self::Sub>> = self.iter().collect();
        let per_sub: Vec<Option<f64>> = handles
            .par_iter()
            .map(|h| -> Result<Option<f64>> {
                let s = read(h)?;
                if s.component_index(component).is_none() {
                    Ok(None)
                } else {
                    Ok(Some(SubField::sum(&*s, component)?))
                }
            })
            .collect::<Result<_>>()?;
        let defining: Vec<f64> = per_sub.into_iter().flatten().collect();
        if defining.is_empty() {
            return Err(PyrucastError::Message(format!(
                "sum: no sub-field defines component {}",
                component
            )));
        }
        Ok(defining.into_iter().sum())
    }

    /// Squared Euclidean norm `xᵀx = Σ v²` over every value of every zone
    /// (Cast3M `XTX`). Thread-count-dependent to the last ULP.
    fn xtx(&self) -> Result<f64> {
        let mut acc = 0.0;
        for h in self.iter() {
            acc += SubField::xtx(&*read(h)?);
        }
        Ok(acc)
    }

    /// Squared Euclidean norm restricted to the named `components`, summed over
    /// every zone that carries any of them (the aggregate twin of
    /// [`SubField::xtx_components`]). A component may be spread across zones;
    /// each zone contributes the squares of whichever selected components it
    /// holds. Errors if **no** zone defines **any** of `components`.
    fn xtx_components(&self, components: &[&str]) -> Result<f64> {
        let mut acc = 0.0;
        let mut any = false;
        for h in self.iter() {
            let s = read(h)?;
            // A zone missing all selected components contributes nothing;
            // one carrying some contributes their squares.
            if components.iter().any(|c| s.component_index(c).is_some()) {
                acc += s.xtx_components(components)?;
                any = true;
            }
        }
        if !any {
            return Err(PyrucastError::Message(format!(
                "xtx_components: no zone defines any of {:?}",
                components
            )));
        }
        Ok(acc)
    }

    /// Zero-copy view of the zones, for operators doing many reads
    /// (gradient, solver, viz, …): one read guard per sub, data read
    /// **in place** in the store for the lifetime of the view.
    fn view(&self) -> Result<FieldView<Self::Sub>> {
        Ok(FieldView {
            components: self.components()?,
            zones: self.iter().map(read).collect::<Result<_>>()?,
        })
    }

    /// Rebuild the aggregate, mapping each sub-field through `f` (the sub is
    /// read, transformed into a fresh sub, re-inserted). Structure is
    /// preserved — no consolidation.
    fn map_subs(&self, f: impl Fn(&Self::Sub) -> Result<Self::Sub> + Sync + Send) -> Result<Self>
    where
        Self: Sized,
    {
        // Compute the zones in parallel — `read` takes concurrent shared locks,
        // and the per-zone work itself may parallelise further (nested rayon is
        // fine). Store mutation (`insert`) stays **serial and in order** so the
        // result decomposition is identical to the sequential version.
        let handles: Vec<&Handle<Self::Sub>> = self.iter().collect();
        let subs: Vec<Self::Sub> = handles
            .par_iter()
            .map(|h| f(&*read(h)?))
            .collect::<Result<_>>()?;
        let mut out = Self::default();
        for s in subs {
            out.add_sub(insert(s))?;
        }
        Ok(out)
    }

    /// Scalar broadcast: a new field with `op(value, rhs)` applied to every
    /// value of every zone.
    fn combine_scalar(&self, op: fn(f64, f64) -> f64, rhs: f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        self.map_subs(|s| Ok(s.map_all(|v| op(v, rhs))))
    }

    /// A new aggregate with `f` applied to **every** value of every zone —
    /// the unary counterpart of [`Field::combine_scalar`], mirroring
    /// [`SubField::map_all`] at the aggregate level.
    fn map_all(&self, f: impl Fn(f64) -> f64 + Copy + Sync + Send) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        self.map_subs(|s| Ok(s.map_all(f)))
    }

    /// Apply `f` **in place** to the named component, on every zone that
    /// defines it. Errors only if **no** zone defines the component.
    fn map_component(
        &self,
        component: &str,
        f: impl Fn(f64) -> f64 + Copy + Sync + Send,
    ) -> Result<()> {
        let mut found = false;
        for h in self.iter() {
            let mut s = write(h)?;
            if s.component_index(component).is_some() {
                s.map_component(component, f)?;
                found = true;
            }
        }
        if !found {
            return Err(PyrucastError::Message(format!(
                "no sub-field defines component {}",
                component
            )));
        }
        Ok(())
    }

    /// Add `scalar` to the named component, in place, on every zone defining it.
    fn add_to_component(&self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, move |v| v + scalar)
    }

    /// Subtract `scalar` from the named component, in place, on every zone.
    fn sub_to_component(&self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, move |v| v - scalar)
    }

    /// Multiply the named component by `scalar`, in place, on every zone.
    fn mul_to_component(&self, component: &str, scalar: f64) -> Result<()> {
        self.map_component(component, move |v| v * scalar)
    }

    /// Divide the named component by `scalar`, in place, on every zone.
    /// Errors if `scalar` is zero.
    fn div_to_component(&self, component: &str, scalar: f64) -> Result<()> {
        if scalar == 0.0 {
            return Err(PyrucastError::Message(
                "div_to_component: division by zero".into(),
            ));
        }
        self.map_component(component, move |v| v / scalar)
    }

    /// Binary combination of two fields, **per (support, component)** with
    /// union/passthrough semantics. The two fields need **not** share the same
    /// zone decomposition:
    ///
    /// - The output covers the **union** of the operands' supports.
    /// - On a support carried by both, the zones are combined component-wise
    ///   ([`SubField::merge_components`]): a component defined on both sides
    ///   becomes `op(self, other)`; a component on only one side **passes
    ///   through unchanged** (raw, for every operator).
    /// - A support carried by only one side has its zone(s) **passed through**
    ///   unchanged.
    ///
    /// Operands are assumed to satisfy the field invariant (at most one zone per
    /// `(support, component)`, as a union enforces), so each support pairs at
    /// most one zone per side; the result inherits the invariant by
    /// construction.
    fn merge_field(&self, other: &Self, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        // Snapshot both sides' zones (clone out of the store) so we never hold
        // two guards at once — safe even when `other` shares handles with
        // `self`.
        let lefts: Vec<Self::Sub> = self
            .iter()
            .map(|h| read(h).map(|g| (*g).clone()))
            .collect::<Result<_>>()?;
        let rights: Vec<Self::Sub> = other
            .iter()
            .map(|h| read(h).map(|g| (*g).clone()))
            .collect::<Result<_>>()?;

        let mut out = Self::default();
        let mut right_used = vec![false; rights.len()];
        // Each left zone: combine with the right zone on the same support if
        // any, else pass through unchanged.
        for l in &lefts {
            match rights.iter().position(|r| l.same_support(r)) {
                Some(j) => {
                    out.add_sub(insert(l.merge_components(&rights[j], op)?))?;
                    right_used[j] = true;
                }
                None => out.add_sub(insert(l.clone()))?,
            }
        }
        // Right zones whose support was absent on the left: pass through.
        for (j, r) in rights.iter().enumerate() {
            if !right_used[j] {
                out.add_sub(insert(r.clone()))?;
            }
        }
        Ok(out)
    }

    /// Scalar product of two fields, summed over the zones they **share** by
    /// support. Mirrors [`SubField::dot`]'s union spirit at the aggregate level:
    /// each support carried by both sides contributes its [`SubField::dot`]
    /// (shared components only); a support — or component — present on only one
    /// side has no counterpart and contributes nothing. The reduction behind
    /// Cast3M's `XTY`/`PSCA`.
    fn dot_field(&self, other: &Self) -> Result<f64>
    where
        Self::Sub: Clone,
    {
        // Snapshot other's zones so we never hold two guards at once — safe
        // even when `other` shares handles with `self` (mirrors merge_field).
        let others: Vec<Self::Sub> = other
            .iter()
            .map(|h| read(h).map(|g| (*g).clone()))
            .collect::<Result<_>>()?;
        let mut acc = 0.0;
        for h in self.iter() {
            let s = read(h)?;
            // At most one right zone shares the support (field invariant).
            if let Some(os) = others.iter().find(|os| s.same_support(os)) {
                acc += s.dot(os)?;
            }
        }
        Ok(acc)
    }

    /// Per-node/-point scalar product of two fields — Cast3M's `PSCA`. Each zone
    /// of `self` carried on a support **shared** with `other` is reduced over
    /// its shared components ([`SubField::pscal`]); the result is a new field
    /// with one `"psca"` zone per shared support. Mirrors [`Field::dot_field`]'s
    /// union spirit: a support (or component) present on only one side has no
    /// counterpart and yields no contribution.
    fn pscal_field(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        // Snapshot other's zones so we never hold two guards at once — safe
        // even when `other` shares handles with `self` (mirrors merge_field).
        let others: Vec<Self::Sub> = other
            .iter()
            .map(|h| read(h).map(|g| (*g).clone()))
            .collect::<Result<_>>()?;
        let mut out = Self::default();
        for h in self.iter() {
            let s = read(h)?;
            // At most one right zone shares the support (field invariant).
            if let Some(os) = others.iter().find(|os| s.same_support(os)) {
                out.add_sub(insert(s.pscal(os)?))?;
            }
        }
        Ok(out)
    }

    /// Targeted update: merge `sub` into the zone(s) of `self` sharing its
    /// support ([`SubField::merge_components`], union of components with
    /// passthrough); every other zone is carried **unchanged**. Errors if no
    /// zone shares the support.
    fn merge_subfield(&self, sub: &Self::Sub, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        let mut out = Self::default();
        let mut matched = false;
        for h in self.iter() {
            let s = read(h)?;
            if s.same_support(sub) {
                out.add_sub(insert(s.merge_components(sub, op)?))?;
                matched = true;
            } else {
                out.add_sub(insert((*s).clone()))?;
            }
        }
        if !matched {
            return Err(PyrucastError::Message(
                "merge_subfield: no zone shares the sub-field's support".into(),
            ));
        }
        Ok(out)
    }
}

impl<A: Aggregate> Field for A where A::Sub: SubField {}

/// Uniform element-wise unary map over **either** a single zone (`Sub*Field`)
/// or an aggregate (`*Field`), returning a new field of the same type.
///
/// It is the minimal bridge that lets a generic operator (e.g.
/// [`crate::ops::field::cos`]) accept both flavours: [`SubField`] and
/// [`Field`] are sibling traits with no common ancestor, so a single generic
/// bound needs this trait. Each impl simply delegates to the matching
/// `map_all` ([`SubField::map_all`] / [`Field::map_all`]); the method keeps a
/// distinct name to avoid colliding with those inherent `map_all` methods at
/// concrete call sites.
pub trait MapValues: Sized {
    /// A new field with `f` applied to every value.
    fn map_values(&self, f: fn(f64) -> f64) -> Result<Self>;
}

impl MapValues for SubNodeField {
    fn map_values(&self, f: fn(f64) -> f64) -> Result<Self> {
        Ok(self.map_all(f))
    }
}

impl MapValues for SubElementField {
    fn map_values(&self, f: fn(f64) -> f64) -> Result<Self> {
        Ok(self.map_all(f))
    }
}

impl MapValues for NodeField {
    fn map_values(&self, f: fn(f64) -> f64) -> Result<Self> {
        self.map_all(f)
    }
}

impl MapValues for ElementField {
    fn map_values(&self, f: fn(f64) -> f64) -> Result<Self> {
        self.map_all(f)
    }
}

/// Fold `op` over one component across every sub that defines it.
fn fold_subs<A>(agg: &A, component: &str, op_name: &str, op: fn(f64, f64) -> f64) -> Result<f64>
where
    A: Aggregate,
    A::Sub: SubField,
{
    // Per-zone fold in parallel (each zone fold is itself parallel), then a
    // serial associative combine. `op` (min/max) is associative & commutative
    // ⇒ thread-count-independent.
    let handles: Vec<&Handle<A::Sub>> = agg.iter().collect();
    let per_sub: Vec<Option<f64>> = handles
        .par_iter()
        .map(|h| -> Result<Option<f64>> {
            let s = read(h)?;
            if s.component_index(component).is_none() {
                Ok(None)
            } else {
                Ok(Some(fold_component(&*s, component, op_name, op)?))
            }
        })
        .collect::<Result<_>>()?;
    let acc = per_sub.into_iter().flatten().reduce(op);
    acc.ok_or_else(|| {
        PyrucastError::Message(format!(
            "{}: no sub-field defines component {}",
            op_name, component
        ))
    })
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::store::{insert, write, Handle};

    fn make_node_field(values: &[f64]) -> SubNodeField {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..values.len())
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
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
        f
    }

    #[test]
    fn subfield_min_max_on_node_field() {
        let f = make_node_field(&[4.0, -1.5, 2.0]);
        assert_eq!(SubField::min(&f, "T").unwrap(), -1.5);
        assert_eq!(SubField::max(&f, "T").unwrap(), 4.0);
    }

    #[test]
    fn subfield_min_max_unknown_component_errors() {
        let f = make_node_field(&[1.0]);
        assert!(SubField::min(&f, "missing").is_err());
        assert!(SubField::max(&f, "missing").is_err());
    }

    #[test]
    fn subfield_min_max_isolates_components() {
        // Two components: min/max must stride over the right offsets.
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            insert(sm)
        };
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set(0, 0, 10.0).unwrap();
        f.set(0, 1, -10.0).unwrap();
        f.set(1, 0, 20.0).unwrap();
        f.set(1, 1, -20.0).unwrap();
        assert_eq!(SubField::min(&f, "U").unwrap(), 10.0);
        assert_eq!(SubField::max(&f, "U").unwrap(), 20.0);
        assert_eq!(SubField::min(&f, "V").unwrap(), -20.0);
        assert_eq!(SubField::max(&f, "V").unwrap(), -10.0);
    }

    #[test]
    fn subfield_min_on_empty_support_errors() {
        let coords = insert(Coords::new(1).unwrap());
        let sm: Handle<SubMesh> = insert(SubMesh::new(coords, ElementType::POI1));
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(SubField::min(&f, "T").is_err());
    }

    #[test]
    fn subfield_sum_and_xtx() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            insert(sm)
        };
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set(0, 0, 10.0).unwrap();
        f.set(0, 1, -10.0).unwrap();
        f.set(1, 0, 20.0).unwrap();
        f.set(1, 1, -20.0).unwrap();
        assert_eq!(SubField::sum(&f, "U").unwrap(), 30.0);
        assert_eq!(SubField::sum(&f, "V").unwrap(), -30.0);
        // xtx = 10² + (-10)² + 20² + (-20)² = 1000.
        assert_eq!(SubField::xtx(&f), 1000.0);
        assert!(SubField::sum(&f, "nope").is_err());
    }

    #[test]
    fn subfield_sum_on_empty_support_is_zero() {
        let coords = insert(Coords::new(1).unwrap());
        let sm: Handle<SubMesh> = insert(SubMesh::new(coords, ElementType::POI1));
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert_eq!(SubField::sum(&f, "T").unwrap(), 0.0);
        assert_eq!(SubField::xtx(&f), 0.0);
    }

    fn make_two_zone_element_field() -> ElementField {
        let coords = insert(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(coords, ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_qua).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        ElementField::with(&fes, &[vec!["k".into()], vec!["E".into(), "k".into()]]).unwrap()
    }

    #[test]
    fn field_components_union_first_seen_order() {
        let ef = make_two_zone_element_field();
        assert_eq!(Field::components(&ef).unwrap(), vec!["k", "E"]);
    }

    #[test]
    fn field_min_max_fold_across_subs() {
        let ef = make_two_zone_element_field();
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("k", 3.0)
            .unwrap();
        {
            let mut s = write(&ef.get(1).unwrap()).unwrap();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 210e9).unwrap();
        }
        assert_eq!(Field::min(&ef, "k").unwrap(), -2.0);
        assert_eq!(Field::max(&ef, "k").unwrap(), 3.0);
        // E exists on the second zone only: folded over that zone alone.
        assert_eq!(Field::min(&ef, "E").unwrap(), 210e9);
        assert_eq!(Field::max(&ef, "E").unwrap(), 210e9);
    }

    #[test]
    fn field_min_unknown_component_errors() {
        let ef = make_two_zone_element_field();
        assert!(Field::min(&ef, "missing").is_err());
        assert!(Field::max(&ef, "missing").is_err());
    }

    #[test]
    fn field_sum_and_xtx_fold_across_subs() {
        let ef = make_two_zone_element_field();
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("k", 3.0)
            .unwrap();
        {
            let mut s = write(&ef.get(1).unwrap()).unwrap();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 5.0).unwrap();
        }
        // Field-level folds equal the sum of the per-zone reductions (no need to
        // know the Gauss-point counts).
        let z0 = SubField::sum(&*read(&ef.get(0).unwrap()).unwrap(), "k").unwrap();
        let z1 = SubField::sum(&*read(&ef.get(1).unwrap()).unwrap(), "k").unwrap();
        assert!((Field::sum(&ef, "k").unwrap() - (z0 + z1)).abs() < 1e-12);
        // E lives on zone 1 only.
        let e1 = SubField::sum(&*read(&ef.get(1).unwrap()).unwrap(), "E").unwrap();
        assert!((Field::sum(&ef, "E").unwrap() - e1).abs() < 1e-12);
        // xtx over the whole field = Σ of the per-zone xtx.
        let x0 = SubField::xtx(&*read(&ef.get(0).unwrap()).unwrap());
        let x1 = SubField::xtx(&*read(&ef.get(1).unwrap()).unwrap());
        assert!((Field::xtx(&ef).unwrap() - (x0 + x1)).abs() < 1e-9);
        assert!(Field::sum(&ef, "missing").is_err());
    }

    #[test]
    fn subfield_xtx_components_selects_a_subset() {
        let (sm, _) = poi1_support(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set(0, 0, 3.0).unwrap(); // U
        f.set(0, 1, 4.0).unwrap(); // V
        f.set(1, 0, 0.0).unwrap();
        f.set(1, 1, 12.0).unwrap();
        // Whole field: 3² + 4² + 0² + 12² = 169.
        assert_eq!(SubField::xtx(&f), 169.0);
        // U only: 3² + 0² = 9.
        assert_eq!(f.xtx_components(&["U"]).unwrap(), 9.0);
        // V only: 4² + 12² = 160.
        assert_eq!(f.xtx_components(&["V"]).unwrap(), 160.0);
        // Both selected ⇒ same as the whole field.
        assert_eq!(f.xtx_components(&["U", "V"]).unwrap(), 169.0);
        // An unknown component alongside a known one is ignored, not an error.
        assert_eq!(f.xtx_components(&["U", "nope"]).unwrap(), 9.0);
        // None present ⇒ error.
        assert!(f.xtx_components(&["nope"]).is_err());
    }

    #[test]
    fn field_xtx_components_folds_selected_across_zones() {
        let ef = make_two_zone_element_field(); // zone0: [k], zone1: [E, k]
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("k", 3.0)
            .unwrap();
        {
            let mut s = write(&ef.get(1).unwrap()).unwrap();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 5.0).unwrap();
        }
        // "k" lives on both zones: Σ of the per-zone k-only xtx.
        let k0 = read(&ef.get(0).unwrap())
            .unwrap()
            .xtx_components(&["k"])
            .unwrap();
        let k1 = read(&ef.get(1).unwrap())
            .unwrap()
            .xtx_components(&["k"])
            .unwrap();
        assert!((ef.xtx_components(&["k"]).unwrap() - (k0 + k1)).abs() < 1e-9);
        // "E" lives on zone 1 only; zone 0 (no E) is skipped, not an error.
        let e1 = read(&ef.get(1).unwrap())
            .unwrap()
            .xtx_components(&["E"])
            .unwrap();
        assert!((ef.xtx_components(&["E"]).unwrap() - e1).abs() < 1e-9);
        // No zone defines "missing" ⇒ error.
        assert!(ef.xtx_components(&["missing"]).is_err());
    }

    // ─── SubField::merge_components / check_same_components ────────────────────

    /// POI1 support over `n` nodes, plus the nodes (for `value(nid, …)`).
    fn poi1_support(n: usize) -> (Handle<SubMesh>, Vec<Node>) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            for nd in &nodes {
                sm.add_cell(&[nd.id()]).unwrap();
            }
            insert(sm)
        };
        (sm, nodes)
    }

    #[test]
    fn subfield_merge_adds_same_support() {
        let (sm, _) = poi1_support(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 0, 2.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        g.set(0, 0, 10.0).unwrap();
        g.set(1, 0, 20.0).unwrap();
        let s = f.merge_components(&g, |a, b| a + b).unwrap();
        assert_eq!(s.get(0, 0).unwrap(), 11.0);
        assert_eq!(s.get(1, 0).unwrap(), 22.0);
    }

    #[test]
    fn subfield_merge_aligns_components_by_name() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 1.0).unwrap();
        f.set_value(nodes[0].id(), "V", 2.0).unwrap();
        // Other carries the same names in the opposite order.
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "U".into()]).unwrap();
        g.set_value(nodes[0].id(), "U", 40.0).unwrap();
        g.set_value(nodes[0].id(), "V", 30.0).unwrap();
        let s = f.merge_components(&g, |a, b| a + b).unwrap();
        assert_eq!(s.value(nodes[0].id(), "U").unwrap(), 41.0);
        assert_eq!(s.value(nodes[0].id(), "V").unwrap(), 32.0);
    }

    #[test]
    fn subfield_merge_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!(f.merge_components(&g, |a, b| a + b).is_err());
    }

    #[test]
    fn subfield_merge_div_by_zero_is_inf() {
        let (sm, _) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        let g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap(); // zero
        let s = f.merge_components(&g, |a, b| a / b).unwrap();
        assert!(s.get(0, 0).unwrap().is_infinite());
    }

    #[test]
    fn check_same_components_accepts_equal_rejects_mismatch() {
        let (sm, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        // Same support, same set (order differs): ok.
        let g = SubNodeField::from_poi1(&sm, vec!["V".into(), "U".into()]).unwrap();
        f.check_same_components(&g).unwrap();
        // Same support, different set: error.
        let h = SubNodeField::from_poi1(&sm, vec!["U".into()]).unwrap();
        assert!(f.check_same_components(&h).is_err());
        // Different support: error.
        let k = SubNodeField::from_poi1(&sm_b, vec!["U".into(), "V".into()]).unwrap();
        assert!(f.check_same_components(&k).is_err());
    }

    #[test]
    fn subfield_merge_components_unions_with_passthrough() {
        // f has [U, V], g has [V, W] on the same support: V combines, U and W
        // pass through raw.
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 1.0).unwrap();
        f.set_value(nodes[0].id(), "V", 2.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "W".into()]).unwrap();
        g.set_value(nodes[0].id(), "V", 20.0).unwrap();
        g.set_value(nodes[0].id(), "W", 7.0).unwrap();
        let s = f.merge_components(&g, |a, b| a + b).unwrap();
        assert_eq!(s.components(), &["U".to_string(), "V".into(), "W".into()]);
        assert_eq!(s.value(nodes[0].id(), "U").unwrap(), 1.0, "f-only: raw");
        assert_eq!(s.value(nodes[0].id(), "V").unwrap(), 22.0, "shared: op");
        assert_eq!(s.value(nodes[0].id(), "W").unwrap(), 7.0, "g-only: raw");
    }

    #[test]
    fn subfield_operator_uses_union_passthrough() {
        // The zone-level `&f + &g` operator is now union/passthrough, no longer
        // strict on component sets.
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set_value(nodes[0].id(), "T", 5.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        g.set_value(nodes[0].id(), "P", 9.0).unwrap();
        // Disjoint components: both pass through, no error.
        let s = (&f + &g).unwrap();
        assert_eq!(s.value(nodes[0].id(), "T").unwrap(), 5.0);
        assert_eq!(s.value(nodes[0].id(), "P").unwrap(), 9.0);
    }

    #[test]
    fn subfield_merge_components_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!(f.merge_components(&g, |a, b| a + b).is_err());
    }

    // ─── SubField::dot (scalar product, strict) ──────────────────────────────

    #[test]
    fn subfield_dot_sums_products() {
        let (sm, _) = poi1_support(3);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        for (i, &(a, b)) in [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)].iter().enumerate() {
            f.set(i, 0, a).unwrap();
            g.set(i, 0, b).unwrap();
        }
        // 1·4 + 2·5 + 3·6 = 32
        assert_eq!(f.dot(&g).unwrap(), 32.0);
    }

    #[test]
    fn subfield_dot_aligns_components_by_name() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 2.0).unwrap();
        f.set_value(nodes[0].id(), "V", 3.0).unwrap();
        // Other carries the same names in the opposite order.
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "U".into()]).unwrap();
        g.set_value(nodes[0].id(), "U", 10.0).unwrap();
        g.set_value(nodes[0].id(), "V", 100.0).unwrap();
        // 2·10 (U) + 3·100 (V) = 320
        assert_eq!(f.dot(&g).unwrap(), 320.0);
    }

    #[test]
    fn subfield_dot_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!(f.dot(&g).is_err());
    }

    #[test]
    fn subfield_dot_disjoint_components_is_zero() {
        // Same support, no shared component: nothing to multiply ⇒ 0.
        let (sm, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        assert_eq!(f.dot(&g).unwrap(), 0.0);
    }

    #[test]
    fn subfield_dot_partial_components_uses_shared_only() {
        // f has [U, V], g has [V] on the same support: only V contributes.
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 2.0).unwrap();
        f.set_value(nodes[0].id(), "V", 3.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into()]).unwrap();
        g.set_value(nodes[0].id(), "V", 10.0).unwrap();
        // Only V: 3·10 = 30 (U has no counterpart).
        assert_eq!(f.dot(&g).unwrap(), 30.0);
    }

    #[test]
    fn field_dot_sums_across_zones() {
        use crate::containers::node_field::NodeField;
        // Two POI1 zones sharing one Coords: dot_field must pair each zone by
        // support and sum both contributions.
        let coords = insert(Coords::new(1).unwrap());
        let node = |x: f64| Node::create_in(coords.clone(), &[x]).unwrap();
        let poi1 = |ns: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in ns {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        let (na, nb, nc) = (node(0.0), node(1.0), node(2.0));
        let sm_a = poi1(&[&na, &nb]);
        let sm_b = poi1(&[&nc]);
        let mut za = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        za.set(0, 0, 2.0).unwrap();
        za.set(1, 0, 3.0).unwrap();
        let mut zb = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        zb.set(0, 0, 5.0).unwrap();
        let mut nf = NodeField::from_sub(za);
        nf.add_sub(insert(zb)).unwrap();
        // Dot with itself = Σ value²: 2² + 3² (zone A) + 5² (zone B) = 38.
        assert_eq!(nf.dot_field(&nf).unwrap(), 38.0);
    }

    #[test]
    fn field_dot_disjoint_supports_is_zero() {
        use crate::containers::node_field::NodeField;
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let a = NodeField::from_sub(SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap());
        let b = NodeField::from_sub(SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap());
        // No zone of `a` shares a support with `b`: nothing to pair ⇒ 0.
        assert_eq!(a.dot_field(&b).unwrap(), 0.0);
    }

    // ─── SubField::pscal (per-node scalar product) ───────────────────────────

    #[test]
    fn subfield_pscal_reduces_over_components_per_node() {
        let (sm, nodes) = poi1_support(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set_value(nodes[0].id(), "UX", 1.0).unwrap();
        f.set_value(nodes[0].id(), "UY", 2.0).unwrap();
        f.set_value(nodes[1].id(), "UX", 3.0).unwrap();
        f.set_value(nodes[1].id(), "UY", 4.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        g.set_value(nodes[0].id(), "UX", 10.0).unwrap();
        g.set_value(nodes[0].id(), "UY", 20.0).unwrap();
        g.set_value(nodes[1].id(), "UX", 30.0).unwrap();
        g.set_value(nodes[1].id(), "UY", 40.0).unwrap();
        let p = f.pscal(&g).unwrap();
        assert_eq!(p.components(), &["psca".to_string()]);
        // node 0: 1·10 + 2·20 = 50 ; node 1: 3·30 + 4·40 = 250
        assert_eq!(p.value(nodes[0].id(), "psca").unwrap(), 50.0);
        assert_eq!(p.value(nodes[1].id(), "psca").unwrap(), 250.0);
    }

    #[test]
    fn subfield_pscal_aligns_components_by_name() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 2.0).unwrap();
        f.set_value(nodes[0].id(), "V", 3.0).unwrap();
        // Same names, opposite order.
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "U".into()]).unwrap();
        g.set_value(nodes[0].id(), "U", 10.0).unwrap();
        g.set_value(nodes[0].id(), "V", 100.0).unwrap();
        let p = f.pscal(&g).unwrap();
        // 2·10 (U) + 3·100 (V) = 320
        assert_eq!(p.value(nodes[0].id(), "psca").unwrap(), 320.0);
    }

    #[test]
    fn subfield_pscal_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!(f.pscal(&g).is_err());
    }

    #[test]
    fn subfield_pscal_uses_shared_components_only() {
        // f has [U, V], g has [V, W] on the same support: only V contributes.
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 2.0).unwrap();
        f.set_value(nodes[0].id(), "V", 3.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "W".into()]).unwrap();
        g.set_value(nodes[0].id(), "V", 10.0).unwrap();
        g.set_value(nodes[0].id(), "W", 5.0).unwrap();
        let p = f.pscal(&g).unwrap();
        // Only V: 3·10 = 30 (U and W have no counterpart).
        assert_eq!(p.value(nodes[0].id(), "psca").unwrap(), 30.0);
    }

    #[test]
    fn field_pscal_field_one_zone_per_zone() {
        use crate::containers::node_field::NodeField;
        let coords = insert(Coords::new(1).unwrap());
        let node = |x: f64| Node::create_in(coords.clone(), &[x]).unwrap();
        let poi1 = |ns: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in ns {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        let (na, nb) = (node(0.0), node(1.0));
        let sm_a = poi1(&[&na]);
        let sm_b = poi1(&[&nb]);
        let mut za = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        za.set(0, 0, 3.0).unwrap();
        let mut zb = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        zb.set(0, 0, 5.0).unwrap();
        let mut nf = NodeField::from_sub(za);
        nf.add_sub(insert(zb)).unwrap();
        // pscal with itself → per-node square, two zones preserved.
        let p = nf.pscal_field(&nf).unwrap();
        let view = p.view().unwrap();
        assert_eq!(view.value(na.id(), "psca").unwrap(), 9.0);
        assert_eq!(view.value(nb.id(), "psca").unwrap(), 25.0);
    }

    // ─── Field-level arithmetic ──────────────────────────────────────────────

    /// Single-zone Lagrange-1 FE space on one TRI3 cell.
    fn one_tri3_fes() -> FiniteElementSpace {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm).unwrap();
        FiniteElementSpace::lagrange1(&mesh).unwrap()
    }

    #[test]
    fn field_combine_scalar_hits_every_zone() {
        let ef = make_two_zone_element_field();
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("k", 1.0)
            .unwrap();
        write(&ef.get(1).unwrap())
            .unwrap()
            .set_uniform("k", 2.0)
            .unwrap();
        let out = ef.combine_scalar(|a, b| a + b, 10.0).unwrap();
        assert_eq!(
            read(&out.get(0).unwrap())
                .unwrap()
                .value(0, 0, "k")
                .unwrap(),
            11.0
        );
        assert_eq!(
            read(&out.get(1).unwrap())
                .unwrap()
                .value(0, 0, "k")
                .unwrap(),
            12.0
        );
    }

    #[test]
    fn field_add_to_component_present_zones_only() {
        let ef = make_two_zone_element_field(); // zone0 ["k"], zone1 ["E","k"]
        write(&ef.get(1).unwrap())
            .unwrap()
            .set_uniform("E", 100.0)
            .unwrap();
        ef.add_to_component("E", 1.0).unwrap(); // E only on zone 1
        assert_eq!(
            read(&ef.get(1).unwrap()).unwrap().value(0, 0, "E").unwrap(),
            101.0
        );
        assert!(ef.add_to_component("missing", 1.0).is_err());
    }

    #[test]
    fn field_merge_field_same_decomposition() {
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into()]).unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 3.0)
            .unwrap();
        write(&g.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 4.0)
            .unwrap();
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        let z = read(&s.get(0).unwrap()).unwrap();
        for gp in 0..z.gauss_count() {
            assert_eq!(z.value(0, gp, "E").unwrap(), 7.0);
        }
    }

    #[test]
    fn field_merge_field_disjoint_supports_unions() {
        // Distinct supports ⇒ union: both zones pass through unchanged.
        let f = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        let g = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 3.0)
            .unwrap();
        write(&g.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 4.0)
            .unwrap();
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        assert_eq!(s.len(), 2, "distinct supports ⇒ two zones");
        assert_eq!(
            read(&s.get(0).unwrap()).unwrap().value(0, 0, "E").unwrap(),
            3.0
        );
        assert_eq!(
            read(&s.get(1).unwrap()).unwrap().value(0, 0, "E").unwrap(),
            4.0
        );
    }

    #[test]
    fn field_merge_field_partial_components_passes_through() {
        // Same support: f has [E], g has [E, nu]. E combines, nu passes through.
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 3.0)
            .unwrap();
        {
            let mut z = write(&g.get(0).unwrap()).unwrap();
            z.set_uniform("E", 4.0).unwrap();
            z.set_uniform("nu", 0.3).unwrap();
        }
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        assert_eq!(s.len(), 1);
        let z = read(&s.get(0).unwrap()).unwrap();
        assert_eq!(z.components(), &["E".to_string(), "nu".to_string()]);
        assert_eq!(z.value(0, 0, "E").unwrap(), 7.0, "shared: op applied");
        assert_eq!(z.value(0, 0, "nu").unwrap(), 0.3, "g-only: passthrough");
    }

    #[test]
    fn field_merge_field_subtraction_passthrough_is_raw() {
        // b-only component under subtraction passes through raw (b, not -b).
        let fes = one_tri3_fes();
        let a = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let b = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
        write(&a.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 10.0)
            .unwrap();
        {
            let mut z = write(&b.get(0).unwrap()).unwrap();
            z.set_uniform("E", 4.0).unwrap();
            z.set_uniform("nu", 0.3).unwrap();
        }
        let s = a.merge_field(&b, |x, y| x - y).unwrap();
        let z = read(&s.get(0).unwrap()).unwrap();
        assert_eq!(z.value(0, 0, "E").unwrap(), 6.0);
        assert_eq!(
            z.value(0, 0, "nu").unwrap(),
            0.3,
            "raw passthrough, not -0.3"
        );
    }

    #[test]
    fn field_merge_subfield_targets_matching_zone() {
        let ef = make_two_zone_element_field(); // zone0 ["k"], zone1 ["E","k"]
        write(&ef.get(0).unwrap())
            .unwrap()
            .set_uniform("k", 1.0)
            .unwrap();
        {
            let mut z1 = write(&ef.get(1).unwrap()).unwrap();
            z1.set_uniform("k", 2.0).unwrap();
            z1.set_uniform("E", 5.0).unwrap();
        }
        // A sub on zone 0's support, k = 10.
        let mut sub = (*read(&ef.get(0).unwrap()).unwrap()).clone();
        sub.set_uniform("k", 10.0).unwrap();
        let out = ef.merge_subfield(&sub, |a, b| a + b).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            read(&out.get(0).unwrap())
                .unwrap()
                .value(0, 0, "k")
                .unwrap(),
            11.0
        );
        let z1 = read(&out.get(1).unwrap()).unwrap();
        assert_eq!(z1.value(0, 0, "k").unwrap(), 2.0);
        assert_eq!(z1.value(0, 0, "E").unwrap(), 5.0);
    }

    #[test]
    fn field_merge_subfield_no_match_errors() {
        let ef = make_two_zone_element_field();
        // A sub on a brand-new, unrelated FE support.
        let other = ElementField::new(&one_tri3_fes(), vec!["k".into()]).unwrap();
        let sub = (*read(&other.get(0).unwrap()).unwrap()).clone();
        assert!(ef.merge_subfield(&sub, |a, b| a + b).is_err());
    }

    // ─── Arithmetic operator sugar (`+ - * /`) ───────────────────────────────

    #[test]
    fn subfield_operators_field_and_scalar() {
        let (sm, _) = poi1_support(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 0, 2.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        g.set(0, 0, 10.0).unwrap();
        g.set(1, 0, 20.0).unwrap();

        // field ∘ field (returns Result) — every operator.
        let add = (&f + &g).unwrap();
        assert_eq!(add.get(0, 0).unwrap(), 11.0);
        assert_eq!(add.get(1, 0).unwrap(), 22.0);
        let sub = (&g - &f).unwrap();
        assert_eq!(sub.get(1, 0).unwrap(), 18.0);
        let mul = (&f * &g).unwrap();
        assert_eq!(mul.get(1, 0).unwrap(), 40.0);
        let div = (&g / &f).unwrap();
        assert_eq!(div.get(1, 0).unwrap(), 10.0);

        // Owned operands also work (`a + b`).
        let owned = (f.clone() + g.clone()).unwrap();
        assert_eq!(owned.get(0, 0).unwrap(), 11.0);

        // field ∘ scalar (pre-existing infallible sugar) stays usable.
        let plus_ten = &f + 10.0;
        assert_eq!(plus_ten.get(1, 0).unwrap(), 12.0);
    }

    #[test]
    fn subfield_operator_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!((&f + &g).is_err());
    }

    #[test]
    fn node_field_operators_field_and_scalar() {
        use crate::containers::node_field::NodeField;
        let (sm, nodes) = poi1_support(2);
        let mut fa = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        fa.set(0, 0, 3.0).unwrap();
        fa.set(1, 0, 4.0).unwrap();
        let mut fb = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        fb.set(0, 0, 1.0).unwrap();
        fb.set(1, 0, 2.0).unwrap();
        let a = NodeField::from_sub(fa);
        let b = NodeField::from_sub(fb);

        // field ∘ field, zone by zone.
        let sum = (&a + &b).unwrap();
        let view = sum.view().unwrap();
        assert_eq!(view.value(nodes[0].id(), "T").unwrap(), 4.0);

        // field ∘ scalar broadcast.
        let doubled = (&a * 2.0).unwrap();
        let v2 = doubled.view().unwrap();
        assert_eq!(v2.value(nodes[0].id(), "T").unwrap(), 6.0);
    }

    #[test]
    fn element_field_operators_field_and_scalar() {
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into()]).unwrap();
        write(&f.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 3.0)
            .unwrap();
        write(&g.get(0).unwrap())
            .unwrap()
            .set_uniform("E", 4.0)
            .unwrap();

        let s = (&f + &g).unwrap();
        let z = read(&s.get(0).unwrap()).unwrap();
        assert_eq!(z.value(0, 0, "E").unwrap(), 7.0);

        let scaled = (&f - 1.0).unwrap();
        let zs = read(&scaled.get(0).unwrap()).unwrap();
        assert_eq!(zs.value(0, 0, "E").unwrap(), 2.0);
    }
}
