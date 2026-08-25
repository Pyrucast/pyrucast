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
//! use pyrucast::atoms::{ElementType, Node};
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::coords::Coords;
//! use pyrucast::containers::node_field::SubNodeField;
//! use pyrucast::handle::Handle;
//!
//! let coords = Handle::new(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let sm = {
//!     let mut sm = SubMesh::new(coords, ElementType::POI1);
//!     sm.add_cell(&[a.id()]).unwrap();
//!     sm.add_cell(&[b.id()]).unwrap();
//!     Handle::new(sm)
//! };
//! let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
//! f.set(0, 0, -3.0).unwrap();
//! f.set(1, 0, 7.5).unwrap();
//! assert_eq!(SubField::min(&f, Some("T")).unwrap(), -3.0);
//! assert_eq!(SubField::max(&f, Some("T")).unwrap(), 7.5);
//! // Sans composante nommée, la réduction porte sur tout le champ.
//! assert_eq!(SubField::min(&f, None).unwrap(), -3.0);
//! ```

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::handle::{Handle, ReadGuard};
use crate::parallel::*;
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

/// Accepts either a **single** component name or a **collection** of names, so
/// the component operators read the same both ways:
/// `field.filter_components("u_x")` and `field.filter_components(["u_x", "u_y"])`
/// (and `field.filter_components(model.primal_vars())`, a `Vec<String>`). It is
/// the Rust twin of the Python `str | list[str]` argument.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// // Le jumeau Rust de l'argument Python `str | list[str]` : les
/// // opérateurs de composantes se lisent pareil dans les deux sens.
/// assert_eq!(temp.filter_components("T")?.components()?, vec!["T".to_string()]);
/// assert_eq!(temp.filter_components(["T"])?.components()?, vec!["T".to_string()]);
/// assert_eq!(temp.filter_components(vec!["T".to_string()])?.components()?,
///            vec!["T".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait IntoComponentNames {
    /// The requested component names, as an owned list.
    fn into_names(self) -> Vec<String>;
}

impl IntoComponentNames for Vec<String> {
    fn into_names(self) -> Vec<String> {
        self
    }
}
impl IntoComponentNames for &[String] {
    fn into_names(self) -> Vec<String> {
        self.to_vec()
    }
}
impl IntoComponentNames for String {
    fn into_names(self) -> Vec<String> {
        vec![self]
    }
}
impl IntoComponentNames for &str {
    fn into_names(self) -> Vec<String> {
        vec![self.to_string()]
    }
}
impl IntoComponentNames for &String {
    fn into_names(self) -> Vec<String> {
        vec![self.clone()]
    }
}
impl IntoComponentNames for Vec<&str> {
    fn into_names(self) -> Vec<String> {
        self.into_iter().map(str::to_string).collect()
    }
}
impl IntoComponentNames for &[&str] {
    fn into_names(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}
impl<const N: usize> IntoComponentNames for [&str; N] {
    fn into_names(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

/// Zero-copy view of a field aggregate's zones: one owned read guard
/// per sub-field plus the union of the component names, built by
/// [`Field::view`]. Holding the view keeps a shared lock on every sub:
/// concurrent reads are free, writes wait until the view is dropped.
///
/// The kind-specific reading methods live next to each concrete sub
/// type, on the `NodeFieldView` and `ElementFieldView` aliases.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// // Une vue en lecture sur **toutes** les zones à la fois : les guards
/// // sont pris une fois, et les lectures concurrentes ne s'attendent pas.
/// let vue = temp.view()?;
/// assert_eq!(vue.components(), &["T".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct FieldView<S: Any + Send + Sync> {
    pub(crate) zones: Vec<ReadGuard<S>>,
    components: Vec<String>,
}

impl<S: Any + Send + Sync> FieldView<S> {
    /// Union of the zones' component names, first-seen order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::field::{Field, SubField};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::node_field::NodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let support = mesh::poi1_from_nodes(&n).unwrap();
    /// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
    /// #                                    vec!["T".into()]).unwrap();
    /// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
    /// // L'union des composantes des zones, dans l'ordre de première
    /// // apparition — la même règle que partout ailleurs dans les agrégats.
    /// assert_eq!(temp.view()?.components(), &["T".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// // Un contrat purement **structurel** : des composantes nommées et un
/// // tampon plat où l'index de composante varie le plus vite. Champs
/// // nodaux et champs par éléments le satisfont tous deux, d'où une
/// // arithmétique écrite une seule fois.
/// let z = temp.get(0)?;
/// let z = z.read();
/// assert_eq!(z.components(), &["T".to_string()]);
/// assert_eq!(z.component_index("T"), Some(0));
/// assert_eq!(z.map_all(f64::sqrt).value(n[0].id(), "T")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait SubField {
    /// Type of the object backing this sub-field's support — a `SubMesh` for
    /// node fields, a `SubFiniteElementSpace` for element fields. Its store
    /// slot identity defines [`SubField::same_support`].
    type Support: Any + Send + Sync;

    /// Handle to the support backing this sub-field.
    fn support(&self) -> Handle<Self::Support>;

    /// Whether `self` and `other` are backed by the **same** support slot
    /// ([`Handle::same_object`]) — the precondition for combining them.
    fn same_support(&self, other: &Self) -> bool {
        self.support().same_object(&other.support())
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

    /// Smallest value of the named component — or, with `None`, the smallest
    /// value of the **whole** field, every component pooled.
    ///
    /// Pooling reads the field as the flat list of its values, in the spirit of
    /// [`SubField::xtx`]: on a field whose components carry different units
    /// (`sigma_xx` next to `sigma_xy`) the answer is "the smallest number in
    /// there", not a physical quantity — name the component when that matters.
    ///
    /// Errors if the component is unknown or the field holds no value.
    fn min(&self, component: Option<&str>) -> Result<f64> {
        fold_component(self, component, "min", f64::min)
    }

    /// Largest value of the named component — or, with `None`, the largest
    /// value of the **whole** field, every component pooled (see
    /// [`SubField::min`]).
    ///
    /// Errors if the component is unknown or the field holds no value.
    fn max(&self, component: Option<&str>) -> Result<f64> {
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

    /// A fresh sub-field on the **same support**, carrying only the components
    /// of `self` that appear in `wanted` — kept in `self`'s **own** order, with
    /// their values copied. The rows line up positionally with `self` (same
    /// support), so the result pairs with it under [`SubField::same_support`].
    ///
    /// A `wanted` entry this sub-field does not carry is silently ignored;
    /// errors only if **none** of `wanted` is present (a sub-field must keep at
    /// least one component). `wanted` is a single name or a list
    /// ([`IntoComponentNames`]). Used at the aggregate level by
    /// [`Field::filter_components`], which additionally **shares this sub's
    /// handle** untouched when the filter would keep every component.
    fn select_components(&self, wanted: impl IntoComponentNames) -> Result<Self>
    where
        Self: Sized,
    {
        let wanted = wanted.into_names();
        // Columns of `self` to keep, in self's order.
        let keep: Vec<usize> = self
            .components()
            .iter()
            .enumerate()
            .filter(|(_, c)| wanted.iter().any(|w| w == *c))
            .map(|(i, _)| i)
            .collect();
        if keep.is_empty() {
            return Err(PyrucastError::Message(format!(
                "select_components: this sub-field carries none of {:?}",
                wanted
            )));
        }
        let names: Vec<String> = keep.iter().map(|&i| self.components()[i].clone()).collect();
        let mut out = self.same_support_with(names)?;
        let in_nc = self.component_count();
        let out_nc = keep.len();
        let sv = self.values();
        let rows = sv.len().checked_div(in_nc.max(1)).unwrap_or(0);
        let outv = out.values_mut();
        for row in 0..rows {
            for (oc, &si) in keep.iter().enumerate() {
                outv[row * out_nc + oc] = sv[row * in_nc + si];
            }
        }
        Ok(out)
    }

    /// A fresh sub-field on the **same support** with component `from` renamed
    /// to `to`; every value is preserved (rename is metadata only, no value
    /// moves). Errors if `from` is absent or `to` already names another
    /// component of this sub-field.
    fn rename_component(&self, from: &str, to: &str) -> Result<Self>
    where
        Self: Sized,
    {
        let i = self.component_index_or_err(from)?;
        if from != to && self.component_index(to).is_some() {
            return Err(PyrucastError::Message(format!(
                "rename_component: target `{}` already names another component",
                to
            )));
        }
        let mut names = self.components().to_vec();
        names[i] = to.to_string();
        let mut out = self.same_support_with(names)?;
        // Same component count ⇒ identical buffer layout; copy values verbatim.
        out.values_mut().copy_from_slice(self.values());
        Ok(out)
    }
}

/// Fold `op` over every value of one component of a [`SubField`].
fn fold_component<S: SubField + ?Sized>(
    field: &S,
    component: Option<&str>,
    op_name: &str,
    op: fn(f64, f64) -> f64,
) -> Result<f64> {
    // Parallel reduction: `op` (min/max) is associative & commutative ⇒ the
    // result is identical to the sequential left-fold for any thread count.
    // With a component, one value per row is read; without one, the values are
    // read flat — the layout is irrelevant when every one of them counts.
    let folded = match component {
        Some(name) => {
            let ci = field
                .component_index(name)
                .ok_or_else(|| PyrucastError::Message(format!("unknown component: {}", name)))?;
            let n_comp = field.component_count();
            field
                .values()
                .par_chunks(n_comp)
                .with_min_len((MIN_PARALLEL_LEN / n_comp).max(1))
                .map(|row| row[ci])
                .reduce_with(op)
        }
        None => field
            .values()
            .par_iter()
            .with_min_len(MIN_PARALLEL_LEN)
            .copied()
            .reduce_with(op),
    };
    folded.ok_or_else(|| {
        PyrucastError::Message(match component {
            Some(name) => format!(
                "{}: no value for component {} (empty support)",
                op_name, name
            ),
            None => format!("{}: the field holds no value", op_name),
        })
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
/// over the subs that define it and error if none does — and, called
/// without a component, over every value of every sub.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// // Ce que l'agrégat ajoute au contrat de zone : les vues d'ensemble et
/// // les opérations qui traversent toutes les zones.
/// assert_eq!(temp.components()?, vec!["T".to_string()]);
/// assert_eq!(temp.map_subs(|s| Ok(s.map_all(f64::sqrt)))?
///     .get(0)?.read().value(n[0].id(), "T")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait Field: Aggregate
where
    Self::Sub: SubField,
{
    /// Union of the subs' component names, first-seen order.
    fn components(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for h in self.iter() {
            let s = h.read();
            for c in s.components() {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
        }
        Ok(out)
    }

    /// Smallest value of `component` across the subs defining it — or, with
    /// `None`, the smallest value of the **whole** field, every component of
    /// every zone pooled (see [`SubField::min`] for what pooling means).
    ///
    /// Errors if no sub defines the component; with `None`, if no zone holds
    /// a single value.
    fn min(&self, component: Option<&str>) -> Result<f64> {
        fold_subs(self, component, "min", f64::min)
    }

    /// Largest value of `component` across the subs defining it — or, with
    /// `None`, the largest value of the **whole** field (see
    /// [`SubField::min`]).
    ///
    /// Errors if no sub defines the component; with `None`, if no zone holds
    /// a single value.
    fn max(&self, component: Option<&str>) -> Result<f64> {
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
                let s = h.read();
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
            acc += SubField::xtx(&*h.read());
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
            let s = h.read();
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
            zones: self.iter().map(|h| h.read()).collect(),
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
            .map(|h| f(&*h.read()))
            .collect::<Result<_>>()?;
        let mut out = Self::default();
        for s in subs {
            out.add_sub(Handle::new(s))?;
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
            let mut s = h.write();
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
        let lefts: Vec<Self::Sub> = self.iter().map(|h| (*h.read()).clone()).collect();
        let rights: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();

        let mut out = Self::default();
        let mut right_used = vec![false; rights.len()];
        // Each left zone: combine with the right zone on the same support if
        // any, else pass through unchanged.
        for l in &lefts {
            match rights.iter().position(|r| l.same_support(r)) {
                Some(j) => {
                    out.add_sub(Handle::new(l.merge_components(&rights[j], op)?))?;
                    right_used[j] = true;
                }
                None => out.add_sub(Handle::new(l.clone()))?,
            }
        }
        // Right zones whose support was absent on the left: pass through.
        for (j, r) in rights.iter().enumerate() {
            if !right_used[j] {
                out.add_sub(Handle::new(r.clone()))?;
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
        let others: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();
        let mut acc = 0.0;
        for h in self.iter() {
            let s = h.read();
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
        let others: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();
        let mut out = Self::default();
        for h in self.iter() {
            let s = h.read();
            // At most one right zone shares the support (field invariant).
            if let Some(os) = others.iter().find(|os| s.same_support(os)) {
                out.add_sub(Handle::new(s.pscal(os)?))?;
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
            let s = h.read();
            if s.same_support(sub) {
                out.add_sub(Handle::new(s.merge_components(sub, op)?))?;
                matched = true;
            } else {
                out.add_sub(Handle::new((*s).clone()))?;
            }
        }
        if !matched {
            return Err(PyrucastError::Message(
                "merge_subfield: no zone shares the sub-field's support".into(),
            ));
        }
        Ok(out)
    }

    /// A new field keeping, in every zone, only the components named in
    /// `wanted` (each zone keeps its matches in its **own** order). A zone is
    /// processed independently:
    ///
    /// - a zone carrying **none** of `wanted` is **dropped**;
    /// - a zone carrying **only** requested components (nothing to strip) has
    ///   its handle **shared** as-is — no sub-field is duplicated;
    /// - a zone carrying some requested and some other components is rebuilt on
    ///   the same support with just the requested ones ([`SubField::select_components`]).
    ///
    /// `wanted` is a single name or a list ([`IntoComponentNames`]) and accepts
    /// a superset of the field's components (extras are ignored), so passing
    /// `model.primal_vars()` to strip a solver result of its dual (Lagrange)
    /// unknowns is the intended use. Errors if **no** zone carries any of
    /// `wanted`.
    fn filter_components(&self, wanted: impl IntoComponentNames) -> Result<Self>
    where
        Self: Sized,
    {
        let wanted = wanted.into_names();
        let mut out = Self::default();
        for h in self.iter() {
            let s = h.read();
            let n_present = s
                .components()
                .iter()
                .filter(|c| wanted.iter().any(|w| w == *c))
                .count();
            if n_present == 0 {
                continue; // zone carries nothing requested → dropped
            }
            if n_present == s.component_count() {
                // Filter is a no-op on this zone: share the handle untouched.
                out.add_sub(h.clone())?;
            } else {
                out.add_sub(Handle::new(s.select_components(wanted.as_slice())?))?;
            }
        }
        if out.is_empty() {
            return Err(PyrucastError::Message(format!(
                "filter_components: no zone carries any of {:?}",
                wanted
            )));
        }
        Ok(out)
    }

    /// A new field with component `from` renamed to `to` in every zone carrying
    /// it. A zone without `from` is carried **unchanged** (handle shared); a
    /// zone with it is rebuilt on the same support ([`SubField::rename_component`],
    /// values preserved). Errors if **no** zone carries `from`, or if a zone
    /// carrying `from` already has a component named `to`.
    fn rename_component(&self, from: &str, to: &str) -> Result<Self>
    where
        Self: Sized,
    {
        let mut out = Self::default();
        let mut found = false;
        for h in self.iter() {
            let s = h.read();
            if s.component_index(from).is_some() {
                out.add_sub(Handle::new(s.rename_component(from, to)?))?;
                found = true;
            } else {
                out.add_sub(h.clone())?;
            }
        }
        if !found {
            return Err(PyrucastError::Message(format!(
                "rename_component: no zone carries component `{}`",
                from
            )));
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// # use pyrucast::containers::field::MapValues;
/// # use pyrucast::ops::field;
/// // Ce qui unifie la zone et l'agrégat pour les maths élément par
/// // élément : une seule définition de `sqrt` sert les quatre types de
/// // champ.
/// assert_eq!(field::sqrt(&temp)?.get(0)?.read().value(n[0].id(), "T")?, 2.0);
/// let z = temp.get(0)?;
/// let z = z.read();
/// assert_eq!(field::sqrt(&*z)?.value(n[0].id(), "T")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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

/// Node-by-node scalar product, uniform over the four field flavours.
///
/// The same shape as [`MapValues`]: a small primitive that lets the operator
/// [`psca`](fn@crate::ops::field::psca) be written once, generically, instead
/// of dispatching over four types. `pscal` (zone) and `pscal_field`
/// (aggregate) do the work; this trait only unifies their names.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::{Field, SubField};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(),
/// #                                    vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 4.0).unwrap();
/// # use pyrucast::ops::field;
/// // Le produit scalaire rend un champ à **une** composante, nommée
/// // `psca`, quelle que soit la saveur du champ d'entrée.
/// let p = field::psca(&temp, &temp)?;
/// assert_eq!(p.get(0)?.read().components(), &["psca".to_string()]);
/// assert_eq!(p.get(0)?.read().value(n[0].id(), "psca")?, 16.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait Pscal: Sized {
    /// A new field of the same flavour, carrying the single `"psca"` component.
    fn pscal_with(&self, other: &Self) -> Result<Self>;
}

impl Pscal for SubNodeField {
    fn pscal_with(&self, other: &Self) -> Result<Self> {
        self.pscal(other)
    }
}

impl Pscal for SubElementField {
    fn pscal_with(&self, other: &Self) -> Result<Self> {
        self.pscal(other)
    }
}

impl Pscal for NodeField {
    fn pscal_with(&self, other: &Self) -> Result<Self> {
        self.pscal_field(other)
    }
}

impl Pscal for ElementField {
    fn pscal_with(&self, other: &Self) -> Result<Self> {
        self.pscal_field(other)
    }
}

/// Fold `op` over one component across every sub that defines it.
fn fold_subs<A>(
    agg: &A,
    component: Option<&str>,
    op_name: &str,
    op: fn(f64, f64) -> f64,
) -> Result<f64>
where
    A: Aggregate,
    A::Sub: SubField,
{
    // Per-zone fold in parallel (each zone fold is itself parallel), then a
    // serial associative combine. `op` (min/max) is associative & commutative
    // ⇒ thread-count-independent.
    //
    // A zone that has nothing to say contributes `None` rather than an error:
    // a zone that does not define the component, or — when every component is
    // pooled — a zone with no value at all. The error is raised once, at the
    // end, if *no* zone contributed.
    let handles: Vec<&Handle<A::Sub>> = agg.iter().collect();
    let per_sub: Vec<Option<f64>> = handles
        .par_iter()
        .map(|h| -> Result<Option<f64>> {
            let s = h.read();
            let silent = match component {
                Some(name) => s.component_index(name).is_none(),
                None => s.values().is_empty(),
            };
            if silent {
                Ok(None)
            } else {
                Ok(Some(fold_component(&*s, component, op_name, op)?))
            }
        })
        .collect::<Result<_>>()?;
    let acc = per_sub.into_iter().flatten().reduce(op);
    acc.ok_or_else(|| {
        PyrucastError::Message(match component {
            Some(name) => format!("{}: no sub-field defines component {}", op_name, name),
            None => format!("{}: the field holds no value", op_name),
        })
    })
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn make_node_field(values: &[f64]) -> SubNodeField {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..values.len())
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            for n in &nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            Handle::new(sm)
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
        assert_eq!(SubField::min(&f, Some("T")).unwrap(), -1.5);
        assert_eq!(SubField::max(&f, Some("T")).unwrap(), 4.0);
    }

    #[test]
    fn subfield_min_max_unknown_component_errors() {
        let f = make_node_field(&[1.0]);
        assert!(SubField::min(&f, Some("missing")).is_err());
        assert!(SubField::max(&f, Some("missing")).is_err());
    }

    #[test]
    fn subfield_min_max_isolates_components() {
        // Two components: min/max must stride over the right offsets.
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            Handle::new(sm)
        };
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set(0, 0, 10.0).unwrap();
        f.set(0, 1, -10.0).unwrap();
        f.set(1, 0, 20.0).unwrap();
        f.set(1, 1, -20.0).unwrap();
        assert_eq!(SubField::min(&f, Some("U")).unwrap(), 10.0);
        assert_eq!(SubField::max(&f, Some("U")).unwrap(), 20.0);
        assert_eq!(SubField::min(&f, Some("V")).unwrap(), -20.0);
        assert_eq!(SubField::max(&f, Some("V")).unwrap(), -10.0);
    }

    #[test]
    fn subfield_min_max_without_component_pool_every_component() {
        // Two components with opposite signs: pooling reads the flat list of
        // values, so the answer comes from whichever component holds it.
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            Handle::new(sm)
        };
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set(0, 0, 10.0).unwrap();
        f.set(0, 1, -10.0).unwrap();
        f.set(1, 0, 20.0).unwrap();
        f.set(1, 1, -20.0).unwrap();
        assert_eq!(SubField::min(&f, None).unwrap(), -20.0); // dans V
        assert_eq!(SubField::max(&f, None).unwrap(), 20.0); // dans U
    }

    #[test]
    fn subfield_min_on_empty_support_errors() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let sm: Handle<SubMesh> = Handle::new(SubMesh::new(coords, ElementType::POI1));
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(SubField::min(&f, Some("T")).is_err());
        assert!(SubField::min(&f, None).is_err());
    }

    #[test]
    fn subfield_sum_and_xtx() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            Handle::new(sm)
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
        let coords = Handle::new(Coords::new(1).unwrap());
        let sm: Handle<SubMesh> = Handle::new(SubMesh::new(coords, ElementType::POI1));
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert_eq!(SubField::sum(&f, "T").unwrap(), 0.0);
        assert_eq!(SubField::xtx(&f), 0.0);
    }

    fn make_two_zone_element_field() -> ElementField {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            Handle::new(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(coords, ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            Handle::new(sm)
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
        ef.get(0).unwrap().write().set_uniform("k", 3.0).unwrap();
        {
            let mut s = ef.get(1).unwrap().write();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 210e9).unwrap();
        }
        assert_eq!(Field::min(&ef, Some("k")).unwrap(), -2.0);
        assert_eq!(Field::max(&ef, Some("k")).unwrap(), 3.0);
        // E exists on the second zone only: folded over that zone alone.
        assert_eq!(Field::min(&ef, Some("E")).unwrap(), 210e9);
        assert_eq!(Field::max(&ef, Some("E")).unwrap(), 210e9);
    }

    #[test]
    fn field_min_max_without_component_pool_every_zone() {
        let ef = make_two_zone_element_field();
        ef.get(0).unwrap().write().set_uniform("k", 3.0).unwrap();
        {
            let mut s = ef.get(1).unwrap().write();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 210e9).unwrap();
        }
        // Toutes zones et toutes composantes confondues.
        assert_eq!(Field::min(&ef, None).unwrap(), -2.0);
        assert_eq!(Field::max(&ef, None).unwrap(), 210e9);
    }

    #[test]
    fn field_min_without_component_on_an_empty_aggregate_errors() {
        let ef = ElementField::default();
        assert!(Field::min(&ef, None).is_err());
        assert!(Field::max(&ef, None).is_err());
    }

    #[test]
    fn field_min_unknown_component_errors() {
        let ef = make_two_zone_element_field();
        assert!(Field::min(&ef, Some("missing")).is_err());
        assert!(Field::max(&ef, Some("missing")).is_err());
    }

    #[test]
    fn field_sum_and_xtx_fold_across_subs() {
        let ef = make_two_zone_element_field();
        ef.get(0).unwrap().write().set_uniform("k", 3.0).unwrap();
        {
            let mut s = ef.get(1).unwrap().write();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 5.0).unwrap();
        }
        // Field-level folds equal the sum of the per-zone reductions (no need to
        // know the Gauss-point counts).
        let z0 = SubField::sum(&*ef.get(0).unwrap().read(), "k").unwrap();
        let z1 = SubField::sum(&*ef.get(1).unwrap().read(), "k").unwrap();
        assert!((Field::sum(&ef, "k").unwrap() - (z0 + z1)).abs() < 1e-12);
        // E lives on zone 1 only.
        let e1 = SubField::sum(&*ef.get(1).unwrap().read(), "E").unwrap();
        assert!((Field::sum(&ef, "E").unwrap() - e1).abs() < 1e-12);
        // xtx over the whole field = Σ of the per-zone xtx.
        let x0 = SubField::xtx(&*ef.get(0).unwrap().read());
        let x1 = SubField::xtx(&*ef.get(1).unwrap().read());
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
        ef.get(0).unwrap().write().set_uniform("k", 3.0).unwrap();
        {
            let mut s = ef.get(1).unwrap().write();
            s.set_uniform("k", -2.0).unwrap();
            s.set_uniform("E", 5.0).unwrap();
        }
        // "k" lives on both zones: Σ of the per-zone k-only xtx.
        let k0 = ef.get(0).unwrap().read().xtx_components(&["k"]).unwrap();
        let k1 = ef.get(1).unwrap().read().xtx_components(&["k"]).unwrap();
        assert!((ef.xtx_components(&["k"]).unwrap() - (k0 + k1)).abs() < 1e-9);
        // "E" lives on zone 1 only; zone 0 (no E) is skipped, not an error.
        let e1 = ef.get(1).unwrap().read().xtx_components(&["E"]).unwrap();
        assert!((ef.xtx_components(&["E"]).unwrap() - e1).abs() < 1e-9);
        // No zone defines "missing" ⇒ error.
        assert!(ef.xtx_components(&["missing"]).is_err());
    }

    // ─── SubField::select_components / rename_component ───────────────────────

    #[test]
    fn subfield_select_components_keeps_subset_in_self_order() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into(), "W".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 1.0).unwrap();
        f.set_value(nodes[0].id(), "V", 2.0).unwrap();
        f.set_value(nodes[0].id(), "W", 3.0).unwrap();
        // Keep V and W; the request order ["W","V"] is ignored — self's order wins.
        // Ergonomic input: a plain `[&str; N]` array (IntoComponentNames).
        let g = f.select_components(["W", "V"]).unwrap();
        assert_eq!(g.components(), &["V".to_string(), "W".into()]);
        assert_eq!(g.value(nodes[0].id(), "V").unwrap(), 2.0);
        assert_eq!(g.value(nodes[0].id(), "W").unwrap(), 3.0);
        // Same support ⇒ pairs under the operators.
        assert!(f.same_support(&g));
        // A single &str selects one component.
        let just_u = f.select_components("U").unwrap();
        assert_eq!(just_u.components(), &["U".to_string()]);
        // Unknown names ignored; none present ⇒ error.
        assert!(f.select_components("nope").is_err());
    }

    #[test]
    fn subfield_rename_component_preserves_values() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 5.0).unwrap();
        f.set_value(nodes[0].id(), "V", 6.0).unwrap();
        let g = f.rename_component("U", "DX").unwrap();
        assert_eq!(g.components(), &["DX".to_string(), "V".into()]);
        assert_eq!(g.value(nodes[0].id(), "DX").unwrap(), 5.0);
        assert_eq!(g.value(nodes[0].id(), "V").unwrap(), 6.0);
        // Absent source, or collision with an existing name ⇒ error.
        assert!(f.rename_component("nope", "X").is_err());
        assert!(f.rename_component("U", "V").is_err());
    }

    #[test]
    fn field_filter_components_shares_or_rebuilds_and_drops() {
        // zone0: [k], zone1: [E, k]
        let ef = make_two_zone_element_field();
        // Keep only "k": zone0 is a no-op (shares its handle), zone1 is rebuilt.
        // Single-name ergonomic input.
        let out = ef.filter_components("k").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(Field::components(&out).unwrap(), vec!["k"]);
        // zone0 handle shared untouched; zone1 rebuilt (fresh slot).
        assert!(out.get(0).unwrap().same_object(&ef.get(0).unwrap()));
        assert!(!out.get(1).unwrap().same_object(&ef.get(1).unwrap()));

        // Keep only "E": zone0 (no E) is dropped, zone1 kept.
        let only_e = ef.filter_components("E").unwrap();
        assert_eq!(only_e.len(), 1);
        assert_eq!(Field::components(&only_e).unwrap(), vec!["E"]);

        // No zone carries any requested component ⇒ error.
        assert!(ef.filter_components("missing").is_err());
    }

    #[test]
    fn field_filter_components_accepts_superset_request() {
        // A solver-result-like field: primal [u_x,u_y] plus dual [lambda].
        let (sm, nodes) = poi1_support(1);
        let mut s = SubNodeField::from_poi1(&sm, vec!["u_x".into(), "u_y".into(), "lambda".into()])
            .unwrap();
        s.set_value(nodes[0].id(), "u_x", 1.0).unwrap();
        s.set_value(nodes[0].id(), "lambda", 9.0).unwrap();
        let f = NodeField::from_sub(s);
        // primal_vars() may name components the field lacks — extras are ignored.
        // A Vec<String> (as primal_vars() returns) flows in directly.
        let primal = f
            .filter_components(vec!["u_x".to_string(), "u_y".to_string(), "T".to_string()])
            .unwrap();
        assert_eq!(Field::components(&primal).unwrap(), vec!["u_x", "u_y"]);
        assert_eq!(primal.value(nodes[0].id(), "u_x").unwrap(), 1.0);
        assert!(primal.value_opt(nodes[0].id(), "lambda").unwrap().is_none());
    }

    #[test]
    fn field_rename_component_renames_matching_zones_only() {
        let ef = make_two_zone_element_field(); // zone0: [k], zone1: [E, k]
        let out = ef.rename_component("E", "young").unwrap();
        // zone0 has no E ⇒ handle shared; zone1 rebuilt with the new name.
        assert!(out.get(0).unwrap().same_object(&ef.get(0).unwrap()));
        assert_eq!(Field::components(&out).unwrap(), vec!["k", "young"]);
        // No zone carries the source ⇒ error.
        assert!(ef.rename_component("missing", "x").is_err());
    }

    // ─── SubField::merge_components / check_same_components ────────────────────

    /// POI1 support over `n` nodes, plus the nodes (for `value(nid, …)`).
    fn poi1_support(n: usize) -> (Handle<SubMesh>, Vec<Node>) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::POI1);
            for nd in &nodes {
                sm.add_cell(&[nd.id()]).unwrap();
            }
            Handle::new(sm)
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
        let coords = Handle::new(Coords::new(1).unwrap());
        let node = |x: f64| Node::create_in(coords.clone(), &[x]).unwrap();
        let poi1 = |ns: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in ns {
                sm.add_cell(&[n.id()]).unwrap();
            }
            Handle::new(sm)
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
        nf.add_sub(Handle::new(zb)).unwrap();
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
        let coords = Handle::new(Coords::new(1).unwrap());
        let node = |x: f64| Node::create_in(coords.clone(), &[x]).unwrap();
        let poi1 = |ns: &[&Node]| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in ns {
                sm.add_cell(&[n.id()]).unwrap();
            }
            Handle::new(sm)
        };
        let (na, nb) = (node(0.0), node(1.0));
        let sm_a = poi1(&[&na]);
        let sm_b = poi1(&[&nb]);
        let mut za = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        za.set(0, 0, 3.0).unwrap();
        let mut zb = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        zb.set(0, 0, 5.0).unwrap();
        let mut nf = NodeField::from_sub(za);
        nf.add_sub(Handle::new(zb)).unwrap();
        // pscal with itself → per-node square, two zones preserved.
        let p = nf.pscal_field(&nf).unwrap();
        let view = p.view().unwrap();
        assert_eq!(view.value(na.id(), "psca").unwrap(), 9.0);
        assert_eq!(view.value(nb.id(), "psca").unwrap(), 25.0);
    }

    // ─── Field-level arithmetic ──────────────────────────────────────────────

    /// Single-zone Lagrange-1 FE space on one TRI3 cell.
    fn one_tri3_fes() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        mesh.add_sub(sm).unwrap();
        FiniteElementSpace::lagrange1(&mesh).unwrap()
    }

    #[test]
    fn field_combine_scalar_hits_every_zone() {
        let ef = make_two_zone_element_field();
        ef.get(0).unwrap().write().set_uniform("k", 1.0).unwrap();
        ef.get(1).unwrap().write().set_uniform("k", 2.0).unwrap();
        let out = ef.combine_scalar(|a, b| a + b, 10.0).unwrap();
        assert_eq!(out.get(0).unwrap().read().value(0, 0, "k").unwrap(), 11.0);
        assert_eq!(out.get(1).unwrap().read().value(0, 0, "k").unwrap(), 12.0);
    }

    #[test]
    fn field_add_to_component_present_zones_only() {
        let ef = make_two_zone_element_field(); // zone0 ["k"], zone1 ["E","k"]
        ef.get(1).unwrap().write().set_uniform("E", 100.0).unwrap();
        ef.add_to_component("E", 1.0).unwrap(); // E only on zone 1
        assert_eq!(ef.get(1).unwrap().read().value(0, 0, "E").unwrap(), 101.0);
        assert!(ef.add_to_component("missing", 1.0).is_err());
    }

    #[test]
    fn field_merge_field_same_decomposition() {
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into()]).unwrap();
        f.get(0).unwrap().write().set_uniform("E", 3.0).unwrap();
        g.get(0).unwrap().write().set_uniform("E", 4.0).unwrap();
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        let z = s.get(0).unwrap().read();
        for gp in 0..z.gauss_count() {
            assert_eq!(z.value(0, gp, "E").unwrap(), 7.0);
        }
    }

    #[test]
    fn field_merge_field_disjoint_supports_unions() {
        // Distinct supports ⇒ union: both zones pass through unchanged.
        let f = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        let g = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        f.get(0).unwrap().write().set_uniform("E", 3.0).unwrap();
        g.get(0).unwrap().write().set_uniform("E", 4.0).unwrap();
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        assert_eq!(s.len(), 2, "distinct supports ⇒ two zones");
        assert_eq!(s.get(0).unwrap().read().value(0, 0, "E").unwrap(), 3.0);
        assert_eq!(s.get(1).unwrap().read().value(0, 0, "E").unwrap(), 4.0);
    }

    #[test]
    fn field_merge_field_partial_components_passes_through() {
        // Same support: f has [E], g has [E, nu]. E combines, nu passes through.
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
        f.get(0).unwrap().write().set_uniform("E", 3.0).unwrap();
        {
            let mut z = g.get(0).unwrap().write();
            z.set_uniform("E", 4.0).unwrap();
            z.set_uniform("nu", 0.3).unwrap();
        }
        let s = f.merge_field(&g, |a, b| a + b).unwrap();
        assert_eq!(s.len(), 1);
        let z = s.get(0).unwrap().read();
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
        a.get(0).unwrap().write().set_uniform("E", 10.0).unwrap();
        {
            let mut z = b.get(0).unwrap().write();
            z.set_uniform("E", 4.0).unwrap();
            z.set_uniform("nu", 0.3).unwrap();
        }
        let s = a.merge_field(&b, |x, y| x - y).unwrap();
        let z = s.get(0).unwrap().read();
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
        ef.get(0).unwrap().write().set_uniform("k", 1.0).unwrap();
        {
            let mut z1 = ef.get(1).unwrap().write();
            z1.set_uniform("k", 2.0).unwrap();
            z1.set_uniform("E", 5.0).unwrap();
        }
        // A sub on zone 0's support, k = 10.
        let mut sub = (*ef.get(0).unwrap().read()).clone();
        sub.set_uniform("k", 10.0).unwrap();
        let out = ef.merge_subfield(&sub, |a, b| a + b).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out.get(0).unwrap().read().value(0, 0, "k").unwrap(), 11.0);
        let z1 = out.get(1).unwrap().read();
        assert_eq!(z1.value(0, 0, "k").unwrap(), 2.0);
        assert_eq!(z1.value(0, 0, "E").unwrap(), 5.0);
    }

    #[test]
    fn field_merge_subfield_no_match_errors() {
        let ef = make_two_zone_element_field();
        // A sub on a brand-new, unrelated FE support.
        let other = ElementField::new(&one_tri3_fes(), vec!["k".into()]).unwrap();
        let sub = (*other.get(0).unwrap().read()).clone();
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
        f.get(0).unwrap().write().set_uniform("E", 3.0).unwrap();
        g.get(0).unwrap().write().set_uniform("E", 4.0).unwrap();

        let s = (&f + &g).unwrap();
        let z = s.get(0).unwrap().read();
        assert_eq!(z.value(0, 0, "E").unwrap(), 7.0);

        let scaled = (&f - 1.0).unwrap();
        let zs = scaled.get(0).unwrap().read();
        assert_eq!(zs.value(0, 0, "E").unwrap(), 2.0);
    }
}
