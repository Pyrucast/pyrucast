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
/// type, on the [`crate::containers::node_field::NodeFieldView`] and
/// [`crate::containers::element_field::ElementFieldView`] aliases.
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
        self.component_index(name).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", name))
        })
    }

    /// Flat value buffer, mutable (same layout as [`SubField::values`]).
    fn values_mut(&mut self) -> &mut [f64];

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
    /// component count, offset = component index).
    fn map_component(&mut self, component: &str, f: impl Fn(f64) -> f64) -> Result<()> {
        let ci = self.component_index_or_err(component)?;
        let ncomp = self.component_count();
        for v in self.values_mut()[ci..].iter_mut().step_by(ncomp) {
            *v = f(*v);
        }
        Ok(())
    }

    /// A clone of `self` with `f` applied to **every** value (all
    /// nodes/points × all components) — the scalar-broadcast primitive.
    fn map_all(&self, f: impl Fn(f64) -> f64) -> Self
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        for v in out.values_mut() {
            *v = f(*v);
        }
        out
    }

    /// Element-by-element binary combination with another sub-field, **strict**:
    /// both must be on the same support ([`SubField::same_support`]) and carry
    /// the same set of components — otherwise an error. Components are aligned
    /// **by name** (order may differ); same support ⇒ same rows in the same
    /// order, so values line up positionally. Division does **not** guard
    /// against zero (numpy-like `inf`/`nan`).
    fn combine(&self, other: &Self, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized + Clone,
    {
        if !self.same_support(other) {
            return Err(PyrucastError::Message(
                "combine: operands are not on the same support".into(),
            ));
        }
        let nc = self.component_count();
        if other.component_count() != nc {
            return Err(PyrucastError::Message(format!(
                "combine: mismatched components ({} vs {})",
                nc,
                other.component_count()
            )));
        }
        // self's components, mapped to other's indices (by name).
        let other_idx = self
            .components()
            .iter()
            .map(|c| other.component_index_or_err(c))
            .collect::<Result<Vec<usize>>>()?;
        let ov = other.values();
        let mut out = self.clone();
        let out_vals = out.values_mut();
        let n_rows = out_vals.len() / nc;
        for row in 0..n_rows {
            for ci in 0..nc {
                let a = out_vals[row * nc + ci];
                let b = ov[row * nc + other_idx[ci]];
                out_vals[row * nc + ci] = op(a, b);
            }
        }
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
    field
        .values()
        .iter()
        .skip(ci)
        .step_by(n_comp)
        .copied()
        .reduce(op)
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
        $crate::__subfield_scalar_op!($T, Add, add, +=);
        $crate::__subfield_scalar_op!($T, Sub, sub, -=);
        $crate::__subfield_scalar_op!($T, Mul, mul, *=);
        $crate::__subfield_scalar_op!($T, Div, div, /=);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __subfield_scalar_op {
    ($T:ty, $Trait:ident, $method:ident, $assign:tt) => {
        impl std::ops::$Trait<f64> for $T {
            type Output = $T;
            fn $method(mut self, rhs: f64) -> $T {
                for v in $crate::containers::field::SubField::values_mut(&mut self) {
                    *v $assign rhs;
                }
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
    fn map_subs(&self, f: impl Fn(&Self::Sub) -> Result<Self::Sub>) -> Result<Self>
    where
        Self: Sized,
    {
        let mut out = Self::default();
        for h in self.iter() {
            let new_sub = f(&*read(h)?)?;
            out.add_sub(insert(new_sub))?;
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
    fn map_all(&self, f: impl Fn(f64) -> f64 + Copy) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        self.map_subs(|s| Ok(s.map_all(f)))
    }

    /// Apply `f` **in place** to the named component, on every zone that
    /// defines it. Errors only if **no** zone defines the component.
    fn map_component(&self, component: &str, f: impl Fn(f64) -> f64 + Copy) -> Result<()> {
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

    /// Binary combination of two fields on the **same decomposition**: each
    /// zone of `self` is paired with the zone of `other` on the same support
    /// ([`SubField::same_support`]) and combined ([`SubField::combine`],
    /// strict on components). **Every** zone on both sides must pair exactly
    /// once — otherwise an error.
    fn combine_field(&self, other: &Self, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        // Snapshot other's zones (clone out of the store) so we never hold two
        // guards at once — safe even when `other` shares handles with `self`.
        let others: Vec<Self::Sub> = other
            .iter()
            .map(|h| read(h).map(|g| (*g).clone()))
            .collect::<Result<_>>()?;
        let mut used = vec![false; others.len()];
        let mut out = Self::default();
        for h in self.iter() {
            let s = read(h)?;
            let mut matched = false;
            for (j, os) in others.iter().enumerate() {
                if used[j] {
                    continue;
                }
                if s.same_support(os) {
                    out.add_sub(insert(s.combine(os, op)?))?;
                    used[j] = true;
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Err(PyrucastError::Message(
                    "combine_field: a zone of the left field has no matching \
                     zone (same support) in the right field"
                        .into(),
                ));
            }
        }
        if used.iter().any(|&u| !u) {
            return Err(PyrucastError::Message(
                "combine_field: a zone of the right field has no matching \
                 zone in the left field"
                    .into(),
            ));
        }
        Ok(out)
    }

    /// Targeted update: combine `sub` into the zone(s) of `self` sharing its
    /// support ([`SubField::combine`], strict on components); every other zone
    /// is carried **unchanged**. Errors if no zone shares the support.
    fn combine_subfield(&self, sub: &Self::Sub, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        let mut out = Self::default();
        let mut matched = false;
        for h in self.iter() {
            let s = read(h)?;
            if s.same_support(sub) {
                out.add_sub(insert(s.combine(sub, op)?))?;
                matched = true;
            } else {
                out.add_sub(insert((*s).clone()))?;
            }
        }
        if !matched {
            return Err(PyrucastError::Message(
                "combine_subfield: no zone shares the sub-field's support".into(),
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
    A: Aggregate + ?Sized,
    A::Sub: SubField,
{
    let mut acc: Option<f64> = None;
    for h in agg.iter() {
        let s = read(h)?;
        let sub_val = if s.component_index(component).is_none() {
            None
        } else {
            Some(fold_component(&*s, component, op_name, op)?)
        };
        if let Some(v) = sub_val {
            acc = Some(match acc {
                Some(a) => op(a, v),
                None => v,
            });
        }
    }
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

    // ─── SubField::combine (binary, strict) ──────────────────────────────────

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
    fn subfield_combine_adds_same_support() {
        let (sm, _) = poi1_support(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 0, 2.0).unwrap();
        let mut g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        g.set(0, 0, 10.0).unwrap();
        g.set(1, 0, 20.0).unwrap();
        let s = f.combine(&g, |a, b| a + b).unwrap();
        assert_eq!(s.get(0, 0).unwrap(), 11.0);
        assert_eq!(s.get(1, 0).unwrap(), 22.0);
    }

    #[test]
    fn subfield_combine_aligns_components_by_name() {
        let (sm, nodes) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["U".into(), "V".into()]).unwrap();
        f.set_value(nodes[0].id(), "U", 1.0).unwrap();
        f.set_value(nodes[0].id(), "V", 2.0).unwrap();
        // Other carries the same names in the opposite order.
        let mut g = SubNodeField::from_poi1(&sm, vec!["V".into(), "U".into()]).unwrap();
        g.set_value(nodes[0].id(), "U", 40.0).unwrap();
        g.set_value(nodes[0].id(), "V", 30.0).unwrap();
        let s = f.combine(&g, |a, b| a + b).unwrap();
        assert_eq!(s.value(nodes[0].id(), "U").unwrap(), 41.0);
        assert_eq!(s.value(nodes[0].id(), "V").unwrap(), 32.0);
    }

    #[test]
    fn subfield_combine_mismatched_support_errors() {
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        assert!(f.combine(&g, |a, b| a + b).is_err());
    }

    #[test]
    fn subfield_combine_mismatched_components_errors() {
        let (sm, _) = poi1_support(1);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let g = SubNodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        assert!(f.combine(&g, |a, b| a + b).is_err());
    }

    #[test]
    fn subfield_combine_div_by_zero_is_inf() {
        let (sm, _) = poi1_support(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        let g = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap(); // zero
        let s = f.combine(&g, |a, b| a / b).unwrap();
        assert!(s.get(0, 0).unwrap().is_infinite());
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
        write(&ef.get(0).unwrap()).unwrap().set_uniform("k", 1.0).unwrap();
        write(&ef.get(1).unwrap()).unwrap().set_uniform("k", 2.0).unwrap();
        let out = ef.combine_scalar(|a, b| a + b, 10.0).unwrap();
        assert_eq!(read(&out.get(0).unwrap()).unwrap().value(0, 0, "k").unwrap(), 11.0);
        assert_eq!(read(&out.get(1).unwrap()).unwrap().value(0, 0, "k").unwrap(), 12.0);
    }

    #[test]
    fn field_add_to_component_present_zones_only() {
        let ef = make_two_zone_element_field(); // zone0 ["k"], zone1 ["E","k"]
        write(&ef.get(1).unwrap()).unwrap().set_uniform("E", 100.0).unwrap();
        ef.add_to_component("E", 1.0).unwrap(); // E only on zone 1
        assert_eq!(read(&ef.get(1).unwrap()).unwrap().value(0, 0, "E").unwrap(), 101.0);
        assert!(ef.add_to_component("missing", 1.0).is_err());
    }

    #[test]
    fn field_combine_field_same_decomposition() {
        let fes = one_tri3_fes();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&fes, vec!["E".into()]).unwrap();
        write(&f.get(0).unwrap()).unwrap().set_uniform("E", 3.0).unwrap();
        write(&g.get(0).unwrap()).unwrap().set_uniform("E", 4.0).unwrap();
        let s = f.combine_field(&g, |a, b| a + b).unwrap();
        let z = read(&s.get(0).unwrap()).unwrap();
        for gp in 0..z.gauss_count() {
            assert_eq!(z.value(0, gp, "E").unwrap(), 7.0);
        }
    }

    #[test]
    fn field_combine_field_mismatched_decomposition_errors() {
        let f = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        let g = ElementField::new(&one_tri3_fes(), vec!["E".into()]).unwrap();
        assert!(f.combine_field(&g, |a, b| a + b).is_err());
    }

    #[test]
    fn field_combine_subfield_targets_matching_zone() {
        let ef = make_two_zone_element_field(); // zone0 ["k"], zone1 ["E","k"]
        write(&ef.get(0).unwrap()).unwrap().set_uniform("k", 1.0).unwrap();
        {
            let mut z1 = write(&ef.get(1).unwrap()).unwrap();
            z1.set_uniform("k", 2.0).unwrap();
            z1.set_uniform("E", 5.0).unwrap();
        }
        // A sub on zone 0's support, k = 10.
        let mut sub = (*read(&ef.get(0).unwrap()).unwrap()).clone();
        sub.set_uniform("k", 10.0).unwrap();
        let out = ef.combine_subfield(&sub, |a, b| a + b).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(read(&out.get(0).unwrap()).unwrap().value(0, 0, "k").unwrap(), 11.0);
        let z1 = read(&out.get(1).unwrap()).unwrap();
        assert_eq!(z1.value(0, 0, "k").unwrap(), 2.0);
        assert_eq!(z1.value(0, 0, "E").unwrap(), 5.0);
    }

    #[test]
    fn field_combine_subfield_no_match_errors() {
        let ef = make_two_zone_element_field();
        // A sub on a brand-new, unrelated FE support.
        let other = ElementField::new(&one_tri3_fes(), vec!["k".into()]).unwrap();
        let sub = (*read(&other.get(0).unwrap()).unwrap()).clone();
        assert!(ef.combine_subfield(&sub, |a, b| a + b).is_err());
    }
}
