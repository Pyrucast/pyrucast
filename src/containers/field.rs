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
//! use pyrucast::containers::mesh::{Configuration, ElementType, Node, SubMesh};
//! use pyrucast::containers::node_field::SubNodeField;
//! use pyrucast::store::insert;
//!
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
//! let sm = {
//!     let mut sm = SubMesh::new(cfg, ElementType::POI1);
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
use crate::error::{PyrucastError, Result};
use crate::persist::Persist;
use crate::store::{read, ReadGuard};
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
}

impl<A: Aggregate> Field for A where A::Sub: SubField {}

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
    use crate::containers::mesh::{Configuration, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::store::{insert, write, Handle};

    fn make_node_field(values: &[f64]) -> SubNodeField {
        let cfg = insert(Configuration::new(1).unwrap());
        let nodes: Vec<Node> = (0..values.len())
            .map(|i| Node::create_in(cfg.clone(), &[i as f64]).unwrap())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::POI1);
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
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::POI1);
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
        let cfg = insert(Configuration::new(1).unwrap());
        let sm: Handle<SubMesh> = insert(SubMesh::new(cfg, ElementType::POI1));
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(SubField::min(&f, "T").is_err());
    }

    fn make_two_zone_element_field() -> ElementField {
        let cfg = insert(Configuration::new(2).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(cfg, ElementType::QUA4);
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
}
