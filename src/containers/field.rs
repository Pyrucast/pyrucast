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
//! // Without a named component, the reduction covers the whole field.
//! assert_eq!(SubField::min(&f, None).unwrap(), -3.0);
//! ```

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::mesh::SubMesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::handle::{Handle, ReadGuard};
use crate::parallel::*;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

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
/// // The Rust twin of the Python argument `str | list[str]`: the component
/// // operators read the same way on both sides.
/// assert_eq!(temp.filter_components("T")?.components(), vec!["T".to_string()]);
/// assert_eq!(temp.filter_components(["T"])?.components(), vec!["T".to_string()]);
/// assert_eq!(temp.filter_components(vec!["T".to_string()])?.components(),
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
/// // A read view on **every** zone at once: the guards are taken once, and
/// // concurrent reads do not wait for each other.
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
    /// // The union of the zones' components, in first-appearance order — the
    /// // same rule as everywhere else in the aggregates.
    /// assert_eq!(temp.view()?.components(), &["T".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn components(&self) -> &[String] {
        &self.components
    }
}

// ─── SubField ───────────────────────────────────────────────────────────────

/// Marks a slot a field does not carry, in a table built by
/// [`SubField::resolve_optional_components`].
///
/// `u32::MAX` and not `Option<u32>`: the table is read in the innermost loop of
/// every physics, and a sentinel keeps it a flat `&[u32]` — one cache line for a
/// whole tensor's worth of slots.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// let mat = SubElementField::from_uniform_per_component(
///     fes.get(0)?, vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?;
/// // `rho` was not supplied: the table says so once for the zone.
/// let table = mat.resolve_optional_components(&["rho"]);
/// assert_eq!(table[0], pyrucast::containers::field::ABSENT_COMPONENT);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const ABSENT_COMPONENT: u32 = u32::MAX;

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
/// // A purely **structural** contract: named components and a flat buffer
/// // where the component index varies fastest. Nodal fields and fields by
/// // elements both satisfy it, hence an arithmetic written once only.
/// let z = temp.get(0)?;
/// let z = z.read();
/// assert_eq!(z.components(), &["T".to_string()]);
/// assert_eq!(z.component_index("T"), Some(0));
/// assert_eq!(z.map_all(f64::sqrt).value(n[0].id(), "T")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait SubField {
    /// Type of the object backing this sub-field's support — a `SubMesh` for
    /// node fields, a `SubFiniteElementSpace` for element fields. Its identity
    /// — the handle's, not its content — defines [`SubField::same_support`].
    type Support: Any + Send + Sync;

    /// Handle to the support backing this sub-field.
    fn support(&self) -> Handle<Self::Support>;

    /// Whether `self` and `other` are backed by the **same** support object
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

    /// Resolve a **canonical component order** into positions in this field —
    /// the upstream half of index-based reading.
    ///
    /// A kernel that runs per Gauss point must not look components up by name:
    /// the search is a string comparison, and it re-proves at every point what
    /// is a property of the *zone*. It reads `values[table[SLOT]]` instead, and
    /// this builds `table` — once per zone, outside any hot loop.
    ///
    /// Every name in `canonical` must be present: a field that cannot serve the
    /// caller's convention is an error here, where it can be named, rather than
    /// a wrong number later. `what` names the field in that message ("material",
    /// "deformation", …), because a bare component name says nothing about which
    /// of three fields is malformed. A field carrying **more** components than
    /// asked is fine — the table simply ignores them.
    ///
    /// Use [`resolve_optional_components`](Self::resolve_optional_components)
    /// for slots a field may legitimately omit.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// // A material laid out "the other way round" from the convention.
    /// let mat = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["nu".into(), "E".into()], &[0.3, 210_000.0])?;
    /// // La table absorbe l'écart d'ordre : la lecture reste juste.
    /// let table = mat.resolve_components(&["E", "nu"], "material")?;
    /// assert_eq!(table, vec![1, 0]);
    /// let row = mat.point_values(0, 0)?;
    /// assert_eq!(row[table[0] as usize], 210_000.0); // E
    /// assert_eq!(row[table[1] as usize], 0.3);       // nu
    /// // A required component that is missing errs here, naming itself.
    /// let err = mat.resolve_components(&["E", "sigma_y"], "material").unwrap_err();
    /// assert!(format!("{err}").contains("sigma_y"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    fn resolve_components(&self, canonical: &[&str], what: &str) -> Result<Vec<u32>> {
        canonical
            .iter()
            .map(|name| {
                self.component_index(name).map(|i| i as u32).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{what}: missing component '{name}' — expected {canonical:?}, \
                             this field carries {:?}",
                        self.components()
                    ))
                })
            })
            .collect()
    }

    /// Resolve a canonical order where a slot may be **absent**, marking each
    /// missing one [`ABSENT_COMPONENT`].
    ///
    /// The counterpart of [`resolve_components`](Self::resolve_components) for
    /// what a field is allowed not to carry: an optional material constant the
    /// caller never supplied, a state component a 2-D kinematics does not span.
    /// Absence is a fact of the zone, so the kernel branches on the table it was
    /// handed — never on a lookup it performs itself.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::{SubField, ABSENT_COMPONENT};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let mat = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["E".into(), "nu".into(), "alpha".into()],
    ///     &[210_000.0, 0.3, 1.2e-5])?;
    /// // `alpha` was supplied, `rho` was not: the table says so, once for the zone.
    /// let table = mat.resolve_optional_components(&["alpha", "rho"]);
    /// assert_eq!(table[0], 2);
    /// assert_eq!(table[1], ABSENT_COMPONENT);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    fn resolve_optional_components(&self, canonical: &[&str]) -> Vec<u32> {
        canonical
            .iter()
            .map(|name| {
                self.component_index(name)
                    .map_or(ABSENT_COMPONENT, |i| i as u32)
            })
            .collect()
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

    /// The point each row stands at, when two zones of one field can share
    /// one — a node, for a nodal field. `None`, the default, when they never
    /// can: two subspaces have no Gauss point in common.
    ///
    /// What lets a field count a point once however many zones carry it
    /// ([`Field::sum`], [`Field::xtx`], [`Field::dot_field`]), and check that
    /// those zones agree on its value. Read in place: the points hold the
    /// support's read guard until dropped.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::SubNodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()])?;
    /// let t = SubNodeField::from_poi1(&support.get(0)?, vec!["T".into()])?;
    /// // A nodal row stands at a node, the one of the support in that place.
    /// let points = t.row_points().expect("a nodal field stands at nodes");
    /// assert_eq!(points.ids(), &[a.id(), b.id()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    fn row_points(&self) -> Option<RowPoints> {
        None
    }

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
        debug_assert_eq!(out.values().len(), rows * out_nc);
        let src = &src;
        // Each output row is written by one task alone, so the result does not
        // depend on the thread count.
        out.values_mut()
            .par_chunks_mut(out_nc)
            .with_min_len((crate::parallel::MIN_PARALLEL_LEN / out_nc.max(1)).max(1))
            .enumerate()
            .for_each(|(row, dst)| {
                for &(oc, si, oi) in src {
                    dst[oc] = match (si, oi) {
                        (Some(si), Some(oi)) => op(sv[row * self_nc + si], ov[row * other_nc + oi]),
                        (Some(si), None) => sv[row * self_nc + si],
                        (None, Some(oi)) => ov[row * other_nc + oi],
                        (None, None) => {
                            unreachable!("output component comes from at least one side")
                        }
                    };
                }
            });
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
/// (zero-copy); the reference versions clone first. The scalar may stand on
/// the left too: `2.0 * field`, `1.0 - field`, `1.0 / field` apply
/// `s ∘ value` to every value.
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
        // The scalar on the left: `s ∘ value`.
        impl std::ops::$Trait<$T> for f64 {
            type Output = $T;
            fn $method(self, mut rhs: $T) -> $T {
                $crate::parallel::map_inplace(
                    $crate::containers::field::SubField::values_mut(&mut rhs),
                    move |v| self $op v,
                );
                rhs
            }
        }
        impl std::ops::$Trait<&$T> for f64 {
            type Output = $T;
            fn $method(self, rhs: &$T) -> $T {
                std::ops::$Trait::$method(self, rhs.clone())
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
/// value), the scalar standing on either side (`&a - 1.0`, `1.0 - &a`).
/// Rebuilding the aggregate can fail, so the output is a [`Result`]:
/// `(&a + &b)?`, `(&a * 2.0)?`.
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
        // scalar ∘ field: `combine_scalar` hands the value first, the scalar second.
        impl std::ops::$Trait<&$T> for f64 {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: &$T) -> Self::Output {
                $crate::containers::field::Field::combine_scalar(rhs, |v, s| s $op v, self)
            }
        }
        impl std::ops::$Trait<$T> for f64 {
            type Output = $crate::error::Result<$T>;
            fn $method(self, rhs: $T) -> Self::Output {
                $crate::containers::field::Field::combine_scalar(&rhs, |v, s| s $op v, self)
            }
        }
    };
}

// ─── Field (aggregate level) ────────────────────────────────────────────────

/// Aggregate-level view of a field.
///
/// Blanket-implemented for every [`Aggregate`] whose sub-object is a
/// [`SubField`]: `NodeField` and `ElementField`. A component may exist on
/// some subs only; `min`/`max` fold over the subs that define it and error if
/// none does — and, called without a component, over every value of every
/// sub.
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
/// // What the aggregate adds to the zone contract: the overall views and the
/// // operations that cross every zone.
/// assert_eq!(temp.components(), vec!["T".to_string()]);
/// assert_eq!(temp.map_subs(|s| Ok(s.map_all(f64::sqrt)))?
///     .get(0)?.read().value(n[0].id(), "T")?, 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait Field: Aggregate
where
    Self::Sub: SubField,
{
    /// Union of the subs' component names, first-seen order.
    fn components(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for h in self.iter() {
            let s = h.read();
            for c in s.components() {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
        }
        out
    }

    /// Smallest value of `component` across the subs defining it — or, with
    /// `None`, the smallest value of the **whole** field, every component of
    /// every zone pooled (see [`SubField::min`] for what pooling means). A zone
    /// with an empty support has no value to give and is passed over.
    ///
    /// Errors if no sub defines the component, or if every one that does is
    /// empty; with `None`, if no zone holds a single value.
    fn min(&self, component: Option<&str>) -> Result<f64> {
        fold_subs(self, component, "min", f64::min)
    }

    /// Largest value of `component` across the subs defining it — or, with
    /// `None`, the largest value of the **whole** field (see
    /// [`SubField::min`]). A zone with an empty support is passed over.
    ///
    /// Errors if no sub defines the component, or if every one that does is
    /// empty; with `None`, if no zone holds a single value.
    fn max(&self, component: Option<&str>) -> Result<f64> {
        fold_subs(self, component, "max", f64::max)
    }

    /// Sum of `component` over the field's points (`Σ` over the whole field
    /// — e.g. the resultant of a nodal force field, one component at a time).
    /// A point several zones carry counts **once**: a node on the seam between
    /// two zones holds one value, whichever zone stores it.
    ///
    /// Errors if no sub defines the component; a defining sub with an empty
    /// support contributes `0.0`. Thread-count-dependent to the last ULP.
    fn sum(&self, component: &str) -> Result<f64> {
        let guards: Vec<ReadGuard<Self::Sub>> = self.iter().map(|h| h.read()).collect();
        let zones: Vec<&Self::Sub> = guards.iter().map(|g| &**g).collect();
        if !zones.iter().any(|z| z.component_index(component).is_some()) {
            return Err(PyrucastError::Message(format!(
                "sum: no sub-field defines component {}",
                component
            )));
        }
        if let [zone] = zones[..] {
            return SubField::sum(zone, component);
        }
        Ok(sum_once(
            &zones,
            |z, c| zones[z].components()[c] == component,
            |z, cols| {
                // The one column of `component` in this zone.
                let (values, width, c) = (zones[z].values(), zones[z].component_count(), cols[0]);
                move |row| values[row * width + c]
            },
        ))
    }

    /// Squared Euclidean norm `xᵀx = Σ v²` over every value of the field
    /// (Cast3M `XTX`), a point several zones carry counted once, as in
    /// [`Field::sum`]. Thread-count-dependent to the last ULP.
    fn xtx(&self) -> f64 {
        let guards: Vec<ReadGuard<Self::Sub>> = self.iter().map(|h| h.read()).collect();
        let zones: Vec<&Self::Sub> = guards.iter().map(|g| &**g).collect();
        match zones[..] {
            [] => 0.0,
            [zone] => SubField::xtx(zone),
            _ => {
                // Zones carrying the same components count whole rows — the
                // usual field; otherwise each zone counts the columns its
                // group shares. Decided once, not at every row.
                let first = zones[0];
                let alike = zones.iter().all(|z| {
                    z.component_count() == first.component_count()
                        && z.components()
                            .iter()
                            .all(|c| first.component_index(c).is_some())
                });
                if alike {
                    sum_once(&zones, |_, _| true, |z, _| whole_row_squares(zones[z]))
                } else {
                    sum_once(&zones, |_, _| true, |z, cols| row_squares(zones[z], cols))
                }
            }
        }
    }

    /// Squared Euclidean norm restricted to the named `components`, summed over
    /// every zone that carries any of them (the aggregate twin of
    /// [`SubField::xtx_components`]), a point several zones carry counted once.
    /// A component may be spread across zones; each zone contributes the
    /// squares of whichever selected components it holds. Errors if **no**
    /// zone defines **any** of `components`.
    fn xtx_components(&self, components: &[&str]) -> Result<f64> {
        let guards: Vec<ReadGuard<Self::Sub>> = self.iter().map(|h| h.read()).collect();
        let zones: Vec<&Self::Sub> = guards.iter().map(|g| &**g).collect();
        // A zone missing all selected components contributes nothing; one
        // carrying some contributes their squares.
        let carries = |z: &&Self::Sub| components.iter().any(|c| z.component_index(c).is_some());
        if !zones.iter().any(carries) {
            return Err(PyrucastError::Message(format!(
                "xtx_components: no zone defines any of {:?}",
                components
            )));
        }
        if let [zone] = zones[..] {
            return zone.xtx_components(components);
        }
        Ok(sum_once(
            &zones,
            |z, c| components.contains(&zones[z].components()[c].as_str()),
            |z, cols| row_squares(zones[z], cols),
        ))
    }

    /// Zero-copy view of the zones, for operators doing many reads
    /// (gradient, deformation, restrict, viz, …): one read guard per sub, data
    /// read **in place** for the lifetime of the view.
    fn view(&self) -> Result<FieldView<Self::Sub>> {
        Ok(FieldView {
            components: self.components(),
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
        // fine). Assembling the result (`add_sub`) stays **serial and in
        // order** so the result decomposition is identical to the sequential
        // version.
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
    /// - A component both sides carry on one support becomes `op(self, other)`,
    ///   wherever on that support each side keeps it.
    /// - A component only one side carries on a support **passes through
    ///   unchanged** (raw, for every operator: `a - b`'s `b`-only component is
    ///   `b`, not `-b`). That is what lets `u + du` move `u_x` alone when `du`
    ///   carries `u_x` only.
    /// - The result keeps `self`'s zones, in order. A component only `other`
    ///   carries joins `self`'s zone on that support when `self` has one zone
    ///   there; it keeps a zone of its own otherwise, as does a support only
    ///   `other` carries.
    ///
    /// A support carries each component in one zone, on each side and in the
    /// result. On a nodal field, a node two supports share must also come out
    /// with one value: a value passed through on one support next to a value
    /// computed on another would break it, and is an error — bring both fields
    /// onto the same supports first (`restrict_like`). The check only runs
    /// when a component was obtained differently from one zone to the next.
    fn merge_field(&self, other: &Self, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        // Snapshot the **right** side's zones (clone them): `other` may share
        // handles with `self`, whose zones are read in place throughout.
        let rights: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();
        let (out, mixed) = merge_zones(self, &rights, op)?;
        check_merged(&out, &mixed)?;
        Ok(out)
    }

    /// Scalar product of two fields: `Σ x·y` over the components both carry on
    /// a support both carry (paired as in [`Field::merge_field`]), a point
    /// several zones carry counted once, as in [`Field::sum`]. A support — or
    /// component — present on only one side has no counterpart and contributes
    /// nothing. The reduction behind Cast3M's `XTY`.
    fn dot_field(&self, other: &Self) -> f64
    where
        Self::Sub: Clone,
    {
        // Snapshot other's zones so we never hold two guards on one zone —
        // `other` may share handles with `self`.
        let rights: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();
        let guards: Vec<ReadGuard<Self::Sub>> = self.iter().map(|h| h.read()).collect();
        let zones: Vec<&Self::Sub> = guards.iter().map(|g| &**g).collect();
        let pairs = partners(&zones, &rights);
        if let [zone] = zones[..] {
            let mut acc = 0.0;
            for (ci, pair) in pairs[0].iter().enumerate() {
                if let Some((j, rc)) = *pair {
                    acc += column_dot(zone, ci, &rights[j], rc);
                }
            }
            return acc;
        }
        sum_once(
            &zones,
            |z, c| pairs[z][c].is_some(),
            |z, cols| {
                // Per counted column: its index, and the right column it
                // pairs with — values, row width, index.
                let (lv, ln) = (zones[z].values(), zones[z].component_count());
                let links: Vec<(usize, Column)> = cols
                    .iter()
                    .map(|&c| {
                        let (j, rc) = pairs[z][c].expect("a counted column has a partner");
                        (c, (rights[j].values(), rights[j].component_count(), rc))
                    })
                    .collect();
                move |row| {
                    links
                        .iter()
                        .map(|&(lc, (rv, rn, rc))| lv[row * ln + lc] * rv[row * rn + rc])
                        .sum()
                }
            },
        )
    }

    /// Per-node/-point scalar product of two fields — Cast3M's `PSCA`. On each
    /// support both sides carry, one `"psca"` zone holding `Σ x·y` over the
    /// components both carry there (paired as in [`Field::merge_field`]); a
    /// support — or component — present on only one side has no counterpart
    /// and yields no contribution. On a nodal field, two supports summing
    /// different components may not disagree at a node they share.
    fn pscal_field(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        let rights: Vec<Self::Sub> = other.iter().map(|h| (*h.read()).clone()).collect();
        let guards: Vec<ReadGuard<Self::Sub>> = self.iter().map(|h| h.read()).collect();
        let zones: Vec<&Self::Sub> = guards.iter().map(|g| &**g).collect();
        let pairs = partners(&zones, &rights);
        let mut out = Self::default();
        // Per output zone, the components its product sums.
        let mut summed: Vec<Vec<String>> = Vec::new();
        let mut done = vec![false; zones.len()];
        for z in 0..zones.len() {
            if done[z] {
                continue;
            }
            // Every zone of `self` on this support: their components add up
            // into the one product of the support.
            let support = zones[z].support();
            let group: Vec<usize> = (z..zones.len())
                .filter(|&k| zones[k].support().same_object(&support))
                .collect();
            for &k in &group {
                done[k] = true;
            }
            if !rights.iter().any(|r| r.support().same_object(&support)) {
                continue;
            }
            let mut terms: Vec<(Column, Column)> = Vec::new();
            let mut names: Vec<String> = Vec::new();
            for &k in &group {
                for (ci, pair) in pairs[k].iter().enumerate() {
                    if let Some((j, rc)) = *pair {
                        let (l, r) = (zones[k], &rights[j]);
                        terms.push((
                            (l.values(), l.component_count(), ci),
                            (r.values(), r.component_count(), rc),
                        ));
                        names.push(l.components()[ci].clone());
                    }
                }
            }
            // Same support ⇒ identical row layout on every zone of the group,
            // on the right and on the fresh output. Each output slot is
            // written once ⇒ the result is independent of the thread count.
            let mut psca = zones[z].same_support_with(vec!["psca".to_string()])?;
            psca.values_mut()
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, o)| {
                    let mut acc = 0.0;
                    for &((lv, ln, lc), (rv, rn, rc)) in &terms {
                        acc += lv[row * ln + lc] * rv[row * rn + rc];
                    }
                    *o = acc;
                });
            names.sort();
            summed.push(names);
            out.add_sub(Handle::new(psca))?;
        }
        if summed.windows(2).any(|w| w[0] != w[1]) {
            check_merged(&out, &["psca".to_string()])?;
        }
        Ok(out)
    }

    /// Targeted update: `self op sub`, `sub` being a zone on a support of
    /// `self` — paired as in [`Field::merge_field`], every other zone carried
    /// **unchanged**. Errors if no zone shares the support, and, on a nodal
    /// field, if the updated zone and an untouched one would disagree at a node
    /// they share.
    fn merge_subfield(&self, sub: &Self::Sub, op: fn(f64, f64) -> f64) -> Result<Self>
    where
        Self: Sized,
        Self::Sub: Clone,
    {
        let support = sub.support();
        if !self
            .iter()
            .any(|h| h.read().support().same_object(&support))
        {
            return Err(PyrucastError::Message(
                "merge_subfield: no zone shares the sub-field's support".into(),
            ));
        }
        let (out, mixed) = merge_zones(self, std::slice::from_ref(sub), op)?;
        check_merged(&out, &mixed)?;
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
/// // What unifies the zone and the aggregate for element-wise mathematics:
/// // a single definition of `sqrt` serves all four field types.
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
/// // The scalar product returns a field with **one** component, named
/// // `psca`, whatever the input field's flavour.
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
    // a zone that does not define the component, or a zone with no value at all
    // — an empty support, which a selection leaves behind on the zones it did
    // not reach. The error is raised once, at the end, if *no* zone
    // contributed.
    let handles: Vec<&Handle<A::Sub>> = agg.iter().collect();
    let per_sub: Vec<Option<f64>> = handles
        .par_iter()
        .map(|h| -> Result<Option<f64>> {
            let s = h.read();
            let silent = s.values().is_empty()
                || component.is_some_and(|name| s.component_index(name).is_none());
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
            Some(name) if agg.iter().any(|h| h.read().component_index(name).is_some()) => {
                format!("{op_name}: no value for component {name} (empty support)")
            }
            Some(name) => format!("{}: no sub-field defines component {}", op_name, name),
            None => format!("{}: the field holds no value", op_name),
        })
    })
}

// ─── Points, pairing and the two rules of a field ───────────────────────────
//
// A field obeys two rules, which every operation keeps:
//
// 1. on one support, a component lives in one zone;
// 2. a point several zones carry — a node, two supports meeting — holds one
//    value per component.
//
// The helpers below are what keeps them: a reduction counts a shared point
// once, an operation pairs zones per (support, component), and a result that
// could break the second rule is checked.

/// The points a zone's rows stand at, read in place: it holds the read guard
/// of the zone's POI1 support, whose cell `i` lists the node of row `i`. What
/// [`SubField::row_points`] returns for a nodal zone.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::Node;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::node_field::SubNodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// let support = mesh::poi1_from_nodes(&[b.clone()])?;
/// let t = SubNodeField::from_poi1(&support.get(0)?, vec!["T".into()])?;
/// let points = t.row_points().expect("a nodal field stands at nodes");
/// assert_eq!(points.ids(), &[b.id()]);
/// // Every point lies below the bound, whichever nodes the zone holds.
/// assert!(points.ids().iter().all(|id| (id.0 as usize) < points.bound()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct RowPoints {
    support: ReadGuard<SubMesh>,
    bound: usize,
}

impl RowPoints {
    /// The points of a POI1 support, whose cell `i` holds the node of row `i`.
    pub(crate) fn of_poi1(support: ReadGuard<SubMesh>) -> Self {
        let bound = support.coords().read().capacity();
        RowPoints { support, bound }
    }

    /// The point of each row, in row order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::SubNodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// let support = mesh::poi1_from_nodes(&[b.clone(), a.clone()])?;
    /// let t = SubNodeField::from_poi1(&support.get(0)?, vec!["T".into()])?;
    /// // In the order of the support, not of the node numbers.
    /// assert_eq!(t.row_points().unwrap().ids(), &[b.id(), a.id()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn ids(&self) -> &[NodeId] {
        self.support.connectivity()
    }

    /// Above every point: the capacity of the `Coords` the points live in,
    /// known without reading the points.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::SubNodeField;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// let support = mesh::poi1_from_nodes(&[a.clone()])?;
    /// let t = SubNodeField::from_poi1(&support.get(0)?, vec!["T".into()])?;
    /// // The zone holds one node, the `Coords` two: the bound counts both.
    /// assert_eq!(t.row_points().unwrap().bound(), coords.read().capacity());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn bound(&self) -> usize {
        self.bound
    }
}

/// `row ↦ Σ v²` over every column of `zone`.
fn whole_row_squares<S: SubField>(zone: &S) -> impl Fn(usize) -> f64 + Sync + use<'_, S> {
    let (values, width) = (zone.values(), zone.component_count());
    move |row| {
        values[row * width..(row + 1) * width]
            .iter()
            .map(|v| v * v)
            .sum()
    }
}

/// `row ↦ Σ v²` over the columns `cols` of `zone`.
fn row_squares<'a, S: SubField>(
    zone: &'a S,
    cols: &[usize],
) -> impl Fn(usize) -> f64 + Sync + use<'a, S> {
    let (values, width) = (zone.values(), zone.component_count());
    let cols = cols.to_vec();
    move |row| {
        let r = &values[row * width..(row + 1) * width];
        cols.iter().map(|&c| r[c] * r[c]).sum()
    }
}

/// Rows a task takes at a time. Chunks, not single rows, are the parallel
/// items: a chunk sums its rows in order and posts its points in runs.
const CHUNK_ROWS: usize = 16 * MIN_PARALLEL_LEN;

/// `Σ term(z, cols)(row)` over every row of `zones`, a point several zones
/// carry counted **once**, by the first zone that carries it. `counts(z, c)`
/// says whether column `c` of zone `z` takes part; `cols` lists the columns of
/// zone `z` counted together — those whose components the same zones count —
/// and `term(z, cols)` is resolved once per zone, before its loop over rows.
///
/// Per such group, one bit per point, and the zones one after the other, each
/// split across the threads: the first counts every row and posts its points;
/// the next ones count only the rows whose point no earlier zone posted, then
/// post theirs; the last only tests. A zone lists each point once, so two
/// threads never handle one point, and which zone counts a point never
/// depends on the threads. Each chunk sums its rows in order and the chunks
/// are summed in order: the result is the same bit for bit whatever the
/// thread count.
fn sum_once<S: SubField, F: Fn(usize) -> f64 + Sync>(
    zones: &[&S],
    counts: impl Fn(usize, usize) -> bool,
    term: impl Fn(usize, &[usize]) -> F,
) -> f64 {
    let rows = |z: usize| zones[z].values().len() / zones[z].component_count().max(1);
    let mut acc = 0.0;
    for group in column_groups(zones, counts) {
        let (key, cols) = (&group.zones, &group.columns);
        let last = key.len() - 1;
        // A point repeats only across two zones or more, of rows standing at
        // points — nodal zones.
        let bound = if last == 0 {
            None
        } else {
            point_bound(zones, key)
        };
        let Some(bound) = bound else {
            for (k, &z) in key.iter().enumerate() {
                acc += zone_pass(rows(z), Pass::Every, term(z, &cols[k]));
            }
            continue;
        };
        let seen: Vec<AtomicU64> = (0..bound.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        for (k, &z) in key.iter().enumerate() {
            let points = zones[z].row_points().expect("point_bound read them");
            let pass = if k == 0 {
                Pass::Post(points.ids(), &seen)
            } else {
                Pass::Test(points.ids(), &seen)
            };
            acc += zone_pass(rows(z), pass, term(z, &cols[k]));
            if k != 0 && k != last {
                post_points(points.ids(), &seen);
            }
        }
    }
    acc
}

/// Components carried by the same zones: those zones, in order, and per zone
/// the columns of the group's components, in one order across the zones —
/// `columns[k][j]` and `columns[l][j]` hold the same component.
struct ColumnGroup {
    zones: Vec<usize>,
    columns: Vec<Vec<usize>>,
}

/// The columns `takes(z, c)` keeps — column `c` of zone `z` — grouped by the
/// zones carrying their component, in the order the components are first met.
fn column_groups<S: SubField>(
    zones: &[&S],
    takes: impl Fn(usize, usize) -> bool,
) -> Vec<ColumnGroup> {
    let mut names: Vec<&str> = Vec::new();
    for z in zones {
        for c in z.components() {
            if !names.contains(&c.as_str()) {
                names.push(c);
            }
        }
    }
    let mut groups: Vec<ColumnGroup> = Vec::new();
    for name in names {
        let cols: Vec<(usize, usize)> = zones
            .iter()
            .enumerate()
            .filter_map(|(z, zone)| {
                let ci = zone.component_index(name)?;
                takes(z, ci).then_some((z, ci))
            })
            .collect();
        if cols.is_empty() {
            continue;
        }
        let key: Vec<usize> = cols.iter().map(|&(z, _)| z).collect();
        let g = match groups.iter().position(|g| g.zones == key) {
            Some(g) => g,
            None => {
                groups.push(ColumnGroup {
                    zones: key,
                    columns: vec![Vec::new(); cols.len()],
                });
                groups.len() - 1
            }
        };
        for (k, &(_, ci)) in cols.iter().enumerate() {
            groups[g].columns[k].push(ci);
        }
    }
    groups
}

/// One past the largest point of the zones `key`, or `None` when their rows
/// stand at no point. Read off the `Coords` capacity when the point set it
/// sizes — one bit per node — costs no more to clear than one read of the
/// points, 4 bytes per row; found by a pass over the points otherwise, for a
/// small field on a large mesh.
pub(crate) fn point_bound<S: SubField>(zones: &[&S], key: &[usize]) -> Option<usize> {
    let (mut capacity, mut rows) = (0, 0);
    for &z in key {
        let points = zones[z].row_points()?;
        capacity = capacity.max(points.bound());
        rows += points.ids().len();
    }
    if capacity <= 32 * rows {
        return Some(capacity);
    }
    let mut bound = 0;
    for &z in key {
        let points = zones[z].row_points().expect("the first loop read them");
        if let Some(top) = points.ids().iter().map(|id| id.0 as usize + 1).max() {
            bound = bound.max(top);
        }
    }
    Some(bound)
}

/// What a pass over a zone's rows does with their points — the point of each
/// row, and the point set — in [`sum_once`].
#[derive(Clone, Copy)]
enum Pass<'a> {
    /// Count every row: no other zone of the group can carry its point.
    Every,
    /// Count every row and post its point: the first zone of the group.
    Post(&'a [NodeId], &'a [AtomicU64]),
    /// Count only the rows whose point no earlier zone posted.
    Test(&'a [NodeId], &'a [AtomicU64]),
}

/// `Σ term(row)` over the rows of one zone that `pass` counts, chunk by chunk
/// across threads; the pass is chosen per chunk, never per row. The bits are
/// read and posted `Relaxed`: a zone only reads bits posted by earlier zones,
/// whose passes returned before it started.
fn zone_pass(rows: usize, pass: Pass, term: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partial: Vec<f64> = (0..rows.div_ceil(CHUNK_ROWS))
        .into_par_iter()
        .map(|k| {
            let chunk = k * CHUNK_ROWS..rows.min((k + 1) * CHUNK_ROWS);
            let mut acc = 0.0;
            match pass {
                Pass::Every => {
                    for row in chunk {
                        acc += term(row);
                    }
                }
                Pass::Post(ids, seen) => {
                    let ids = &ids[chunk.clone()];
                    let mut run = Run::default();
                    for (row, id) in chunk.zip(ids) {
                        run.post(seen, id.0 as usize);
                        acc += term(row);
                    }
                    run.flush(seen);
                }
                Pass::Test(ids, seen) => {
                    let ids = &ids[chunk.clone()];
                    for (row, id) in chunk.zip(ids) {
                        if !is_posted(seen, id.0 as usize) {
                            acc += term(row);
                        }
                    }
                }
            }
            acc
        })
        .collect();
    partial.iter().sum()
}

/// Post every point of `ids`, chunk by chunk across threads.
fn post_points(ids: &[NodeId], seen: &[AtomicU64]) {
    ids.par_chunks(CHUNK_ROWS).for_each(|chunk| {
        let mut run = Run::default();
        for id in chunk {
            run.post(seen, id.0 as usize);
        }
        run.flush(seen);
    });
}

/// Whether point `p` is posted in `seen`.
fn is_posted(seen: &[AtomicU64], p: usize) -> bool {
    seen[p / 64].load(Ordering::Relaxed) & (1 << (p % 64)) != 0
}

/// Points posted in runs: consecutive points sharing a 64-bit word of the
/// point set cost one atomic `OR` together.
#[derive(Default)]
struct Run {
    word: usize,
    bits: u64,
}

impl Run {
    fn post(&mut self, seen: &[AtomicU64], p: usize) {
        if p / 64 != self.word {
            self.flush(seen);
            self.word = p / 64;
        }
        self.bits |= 1 << (p % 64);
    }

    fn flush(&mut self, seen: &[AtomicU64]) {
        if self.bits != 0 {
            seen[self.word].fetch_or(self.bits, Ordering::Relaxed);
            self.bits = 0;
        }
    }
}

/// `Σ x·y` of column `lc` of `left` against column `rc` of `right`, two zones
/// on one support (same rows, same order).
fn column_dot<S: SubField + ?Sized>(left: &S, lc: usize, right: &S, rc: usize) -> f64 {
    let (ln, rn) = (
        left.component_count().max(1),
        right.component_count().max(1),
    );
    let (lv, rv) = (left.values(), right.values());
    (0..lv.len() / ln)
        .into_par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|row| lv[row * ln + lc] * rv[row * rn + rc])
        .sum()
}

/// For each column of each left zone, the right zone carrying that component
/// on the same support and its column there — `None` when none does. One at
/// most: on a support, a component lives in one zone.
fn partners<S: SubField>(lefts: &[&S], rights: &[S]) -> Vec<Vec<Option<(usize, usize)>>> {
    lefts
        .iter()
        .map(|l| {
            let support = l.support();
            let there: Vec<usize> = (0..rights.len())
                .filter(|&j| rights[j].support().same_object(&support))
                .collect();
            l.components()
                .iter()
                .map(|name| {
                    there
                        .iter()
                        .find_map(|&j| rights[j].component_index(name).map(|rc| (j, rc)))
                })
                .collect()
        })
        .collect()
}

/// A column read in place: its zone's values, their row width, its index.
type Column<'a> = (&'a [f64], usize, usize);

/// Where a column of a combined zone reads: a left column and its right
/// partner — (zone, column) — or one of the two alone.
#[derive(Clone, Copy)]
enum Source {
    Both(usize, (usize, usize)),
    Left(usize),
    Right((usize, usize)),
}

/// How a column of a combined zone was obtained. Two zones whose columns of
/// one component were obtained the same way agree wherever they meet — each
/// side holds one value per point; only a component obtained differently from
/// one zone to the next can break the second rule.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Taken {
    Both,
    Left,
    Right,
}

/// `left op rights`, paired per (support, component) — the body of
/// [`Field::merge_field`] and [`Field::merge_subfield`]. Returns the result and
/// the components obtained differently from one zone to the next (see
/// [`Taken`]), which alone can break the second rule.
fn merge_zones<A>(left: &A, rights: &[A::Sub], op: fn(f64, f64) -> f64) -> Result<(A, Vec<String>)>
where
    A: Aggregate,
    A::Sub: SubField + Clone,
{
    let guards: Vec<ReadGuard<A::Sub>> = left.iter().map(|h| h.read()).collect();
    let lefts: Vec<&A::Sub> = guards.iter().map(|g| &**g).collect();
    let pairs = partners(&lefts, rights);
    let mut used: Vec<Vec<bool>> = rights
        .iter()
        .map(|r| vec![false; r.component_count()])
        .collect();
    let mut how: Vec<(String, Taken, bool)> = Vec::new();
    let mut note = |name: &str, t: Taken| match how.iter_mut().find(|(n, _, _)| n == name) {
        Some((_, first, mixed)) => *mixed |= *first != t,
        None => how.push((name.to_string(), t, false)),
    };
    let mut out = A::empty();
    for (z, l) in lefts.iter().enumerate() {
        let support = l.support();
        let alone = lefts
            .iter()
            .filter(|k| k.support().same_object(&support))
            .count()
            == 1;
        let mut components = l.components().to_vec();
        let mut cols: Vec<Source> = Vec::new();
        for (lc, pair) in pairs[z].iter().enumerate() {
            let (source, taken) = match *pair {
                Some((j, rc)) => {
                    used[j][rc] = true;
                    (Source::Both(lc, (j, rc)), Taken::Both)
                }
                None => (Source::Left(lc), Taken::Left),
            };
            cols.push(source);
            note(&components[lc], taken);
        }
        // What only the right carries on this support joins the left zone
        // when it is the only one there; with several, none is *the* zone.
        if alone {
            for (j, r) in rights.iter().enumerate() {
                if !r.support().same_object(&support) {
                    continue;
                }
                for (rc, name) in r.components().iter().enumerate() {
                    if !used[j][rc] {
                        used[j][rc] = true;
                        note(name, Taken::Right);
                        components.push(name.clone());
                        cols.push(Source::Right((j, rc)));
                    }
                }
            }
        }
        let zone = if cols.iter().all(|c| matches!(c, Source::Left(_))) {
            (**l).clone()
        } else {
            combine_columns(*l, rights, components, &cols, op)?
        };
        out.add_sub(Handle::new(zone))?;
    }
    for (j, r) in rights.iter().enumerate() {
        let rest: Vec<String> = r
            .components()
            .iter()
            .enumerate()
            .filter(|&(rc, _)| !used[j][rc])
            .map(|(_, name)| name.clone())
            .collect();
        if rest.is_empty() {
            continue;
        }
        for name in &rest {
            note(name, Taken::Right);
        }
        let zone = if rest.len() == r.component_count() {
            r.clone()
        } else {
            r.select_components(rest)?
        };
        out.add_sub(Handle::new(zone))?;
    }
    let mixed = how
        .into_iter()
        .filter(|(_, _, mixed)| *mixed)
        .map(|(name, _, _)| name)
        .collect();
    Ok((out, mixed))
}

/// A zone on `left`'s support whose column `k` is named `components[k]` and
/// filled, row by row, from `cols[k]`: `op(left, right)` when it reads both
/// sides, the one value it reads otherwise. Same support ⇒ same rows in the
/// same order on every zone read. The columns are sorted by kind once, so the
/// loop over rows tests nothing.
fn combine_columns<S: SubField>(
    left: &S,
    rights: &[S],
    components: Vec<String>,
    cols: &[Source],
    op: fn(f64, f64) -> f64,
) -> Result<S> {
    let mut out = left.same_support_with(components)?;
    let nc = out.component_count();
    let (lv, ln) = (left.values(), left.component_count());
    let right =
        |j: usize, rc: usize| -> Column { (rights[j].values(), rights[j].component_count(), rc) };
    // Per kind, (output column, what it reads).
    let (mut both, mut left_only, mut right_only) = (Vec::new(), Vec::new(), Vec::new());
    for (k, source) in cols.iter().enumerate() {
        match *source {
            Source::Both(lc, (j, rc)) => both.push((k, lc, right(j, rc))),
            Source::Left(lc) => left_only.push((k, lc)),
            Source::Right((j, rc)) => right_only.push((k, right(j, rc))),
        }
    }
    // Each output row is written by one task alone, so the result does not
    // depend on the thread count.
    out.values_mut()
        .par_chunks_mut(nc.max(1))
        .with_min_len((MIN_PARALLEL_LEN / nc.max(1)).max(1))
        .enumerate()
        .for_each(|(row, dst)| {
            let l = &lv[row * ln..(row + 1) * ln];
            for &(k, lc, (rv, rn, rc)) in &both {
                dst[k] = op(l[lc], rv[row * rn + rc]);
            }
            for &(k, lc) in &left_only {
                dst[k] = l[lc];
            }
            for &(k, (rv, rn, rc)) in &right_only {
                dst[k] = rv[row * rn + rc];
            }
        });
    Ok(out)
}

/// The second rule on the result of an operation, for the `components`
/// obtained differently from one zone to the next — the only ones that can
/// break it.
fn check_merged<A>(out: &A, components: &[String]) -> Result<()>
where
    A: Aggregate,
    A::Sub: SubField,
{
    if components.is_empty() {
        return Ok(());
    }
    check_points_agree(out, components).map_err(|e| {
        PyrucastError::Message(format!(
            "{e}: the two fields do not carry this component on the same \
             supports, so the zones meeting at this node were not combined alike \
             — bring both fields onto the same supports first (restrict_like)"
        ))
    })
}

/// The second rule for `components`: every point several zones of `field`
/// carry holds one value per component, compared exactly. A field whose zones
/// share no point passes at once. Errors on the first point where two zones
/// disagree.
///
/// One pass per group of components the same zones carry
/// ([`column_groups`]) — usually a single one: per point, the first (zone,
/// row) holding it; at a point met again, every component of the group is
/// compared with that row. Values are read there only.
pub(crate) fn check_points_agree<A>(field: &A, components: &[String]) -> Result<()>
where
    A: Aggregate,
    A::Sub: SubField,
{
    if field.len() < 2 {
        return Ok(());
    }
    let guards: Vec<ReadGuard<A::Sub>> = field.iter().map(|h| h.read()).collect();
    let zones: Vec<&A::Sub> = guards.iter().map(|g| &**g).collect();
    let checked = |z: usize, c: usize| components.contains(&zones[z].components()[c]);
    const UNSEEN: u32 = u32::MAX;
    for group in column_groups(&zones, checked) {
        let (key, cols) = (&group.zones, &group.columns);
        // A point repeats only across two zones or more, of rows standing at
        // points — nodal zones.
        if key.len() < 2 {
            continue;
        }
        let Some(bound) = point_bound(&zones, key) else {
            continue;
        };
        // Per point, the first (zone of the group, row) holding it. Points are
        // read in place, one zone at a time: two zones may share a support,
        // whose lock is never taken twice.
        let mut first = vec![(UNSEEN, 0u32); bound];
        let points = zones[key[0]].row_points().expect("point_bound read them");
        for (row, p) in points.ids().iter().enumerate() {
            first[p.0 as usize] = (0, row as u32);
        }
        drop(points);
        for (k, &z) in key.iter().enumerate().skip(1) {
            let points = zones[z].row_points().expect("point_bound read them");
            let (values, width) = (zones[z].values(), zones[z].component_count());
            for (row, p) in points.ids().iter().enumerate() {
                let slot = &mut first[p.0 as usize];
                if slot.0 == UNSEEN {
                    *slot = (k as u32, row as u32);
                    continue;
                }
                let (hk, hrow) = (slot.0 as usize, slot.1 as usize);
                let held = zones[key[hk]];
                let (hv, hw) = (held.values(), held.component_count());
                for (&hc, &c) in cols[hk].iter().zip(&cols[k]) {
                    let (w, v) = (hv[hrow * hw + hc], values[row * width + c]);
                    if w != v {
                        return Err(PyrucastError::Message(format!(
                            "incoherent {}: node {}, component {}: {} \u{2260} {}",
                            A::type_name(),
                            p,
                            held.components()[hc],
                            w,
                            v
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

/// The first rule, for a zone about to join `field`: refuse it when a zone of
/// `field` already carries one of its components on its support.
pub(crate) fn check_component_free<A>(field: &A, zone: &Handle<A::Sub>) -> Result<()>
where
    A: Aggregate,
    A::Sub: SubField,
{
    // Read the newcomer once, guard dropped: it may already be in `field`.
    let (support, components) = {
        let z = zone.read();
        (z.support(), z.components().to_vec())
    };
    for (i, h) in field.iter().enumerate() {
        let held = h.read();
        if !held.support().same_object(&support) {
            continue;
        }
        if let Some(c) = components
            .iter()
            .find(|c| held.component_index(c).is_some())
        {
            return Err(PyrucastError::Message(format!(
                "{}: zone {i} already carries {c} on this support — on one support, \
                 each component lives in one zone",
                A::type_name()
            )));
        }
    }
    Ok(())
}

/// `a + b` between partial sums on their way to an assembly
/// ([`NodeField::summed`]): what two terms load on one support adds up, the
/// rest stays side by side, and a node two supports share keeps one partial
/// sum in each — what the assembly adds next. Not a field yet, so the second
/// rule is not checked.
pub(crate) fn add_partials<A>(a: &A, b: &A) -> Result<A>
where
    A: Aggregate,
    A::Sub: SubField + Clone,
{
    let rights: Vec<A::Sub> = b.iter().map(|h| (*h.read()).clone()).collect();
    Ok(merge_zones(a, &rights, |x, y| x + y)?.0)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::{NodeField, SubNodeField};
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
        assert_eq!(Field::components(&ef), vec!["k", "E"]);
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
        assert!((Field::xtx(&ef) - (x0 + x1)).abs() < 1e-9);
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
        assert_eq!(Field::components(&out), vec!["k"]);
        // zone0 handle shared untouched; zone1 rebuilt (a new zone).
        assert!(out.get(0).unwrap().same_object(&ef.get(0).unwrap()));
        assert!(!out.get(1).unwrap().same_object(&ef.get(1).unwrap()));

        // Keep only "E": zone0 (no E) is dropped, zone1 kept.
        let only_e = ef.filter_components("E").unwrap();
        assert_eq!(only_e.len(), 1);
        assert_eq!(Field::components(&only_e), vec!["E"]);

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
        assert_eq!(Field::components(&primal), vec!["u_x", "u_y"]);
        assert_eq!(primal.value(nodes[0].id(), "u_x").unwrap(), 1.0);
        assert!(primal.value_opt(nodes[0].id(), "lambda").is_none());
    }

    #[test]
    fn field_rename_component_renames_matching_zones_only() {
        let ef = make_two_zone_element_field(); // zone0: [k], zone1: [E, k]
        let out = ef.rename_component("E", "young").unwrap();
        // zone0 has no E ⇒ handle shared; zone1 rebuilt with the new name.
        assert!(out.get(0).unwrap().same_object(&ef.get(0).unwrap()));
        assert_eq!(Field::components(&out), vec!["k", "young"]);
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
        // The zone-level `&f + &g` operator is union/passthrough: it does not
        // require the same component sets.
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
        assert_eq!(nf.dot_field(&nf), 38.0);
    }

    #[test]
    fn field_dot_disjoint_supports_is_zero() {
        use crate::containers::node_field::NodeField;
        let (sm_a, _) = poi1_support(1);
        let (sm_b, _) = poi1_support(1);
        let a = NodeField::from_sub(SubNodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap());
        let b = NodeField::from_sub(SubNodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap());
        // No zone of `a` shares a support with `b`: nothing to pair ⇒ 0.
        assert_eq!(a.dot_field(&b), 0.0);
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
        // Distinct supports ⇒ union: both zones pass through unchanged. Two
        // spaces on one mesh are two supports on one `Coords`.
        let fes = one_tri3_fes();
        let other = FiniteElementSpace::lagrange1(&fes.mesh().unwrap()).unwrap();
        let f = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let g = ElementField::new(&other, vec!["E".into()]).unwrap();
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

    /// A unit square cut along its diagonal into two TRI3 zones that share
    /// nodes 1 and 2: the left zone, the right one, and the two together.
    fn square_in_two() -> (Mesh, Mesh, Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let zone = |a: usize, b: usize, c: usize| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n[a].id(), n[b].id(), n[c].id()]).unwrap();
            Mesh::from_submesh(sm)
        };
        let (left, right) = (zone(0, 1, 2), zone(1, 3, 2));
        let both = left.union(&right).unwrap();
        (left, right, both, n)
    }

    #[test]
    fn a_node_two_zones_share_counts_once_in_the_reductions() {
        let (_, _, both, _) = square_in_two();
        let u = NodeField::new(&both, vec!["T".into()]).unwrap();
        u.add_to_component("T", 1.0).unwrap();
        // Four nodes, two of them in both zones: each counted once.
        assert_eq!(Field::sum(&u, "T").unwrap(), 4.0);
        assert_eq!(Field::xtx(&u), 4.0);
        assert_eq!(Field::xtx_components(&u, &["T"]).unwrap(), 4.0);
        assert_eq!(u.dot_field(&u), 4.0);
        // The same values in a single zone give the same numbers.
        let one = NodeField::new(
            &crate::ops::mesh::poi1_from_nodes(&square_in_two().3).unwrap(),
            vec!["T".into()],
        )
        .unwrap();
        one.add_to_component("T", 1.0).unwrap();
        assert_eq!(Field::sum(&one, "T").unwrap(), 4.0);
    }

    /// A nodal field on `zones` (node indices into `n`), its value at node
    /// `i` set to `f(i)` in every zone carrying it — the second rule kept.
    fn field_on(n: &[Node], zones: &[&[usize]], f: impl Fn(usize) -> f64) -> NodeField {
        let mut mesh = Mesh::default();
        for z in zones {
            let nodes: Vec<Node> = z.iter().map(|&i| n[i].clone()).collect();
            mesh.add_sub(
                crate::ops::mesh::poi1_from_nodes(&nodes)
                    .unwrap()
                    .get(0)
                    .unwrap(),
            )
            .unwrap();
        }
        let u = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        for h in &u {
            let mut zone = h.write();
            for nid in zone.node_ids() {
                let i = n.iter().position(|m| m.id() == nid).unwrap();
                zone.set_by_node(nid, 0, f(i)).unwrap();
            }
        }
        u
    }

    fn nodes_on_a_line(count: usize) -> Vec<Node> {
        let coords = Handle::new(Coords::new(1).unwrap());
        (0..count)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect()
    }

    #[test]
    fn three_zones_in_a_row_count_each_node_once() {
        let n = nodes_on_a_line(6);
        // The middle zone shares node 2 with the first and node 4 with the
        // last, which also takes node 0 back from the first.
        let u = field_on(&n, &[&[0, 1, 2], &[2, 3, 4], &[4, 5, 0]], |i| {
            (i + 1) as f64
        });
        let distinct: f64 = (1..=6).map(|v| v as f64).sum();
        let squares: f64 = (1..=6).map(|v| (v * v) as f64).sum();
        assert_eq!(Field::sum(&u, "T").unwrap(), distinct);
        assert_eq!(Field::xtx(&u), squares);
        assert_eq!(Field::xtx_components(&u, &["T"]).unwrap(), squares);
        assert_eq!(u.dot_field(&u), squares);
    }

    #[test]
    fn zones_carrying_different_components_count_each_value_once() {
        let n = nodes_on_a_line(4);
        let mut mesh = Mesh::default();
        for z in [&n[0..3], &n[2..4]] {
            mesh.add_sub(
                crate::ops::mesh::poi1_from_nodes(z)
                    .unwrap()
                    .get(0)
                    .unwrap(),
            )
            .unwrap();
        }
        // T on both zones, which share node 2; P on the first zone only.
        let u = NodeField::with(&mesh, &[vec!["T".into(), "P".into()], vec!["T".into()]]).unwrap();
        u.add_to_component("T", 1.0).unwrap();
        u.add_to_component("P", 2.0).unwrap();
        // Four distinct nodes carry T = 1, three carry P = 2.
        assert_eq!(Field::xtx(&u), 4.0 * 1.0 + 3.0 * 4.0);
        assert_eq!(Field::xtx_components(&u, &["T"]).unwrap(), 4.0);
    }

    #[test]
    fn a_small_field_on_a_large_coords_counts_each_node_once() {
        // Six nodes carried out of ten thousand: the point set is sized by a
        // pass over the six rather than by the whole `Coords`.
        let n = nodes_on_a_line(10_000);
        let u = field_on(&n, &[&[9_000, 9_001, 9_999], &[9_999, 3, 4]], |_| 1.0);
        assert_eq!(Field::sum(&u, "T").unwrap(), 5.0);
        assert_eq!(Field::xtx(&u), 5.0);
    }

    #[test]
    fn reductions_do_not_depend_on_the_thread_count() {
        // Two zones of several chunks each, sharing a band of nodes, with
        // values whose sum rounds differently in a different order.
        let n = nodes_on_a_line(30_000);
        let left: Vec<usize> = (0..16_000).collect();
        let right: Vec<usize> = (14_000..30_000).collect();
        let f = |i: usize| (i as f64).sin() * 1e3 + 1.0 / (i as f64 + 1.0);
        let u = field_on(&n, &[&left, &right], f);
        let reduce = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                [
                    Field::sum(&u, "T").unwrap(),
                    Field::xtx(&u),
                    u.dot_field(&u),
                ]
            })
        };
        let one = reduce(1);
        assert_eq!(one.map(f64::to_bits), reduce(4).map(f64::to_bits));
        // Each node once: the sum over the 30 000 distinct nodes.
        let distinct: f64 = (0..30_000).map(f).sum();
        assert!((one[0] - distinct).abs() < 1e-9 * distinct.abs().max(1.0));
    }

    #[test]
    fn regions_meeting_at_a_node_cannot_be_combined_unlike() {
        let (left, _, both, n) = square_in_two();
        let g = NodeField::new(&left, vec!["T".into()]).unwrap();
        g.add_to_component("T", 1.0).unwrap();
        let c = NodeField::new(&both, vec!["T".into()]).unwrap();
        c.add_to_component("T", 10.0).unwrap();
        // Left zone: computed from both fields; right zone: `c` alone. At the
        // shared nodes the two would disagree (11 against 10).
        let err = (&g + &c).unwrap_err().to_string();
        assert!(err.contains("restrict_like"), "{err}");
        // On the same supports, the sum is computed everywhere alike.
        let g_on_both = crate::ops::node_field::restrict(&g, &both).unwrap();
        let s = (&g_on_both + &c).unwrap();
        assert_eq!(s.value(n[1].id(), "T").unwrap(), 11.0);
        for z in &s {
            assert_eq!(z.read().value(n[1].id(), "T").unwrap(), 11.0);
        }
    }

    #[test]
    fn a_component_moved_on_every_zone_needs_no_check() {
        let (_, _, both, n) = square_in_two();
        let u = NodeField::new(&both, vec!["u_x".into(), "u_y".into()]).unwrap();
        u.add_to_component("u_y", 5.0).unwrap();
        let du = NodeField::new(&both, vec!["u_x".into()]).unwrap();
        du.add_to_component("u_x", 1.0).unwrap();
        // `u_x` computed in every zone, `u_y` passed through in every zone.
        let s = (&u + &du).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.value(n[1].id(), "u_x").unwrap(), 1.0);
        assert_eq!(s.value(n[1].id(), "u_y").unwrap(), 5.0);
    }

    #[test]
    fn side_by_side_element_zones_pair_per_component() {
        let fes = one_tri3_fes();
        let left = ElementField::new(&fes, vec!["E".into(), "k".into()]).unwrap();
        {
            let mut z = left.get(0).unwrap().write();
            z.set_uniform("E", 10.0).unwrap();
            z.set_uniform("k", 20.0).unwrap();
        }
        let a = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
        {
            let mut z = a.get(0).unwrap().write();
            z.set_uniform("E", 1.0).unwrap();
            z.set_uniform("nu", 0.3).unwrap();
        }
        let b = ElementField::new(&fes, vec!["k".into()]).unwrap();
        b.get(0).unwrap().write().set_uniform("k", 2.0).unwrap();
        // Two zones on one subspace, each component in one of them.
        let right = a.union(&b).unwrap();
        assert_eq!(right.len(), 2);

        let s = (&left + &right).unwrap();
        assert_eq!(s.len(), 1, "the right's extras join the left's only zone");
        let z = s.get(0).unwrap().read();
        assert_eq!(z.value(0, 0, "E").unwrap(), 11.0);
        assert_eq!(z.value(0, 0, "k").unwrap(), 22.0);
        assert_eq!(z.value(0, 0, "nu").unwrap(), 0.3);
        drop(z);
        // Paired the same way from both ends: the product is symmetric.
        let points = left.get(0).unwrap().read().gauss_count() as f64;
        assert_eq!(left.dot_field(&right), 50.0 * points);
        assert_eq!(right.dot_field(&left), 50.0 * points);
        // One product per subspace, over every component both carry.
        let p = left.pscal_field(&right).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.get(0).unwrap().read().value(0, 0, "psca").unwrap(), 50.0);
        let p = right.pscal_field(&left).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.get(0).unwrap().read().value(0, 0, "psca").unwrap(), 50.0);
    }

    #[test]
    fn extras_keep_a_zone_of_their_own_when_the_left_has_several() {
        let fes = one_tri3_fes();
        let left = ElementField::new(&fes, vec!["E".into()])
            .unwrap()
            .union(&ElementField::new(&fes, vec!["k".into()]).unwrap())
            .unwrap();
        let right = ElementField::new(&fes, vec!["E".into(), "alpha".into()]).unwrap();
        right
            .get(0)
            .unwrap()
            .write()
            .set_uniform("alpha", 1e-5)
            .unwrap();
        let s = (&left + &right).unwrap();
        // The left's two zones, then `alpha` alone: no zone is *the* one.
        let comps: Vec<Vec<String>> = s.iter().map(|h| h.read().components().to_vec()).collect();
        assert_eq!(
            comps,
            vec![
                vec!["E".to_string()],
                vec!["k".into()],
                vec!["alpha".into()]
            ]
        );
    }

    #[test]
    fn a_component_joins_a_support_once() {
        let (left, _, _, _) = square_in_two();
        let mut f = NodeField::new(&left, vec!["T".into()]).unwrap();
        let again = NodeField::new(&left, vec!["T".into(), "q".into()]).unwrap();
        let err = f.add_sub(again.get(0).unwrap()).unwrap_err().to_string();
        assert!(err.contains("already carries T"), "{err}");
        // Another component on the same support is welcome.
        f.add_sub(
            NodeField::new(&left, vec!["q".into()])
                .unwrap()
                .get(0)
                .unwrap(),
        )
        .unwrap();

        let fes = one_tri3_fes();
        let mut e = ElementField::new(&fes, vec!["E".into()]).unwrap();
        let twice = ElementField::new(&fes, vec!["E".into()]).unwrap();
        assert!(e.add_sub(twice.get(0).unwrap()).is_err());
    }

    #[test]
    fn a_field_is_not_built_on_a_zone_listed_twice() {
        let (left, _, _, _) = square_in_two();
        let mut twice = Mesh::empty();
        twice.add_sub(left.get(0).unwrap()).unwrap();
        twice.add_sub(left.get(0).unwrap()).unwrap();
        let err = NodeField::new(&twice, vec!["T".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("zones 0 and 1"), "{err}");
        // Two component lists on the twice-listed zone: two zones, each its own.
        let f = NodeField::with(&twice, &[vec!["T".into()], vec!["q".into()]]).unwrap();
        assert_eq!(f.len(), 2);
        assert!(NodeField::with(&twice, &[vec!["T".into()], vec!["T".into()]]).is_err());

        let fes = one_tri3_fes();
        let mut fes_twice = FiniteElementSpace::empty();
        fes_twice.add_sub(fes.get(0).unwrap()).unwrap();
        fes_twice.add_sub(fes.get(0).unwrap()).unwrap();
        let err = ElementField::new(&fes_twice, vec!["E".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("subspaces 0 and 1"), "{err}");
        assert!(ElementField::with(&fes_twice, &[vec!["E".into()], vec!["E".into()]]).is_err());
    }

    /// A zone on an empty support — what a selection leaves on the zones it
    /// did not reach — has no value to give: `min` / `max` pass it over, and
    /// only a field whose every defining zone is empty has no answer.
    #[test]
    fn min_and_max_pass_over_a_zone_with_an_empty_support() {
        let full = make_node_field(&[3.0, -2.0, 7.0]);
        let coords = full.support().read().coords();
        let empty_support = Handle::new(SubMesh::new(coords, ElementType::POI1));
        let empty = SubNodeField::from_poi1(&empty_support, vec!["T".into()]).unwrap();
        let mut field = NodeField::default();
        field.add_sub(Handle::new(empty)).unwrap();
        field.add_sub(Handle::new(full)).unwrap();
        assert_eq!(Field::max(&field, Some("T")).unwrap(), 7.0);
        assert_eq!(Field::min(&field, Some("T")).unwrap(), -2.0);

        let mut only_empty = NodeField::default();
        only_empty
            .add_sub(Handle::new(
                SubNodeField::from_poi1(&empty_support, vec!["T".into()]).unwrap(),
            ))
            .unwrap();
        let err = Field::max(&only_empty, Some("T")).unwrap_err();
        assert!(
            format!("{err}").contains("empty support"),
            "unexpected: {err}"
        );
        let err = Field::max(&only_empty, Some("w")).unwrap_err();
        assert!(
            format!("{err}").contains("no sub-field"),
            "unexpected: {err}"
        );
    }
}
