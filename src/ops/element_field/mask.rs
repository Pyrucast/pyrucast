//! Value-range masking — turn a field into a 0/1 indicator of the **same
//! shape**, testing each value against a `[lower, upper]` band, component by
//! component (Cast3M's `MASQUE`).
//!
//! Unlike [`select`](crate::ops::mesh::select_nodes), which extracts the passing part of the
//! support into a `Mesh`, `mask` keeps the field's exact structure (same
//! zones, same support, same components) and only rewrites the values:
//! `1.0` where the band holds, `0.0` where it does not. The result is
//! therefore multipliable term by term with the input — `field *
//! mask(field, ge=0)` zeroes the out-of-band values, component by
//! component.
//!
//! There is **no** AND across components here (that is [`select`](crate::ops::mesh::select_nodes)'s job):
//! each value stands on its own, so the mask is per component.
//!
//! # Component filter
//!
//! - `components = None` ⇒ every component is tested (gets a real 0/1);
//! - `components = Some(list)` ⇒ only those components are tested; the
//!   others are left at `1.0` (identity for the product), so masking a
//!   subset of components leaves the rest untouched. A zone that does not
//!   carry a listed component is left all-`1.0` (the filter cannot apply).
//!
//! # Bounds
//!
//! The `[lower, upper]` band comes from a shared [`Band`],
//! built from the four comparison bounds `ge` / `gt` / `le` / `lt` — each
//! side open, inclusive, or strict.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::{Field, SubField};
use crate::error::Result;

use crate::atoms::Band;
use crate::ops::mesh::select::components_to_test;

/// Rewrite `sub` into its 0/1 mask: each **tested** component becomes `1.0`
/// inside the band and `0.0` outside; untested components stay `1.0`.
///
/// Components are the innermost storage axis for both field flavours, so a
/// single flat pass with `idx % ncomp` handles nodes and Gauss points alike.
fn mask_zone<S>(sub: &S, band: &Band, components: &Option<Vec<String>>) -> S
where
    S: SubField + Clone,
{
    let ncomp = sub.component_count();
    // Which component indices are tested? `None` ⇒ the zone lacks a
    // requested component — nothing to test, leave it identity (all 1.0).
    let mut tested = vec![false; ncomp];
    if let Some(idx) = components_to_test(sub.components(), components) {
        for i in idx {
            tested[i] = true;
        }
    }
    let mut out = sub.clone();
    for (i, v) in out.values_mut().iter_mut().enumerate() {
        *v = if tested[i % ncomp] {
            f64::from(band.contains(*v))
        } else {
            1.0
        };
    }
    out
}

/// Per-component 0/1 mask of `field` against `band`, zone by zone. Returns a
/// `NodeField` with the **same structure** (zones, support, components) as
/// the input. See the [module documentation](self) for the component-filter
/// semantics.
/// Per-component 0/1 mask of `field` against `band`, zone by zone (one value
/// per Gauss point × component). Returns an `ElementField` with the same
/// structure as the input.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// # let mut f = ElementField::new(&fes, vec!["q".into()])?;
/// # f.get(0)?.write().set_uniform("q", 5.0)?;
/// // Un indicateur 0/1 aux points de Gauss, de même forme que l'entrée.
/// let haute = Band::new(Some(10.0), None, None, None)?;
/// assert_eq!(element_field::mask(&f, &haute, None)?
///     .get(0)?.read().value(0, 0, "q")?, 0.0);
/// let basse = Band::new(Some(1.0), None, None, None)?;
/// assert_eq!(element_field::mask(&f, &basse, None)?
///     .get(0)?.read().value(0, 0, "q")?, 1.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn mask(
    field: &ElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> Result<ElementField> {
    field.map_subs(|s| Ok(mask_zone(s, band, &components)))
}

/// Single-zone [`mask`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// # let mut f = ElementField::new(&fes, vec!["q".into()])?;
/// # f.get(0)?.write().set_uniform("q", 5.0)?;
/// let haute = Band::new(Some(10.0), None, None, None)?;
/// let m = element_field::mask_sub(&f.get(0)?.read(), &haute, None);
/// assert_eq!(m.value(0, 0, "q")?, 0.0); // 5 < 10
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn mask_sub(
    sub: &SubElementField,
    band: &Band,
    components: Option<Vec<String>>,
) -> SubElementField {
    mask_zone(sub, band, &components)
}
