//! Consolidate an [`ElementField`]: fuse zones sharing the **same support**.
//!
//! The element-field twin of [`crate::ops::node_field::consolidate`](fn@crate::ops::node_field::consolidate). Sub-fields
//! defined on the *same* `SubFiniteElementSpace` (matched by handle identity,
//! [`crate::handle::Handle::same_object`]) are fused into a single
//! [`SubElementField`] carrying the **union of their components**. A
//! component defined by several of those sub-fields must hold the **same**
//! value at every `(cell, Gauss point)` (exact comparison) — anything else
//! is an error. Zones on distinct supports stay separate.
//!
//! Unlike node fields, element fields have no notion of nodes shared across
//! supports, so there is no cross-support check: distinct
//! `SubFiniteElementSpace`s are independent.
//!
//! A support group reduced to a single sub-field is **shared** (same handle,
//! no copy) in the result.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::parallel::*;

/// Fuse the zones of `field` that share the same support
/// `SubFiniteElementSpace`. See the module documentation. Errors if two
/// zones on the same support disagree on a shared `(cell, gauss, component)`.
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
/// // Le pendant par éléments : une zone par support, composantes réunies.
/// let a = ElementField::new(&fes, vec!["s_xx".into()])?;
/// let b = ElementField::new(&fes, vec!["s_yy".into()])?;
/// let deux = a.union(&b)?;
/// assert_eq!(deux.len(), 2);
/// let une = element_field::consolidate(&deux)?;
/// assert_eq!(une.len(), 1);
/// assert_eq!(une.get(0)?.read().component_count(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn consolidate(field: &ElementField) -> Result<ElementField> {
    // Group the zone handles by FE-subspace identity, first-seen order.
    let mut groups: Vec<Vec<Handle<SubElementField>>> = Vec::new();
    for h in field {
        let fespace = h.read().support();
        match groups
            .iter_mut()
            .find(|g| g[0].read().support().same_object(&fespace))
        {
            Some(g) => g.push(h.clone()),
            None => groups.push(vec![h.clone()]),
        }
    }

    let mut out = ElementField::empty();
    for group in &groups {
        // A group of one keeps its zone: shared, never copied. Snapshotting it
        // would copy a material field's whole value array to hand it back
        // unchanged.
        if let [single] = group.as_slice() {
            out.add_sub(single.clone())?;
            continue;
        }

        // Read the group's zones for the whole fusion — concurrent shared locks,
        // no copy.
        let zones: Vec<_> = group.iter().map(|h| h.read()).collect();

        // Union of the group's components, first-seen order across the zones.
        let mut components: Vec<String> = Vec::new();
        for z in &zones {
            for c in z.components() {
                if !components.contains(c) {
                    components.push(c.clone());
                }
            }
        }
        // All zones share the support, hence the same (cell, gauss) layout.
        let mut fused = SubElementField::new(zones[0].support(), components.clone())?;
        let out_nc = components.len();
        let n_gauss = fused.gauss_count().max(1);

        // Resolve, **once per zone**, where each of its components lands in the
        // fused layout and whether this zone is the one that writes it. Looking
        // the name up at every Gauss point instead would re-prove, a hundred
        // million times over, a property of the zone.
        let mut written = vec![false; out_nc];
        let mut plans: Vec<Vec<(usize, usize, bool)>> = Vec::with_capacity(zones.len());
        for z in &zones {
            let plan = z
                .components()
                .iter()
                .enumerate()
                .map(|(ci, comp)| {
                    let oc = components
                        .iter()
                        .position(|c| c == comp)
                        .expect("the union of the group's components contains each of them");
                    let first = !written[oc];
                    written[oc] = true;
                    (ci, oc, first)
                })
                .collect();
            plans.push(plan);
        }

        // One flat pass per zone, parallel over the (cell, gauss) rows: each row
        // of the fused buffer is written by one task alone. A component several
        // zones carry must agree at every point — checked by index, in the same
        // pass.
        for (z, plan) in zones.iter().zip(&plans) {
            let src_nc = z.component_count();
            let src = z.values();
            fused
                .values_mut()
                .par_chunks_mut(out_nc)
                .with_min_len((MIN_PARALLEL_LEN / out_nc.max(1)).max(1))
                .enumerate()
                .try_for_each(|(row, dst)| -> Result<()> {
                    for &(ci, oc, first) in plan {
                        let v = src[row * src_nc + ci];
                        if first {
                            dst[oc] = v;
                        } else if dst[oc] != v {
                            return Err(PyrucastError::Message(format!(
                                "incoherent ElementField on shared support: \
                                 cell {}, gauss {}, component {}: {} ≠ {}",
                                row / n_gauss,
                                row % n_gauss,
                                components[oc],
                                dst[oc],
                                v
                            )));
                        }
                    }
                    Ok(())
                })?;
        }
        out.add_sub(Handle::new(fused))?;
    }
    Ok(out)
}

/// Validate an [`ElementField`]'s zone decomposition: **no component may be
/// carried by more than one zone on the same support**.
///
/// This is the invariant a union (`a | b`) now enforces instead of fusing
/// zones. Two zones on the same support are allowed as long as their component
/// sets are disjoint — they stay side by side, no new [`SubElementField`] is
/// built. A component appearing twice on one support is a genuine duplicate
/// (readers resolve one zone per `(support, component)`; folding consumers such
/// as the VTK export or [`crate::ops::measure::integral_element`] would otherwise
/// double-count), so it is rejected. Call
/// [`consolidate`] explicitly to fuse zones instead.
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
/// // Deux zones sur le même support **peuvent** coexister si leurs
/// // composantes sont disjointes — c'est ce que ce contrôle autorise.
/// let a = ElementField::new(&fes, vec!["s_xx".into()])?;
/// let b = ElementField::new(&fes, vec!["s_yy".into()])?;
/// let deux = a.union(&b)?;
/// assert_eq!(deux.len(), 2);
/// assert!(element_field::check_unique_component_per_support(&deux).is_ok());
/// // La même composante portée deux fois sur un support, non : l'union
/// // elle-même la refuse, en nommant la composante fautive.
/// let doublon = ElementField::new(&fes, vec!["s_xx".into()])?;
/// assert!(a.union(&doublon).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn check_unique_component_per_support(field: &ElementField) -> Result<()> {
    // (support identity, component) already seen. A set, not a scanned list:
    // this runs on every union, and a field of many zones would otherwise cost
    // a string comparison per pair of components.
    let mut seen: std::collections::HashSet<(usize, String)> = std::collections::HashSet::new();
    for h in field {
        let s = h.read();
        let support = s.support();
        for comp in s.components() {
            let key = (support.id(), comp.clone());
            if !seen.insert(key) {
                return Err(PyrucastError::Message(format!(
                    "ElementField: component {comp} is carried by two zones on \
                     the same support {support}. Component fields must be unique \
                     per support; call consolidate to fuse zones that \
                     legitimately share a support."
                )));
            }
        }
    }
    Ok(())
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::Field;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Single-subspace Lagrange-1 FE space over one TRI3 cell.
    fn one_tri3_fes() -> FiniteElementSpace {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        FiniteElementSpace::lagrange1(&mesh).unwrap()
    }

    /// One-zone field on `fes` (single subspace), components `comps`.
    fn field_on(fes: &FiniteElementSpace, comps: Vec<String>) -> ElementField {
        ElementField::new(fes, comps).unwrap()
    }

    /// Two-zone field built without triggering the union's finalize.
    fn two_zone(a: &ElementField, b: &ElementField) -> ElementField {
        let mut f = ElementField::default();
        f.add_sub(a.get(0).unwrap()).unwrap();
        f.add_sub(b.get(0).unwrap()).unwrap();
        f
    }

    #[test]
    fn same_support_distinct_components_fuse() {
        let fes = one_tri3_fes();
        let a = field_on(&fes, vec!["E".into()]);
        let b = field_on(&fes, vec!["nu".into()]);
        a.get(0).unwrap().write().set_uniform("E", 210.0).unwrap();
        b.get(0).unwrap().write().set_uniform("nu", 0.3).unwrap();

        let c = consolidate(&two_zone(&a, &b)).unwrap();
        assert_eq!(c.len(), 1, "same support ⇒ one fused zone");
        assert_eq!(Field::components(&c).unwrap(), vec!["E", "nu"]);
        let s = c.get(0).unwrap().read();
        assert_eq!(s.value(0, 0, "E").unwrap(), 210.0);
        assert_eq!(s.value(0, 0, "nu").unwrap(), 0.3);
    }

    #[test]
    fn union_same_support_distinct_components_stays_separate() {
        // New convention: a union no longer fuses zones. Two zones on the same
        // support carrying disjoint components are kept side by side.
        let fes = one_tri3_fes();
        let a = field_on(&fes, vec!["E".into()]);
        let b = field_on(&fes, vec!["nu".into()]);
        let c = a.union(&b).unwrap();
        assert_eq!(c.len(), 2, "same support, disjoint components ⇒ two zones");
    }

    #[test]
    fn union_same_support_shared_component_is_rejected() {
        // Same component on the same support is a duplicate ⇒ union errors,
        // regardless of whether the values agree.
        let fes = one_tri3_fes();
        let a = field_on(&fes, vec!["E".into()]);
        let b = field_on(&fes, vec!["E".into()]);
        a.get(0).unwrap().write().set_uniform("E", 1.0).unwrap();
        b.get(0).unwrap().write().set_uniform("E", 1.0).unwrap();
        // Even with identical values, the duplicate component is rejected.
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn distinct_supports_stay_separate() {
        let fes_a = one_tri3_fes();
        let fes_b = one_tri3_fes();
        let a = field_on(&fes_a, vec!["E".into()]);
        let b = field_on(&fes_b, vec!["nu".into()]);
        let c = a.union(&b).unwrap();
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn check_unique_accepts_disjoint_rejects_duplicate() {
        let fes = one_tri3_fes();
        let a = field_on(&fes, vec!["E".into()]);
        let b = field_on(&fes, vec!["nu".into()]);
        let dup = field_on(&fes, vec!["E".into()]);
        // Disjoint components on the same support: valid.
        check_unique_component_per_support(&two_zone(&a, &b)).unwrap();
        // Same component twice on the same support: rejected.
        assert!(check_unique_component_per_support(&two_zone(&a, &dup)).is_err());
    }
}
