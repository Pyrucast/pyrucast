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

/// Fuse the zones of `field` that share the same support
/// `SubFiniteElementSpace`. See the module documentation. Errors if two
/// zones on the same support disagree on a shared `(cell, gauss, component)`.
pub fn consolidate(field: &ElementField) -> Result<ElementField> {
    struct Snap {
        handle: Handle<SubElementField>,
        fespace: Handle<crate::containers::finite_element_space::SubFiniteElementSpace>,
        components: Vec<String>,
        n_cells: usize,
        n_gauss: usize,
        values: Vec<f64>,
    }
    let mut snaps: Vec<Snap> = Vec::with_capacity(field.len());
    for h in field {
        let (fespace, components, n_cells, n_gauss, values) = {
            let s = h.read();
            (
                s.support(),
                s.components().to_vec(),
                s.cell_count(),
                s.gauss_count(),
                s.values().to_vec(),
            )
        };
        snaps.push(Snap {
            handle: h.clone(),
            fespace,
            components,
            n_cells,
            n_gauss,
            values,
        });
    }

    // Group sub indices by FE-subspace handle identity, first-seen order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, snap) in snaps.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|idxs| snaps[idxs[0]].fespace.same_object(&snap.fespace))
        {
            Some(idxs) => idxs.push(i),
            None => groups.push(vec![i]),
        }
    }

    let mut out = ElementField::default();
    for idxs in &groups {
        if let [single] = idxs.as_slice() {
            out.add_sub(snaps[*single].handle.clone())?;
            continue;
        }

        // Union of the group's components, first-seen order across the subs.
        let mut components: Vec<String> = Vec::new();
        for &i in idxs {
            for c in &snaps[i].components {
                if !components.contains(c) {
                    components.push(c.clone());
                }
            }
        }

        // All subs share the support, hence the same (cell, gauss) layout.
        let support = snaps[idxs[0]].fespace.clone();
        let mut fused = SubElementField::new(support, components)?;

        // A component shared by several subs must agree at every point.
        let mut written: std::collections::HashSet<String> = std::collections::HashSet::new();
        for &i in idxs {
            let snap = &snaps[i];
            let ncomp = snap.components.len();
            for (ci, comp) in snap.components.iter().enumerate() {
                let first_writer = written.insert(comp.clone());
                for cell in 0..snap.n_cells {
                    for g in 0..snap.n_gauss {
                        let v = snap.values[(cell * snap.n_gauss + g) * ncomp + ci];
                        if first_writer {
                            fused.set_value(cell, g, comp, v)?;
                        } else {
                            let existing = fused.value(cell, g, comp)?;
                            if existing != v {
                                return Err(PyrucastError::Message(format!(
                                    "incoherent ElementField on shared support: \
                                     cell {}, gauss {}, component {}: {} ≠ {}",
                                    cell, g, comp, existing, v
                                )));
                            }
                        }
                    }
                }
            }
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
pub fn check_unique_component_per_support(field: &ElementField) -> Result<()> {
    // (support identity, component) already seen.
    let mut seen: Vec<(usize, String)> = Vec::new();
    for h in field {
        let s = h.read();
        let support = s.support();
        for comp in s.components() {
            let key = (support.id(), comp.clone());
            if seen.contains(&key) {
                return Err(PyrucastError::Message(format!(
                    "ElementField: component {comp} is carried by two zones on \
                     the same support {support}. Component fields must be unique \
                     per support; call consolidate to fuse zones that \
                     legitimately share a support."
                )));
            }
            seen.push(key);
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
