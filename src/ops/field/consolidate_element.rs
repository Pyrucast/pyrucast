//! Consolidate an [`ElementField`]: fuse zones sharing the **same support**.
//!
//! The element-field twin of [`crate::ops::field::consolidate`](fn@crate::ops::field::consolidate). Sub-fields
//! defined on the *same* `SubFiniteElementSpace` (matched by handle identity,
//! [`crate::store::Handle::same_slot`]) are fused into a single
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
use crate::store::{insert, read, Handle};

/// Fuse the zones of `field` that share the same support
/// `SubFiniteElementSpace`. See the module documentation. Errors if two
/// zones on the same support disagree on a shared `(cell, gauss, component)`.
pub fn consolidate_element(field: &ElementField) -> Result<ElementField> {
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
            let s = read(h)?;
            (
                s.support(),
                s.components().to_vec(),
                s.cell_count(),
                s.gauss_count(),
                s.values().to_vec(),
            )
        };
        snaps.push(Snap { handle: h.clone(), fespace, components, n_cells, n_gauss, values });
    }

    // Group sub indices by FE-subspace handle identity, first-seen order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, snap) in snaps.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|idxs| snaps[idxs[0]].fespace.same_slot(&snap.fespace))
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
        out.add_sub(insert(fused))?;
    }
    Ok(out)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::field::Field;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::store::insert;

    /// Single-subspace Lagrange-1 FE space over one TRI3 cell.
    fn one_tri3_fes() -> FiniteElementSpace {
        let coords = insert(Coords::new(2).unwrap());
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
        crate::store::write(&a.get(0).unwrap()).unwrap().set_uniform("E", 210.0).unwrap();
        crate::store::write(&b.get(0).unwrap()).unwrap().set_uniform("nu", 0.3).unwrap();

        let c = consolidate_element(&two_zone(&a, &b)).unwrap();
        assert_eq!(c.len(), 1, "same support ⇒ one fused zone");
        assert_eq!(Field::components(&c).unwrap(), vec!["E", "nu"]);
        let s = read(&c.get(0).unwrap()).unwrap();
        assert_eq!(s.value(0, 0, "E").unwrap(), 210.0);
        assert_eq!(s.value(0, 0, "nu").unwrap(), 0.3);
    }

    #[test]
    fn same_support_shared_component_must_agree() {
        let fes = one_tri3_fes();
        let a = field_on(&fes, vec!["E".into()]);
        let b = field_on(&fes, vec!["E".into()]);
        crate::store::write(&a.get(0).unwrap()).unwrap().set_uniform("E", 1.0).unwrap();
        crate::store::write(&b.get(0).unwrap()).unwrap().set_uniform("E", 2.0).unwrap();
        // The union's finalize fuses and detects the conflict → `|` errors.
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
}
