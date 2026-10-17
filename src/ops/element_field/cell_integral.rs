//! Integral of a per-element field over **each cell** — one value per cell and
//! component, `∫_cell f dΩ` by the field's own quadrature (Cast3M `INTG`
//! `'ELEM'`).
//!
//! Where [`integral_element`](fn@crate::ops::measure::integral::integral_element)
//! sums over the whole support into one number, this keeps the cells apart.
//! The result needs a home with one point per cell: the caller hands over a
//! space on the same mesh built with the `Reduced` quadrature — a single point
//! at the centroid of each cell, Cast3M's `'GRAVITE'` support. Integrating `1`
//! gives the measure of every cell (its area, or its revolved volume on a body
//! of revolution), which is how an incremental solver weighs its nodes.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;

/// The integral of every component of `field` over each cell, defined on
/// `target`.
///
/// `target` spans the same mesh as `field` with **one point per cell** —
/// `FiniteElementSpace::with(mesh, &[(Lagrange1, Reduced)])`. Each of its zones
/// receives one output zone per input zone on the same submesh, with the same
/// components: the value at a cell is `Σ_g f(g)·|J|·w_g` over that cell's
/// Gauss points, the measure carrying `2πr` on a body of revolution.
///
/// Errors when a zone of `target` has more than one point per cell, or when
/// `field` has nothing on one of its submeshes.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node, QuadratureRule};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::element_field;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # sm.add_cell(&[n[0].id(), n[2].id(), n[3].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// let fes = FiniteElementSpace::lagrange1(&maillage)?;
/// let centres = FiniteElementSpace::with(&maillage, &[(Interpolation::Lagrange1, QuadratureRule::Reduced)])?;
/// let un = ElementField::new(&fes, vec!["un".into()])?;
/// un.get(0)?.write().set_uniform("un", 1.0)?;
/// // The integral of 1 over each cell is its area: two triangles of 1.
/// let aires = element_field::cell_integral(&un, &centres)?;
/// let z = aires.get(0)?;
/// assert_eq!(z.read().gauss_count(), 1);
/// assert!((z.read().value(1, 0, "un")? - 1.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn cell_integral(field: &ElementField, target: &FiniteElementSpace) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for target_zone in target {
        let submesh = target_zone.read().submesh();
        let mut found = false;
        for zone in field {
            let (fespace, components) = {
                let z = zone.read();
                (z.support(), z.components().to_vec())
            };
            if !fespace.read().submesh().same_object(&submesh) {
                continue;
            }
            found = true;
            let n = components.len();
            let integral = kernel::cellwise(
                &fespace,
                zone,
                target_zone,
                components,
                |geom, rows, out| {
                    out.fill(0.0);
                    for g in 0..geom.n_gauss {
                        let w = geom.det_j_w(g);
                        for (k, slot) in out.iter_mut().enumerate() {
                            *slot += rows[g * n + k] * w;
                        }
                    }
                    Ok(())
                },
            )?;
            out.add_sub(Handle::new(integral))?;
        }
        if !found {
            return Err(PyrucastError::Message(
                "cell_integral: the field has no zone on one of the target's submeshes — the \
                 target must be built on the field's own mesh"
                    .into(),
            ));
        }
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Interpolation, Node, QuadratureRule};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::ops::measure::integral::integral_element;

    fn quads(axisymmetric: bool) -> Mesh {
        let coords = Handle::new(if axisymmetric {
            Coords::axisymmetric().unwrap()
        } else {
            Coords::new(2).unwrap()
        });
        let n: Vec<Node> = [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.5, 0.0],
            [1.0, 1.0],
            [2.0, 1.2],
            [3.5, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        sm.add_cell(&[n[0].id(), n[1].id(), n[4].id(), n[3].id()])
            .unwrap();
        sm.add_cell(&[n[1].id(), n[2].id(), n[5].id(), n[4].id()])
            .unwrap();
        Mesh::from_submesh(sm)
    }

    fn reduced(mesh: &Mesh) -> FiniteElementSpace {
        FiniteElementSpace::with(mesh, &[(Interpolation::Lagrange1, QuadratureRule::Reduced)])
            .unwrap()
    }

    /// The independent oracle: the cell integrals add up to the whole-support
    /// integral `measure::integral` computes — plane and revolved alike.
    #[test]
    fn cells_add_up_to_the_whole_integral() {
        for axi in [false, true] {
            let mesh = quads(axi);
            let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
            let f = ElementField::new(&fes, vec!["f".into()]).unwrap();
            {
                let z = f.get(0).unwrap();
                let mut w = z.write();
                for c in 0..2 {
                    for g in 0..w.gauss_count() {
                        w.set_value(c, g, "f", 1.0 + c as f64 + 0.25 * g as f64)
                            .unwrap();
                    }
                }
            }
            let per_cell = cell_integral(&f, &reduced(&mesh)).unwrap();
            let z = per_cell.get(0).unwrap();
            let sum: f64 = (0..2).map(|c| z.read().value(c, 0, "f").unwrap()).sum();
            let whole = integral_element(&f, "f").unwrap();
            assert!(
                (sum - whole).abs() < 1e-10 * whole.abs(),
                "axi={axi}: {sum} ≠ {whole}"
            );
        }
    }

    #[test]
    fn a_target_with_several_points_per_cell_is_refused() {
        let mesh = quads(false);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let f = ElementField::new(&fes, vec!["f".into()]).unwrap();
        let err = cell_integral(&f, &fes).unwrap_err();
        assert!(format!("{err}").contains("one point per cell"), "{err}");
    }

    #[test]
    fn a_target_on_another_mesh_is_refused() {
        let fes = FiniteElementSpace::lagrange1(&quads(false)).unwrap();
        let f = ElementField::new(&fes, vec!["f".into()]).unwrap();
        let err = cell_integral(&f, &reduced(&quads(false))).unwrap_err();
        assert!(format!("{err}").contains("no zone"), "{err}");
    }
}
