//! Material axes turned with the part, at the Gauss points — for a law that
//! reads an oriented material and is computed on the current configuration.
//!
//! An orthotropic or anisotropic material carries its axes as components of
//! the material field (`V1X, V1Y` in 2-D, `V1X…V2Z` in 3-D, see
//! [`symmetry`](crate::models::symmetry)). They are given on the reference
//! configuration. A conduction or a diffusion computed on the deformed part
//! reads the material as it is, its axes turned by the rotation `R` of the
//! polar decomposition `F = R U`: `v = R v₀`, so that `k = R k₀ Rᵀ`. The
//! coefficients themselves do not change — the stretch `U` leaves them as the
//! material is.

use nalgebra::{Matrix3, Vector3};

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::continuum::finite_strain;
use crate::models::continuum::internal_force::full_matrix_reads;
use crate::models::kernel;
use crate::models::symmetry::{FRAME_2D, FRAME_3D};

/// `field` with its material axes turned by the rotation `R` of `F = R U` at
/// each point — `F` read from `gradient`, the field
/// [`deformation`](fn@crate::ops::element_field::deformation) produces under
/// the measure `gradient` (`F_xx, F_xy, …`), computed on the **reference**
/// configuration. Every other component is copied as it is.
///
/// A zone without axes (an isotropic material) has nothing to turn: it is
/// handed back as it is, shared, not copied. In 2-D the rotation is the
/// in-plane one; the out-of-plane axis stays.
///
/// Errors: a zone carrying part of the axes only (`V1X` without `V1Y`); no
/// gradient zone on the same FE space carrying the `F_*` read.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// use pyrucast::ops::element_field::{deformation, material_field, rotate_frame, StrainMeasure};
/// # use pyrucast::ops::model;
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let mesh = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&mesh)?;
/// let conduction = model::heat_conduction_with_symmetry(&fes, MaterialSymmetry::Orthotropic)?;
/// let materials = material_field(
///     &conduction,
///     &[("k_1", 9.0), ("k_2", 1.0), ("k_3", 1.0), ("V1X", 1.0), ("V1Y", 0.0)],
/// )?;
/// // A quarter turn of the part: x → y, y → −x, so u = (−x − y, x − y).
/// let u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
/// {
///     let mut z = u.get(0)?.write();
///     z.set_value(n[1].id(), "u_x", -1.0)?;
///     z.set_value(n[1].id(), "u_y", 1.0)?;
///     z.set_value(n[2].id(), "u_x", -1.0)?;
///     z.set_value(n[2].id(), "u_y", -1.0)?;
/// }
/// let f = deformation(&u, &fes, StrainMeasure::Gradient)?;
/// let turned = rotate_frame(&materials, &f)?;
/// // The strong axis, along x at rest, is along y on the turned part.
/// let z = turned.get(0)?.read();
/// assert!(z.value(0, 0, "V1X")?.abs() < 1e-12);
/// assert!((z.value(0, 0, "V1Y")? - 1.0).abs() < 1e-12);
/// assert_eq!(z.value(0, 0, "k_1")?, 9.0); // the material itself is unchanged
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn rotate_frame(field: &ElementField, gradient: &ElementField) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for zone in field {
        out.add_sub(rotate_zone(zone, gradient)?)?;
    }
    Ok(out)
}

fn rotate_zone(
    zone: &Handle<SubElementField>,
    gradient: &ElementField,
) -> Result<Handle<SubElementField>> {
    // Everything is settled here, once per zone: which axes are read, which
    // entries of F, where each one sits.
    let (fespace, space_dim, components) = {
        let z = zone.read();
        let fespace = z.support();
        let space_dim = fespace.read().space_dim();
        (fespace, space_dim, z.components().to_vec())
    };
    let frame: &[&str] = if space_dim == 2 { &FRAME_2D } else { &FRAME_3D };
    let carried = frame
        .iter()
        .filter(|c| components.iter().any(|n| n == *c))
        .count();
    if carried == 0 {
        return Ok(zone.clone());
    }
    if carried < frame.len() {
        return Err(PyrucastError::Message(format!(
            "rotate_frame: the zone carries part of the material axes {frame:?} only — \
             it carries {components:?}"
        )));
    }
    let axes = zone.read().resolve_components(frame, "material axes")?;
    let f_names = full_matrix_reads("F", space_dim);
    let f_zone = gradient.sub_for_fespace_reading(&fespace, &f_names)?;
    let f_entries: Vec<(usize, usize, u32)> = {
        let refs: Vec<&str> = f_names.iter().map(String::as_str).collect();
        let idx = f_zone.read().resolve_components(&refs, "gradient")?;
        (0..space_dim)
            .flat_map(|i| (0..space_dim).map(move |j| (i, j)))
            .zip(idx)
            .map(|((i, j), c)| (i, j, c))
            .collect()
    };
    // One vector per `space_dim` columns: V1 alone in 2-D, V1 then V2 in 3-D.
    let vectors: Vec<&[u32]> = axes.chunks(space_dim).collect();

    let rotated = kernel::element_pointwise(
        &fespace,
        zone,
        &f_zone,
        &f_zone,
        components,
        |_geom, _g, row, f_row, _, out| {
            out.copy_from_slice(row);
            // F is the identity outside the plane: the in-plane rotation.
            let mut f = Matrix3::<f64>::identity();
            for &(i, j, c) in &f_entries {
                f[(i, j)] = f_row[c as usize];
            }
            let r = finite_strain::polar_rotation(&f);
            for columns in &vectors {
                let mut v = Vector3::zeros();
                for (a, &c) in columns.iter().enumerate() {
                    v[a] = row[c as usize];
                }
                let turned = r * v;
                for (a, &c) in columns.iter().enumerate() {
                    out[c as usize] = turned[a];
                }
            }
            Ok(())
        },
    )?;
    Ok(Handle::new(rotated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::NodeField;
    use crate::coords::Coords;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::ops::element_field::{deformation, material_field, StrainMeasure};
    use crate::ops::model;

    /// A unit TET4 and the space on it.
    fn tet() -> (Mesh, FiniteElementSpace, Vec<Node>) {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<Node> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        let ids: Vec<_> = n.iter().map(Node::id).collect();
        sm.add_cell(&ids).unwrap();
        let mesh = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        (mesh, fes, n)
    }

    /// The displacement `u = (F − I) X` of a uniform gradient.
    fn displacement(mesh: &Mesh, nodes: &[Node], f: &Matrix3<f64>) -> NodeField {
        let names = vec!["u_x".into(), "u_y".into(), "u_z".into()];
        let u = NodeField::new(mesh, names).unwrap();
        let mut z = u.get(0).unwrap().write();
        for n in nodes {
            let p = n.position().unwrap();
            let x = Vector3::new(p[0], p[1], p[2]);
            let d = (f - Matrix3::identity()) * x;
            for (a, name) in ["u_x", "u_y", "u_z"].iter().enumerate() {
                z.set_value(n.id(), name, d[a]).unwrap();
            }
        }
        drop(z);
        u
    }

    /// The axes follow R, not F: under a stretch with shear, `R v₀` differs
    /// from the convected `F v₀ / |F v₀|`, and it is `R v₀` that comes out —
    /// with `V2` turned as well and the coefficients untouched.
    #[test]
    fn the_axes_turn_by_the_rotation_of_the_gradient() {
        let (mesh, fes, nodes) = tet();
        let conduction =
            model::heat_conduction_with_symmetry(&fes, MaterialSymmetry::Orthotropic).unwrap();
        let (v1, v2) = (Vector3::new(1.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 2.0));
        let materials = material_field(
            &conduction,
            &[
                ("k_1", 5.0),
                ("k_2", 2.0),
                ("k_3", 1.0),
                ("V1X", v1[0]),
                ("V1Y", v1[1]),
                ("V1Z", v1[2]),
                ("V2X", v2[0]),
                ("V2Y", v2[1]),
                ("V2Z", v2[2]),
            ],
        )
        .unwrap();
        let r0 = nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), 0.6).into_inner();
        let u0 = Matrix3::new(1.5, 0.4, 0.0, 0.4, 0.9, 0.0, 0.0, 0.0, 1.2);
        let f = r0 * u0;
        let gradient = deformation(
            &displacement(&mesh, &nodes, &f),
            &fes,
            StrainMeasure::Gradient,
        )
        .unwrap();

        let turned = rotate_frame(&materials, &gradient).unwrap();
        let z = turned.get(0).unwrap().read();
        let read = |names: [&str; 3]| {
            Vector3::from_iterator(names.iter().map(|c| z.value(0, 0, c).unwrap()))
        };
        let (w1, w2) = (read(["V1X", "V1Y", "V1Z"]), read(["V2X", "V2Y", "V2Z"]));
        assert!((w1 - r0 * v1).norm() < 1e-12, "{w1}");
        assert!((w2 - r0 * v2).norm() < 1e-12, "{w2}");
        let convected = (f * v1).normalize() * v1.norm();
        assert!((w1 - convected).norm() > 1e-2);
        assert_eq!(z.value(0, 0, "k_1").unwrap(), 5.0);
    }

    /// An isotropic zone has nothing to turn: it comes back shared. Part of
    /// the axes is an error.
    #[test]
    fn a_zone_without_axes_is_shared_and_part_of_them_refused() {
        let (mesh, fes, nodes) = tet();
        let conduction = model::heat_conduction(&fes).unwrap();
        let isotropic = material_field(&conduction, &[("k", 1.0)]).unwrap();
        let gradient = deformation(
            &displacement(&mesh, &nodes, &Matrix3::identity()),
            &fes,
            StrainMeasure::Gradient,
        )
        .unwrap();
        let same = rotate_frame(&isotropic, &gradient).unwrap();
        assert!(same.get(0).unwrap().same_object(&isotropic.get(0).unwrap()));

        let partial = ElementField::new(&fes, vec!["k".into(), "V1X".into()]).unwrap();
        assert!(rotate_frame(&partial, &gradient).is_err());
    }
}
