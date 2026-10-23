//! A stress carried from one measure to another at the Gauss points, given the
//! deformation gradient `F` — the stress half of Cast3M's `CONF`, between the
//! reference configuration and the current one.
//!
//! A law reading a large-transformation measure writes a **Lagrangian** stress
//! on the reference configuration: the second Piola–Kirchhoff `S` (dual of
//! Green–Lagrange). What the rest of the computation needs is another measure
//! of the same state: the first Piola–Kirchhoff `P = F S` for the internal
//! forces on the reference configuration, the Cauchy `σ` for the geometric
//! stiffness on the current one and for the results. This operator converts,
//! point by point; nothing is transported, the state itself never changes
//! configuration.

use nalgebra::Matrix3;

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::continuum::finite_strain;
use crate::models::continuum::internal_force::{full_matrix_reads, voigt_matrix_reads};
use crate::models::kernel;
use crate::named::Named;

/// Which stress [`stress_measure`] produces — and which one it recognises in its
/// input, by the names it carries.
///
/// | measure | prefix | components | dual of |
/// |---|---|---|---|
/// | [`Cauchy`](Self::Cauchy) | `sigma_` | `i ≤ j` | — (current configuration) |
/// | [`Kirchhoff`](Self::Kirchhoff) | `tau_` | `i ≤ j` | — (`τ = J σ`) |
/// | [`FirstPiolaKirchhoff`](Self::FirstPiolaKirchhoff) | `P_` | every `(i, j)` | the gradient `F` |
/// | [`SecondPiolaKirchhoff`](Self::SecondPiolaKirchhoff) | `S_` | `i ≤ j` | Green–Lagrange `E` |
/// | [`Hencky`](Self::Hencky) | `T_` | `i ≤ j` | Hencky `H = ½ ln C` |
///
/// The Lagrangian ones are the duals of a strain measure; Cauchy and Kirchhoff
/// are Eulerian and the dual of no total measure, which is why the target is
/// named after the stress rather than after a strain.
///
/// ```
/// use pyrucast::named::Named;
/// use pyrucast::ops::element_field::StressMeasure;
///
/// assert_eq!(StressMeasure::parse("pk2")?, StressMeasure::SecondPiolaKirchhoff);
/// // The usual letters are accepted too.
/// assert_eq!(StressMeasure::parse("sigma")?, StressMeasure::Cauchy);
/// assert_eq!(StressMeasure::FirstPiolaKirchhoff.prefix(), "P");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StressMeasure {
    /// The true stress `σ`, force per current area.
    Cauchy,
    /// The Kirchhoff stress `τ = J σ`.
    Kirchhoff,
    /// The nominal stress `P = F S`, force per reference area — not symmetric.
    FirstPiolaKirchhoff,
    /// The second Piola–Kirchhoff stress `S`, dual of Green–Lagrange.
    SecondPiolaKirchhoff,
    /// The dual `T` of the Hencky strain `H = ½ ln C`, what a law reading
    /// Hencky writes: `S = T : ∂H/∂E`.
    Hencky,
}

impl StressMeasure {
    /// The component prefix: a component is named `<prefix>_<ai><aj>`.
    ///
    /// ```
    /// use pyrucast::ops::element_field::StressMeasure;
    ///
    /// // Cauchy keeps the `sigma_` every small-strain law writes.
    /// assert_eq!(StressMeasure::Cauchy.prefix(), "sigma");
    /// assert_eq!(StressMeasure::SecondPiolaKirchhoff.prefix(), "S");
    /// ```
    pub fn prefix(self) -> &'static str {
        match self {
            StressMeasure::Cauchy => "sigma",
            StressMeasure::Kirchhoff => "tau",
            StressMeasure::FirstPiolaKirchhoff => "P",
            StressMeasure::SecondPiolaKirchhoff => "S",
            StressMeasure::Hencky => "T",
        }
    }

    /// The in-plane entries `(i, j)` this measure stores in a `space_dim` space,
    /// in the order of [`names`](Self::names): `i ≤ j` for a symmetric one,
    /// every pair, row-major, for `P`.
    fn slots(self, space_dim: usize) -> Vec<(usize, usize)> {
        let symmetric = self != StressMeasure::FirstPiolaKirchhoff;
        let mut slots = Vec::with_capacity(space_dim * space_dim);
        for i in 0..space_dim {
            let first = if symmetric { i } else { 0 };
            for j in first..space_dim {
                slots.push((i, j));
            }
        }
        slots
    }

    /// The in-plane components of this measure in a `space_dim` space.
    fn names(self, space_dim: usize) -> Vec<String> {
        if self == StressMeasure::FirstPiolaKirchhoff {
            full_matrix_reads(self.prefix(), space_dim)
        } else {
            voigt_matrix_reads(self.prefix(), space_dim)
        }
    }

    /// `S` from this measure.
    fn to_pk2(self) -> Convert {
        match self {
            StressMeasure::Cauchy => finite_strain::pk2_from_cauchy,
            StressMeasure::Kirchhoff => finite_strain::pk2_from_kirchhoff,
            StressMeasure::FirstPiolaKirchhoff => finite_strain::pk2_from_pk1,
            StressMeasure::SecondPiolaKirchhoff => same,
            StressMeasure::Hencky => finite_strain::pk2_from_hencky,
        }
    }

    /// This measure from `S`.
    fn from_pk2(self) -> Convert {
        match self {
            StressMeasure::Cauchy => finite_strain::cauchy_from_pk2,
            StressMeasure::Kirchhoff => finite_strain::kirchhoff_from_pk2,
            StressMeasure::FirstPiolaKirchhoff => finite_strain::pk1_from_pk2,
            StressMeasure::SecondPiolaKirchhoff => same,
            StressMeasure::Hencky => finite_strain::hencky_from_pk2,
        }
    }
}

impl Named for StressMeasure {
    const LABEL: &'static str = "stress measure";
    const VALUES: &'static [Self] = &[
        StressMeasure::Cauchy,
        StressMeasure::Kirchhoff,
        StressMeasure::FirstPiolaKirchhoff,
        StressMeasure::SecondPiolaKirchhoff,
        StressMeasure::Hencky,
    ];

    fn name(self) -> &'static str {
        match self {
            StressMeasure::Cauchy => "cauchy",
            StressMeasure::Kirchhoff => "kirchhoff",
            StressMeasure::FirstPiolaKirchhoff => "pk1",
            StressMeasure::SecondPiolaKirchhoff => "pk2",
            StressMeasure::Hencky => "hencky",
        }
    }

    // The letters the literature uses.
    fn aliases() -> &'static [(&'static str, Self)] {
        &[
            ("sigma", StressMeasure::Cauchy),
            ("tau", StressMeasure::Kirchhoff),
            ("P", StressMeasure::FirstPiolaKirchhoff),
            ("S", StressMeasure::SecondPiolaKirchhoff),
            ("T", StressMeasure::Hencky),
        ]
    }
}

/// One conversion of the point algebra, chosen once per zone: the Gauss loop
/// calls it and tests nothing.
type Convert = fn(&Matrix3<f64>, &Matrix3<f64>) -> Matrix3<f64>;

fn same(s: &Matrix3<f64>, _f: &Matrix3<f64>) -> Matrix3<f64> {
    *s
}

/// `stress` converted to the measure `target`, point by point, with the
/// deformation gradient of `gradient` — the field
/// [`deformation`](fn@crate::ops::element_field::deformation) produces under
/// the measure `gradient` (`F_xx, F_xy, …`, hoop `F_zz` on a body of
/// revolution).
///
/// The measure of `stress` is recognised by its **names** — `sigma_*`, `tau_*`,
/// `P_*`, `S_*` or `T_*` — exactly one per zone. In 2-D the out-of-plane `zz` is read
/// when the stress carries it (the hoop of a body of revolution, the `σ_zz` a
/// plane-strain law echoes) and written back under the target's name; `F_zz` is
/// 1 there unless the gradient carries it.
///
/// Errors: a zone carrying none of the five measures, or more than one; no
/// gradient zone on the same FE space carrying the `F_*` read.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// use pyrucast::ops::element_field::{deformation, stress_measure, StrainMeasure, StressMeasure};
/// # let coords = Handle::new(Coords::new(2)?);
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let mesh = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&mesh)?;
/// // A uniform stretch λ = 2 along x: F = diag(2, 1).
/// let u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
/// u.get(0)?.write().set_value(n[1].id(), "u_x", 1.0)?;
/// let f = deformation(&u, &fes, StrainMeasure::Gradient)?;
/// // A second Piola–Kirchhoff stress S_xx = 10 …
/// let s = ElementField::new(&fes, vec!["S_xx".into(), "S_xy".into(), "S_yy".into()])?;
/// s.get(0)?.write().set_uniform("S_xx", 10.0)?;
/// // … is a nominal stress P_xx = λ S_xx and a true stress σ_xx = λ² S_xx / J.
/// let p = stress_measure(&s, &f, StressMeasure::FirstPiolaKirchhoff)?;
/// let sigma = stress_measure(&s, &f, StressMeasure::Cauchy)?;
/// assert!((p.get(0)?.read().value(0, 0, "P_xx")? - 20.0).abs() < 1e-12);
/// assert!((sigma.get(0)?.read().value(0, 0, "sigma_xx")? - 20.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn stress_measure(
    stress: &ElementField,
    gradient: &ElementField,
    target: StressMeasure,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for zone in stress {
        out.add_sub(Handle::new(convert_zone(zone, gradient, target)?))?;
    }
    Ok(out)
}

fn convert_zone(
    zone: &Handle<SubElementField>,
    gradient: &ElementField,
    target: StressMeasure,
) -> Result<SubElementField> {
    // Everything is settled here, once per zone: which measure comes in, which
    // components are read and written, which conversion runs.
    let (fespace, space_dim, axisymmetric, components) = {
        let z = zone.read();
        let fespace = z.support();
        let (space_dim, axisymmetric) = {
            let s = fespace.read();
            (s.space_dim(), s.is_axisymmetric())
        };
        (fespace, space_dim, axisymmetric, z.components().to_vec())
    };
    let carries = |names: &[String]| names.iter().all(|n| components.contains(n));
    let found: Vec<StressMeasure> = StressMeasure::VALUES
        .iter()
        .copied()
        .filter(|m| carries(&m.names(space_dim)))
        .collect();
    let source = match found.as_slice() {
        [one] => *one,
        [] => {
            return Err(PyrucastError::Message(format!(
            "stress_measure: no stress among sigma_*, tau_*, P_*, S_*, T_* in this {space_dim}-D \
                 zone, which carries {components:?}"
        )))
        }
        several => {
            return Err(PyrucastError::Message(format!(
                "stress_measure: the zone carries several stresses ({}), so which one to convert \
                 is ambiguous — pass them in separate fields",
                several
                    .iter()
                    .map(|m| m.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )))
        }
    };
    // In 2-D the out-of-plane component closes the lists when it is there.
    let zz = space_dim == 2 && components.contains(&format!("{}_zz", source.prefix()));
    let mut reads = source.names(space_dim);
    let mut writes = target.names(space_dim);
    let mut read_slots = source.slots(space_dim);
    let mut write_slots = target.slots(space_dim);
    if zz {
        reads.push(format!("{}_zz", source.prefix()));
        writes.push(format!("{}_zz", target.prefix()));
        read_slots.push((2, 2));
        write_slots.push((2, 2));
    }
    let mut f_names = full_matrix_reads("F", space_dim);
    let mut f_slots = StressMeasure::FirstPiolaKirchhoff.slots(space_dim);
    if axisymmetric {
        f_names.push("F_zz".to_string());
        f_slots.push((2, 2));
    }
    let f_zone = gradient.sub_for_fespace_reading(&fespace, &f_names)?;
    // Where each matrix entry is read: `(i, j, column)`, the lower triangle of
    // a symmetric stress pointing at the same column as the upper one. The
    // Gauss loop below walks these tables and tests nothing.
    let stress_entries = {
        let refs: Vec<&str> = reads.iter().map(String::as_str).collect();
        let idx = zone.read().resolve_components(&refs, "stress")?;
        let mut entries = Vec::with_capacity(9);
        for (&(i, j), &c) in read_slots.iter().zip(&idx) {
            entries.push((i, j, c));
            if i != j && source != StressMeasure::FirstPiolaKirchhoff {
                entries.push((j, i, c));
            }
        }
        entries
    };
    let f_entries: Vec<(usize, usize, u32)> = {
        let refs: Vec<&str> = f_names.iter().map(String::as_str).collect();
        let idx = f_zone.read().resolve_components(&refs, "gradient")?;
        f_slots
            .iter()
            .zip(&idx)
            .map(|(&(i, j), &c)| (i, j, c))
            .collect()
    };
    let (to_s, from_s) = (source.to_pk2(), target.from_pk2());

    kernel::element_pointwise(
        &fespace,
        zone,
        &f_zone,
        &f_zone,
        writes,
        |_geom, _g, row, f_row, _, out| {
            // F is the identity outside the plane unless the gradient says
            // otherwise; the stress is zero there unless it carries `zz`.
            let mut f = Matrix3::<f64>::identity();
            for &(i, j, c) in &f_entries {
                f[(i, j)] = f_row[c as usize];
            }
            let mut tensor = Matrix3::<f64>::zeros();
            for &(i, j, c) in &stress_entries {
                tensor[(i, j)] = row[c as usize];
            }
            let result = from_s(&to_s(&tensor, &f), &f);
            for (slot, &(i, j)) in out.iter_mut().zip(&write_slots) {
                *slot = result[(i, j)];
            }
            Ok(())
        },
    )
}
