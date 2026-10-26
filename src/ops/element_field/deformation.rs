//! Deformation measures of a displacement field at the Gauss points — the
//! mechanical counterpart of [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient).
//!
//! One operator, `(displacement, fespace, measure) → ElementField`. The
//! [`StrainMeasure`] chooses *which* deformation is produced: the linearized
//! strain `ε`, the deformation gradient `F`, or one of the finite measures
//! built on it (Green–Lagrange, Cauchy-Green, Hencky, Euler–Almansi). The
//! caller then decides which one to feed
//! [`crate::ops::element_field::behavior::integrate`].

use nalgebra::Matrix3;

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::Field;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::ops::element_field::gradient::AXES;

/// Which deformation [`deformation`] produces from a displacement `u`.
///
/// With `H = ∇u` and `F = I + H`:
///
/// | measure | formula | prefix | components |
/// |---|---|---|---|
/// | [`Linear`](Self::Linear) | `ε = ½(H + Hᵀ)` | `eps_` | `i ≤ j` |
/// | [`Gradient`](Self::Gradient) | `F = I + H` | `F_` | every `(i, j)` |
/// | [`GreenLagrange`](Self::GreenLagrange) | `E = ½(FᵀF − I)` | `E_` | `i ≤ j` |
/// | [`RightCauchyGreen`](Self::RightCauchyGreen) | `C = FᵀF` | `C_` | `i ≤ j` |
/// | [`LeftCauchyGreen`](Self::LeftCauchyGreen) | `b = FFᵀ` | `b_` | `i ≤ j` |
/// | [`Hencky`](Self::Hencky) | `½ ln C` | `H_` | `i ≤ j` |
/// | [`EulerAlmansi`](Self::EulerAlmansi) | `e = ½(I − b⁻¹)` | `e_` | `i ≤ j` |
///
/// Each measure has its own prefix, so one is never read for another.
///
/// ```
/// use pyrucast::named::Named;
/// use pyrucast::ops::element_field::StrainMeasure;
///
/// assert_eq!(StrainMeasure::parse("green_lagrange")?, StrainMeasure::GreenLagrange);
/// // The usual one-letter names are accepted too.
/// assert_eq!(StrainMeasure::parse("F")?, StrainMeasure::Gradient);
/// assert_eq!(StrainMeasure::Hencky.prefix(), "H");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum StrainMeasure {
    /// Linearized (small-strain) tensor `ε = ½(∇u + ∇uᵀ)`, in **tensor**
    /// convention (`eps_xy` is half the engineering shear `γ`).
    #[default]
    Linear,
    /// Deformation gradient `F = I + ∇u` — not symmetric, so every entry is
    /// produced, row-major.
    Gradient,
    /// Green–Lagrange strain `E = ½(FᵀF − I)`, the Lagrangian measure of large
    /// displacements.
    GreenLagrange,
    /// Right Cauchy-Green tensor `C = FᵀF`.
    RightCauchyGreen,
    /// Left Cauchy-Green (Finger) tensor `b = FFᵀ`.
    LeftCauchyGreen,
    /// Hencky (logarithmic) strain `½ ln C = ln U`, through the spectral
    /// decomposition of `C`.
    Hencky,
    /// Euler–Almansi strain `e = ½(I − b⁻¹)`, the Eulerian counterpart of
    /// Green–Lagrange.
    EulerAlmansi,
}

impl StrainMeasure {
    /// The component prefix: a component is named `<prefix>_<ai><aj>`.
    ///
    /// ```
    /// use pyrucast::ops::element_field::StrainMeasure;
    ///
    /// // Linear keeps the historical `eps_`, read by every behaviour law.
    /// assert_eq!(StrainMeasure::Linear.prefix(), "eps");
    /// assert_eq!(StrainMeasure::EulerAlmansi.prefix(), "e");
    /// ```
    pub fn prefix(self) -> &'static str {
        match self {
            StrainMeasure::Linear => "eps",
            StrainMeasure::Gradient => "F",
            StrainMeasure::GreenLagrange => "E",
            StrainMeasure::RightCauchyGreen => "C",
            StrainMeasure::LeftCauchyGreen => "b",
            StrainMeasure::Hencky => "H",
            StrainMeasure::EulerAlmansi => "e",
        }
    }

    /// Whether the measure is a symmetric tensor — only the entries `i ≤ j`
    /// are then produced.
    fn is_symmetric(self) -> bool {
        self != StrainMeasure::Gradient
    }
}

impl crate::named::Named for StrainMeasure {
    const LABEL: &'static str = "strain measure";
    const VALUES: &'static [Self] = &[
        StrainMeasure::Linear,
        StrainMeasure::Gradient,
        StrainMeasure::GreenLagrange,
        StrainMeasure::RightCauchyGreen,
        StrainMeasure::LeftCauchyGreen,
        StrainMeasure::Hencky,
        StrainMeasure::EulerAlmansi,
    ];

    fn name(self) -> &'static str {
        match self {
            StrainMeasure::Linear => "linear",
            StrainMeasure::Gradient => "gradient",
            StrainMeasure::GreenLagrange => "green_lagrange",
            StrainMeasure::RightCauchyGreen => "right_cauchy_green",
            StrainMeasure::LeftCauchyGreen => "left_cauchy_green",
            StrainMeasure::Hencky => "hencky",
            StrainMeasure::EulerAlmansi => "euler_almansi",
        }
    }

    // The one-letter names the literature uses.
    fn aliases() -> &'static [(&'static str, Self)] {
        &[
            ("F", StrainMeasure::Gradient),
            ("C", StrainMeasure::RightCauchyGreen),
            ("b", StrainMeasure::LeftCauchyGreen),
            ("log", StrainMeasure::Hencky),
            ("almansi", StrainMeasure::EulerAlmansi),
        ]
    }
}

// ─── The measures, from `H = ∇u` ───────────────────────────────────────────
//
// Pure, branch-free, on the stack: they run at every Gauss point. A degenerate
// `F` (det F = 0) yields inf/NaN for Hencky and Almansi — the mathematical
// value, not a checked case.

/// `ε = ½(H + Hᵀ)` — summed in the order the operator always used, so the
/// linear measure stays bit-for-bit what it was.
fn linear(h: &Matrix3<f64>) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| 0.5 * (h[(i, j)] + h[(j, i)]))
}

fn gradient(h: &Matrix3<f64>) -> Matrix3<f64> {
    Matrix3::identity() + h
}

fn right_cauchy_green(h: &Matrix3<f64>) -> Matrix3<f64> {
    let f = gradient(h);
    f.transpose() * f
}

fn left_cauchy_green(h: &Matrix3<f64>) -> Matrix3<f64> {
    let f = gradient(h);
    f * f.transpose()
}

fn green_lagrange(h: &Matrix3<f64>) -> Matrix3<f64> {
    0.5 * (right_cauchy_green(h) - Matrix3::identity())
}

/// `½ ln C = ½ Σ ln λₖ nₖ nₖᵀ`, `(λₖ, nₖ)` the eigenpairs of `C`.
fn hencky(h: &Matrix3<f64>) -> Matrix3<f64> {
    let eig = right_cauchy_green(h).symmetric_eigen();
    let mut out = Matrix3::zeros();
    for k in 0..3 {
        let n = eig.eigenvectors.column(k);
        out += (0.5 * eig.eigenvalues[k].ln()) * (n * n.transpose());
    }
    out
}

/// `½(I − b⁻¹)`, `b⁻¹` as adjugate over determinant: no `Option` to unwrap
/// at the Gauss point — a singular `b` gives inf, its mathematical value.
fn euler_almansi(h: &Matrix3<f64>) -> Matrix3<f64> {
    let b = left_cauchy_green(h);
    let adj = Matrix3::new(
        b[(1, 1)] * b[(2, 2)] - b[(1, 2)] * b[(2, 1)],
        b[(0, 2)] * b[(2, 1)] - b[(0, 1)] * b[(2, 2)],
        b[(0, 1)] * b[(1, 2)] - b[(0, 2)] * b[(1, 1)],
        b[(1, 2)] * b[(2, 0)] - b[(1, 0)] * b[(2, 2)],
        b[(0, 0)] * b[(2, 2)] - b[(0, 2)] * b[(2, 0)],
        b[(0, 2)] * b[(1, 0)] - b[(0, 0)] * b[(1, 2)],
        b[(1, 0)] * b[(2, 1)] - b[(1, 1)] * b[(2, 0)],
        b[(0, 1)] * b[(2, 0)] - b[(0, 0)] * b[(2, 1)],
        b[(0, 0)] * b[(1, 1)] - b[(0, 1)] * b[(1, 0)],
    );
    let det = b[(0, 0)] * adj[(0, 0)] + b[(0, 1)] * adj[(1, 0)] + b[(0, 2)] * adj[(2, 0)];
    0.5 * (Matrix3::identity() - adj / det)
}

/// Deformation of a displacement field `u` at the Gauss points of every
/// subspace of `fespace`, in the chosen [`StrainMeasure`].
///
/// `u` must carry exactly `space_dim` components, taken **in order** as the
/// displacement along x, y, z. The result has one component
/// `<prefix>_<ai><aj>` per produced entry ([`StrainMeasure::prefix`]):
/// `i ≤ j` for a symmetric measure, in order `xx, xy, …, yy, …`; every
/// `(i, j)`, row-major, for the gradient `F`. In 2-D, the out-of-plane
/// direction is undeformed (`F_zz = 1`) and its entries are not produced.
///
/// On an [axisymmetric](crate::coords::Coords::axisymmetric) subspace a
/// component `<prefix>_zz` is appended: the **hoop** entry, built on
/// `H_θθ = u_r / r`, which the meridian gradient cannot express
/// (`eps_zz = u_r / r`, `F_zz = 1 + u_r / r`, …).
///
/// A degenerate deformation (`det F = 0`) gives non-finite Hencky and
/// Almansi values: nothing is checked at the Gauss point.
///
/// Runs on the shared parallel driver `models::kernel::nodal_pointwise`,
/// like [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// use pyrucast::ops::element_field::StrainMeasure;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // An overall **translation** deforms nothing.
/// let u = NodeField::from_submesh(&support.get(0)?, vec!["u_x".into(), "u_y".into()])?;
/// u.get(0)?.write().add_to_component("u_x", 0.5)?;
/// let eps = element_field::deformation(&u, &fes, StrainMeasure::Linear)?;
/// assert!(eps.get(0)?.read().value(0, 0, "eps_xx")?.abs() < 1e-12);
/// // A uniform stretch u_x = 0.1·x (λ = 1.1) reads differently in each measure.
/// # let x = node_field::positions(&maillage, None)?;
/// let etire = NodeField::from_submesh(&support.get(0)?, vec!["u_x".into(), "u_y".into()])?;
/// for i in 0..3 {
///     let v = x.get(0)?.read().value(n[i].id(), "X")? * 0.1;
///     etire.get(0)?.write().set_value(n[i].id(), "u_x", v)?;
/// }
/// let lu = |m, c: &str| -> pyrucast::Result<f64> {
///     element_field::deformation(&etire, &fes, m)?.get(0)?.read().value(0, 0, c)
/// };
/// assert!((lu(StrainMeasure::Linear, "eps_xx")? - 0.1).abs() < 1e-12);
/// assert!((lu(StrainMeasure::GreenLagrange, "E_xx")? - 0.105).abs() < 1e-12);
/// assert!((lu(StrainMeasure::Hencky, "H_xx")? - 1.1_f64.ln()).abs() < 1e-12);
/// assert!((lu(StrainMeasure::EulerAlmansi, "e_xx")? - 0.5 * (1.0 - 1.0 / 1.21)).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn deformation(
    u: &NodeField,
    fespace: &FiniteElementSpace,
    measure: StrainMeasure,
) -> Result<ElementField> {
    let components = Field::components(u);
    let view = u.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        out.add_sub(Handle::new(deformation_zone(
            sub,
            &view,
            &components,
            measure,
        )?))?;
    }
    Ok(out)
}

/// [`deformation`] on one zone, reading the displacement from `components`
/// (named, along x, y, z in that order) — the per-zone body shared with
/// [`strain`](fn@crate::ops::element_field::strain), which names a law's
/// own displacement instead of every component of a solution.
pub(crate) fn deformation_zone(
    sub: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    components: &[String],
    measure: StrainMeasure,
) -> Result<SubElementField> {
    // Everything is settled here, once per zone: the Gauss loop below
    // neither tests nor allocates.
    kernel::require_field_basis(sub, "shape values")?;
    let (space_dim, axisymmetric) = {
        let s = sub.read();
        (s.space_dim(), s.is_axisymmetric())
    };
    if components.len() != space_dim {
        return Err(PyrucastError::Message(format!(
            "deformation: the displacement field carries {} component(s) but the FE \
             space is {}-D — expected exactly one displacement component per axis",
            components.len(),
            space_dim
        )));
    }
    // The entries (i, j) to write and their names: `i ≤ j` for a symmetric
    // measure, every pair for F.
    let prefix = measure.prefix();
    let mut slots = Vec::with_capacity(space_dim * space_dim + 1);
    for i in 0..space_dim {
        let first = if measure.is_symmetric() { i } else { 0 };
        for j in first..space_dim {
            slots.push((i, j));
        }
    }
    // On a body of revolution the hoop entry is an independent component.
    // It is appended last and named `zz` (Cast3M's naming, x = r and y = z),
    // free of clashes since the meridian plane only spans x and y. It is one
    // more table entry, not a branch at the Gauss point.
    if axisymmetric {
        slots.push((2, 2));
    }
    let names = slots
        .iter()
        .map(|&(i, j)| format!("{prefix}_{}{}", AXES[i], AXES[j]))
        .collect();
    // The measure is chosen here, by monomorphisation: each arm compiles
    // its own kernel, with no `match` at the Gauss point.
    match measure {
        StrainMeasure::Linear => produce(sub, view, components, names, &slots, linear),
        StrainMeasure::Gradient => produce(sub, view, components, names, &slots, gradient),
        StrainMeasure::GreenLagrange => {
            produce(sub, view, components, names, &slots, green_lagrange)
        }
        StrainMeasure::RightCauchyGreen => {
            produce(sub, view, components, names, &slots, right_cauchy_green)
        }
        StrainMeasure::LeftCauchyGreen => {
            produce(sub, view, components, names, &slots, left_cauchy_green)
        }
        StrainMeasure::Hencky => produce(sub, view, components, names, &slots, hencky),
        StrainMeasure::EulerAlmansi => produce(sub, view, components, names, &slots, euler_almansi),
    }
}

/// One zone, one measure: picks the axisymmetric or Cartesian kernel — a fact
/// of the zone — and hands it to the shared driver.
fn produce<M>(
    sub: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    components: &[String],
    names: Vec<String>,
    slots: &[(usize, usize)],
    measure: M,
) -> Result<SubElementField>
where
    M: Fn(&Matrix3<f64>) -> Matrix3<f64> + Sync,
{
    if sub.read().is_axisymmetric() {
        kernel::nodal_pointwise(sub, view, components, names, |geom, g, dofs, out| {
            point::<true, M>(geom, g, dofs, out, slots, &measure)
        })
    } else {
        kernel::nodal_pointwise(sub, view, components, names, |geom, g, dofs, out| {
            point::<false, M>(geom, g, dofs, out, slots, &measure)
        })
    }
}

/// Point kernel: `H = ∇u` with `∂u_i/∂x_j = Σ_k u_i(k)·∂N_k/∂x_j`, then the
/// measure, then the requested entries. `dofs` holds the cell's displacements,
/// node-major: the driver gathered them once, so this is arithmetic alone.
/// `AXI` is a constant: the hoop term costs nothing where it does not apply.
#[inline]
fn point<const AXI: bool, M>(
    geom: &kernel::CellGeom,
    g: usize,
    dofs: &[f64],
    out: &mut [f64],
    slots: &[(usize, usize)],
    measure: &M,
) -> Result<()>
where
    M: Fn(&Matrix3<f64>) -> Matrix3<f64>,
{
    let sd = geom.space_dim;
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    let dn_dx = &mut dn_buf[..geom.n_nodes * sd];
    geom.dn_dx(g, dn_dx)?;
    let mut h = Matrix3::<f64>::zeros();
    for i in 0..sd {
        for j in 0..sd {
            let mut dij = 0.0;
            for k in 0..geom.n_nodes {
                dij += dofs[k * sd + i] * dn_dx[k * sd + j];
            }
            h[(i, j)] = dij;
        }
    }
    if AXI {
        // H_θθ = u_r / r, with u_r interpolated at the Gauss point.
        let mut n_buf = [0.0_f64; MAX_CELL_DOFS];
        let n = geom.field_n_at_g(g, &mut n_buf);
        let mut u_r = 0.0;
        for k in 0..geom.n_nodes {
            u_r += dofs[k * sd] * n[k];
        }
        h[(2, 2)] = u_r / geom.radius(g);
    }
    let m = measure(&h);
    for (o, &(i, j)) in out.iter_mut().zip(slots) {
        *o = m[(i, j)];
    }
    Ok(())
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Linear displacement `u_x = 2x + 0.5y`, `u_y = 0.1x + 3y` on a TRI3.
    /// ⇒ ε_xx = 2, ε_yy = 3, ε_xy = ½(0.5 + 0.1) = 0.3.
    #[test]
    fn linear_displacement_gives_constant_strain() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        // u_x = 2x + 0.5y, u_y = 0.1x + 3y at the three nodes.
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 2.0 * x + 0.5 * y).unwrap();
            u.set_value(n.id(), "u_y", 0.1 * x + 3.0 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes, StrainMeasure::Linear).unwrap();
        {
            let s = strain.get(0).unwrap().read();
            assert_eq!(
                s.components(),
                &[
                    "eps_xx".to_string(),
                    "eps_xy".to_string(),
                    "eps_yy".to_string()
                ]
            );
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "eps_xx").unwrap() - 2.0).abs() < 1e-12);
                assert!((s.value(0, g, "eps_yy").unwrap() - 3.0).abs() < 1e-12);
                assert!((s.value(0, g, "eps_xy").unwrap() - 0.3).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rejects_displacement_with_wrong_component_count() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // 2-D space but a single displacement component.
        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let u = NodeField::from_sub(SubNodeField::from_poi1(&support, vec!["u_x".into()]).unwrap());
        let err = deformation(&u, &fes, StrainMeasure::Linear).unwrap_err();
        assert!(format!("{err}").contains("one displacement component per axis"));
    }

    /// A single cell whose nodal displacements are `u = A·x` — a homogeneous
    /// deformation, so `∇u = A` exactly at every Gauss point. A TRI3 in 2-D, a
    /// TET4 in 3-D, a TRI3 in the meridian plane when `axisymmetric` (kept off
    /// the axis, r ∈ [1, 2]).
    fn homogeneous(
        a: &[[f64; 3]; 3],
        dim: usize,
        axisymmetric: bool,
    ) -> (NodeField, FiniteElementSpace) {
        let coords = Handle::new(if axisymmetric {
            Coords::axisymmetric().unwrap()
        } else {
            Coords::new(dim as u8).unwrap()
        });
        let (ty, pts): (ElementType, Vec<[f64; 3]>) = if dim == 2 {
            let o = if axisymmetric { 1.0 } else { 0.0 };
            (
                ElementType::TRI3,
                vec![[o, 0.0, 0.0], [o + 1.0, 0.0, 0.0], [o, 1.0, 0.0]],
            )
        } else {
            (
                ElementType::TET4,
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        };
        let nodes: Vec<Node> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), &p[..dim]).unwrap())
            .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ty));
        let ids: Vec<_> = nodes.iter().map(|n| n.id()).collect();
        mesh.add_cell(&ids).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&nodes).unwrap());
        let names: Vec<String> = AXES[..dim].iter().map(|a| format!("u_{a}")).collect();
        let mut u = SubNodeField::from_poi1(&support, names.clone()).unwrap();
        for (n, p) in nodes.iter().zip(&pts) {
            for i in 0..dim {
                let v: f64 = (0..dim).map(|j| a[i][j] * p[j]).sum();
                u.set_value(n.id(), &names[i], v).unwrap();
            }
        }
        (NodeField::from_sub(u), fes)
    }

    /// Every component `<prefix>_<ai><aj>` of `measure` at every Gauss point,
    /// against the expected `want(i, j)`.
    fn assert_measure(
        u: &NodeField,
        fes: &FiniteElementSpace,
        measure: StrainMeasure,
        want: impl Fn(usize, usize) -> f64,
        tol: f64,
    ) {
        let ef = deformation(u, fes, measure).unwrap();
        let s = ef.get(0).unwrap().read();
        for name in s.components() {
            let tail = name
                .strip_prefix(&format!("{}_", measure.prefix()))
                .unwrap();
            let axis = |c| AXES.iter().position(|a| a.starts_with(c)).unwrap();
            let mut chars = tail.chars();
            let (i, j) = (axis(chars.next().unwrap()), axis(chars.next().unwrap()));
            for g in 0..s.gauss_count() {
                let got = s.value(0, g, name).unwrap();
                let w = want(i, j);
                assert!(
                    (got - w).abs() < tol,
                    "{measure:?} {name}: got {got}, want {w}"
                );
            }
        }
    }

    fn delta(i: usize, j: usize) -> f64 {
        if i == j {
            1.0
        } else {
            0.0
        }
    }

    /// The independent reference: dense 3×3 algebra written out by hand, not
    /// the operator's own path.
    fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut t = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                t[i][j] = a[j][i];
            }
        }
        t
    }

    #[test]
    fn names_follow_the_measure_and_the_dimension() {
        let (u, fes) = homogeneous(&[[0.0; 3]; 3], 2, false);
        let names = |m| {
            deformation(&u, &fes, m)
                .unwrap()
                .get(0)
                .unwrap()
                .read()
                .components()
                .to_vec()
        };
        assert_eq!(
            names(StrainMeasure::Gradient),
            ["F_xx", "F_xy", "F_yx", "F_yy"]
        );
        assert_eq!(
            names(StrainMeasure::GreenLagrange),
            ["E_xx", "E_xy", "E_yy"]
        );
        assert_eq!(
            names(StrainMeasure::LeftCauchyGreen),
            ["b_xx", "b_xy", "b_yy"]
        );
        let (u, fes) = homogeneous(&[[0.0; 3]; 3], 3, false);
        let gradient = deformation(&u, &fes, StrainMeasure::Gradient).unwrap();
        assert_eq!(gradient.get(0).unwrap().read().components().len(), 9);
        let hencky = deformation(&u, &fes, StrainMeasure::Hencky).unwrap();
        assert_eq!(
            hencky.get(0).unwrap().read().components(),
            ["H_xx", "H_xy", "H_xz", "H_yy", "H_yz", "H_zz"]
        );
        let (u, fes) = homogeneous(&[[0.0; 3]; 3], 2, true);
        let almansi = deformation(&u, &fes, StrainMeasure::EulerAlmansi).unwrap();
        assert_eq!(
            almansi.get(0).unwrap().read().components(),
            ["e_xx", "e_xy", "e_yy", "e_zz"]
        );
    }

    #[test]
    fn names_round_trip() {
        use crate::named::Named;
        for &m in StrainMeasure::VALUES {
            assert_eq!(StrainMeasure::parse(m.name()).unwrap(), m);
        }
        assert_eq!(
            StrainMeasure::parse("b").unwrap(),
            StrainMeasure::LeftCauchyGreen
        );
        assert_eq!(StrainMeasure::parse("log").unwrap(), StrainMeasure::Hencky);
    }

    /// A finite rigid rotation (30°) strains nothing in any finite measure,
    /// while the linearized ε does not vanish — the reason they exist.
    #[test]
    fn finite_rotation_is_strain_free() {
        let (c, s) = (30_f64.to_radians().cos(), 30_f64.to_radians().sin());
        // R about z; F = R ⇒ H = R − I.
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut a = r;
        for (i, row) in a.iter_mut().enumerate() {
            row[i] -= 1.0;
        }
        for dim in [2, 3] {
            let (u, fes) = homogeneous(&a, dim, false);
            for m in [
                StrainMeasure::GreenLagrange,
                StrainMeasure::Hencky,
                StrainMeasure::EulerAlmansi,
            ] {
                assert_measure(&u, &fes, m, |_, _| 0.0, 1e-12);
            }
            for m in [
                StrainMeasure::RightCauchyGreen,
                StrainMeasure::LeftCauchyGreen,
            ] {
                assert_measure(&u, &fes, m, delta, 1e-12);
            }
            assert_measure(&u, &fes, StrainMeasure::Gradient, |i, j| r[i][j], 1e-12);
            assert_measure(
                &u,
                &fes,
                StrainMeasure::Linear,
                |i, j| if i == j && i < 2 { c - 1.0 } else { 0.0 },
                1e-12,
            );
        }
    }

    /// Simple shear `u_x = γ·y`, in 3-D: the closed forms of every measure.
    #[test]
    fn simple_shear_matches_the_closed_forms() {
        let gamma = 0.4;
        let a = [[0.0, gamma, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut f = a;
        for (i, row) in f.iter_mut().enumerate() {
            row[i] += 1.0;
        }
        let c = mul(&transpose(&f), &f);
        let b = mul(&f, &transpose(&f));
        let (u, fes) = homogeneous(&a, 3, false);
        assert_measure(
            &u,
            &fes,
            StrainMeasure::RightCauchyGreen,
            |i, j| c[i][j],
            1e-12,
        );
        assert_measure(
            &u,
            &fes,
            StrainMeasure::LeftCauchyGreen,
            |i, j| b[i][j],
            1e-12,
        );
        // E = ½(C − I): E_xy = γ/2, E_yy = γ²/2.
        assert_measure(
            &u,
            &fes,
            StrainMeasure::GreenLagrange,
            |i, j| match (i, j) {
                (0, 1) | (1, 0) => gamma / 2.0,
                (1, 1) => gamma * gamma / 2.0,
                _ => 0.0,
            },
            1e-12,
        );
        // b⁻¹ = [[1, −γ, 0], [−γ, 1+γ², 0], [0, 0, 1]] ⇒ e_xy = γ/2, e_yy = −γ²/2.
        assert_measure(
            &u,
            &fes,
            StrainMeasure::EulerAlmansi,
            |i, j| match (i, j) {
                (0, 1) | (1, 0) => gamma / 2.0,
                (1, 1) => -gamma * gamma / 2.0,
                _ => 0.0,
            },
            1e-12,
        );
        // Hencky has a closed form in simple shear too: with sinh β = γ/2,
        // ln U = β/√(1+γ²/4)·[[−γ/2, 1, 0], [1, γ/2, 0], [0, 0, 0]].
        let beta = (gamma / 2.0).asinh();
        let k = beta / (1.0 + gamma * gamma / 4.0).sqrt();
        assert_measure(
            &u,
            &fes,
            StrainMeasure::Hencky,
            |i, j| match (i, j) {
                (0, 0) => -k * gamma / 2.0,
                (1, 1) => k * gamma / 2.0,
                (0, 1) | (1, 0) => k,
                _ => 0.0,
            },
            1e-12,
        );
    }

    /// Small displacements: every strain measure agrees with ε to second order.
    #[test]
    fn finite_measures_reduce_to_epsilon_for_small_gradients() {
        let s = 1e-6;
        let a = [
            [0.3 * s, 0.5 * s, -0.2 * s],
            [0.1 * s, -0.4 * s, 0.7 * s],
            [0.6 * s, 0.2 * s, 0.9 * s],
        ];
        let (u, fes) = homogeneous(&a, 3, false);
        let eps = |i: usize, j: usize| 0.5 * (a[i][j] + a[j][i]);
        for m in [
            StrainMeasure::GreenLagrange,
            StrainMeasure::Hencky,
            StrainMeasure::EulerAlmansi,
        ] {
            assert_measure(&u, &fes, m, eps, 1e-11);
        }
    }

    /// Axisymmetric radial expansion `u_r = a·r`: F_θθ = 1 + a, and the hoop
    /// entry of every measure follows from it.
    #[test]
    fn axisymmetric_hoop_entry() {
        let a = 0.2;
        let (u, fes) = homogeneous(&[[a, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0; 3]], 2, true);
        let hoop = |m| {
            let ef = deformation(&u, &fes, m).unwrap();
            let s = ef.get(0).unwrap().read();
            let name = format!("{}_zz", StrainMeasure::prefix(m));
            s.value(0, 0, &name).unwrap()
        };
        let lambda: f64 = 1.0 + a;
        assert!((hoop(StrainMeasure::Linear) - a).abs() < 1e-12);
        assert!((hoop(StrainMeasure::Gradient) - lambda).abs() < 1e-12);
        assert!((hoop(StrainMeasure::GreenLagrange) - (a + a * a / 2.0)).abs() < 1e-12);
        assert!((hoop(StrainMeasure::RightCauchyGreen) - lambda * lambda).abs() < 1e-12);
        assert!((hoop(StrainMeasure::Hencky) - lambda.ln()).abs() < 1e-12);
        assert!(
            (hoop(StrainMeasure::EulerAlmansi) - 0.5 * (1.0 - 1.0 / (lambda * lambda))).abs()
                < 1e-12
        );
    }
}
