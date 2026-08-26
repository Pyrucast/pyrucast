//! [`Interpolation`] — the Lagrange degree of an element.
//!
//! The enum is a **tag**, not a table: the shape functions themselves live in
//! each element's own file, behind
//! [`ElementKind::shape_into`](super::ElementKind::shape_into). Every method
//! here forwards to [`ElementType::as_kind`], so a new element type needs no
//! change in this file.
//!
//! That holds for the **Lagrange** families, whose pairing with an element type
//! is a bijection — a `TRI3` is Lagrange-1 and a `TRI6` is Lagrange-2, never the
//! other way round, which is what
//! [`ElementKind::degree`](super::ElementKind::degree) states.
//!
//! [`Hermite3`](Interpolation::Hermite3) is the exception that shows why the
//! enum is worth keeping separate. Its basis has **two functions per node** —
//! a value and a slope — so it cannot be a property of the element type, and it
//! lives in [`super::hermite`] instead. A `SEG2` therefore carries a Lagrange-1
//! *or* a Hermite-3 field depending on what the space declares, while its
//! **geometry** stays Lagrange-1 in both cases: the element becomes
//! subparametric, which is precisely the case this enum existed to leave room
//! for.

use crate::atoms::ElementType;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported finite-element interpolations.
///
/// Names mix cast3m and standard FE terminology. See
/// [`crate::atoms::element_type`] for the conventions on reference frames and
/// node order.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Interpolation};
/// // Le degré d'interpolation du **champ**, distinct de celui de la
/// // géométrie : un élément peut être sous-paramétrique.
/// assert_eq!(Interpolation::Lagrange1.shape_count(ElementType::TRI3), 3);
/// // Hermite porte deux fonctions par nœud — une valeur, une pente.
/// assert_eq!(Interpolation::Hermite3.shape_count(ElementType::SEG2), 4);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    /// Linear Lagrange (P1 / Q1). One shape function per geometric node, equal
    /// to 1 at "its" node and 0 at the others. Defined for the linear element
    /// types (`SEG2`, `TRI3`, `QUA4`, `TET4`, `PYRA5`, `PENTA6`, `HEX8`).
    Lagrange1,
    /// Quadratic Lagrange (P2 / Q2 serendipity). One shape function per
    /// geometric node — corners **and** mid-edge nodes. Defined for the
    /// quadratic element types (`SEG3`, `TRI6`, `QUA8`, `QUA9`, `TET10`,
    /// `PENTA15`, `HEX20`, `HEX27`). `QUA8`/`HEX20`/`PENTA15` are serendipity
    /// (edge nodes only); `QUA9`/`HEX27` carry the face/centre nodes too.
    Lagrange2,
    /// Cubic **Hermite** (C¹), on `SEG2` only. **Two** shape functions per node
    /// — a value and a slope — so a segment carries four, and the interpolated
    /// field is continuous *in its derivative* across elements. That is what a
    /// fourth-order equation such as Euler-Bernoulli's `(EIw'')'' = q` demands,
    /// and what no Lagrange basis provides.
    ///
    /// The geometry stays Lagrange-1: the element is **subparametric**.
    ///
    /// > New variants go at the **end** — `bincode` serialises the index.
    Hermite3,
    /// **No field basis at all**: the formulation owns it.
    ///
    /// A structural element whose element matrix is a closed form — a bar, a
    /// frame — never evaluates a shape function. Its basis was integrated once,
    /// on paper, and only the result is in the code. Declaring a Lagrange
    /// interpolation on such a space states something the physics does not use,
    /// and — worse — something that may be **false**: the closed form of an
    /// exact frame element comes from cubic/quadratic functions, not linear
    /// ones.
    ///
    /// This variant says so. The **geometry** stays Lagrange, so coordinates,
    /// Jacobian and measure are all well defined and the whole assembly
    /// plumbing is unchanged; only the interpolation of the *unknown* is
    /// declared absent. Any operator that tries to evaluate a field inside such
    /// an element gets an error naming the situation, instead of silently
    /// falling back to a linear basis the formulation never used.
    ///
    /// It is named for **where the basis lives**, not for how the integral was
    /// done: a formulation could own its basis and still integrate it
    /// internally, and this would still be the right declaration.
    ModelEmbedded,
}

impl Interpolation {
    /// Whether this interpolation is defined for `element_type`.
    ///
    /// For the Lagrange families this asks whether it is that element's own
    /// degree; `Hermite3` is defined on `SEG2` alone. `POI1` has no reference
    /// frame and is always rejected.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// assert!(Interpolation::Lagrange1.is_compatible_with(ElementType::TRI3));
    /// // Hermite cubique n'est défini que sur le segment.
    /// assert!(Interpolation::Hermite3.is_compatible_with(ElementType::SEG2));
    /// assert!(!Interpolation::Hermite3.is_compatible_with(ElementType::TRI3));
    /// ```
    pub fn is_compatible_with(self, element_type: ElementType) -> bool {
        match self {
            Self::Lagrange1 | Self::Lagrange2 => element_type.as_kind().degree() == Some(self),
            Self::Hermite3 => element_type == ElementType::SEG2,
            // Any element whose geometry is defined: the field basis is the
            // model's business, the space only has to place the points.
            Self::ModelEmbedded => element_type.as_kind().degree().is_some(),
        }
    }

    /// Whether the space declares **no** field basis, the formulation owning it.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// // « Aucune base de champ » : la formulation possède la sienne, sous
    /// // forme close — une barre, un portique.
    /// assert!(Interpolation::ModelEmbedded.is_model_embedded());
    /// assert!(!Interpolation::Lagrange1.is_model_embedded());
    /// ```
    pub fn is_model_embedded(self) -> bool {
        matches!(self, Self::ModelEmbedded)
    }

    /// Number of **shape functions** per cell — which is the number of nodes
    /// only for the Lagrange families.
    ///
    /// This is the stride every reference-space table is built on. Reading it
    /// instead of `nodes_per_cell()` is what lets a basis carry more than one
    /// function per node.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// // Le nombre de nœuds pour les familles de Lagrange…
    /// assert_eq!(Interpolation::Lagrange1.shape_count(ElementType::QUA4), 4);
    /// assert_eq!(Interpolation::Lagrange2.shape_count(ElementType::QUA8), 8);
    /// // …et le double pour Hermite, qui tabule aussi les pentes.
    /// assert_eq!(Interpolation::Hermite3.shape_count(ElementType::SEG2), 4);
    /// ```
    pub fn shape_count(self, element_type: ElementType) -> usize {
        match self {
            Self::Lagrange1 | Self::Lagrange2 => element_type.nodes_per_cell(),
            Self::Hermite3 => super::hermite::HERMITE3_SHAPE_COUNT,
            // None, and that is the statement — not "zero functions" but "not
            // this space's to say".
            Self::ModelEmbedded => 0,
        }
    }

    /// Degrees of freedom each node carries **per scalar field**: 1 for
    /// Lagrange (the value), 2 for Hermite (the value and its slope).
    ///
    /// A physics reads this to know whether a nodal variable it declares is an
    /// independent field or the derivative of another one.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// assert_eq!(Interpolation::Lagrange1.dofs_per_node(), 1);
    /// assert_eq!(Interpolation::Hermite3.dofs_per_node(), 2); // valeur + pente
    /// ```
    pub fn dofs_per_node(self) -> usize {
        match self {
            Self::Lagrange1 | Self::Lagrange2 => 1,
            Self::Hermite3 => 2,
            Self::ModelEmbedded => 0,
        }
    }

    /// Whether the basis is C¹ — i.e. interpolates a slope as well as a value.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// assert!(Interpolation::Hermite3.is_hermite());
    /// assert!(!Interpolation::Lagrange2.is_hermite());
    /// ```
    pub fn is_hermite(self) -> bool {
        matches!(self, Self::Hermite3)
    }

    /// Short name (cast3m-style).
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// assert_eq!(Interpolation::Lagrange1.name(), "LAGRANGE1");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Lagrange1 => "LAGRANGE1",
            Self::Lagrange2 => "LAGRANGE2",
            Self::Hermite3 => "HERMITE3",
            Self::ModelEmbedded => "MODEL_EMBEDDED",
        }
    }

    /// Evaluate the shape functions `N_i(ξ)` at the reference point `xi`.
    ///
    /// Returns a flat `Vec<f64>` of length
    /// [`shape_count`](Self::shape_count) — the cell's nodes for Lagrange, and
    /// `[w_A, w'_A, w_B, w'_B]` for Hermite. `xi` must have length
    /// `element_type.topological_dim()`.
    ///
    /// # Errors
    ///
    /// - `xi` has the wrong length;
    /// - the `(self, element_type)` pair is not supported (`POI1`, a degree
    ///   that is not the element's own, …).
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// // Au centre du triangle de référence, les trois N_i valent 1/3 —
    /// // et somment à 1, comme partout.
    /// let n = Interpolation::Lagrange1.shape(ElementType::TRI3, &[1.0 / 3.0, 1.0 / 3.0])?;
    /// assert!(n.iter().all(|v| (v - 1.0 / 3.0).abs() < 1e-12));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn shape(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check(element_type, xi)?;
        match self {
            Self::Lagrange1 | Self::Lagrange2 => Ok(element_type.as_kind().shape(xi)),
            Self::Hermite3 => Ok(super::hermite::shape(xi[0]).to_vec()),
            Self::ModelEmbedded => Err(Self::embedded_refusal("shape functions")),
        }
    }

    /// The message every field evaluator shares — one wording, so a caller
    /// meets the same explanation whichever accessor it reached for.
    fn embedded_refusal(what: &str) -> PyrucastError {
        PyrucastError::Message(format!(
            "interpolation MODEL_EMBEDDED declares no field basis, so it has no {what}: the \
             formulation owns its interpolation (a closed-form structural element). Interpolating \
             a field inside such an element is the formulation's business, not the space's."
        ))
    }

    /// Evaluate the reference derivatives `∂N_i/∂ξ_j` at `xi`.
    ///
    /// Returns a flat row-major buffer of length
    /// `shape_count × topological_dim`, where entry
    /// `[i * topological_dim + j]` is `∂N_i/∂ξ_j`.
    ///
    /// # Errors
    ///
    /// Same as [`shape`](Self::shape).
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// // Sur un TRI3 les dérivées sont constantes : ∂N/∂ξ = [-1, 1, 0].
    /// let d = Interpolation::Lagrange1.dshape_dxi(ElementType::TRI3, &[0.0, 0.0])?;
    /// assert_eq!(d.len(), 3 * 2); // ∂N_i/∂ξ_k, à plat
    /// // Elles somment au vecteur nul : la partition de l'unité, dérivée.
    /// assert!((d[0] + d[2] + d[4]).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn dshape_dxi(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check(element_type, xi)?;
        match self {
            Self::Lagrange1 | Self::Lagrange2 => Ok(element_type.as_kind().dshape(xi)),
            Self::Hermite3 => Ok(super::hermite::dshape(xi[0]).to_vec()),
            Self::ModelEmbedded => Err(Self::embedded_refusal("reference derivatives")),
        }
    }

    /// Evaluate the reference **second** derivatives `∂²N_i/∂ξ²` at `xi`.
    ///
    /// Defined for `Hermite3` only, and that is not an oversight: a C¹ basis is
    /// the case where a second derivative is a *primary* quantity — the
    /// curvature a beam is bent by. The Lagrange families would need
    /// second-derivative tables in every element file, plus the
    /// `∂J/∂ξ` term of the chain rule on a curved element, and nothing consumes
    /// them yet; adding an untested path would be worse than erroring.
    ///
    /// # Errors
    ///
    /// Same as [`shape`](Self::shape), plus a Lagrange family.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Interpolation};
    /// // Les dérivées **secondes** : ce qu'exige une équation d'ordre quatre,
    /// // et que seules les familles C¹ tabulent.
    /// let d2 = Interpolation::Hermite3.d2shape_dxi2(ElementType::SEG2, &[0.0])?;
    /// assert!(d2.iter().any(|v| v.abs() > 1e-12));
    /// // Une base de Lagrange n'en a pas — et le dit plutôt que de rendre zéro.
    /// assert!(Interpolation::Lagrange1.d2shape_dxi2(ElementType::SEG2, &[0.0]).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn d2shape_dxi2(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check(element_type, xi)?;
        match self {
            Self::Hermite3 => Ok(super::hermite::d2shape(xi[0]).to_vec()),
            Self::Lagrange1 | Self::Lagrange2 => Err(PyrucastError::Message(format!(
                "interpolation {self}: second derivatives are only defined for the C¹ families \
                 (HERMITE3); a Lagrange basis has none tabulated"
            ))),
            Self::ModelEmbedded => Err(Self::embedded_refusal("second derivatives")),
        }
    }

    /// Validate the pair and the point's arity in one go — the guard both
    /// evaluators share.
    fn check(self, element_type: ElementType, xi: &[f64]) -> Result<()> {
        if !self.is_compatible_with(element_type) {
            return Err(PyrucastError::Message(format!(
                "interpolation {} is not defined for {}",
                self, element_type
            )));
        }
        let expected = element_type.topological_dim();
        if xi.len() != expected {
            return Err(PyrucastError::Message(format!(
                "xi has length {}, expected {} for {}",
                xi.len(),
                expected,
                element_type
            )));
        }
        Ok(())
    }
}

impl crate::named::Named for Interpolation {
    const LABEL: &'static str = "interpolation";
    const VALUES: &'static [Self] = &[
        Self::Lagrange1,
        Self::Lagrange2,
        Self::Hermite3,
        Self::ModelEmbedded,
    ];

    fn name(self) -> &'static str {
        Interpolation::name(self)
    }

    fn aliases() -> &'static [(&'static str, Self)] {
        &[
            ("LAG1", Self::Lagrange1),
            ("LAG2", Self::Lagrange2),
            ("HER3", Self::Hermite3),
            ("MODELEMBEDDED", Self::ModelEmbedded),
            ("EMBEDDED", Self::ModelEmbedded),
        ]
    }
}

impl fmt::Display for Interpolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl crate::dump::Dump for Interpolation {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::named::Named;

    /// Sum of Lagrange shape functions equals 1 everywhere (partition of
    /// unity).
    fn check_partition_of_unity(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let n = interp.shape(et, xi).unwrap();
        let s: f64 = n.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "{} on {} at xi={:?}: sum N_i = {} ≠ 1",
            interp,
            et,
            xi,
            s
        );
    }

    /// Sum of reference derivatives along each direction is 0 (partition
    /// of unity, differentiated).
    fn check_derivatives_sum_to_zero(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let dn = interp.dshape_dxi(et, xi).unwrap();
        let n_nodes = et.nodes_per_cell();
        let ref_dim = et.topological_dim();
        for j in 0..ref_dim {
            let mut s = 0.0;
            for i in 0..n_nodes {
                s += dn[i * ref_dim + j];
            }
            assert!(
                s.abs() < 1e-12,
                "{} on {}: Σ_i dN_i/dξ_{} = {} ≠ 0",
                interp,
                et,
                j,
                s
            );
        }
    }

    /// At node `i`, `N_i = 1` and `N_j = 0` for `j ≠ i` (Kronecker
    /// property of Lagrange interpolations).
    fn check_kronecker(interp: Interpolation, et: ElementType, ref_nodes: &[&[f64]]) {
        for (i, xi) in ref_nodes.iter().enumerate() {
            let n = interp.shape(et, xi).unwrap();
            for (j, &v) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-12,
                    "{} on {} at node {}: N_{} = {} ≠ {}",
                    interp,
                    et,
                    i,
                    j,
                    v,
                    expected
                );
            }
        }
    }

    /// Central-difference estimate of the reference derivatives, used to
    /// validate the hand-written analytic `dshape_dxi` of the quadratic
    /// elements.
    fn central_diff_dshape(interp: Interpolation, et: ElementType, xi: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        let n_nodes = et.nodes_per_cell();
        let d = et.topological_dim();
        let mut out = vec![0.0; n_nodes * d];
        for k in 0..d {
            let mut xp = xi.to_vec();
            let mut xm = xi.to_vec();
            xp[k] += h;
            xm[k] -= h;
            let np = interp.shape(et, &xp).unwrap();
            let nm = interp.shape(et, &xm).unwrap();
            for i in 0..n_nodes {
                out[i * d + k] = (np[i] - nm[i]) / (2.0 * h);
            }
        }
        out
    }

    fn check_dshape_matches_fd(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let ana = interp.dshape_dxi(et, xi).unwrap();
        let fd = central_diff_dshape(interp, et, xi);
        for (k, (a, f)) in ana.iter().zip(&fd).enumerate() {
            assert!(
                (a - f).abs() < 1e-5,
                "{} on {} at {:?}: dN[{}] analytic {} vs FD {}",
                interp,
                et,
                xi,
                k,
                a,
                f
            );
        }
    }

    #[test]
    fn lagrange2_all_types() {
        let samples: &[(ElementType, &[&[f64]])] = &[
            (ElementType::SEG3, &[&[-0.4], &[0.2], &[0.9]]),
            (ElementType::TRI6, &[&[0.2, 0.3], &[0.1, 0.6]]),
            (ElementType::QUA8, &[&[-0.3, 0.5], &[0.7, -0.2]]),
            (ElementType::QUA9, &[&[-0.3, 0.5], &[0.7, -0.2]]),
            (ElementType::TET10, &[&[0.2, 0.3, 0.1], &[0.1, 0.1, 0.5]]),
            (ElementType::PENTA15, &[&[0.2, 0.3, 0.4], &[0.1, 0.5, 0.8]]),
            (ElementType::HEX20, &[&[-0.3, 0.5, 0.2], &[0.6, -0.4, 0.9]]),
            (ElementType::HEX27, &[&[-0.3, 0.5, 0.2], &[0.6, -0.4, 0.9]]),
        ];
        for &(et, pts) in samples {
            // Kronecker delta at the nodes.
            check_kronecker(Interpolation::Lagrange2, et, et.as_kind().ref_nodes());
            // Partition of unity, derivative sum, and analytic-vs-FD gradient.
            for xi in pts {
                check_partition_of_unity(Interpolation::Lagrange2, et, xi);
                check_derivatives_sum_to_zero(Interpolation::Lagrange2, et, xi);
                check_dshape_matches_fd(Interpolation::Lagrange2, et, xi);
            }
        }
    }

    #[test]
    fn lagrange_degree_matches_element_type() {
        // Degree mismatch is rejected both ways.
        assert!(Interpolation::Lagrange2.is_compatible_with(ElementType::TRI6));
        assert!(!Interpolation::Lagrange2.is_compatible_with(ElementType::TRI3));
        assert!(!Interpolation::Lagrange1.is_compatible_with(ElementType::TRI6));
        assert!(Interpolation::Lagrange1
            .shape(ElementType::HEX20, &[0.0; 3])
            .is_err());
        assert!(Interpolation::Lagrange2
            .shape(ElementType::HEX8, &[0.0; 3])
            .is_err());
    }

    #[test]
    fn lagrange1_seg2() {
        // Reference nodes: ξ = ±1
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::SEG2,
            ElementType::SEG2.as_kind().ref_nodes(),
        );
        for xi in [-1.0, -0.3, 0.0, 0.7, 1.0] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
        }
    }

    #[test]
    fn lagrange1_tri3() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TRI3,
            ElementType::TRI3.as_kind().ref_nodes(),
        );
        for (a, b) in [(0.25, 0.25), (1.0 / 3.0, 1.0 / 3.0), (0.5, 0.0), (0.0, 0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_qua4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::QUA4,
            ElementType::QUA4.as_kind().ref_nodes(),
        );
        for (a, b) in [(0.0, 0.0), (0.3, -0.7), (-0.5, 0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_tet4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TET4,
            ElementType::TET4.as_kind().ref_nodes(),
        );
        for (a, b, c) in [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
        }
    }

    #[test]
    fn lagrange1_hex8() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::HEX8,
            ElementType::HEX8.as_kind().ref_nodes(),
        );
        for (a, b, c) in [(0.0, 0.0, 0.0), (0.3, -0.7, 0.5), (-0.5, 0.5, -0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
        }
    }

    #[test]
    fn lagrange1_penta6() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::PENTA6,
            ElementType::PENTA6.as_kind().ref_nodes(),
        );
        for (a, b, c) in [(1.0 / 3.0, 1.0 / 3.0, 0.5), (0.1, 0.2, 0.7)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::PENTA6, &[a, b, c]);
            check_derivatives_sum_to_zero(
                Interpolation::Lagrange1,
                ElementType::PENTA6,
                &[a, b, c],
            );
        }
    }

    #[test]
    fn rejects_poi1() {
        assert!(!Interpolation::Lagrange1.is_compatible_with(ElementType::POI1));
        assert!(Interpolation::Lagrange1
            .shape(ElementType::POI1, &[])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::POI1, &[])
            .is_err());
    }

    #[test]
    fn rejects_bad_xi_length() {
        assert!(Interpolation::Lagrange1
            .shape(ElementType::SEG2, &[0.0, 0.0])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::TRI3, &[0.0])
            .is_err());
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", Interpolation::Lagrange1), "LAGRANGE1");
        assert_eq!(
            Interpolation::from_name("lagrange1"),
            Some(Interpolation::Lagrange1)
        );
        assert_eq!(
            Interpolation::from_name("LAG1"),
            Some(Interpolation::Lagrange1)
        );
        assert_eq!(Interpolation::from_name("unknown"), None);
    }

    /// Interior sample points of the reference pyramid, well clear of the
    /// apex and one right up against it.
    const PYRA5_SAMPLES: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [0.5, -0.25, 0.25],
        [-0.9, 0.9, 0.05],
        [0.0, 0.0, 0.5],
        [0.02, -0.01, 0.98],
        [0.0, 0.0, 1.0],
    ];

    #[test]
    fn pyra5_is_a_partition_of_unity_up_to_the_apex() {
        for xi in PYRA5_SAMPLES {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::PYRA5, &xi);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::PYRA5, &xi);
        }
    }

    #[test]
    fn pyra5_shape_functions_are_one_at_their_own_node() {
        let nodes = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        for (j, xi) in nodes.iter().enumerate() {
            let n = Interpolation::Lagrange1
                .shape(ElementType::PYRA5, xi)
                .unwrap();
            for (i, &v) in n.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-12, "N_{i} at node {j} = {v}");
            }
        }
    }

    #[test]
    fn pyra5_derivatives_match_finite_differences() {
        // The rational term is where a hand-derived derivative goes wrong, so
        // it is checked against the shape functions themselves.
        let h = 1e-6;
        for xi in [[0.1, -0.2, 0.3], [-0.4, 0.4, 0.1], [0.0, 0.0, 0.6]] {
            let d = Interpolation::Lagrange1
                .dshape_dxi(ElementType::PYRA5, &xi)
                .unwrap();
            for j in 0..3 {
                let (mut lo, mut hi) = (xi, xi);
                lo[j] -= h;
                hi[j] += h;
                let a = Interpolation::Lagrange1
                    .shape(ElementType::PYRA5, &lo)
                    .unwrap();
                let b = Interpolation::Lagrange1
                    .shape(ElementType::PYRA5, &hi)
                    .unwrap();
                for i in 0..5 {
                    let fd = (b[i] - a[i]) / (2.0 * h);
                    assert!(
                        (d[i * 3 + j] - fd).abs() < 1e-7,
                        "∂N_{i}/∂ξ_{j} at {xi:?}: {} vs {fd} by finite difference",
                        d[i * 3 + j]
                    );
                }
            }
        }
    }

    #[test]
    fn pyra5_shape_functions_are_bilinear_on_the_base_and_linear_on_an_edge() {
        // On ζ = 0 the pyramid must reduce to a QUA4, so a hexahedron and a
        // pyramid sharing a face agree along it.
        for &(a, b) in &[(0.3, -0.7), (-0.5, 0.5), (1.0, 1.0)] {
            let n = Interpolation::Lagrange1
                .shape(ElementType::PYRA5, &[a, b, 0.0])
                .unwrap();
            let q = Interpolation::Lagrange1
                .shape(ElementType::QUA4, &[a, b])
                .unwrap();
            for i in 0..4 {
                assert!(
                    (n[i] - q[i]).abs() < 1e-12,
                    "base node {i}: {} vs {}",
                    n[i],
                    q[i]
                );
            }
            assert!(n[4].abs() < 1e-12, "the apex must not reach the base");
        }
        // And along the edge from base node 0 to the apex it is linear, so a
        // tetrahedron sharing that edge agrees too.
        for t in [0.0, 0.25, 0.5, 0.75] {
            let xi = [-(1.0 - t), -(1.0 - t), t];
            let n = Interpolation::Lagrange1
                .shape(ElementType::PYRA5, &xi)
                .unwrap();
            assert!((n[0] - (1.0 - t)).abs() < 1e-12, "N_0 at t={t} = {}", n[0]);
            assert!((n[4] - t).abs() < 1e-12, "N_4 at t={t} = {}", n[4]);
        }
    }
}
