//! Material symmetry — **isotropic**, **orthotropic**, **anisotropic**.
//!
//! Symmetry is an axis of its own, orthogonal to the kinematic hypothesis
//! ([`ElasticityModel`]) and to the physics: elasticity, heat conduction and
//! Fickian diffusion all read their coefficients through this module. A physics
//! stores a [`MaterialSymmetry`] and asks here for its constitutive matrix; it
//! never parses a frame or inverts a compliance itself.
//!
//! ## The frame is given by vectors, not angles
//!
//! An orthotropic (or anisotropic) material carries **material axes**, supplied
//! as ordinary components of the material field:
//!
//! | space | components | meaning |
//! |---|---|---|
//! | 2-D | `V1X`, `V1Y` | the first material axis; the second is its in-plane normal |
//! | 3-D | `V1X…V1Z`, `V2X…V2Z` | the first two axes; the third is `V1 × V2` |
//!
//! They are orthonormalised here (Gram-Schmidt), so `V2` need only be
//! *approximately* perpendicular to `V1` — it is the plane they span that
//! matters. Vectors rather than Euler angles: no convention to remember, no
//! gimbal case, and the frame varies naturally from cell to cell (a wound
//! composite, a rolled plate) since it travels through the material field like
//! any other coefficient.
//!
//! ## How the constitutive matrix is built
//!
//! Always the same three steps, whatever the physics:
//!
//! 1. build the coefficient tensor **in the material axes** (where orthotropy is
//!    diagonal and the anisotropic constants are given);
//! 2. rotate it to the global axes;
//! 3. reduce it to the model (plane stress / plane strain / axisymmetric / solid).
//!
//! The rotation of the elastic tensor goes through the **fourth-order tensor**
//! `C_ijkl`, not through a 6×6 Bond matrix. It costs `3⁸` multiply-adds once per
//! cell — nothing, since the material is constant per cell — and it removes the
//! whole family of index and factor-of-two mistakes that Voigt rotation
//! formulas invite. With **engineering** shear the Voigt ↔ tensor map carries no
//! factor at all: `C_ijkl = D[voigt(i,j)][voigt(k,l)]`.
//!
//! **Isotropy is left untouched**: it short-circuits to
//! [`elasticity::constitutive`] and to
//! the plain scalar conductivity, so every existing assembly keeps its exact
//! numbers.

use crate::containers::element_field::SubElementField;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::{self, ElasticityModel};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// Which material symmetry the coefficients of a physics obey.
///
/// Mirrors Cast3M, where `ISOTROPE` / `ORTHOTROPE` / `ANISOTROPE` qualifies the
/// **material** of a formulation rather than naming a different model — hence an
/// axis carried by the existing physics, not three duplicated ones.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Un axe porté par la physique existante, pas trois physiques dupliquées :
/// // c'est le **matériau** qui est isotrope, orthotrope ou anisotrope.
/// assert!(!MaterialSymmetry::Isotropic.has_frame());
/// assert!(MaterialSymmetry::Orthotropic.has_frame());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialSymmetry {
    /// Two constants, no privileged direction.
    #[default]
    Isotropic,
    /// Three orthogonal material axes, each with its own moduli.
    Orthotropic,
    /// The full tensor of constants, in a material frame.
    Anisotropic,
}

impl MaterialSymmetry {
    /// Parse from a lowercase tag (`"isotropic"`, `"orthotropic"`,
    /// `"anisotropic"`) — the Python-facing spelling.
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
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let mut mat = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
    /// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    /// assert_eq!(MaterialSymmetry::from_tag("orthotropic"),
    ///            Some(MaterialSymmetry::Orthotropic));
    /// assert_eq!(MaterialSymmetry::from_tag("ORTHOTROPE"), None); // minuscules
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "isotropic" => Some(Self::Isotropic),
            "orthotropic" => Some(Self::Orthotropic),
            "anisotropic" => Some(Self::Anisotropic),
            _ => None,
        }
    }

    /// The lowercase tag for this symmetry (the inverse of [`from_tag`](Self::from_tag)).
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
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let mut mat = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
    /// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    /// assert_eq!(MaterialSymmetry::Anisotropic.to_tag(), "anisotropic");
    /// // Réciproque exacte de `from_tag`.
    /// assert_eq!(MaterialSymmetry::from_tag(MaterialSymmetry::Anisotropic.to_tag()),
    ///            Some(MaterialSymmetry::Anisotropic));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Isotropic => "isotropic",
            Self::Orthotropic => "orthotropic",
            Self::Anisotropic => "anisotropic",
        }
    }

    /// Whether this symmetry needs material axes (everything but isotropy).
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
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let mut mat = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
    /// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    /// // Un repère matériau n'a de sens que s'il y a une direction privilégiée.
    /// assert!(!MaterialSymmetry::Isotropic.has_frame());
    /// assert!(MaterialSymmetry::Anisotropic.has_frame());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn has_frame(self) -> bool {
        self != Self::Isotropic
    }
}

impl std::fmt::Display for MaterialSymmetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

// ─── The material frame ─────────────────────────────────────────────────────

/// Frame components required in 2-D: the first axis, in the plane.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // En 2-D, le premier axe suffit : le second est sa normale dans le plan.
/// assert_eq!(symmetry::FRAME_2D, ["V1X", "V1Y"]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const FRAME_2D: [&str; 2] = ["V1X", "V1Y"];
/// Frame components required in 3-D: the first two axes (the third is `V1 × V2`).
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // En 3-D, deux axes ; le troisième est V1 × V2.
/// assert_eq!(symmetry::FRAME_3D.len(), 6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const FRAME_3D: [&str; 6] = ["V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z"];

/// The frame components a symmetry requires in a space of dimension `space_dim`
/// — empty for isotropy, which has no privileged direction.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Ce que le champ matériau doit porter **en plus** des constantes.
/// assert!(symmetry::frame_components(MaterialSymmetry::Isotropic, 3).is_empty());
/// assert_eq!(symmetry::frame_components(MaterialSymmetry::Orthotropic, 2).len(), 2);
/// assert_eq!(symmetry::frame_components(MaterialSymmetry::Orthotropic, 3).len(), 6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn frame_components(symmetry: MaterialSymmetry, space_dim: usize) -> &'static [&'static str] {
    if !symmetry.has_frame() {
        &[]
    } else if space_dim == 2 {
        &FRAME_2D
    } else {
        &FRAME_3D
    }
}

/// The rotation `R` taking **material** axes to **global** axes (its columns are
/// the material axes expressed globally), read from the material field of `cell`.
///
/// In 3-D, `V1` and `V2` are orthonormalised by Gram-Schmidt and the third axis
/// closes a right-handed frame. In 2-D only `V1` is read: the second axis is its
/// in-plane normal and the third is the out-of-plane direction, which keeps the
/// hoop direction of an axisymmetric model as a material axis — as it must be.
///
/// Errors on a degenerate frame (a null `V1`, or a `V2` parallel to it), which
/// would otherwise produce a silently meaningless material.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // R mène des axes **matériau** aux axes globaux : ses colonnes sont les
/// // axes matériau vus globalement. Ici V1 = (0, 1) : un quart de tour.
/// let r = symmetry::frame_rotation(&mat, 0, 2)?;
/// assert!((r[(0, 0)] - 0.0).abs() < 1e-12 && (r[(1, 0)] - 1.0).abs() < 1e-12);
/// // Un repère dégénéré est refusé plutôt que silencieusement absurde.
/// mat.set_uniform("V1Y", 0.0)?;
/// assert!(symmetry::frame_rotation(&mat, 0, 2).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn frame_rotation(
    material: &SubElementField,
    cell: usize,
    space_dim: usize,
) -> Result<Matrix3<f64>> {
    let read = |name: &str| material.value(cell, 0, name);
    let degenerate = |what: &str| {
        PyrucastError::Message(format!(
            "material frame: {what} — the orthotropy axes must be non-null and independent \
             (V1, V2 are orthonormalised, so V2 need only be roughly perpendicular to V1)"
        ))
    };

    if space_dim == 2 {
        let v1 = Vector3::new(read("V1X")?, read("V1Y")?, 0.0);
        let e1 = v1
            .try_normalize(f64::EPSILON)
            .ok_or_else(|| degenerate("V1 is null"))?;
        // The in-plane normal, then the out-of-plane axis: a right-handed frame
        // whose third axis is `z` (the hoop direction in axisymmetry).
        let e2 = Vector3::new(-e1[1], e1[0], 0.0);
        let e3 = Vector3::new(0.0, 0.0, 1.0);
        return Ok(Matrix3::from_columns(&[e1, e2, e3]));
    }

    let v1 = Vector3::new(read("V1X")?, read("V1Y")?, read("V1Z")?);
    let v2 = Vector3::new(read("V2X")?, read("V2Y")?, read("V2Z")?);
    let e1 = v1
        .try_normalize(f64::EPSILON)
        .ok_or_else(|| degenerate("V1 is null"))?;
    // Gram-Schmidt: strip from V2 whatever it has along V1.
    let e2 = (v2 - e1 * v2.dot(&e1))
        .try_normalize(f64::EPSILON)
        .ok_or_else(|| degenerate("V2 is null or parallel to V1"))?;
    let e3 = e1.cross(&e2);
    Ok(Matrix3::from_columns(&[e1, e2, e3]))
}

// ─── Voigt ↔ fourth-order tensor ────────────────────────────────────────────

/// Voigt slot of a tensor index pair, in this crate's order
/// `[xx, yy, zz, yz, xz, xy]`. Symmetric, so `(i,j)` and `(j,i)` agree.
const VOIGT_OF: [[usize; 3]; 3] = [[0, 5, 4], [5, 1, 3], [4, 3, 2]];

/// A fourth-order stiffness tensor `C_ijkl`.
type Tensor4 = [[[[f64; 3]; 3]; 3]; 3];

/// `C_ijkl = D[voigt(i,j)][voigt(k,l)]` — exact, no factor, because `D` is
/// expressed against **engineering** shear (`γ = 2ε`), which is precisely what
/// absorbs the double-counting of the off-diagonal pairs in `C_ijkl ε_kl`.
fn to_tensor(d: &[[f64; 6]; 6]) -> Tensor4 {
    let mut c = [[[[0.0; 3]; 3]; 3]; 3];
    for (i, ci) in c.iter_mut().enumerate() {
        for (j, cij) in ci.iter_mut().enumerate() {
            for (k, cijk) in cij.iter_mut().enumerate() {
                for (l, v) in cijk.iter_mut().enumerate() {
                    *v = d[VOIGT_OF[i][j]][VOIGT_OF[k][l]];
                }
            }
        }
    }
    c
}

/// The inverse map: pick the representative index pair of each Voigt slot.
fn from_tensor(c: &Tensor4) -> [[f64; 6]; 6] {
    const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut d = [[0.0; 6]; 6];
    for (a, &(i, j)) in PAIRS.iter().enumerate() {
        for (b, &(k, l)) in PAIRS.iter().enumerate() {
            d[a][b] = c[i][j][k][l];
        }
    }
    d
}

/// Rotate a fourth-order tensor from material to global axes:
/// `C'_pqrs = R_pi R_qj R_rk R_sl C_ijkl`.
fn rotate_tensor(c: &Tensor4, r: &Matrix3<f64>) -> Tensor4 {
    // Four successive single-index contractions instead of one 8-fold loop:
    // 4·3⁵ = 972 multiply-adds rather than 3⁸ = 6561, and each pass is a plain
    // matrix product along one axis.
    let mut a = *c;
    for _axis in 0..4 {
        let mut b = [[[[0.0; 3]; 3]; 3]; 3];
        for (p, bp) in b.iter_mut().enumerate() {
            for (j, bpj) in bp.iter_mut().enumerate() {
                for (k, bpjk) in bpj.iter_mut().enumerate() {
                    for (l, v) in bpjk.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for i in 0..3 {
                            acc += r[(p, i)] * a[i][j][k][l];
                        }
                        *v = acc;
                    }
                }
            }
        }
        // Cycle the axes so the next pass contracts the following index.
        let mut rotated = [[[[0.0; 3]; 3]; 3]; 3];
        for (i, ri) in rotated.iter_mut().enumerate() {
            for (j, rij) in ri.iter_mut().enumerate() {
                for (k, rijk) in rij.iter_mut().enumerate() {
                    for (l, v) in rijk.iter_mut().enumerate() {
                        *v = b[j][k][l][i];
                    }
                }
            }
        }
        a = rotated;
    }
    a
}

/// Rotate a 6×6 engineering-Voigt matrix from material to global axes.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// # use nalgebra::Matrix3;
/// // Une rotation identité ne change rien…
/// let d = symmetry::orthotropic_from_constants(
///     [210e3, 10e3, 10e3], [0.3, 0.3, 0.3], [5e3, 5e3, 5e3])?;
/// assert_eq!(symmetry::rotate_voigt(&d, &Matrix3::identity()), d);
/// // …un quart de tour échange les deux premières directions.
/// let r = symmetry::frame_rotation(&mat, 0, 2)?;
/// let dr = symmetry::rotate_voigt(&d, &r);
/// assert!((dr[0][0] - d[1][1]).abs() < 1e-6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn rotate_voigt(d: &[[f64; 6]; 6], r: &Matrix3<f64>) -> [[f64; 6]; 6] {
    from_tensor(&rotate_tensor(&to_tensor(d), r))
}

// ─── Elasticity ─────────────────────────────────────────────────────────────

/// Orthotropic elastic constants, in the material axes.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Neuf constantes dans les axes matériau : trois modules, trois
/// // coefficients de Poisson, trois modules de cisaillement.
/// assert_eq!(symmetry::ORTHOTROPIC_ELASTIC[0], "E_1");
/// assert_eq!(symmetry::ORTHOTROPIC_ELASTIC.len(), 9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const ORTHOTROPIC_ELASTIC: [&str; 9] = [
    "E_1", "E_2", "E_3", "nu_12", "nu_13", "nu_23", "G_12", "G_13", "G_23",
];

/// The 21 independent constants of a general anisotropic stiffness, named
/// `C_<i><j>` over the upper triangle of the Voigt matrix (`1..=6`, this crate's
/// order `[xx, yy, zz, yz, xz, xy]`).
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Les 21 constantes indépendantes, sur le triangle supérieur de Voigt.
/// assert_eq!(symmetry::ANISOTROPIC_ELASTIC[0], "C_11");
/// assert_eq!(symmetry::ANISOTROPIC_ELASTIC.len(), 21);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const ANISOTROPIC_ELASTIC: [&str; 21] = [
    "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_22", "C_23", "C_24", "C_25", "C_26", "C_33",
    "C_34", "C_35", "C_36", "C_44", "C_45", "C_46", "C_55", "C_56", "C_66",
];

/// Full 3-D orthotropic stiffness in the material axes: invert the normal
/// compliance block, and put the three shear moduli on the diagonal.
///
/// The compliance uses the reciprocity relations `ν_ji/E_j = ν_ij/E_i`, so only
/// the three « major » Poisson ratios are asked for. Errors if the resulting
/// compliance is singular — which is what an inconsistent set of ratios
/// (`ν_12² > E_1/E_2`, …) produces, and which would otherwise assemble a
/// non-positive stiffness in silence.
fn orthotropic_stiffness(material: &SubElementField, cell: usize) -> Result<[[f64; 6]; 6]> {
    let v = |name: &str| material.value(cell, 0, name);
    orthotropic_from_constants(
        [v("E_1")?, v("E_2")?, v("E_3")?],
        [v("nu_12")?, v("nu_13")?, v("nu_23")?],
        [v("G_12")?, v("G_13")?, v("G_23")?],
    )
}

/// The same, from bare constants — the arithmetic core, so it can be exercised
/// without a material field. `nu` is `[ν_12, ν_13, ν_23]`, `g` is `[G_12, G_13, G_23]`.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Le cœur arithmétique, exerçable sans champ matériau. Un matériau
/// // orthotrope dégénéré en isotrope redonne la matrice isotrope.
/// let d = symmetry::orthotropic_from_constants(
///     [210e3, 210e3, 210e3], [0.3, 0.3, 0.3], [80769.0, 80769.0, 80769.0])?;
/// assert!((d[0][0] - d[1][1]).abs() < 1e-6);
/// assert!((d[0][1] - d[0][2]).abs() < 1e-6);
/// // Un module nul ou négatif est refusé ; la souplesse doit par ailleurs
/// // rester inversible.
/// assert!(symmetry::orthotropic_from_constants(
///     [0.0, 210e3, 210e3], [0.3, 0.3, 0.3], [80769.0, 80769.0, 80769.0]).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn orthotropic_from_constants(e: [f64; 3], nu: [f64; 3], g: [f64; 3]) -> Result<[[f64; 6]; 6]> {
    let ([e1, e2, e3], [nu12, nu13, nu23], [g12, g13, g23]) = (e, nu, g);
    for (name, value) in [("E_1", e1), ("E_2", e2), ("E_3", e3)] {
        if value <= 0.0 {
            return Err(PyrucastError::Message(format!(
                "orthotropic elasticity: {name} = {value} — the Young moduli must be positive"
            )));
        }
    }
    let compliance = Matrix3::new(
        1.0 / e1,
        -nu12 / e1,
        -nu13 / e1,
        -nu12 / e1,
        1.0 / e2,
        -nu23 / e2,
        -nu13 / e1,
        -nu23 / e2,
        1.0 / e3,
    );
    let normal = compliance.try_inverse().ok_or_else(|| {
        PyrucastError::Message(
            "orthotropic elasticity: the compliance is singular — check the Poisson ratios \
             (a physical set satisfies nu_ij² < E_i/E_j)"
                .into(),
        )
    })?;
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = normal[(i, j)];
        }
    }
    // Voigt order [xx, yy, zz, yz, xz, xy] ⇒ slot 3 is `yz` (G_23), 4 is `xz`
    // (G_13), 5 is `xy` (G_12).
    d[3][3] = g23;
    d[4][4] = g13;
    d[5][5] = g12;
    Ok(d)
}

/// Full 3-D anisotropic stiffness in the material axes, read straight from the
/// 21 upper-triangle components and mirrored.
fn anisotropic_stiffness(material: &SubElementField, cell: usize) -> Result<[[f64; 6]; 6]> {
    let mut d = [[0.0; 6]; 6];
    let mut names = ANISOTROPIC_ELASTIC.iter();
    for i in 0..6 {
        for j in i..6 {
            let name = names.next().expect("21 names cover the upper triangle");
            let value = material.value(cell, 0, name)?;
            d[i][j] = value;
            d[j][i] = value;
        }
    }
    Ok(d)
}

/// The constitutive (Voigt) matrix of a cell, in **global** axes and reduced to
/// `model` — the single entry point every mechanical kernel uses.
///
/// Isotropy short-circuits to [`elasticity::constitutive`], keeping its exact
/// closed forms (and therefore the exact numbers of every assembly that predates
/// this module). The other two build the full 3-D stiffness in the material
/// axes, rotate it by the frame, then reduce.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// # let iso = SubElementField::from_uniform_per_component(
/// #     fes.get(0)?, vec!["E".into(), "nu".into()], &[210e3, 0.3])?;
/// // L'unique porte d'entrée des noyaux mécaniques. L'isotropie
/// // court-circuite vers les formes closes, donc rien ne bouge pour elle.
/// let d = symmetry::elastic_constitutive(
///     &iso, 0, MaterialSymmetry::Isotropic, ElasticityModel::PlaneStress, 2)?;
/// assert_eq!(d.len(), 3);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elastic_constitutive(
    material: &SubElementField,
    cell: usize,
    symmetry: MaterialSymmetry,
    model: ElasticityModel,
    space_dim: usize,
) -> Result<Vec<Vec<f64>>> {
    if symmetry == MaterialSymmetry::Isotropic {
        return Ok(elasticity::constitutive(
            material.value(cell, 0, "E")?,
            material.value(cell, 0, "nu")?,
            model,
            space_dim,
        ));
    }
    let d_mat = match symmetry {
        MaterialSymmetry::Orthotropic => orthotropic_stiffness(material, cell)?,
        MaterialSymmetry::Anisotropic => anisotropic_stiffness(material, cell)?,
        MaterialSymmetry::Isotropic => unreachable!("handled above"),
    };
    let r = frame_rotation(material, cell, space_dim)?;
    Ok(reduce_to_model(&rotate_voigt(&d_mat, &r), model))
}

/// Reduce a full-3-D engineering-Voigt matrix to the model's `v×v` matrix: the
/// `[xx, yy, xy]` block for plane strain, its **static condensation** on `ε_zz`
/// (so `σ_zz = 0`) for plane stress, the `[rr, zz, θθ, rz]` block for
/// axisymmetric, the full `6×6` for the solid.
///
/// Shared by the anisotropic constitutive path and by the consistent tangent of
/// the non-linear laws — the reduction is a property of the *kinematics*, not of
/// the constitutive law that produced the 6×6.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // La réduction tient à la **cinématique**, pas à la loi qui a produit
/// // la 6×6 — d'où son partage avec les tangentes non linéaires.
/// let d = symmetry::orthotropic_from_constants(
///     [210e3, 210e3, 210e3], [0.3, 0.3, 0.3], [80769.0, 80769.0, 80769.0])?;
/// assert_eq!(symmetry::reduce_to_model(&d, ElasticityModel::PlaneStrain).len(), 3);
/// assert_eq!(symmetry::reduce_to_model(&d, ElasticityModel::Solid).len(), 6);
/// // Contraintes planes : condensation statique sur ε_zz, donc σ_zz = 0 —
/// // le terme (0,0) y est plus **petit** qu'en déformations planes.
/// let cp = symmetry::reduce_to_model(&d, ElasticityModel::PlaneStress);
/// let dp = symmetry::reduce_to_model(&d, ElasticityModel::PlaneStrain);
/// assert!(cp[0][0] < dp[0][0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn reduce_to_model(d3: &[[f64; 6]; 6], model: ElasticityModel) -> Vec<Vec<f64>> {
    /// Where each axisymmetric Voigt slot `[rr, zz, θθ, rz]` sits in the 3-D order.
    const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];
    /// The in-plane slots `[xx, yy, xy]`.
    const PLANE: [usize; 3] = [0, 1, 5];
    match model {
        // Axisymmetric: the plain [rr, zz, θθ, rz] sub-block. No condensation —
        // all four strains are prescribed (the hoop is measured, not assumed).
        ElasticityModel::Axisymmetric => AXI_TO_3D
            .iter()
            .map(|&i| AXI_TO_3D.iter().map(|&j| d3[i][j]).collect())
            .collect(),
        ElasticityModel::Solid => d3.iter().map(|r| r.to_vec()).collect(),
        ElasticityModel::PlaneStrain => PLANE
            .iter()
            .map(|&i| PLANE.iter().map(|&j| d3[i][j]).collect())
            .collect(),
        ElasticityModel::PlaneStress => {
            // Condense the out-of-plane normal `zz` (index 2) so σ_zz = 0:
            // D2[i][j] = D3[i][j] − D3[i][2]·D3[2][j]/D3[2][2].
            let z = 2usize;
            let dzz = d3[z][z];
            let cond = |i: usize, j: usize| d3[i][j] - d3[i][z] * d3[z][j] / dzz;
            PLANE
                .iter()
                .map(|&i| PLANE.iter().map(|&j| cond(i, j)).collect())
                .collect()
        }
    }
}

// ─── Scalar transport (conduction, diffusion) ───────────────────────────────

/// Orthotropic conductivities / diffusivities, in the material axes. The prefix
/// is the physics' own (`k` for heat, `D` for Fick).
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Le préfixe est celui de la physique : `k` pour la chaleur, `D` pour Fick.
/// assert_eq!(symmetry::orthotropic_scalar("k"),
///            ["k_1".to_string(), "k_2".to_string(), "k_3".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn orthotropic_scalar(prefix: &str) -> [String; 3] {
    [
        format!("{prefix}_1"),
        format!("{prefix}_2"),
        format!("{prefix}_3"),
    ]
}

/// The six independent components of a symmetric anisotropic conductivity /
/// diffusivity tensor, upper triangle.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Les six composantes du triangle supérieur d'un tenseur symétrique.
/// assert_eq!(symmetry::anisotropic_scalar("D")[0], "D_11");
/// assert_eq!(symmetry::anisotropic_scalar("D").len(), 6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn anisotropic_scalar(prefix: &str) -> [String; 6] {
    [
        format!("{prefix}_11"),
        format!("{prefix}_12"),
        format!("{prefix}_13"),
        format!("{prefix}_22"),
        format!("{prefix}_23"),
        format!("{prefix}_33"),
    ]
}

/// The conductivity / diffusivity **tensor** of a cell in global axes, as a
/// `space_dim × space_dim` matrix (the leading block of the 3-D tensor).
///
/// `prefix` names the physics' coefficient (`"k"`, `"D"`). Isotropy reads the
/// bare `prefix` component **at the Gauss point** `g` — conduction has always
/// allowed a conductivity varying inside a cell, and that is preserved;
/// orthotropy and anisotropy read their constants per cell, like the mechanical
/// moduli.
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
/// # use pyrucast::models::elasticity::ElasticityModel;
/// # use pyrucast::models::symmetry::{self, MaterialSymmetry};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let mut mat = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["k_1".into(), "k_2".into(), "k_3".into(), "V1X".into(), "V1Y".into()],
/// #     &[9.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
/// // Orthotrope, axes tournés d'un quart de tour : la conductivité forte
/// // (k_1 = 9) se retrouve **sur y**, pas sur x.
/// let k = symmetry::transport_tensor(
///     &mat, 0, 0, MaterialSymmetry::Orthotropic, 2, "k")?;
/// assert!((k[(1, 1)] - 9.0).abs() < 1e-12);
/// assert!((k[(0, 0)] - 1.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn transport_tensor(
    material: &SubElementField,
    cell: usize,
    g: usize,
    symmetry: MaterialSymmetry,
    space_dim: usize,
    prefix: &str,
) -> Result<Matrix3<f64>> {
    let k3 = match symmetry {
        MaterialSymmetry::Isotropic => {
            let k = material.value(cell, g, prefix)?;
            Matrix3::from_diagonal_element(k)
        }
        MaterialSymmetry::Orthotropic => {
            let n = orthotropic_scalar(prefix);
            let d = Matrix3::from_diagonal(&Vector3::new(
                material.value(cell, 0, &n[0])?,
                material.value(cell, 0, &n[1])?,
                material.value(cell, 0, &n[2])?,
            ));
            let r = frame_rotation(material, cell, space_dim)?;
            r * d * r.transpose()
        }
        MaterialSymmetry::Anisotropic => {
            let n = anisotropic_scalar(prefix);
            let v = |i: usize| material.value(cell, 0, &n[i]);
            let (k11, k12, k13) = (v(0)?, v(1)?, v(2)?);
            let (k22, k23, k33) = (v(3)?, v(4)?, v(5)?);
            let d = Matrix3::new(k11, k12, k13, k12, k22, k23, k13, k23, k33);
            let r = frame_rotation(material, cell, space_dim)?;
            r * d * r.transpose()
        }
    };
    Ok(k3)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A rotation about `z` by `theta`, as the frame vectors would give it.
    fn rot_z(theta: f64) -> Matrix3<f64> {
        let (c, s) = (theta.cos(), theta.sin());
        Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    fn isotropic_6x6(e: f64, nu: f64) -> [[f64; 6]; 6] {
        let d = elasticity::constitutive(e, nu, ElasticityModel::Solid, 3);
        let mut out = [[0.0; 6]; 6];
        for (i, row) in d.iter().enumerate() {
            out[i].copy_from_slice(row);
        }
        out
    }

    #[test]
    fn tags_round_trip() {
        for s in [
            MaterialSymmetry::Isotropic,
            MaterialSymmetry::Orthotropic,
            MaterialSymmetry::Anisotropic,
        ] {
            assert_eq!(MaterialSymmetry::from_tag(s.to_tag()), Some(s));
        }
        assert_eq!(MaterialSymmetry::from_tag("cubic"), None);
    }

    #[test]
    fn voigt_tensor_round_trip_is_the_identity() {
        // An arbitrary symmetric 6×6 must survive D → C_ijkl → D untouched.
        let mut d = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in i..6 {
                let v = (i * 6 + j) as f64 + 1.0;
                d[i][j] = v;
                d[j][i] = v;
            }
        }
        let back = from_tensor(&to_tensor(&d));
        for i in 0..6 {
            for j in 0..6 {
                assert!((back[i][j] - d[i][j]).abs() < 1e-12, "slot ({i},{j})");
            }
        }
    }

    #[test]
    fn rotating_an_isotropic_stiffness_changes_nothing() {
        // Isotropy is invariant under every rotation — the sharpest available
        // check that the fourth-order rotation carries no stray index or factor.
        let d = isotropic_6x6(210e9, 0.3);
        for theta in [0.1, 0.7, 1.3, 2.9] {
            let r = rot_z(theta);
            let rotated = rotate_voigt(&d, &r);
            for i in 0..6 {
                for j in 0..6 {
                    assert!(
                        (rotated[i][j] - d[i][j]).abs() < 1e-3,
                        "theta = {theta}, slot ({i},{j}): {} vs {}",
                        rotated[i][j],
                        d[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn rotating_by_a_quarter_turn_swaps_the_material_axes() {
        // A 90° turn about `z` sends axis 1 onto axis 2: the rotated stiffness
        // must be the one of a material whose first two axes are exchanged.
        let mut d = [[0.0; 6]; 6];
        d[0][0] = 3.0;
        d[1][1] = 5.0;
        d[2][2] = 7.0;
        d[3][3] = 11.0;
        d[4][4] = 13.0;
        d[5][5] = 17.0;
        let rotated = rotate_voigt(&d, &rot_z(std::f64::consts::FRAC_PI_2));
        assert!((rotated[0][0] - 5.0).abs() < 1e-9);
        assert!((rotated[1][1] - 3.0).abs() < 1e-9);
        assert!((rotated[2][2] - 7.0).abs() < 1e-9);
        // `xy` shear is unchanged by a turn about `z`; `yz` and `xz` swap.
        assert!((rotated[5][5] - 17.0).abs() < 1e-9);
        assert!((rotated[3][3] - 13.0).abs() < 1e-9);
        assert!((rotated[4][4] - 11.0).abs() < 1e-9);
    }

    #[test]
    fn orthotropy_with_equal_constants_is_isotropy() {
        // Feeding the orthotropic law the isotropic constants must reproduce the
        // isotropic stiffness exactly — the two code paths meet.
        let (e, nu) = (210e9, 0.3);
        let g = e / (2.0 * (1.0 + nu));
        let d = orthotropic_from_constants([e, e, e], [nu, nu, nu], [g, g, g])
            .expect("isotropic constants are admissible");
        let iso = isotropic_6x6(e, nu);
        for i in 0..6 {
            for j in 0..6 {
                assert!((d[i][j] - iso[i][j]).abs() < 1e-3, "slot ({i},{j})");
            }
        }
    }

    #[test]
    fn plane_stress_reduction_kills_the_out_of_plane_stress() {
        // Condensing the solid stiffness on ε_zz must give back the closed-form
        // plane-stress matrix.
        let (e, nu) = (210e9, 0.3);
        let reduced = reduce_to_model(&isotropic_6x6(e, nu), ElasticityModel::PlaneStress);
        let expect = elasticity::constitutive(e, nu, ElasticityModel::PlaneStress, 2);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reduced[i][j] - expect[i][j]).abs() < 1e-3,
                    "slot ({i},{j}): {} vs {}",
                    reduced[i][j],
                    expect[i][j]
                );
            }
        }
    }

    #[test]
    fn frame_components_follow_the_symmetry_and_the_dimension() {
        assert!(frame_components(MaterialSymmetry::Isotropic, 3).is_empty());
        assert_eq!(
            frame_components(MaterialSymmetry::Orthotropic, 2),
            &FRAME_2D
        );
        assert_eq!(
            frame_components(MaterialSymmetry::Anisotropic, 3),
            &FRAME_3D
        );
    }
}
