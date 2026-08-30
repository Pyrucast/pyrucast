# Archive — pression suiveuse (`follower_pressure`)

**Retirée du code le 2026-08-30**, sur la branche `noyaux-paralleles`.

## Pourquoi

`follower_pressure` était la seule physique à ne produire **aucune** matrice :
son `contributions` renvoyait `Vec::new()` et son `stiffness_layout` déclarait
`symmetric: false` uniquement pour que `build_internal_forces` y trouve son
support et ses fespaces. Deux mensonges de modélisation — elle disait « j'ai un
bloc de raideur » pour obtenir « la géométrie sur laquelle j'intègre mes forces ».

Le trait `Domain` classe les sous-modèles par **le type de matrice qu'ils
produisent**. Garder la pression suiveuse aurait obligé à scinder `Domain` en
deux traits pour ce seul membre. Elle sort donc, le temps de lui rendre une place
juste — vraisemblablement une capacité « charge » distincte de `Domain`, avec sa
propre déclaration de géométrie d'intégration plutôt qu'un `stiffness_layout`
emprunté.

Rien n'était cassé : les tests passaient, la physique était juste. C'est son
rangement dans la hiérarchie des traits qui ne l'était pas.

## Ce qu'il faudra refaire pour la remettre

Chaque point ci-dessous a été défait ; les remettre, dans cet ordre, restaure la
physique. Les numéros de ligne sont ceux d'avant le retrait, donnés comme repère.

### Code Rust

| Fichier | Ce qu'il y avait |
|---|---|
| `src/models/follower_pressure.rs` | le module entier (ci-dessous) |
| `src/models/mod.rs:53` | `pub mod follower_pressure;` |
| `src/lib.rs:124` | `models::follower_pressure::follower_pressure_py::follower_pressure,` dans l'enregistrement pyo3 |
| `src/ops/model/mod.rs:75` | `pub use crate::models::follower_pressure::follower_pressure;` |
| `src/ops/model/mod.rs:12` | la mention dans l'inventaire en tête de module |
| `src/containers/model.rs:133` | `follower_pressure` dans la liste d'imports |
| `src/containers/model.rs:250` | la variante `SubModel::FollowerPressure(follower_pressure::FollowerPressure)` |
| `src/containers/model.rs:309` | le bras `SubModel::FollowerPressure(p) => p,` de `as_kind` |
| `src/containers/model.rs:753-788` | le constructeur `SubModel::follower_pressure` et son doctest |
| `tests/follower_pressure.rs` | le test d'intégration (ci-dessous) |

`CellGeom::tangents` et `CellGeom::normal_from_tangents`
(`src/models/kernel.rs`) ont été **conservées** : ce sont leurs seuls appelants
hors de `kernel.rs` qui disparaissaient, et la pression suiveuse en a besoin au
retour. Leur documentation invoque toujours « a follower load » comme motivation.

### Python

| Fichier | Ce qu'il y avait |
|---|---|
| `python/pyrucast/model.py:33` | `follower_pressure as follower_pressure,` |
| `python/pyrucast/model.py:64` | `"follower_pressure",` dans `__all__` |
| `python/pyrucast/_pyrucast/__init__.pyi:63,3242` | l'entrée d'`__all__` et le stub de la fonction |
| `tests/python/test_doc_mecanique.py:258-267` | la section `pression_suiveuse` (ci-dessous) |

### Book et tables de correspondance

| Fichier | Ce qu'il y avait |
|---|---|
| `book/src/SUMMARY.md:34` | `    - [Pression suiveuse](mecanique/pression-suiveuse.md)` |
| `book/src/mecanique.md:18` | l'entrée de la liste des pages mécanique |
| `book/src/mecanique/pression-suiveuse.md` | la page entière (ci-dessous) |
| `book/src/physiques.md:48,90,117,175` | l'entrée de la liste, la ligne du tableau des physiques et deux mentions en prose |
| `book/src/model.md:115,153` | la ligne du tableau des opérateurs et l'entrée de la liste `Mechanical` |
| `book/src/correspondance-rust-python.md:244` | la ligne de correspondance |
| `book/src/operateurs.md:18` | la mention dans l'énumération `ops::model` |
| `modèle_castem.csv:256` | la colonne « Équivalent pyrucast » de `CHARGEMENT;PRESSION`, ramenée à `—` |

---

## `src/models/follower_pressure.rs`

```rust
//! Follower pressure — a load that turns with the surface it acts on.
//!
//! A pressure is always normal to the surface it presses. As the body deforms,
//! that surface **moves and tilts**, so the load direction moves with it: a
//! pressure is a *follower* load. Ignoring this is exact only in small
//! displacements; on an inflating membrane, a buckling shell or a rotating blade
//! it is not.
//!
//! ```text
//! t = −p · n(u)          n(u) the normal of the **deformed** surface
//! ```
//!
//! ## Why it is a model and not a load
//!
//! A dead load is built once with
//! [`flux`](fn@crate::ops::node_field::flux) and never looked at again. A follower
//! pressure cannot be: its direction depends on the current displacement, so it
//! has to be **recomputed at every residual evaluation**. That is precisely what
//! a physics does — it integrates a behaviour and contributes to the internal
//! forces — so it is one:
//!
//! ```text
//! u  ──gradient──▶  ∇_s u  ──integrate_behavior──▶  t(u)  ──internal_forces──▶  f(u)
//! ```
//!
//! The behaviour integration is where the direction is refreshed. Nothing else
//! in the pipeline changes.
//!
//! ## The deformed normal, from the deformed tangents
//!
//! The direction **and** the area change both come from the tangents of the
//! surface. If `a_k = ∂x/∂ξ_k` are the reference tangents, the deformed ones are
//!
//! ```text
//! ā_k = a_k + ∂u/∂ξ_k = a_k + (∇_s u)·a_k
//! ```
//!
//! and the normal times the area ratio is their cross product (their −90° turn
//! in 2-D) divided by the reference one:
//!
//! ```text
//! t = −p · (ā₁ × ā₂) / |a₁ × a₂|         (3-D)
//! t = −p · (ā_y, −ā_x) / |a|             (2-D)
//! ```
//!
//! Keeping the traction **referential** is what lets the internal-force integral
//! use the ordinary reference measure: the formulation stays total-Lagrangian,
//! and with no displacement it gives back `t = −p·N` exactly.
//!
//! ### Why not Nanson
//!
//! `n da = det(F)·F⁻ᵀ·N dA` is the textbook route, and it is the wrong one
//! **here**. On a manifold the tangential gradient has no component along the
//! normal, so `I + ∇_s u` is not a deformation gradient: a quarter-turn of the
//! surface sends its determinant to zero and the formula blows up on a
//! perfectly ordinary rotation. The tangents never degenerate that way — they
//! rotate with the surface — so they are what a surface load must be built on.
//!
//! ## Orientation is the mesh's business
//!
//! The normal follows the boundary mesh's **winding**
//! ([`CellGeom::normal`](crate::models::kernel::CellGeom::normal)). A positive
//! `p` pushes **against** it — compressive — so an outward-wound boundary gives
//! the usual sign. This is the one place where a boundary mesh's orientation
//! matters: contrast [`boundary_transfer`](crate::models::boundary_transfer), whose direction
//! is already consumed in writing `q·n` and which is therefore
//! orientation-blind.
//!
//! ## What it contributes
//!
//! Internal forces only. It declares a `stiffness_layout` — that is what the
//! internal-force scatter is driven from — but **contributes no matrix**: its
//! `contributions()` is empty for every kind. The load-correction stiffness
//! `∂f/∂u` (non-symmetric) is not implemented; a Newton loop converges without
//! it, more slowly.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::owned_components;
use crate::models::tensor::{dual_name, primal_name};
use crate::models::ZoneLayout;
use crate::models::{
    CellGeom, Contribution, Domain, MatrixKind, MatrixLayout, Physics, SubModelKind,
};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Required material component: the pressure.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::follower_pressure;
/// // La pression appliquée, fournie au moment de l'assemblage.
/// assert_eq!(follower_pressure::MATERIAL_COMPONENT, "p");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL_COMPONENT: &str = "p";
/// Material contract returned by [`Domain::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Behaviour-**output** components: the referential traction.
fn traction_names(space_dim: usize) -> Vec<String> {
    (0..space_dim).map(|a| format!("t_{}", AXES[a])).collect()
}

/// Follower pressure on a boundary FE subspace.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::follower_pressure::{self, FollowerPressure};
/// // Une pression qui tourne avec la surface : une seule constante.
/// let f = FollowerPressure::new(zone.clone())?;
/// assert_eq!(f.material_components(),
///            vec![follower_pressure::MATERIAL_COMPONENT.to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct FollowerPressure {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the boundary's unique nodes.
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
}

impl FollowerPressure {
    /// Follower pressure on a **boundary** FE subspace — an edge mesh in 2-D, a
    /// surface mesh in 3-D. Errors on anything else: a pressure acts on a
    /// surface, and a cell that fills its space has no normal to follow.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::follower_pressure::{self, FollowerPressure};
    /// // Une pression qui tourne avec la surface : une seule constante.
    /// let f = FollowerPressure::new(zone.clone())?;
    /// assert_eq!(f.material_components(),
    ///            vec![follower_pressure::MATERIAL_COMPONENT.to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let (submesh, space_dim, ref_dim) = {
            let s = fespace.read();
            (s.submesh(), s.space_dim(), s.ref_dim()?)
        };
        if ref_dim + 1 != space_dim {
            return Err(PyrucastError::Message(format!(
                "FollowerPressure: a {ref_dim}-D element in a {space_dim}-D space is not a \
                 boundary — a pressure acts on a surface (SEG2 in 2-D, TRI3/QUA4 in 3-D), \
                 and needs a normal to follow"
            )));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
        })
    }
}

impl SubModelKind for FollowerPressure {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    /// Declared so the internal-force scatter knows which subspace and support
    /// to run on — **not** to contribute a matrix. See
    /// [`contributions`](Self::contributions).
    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: false,
        })
    }

    /// A follower pressure contributes **no matrix at all** — it is a load, and
    /// its whole effect lives in the internal forces. Overriding this (rather
    /// than dropping the layout) is what lets it keep a `stiffness_layout` for
    /// the internal-force scatter without an assembler ever asking it for an
    /// element matrix it does not have.
    fn contributions(
        &self,
        _kind: MatrixKind,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        Ok(Vec::new())
    }

    /// The consistent nodal load `f_{i,a} = ∫_Γ N_i · t_a dΓ`, integrated on the
    /// **reference** surface — the traction already carries the area change.
    fn internal_force_reads(&self) -> Vec<String> {
        traction_names(self.space_dim)
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let d = self.space_dim;
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g);
            let w = geom.det_j_w(g);
            for i in 0..geom.n_nodes {
                for (a, &comp) in lay.iter().enumerate() {
                    fe[i * d + a] += shape[i] * stress.get(geom.cell, g, comp as usize)? * w;
                }
            }
        }
        Ok(())
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "FollowerPressure"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<FollowerPressure>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

impl Domain for FollowerPressure {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Vec<String> {
        owned_components(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        traction_names(self.space_dim)
    }

    /// The referential traction at one Gauss point, from the deformed tangents.
    ///
    /// This is where the direction is refreshed: call it again with an updated
    /// displacement and the load has turned with the surface.
    fn deformation_reads(&self) -> Vec<String> {
        let d = self.space_dim;
        let mut names = Vec::with_capacity(d * d);
        for a in 0..d {
            for b in 0..d {
                names.push(format!("grad_u_{}_{}", AXES[a], AXES[b]));
            }
        }
        names
    }

    fn integrate_point(
        &self,
        geom: &CellGeom,
        g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        let (cell, d) = (geom.cell, self.space_dim);
        let p = material[lay.material[0] as usize];

        // ∇_s u, the tangential gradient of the displacement, row-major over
        // `deformation_reads`' own `(a, b)` order.
        let mut grad = [0.0_f64; 9];
        for k in 0..d * d {
            grad[k] = deformation[lay.deformation[k] as usize];
        }

        // The deformed tangents ā_k = a_k + (∇_s u)·a_k, and the reference
        // measure |a₁ × a₂| that turns the result into a *referential* traction.
        // Both live on the stack: a surface cell has at most two tangents in 3-D.
        let n_tan = d - 1;
        let mut reference = [0.0_f64; 6];
        geom.tangents(g, &mut reference[..n_tan * d]);
        let mut deformed = [0.0_f64; 6];
        for k in 0..n_tan {
            for i in 0..d {
                deformed[k * d + i] = reference[k * d + i]
                    + (0..d)
                        .map(|j| grad[i * d + j] * reference[k * d + j])
                        .sum::<f64>();
            }
        }

        let mut n_ref = [0.0_f64; 3];
        CellGeom::normal_from_tangents(&reference, n_tan, d, &mut n_ref)?;
        let area_ref = n_ref[..d].iter().map(|v| v * v).sum::<f64>().sqrt();
        if area_ref <= f64::EPSILON {
            return Err(PyrucastError::Message(format!(
                "FollowerPressure: cell {cell} is degenerate at Gauss point {g} (null area)"
            )));
        }
        let mut n_def = [0.0_f64; 3];
        CellGeom::normal_from_tangents(&deformed, n_tan, d, &mut n_def)?;

        // The pressure pushes **against** the normal, hence the minus sign; the
        // magnitude of `n_def` already carries the area change.
        for (a, o) in out.iter_mut().enumerate().take(d) {
            *o = -p * n_def[a] / area_ref;
        }
        Ok(())
    }
}

crate::physics_operator! {
    /// Follower-pressure `Model` spanning **every** subspace of a *boundary*
    /// `fes`. Parent-level operator; `p` is supplied at assembly time.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::models::{Physics, RelationSense};
    /// # use pyrucast::ops::mesh;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let mut bord = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # bord.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let fes_bord = FiniteElementSpace::lagrange1(&Mesh::from_submesh(bord))?;
    /// let m = model::follower_pressure(&fes_bord)?;
    /// assert_eq!(m.len(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn follower_pressure(fes) via SubModel::follower_pressure;
    python: "`model.follower_pressure(fespace)` — a pressure that **turns with the\nsurface** it acts on, on a *boundary* `fespace` (an edge mesh in 2-D, a\nsurface mesh in 3-D). Material: `p`, the pressure.\n\nUnlike a dead load built once with `flux(...)`, its direction depends on\nthe current displacement, so it is recomputed at each residual\nevaluation:\n\n```text\nu → element_field.gradient → integrate_behavior → node_field.internal_forces\n```\n\nIt contributes **no matrix** — only internal forces. A positive `p`\npushes *against* the boundary mesh's own normal, which follows its\nwinding: orienting the boundary outwards gives the usual compressive\nsign."
}
```

## `tests/follower_pressure.rs`

```rust
//! Follower pressure — a load that turns with the surface.
//!
//! The whole claim of a follower load is that its **direction depends on the
//! displacement**. So the tests do not check a value against a formula once;
//! they check that the load *moves* the way the surface does, and that it
//! degenerates to the dead load when the surface does not move.
//!
//! The test surface is the `x = 1` edge of the unit square — a unit-length
//! SEG2 wound so that its normal points along `+x`, i.e. outwards. A positive
//! pressure therefore pushes along `−x`.
//!
//! Three regimes pin the law down:
//!
//! | displacement | expected load |
//! |---|---|
//! | none | `(−p, 0)` — the dead load exactly |
//! | rigid rotation by `θ` | `(−p cosθ, −p sinθ)` — same magnitude, turned by `θ` |
//! | uniform stretch `λ` along the edge | `(−pλ, 0)` — the deformed **area** has grown |
//!
//! The second is the one a non-follower load gets wrong, and the third is the
//! one a follower load that only rotates its direction, forgetting the area
//! change, gets wrong.
//!
//! Single source for the « pression suiveuse » example of the mechanics book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::element_field;
use pyrucast::ops::model;
use pyrucast::Result;

const P: f64 = 3.0; // pressure

#[test]
fn an_undeformed_surface_gives_back_the_dead_load() -> Result<()> {
    let edge = Edge::new()?;
    let (fx, fy) = edge.load(|_x, _y| (0.0, 0.0))?;
    // Outward normal along +x, pressure pushing against it.
    assert!((fx + P).abs() < 1e-12, "f_x = {fx}");
    assert!(fy.abs() < 1e-12, "f_y = {fy}");
    Ok(())
}

#[test]
fn the_load_turns_with_the_surface() -> Result<()> {
    let edge = Edge::new()?;
    for angle_deg in [10.0_f64, 45.0, 90.0, 170.0] {
        let a = angle_deg.to_radians();
        let (c, s) = (a.cos(), a.sin());
        // A rigid rotation of the whole plane: u = (R − I)·x.
        let (fx, fy) = edge.load(|x, y| (c * x - s * y - x, s * x + c * y - y))?;
        // Same magnitude, turned by exactly the same angle.
        assert!(
            (fx + P * c).abs() < 1e-10 && (fy + P * s).abs() < 1e-10,
            "{angle_deg}°: ({fx}, {fy}), expected ({}, {})",
            -P * c,
            -P * s
        );
        let magnitude = (fx * fx + fy * fy).sqrt();
        assert!((magnitude - P).abs() < 1e-10, "magnitude {magnitude}");
    }
    Ok(())
}
// ANCHOR_END: example

/// Stretching the edge grows the surface the pressure acts on, so the total load
/// grows with it. That factor rides on the **magnitude** of the deformed normal,
/// which a follower load that only rotates its direction would normalise away.
#[test]
fn stretching_the_surface_grows_the_load() -> Result<()> {
    let edge = Edge::new()?;
    for lambda in [0.5_f64, 1.0, 1.5, 2.0] {
        // Stretch along the edge (y), leaving x alone.
        let (fx, fy) = edge.load(|_x, y| (0.0, (lambda - 1.0) * y))?;
        assert!(
            (fx + P * lambda).abs() < 1e-10,
            "λ = {lambda}: f_x = {fx}, expected {}",
            -P * lambda
        );
        assert!(fy.abs() < 1e-12, "f_y = {fy}");
    }
    Ok(())
}

/// A pressure acts on a surface. Handing it a cell that fills its space is a
/// modelling error — there is no normal to follow — and is reported as one.
#[test]
fn a_volumetric_subspace_is_rejected() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        .iter()
        .map(|c| Node::create_in(coords.clone(), c))
        .collect::<Result<_>>()?;
    let mut square = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    square.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())?;
    let err = model::follower_pressure(&FiniteElementSpace::lagrange1(&square)?).unwrap_err();
    assert!(err.to_string().contains("not a boundary"), "{err}");
    Ok(())
}

/// Reversing the winding reverses the normal, hence the load. The orientation of
/// the boundary mesh is the user's statement of which side is « outside », and
/// nothing else can supply it.
#[test]
fn reversing_the_winding_reverses_the_load() -> Result<()> {
    let forward = Edge::new()?;
    let backward = Edge::reversed()?;
    let (fx, _) = forward.load(|_x, _y| (0.0, 0.0))?;
    let (bx, _) = backward.load(|_x, _y| (0.0, 0.0))?;
    assert!((fx + bx).abs() < 1e-12, "{fx} and {bx} must be opposite");
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The `x = 1` edge of the unit square, as a follower-pressure model.
struct Edge {
    nodes: Vec<Node>,
    fes: FiniteElementSpace,
    model: Model,
    materials: ElementField,
}

impl Edge {
    /// Wound bottom → top, so the normal points along `+x` (outwards).
    fn new() -> Result<Self> {
        Self::build(false)
    }

    /// Wound top → bottom: the same geometry, the opposite normal.
    fn reversed() -> Result<Self> {
        Self::build(true)
    }

    fn build(reversed: bool) -> Result<Self> {
        let coords = Handle::new(Coords::new(2)?);
        let bottom = Node::create_in(coords.clone(), &[1.0, 0.0])?;
        let top = Node::create_in(coords.clone(), &[1.0, 1.0])?;
        let nodes = if reversed {
            vec![top, bottom]
        } else {
            vec![bottom, top]
        };
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[nodes[0].id(), nodes[1].id()])?;
        let fes = FiniteElementSpace::lagrange1(&mesh)?;
        let model = model::follower_pressure(&fes)?;
        let materials = element_field::material_field(&model, &[("p", P)])?;
        Ok(Self {
            nodes,
            fes,
            model,
            materials,
        })
    }

    /// The **total** nodal load for a displacement field given as `u(x, y)`.
    ///
    /// This is the follower pipeline end to end: the displacement is
    /// differentiated on the surface, the behaviour turns that into a traction,
    /// and the internal forces integrate it. Calling it again with another
    /// displacement is what « the load follows » means.
    fn load(&self, u: impl Fn(f64, f64) -> (f64, f64)) -> Result<(f64, f64)> {
        let sm = Handle::new(SubMesh::poi1_from_nodes(&self.nodes)?);
        let mut field = SubNodeField::from_poi1(&sm, vec!["u_x".to_string(), "u_y".to_string()])?;
        for n in &self.nodes {
            let p = n.position()?;
            let (ux, uy) = u(p[0], p[1]);
            field.set_value(n.id(), "u_x", ux)?;
            field.set_value(n.id(), "u_y", uy)?;
        }
        let displacement = NodeField::from_sub(field);

        let gradient = element_field::gradient(&displacement, &self.fes)?;
        let traction = element_field::behavior::integrate(
            &self.model,
            &gradient,
            None,
            &self.materials,
            None,
        )?;
        let forces = pyrucast::ops::node_field::internal_forces(&traction, &self.model)?;
        let mut total = (0.0, 0.0);
        for n in &self.nodes {
            total.0 += forces.value(n.id(), "f_x")?;
            total.1 += forces.value(n.id(), "f_y")?;
        }
        Ok(total)
    }
}
```

## `tests/python/test_doc_mecanique.py` — section `pression_suiveuse`

```python
# ANCHOR: pression_suiveuse
bord = pyrucast.FiniteElementSpace(maillage_de_bord)
charge = pyrucast.model.follower_pressure(bord)
materials = pyrucast.element_field.material_field(charge, [("p", 1.0e5)])

# À chaque itération : la direction se recalcule depuis le déplacement courant.
gradient = pyrucast.element_field.gradient(u, bord)
traction = pyrucast.element_field.integrate_behavior(charge, gradient, materials)
f = pyrucast.node_field.internal_forces(traction, charge)
# ANCHOR_END: pression_suiveuse
```

## `book/src/mecanique/pression-suiveuse.md`

````markdown
# Pression suiveuse

## Introduction

Une pression est toujours **normale à la surface** sur laquelle elle s'exerce.
Quand le corps se déforme, cette surface bouge et bascule : la direction de la
charge bouge avec elle. C'est une charge **suiveuse**.

L'ignorer n'est exact qu'en petits déplacements. Sur une membrane qui se gonfle,
une coque qui flambe, une aube qui tourne, la différence n'est pas un détail :
c'est elle qui décide de la charge critique.

\\[
\mathbf t = -p\\,\mathbf n(u),
\\]

\\( \mathbf n(u) \\) étant la normale de la surface **déformée**.

Les degrés de liberté sont ceux de la mécanique — déplacement `u_x, u_y(, u_z)`,
force nodale `f_x, …` — et le modèle vit sur un maillage de **bord** : SEG2 en
2-D, TRI3/QUA4 en 3-D.

## Pourquoi c'est un modèle et pas un chargement

Une charge morte se construit une fois avec
[`flux`](../operateurs/champs.md) et ne se regarde plus. Une pression suiveuse ne
le peut pas : sa direction dépend du déplacement courant, donc elle doit être
**recalculée à chaque évaluation du résidu**. C'est exactement ce que fait une
physique — elle intègre un comportement et contribue aux forces internes — donc
c'en est une :

```text
u  ──gradient──▶  ∇_s u  ──integrate_behavior──▶  t(u)  ──internal_forces──▶  f(u)
```

C'est dans l'intégration du comportement que la direction se rafraîchit. Rien
d'autre dans la chaîne ne change, et la boucle de Newton reste pilotée depuis
Python comme les autres non-linéarités.

## Équations continues résolues

Le travail virtuel de la pression sur la configuration **déformée** :

\\[
\delta W = -\int_{\gamma} p\\, \mathbf{n}\cdot\delta\mathbf{u}\\; da
\\]

où \\(\gamma\\) et \\(\mathbf{n}\\) sont la surface et la normale *actuelles*.
Tout le travail consiste à ramener cette intégrale sur la configuration de
référence, ce qui demande à la fois la **rotation** de la normale et le
**changement d'aire**.

## Forme discrétisée — par les tangentes déformées

Les deux viennent des tangentes de la surface. Si \\(a_k = \partial x/\partial
\xi_k\\) sont les tangentes de référence, les tangentes déformées sont

\\[
\bar{a}_k = a_k + \frac{\partial u}{\partial \xi_k} = a_k + (\nabla_s u)\cdot a_k
\\]

et la normale multipliée par le rapport d'aires est leur produit vectoriel (leur
rotation de −90° en 2-D), divisé par celui de référence :

\\[
\mathbf t = -p\\;\frac{\bar a_1 \times \bar a_2}{\lVert a_1 \times a_2 \rVert}
\quad \text{(3-D)},
\qquad
\mathbf t = -p\\;\frac{(\bar a_y,\\; -\bar a_x)}{\lVert a \rVert}
\quad \text{(2-D)},
\\]

les forces s'en déduisant par la mesure de **référence**, comme n'importe quelle
force interne :

\\[
f_i = \int_{\Gamma_0} N_i\\,\mathbf t\\; d\Gamma_0 .
\\]

Garder la traction **référentielle** est ce qui permet à l'intégrale des forces
internes d'utiliser la mesure de référence habituelle : la formulation reste
totalement lagrangienne, et sans déplacement elle redonne exactement `t = −p·N`.

### Pourquoi pas Nanson

\\(n\\,da = \det(F)\\,F^{-T}N\\,dA\\) est la route classique, et c'est la
**mauvaise** ici. Sur une variété, le gradient tangentiel n'a aucune composante
selon la normale : \\(I + \nabla_s u\\) n'est donc pas un gradient de
transformation. Un quart de tour de la surface envoie son déterminant à zéro et
la formule explose sur une rotation parfaitement ordinaire. Les tangentes, elles,
ne dégénèrent jamais ainsi — elles tournent avec la surface. C'est sur elles
qu'une charge surfacique doit être bâtie.

C'est le genre d'écueil qu'un test attrape et qu'une relecture laisse passer :
la rotation à 90° est dans la suite de tests pour cette raison.

## Variables et matériau

| | |
|---|---|
| primales | `u_x, u_y(, u_z)` |
| duales | `f_x, f_y(, f_z)` |
| matériau | `p` (la pression) |
| entrée du comportement | `grad_u_x_x`, … (le gradient surfacique de `u`) |
| sortie du comportement | `t_x, t_y(, t_z)` (la traction référentielle) |
| nature | `Mechanical` |

### L'orientation est l'affaire du maillage

La normale suit le **sens de parcours** du maillage de bord. Un `p` positif
pousse *contre* elle — compression — donc un bord orienté vers l'extérieur donne
le signe habituel.

C'est le seul endroit où l'orientation d'un maillage de bord compte. Par
contraste, la [convection](../thermique.md#convection-de-surface-robin--film) et
le [rayonnement](../thermique.md#rayonnement-à-linfini-stefan-boltzmann) y sont
aveugles : leur direction est déjà consommée en écrivant `q·n`, et la mesure
`det_j_w` est une magnitude invariante.

## Ce qu'elle contribue

Des forces internes, et rien d'autre. Elle déclare un `stiffness_layout` — c'est
de là que la dispersion des forces internes est pilotée — mais ses
`contributions()` sont **vides** pour tous les genres de matrice. La raideur de
suivi \\(\partial f/\partial u\\) (non symétrique) n'est pas implémentée : une
boucle de Newton converge sans elle, plus lentement.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/follower_pressure.rs:example}}
```

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_mecanique.py:pression_suiveuse}}
```

## Compléments

**Ce que ça vaut comme vérification.** Une charge suiveuse ne se contrôle pas en
comparant une valeur à une formule une fois : elle se contrôle en vérifiant
qu'elle **bouge** comme la surface. Trois régimes l'épinglent :

| déplacement | charge attendue |
|---|---|
| aucun | `(−p, 0)` — exactement la charge morte |
| rotation rigide de `θ` | `(−p cosθ, −p sinθ)` — même module, tournée de `θ` |
| étirement `λ` le long du bord | `(−pλ, 0)` — l'aire déformée a grandi |

Le deuxième est ce qu'une charge non suiveuse rate ; le troisième est ce que
rate une charge suiveuse qui se contenterait de tourner sa direction en oubliant
le changement d'aire.
````

