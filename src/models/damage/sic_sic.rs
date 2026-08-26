//! SiC/SiC — **orthotropic** damage of a woven ceramic-matrix composite.
//!
//! A SiC/SiC composite is a silicon-carbide matrix reinforced by SiC fibre tows,
//! usually woven. It fails nothing like a metal or like concrete: the matrix
//! cracks first, in planes normal to the tow directions, while the fibres keep
//! carrying load across those cracks. The stiffness therefore falls **by
//! direction**, and by very different amounts in each.
//!
//! No isotropic damage variable can express that. This law carries **one damage
//! per material direction**:
//!
//! ```text
//! d_i = d_max,i · (1 − exp(−(⟨ε_i⟩₊ − ε_0,i)/ε_c,i))      for ⟨ε_i⟩₊ > ε_0,i
//! ```
//!
//! and degrades the orthotropic stiffness direction by direction. The positive
//! part is what matters: a matrix crack opens in **extension** and closes again
//! in compression, so a direction under compression is not degraded at all.
//!
//! ## The material frame is the weave
//!
//! The damage directions are the **material axes**, supplied exactly as for
//! [orthotropic elasticity](crate::models::symmetry) — by the vectors `V1`, `V2`
//! carried in the material field. That is not a coincidence: for a woven
//! composite they *are* the tow directions, and reusing the same frame means a
//! curved part (a wound tube, a shaped panel) gets its damage directions right
//! for free, cell by cell.
//!
//! ## Saturation, not failure
//!
//! Each `d_i` saturates at `d_max,i` rather than reaching one. That is the
//! physical statement that matrix cracking **does not** take the whole
//! stiffness: the fibres remain, and a saturated composite still carries load
//! along its tows. A law that let the damage reach one would predict a collapse
//! that does not happen.

use super::DamageLawKind;
use crate::error::Result;
use crate::models::damage::DamageLaw;
use crate::models::damage::{lame, pos, DamageUpdate, MatRead};
use crate::models::elasticity::ElasticityModel;
use crate::models::symmetry;
use nalgebra::Matrix3;

/// The law's material contract: the elastic constants, then a threshold, a
/// characteristic strain and a saturation per direction — plus the material
/// frame, which the assembler resolves like any other component.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_0_1".into(), "eps_c_1".into(), "d_max_1".into(), "eps_0_2".into(), "eps_c_2".into(), "d_max_2".into(), "eps_0_3".into(), "eps_c_3".into(), "d_max_3".into(), "V1X".into(), "V1Y".into()], &[200000.0, 0.2, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 1.0, 0.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // Trois directions de tissage, chacune avec son seuil, sa saturation et
/// // son endommagement maximal — plus l'axe du repère matériau.
/// assert!(damage::sic_sic::MATERIAL_2D.contains(&"V1X"));
/// assert!(!damage::sic_sic::MATERIAL_2D.contains(&"V2X")); // 2-D : un seul axe
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL_2D: &[&str] = &[
    "E", "nu", "eps_0_1", "eps_c_1", "d_max_1", "eps_0_2", "eps_c_2", "d_max_2", "eps_0_3",
    "eps_c_3", "d_max_3", "V1X", "V1Y",
];
/// The same, with the two axes a 3-D frame needs.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_0_1".into(), "eps_c_1".into(), "d_max_1".into(), "eps_0_2".into(), "eps_c_2".into(), "d_max_2".into(), "eps_0_3".into(), "eps_c_3".into(), "d_max_3".into(), "V1X".into(), "V1Y".into()], &[200000.0, 0.2, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 1.0, 0.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // En 3-D, le repère demande deux axes ; le troisième est V1 × V2.
/// assert!(damage::sic_sic::MATERIAL_3D.contains(&"V2X"));
/// assert!(damage::sic_sic::MATERIAL_3D.len() > damage::sic_sic::MATERIAL_2D.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL_3D: &[&str] = &[
    "E", "nu", "eps_0_1", "eps_c_1", "d_max_1", "eps_0_2", "eps_c_2", "d_max_2", "eps_0_3",
    "eps_c_3", "d_max_3", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];

/// One SiC/SiC step.
///
/// `prev` carries the three history variables `κ_i = max_t ⟨ε_i⟩₊`, one per
/// material direction.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_0_1".into(), "eps_c_1".into(), "d_max_1".into(), "eps_0_2".into(), "eps_c_2".into(), "d_max_2".into(), "eps_0_3".into(), "eps_c_3".into(), "d_max_3".into(), "V1X".into(), "V1Y".into()], &[200000.0, 0.2, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 0.0001, 0.01, 0.9, 1.0, 0.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // Un endommagement **par direction de tissage** : l'état en porte six,
/// // trois seuils et trois endommagements.
/// let u = damage::sic_sic::update(&[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 6], &mat, 2)?;
/// assert_eq!(u.vars.len(), 6);
/// // Une traction selon le premier axe n'endommage que celui-là.
/// assert!(u.vars[3] > 0.0);
/// assert_eq!(u.vars[4], 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn update(
    eps: &[f64; 6],
    prev: &[f64],
    mat: &MatRead,
    space_dim: usize,
) -> Result<DamageUpdate> {
    let e = mat.get("E")?;
    let nu = mat.get("nu")?;
    let (lambda, mu) = lame(e, nu);

    // The weave directions — the same frame an orthotropic elasticity would use.
    let r = symmetry::frame_rotation(mat.field, mat.cell, space_dim)?;

    // The strain in the material axes: `ε_mat = Rᵀ ε R`.
    let eps_global = Matrix3::new(
        eps[0], eps[5], eps[4], eps[5], eps[1], eps[3], eps[4], eps[3], eps[2],
    );
    let eps_mat = r.transpose() * eps_global * r;

    // One damage per direction, driven by the **positive** normal strain there:
    // a matrix crack opens in extension and closes in compression.
    let mut damages = [0.0_f64; 3];
    let mut kappas = [0.0_f64; 3];
    for i in 0..3 {
        let driver = pos(eps_mat[(i, i)]);
        let kappa = prev.get(i).copied().unwrap_or(0.0).max(driver);
        kappas[i] = kappa;
        let eps_0 = mat.get(&format!("eps_0_{}", i + 1))?;
        let eps_c = mat.get(&format!("eps_c_{}", i + 1))?.max(1e-30);
        let d_max = mat.get(&format!("d_max_{}", i + 1))?;
        damages[i] = if kappa > eps_0 {
            (d_max * (1.0 - (-(kappa - eps_0) / eps_c).exp())).clamp(0.0, 1.0 - 1e-12)
        } else {
            0.0
        };
    }

    // Degrade the stiffness **in the material axes**, direction by direction.
    // The normal block is scaled by `(1−d_i)(1−d_j)`, which keeps the operator
    // symmetric and degrades a coupling term as much as the weaker of the two
    // directions it couples; each shear takes the pair it shears.
    let tr = eps_mat[(0, 0)] + eps_mat[(1, 1)] + eps_mat[(2, 2)];
    let mut sigma_mat = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            let intact = if i == j {
                lambda * tr + 2.0 * mu * eps_mat[(i, i)]
            } else {
                2.0 * mu * eps_mat[(i, j)]
            };
            sigma_mat[(i, j)] = (1.0 - damages[i]).sqrt() * (1.0 - damages[j]).sqrt() * intact;
        }
    }

    // …then back to the global axes.
    let sigma_global = r * sigma_mat * r.transpose();
    let sigma = [
        sigma_global[(0, 0)],
        sigma_global[(1, 1)],
        sigma_global[(2, 2)],
        sigma_global[(1, 2)],
        sigma_global[(0, 2)],
        sigma_global[(0, 1)],
    ];

    Ok(DamageUpdate {
        sigma,
        // A scalar summary for visualisation; the state is the three below.
        damage: damages.iter().cloned().fold(0.0_f64, f64::max),
        vars: vec![
            kappas[0], kappas[1], kappas[2], damages[0], damages[1], damages[2],
        ],
    })
}

/// SiC/SiC — orthotropic damage, three directions.
pub(crate) struct SicSic;

impl DamageLawKind for SicSic {
    fn material_components(&self, space_dim: usize) -> &'static [&'static str] {
        if space_dim == 2 {
            MATERIAL_2D
        } else {
            MATERIAL_3D
        }
    }

    fn internal_names(&self) -> Vec<String> {
        vec![
            "kappa_1".into(),
            "kappa_2".into(),
            "kappa_3".into(),
            "d_1".into(),
            "d_2".into(),
            "d_3".into(),
        ]
    }

    fn update(
        &self,
        eps: &[f64; 6],
        prev: &[f64],
        mat: &MatRead,
        space_dim: usize,
    ) -> Result<DamageUpdate> {
        update(eps, prev, mat, space_dim)
    }
}

crate::physics_operator! {
    /// [`model::damage_sic_sic`](crate::ops::model::damage_sic_sic()) — **orthotropic** damage of a
    /// woven SiC/SiC ceramic-matrix composite: one damage per weave direction.
    /// Material `E`, `nu`, then `eps_0_i`, `eps_c_i`, `d_max_i` for `i = 1..3`,
    /// plus the material axes (`V1X, V1Y[, V1Z, V2X…]`).
    ///
    /// The matrix cracks in planes normal to the tows while the fibres keep
    /// carrying load, so the stiffness falls **by direction** and by very
    /// different amounts — which no scalar damage can express. The directions
    /// are the same material axes an orthotropic elasticity uses, so a curved
    /// part gets them right cell by cell.
    ///
    /// Each damage **saturates** at `d_max_i` rather than reaching one: matrix
    /// cracking does not take the whole stiffness, and a law that let it would
    /// predict a collapse that does not happen. State: `kappa_1..3`, `d_1..3`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let m = model::damage_sic_sic(&fes, ElasticityModel::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn damage_sic_sic(fes, model: ElasticityModel) = crate::ops::model::damage_with_law, DamageLaw::SicSic;
    python: "`model.damage_sic_sic(fespace, model)` — **orthotropic** damage of a\nwoven SiC/SiC ceramic-matrix composite: one damage per weave direction.\nMaterial `E`, `nu`, then `eps_0_i`, `eps_c_i`, `d_max_i` for `i = 1..3`,\nplus the material axes (`V1X, V1Y[, V1Z, V2X…]`).\n\nThe matrix cracks in planes normal to the tows while the fibres keep\ncarrying load, so the stiffness falls **by direction** and by very\ndifferent amounts — which no scalar damage can express. The directions\nare the same material axes an orthotropic elasticity uses, so a curved\npart gets them right cell by cell.\n\nEach damage **saturates** at `d_max_i` rather than reaching one: matrix\ncracking does not take the whole stiffness, and a law that let it would\npredict a collapse that does not happen. State: `kappa_1..3`, `d_1..3`."
}
