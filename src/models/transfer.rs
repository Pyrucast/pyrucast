//! The exchange law `h(a − b)` — what a boundary and an interface share.
//!
//! One law, two situations, and the only difference is **which side of the
//! equals sign the far medium lives on**:
//!
//! | | the far side is… | where it goes |
//! |---|---|---|
//! | [boundary](crate::models::boundary_transfer) | a **datum** (an ambient value) | the right-hand side, `h·a_ext ∫N dΓ` |
//! | [interface](crate::models::interface_transfer) | an **unknown** (the other mesh's field) | the matrix, as a coupling block |
//!
//! The off-diagonal block *is* the right-hand side made implicit. That is why
//! the two physics share this module rather than each carrying a copy of
//! `∫NᵢNⱼ`: the boundary kernel is the interface kernel with both sides on the
//! same cell.
//!
//! ## What is transferred is the caller's to say
//!
//! Neither physics knows what it transports. It is given a list of
//! `(primal, dual)` pairs — the same shape
//! [`embedded`](crate::models::embedded) and [`contact`](crate::models::contact)
//! take, the two other laws that tie meshes together — and everything else
//! follows from it:
//!
//! ```text
//! transferred      material          behaviour input     behaviour output
//! ("T", "q")       h_T               T      / jump_T     flux_T
//! ("c_H2","j_H2")  h_c_H2            c_H2   / jump_c_H2  flux_c_H2
//! ("u_x", "f_x")   h_u_x             u_x    / jump_u_x   flux_u_x
//! ```
//!
//! One coefficient per transferred quantity, named after it. A surface exchange
//! on displacements is then a **Winkler elastic foundation**, and an interface
//! one a joint of finite stiffness — neither needed a line of new physics.
//!
//! ## When *not* to use it on displacements
//!
//! Tying two surfaces by making `h` large is a penalty method, and a
//! [`Mpc`](crate::models::mpc) does it exactly, without degrading the
//! conditioning. The test is where the number comes from: **if `h` comes from a
//! measurement this is physics; if `h` was chosen "large enough", it wanted a
//! constraint.**

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::error::{PyrucastError, Result};
use crate::models::{CellGeom, ElementLayout, Physics};

/// The material component carrying the exchange coefficient of one transferred
/// quantity — `h_T`, `h_c_H2`, `h_u_x`.
///
/// Derived rather than tabulated, which is what made
/// [`Domain::material_components`](crate::models::Domain::material_components)
/// owned: the name is not known until the caller says what is transferred.
///
/// ```
/// # use pyrucast::models::transfer;
/// // Le nom n'est pas tabulé : il se déduit de ce que l'appelant transporte.
/// assert_eq!(transfer::coefficient_name("T"), "h_T");
/// assert_eq!(transfer::coefficient_name("c_H2"), "h_c_H2");
/// ```
pub fn coefficient_name(primal: &str) -> String {
    format!("h_{primal}")
}

/// The behaviour **output** component of one transferred quantity: the exchanged
/// flux density `h·(a − b)`.
///
/// ```
/// # use pyrucast::models::transfer;
/// // La **sortie** de la loi : la densité de flux échangée.
/// assert_eq!(transfer::flux_name("T"), "flux_T");
/// ```
pub fn flux_name(primal: &str) -> String {
    format!("flux_{primal}")
}

/// The behaviour **input** component of an interface law: the jump of one
/// transferred quantity across it, `a₁ − a₂`.
///
/// A boundary law needs no such name — its input is the field itself, the far
/// side being a datum rather than a second unknown.
///
/// ```
/// # use pyrucast::models::transfer;
/// // L'**entrée** d'une loi d'interface : le saut a₁ − a₂. Une loi de bord
/// // n'en a pas besoin — son entrée est le champ lui-même.
/// assert_eq!(transfer::jump_name("T"), "jump_T");
/// ```
pub fn jump_name(primal: &str) -> String {
    format!("jump_{primal}")
}

/// Check a caller's transferred list, and return the material contract it
/// implies. An empty list is a caller error: a law that transfers nothing has no
/// matrix to assemble and no coefficient to read.
///
/// ```
/// # use pyrucast::models::transfer;
/// let paires = vec![("T".to_string(), "q".to_string())];
/// assert_eq!(transfer::material_contract("BoundaryTransfer", &paires)?,
///            vec!["h_T".to_string()]);
/// // Une loi qui ne transporte rien n'a ni matrice ni coefficient : c'est
/// // une erreur d'appel, pas un cas dégénéré à laisser passer.
/// assert!(transfer::material_contract("BoundaryTransfer", &[]).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn material_contract(label: &str, components: &[(String, String)]) -> Result<Vec<String>> {
    if components.is_empty() {
        return Err(PyrucastError::Message(format!(
            "{label}: nothing to transfer — give at least one (primal, dual) pair, e.g. \
             [(\"T\", \"q\")] for a thermal exchange or [(\"u_x\", \"f_x\"), …] for an elastic \
             one"
        )));
    }
    Ok(components
        .iter()
        .map(|(primal, _)| coefficient_name(primal))
        .collect())
}

/// The `'static` singleton slice of one nature, so a physics whose nature is
/// **stored** can still answer [`SubModelKind::physics`](crate::models::SubModelKind::physics)
/// without the trait
/// having to hand out owned data.
///
/// A transfer law's nature cannot be deduced from its variable names, which the
/// caller chooses freely, so it is declared and kept; but there are finitely
/// many natures, and each has exactly one slice.
///
/// ```
/// # use pyrucast::models::transfer;
/// # use pyrucast::models::Physics;
/// // La nature d'une loi de transfert ne se déduit pas des noms de
/// // variables, que l'appelant choisit : elle est déclarée, puis rendue
/// // ici sous la forme d'une tranche `'static`.
/// assert_eq!(transfer::physics_slice(Physics::Thermal), &[Physics::Thermal]);
/// ```
pub fn physics_slice(physics: Physics) -> &'static [Physics] {
    match physics {
        Physics::Mechanical => &[Physics::Mechanical],
        Physics::Thermal => &[Physics::Thermal],
        Physics::Constraint => &[Physics::Constraint],
        Physics::Other => &[Physics::Other],
        Physics::Diffusion => &[Physics::Diffusion],
        Physics::Radiation => &[Physics::Radiation],
    }
}

/// `sign · h ∫_Γ N_i^row N_j^col dΓ`, one uncoupled sub-block per transferred
/// quantity.
///
/// The four blocks of an interface come from this one function: the two diagonal
/// ones with `row_geom == col_geom` and `sign = +1`, the two off-diagonal ones
/// with the facing cell and `sign = −1`. A boundary term is the diagonal case
/// alone.
///
/// `ke` is laid out [`NodesThenVars`](crate::containers::matrix::DofOrdering),
/// so a node's variables are contiguous. Quantities do **not** couple to each
/// other — heat crossing a joint does not drive hydrogen across it — which is
/// why only the diagonal in the variable index is written.
///
/// The measure comes from the **row** side: on a conforming interface the two
/// carry the same surface, so either would do, and taking the row side keeps the
/// four blocks integrated identically — which is what makes them sum to a
/// consistent operator.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::transfer;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["h_T".into()], &[2.0]).unwrap());
/// # use pyrucast::models::ElementLayout;
/// // Un coefficient `h_<primal>` par couple transféré, dans l'ordre que
/// // `material_components` déclare — ici un seul, donc l'identité.
/// let lay = ElementLayout {
///     material: vec![0], optional_material: vec![], state: vec![],
/// };
/// // h ∫ N_i N_j dΓ : la mesure vient du côté **ligne**, ce qui fait que
/// // les quatre blocs d'une interface s'intègrent identiquement.
/// let bloc = assemble_block(
///     std::slice::from_ref(&zone), &support, &support,
///     vec!["q".into()], vec!["T".into()], DofOrdering::NodesThenVars, true,
///     &mat, None,
///     |geoms, m, _s, ke| {
///         transfer::exchange_matrix(&geoms[0], &geoms[0], m, &lay, 1.0, ke)
///     },
/// )?;
/// // La somme des entrées vaut h × la longueur du segment.
/// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
/// assert!((total - 2.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn exchange_matrix(
    row_geom: &CellGeom,
    col_geom: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    sign: f64,
    ke: &mut [f64],
) -> Result<()> {
    // One coefficient `h_<primal>` per transferred pair, in the order
    // `material_components` declares — resolved once for the zone.
    let coefficients = &lay.material;
    let n_vars = coefficients.len();
    let row_stride = col_geom.n_nodes * n_vars;
    for g in 0..row_geom.n_gauss {
        let row_shape = row_geom.n_at_g(g);
        let col_shape = col_geom.n_at_g(g);
        let w = row_geom.det_j_w(g);
        let row = material.row(row_geom.cell, g);
        for (v, &comp) in coefficients.iter().enumerate() {
            let hw = sign * row[comp as usize] * w;
            if hw == 0.0 {
                continue;
            }
            for i in 0..row_geom.n_nodes {
                let row = (i * n_vars + v) * row_stride;
                let hw_ni = hw * row_shape[i];
                for j in 0..col_geom.n_nodes {
                    ke[row + j * n_vars + v] += hw_ni * col_shape[j];
                }
            }
        }
    }
    Ok(())
}

/// `f_i = ∫ N_i · flux dΓ` — the internal nodal fluxes of an exchange law,
/// weighted by `N` and not by `Bᵀ`.
///
/// A film integrand is a flux **density**, not a gradient-conjugate quantity, so
/// the continuum default would be wrong here. For a linear law this equals
/// `(K·u)_i`, which is the invariant the internal forces must satisfy.
///
/// It **cannot fail**: the zone settled the shape of every read before the
/// parallel region, so this slices rows and adds. The `Result` is the trait's,
/// not its own — [`crate::models::SubModelKind::internal_force_element`] returns
/// one because other implementers need it.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::assemble_block;
/// # use pyrucast::models::transfer;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["h_T".into()], &[2.0]).unwrap());
/// # let flux = SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["flux_T".into()], &[1.0])?;
/// # let fe = std::sync::Mutex::new(vec![0.0; 2]);
/// // ∫ Nᵀ · flux dΓ — le résidu d'une loi de transfert, du côté nodal. Les
/// // composantes lues sont résolues **une fois**, avant la boucle.
/// let lay = flux.resolve_components(&["flux_T"], "flux")?;
/// // Une densité unité sur un segment de longueur 1 se partage en deux.
/// pyrucast::models::kernel::reduce_cells(&zone, |geom| {
///     transfer::internal_force(geom, &flux, &lay, &mut fe.lock().unwrap())?;
///     Ok(0.0)
/// })?;
/// assert!((fe.lock().unwrap().iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn internal_force(
    geom: &CellGeom,
    stress: &SubElementField,
    lay: &[u32],
    fe: &mut [f64],
) -> Result<()> {
    let n_vars = lay.len();
    let stride = stress.component_count();
    let values = stress.values();
    for g in 0..geom.n_gauss {
        let shape = geom.n_at_g(g);
        let w = geom.det_j_w(g);
        let start = (geom.cell * geom.n_gauss + g) * stride;
        for (v, &comp) in lay.iter().enumerate() {
            let flux_w = values[start + comp as usize] * w;
            for i in 0..geom.n_nodes {
                fe[i * n_vars + v] += shape[i] * flux_w;
            }
        }
    }
    Ok(())
}
