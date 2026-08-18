//! Source des exemples Rust de `book/src/fe-space.md`.
//!
//! La page tire ces fonctions par `{{#include …:ancre}}` et `cargo test` les
//! exécute. L'ancre couvre la **fonction entière**, signature comprise.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node, QuadratureRule};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::Result;

/// Le triangle (0,0), (2,0), (0,2) et son espace EF par défaut.
fn triangle() -> Result<(Handle<Coords>, Mesh, FiniteElementSpace)> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[2.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 2.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    mesh.add_cell(&[a.id(), b.id(), c.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((coords, mesh, fes))
}

// ANCHOR: constructeur
#[test]
fn le_constructeur_par_defaut_pose_lagrange1_et_gauss() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[2.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 2.0])?;

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    mesh.add_cell(&[a.id(), b.id(), c.id()])?;

    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let sub = fes.get(0)?;
    let s = sub.read();
    assert_eq!(s.gauss_count(), 3);
    // Le triangle (0,0), (2,0), (0,2) a |J| = 4 partout :
    // mapping affine, det(J) = 4 = 2 × aire physique du triangle (1/2 × 2 × 2).
    for g in 0..s.gauss_count() {
        let dj = s.det_jacobian(0, g)?;
        assert!((dj - 4.0).abs() < 1e-12);
    }
    Ok(())
}
// ANCHOR_END: constructeur

// ANCHOR: constructeur_explicite
#[test]
fn le_constructeur_explicite_choisit_par_sous_maillage() -> Result<()> {
    let (_, mesh, _) = triangle()?;

    let fes =
        FiniteElementSpace::with(&mesh, &[(Interpolation::Lagrange1, QuadratureRule::Gauss)])?;

    assert_eq!(fes.len(), 1);
    Ok(())
}
// ANCHOR_END: constructeur_explicite

// ANCHOR: evaluations
#[test]
fn evaluer_les_grandeurs_sur_une_cellule() -> Result<()> {
    let (_, _, fes) = triangle()?;
    let sub = fes.get(0)?;

    let s = sub.read();
    for cell_idx in 0..s.cell_count()? {
        for g in 0..s.gauss_count() {
            let n = s.n_at_g(g)?; // N_i(ξ_g)
            let dn = s.dn_at_g(g)?; // ∂N_i/∂ξ_k(ξ_g)
            let jac = s.jacobian(cell_idx, g)?;
            let det_j = s.det_jacobian(cell_idx, g)?;
            let dn_dx = s.dn_dx(cell_idx, g)?;
            // … utiliser ces buffers dans l'assemblage matrice-élémentaire …
            let _ = (n, dn, jac, det_j, dn_dx);
        }
    }
    Ok(())
}
// ANCHOR_END: evaluations

// ANCHOR: deplacement
#[test]
fn deplacer_un_noeud_change_les_evaluations_a_venir() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let sub = FiniteElementSpace::lagrange1(&mesh)?.get(0)?;

    // SEG2 initial : nœuds en x=0 et x=1 → |J| = 0.5 (longueur 1 sur [-1,+1]).
    let dj_before = sub.read().det_jacobian(0, 0)?;
    assert!((dj_before - 0.5).abs() < 1e-12);

    // Étirement : on déplace le second nœud en x=4 → |J| = 2.0 (longueur 4 sur [-1,+1]).
    coords.write().set_position(b.id(), &[4.0, 0.0])?;
    let dj_after = sub.read().det_jacobian(0, 0)?;
    assert!((dj_after - 2.0).abs() < 1e-12);
    Ok(())
}
// ANCHOR_END: deplacement
