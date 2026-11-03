//! Source of the Rust examples of `book/src/matrix.md`.
//!
//! The page pulls these functions through `{{#include …:anchor}}` and the test
//! suite runs them. The anchor covers the **whole function**, signature
//! included: in Rust all code lives in a `fn`, and mdbook does not strip the
//! indentation.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::RelationSense;
use pyrucast::ops::model;
use pyrucast::ops::{element_field, matrix, mesh, solver};
use pyrucast::Result;

/// A thermal bar of two SEG2, Dirichlet on the left: the model, its materials
/// and the loading that imposes `T = 1` there.
fn barre() -> Result<(
    Model,
    pyrucast::containers::element_field::ElementField,
    NodeField,
)> {
    let coords = Handle::new(Coords::new(1)?);
    let n: Vec<Node> = (0..3)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 / 2.0]).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords, ElementType::SEG2);
    for i in 0..2 {
        sm.add_cell(&[n[i].id(), n[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;

    let imposed = mesh::poi1_from_nodes(std::slice::from_ref(&n[0]))?;
    let conduction = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
    let appui = model::dirichlet(&imposed, &conduction, "T", RelationSense::Equality)?;
    let mult = appui.multiplier_mesh()?;
    let mult_node = mult.node(0, 0, 0)?;
    let model = conduction.union(&appui)?;
    let materials =
        element_field::material_field(&model, &[("k", 1.0), ("rho", 1.0), ("cp", 1.0)])?;

    let rhs = NodeField::from_submesh(&mult.get(0)?, vec!["imposed".into()])?;
    rhs.get(0)?
        .write()
        .set_value(mult_node.id(), "imposed", 1.0)?;
    Ok((model, materials, rhs))
}

// ANCHOR: bloc_ecrit
#[test]
fn ecrire_un_bloc() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let support = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        Handle::new(sm)
    };

    let mut bloc = SubMatrix::new(
        support.clone(),  // support des lignes
        support.clone(),  // support des colonnes
        vec!["q".into()], // variables duales : les lignes
        vec!["T".into()], // variables primales : les colonnes
        DofOrdering::NodesThenVars,
        Symmetry::Full, // symétrique à lui seul
    );
    // K = [[2, -1], [-1, 2]]
    bloc.add_entry(a.id(), "q", a.id(), "T", 2.0)?;
    bloc.add_entry(a.id(), "q", b.id(), "T", -1.0)?;
    bloc.add_entry(b.id(), "q", a.id(), "T", -1.0)?;
    bloc.add_entry(b.id(), "q", b.id(), "T", 2.0)?;

    let mut k = Matrix::empty();
    k.add_sub(Handle::new(bloc))?;
    k.assemble()?; // assemble les blocs écrits

    assert_eq!((k.n_rows()?, k.n_cols()?), (2, 2));
    assert!(k.symmetric());
    Ok(())
}
// ANCHOR_END: bloc_ecrit

// ANCHOR: somme
#[test]
fn additionner_puis_resoudre() -> Result<()> {
    let (model, materials, rhs) = barre()?;
    let k = matrix::stiffness(&model, &materials)?;
    let m = matrix::mass(&model, &materials)?;
    let dt = 0.1;

    // `/` ne peut pas échouer et rend une `Matrix` ; `+` rend un `Result`,
    // puisqu'un opérande peut ne pas avoir de valeurs.
    let systeme = (&m / dt + &k)?;
    let t = solver::solve(&systeme, &rhs)?;

    assert!(t.node_count() > 0);
    Ok(())
}
// ANCHOR_END: somme

// ANCHOR: lecture
#[test]
fn lire_une_matrice_assemblee() -> Result<()> {
    let (model, materials, _) = barre()?;
    let k = matrix::stiffness(&model, &materials)?;
    let a: NodeId = k.row_mesh()?.node(0, 0, 0)?.id();
    let x = NodeField::from_submesh(&k.col_mesh()?.get(0)?, vec!["T".into()])?;

    // Une valeur.
    let v: f64 = k.get(a, "q", a, "T")?;

    // Les trois tableaux de la matrice creuse, prêtés sans copie…
    let (debuts, colonnes, valeurs): (&[usize], &[usize], &[f64]) = k.csr_arrays()?;
    assert_eq!(debuts.len(), k.n_rows()? + 1);
    assert_eq!(colonnes.len(), valeurs.len());
    // …ou des objets neufs, de nalgebra et nalgebra-sparse.
    let plein: nalgebra::DMatrix<f64> = k.to_dmatrix()?;
    let csr: nalgebra_sparse::CsrMatrix<f64> = k.to_csr()?;
    let csc: nalgebra_sparse::CscMatrix<f64> = k.to_csc()?;

    // Le produit par un champ : `x` est lu aux colonnes, le résultat est posé
    // sur les lignes.
    let y: NodeField = (&k * &x)?;

    let _ = (v, plein, csr, csc, y);
    Ok(())
}
// ANCHOR_END: lecture
