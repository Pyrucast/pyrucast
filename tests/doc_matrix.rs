//! Source des exemples Rust de `book/src/matrix.md`.
//!
//! La page tire ces fonctions par `{{#include …:ancre}}` et `cargo test` les
//! exécute. L'ancre couvre la **fonction entière**, signature comprise : en
//! Rust tout code vit dans un `fn`, et mdbook n'enlève pas l'indentation.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::{Physics, RelationSense};
use pyrucast::ops::{element_field, matrix, mesh, solver};
use pyrucast::Result;

/// Une barre thermique à deux SEG2, Dirichlet à gauche : le modèle, ses
/// matériaux, le nœud-multiplicateur et son chargement.
fn barre() -> Result<(
    Model,
    pyrucast::containers::element_field::ElementField,
    Node,
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
    let mult = mesh::barycenter(&imposed)?;
    let mult_node = mult.node(0, 0, 0)?;
    let model = Model::heat_conduction(&fes)?.union(&Model::dirichlet(
        "T".into(),
        "q".into(),
        &imposed,
        &mult,
        None,
        None,
        RelationSense::Equality,
    )?)?;
    let materials = element_field::material_field(&model, &[("k", 1.0)])?;

    let mut rhs = NodeField::from_submesh(&mult.get(0)?, vec!["imposed_T".into()])?;
    rhs.get(0)?
        .write()
        .set_value(mult_node.id(), "imposed_T", 1.0)?;
    Ok((model, materials, mult_node, rhs))
}

// ANCHOR: filtrage
#[test]
fn filtrer_une_matrice_par_nature() -> Result<()> {
    let (model, materials, _, _) = barre()?;
    let k = matrix::stiffness(&model, &materials)?;

    let k_meca = k.filter(Physics::Mechanical)?; // blocs au moins mécaniques
    let natures = k.physics()?; // ex. [Thermal, Constraint]

    assert!(k_meca.is_empty()); // ce modèle est thermique
    assert!(natures.contains(&Physics::Thermal));
    Ok(())
}
// ANCHOR_END: filtrage

// ANCHOR: facteur
#[test]
fn diviser_une_matrice_ne_reecrit_aucune_valeur() -> Result<()> {
    let (model, materials, _, _) = barre()?;
    let m = matrix::stiffness(&model, &materials)?;
    let a = m.row_mesh()?.node(0, 0, 0)?.id();
    let dt = 0.1;

    let mut m_dt = (&m / dt)?; // facteur = 1/dt sur chaque bloc, aucune valeur réécrite
                               // Le facteur d'un bloc **calculé** ne se matérialise qu'à l'assemblage :
                               // sans ce `assemble`, la relecture rendrait des zéros.
    m_dt.assemble()?;
    assert_eq!(m.get(a, "q", a, "T")?, m_dt.get(a, "q", a, "T")? * dt); // m inchangée
    Ok(())
}
// ANCHOR_END: facteur

// ANCHOR: somme
#[test]
fn composer_deux_matrices_puis_resoudre() -> Result<()> {
    let (model, materials, _, rhs) = barre()?;
    let k = matrix::stiffness(&model, &materials)?;
    let m = matrix::stiffness(&model, &materials)?;
    let dt = 0.1;

    // Composition : `union` côté Rust — le `|` de la surface Python n'a pas
    // d'équivalent en surcharge d'opérateur ici.
    let mut sys = (&m / dt)?.union(&k)?;
    sys.assemble()?; // requis dès qu'un bloc calculé est présent
    let u = solver::lu::solve(&sys, &rhs)?;

    assert!(u.node_count()? > 0);
    Ok(())
}
// ANCHOR_END: somme

// ANCHOR: bloc_carre
#[test]
fn les_entrees_vivent_dans_un_bloc() -> Result<()> {
    // Les entrées vivent dans un **bloc**, jamais dans l'agrégat : un bloc
    // connaît ses supports POI1 (lignes et colonnes) et ses noms de variables.
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let support = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        Handle::new(sm)
    };

    let mut block = SubMatrix::new(
        support.clone(),  // support des lignes
        support.clone(),  // support des colonnes (carré ici)
        vec!["q".into()], // variables duales   → lignes
        vec!["T".into()], // variables primales → colonnes
        DofOrdering::NodesThenVars,
        true, // symétrique
    )?;

    // Modèle simple à 2 nœuds (segment) :
    //   K = [[ 2, -1], [-1,  2]]
    block.add_entry(a.id(), "q", a.id(), "T", 2.0)?;
    block.add_entry(a.id(), "q", b.id(), "T", -1.0)?;
    block.add_entry(b.id(), "q", a.id(), "T", -1.0)?;
    block.add_entry(b.id(), "q", b.id(), "T", 2.0)?;

    let mut k = Matrix::empty();
    k.add_sub(Handle::new(block))?;
    k.finalize()?; // requis avant tout usage solveur

    assert_eq!(k.n_rows()?, 2);
    assert_eq!(k.n_cols()?, 2);
    assert!(k.symmetric()?);
    Ok(())
}
// ANCHOR_END: bloc_carre

// ANCHOR: bloc_rectangulaire
#[test]
fn un_bloc_de_lagrange_est_rectangulaire() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let support = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        Handle::new(sm)
    };

    // 2 contraintes : les multiplicateurs m0/m1 lient les nœuds primaires a/b.
    // Le bloc est rectangulaire dès que les deux supports diffèrent — ici ils
    // ont la même taille, mais ce sont deux nuages de nœuds distincts.
    let m0 = Node::create_in(coords.clone(), &[0.0])?;
    let m1 = Node::create_in(coords.clone(), &[1.0])?;
    let mult_support = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[m0.id()])?;
        sm.add_cell(&[m1.id()])?;
        Handle::new(sm)
    };
    let mut block = SubMatrix::new(
        mult_support,
        support.clone(),
        vec!["T".into()],
        vec!["T".into()],
        DofOrdering::NodesThenVars,
        false,
    )?;
    block.add_entry(m0.id(), "T", a.id(), "T", 1.0)?;
    block.add_entry(m1.id(), "T", b.id(), "T", 1.0)?;

    let mut c = Matrix::empty();
    c.add_sub(Handle::new(block))?;
    c.finalize()?;
    assert_eq!(c.n_rows()?, 2);
    assert_eq!(c.n_cols()?, 2);
    // "T" est interné une seule fois dans la table de noms même s'il apparaît
    // côté ligne ET côté colonne (la collision est résolue par les `NodeId`
    // distincts : les multiplicateurs sont des nœuds à part entière).
    assert_eq!(c.field_names()?.len(), 1);
    Ok(())
}
// ANCHOR_END: bloc_rectangulaire

// ANCHOR: lecture
#[test]
fn lire_une_matrice_assemblee() -> Result<()> {
    let (model, materials, _, _) = barre()?;
    let k = matrix::stiffness(&model, &materials)?;
    let a: NodeId = k.row_mesh()?.node(0, 0, 0)?.id();
    let x = NodeField::from_submesh(&k.col_mesh()?.get(0)?, vec!["T".into()])?;

    // Toutes ces lectures traversent l'état assemblé : elles rendent un
    // `Result` et échouent tant que `finalize()` (ou `assemble()`) n'a pas
    // été appelé.

    // Valeur à une coordonnée (somme de toutes les entrées COO à ce point).
    let v: f64 = k.get(a, "q", a, "T")?;

    // Vue dense ligne-major (flat Vec, pratique pour Python).
    let d: Vec<f64> = k.dense()?;
    assert_eq!(d.len(), k.n_rows()? * k.n_cols()?);

    // Vue dense typée nalgebra (column-major DMatrix), prête pour LU/Cholesky.
    let m: nalgebra::DMatrix<f64> = k.to_dmatrix()?;

    // Vues creuses nalgebra-sparse, prêtes pour les solveurs creux.
    let csr: &nalgebra_sparse::CsrMatrix<f64> = k.to_csr()?;
    let csc: nalgebra_sparse::CscMatrix<f64> = k.to_csc()?;

    // Itération sur les triplets bruts (ordre d'insertion préservé). Une
    // entrée est un 5-uplet `(nœud ligne, var duale, nœud colonne, var
    // primale, valeur)` — les noms de variables y sont déjà résolus.
    for (row_node, row_var, col_node, col_var, value) in k.iter_entries()? {
        let _ = (row_node, row_var, col_node, col_var, value);
    }

    // Produit matrice · champ : `x` est lu aux DOFs *colonnes* (vars
    // **primales**), le résultat est un `NodeField` sur les DOFs *lignes*
    // (vars **duales**) — `K · u = f`. L'opérateur `*` en est le sucre.
    let y: NodeField = k.mul_field(&x)?;
    let y: NodeField = (&k * &x)?;

    let _ = (v, m, csr, csc, y);
    Ok(())
}
// ANCHOR_END: lecture
