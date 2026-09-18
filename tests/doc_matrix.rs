//! Source of the Rust examples of `book/src/matrix.md`.
//!
//! The page pulls these functions through `{{#include …:anchor}}` and
//! exécute. L'ancre couvre la **fonction entière**, signature comprise : en
//! Rust all code lives in a `fn`, and mdbook does not strip the indentation.
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
use pyrucast::ops::model;
use pyrucast::ops::{element_field, matrix, mesh, solver};
use pyrucast::Result;

/// A thermal bar of two SEG2, Dirichlet on the left: the model, its materials,
/// the multiplier node and its loading.
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
    let conduction = model::heat_conduction(&fes)?;
    let model = conduction.union(&model::dirichlet(
        &conduction,
        "T",
        &imposed,
        &mult,
        RelationSense::Equality,
    )?)?;
    let materials = element_field::material_field(&model, &[("k", 1.0)])?;

    let rhs = NodeField::from_submesh(&mult.get(0)?, vec!["imposed_T".into()])?;
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
    let natures = k.physics(); // ex. [Thermal, Constraint]

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
                               // A **computed** block's factor materializes only at assembly: without this
                               // `assemble`, reading back would return zeros.
    m_dt.assemble()?;
    assert_eq!(m.get(a, "q", a, "T"), m_dt.get(a, "q", a, "T") * dt); // m inchangée
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
    // The entries live in a **block**, never in the aggregate: a block knows its
    // POI1 supports (rows and columns) and its variable names.
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

    // A simple 2-node model (a segment):
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
    assert!(k.symmetric());
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

    // 2 constraints: the multipliers m0/m1 tie the primary nodes a/b.
    // The block is rectangular as soon as the two supports differ — here they
    // have the same size, but they are two distinct node clouds.
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
    // "T" is interned once only in the name table even though it appears on the
    // row side AND the column side (the collision is settled by the distinct
    // `NodeId`: the multipliers are nodes in their own right).
    assert_eq!(c.field_names().len(), 1);
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
    // `Result` and fail until `finalize()` (or `assemble()`) has
    // été appelé.

    // Value at a coordinate (the sum of every COO entry at that point).
    let v: f64 = k.get(a, "q", a, "T");

    // Dense row-major view (a flat Vec, handy for Python).
    let d: Vec<f64> = k.dense()?;
    assert_eq!(d.len(), k.n_rows()? * k.n_cols()?);

    // Typed nalgebra dense view (column-major DMatrix), ready for LU/Cholesky.
    let m: nalgebra::DMatrix<f64> = k.to_dmatrix()?;

    // nalgebra-sparse sparse views, ready for the sparse solvers. `to_csr`
    // materializes; `csr_arrays` borrows the three arrays without copying.
    let csr: nalgebra_sparse::CsrMatrix<f64> = k.to_csr()?;
    let csc: nalgebra_sparse::CscMatrix<f64> = k.to_csc()?;
    let (offsets, cols, vals): (&[usize], &[usize], &[f64]) = k.csr_arrays()?;
    assert_eq!(offsets.len(), k.n_rows()? + 1);
    assert_eq!(cols.len(), vals.len());

    // Iteration over the raw triplets (insertion order preserved). An entry is a
    // 5-tuple `(row node, dual var, column node, primal var, value)` — the
    // variable names are already resolved there.
    for (row_node, row_var, col_node, col_var, value) in k.iter_entries() {
        let _ = (row_node, row_var, col_node, col_var, value);
    }

    // Matrix · field product: `x` is read at the *column* DOFs (**primal** vars),
    // the result is a `NodeField` on the *row* DOFs (**dual** vars) — `K · u = f`.
    // The `*` operator is its sugar.
    let y: NodeField = k.mul_field(&x)?;
    let y_sucre: NodeField = (&k * &x)?; // le même produit, en opérateur

    let _ = (v, m, csr, csc, y, y_sucre);
    Ok(())
}
// ANCHOR_END: lecture
