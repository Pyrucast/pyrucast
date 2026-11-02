//! Source of the Rust examples of `book/src/conventions.md` and `book/src/model.md`.
//!
//! The pages pull these functions through `{{#include …:anchor}}` and
//! runs them. The anchor covers the **whole function**, signature included.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::{Physics, RelationSense};
use pyrucast::ops::model;
use pyrucast::ops::{element_field, matrix, mesh};
use pyrucast::Result;

// ANCHOR: erreurs
#[test]
fn une_erreur_se_lit_et_se_filtre() {
    // Dimension nulle — erreur attendue.
    let err = Coords::new(0).unwrap_err();
    assert!(err.to_string().contains("dim must be ≥ 1"));

    // Pattern matching on the variants.
    match Coords::new(0) {
        Ok(_) => unreachable!(),
        Err(pyrucast::PyrucastError::Message(msg)) => println!("erreur : {msg}"),
        Err(e) => println!("autre erreur : {e}"),
    }
}
// ANCHOR_END: erreurs

// ANCHOR: serialisation
use pyrucast::archive::Portable;
use pyrucast::models::symmetry::MaterialSymmetry;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct Pt {
    x: f64,
    y: f64,
}

#[test]
fn un_seul_mecanisme_de_serialisation() {
    let original = Pt { x: 1.5, y: -2.0 };
    let bytes = original.to_bytes().unwrap();
    let restored = Pt::from_bytes(&bytes).unwrap();
    assert_eq!(original, restored);
}
// ANCHOR_END: serialisation

// ANCHOR: modele
#[test]
fn un_modele_se_declare_et_s_assemble() -> Result<()> {
    // Une barre [0, 1] d'un seul SEG2.
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut maillage = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    maillage.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&maillage)?;

    // La conduction, puis l'appui qui fige T en `a` ; `union` est le `|` de
    // Python.
    let conduction = model::heat_conduction(&fes, MaterialSymmetry::Isotropic)?;
    let impose = mesh::poi1_from_nodes(std::slice::from_ref(&a))?;
    let appui = model::dirichlet(&impose, &conduction, "T", RelationSense::Equality)?;
    let modele = conduction.union(&appui)?;

    // Le matériau arrive à l'assemblage ; l'appui n'en demande pas.
    let materiaux = element_field::material_field(&modele, &[("k", 1.0)])?;
    let k = matrix::stiffness(&modele, &materiaux)?;
    assert_eq!(k.n_rows()?, 3); // 2 nœuds + 1 nœud-multiplicateur

    // L'appui fige une température : il est thermique aussi.
    let thermique = modele.filter(Physics::Thermal);
    let seule = thermique.exclude(Physics::Constraint);
    assert_eq!((thermique.len(), seule.len()), (2, 1));
    Ok(())
}
// ANCHOR_END: modele
