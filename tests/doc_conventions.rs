//! Source des exemples Rust de `book/src/conventions.md` et `book/src/model.md`.
//!
//! Les pages tirent ces fonctions par `{{#include …:ancre}}` et `cargo test`
//! les exécute. L'ancre couvre la **fonction entière**, signature comprise.
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

    // Pattern-matching sur les variantes.
    match Coords::new(0) {
        Ok(_) => unreachable!(),
        Err(pyrucast::PyrucastError::Message(msg)) => println!("erreur : {msg}"),
        Err(e) => println!("autre erreur : {e}"),
    }
}
// ANCHOR_END: erreurs

// ANCHOR: affichage
#[test]
fn debug_montre_la_structure_display_le_resume() {
    let coords = Handle::new(Coords::new(2).unwrap());
    let c = coords.read();
    println!("{:?}", *c); // vue structurelle (Debug)
    println!("{}", *c); // vue résumée (Display)
}
// ANCHOR_END: affichage

// ANCHOR: serialisation
use pyrucast::archive::Portable;

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
    // 1-D : maillage [0, 1] à un seul SEG2.
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Modèle : conduction (le matériau est fourni à l'assemblage, pas ici)
    // + Dirichlet à gauche. Constructeurs au niveau parent (balaient les
    // sous-espaces de `fes`), composés par `union` — on ne construit jamais
    // de `SubModel` à la main (cf. CONVENTIONS.md).
    let hc = model::heat_conduction(&fes)?;
    // Maillage des nœuds imposés + support des multiplicateurs (barycenter
    // colocalise des nœuds neufs). Le modèle ne crée aucun nœud lui-même.
    let imposed = mesh::poi1_from_nodes(std::slice::from_ref(&a))?;
    let multiplier = mesh::barycenter(&imposed)?;
    let dir = model::dirichlet(
        "T".into(),
        "q".into(),
        &imposed,
        &multiplier,
        None,
        None,
        RelationSense::Equality,
    )?;
    let model = hc.union(&dir)?;

    // Matériau k = 1, appliqué aux sous-modèles qui en ont besoin (Dirichlet
    // est automatiquement ignoré), puis assemblage.
    let materials = element_field::material_field(&model, &[("k", 1.0)])?;
    let k = matrix::stiffness(&model, &materials)?;
    assert_eq!(k.n_rows()?, 3); // 2 nœuds physiques + 1 multiplicateur
    Ok(())
}
// ANCHOR_END: modele

// ANCHOR: filtrer_par_nature
#[test]
fn filtrer_un_modele_et_sa_matrice_par_nature() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let model = model::heat_conduction(&fes)?;
    let materials = element_field::material_field(&model, &[("k", 1.0)])?;
    let k = matrix::stiffness(&model, &materials)?;

    let meca = model.filter(Physics::Mechanical)?; // sous-modèles au moins mécaniques
    let k_meca = k.filter(Physics::Mechanical)?; // blocs au moins mécaniques (non assemblés)
    let natures = k.physics(); // ex. [Thermal, Constraint]

    assert!(meca.is_empty() && k_meca.is_empty()); // ce modèle est thermique
    assert!(natures.contains(&Physics::Thermal));
    Ok(())
}
// ANCHOR_END: filtrer_par_nature
