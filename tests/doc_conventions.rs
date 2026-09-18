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
    // 1-D: a [0, 1] mesh with a single SEG2.
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Model: conduction (the material is supplied at assembly, not here) +
    // Dirichlet on the left. Constructors at the parent level (they sweep
    // `fes`'s subspaces), composed with `union` — a `SubModel` is never built by
    // hand (see CONVENTIONS.md).
    let hc = model::heat_conduction(&fes)?;
    // Mesh of the imposed nodes + support of the multipliers (barycenter
    // co-locates fresh nodes). The model creates no node itself.
    let imposed = mesh::poi1_from_nodes(std::slice::from_ref(&a))?;
    let multiplier = mesh::barycenter(&imposed)?;
    let dir = model::dirichlet(&hc, "T", &imposed, &multiplier, RelationSense::Equality)?;
    let model = hc.union(&dir)?;

    // Material k = 1, applied to the sub-models that need it (Dirichlet is
    // skipped automatically), then assembly.
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
