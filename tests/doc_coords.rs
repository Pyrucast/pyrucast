//! Source des exemples Rust de `book/src/coords.md`.
//!
//! La page tire ces fonctions par `{{#include …:ancre}}` et `cargo test` les
//! exécute. L'ancre couvre la **fonction entière**, signature comprise : en
//! Rust tout code vit dans un `fn`, et mdbook n'enlève pas l'indentation d'un
//! extrait inclus — montrer la fonction est donc plus honnête que montrer un
//! corps décalé de quatre espaces.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

// ANCHOR: repere
use pyrucast::coords::Coords;

#[test]
fn le_repere_se_choisit_a_la_construction() {
    // Cartésien (par défaut) : dim libre.
    let plan = Coords::new(2).unwrap();
    assert!(!plan.is_axisymmetric());

    // Révolution : la dimension vaut nécessairement 2, donc pas d'argument.
    let axi = Coords::axisymmetric().unwrap();
    assert_eq!(axi.dim(), 2);
    assert!(axi.is_axisymmetric());
}
// ANCHOR_END: repere

// ANCHOR: refcount
use pyrucast::handle::Handle;

#[test]
fn un_noeud_survit_tant_qu_on_le_tient() {
    let coords = Handle::new(Coords::new(2).unwrap());
    // add_node initialise refcount = 1 ; sans décrément, le nœud est protégé.
    let id = coords.write().add_node(&[0.0, 0.0]).unwrap();
    assert_eq!(coords.write().gc(), 0);

    // Après décrément, gc ramasse.
    coords.write().decref(id).unwrap();
    assert_eq!(coords.write().gc(), 1);
}
// ANCHOR_END: refcount

// ANCHOR: configurations
#[test]
fn une_seconde_configuration_clone_la_courante() {
    let coords = Handle::new(Coords::new(2).unwrap());
    let id = coords.write().add_node(&[0.0, 0.0]).unwrap();

    let c2 = coords.write().add_config("deformed");
    coords.write().select(c2).unwrap();
    // les `set_position` suivants modifient désormais la configuration "deformed".
    coords.write().set_position(id, &[0.1, 0.05]).unwrap();

    coords.write().select(0).unwrap();
    assert_eq!(coords.read().position(id).unwrap(), vec![0.0, 0.0]);
    coords.write().select(c2).unwrap();
    assert_eq!(coords.read().position(id).unwrap(), vec![0.1, 0.05]);
}
// ANCHOR_END: configurations

// ANCHOR: permutation
#[test]
fn une_permutation_renumerote_pour_le_solveur() {
    let coords = Handle::new(Coords::new(2).unwrap());
    // Trois nœuds créés ; ids = 0, 1, 2.
    coords.write().add_node(&[0.0, 0.0]).unwrap();
    coords.write().add_node(&[1.0, 0.0]).unwrap();
    coords.write().add_node(&[0.5, 1.0]).unwrap();

    // Permutation posée à la main (le calcul automatique reste à écrire).
    coords.write().set_permutation(vec![2, 0, 1]).unwrap();
    // permutation[0] = 2 : le nœud d'id 0 est en position solveur 2.
    println!("{:?}", coords.read().permutation());

    // Retour à l'identité.
    coords.write().clear_permutation();
    assert!(coords.read().permutation().is_none());
}
// ANCHOR_END: permutation
