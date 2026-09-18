//! Source of the Rust examples of `book/src/coords.md`.
//!
//! The page pulls these functions through `{{#include …:anchor}}` and
//! exécute. L'ancre couvre la **fonction entière**, signature comprise : en
//! Rust all code lives in a `fn`, and mdbook does not strip an included
//! excerpt's indentation — showing the function is therefore more honest than
//! corps décalé de quatre espaces.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

// ANCHOR: repere
use pyrucast::coords::Coords;

#[test]
fn le_repere_se_choisit_a_la_construction() {
    // Cartesian (the default): free dim.
    let plan = Coords::new(2).unwrap();
    assert!(!plan.is_axisymmetric());

    // Revolution: the dimension is necessarily 2, hence no argument.
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
    // add_node initializes refcount = 1; without a decrement, the node is protected.
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
    // the following `set_position` now change the "deformed" configuration.
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

    // Permutation set by hand (the automatic computation is still to be written).
    coords.write().set_permutation(vec![2, 0, 1]).unwrap();
    // permutation[0] = 2: the node with id 0 is at solver position 2.
    println!("{:?}", coords.read().permutation());

    // Back to the identity.
    coords.write().clear_permutation();
    assert!(coords.read().permutation().is_none());
}
// ANCHOR_END: permutation
