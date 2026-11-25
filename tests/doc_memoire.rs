//! Source of the Rust examples of `book/src/memory-model.md` and
//! `book/src/developper/interrompre-une-fonction.md`.
//!
//! The pages pull these functions through `{{#include …:anchor}}` and
//! `cargo test` runs them. The anchor covers the **whole function**, signature
//! included: in Rust all code lives in a `fn`, and mdbook does not strip the
//! indentation — showing the function is therefore more honest than showing a
//! fragment of its body.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::mesh::triangulate_surface_cancellable;
use pyrucast::PyrucastError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Two tickets on the same object; the object dies with the last one.
// ANCHOR: partage
#[test]
fn un_handle_est_un_ticket() {
    let h1 = Handle::new(Coords::new(2).unwrap()); // premier ticket
    let h2 = h1.clone(); // même objet
    assert!(h1.same_object(&h2));
    drop(h1); // h2 tient encore l'objet
    drop(h2); // dernier ticket → l'objet est détruit
}
// ANCHOR_END: partage

/// The lock covers *this* object alone, and lasts as long as the guard.
// ANCHOR: guards
#[test]
fn un_guard_par_objet() {
    let handle = Handle::new(Coords::new(2).unwrap());

    let coords = handle.read(); // verrou lecture sur CET objet seul
    println!("dim = {}", coords.dim()); // coords se comporte comme un &Coords
    drop(coords); // (ou fin de portée) → verrou relâché

    handle.write().add_node(&[0.0, 0.0]).unwrap(); // verrou écriture, le temps de l'appel
}
// ANCHOR_END: guards

/// A buffer view counts as a reader: writing does not wait for it, it refuses.
// ANCHOR: pret
#[test]
fn une_vue_fait_refuser_l_ecriture() {
    let handle = Handle::new(Coords::new(2).unwrap());

    let vue = handle.lend(); // un tableau numpy commence à lire l'objet
    let refus = handle.write_checked(); // écrire est refusé…
    assert!(matches!(refus, Err(PyrucastError::Lent(1)))); // …tant qu'une vue le lit
    drop(vue); // le tableau disparaît

    let mut coords = handle.write_checked().unwrap(); // écrire passe
    coords.add_node(&[0.0, 0.0]).unwrap();
}
// ANCHOR_END: pret

/// A shared cancellation token, armed by whoever you like.
// ANCHOR: annulation
#[test]
fn un_jeton_partage_interrompt_le_mailleur() {
    let coords = Handle::new(Coords::new(2).unwrap());
    let coins: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    for i in 0..4 {
        sm.add_cell(&[coins[i].id(), coins[(i + 1) % 4].id()])
            .unwrap();
    }
    let contour = Mesh::from_submesh(sm);

    let stop = Arc::new(AtomicBool::new(false));
    // A Ctrl+C handler (the `ctrlc` crate, to add to your own Cargo.toml), a
    // supervising thread, a timeout… all arm the same token:
    //     let s = stop.clone();
    //     ctrlc::set_handler(move || s.store(true, Ordering::Relaxed)).ok();

    let mesh = triangulate_surface_cancellable(&contour, ElementType::TRI3, Some(0.5), &*stop);
    assert!(mesh.is_ok()); // rien n'a armé le jeton : le maillage aboutit

    // Token armed in advance: the mesher stops at the first checkpoint.
    stop.store(true, Ordering::Relaxed);
    let interrompu =
        triangulate_surface_cancellable(&contour, ElementType::TRI3, Some(0.5), &*stop);
    assert!(interrompu.is_err());
    // `Deadline::after(Duration::from_secs(10))` would work just as well.
}
// ANCHOR_END: annulation
