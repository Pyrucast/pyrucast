//! Source of the Rust examples on the book's container pages.
//!
//! Couvre `introduction.md`, `node.md`, `mesh.md`, `node-field.md`,
//! `element-field.md`, `evolution.md`, `sauvegarde.md`, `compilation.md` et
//! `operateurs/assemblage.md`. The pages pull these functions through
//! `{{#include …:anchor}}` and `cargo test` runs them. The anchor covers the
//! **fonction entière**, signature comprise.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::evolution::{
    Evolution, Interpolated, OutOfRange, SubEvolution, SubValue,
};
use pyrucast::containers::field::{Field, SubField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::model;
use pyrucast::ops::{element_field, matrix, mesh};
use pyrucast::{archive, Result};

// ANCHOR: premiers_pas
#[test]
fn un_maillage_minimal() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;

    let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    sm.add_cell(&[a.id(), b.id()])?;

    // The aggregate does not carry the `Coords`: the submeshes are what hold it.
    // `Mesh::from_submesh(sm)` is the shortcut for the single-submesh case.
    let mut mesh = Mesh::empty();
    mesh.add_sub(Handle::new(sm))?;
    println!("{}", mesh); // Mesh: 1 submesh(es), 1 cell(s) total
    Ok(())
}
// ANCHOR_END: premiers_pas

// ANCHOR: noeud
#[test]
fn un_noeud_est_un_compteur_de_references() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let n = Node::create_in(coords.clone(), &[1.0, 2.0])?;
    let m = n.clone(); // refcount = 2
    drop(n); // refcount = 1
    drop(m); // refcount = 0
    coords.write().gc(); // collecte
    Ok(())
}
// ANCHOR_END: noeud

// ANCHOR: maillage
#[test]
fn un_maillage_se_compose_de_zones() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.5, 1.0])?;

    let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    sm.add_cell(&[a.id(), b.id(), c.id()])?;

    let sm_handle = Handle::new(sm);
    let mut mesh = Mesh::empty(); // l'agrégat ne porte pas la `Coords`
    mesh.add_sub(sm_handle)?;
    assert_eq!(mesh.cell_count(), 1);
    Ok(())
}
// ANCHOR_END: maillage

// ANCHOR: champ_nodal
#[test]
fn un_champ_aux_noeuds_s_ecrit_par_zone_et_se_lit_par_agregat() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;

    // Support : SubMesh POI1 contenant a et b.
    let sm = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        Handle::new(sm)
    };

    // Champ de déplacement 2D mono-zone : composantes UX, UY.
    let u = NodeField::from_submesh(&sm, vec!["UX".into(), "UY".into()])?;

    // Écriture : via la zone. Lecture : via l'agrégat (ou la zone).
    u.get(0)?.write().set_value(a.id(), "UX", 1.5)?;
    assert_eq!(u.value(a.id(), "UX")?, 1.5);
    assert_eq!(u.value(b.id(), "UX")?, 0.0); // valeur par défaut

    // From a multi-zone mesh: one SubNodeField per submesh.
    let mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::POI1));
    let field = NodeField::new(&mesh, vec!["T".into()])?;
    assert_eq!(field.len(), mesh.len());
    field.check()?; // zones cohérentes aux interfaces
    Ok(())
}
// ANCHOR_END: champ_nodal

// ANCHOR: champ_gauss
#[test]
fn un_champ_aux_points_de_gauss_porte_le_materiau() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
    mesh.add_cell(&[a.id(), b.id(), c.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // 2-D linear elasticity: two material properties, one zone (one subspace).
    let mat = ElementField::new(&fes, vec!["E".into(), "nu".into()])?;
    {
        let mut z = mat.get(0)?.write(); // la zone (SubElementField) — guard
        z.set_uniform("E", 210e9)?; // module d'Young constant
        z.set_uniform("nu", 0.3)?; // Poisson constant
        assert_eq!(z.value(0, 0, "E")?, 210e9);
    }

    // Components per subspace (multi-material):
    let mat2 = ElementField::with(
        &fes,
        &[vec!["E".into(), "nu".into()]], // une liste par sous-espace
    )?;

    // Statistics and arithmetic at the aggregate level.
    assert_eq!(Field::max(&mat, Some("E"))?, 210e9);
    let scaled = &mat * 1.1; // nouveau champ (référence : préserve `mat`)
    mat.mul_to_component("E", 0.95)?; // en place, seulement "E"
    let _ = (mat2, scaled);
    Ok(())
}
// ANCHOR_END: champ_gauss

// ANCHOR: evolution
#[test]
fn une_evolution_interpole_scalaires_et_champs() -> Result<()> {
    // Courbe scalaire X→Y : 0→10, 1→20.
    let se = SubEvolution::new(
        vec![(0.0, SubValue::Scalar(10.0)), (1.0, SubValue::Scalar(20.0))],
        OutOfRange::Error,
    )?;
    match se.interpolate(0.5, None)? {
        SubValue::Scalar(v) => assert_eq!(v, 15.0),
        _ => unreachable!(),
    }
    // Hors plage : Error (défaut) lève ; surcharge Clamp → extrémité.
    assert!(se.interpolate(2.0, None).is_err());

    // Agrégat scalaire → liste de flottants.
    let e = Evolution::from_scalars(vec![(0.0, 10.0), (1.0, 20.0)], OutOfRange::Error)?;
    match e.interpolate(0.5, None)? {
        Interpolated::Scalars(v) => assert_eq!(v, vec![15.0]),
        _ => unreachable!(),
    }
    Ok(())
}
// ANCHOR_END: evolution

// ANCHOR: archive
#[test]
fn sauver_et_relire_un_graphe_d_objets() -> Result<()> {
    // `tempfile` is not a dependency of the project: a unique name in the system's
    // temporary directory is enough.
    let chemin = std::env::temp_dir().join(format!("pyrucast_doc_{}.pyr", std::process::id()));
    let chemin = chemin.to_str().unwrap().to_string();
    let chemin = chemin.as_str();

    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let temperature = NodeField::new(&mesh::to_poi1(&mesh)?, vec!["T".into()])?;

    archive::save(
        chemin,
        &[
            ("maillage fin", &mesh as &dyn archive::ArchiveRoot),
            ("T (°C)", &temperature),
            ("pas de temps", &0.05_f64),
        ],
    )?;

    let mut objets = archive::load(chemin)?;
    let mesh2 = objets.mesh("maillage fin")?; // erreur nommant clef, type attendu, type trouvé
    let dt = objets.float("pas de temps")?;

    assert_eq!(mesh2.cell_count(), 1);
    assert_eq!(dt, 0.05);
    Ok(())
}
// ANCHOR_END: archive

// ANCHOR: mailler_en_rust
#[test]
fn mailler_une_surface_depuis_un_contour() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let coins: Vec<Node> = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords, ElementType::SEG2);
    for i in 0..4 {
        sm.add_cell(&[coins[i].id(), coins[(i + 1) % 4].id()])?;
    }
    let contour = Mesh::from_submesh(sm);

    let mesh = mesh::triangulate_surface(&contour, ElementType::TRI3, Some(1.0))?;

    assert!(mesh.cell_count() > 0);
    Ok(())
}
// ANCHOR_END: mailler_en_rust

// ANCHOR: reassembler
#[test]
fn ajouter_un_bloc_invalide_l_assemblage() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    m.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&m)?;
    let imposed = mesh::poi1_from_nodes(std::slice::from_ref(&a))?;
    let mult = mesh::barycenter(&imposed)?;
    let conduction = model::heat_conduction(&fes)?;
    let model = conduction.union(&model::dirichlet(
        &conduction,
        "T",
        &imposed,
        &mult,
        Default::default(),
    )?)?;
    let materials = element_field::material_field(&model, &[("k", 1.0)])?;
    let support = mesh::to_poi1(&m)?.get(0)?;
    let bloc_supplementaire = SubMatrix::new(
        support.clone(),
        support,
        vec!["q".into()],
        vec!["T".into()],
        DofOrdering::NodesThenVars,
        Symmetry::Full,
    );

    let mut k = matrix::stiffness(&model, &materials)?;
    k.add_sub(Handle::new(bloc_supplementaire))?; // invalide l'état assemblé
    k.assemble()?; // réassemble, nouveau bloc inclus

    assert!(k.n_rows()? > 0);
    Ok(())
}
// ANCHOR_END: reassembler
