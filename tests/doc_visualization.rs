//! Source des exemples Rust de `book/src/visualization.md`.
//!
//! Tout le fichier est derrière `#![cfg(feature = "viz")]` : sans la feature,
//! `pyrucast::viz` n'existe pas et le fichier se compile à vide. `check_rust`
//! lance `cargo test --features viz`, qui l'exerce pour de bon.
//!
//! Les extraits écrivent des SVG sous des noms courts ; le module bascule dans
//! un dossier jetable et **rend** le répertoire courant à la fin.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

#![cfg(feature = "viz")]

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, RgbColor};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::mesh;
use pyrucast::viz::{ColorScale, Colormap, FieldArg, MeshStyle, Revolve, View};
use pyrucast::Result;

/// Un dossier jetable, et le triangle 3-D dont tous les extraits se servent.
fn scene() -> Result<(std::path::PathBuf, Handle<Coords>, Mesh)> {
    let dossier = std::env::temp_dir().join(format!("pyrucast_viz_{}", std::process::id()));
    std::fs::create_dir_all(&dossier).unwrap();
    let coords = Handle::new(Coords::new(3)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0])?;
    let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    sm.add_cell(&[a.id(), b.id(), c.id()])?;
    Ok((dossier, coords, Mesh::from_submesh(sm)))
}

// ANCHOR: vues
#[test]
fn les_vues_predefinies() {
    let _ = View::front(); // yaw=0, pitch=0      : caméra en +X
    let _ = View::side(); // yaw=90, pitch=0     : caméra en +Y
    let _ = View::top(); // yaw=0, pitch=90     : vue du dessus
    let _ = View::iso(); // yaw=45, pitch≈35.26 : isométrique
    let _ = View::default(); // = iso()
}
// ANCHOR_END: vues

// ANCHOR: export
#[test]
fn exporter_un_sous_maillage_en_svg() -> Result<()> {
    let (dossier, coords, _) = scene()?;
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0])?;
    let mut sm = SubMesh::new(coords, ElementType::TRI3);
    sm.add_cell(&[a.id(), b.id(), c.id()])?;

    // Export vectoriel.
    sm.plot(View::iso(), Some(&dossier.join("triangle.svg")))?;
    // Fenêtre interactive (feature `viz-interactive`).
    // sm.plot(View::default(), None).unwrap();
    Ok(())
}
// ANCHOR_END: export

// ANCHOR: couleur
#[test]
fn chaque_zone_porte_sa_couleur() -> Result<()> {
    let (_, coords, maillage) = scene()?;
    let mut sm = SubMesh::new(coords, ElementType::TRI3);
    sm.set_face_color(RgbColor::new(220, 60, 60));
    assert_eq!(sm.face_color(), RgbColor::new(220, 60, 60));

    // La même couleur pour **toutes** les zones d'un maillage, sans boucle :
    // la méthode rend le maillage, donc elle s'enchaîne.
    let bleu = RgbColor::new(60, 60, 220);
    assert_eq!(maillage.set_face_color(bleu).cell_count(), 1);
    assert!(maillage.iter().all(|z| z.read().face_color() == bleu));
    Ok(())
}
// ANCHOR_END: couleur

// ANCHOR: champ
#[test]
fn tracer_un_champ_avec_son_echelle() -> Result<()> {
    let (dossier, _, mesh) = scene()?;
    let poi1_h = mesh::to_poi1(&mesh)?.get(0)?;

    // Champ déplacement à 2 composantes "UX" / "UY" sur un POI1. `FieldArg`
    // prend l'**agrégat** : une zone seule se remonte par `NodeField::from_sub`.
    let sub = SubNodeField::from_poi1(&poi1_h, vec!["UX".into(), "UY".into()])?;
    // ... remplissage ...
    let u = NodeField::from_sub(sub);

    // Échelle auto, viridis, première composante, rendu interpolé niveau 4.
    mesh.plot_with_field(
        View::default(),
        Some(&dossier.join("ux.svg")),
        FieldArg::Node(&u),
        None,
        ColorScale::default(),
        4,
        None, // titre
    )?;

    // Composante "UY", colormap coolwarm, bornes fixées à [-1, 1], plat.
    let scale = ColorScale {
        cmap: Colormap::CoolWarm,
        vmin: Some(-1.0),
        vmax: Some(1.0),
    };
    mesh.plot_with_field(
        View::default(),
        Some(&dossier.join("uy.svg")),
        FieldArg::Node(&u),
        Some("UY"),
        scale,
        0,
        None, // titre
    )?;
    Ok(())
}
// ANCHOR_END: champ

// ANCHOR: style
#[test]
fn peau_opaque_ou_fil_de_fer() -> Result<()> {
    let (dossier, _, mesh) = scene()?;

    // Peau opaque (équivalent de plot).
    mesh.plot_styled(
        View::iso(),
        Some(&dossier.join("solide.svg")),
        MeshStyle::Surface,
        None, // titre
    )?;
    // Fil de fer : toutes les arêtes.
    mesh.plot_styled(
        View::iso(),
        Some(&dossier.join("fil.svg")),
        MeshStyle::Wireframe,
        None, // titre
    )?;
    Ok(())
}
// ANCHOR_END: style

/// Une section axisymétrique : la révolution n'a de sens que sur elle.
fn section_axisymetrique() -> Result<(std::path::PathBuf, Mesh)> {
    let dossier = std::env::temp_dir().join(format!("pyrucast_rev_{}", std::process::id()));
    std::fs::create_dir_all(&dossier).unwrap();
    let coords = Handle::new(Coords::axisymmetric()?);
    let n: Vec<Node> = [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords, ElementType::TRI3);
    sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    Ok((dossier, Mesh::from_submesh(sm)))
}

// ANCHOR: revolution
#[test]
fn le_corps_de_revolution_se_demande_dans_la_vue() -> Result<()> {
    // Le maillage doit être **axisymétrique** : c'est son repère qui donne
    // l'axe autour duquel le balayage tourne.
    let (dossier, mesh) = section_axisymetrique()?;

    let vue = View {
        revolve: Some(Revolve::full()),
        ..View::iso()
    };
    mesh.plot(vue, Some(&dossier.join("piece.svg")))?;

    // Balayage partiel, ou finesse angulaire choisie à la main.
    let _ = Revolve::new(270.0).unwrap(); // un secteur par 10°
    let _ = Revolve::with_sectors(360.0, 72).unwrap(); // silhouette plus lisse
    Ok(())
}
// ANCHOR_END: revolution
