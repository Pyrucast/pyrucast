//! Saving and reloading a graph of objects.
//!
//! The requirement these tests exist for: *the objects must be readable back,
//! keep their coherence, and introduce no duplication — two fields on one
//! support stay two fields on one support.* Everything else here follows from
//! that, or guards the rule that decides what the file carries.

use pyrucast::aggregate::Aggregate;
use pyrucast::archive::{self, ArchiveRoot};
use pyrucast::atoms::element_type::ElementType;
use pyrucast::atoms::node::Node;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::error::Result;
use pyrucast::handle::Handle;
use pyrucast::ops::model;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pyrucast_archive_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A segment mesh of `n` cells along x, its coordinates, and its node ids.
fn line_with_ids(n: usize) -> Result<(Handle<Coords>, Mesh, Vec<pyrucast::atoms::NodeId>)> {
    let coords = Handle::new(Coords::new(2)?);
    let nodes: Vec<Node> = (0..=n)
        .map(|i| Node::create_in(coords.clone(), &[i as f64, 0.0]).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    for i in 0..n {
        sm.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let mesh = Mesh::from_submesh(sm);
    let ids = nodes.iter().map(|n| n.id()).collect();
    Ok((coords, mesh, ids))
}

/// The same, when the node ids are not needed.
fn line(n: usize) -> Result<(Handle<Coords>, Mesh)> {
    let (c, m, _) = line_with_ids(n)?;
    Ok((c, m))
}

// ─── The requirement ────────────────────────────────────────────────────────

/// Two fields built on one support come back on **one** support, not two
/// copies of it. Written through one, read through the other.
#[test]
fn sharing_survives_the_round_trip() -> Result<()> {
    let (_coords, mesh) = line(3)?;
    let poi = mesh.get(0).unwrap().read().to_poi1()?;

    let t = NodeField::from_sub(SubNodeField::from_poi1(&poi, vec!["T".into()])?);
    let f = NodeField::from_sub(SubNodeField::from_poi1(&poi, vec!["f".into()])?);
    assert!(
        t.get(0).unwrap().read().support().same_object(&poi),
        "the fixture itself must share the support"
    );

    let path = tmp("sharing.pyr");
    archive::save(
        &path,
        &[
            ("temperature", &t as &dyn ArchiveRoot),
            ("force", &f),
            ("maillage", &mesh),
        ],
    )?;

    let mut back = archive::load(&path)?;
    let t2 = back.node_field("temperature")?;
    let f2 = back.node_field("force")?;

    let st = t2.get(0).unwrap().read().support();
    let sf = f2.get(0).unwrap().read().support();
    assert!(
        st.same_object(&sf),
        "the two reloaded fields must share ONE support, not two copies"
    );

    // Not merely equal by value: writing through one is seen through the other.
    st.write()
        .set_face_color(pyrucast::atoms::RgbColor::new(7, 8, 9));
    assert_eq!(
        sf.read().face_color(),
        pyrucast::atoms::RgbColor::new(7, 8, 9)
    );
    Ok(())
}

/// The mesh saved alongside shares its coordinates with the fields' support.
#[test]
fn one_coords_for_the_whole_file() -> Result<()> {
    let (coords, mesh) = line(2)?;
    let poi = mesh.get(0).unwrap().read().to_poi1()?;
    let t = NodeField::from_sub(SubNodeField::from_poi1(&poi, vec!["T".into()])?);

    let path = tmp("one_coords.pyr");
    archive::save(
        &path,
        &[
            ("maillage", &mesh as &dyn ArchiveRoot),
            ("T", &t),
            ("coords", &coords),
        ],
    )?;

    let mut back = archive::load(&path)?;
    let mesh2 = back.mesh("maillage")?;
    let t2 = back.node_field("T")?;
    let c2 = back.coords("coords")?;

    let from_mesh = mesh2.get(0).unwrap().read().coords();
    let from_field = t2.get(0).unwrap().read().coords();
    assert!(from_mesh.same_object(&c2));
    assert!(from_field.same_object(&c2));
    Ok(())
}

// ─── The counters ───────────────────────────────────────────────────────────

/// Node refcounts come back as a fresh build would leave them: exactly the
/// references the file carries, recounted from zero.
#[test]
fn node_counts_are_those_of_a_fresh_build() -> Result<()> {
    let (coords, mesh, ids) = line_with_ids(2)?;
    // The fixture has already dropped its `Node` atoms, so what protects these
    // nodes now is the submesh alone — which is precisely what the file holds.

    let path = tmp("counts.pyr");
    archive::save(&path, &[("maillage", &mesh as &dyn ArchiveRoot)])?;

    let mut back = archive::load(&path)?;
    let mesh2 = back.mesh("maillage")?;
    let c2 = mesh2.get(0).unwrap().read().coords();

    for &id in &ids {
        assert_eq!(
            c2.read().refcount(id),
            coords.read().refcount(id),
            "node {id:?} must come back counted exactly as it is counted live"
        );
    }
    // Nothing the submesh holds is collectable.
    assert_eq!(c2.write().gc(), 0);
    Ok(())
}

/// A `Node` the caller holds is **not** archived: it is an atom of their stack.
/// The reloaded graph is therefore protected by one reference less — the
/// documented consequence of recounting from zero.
#[test]
fn a_node_the_caller_holds_is_not_archived() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    sm.add_cell(&[a.id(), b.id()])?;
    let mesh = Mesh::from_submesh(sm);

    let path = tmp("loose_node.pyr");
    archive::save(&path, &[("maillage", &mesh as &dyn ArchiveRoot)])?;

    let mut back = archive::load(&path)?;
    let mesh2 = back.mesh("maillage")?;
    let c2 = mesh2.get(0).unwrap().read().coords();

    // Live: the Node atom plus the submesh. Reloaded: the submesh alone.
    assert_eq!(coords.read().refcount(a.id()), 2);
    assert_eq!(c2.read().refcount(a.id()), 1);

    // And a bare Coords, saved with nothing that uses it, comes back with its
    // nodes at zero — a `gc()` will collect them, exactly as documented.
    let alone = tmp("bare_coords.pyr");
    archive::save(&alone, &[("c", &coords as &dyn ArchiveRoot)])?;
    let bare = archive::load(&alone)?.coords("c")?;
    assert_eq!(bare.read().refcount(a.id()), 0);
    assert_eq!(bare.write().gc(), 2, "both nodes are collectable");
    Ok(())
}

// ─── The rule: recomputable ⇒ not written ───────────────────────────────────

/// A nodal field weighs its values, not its values plus a copy of its support's
/// connectivity — and comes back with that copy rebuilt.
#[test]
fn what_is_recomputable_is_not_written() -> Result<()> {
    let n = 2000;
    let (_coords, mesh) = line(n)?;
    let poi = mesh.get(0).unwrap().read().to_poi1()?;
    let t = NodeField::from_sub(SubNodeField::from_poi1(&poi, vec!["T".into()])?);

    let with_mesh = tmp("field_and_mesh.pyr");
    let field_only = tmp("field_only.pyr");
    archive::save(&with_mesh, &[("m", &mesh as &dyn ArchiveRoot), ("T", &t)])?;
    archive::save(&field_only, &[("T", &t as &dyn ArchiveRoot)])?;

    // The field's own node list is not in the file: adding the mesh (which does
    // carry a connectivity) must cost roughly one connectivity, not two.
    let a = std::fs::metadata(&with_mesh).unwrap().len();
    let b = std::fs::metadata(&field_only).unwrap().len();
    let one_connectivity = (n as u64 + 1) * 4;
    assert!(
        a - b < 3 * one_connectivity,
        "adding the mesh cost {} bytes, about {} connectivities — the field is \
         carrying its node list after all",
        a - b,
        (a - b) / one_connectivity
    );

    // And it is back, correct, after the reload.
    let mut back = archive::load(&field_only)?;
    let t2 = back.node_field("T")?;
    assert_eq!(t2.get(0).unwrap().read().node_count(), n + 1);
    Ok(())
}

// ─── The format ─────────────────────────────────────────────────────────────

/// Saving the same objects twice gives the same bytes.
#[test]
fn the_file_is_reproducible() -> Result<()> {
    let (_coords, mesh) = line(4)?;
    let poi = mesh.get(0).unwrap().read().to_poi1()?;
    let t = NodeField::from_sub(SubNodeField::from_poi1(&poi, vec!["T".into()])?);

    let (a, b) = (tmp("repro_a.pyr"), tmp("repro_b.pyr"));
    for path in [&a, &b] {
        archive::save(
            path,
            &[
                ("zzz", &t as &dyn ArchiveRoot),
                ("aaa", &mesh),
                ("n", &4_i64),
            ],
        )?;
    }
    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "two saves of the same objects must be byte-identical"
    );
    Ok(())
}

/// An unknown format number is refused, naming both versions.
#[test]
fn an_unknown_format_is_refused() -> Result<()> {
    let (_coords, mesh) = line(1)?;
    let path = tmp("version.pyr");
    archive::save(&path, &[("m", &mesh as &dyn ArchiveRoot)])?;

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8] = 99;
    std::fs::write(&path, &bytes).unwrap();

    let e = archive::load(&path).unwrap_err().to_string();
    assert!(e.contains("99"), "should name the file's version: {e}");
    assert!(e.contains("format"), "should say what went wrong: {e}");
    Ok(())
}

/// A file that is not an archive is refused on its signature, not by a partial
/// decode.
#[test]
fn a_foreign_file_is_refused() {
    let path = tmp("foreign.pyr");
    std::fs::write(&path, b"this is not an archive at all").unwrap();
    let e = archive::load(&path).unwrap_err().to_string();
    assert!(e.contains("signature"), "{e}");
}

// ─── Simple values ──────────────────────────────────────────────────────────

#[test]
fn simple_values_travel_too() -> Result<()> {
    let path = tmp("values.pyr");
    archive::save(
        &path,
        &[
            ("actif", &true as &dyn ArchiveRoot),
            ("pas", &12_i64),
            ("dt", &0.05_f64),
            ("cas", &"charge répartie".to_string()),
            ("instants", &vec![0.0_f64, 0.1, 0.2]),
            ("noms", &vec!["a".to_string(), "b".to_string()]),
        ],
    )?;

    let back = archive::load(&path)?;
    assert!(back.bool("actif")?);
    assert_eq!(back.int("pas")?, 12);
    assert_eq!(back.float("dt")?, 0.05);
    assert_eq!(back.text("cas")?, "charge répartie");
    assert_eq!(back.floats("instants")?, vec![0.0, 0.1, 0.2]);
    assert_eq!(back.texts("noms")?, vec!["a", "b"]);
    Ok(())
}

/// The error names the key, what was expected and what was there.
#[test]
fn a_wrong_type_says_so() -> Result<()> {
    let path = tmp("wrong_type.pyr");
    archive::save(&path, &[("dt", &0.05_f64 as &dyn ArchiveRoot)])?;
    let mut back = archive::load(&path)?;

    let e = back.mesh("dt").unwrap_err().to_string();
    assert!(
        e.contains("dt") && e.contains("float") && e.contains("Mesh"),
        "{e}"
    );

    let e = back.float("absent").unwrap_err().to_string();
    assert!(
        e.contains("absent") && e.contains("dt"),
        "should list what is there: {e}"
    );
    Ok(())
}

// ─── The scope ──────────────────────────────────────────────────────────────

/// Outside an archive a handle has no meaning, and saying so is the job of an
/// error — not of a byte written at random.
#[test]
fn a_handle_alone_cannot_be_serialized() -> Result<()> {
    use pyrucast::archive::Portable;
    let (_coords, mesh) = line(1)?;
    let sm = mesh.get(0).unwrap();
    let e = sm.read().to_bytes().unwrap_err().to_string();
    assert!(e.contains("archive"), "{e}");
    Ok(())
}

// ─── A whole study ──────────────────────────────────────────────────────────

/// Mesh, FE space, model, material, load, stiffness — saved together, reloaded,
/// then reassembled and solved. The answer must be the one from before, to the
/// bit: what came back is the same problem, not a lookalike.
#[test]
fn a_whole_study_makes_the_round_trip() -> Result<()> {
    use pyrucast::containers::finite_element_space::FiniteElementSpace;
    use pyrucast::ops::solver::lu::solve;

    const K: f64 = 1.0;
    const Q: f64 = 10.0;
    const T_IMPOSED: f64 = 20.0;
    const N: usize = 4;
    let h = 1.0 / N as f64;

    let coords = Handle::new(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(
        nodes.last().unwrap(),
    ))?);
    let multiplier = pyrucast::ops::mesh::barycenter(&imposed)?;
    let mult = multiplier.node(0, 0, 0)?.id();
    let conduction = model::heat_conduction(&fes)?;
    let model = conduction.union(&model::dirichlet(
        &conduction,
        "T",
        &imposed,
        &multiplier,
        Default::default(),
    )?)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("k", K)])?;

    let node0 = nodes[0].id();
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[node0])?;
    load_sm.add_cell(&[mult])?;
    let load_sm = Handle::new(load_sm);
    let mut r = SubNodeField::from_poi1(&load_sm, vec!["imposed_T".into(), "q".into()])?;
    r.set_value(node0, "q", Q)?;
    r.set_value(mult, "imposed_T", T_IMPOSED)?;
    let rhs = NodeField::from_sub(r);

    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let before = solve(&stiffness, &rhs)?;

    // ── Everything into one file ───────────────────────────────────────────
    let path = tmp("study.pyr");
    archive::save(
        &path,
        &[
            ("maillage", &mesh as &dyn ArchiveRoot),
            ("espace", &fes),
            ("modele", &model),
            ("materiaux", &materials),
            ("chargement", &rhs),
            ("rigidite", &stiffness),
            ("solution", &before),
            ("k", &K),
        ],
    )?;

    let mut back = archive::load(&path)?;
    let model2 = back.model("modele")?;
    let materials2 = back.element_field("materiaux")?;
    let rhs2 = back.node_field("chargement")?;
    let solution2 = back.node_field("solution")?;
    assert_eq!(back.float("k")?, K);

    // Reassembling from the reloaded model must give the same answer. The CSR
    // and the factorization were not in the file — they rebuild themselves.
    let stiffness2 = pyrucast::ops::matrix::stiffness(&model2, &materials2)?;
    let after = solve(&stiffness2, &rhs2)?;

    for node in &nodes {
        assert_eq!(
            after.value(node.id(), "T")?,
            before.value(node.id(), "T")?,
            "reassembling the reloaded study must be bit-for-bit the same"
        );
        // And the solution field saved alongside came back untouched.
        assert_eq!(
            solution2.value(node.id(), "T")?,
            before.value(node.id(), "T")?
        );
    }
    assert_eq!(
        after.value(mult, "lambda_T")?,
        before.value(mult, "lambda_T")?
    );
    Ok(())
}

// ─── A cycle ────────────────────────────────────────────────────────────────

/// The written graph is acyclic only *because* caches are not written — nothing
/// in the type system enforces it. So the writer must refuse a cycle by naming
/// it, rather than recurse until the stack gives out.
#[test]
fn a_cycle_is_refused_by_name() -> Result<()> {
    use pyrucast::archive::Archivable;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Default)]
    struct Knot {
        next: Option<Handle<Knot>>,
    }
    impl Archivable for Knot {
        const TAG: &'static str = "Knot";
    }

    let knot = Handle::new(Knot::default());
    knot.write().next = Some(knot.clone()); // tie it to itself

    let path = tmp("cycle.pyr");
    let e = archive::save(&path, &[("noeud", &knot as &dyn ArchiveRoot)])
        .unwrap_err()
        .to_string();
    assert!(e.contains("cycle"), "should say what is wrong: {e}");
    assert!(e.contains("Knot"), "should name the object: {e}");
    Ok(())
}
