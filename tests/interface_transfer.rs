//! Exchange law across an interface — the first inter-mesh coupling.
//!
//! Two unit squares laid side by side, `[0,1]×[0,1]` and `[1,2]×[0,1]`, that do
//! **not** share their junction: at `x = 1` each carries its own pair of nodes,
//! and the two bodies are tied only by the exchange law `j·n = h(c₁ − c₂)`.
//!
//! A uniform flux density `q` enters at `x = 0` and the concentration is imposed
//! at `x = 2`. In steady state the same `q` crosses everything, so the profile
//! is piecewise linear with a **jump** at the interface:
//!
//! ```text
//! drop across each square : q/D          jump across the interface : q/h
//! ```
//!
//! The jump is the whole point. It is what an interface law adds over a shared
//! node, and it is carried entirely by the two **off-diagonal** blocks — those
//! whose rows live on one mesh and whose columns live on the other, which is
//! what `Contribution::Coupling` exists for. A stiff interface (`h → ∞`)
//! recovers the continuous solution, which the second test checks.
//!
//! Single source for the « transfert d'interface » example of the diffusion book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::Physics;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

/// The diffusing species — every name of the Fick physics carries it.
const SPECIES: &str = "H2";

const D: f64 = 2.0; // diffusivity, both squares
const Q: f64 = 10.0; // flux density injected at x = 0
const C_RIGHT: f64 = 1.0; // concentration imposed at x = 2

#[test]
fn an_interface_law_makes_the_field_jump() -> Result<()> {
    const H: f64 = 5.0; // transfer coefficient
    let (geom, solution) = solve_two_squares(H)?;

    let c = |n: &Node| solution.value(n.id(), &format!("c_{SPECIES}"));
    let far_left = c(&geom.left[0])?; // (0, 0)
    let left_face = c(&geom.left[1])?; // (1, 0), left side
    let right_face = c(&geom.right[0])?; // (1, 0), right side
    let far_right = c(&geom.right[1])?; // (2, 0)

    let tol = 1e-10;
    assert!((far_right - C_RIGHT).abs() < tol, "c(2) = {far_right}");
    // Slope q/D over the unit width of each square.
    assert!(
        (far_left - left_face - Q / D).abs() < tol,
        "left square: {far_left} → {left_face}"
    );
    assert!(
        (right_face - far_right - Q / D).abs() < tol,
        "right square: {right_face} → {far_right}"
    );
    // …and the jump across the interface is q/h — the exchange law itself.
    let jump = left_face - right_face;
    assert!(
        (jump - Q / H).abs() < tol,
        "jump = {jump}, expected {}",
        Q / H
    );
    Ok(())
}
// ANCHOR_END: example

/// A stiff interface (`h → ∞`) must recover the single continuous body: the jump
/// vanishes as `q/h`, so the two facing values converge to each other.
/// L'interface rend son terme de résidu **des deux côtés**, égaux et opposés.
///
/// `∫h(a₁−a₂)N` sur A et son opposé sur B : l'intégrale d'une différence est la
/// différence des intégrales, et chacune se disperse sur son propre espace. Ce
/// qui est couplé, c'est la matrice — lignes sur A, colonnes sur B — pas le
/// vecteur, qui ne produit qu'un nombre par nœud. L'interface n'a donc plus de
/// loi : son `h·(a₁−a₂)` est le coefficient de son propre opérateur appliqué au
/// saut, pas un comportement.
/// Le résidu de l'interface **est** sa matrice appliquée à la solution.
///
/// L'opérateur est linéaire : `r = K·u` n'est pas une approximation mais une
/// identité, et c'est le seul test qui fixe les *valeurs*. L'antisymétrie, elle,
/// survit à une erreur partagée par les deux côtés — les deux contributions
/// naissent du même saut, au signe près. C'est ce qu'il a fallu pour voir que le
/// saut se lisait par position dans un champ interpolé qui porte **toutes** les
/// composantes de la solution, multiplicateur de Dirichlet compris.
#[test]
fn the_interface_residual_is_its_matrix_applied_to_the_solution() -> Result<()> {
    const H: f64 = 5.0;
    let (geom, model, materials) = two_square_model(H)?;
    let (_, solution) = solve_two_squares(H)?;

    let interface = {
        let mut m = Model::empty();
        for h in &model {
            if h.read().as_kind().label() == "InterfaceTransfer" {
                m.add_sub(h.clone())?;
            }
        }
        m
    };
    let state = ElementField::empty();
    let r = pyrucast::ops::node_field::internal_forces(&interface, &state, &solution, &materials)?;
    let k = pyrucast::ops::matrix::stiffness(&interface, &materials)?;
    let ku = (&k * &solution)?;

    let dual = format!("j_{SPECIES}");
    let mut vus = 0;
    for n in [&geom.left[1], &geom.left[2], &geom.right[0], &geom.right[3]] {
        let got = r.value(n.id(), &dual)?;
        let want = ku.value(n.id(), &dual)?;
        assert!(
            (got - want).abs() < 1e-9 * want.abs().max(1.0),
            "r ≠ K·u en {:?} : {got} vs {want}",
            n.id()
        );
        vus += 1;
    }
    assert_eq!(vus, 4);
    Ok(())
}

#[test]
fn the_interface_renders_equal_and_opposite_fluxes() -> Result<()> {
    const H: f64 = 5.0;
    let (geom, model, materials) = two_square_model(H)?;
    let (_, solution) = solve_two_squares(H)?;

    // Le résidu de l'interface seule : on pointe le sous-modèle qu'on veut.
    let interface = {
        let mut m = Model::empty();
        for h in &model {
            if h.read().as_kind().label() == "InterfaceTransfer" {
                m.add_sub(h.clone())?;
            }
        }
        m
    };
    let state = pyrucast::containers::element_field::ElementField::empty();
    let f = pyrucast::ops::node_field::internal_forces(&interface, &state, &solution, &materials)?;

    let dual = format!("j_{SPECIES}");
    // Somme sur chaque côté de l'interface : les deux faces qui se regardent.
    let cote_a: f64 = [&geom.left[1], &geom.left[2]]
        .iter()
        .map(|n| f.value(n.id(), &dual).unwrap_or(0.0))
        .sum();
    let cote_b: f64 = [&geom.right[0], &geom.right[3]]
        .iter()
        .map(|n| f.value(n.id(), &dual).unwrap_or(0.0))
        .sum();

    assert!(cote_a.abs() > 1e-9, "l'interface doit transporter un flux");
    assert!(
        (cote_a + cote_b).abs() < 1e-9 * cote_a.abs().max(1.0),
        "les deux côtés doivent s'annuler : {cote_a} et {cote_b}"
    );
    Ok(())
}

#[test]
fn a_stiff_interface_recovers_a_continuous_body() -> Result<()> {
    let mut previous = f64::INFINITY;
    for h in [1e2, 1e4, 1e6] {
        let (geom, solution) = solve_two_squares(h)?;
        let jump = solution.value(geom.left[1].id(), &format!("c_{SPECIES}"))?
            - solution.value(geom.right[0].id(), &format!("c_{SPECIES}"))?;
        assert!(
            (jump - Q / h).abs() < 1e-8 * jump.abs().max(1.0),
            "h = {h}: jump {jump} ≠ {}",
            Q / h
        );
        assert!(
            jump < previous,
            "the jump must shrink with h: {jump} ≥ {previous}"
        );
        previous = jump;
    }
    assert!(previous < 1e-4, "residual jump {previous}");
    Ok(())
}

/// The exchange term is symmetric overall (`+K −K / −K +K`), so the assembled
/// stiffness stays symmetric even though **each** coupling block alone is not.
/// That is the structural check that the four blocks land where they should.
#[test]
fn the_four_blocks_sum_to_a_symmetric_operator() -> Result<()> {
    let (_, model, materials) = two_square_model(3.0)?;
    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let dense = k.dense()?;
    let n = k.row_dofs()?.len();
    assert_eq!(dense.len(), n * n);
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (dense[i * n + j], dense[j * n + i]);
            assert!((a - b).abs() < 1e-12, "K[{i},{j}] = {a} ≠ {b} = K[{j},{i}]");
        }
    }
    // The coupling really is there: some entry links a left-side DOF to a
    // right-side one, which a pair of independent bodies could never produce.
    let coupling = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .any(|(i, j)| i != j && dense[i * n + j].abs() > 1e-12);
    assert!(coupling, "the interface must couple the two meshes");
    Ok(())
}

/// A non-conforming interface is a meshing problem, and is reported as one
/// rather than resolved by a silent projection.
#[test]
fn a_non_conforming_interface_is_rejected() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let node = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]);
    let edge = |a: &Node, b: &Node| -> Result<FiniteElementSpace> {
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        m.add_cell(&[a.id(), b.id()])?;
        FiniteElementSpace::lagrange1(&m)
    };
    let (a0, a1) = (node(1.0, 0.0)?, node(1.0, 1.0)?);
    // The facing edge sits at x = 1.5: the two sides do not describe one surface.
    let (b0, b1) = (node(1.5, 0.0)?, node(1.5, 1.0)?);
    let err = model::interface_transfer(
        &edge(&a0, &a1)?,
        &edge(&b0, &b1)?,
        vec![(format!("c_{SPECIES}"), format!("j_{SPECIES}"))],
        Physics::Diffusion,
        1e-9,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not node-conforming"), "unexpected: {msg}");
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The two squares' nodes, `[bottom-left, bottom-right, top-right, top-left]`.
struct Geometry {
    left: Vec<Node>,
    right: Vec<Node>,
}

/// Two QUA4 squares with **duplicated** nodes at `x = 1`, their diffusion models,
/// and the interface law tying them.
fn two_square_model(h: f64) -> Result<(Geometry, Model, ElementField)> {
    let coords = Handle::new(Coords::new(2)?);
    let node = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]);

    // Left square [0,1]², right square [1,2]×[0,1]. The two pairs at x = 1 are
    // distinct node objects at the same position: that is what lets `c` jump.
    let left = vec![
        node(0.0, 0.0)?,
        node(1.0, 0.0)?,
        node(1.0, 1.0)?,
        node(0.0, 1.0)?,
    ];
    let right = vec![
        node(1.0, 0.0)?,
        node(2.0, 0.0)?,
        node(2.0, 1.0)?,
        node(1.0, 1.0)?,
    ];

    let square = |ns: &[Node]| -> Result<FiniteElementSpace> {
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        m.add_cell(&ns.iter().map(|n| n.id()).collect::<Vec<_>>())?;
        FiniteElementSpace::lagrange1(&m)
    };
    let edge = |a: &Node, b: &Node| -> Result<FiniteElementSpace> {
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        m.add_cell(&[a.id(), b.id()])?;
        FiniteElementSpace::lagrange1(&m)
    };

    // The interface: the x = 1 edge of each square, node for node
    // (bottom → top on both sides, so local node k faces local node k).
    let face_left = edge(&left[1], &left[2])?;
    let face_right = edge(&right[0], &right[3])?;

    // Concentration imposed on the far-right edge (x = 2).
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(&[
        right[1].clone(),
        right[2].clone(),
    ])?);
    let multiplier = mesh::barycenter(&imposed)?;

    // La diffusion des deux carrés : c'est elle que l'appui contraint.
    let fick_pair =
        model::fick(&square(&left)?, SPECIES)?.union(&model::fick(&square(&right)?, SPECIES)?)?;
    let model = fick_pair
        .union(&model::interface_transfer(
            &face_left,
            &face_right,
            vec![(format!("c_{SPECIES}"), format!("j_{SPECIES}"))],
            Physics::Diffusion,
            1e-9,
        )?)?
        .union(&model::dirichlet(
            &fick_pair,
            &format!("c_{SPECIES}"),
            &imposed,
            &multiplier,
            Default::default(),
        )?)?;

    // The inlet: uniform flux density Q on the far-left edge (x = 0). A load
    // is a term of the model, so it joins it here and its density joins the
    // material below.
    let inlet = edge(&left[0], &left[3])?;
    let model = model.union(&model::flux(
        &inlet,
        format!("j_{SPECIES}"),
        Physics::Diffusion,
    )?)?;

    // One material field for the whole model: the squares ask for `D`, the
    // interface for `h`, the inlet for its density, and each resolves its own
    // zone by its components.
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            (&format!("D_{SPECIES}"), D),
            (&format!("h_c_{SPECIES}"), h),
            (&format!("phi_j_{SPECIES}"), Q),
        ],
    )?;
    Ok((Geometry { left, right }, model, materials))
}

/// Build, load and solve the two-square problem for a given transfer coefficient.
fn solve_two_squares(h: f64) -> Result<(Geometry, NodeField)> {
    let (geom, model, materials) = two_square_model(h)?;
    let coords = geom.left[0].coords();

    // The inlet's consistent nodal loads: the model carries the term, we ask
    // it for its value.
    let influx = pyrucast::ops::node_field::external_forces(&model, &materials)?;

    // The imposed concentration, on the Dirichlet multiplier nodes.
    let mult_mesh = model.multiplier_mesh()?;
    let mut imposed_sm = SubMesh::new(coords, ElementType::POI1);
    let mut mults = Vec::new();
    for i in 0..mult_mesh.len() {
        let cells = mult_mesh.get(i)?.read().cell_count();
        for cell in 0..cells {
            let id = mult_mesh.node(i, cell, 0)?.id();
            imposed_sm.add_cell(&[id])?;
            mults.push(id);
        }
    }
    let imposed_sm = Handle::new(imposed_sm);
    let mut imposed = SubNodeField::from_poi1(&imposed_sm, vec![format!("imposed_c_{SPECIES}")])?;
    for id in &mults {
        imposed.set_value(*id, &format!("imposed_c_{SPECIES}"), C_RIGHT)?;
    }

    let rhs = influx.union(&NodeField::from_sub(imposed))?;
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;
    Ok((geom, solution))
}
