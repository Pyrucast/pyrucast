//! Worked thermal example on a **square**, exercised end-to-end through the
//! public API.
//!
//! Steady conduction on the unit square `[0,1]²` (structured `N×N` QUA4 grid):
//! a heat source (uniform Neumann flux, total `Q`) on the **left** edge
//! `x = 0`, an imposed temperature `T = 20` (Dirichlet) on the **right** edge
//! `x = 1`, and **insulated** top/bottom edges (the natural, do-nothing BC).
//!
//! With the lateral edges insulated the field is independent of `y` and the
//! square reduces to the 1-D line: `u(x) = 20 + (Q/k)·(1 − x)`. The **total**
//! reaction (sum of the Lagrange multipliers over the imposed edge) equals the
//! injected flux `Q`.
//!
//! Teaching point: a *distributed* edge flux is applied as **consistent nodal
//! loads** — for a uniform flux on linear elements, `Q·h` at interior edge
//! nodes and `Q·h/2` at the two corners (they sum to `Q`).
//!
//! Single source for the « carré » example of the book chapter
//! *« Conduction thermique »*; runs under `cargo test`.

// ANCHOR: square
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::Physics;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

#[test]
fn thermal_square_recovers_analytical_solution() -> Result<()> {
    // ── Données du problème ────────────────────────────────────────────────
    const K: f64 = 1.0; // conductivité
    const Q: f64 = 10.0; // flux de chaleur TOTAL injecté sur le bord gauche
    const T_IMPOSED: f64 = 20.0; // température imposée sur le bord droit
    const N: usize = 4; // N×N éléments QUA4
    let h = 1.0 / N as f64;

    // ── Maillage : grille structurée (N+1)×(N+1) de QUA4 sur [0,1]² ─────────
    let coords = Handle::new(Coords::new(2)?);
    let idx = |i: usize, j: usize| j * (N + 1) + i; // nœud colonne i, ligne j
    let mut grid: Vec<Node> = Vec::with_capacity((N + 1) * (N + 1));
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * h, j as f64 * h],
            )?);
        }
    }
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..N {
        for i in 0..N {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Dirichlet T = 20 sur le bord droit (x = 1) ─────────────────────────
    let right_nodes: Vec<Node> = (0..=N).map(|j| grid[idx(N, j)].clone()).collect();
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(&right_nodes)?);
    let multiplier = mesh::barycenter(&imposed)?;
    let mults: Vec<Node> = (0..=N)
        .map(|j| multiplier.node(0, j, 0))
        .collect::<Result<_>>()?;

    let conduction = model::heat_conduction(&fes)?;
    let dirichlet = model::dirichlet(&conduction, "T", &imposed, &multiplier, Default::default())?;
    let model = conduction.union(&dirichlet)?;
    // ── Chargement ─────────────────────────────────────────────────────────
    // Source : flux uniforme (densité Q) sur le bord gauche, transformé en
    // charges nodales cohérentes par l'opérateur `flux` (Cast3m FLUX) — plus de
    // répartition Q·h / Q·h/2 à la main. Le bord est un maillage SEG2 bâti sur
    // les nœuds de la grille ; il s'intègre comme une ligne.
    let mut left_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        left_edge.add_cell(&[grid[idx(0, j)].id(), grid[idx(0, j + 1)].id()])?;
    }
    let left_fes = FiniteElementSpace::lagrange1(&left_edge)?;
    let model = model.union(&model::flux(&left_fes, "q".into(), Physics::Thermal)?)?;

    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("k", K), ("phi_q", Q)])?;
    let source = pyrucast::ops::node_field::external_forces(&model, &materials)?;

    // Valeur imposée T = 20 au slot "imposed_T" des nœuds-multiplicateurs.
    let mut imposed_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    for m in &mults {
        imposed_sm.add_cell(&[m.id()])?;
    }
    let imposed_sm = Handle::new(imposed_sm);
    let mut imposed_load = SubNodeField::from_poi1(&imposed_sm, vec!["imposed_T".into()])?;
    for m in &mults {
        imposed_load.set_value(m.id(), "imposed_T", T_IMPOSED)?;
    }

    // Chargement = flux du bord + valeurs imposées (union des zones).
    let rhs = source.union(&NodeField::from_sub(imposed_load))?;

    // ── Assemblage + résolution ────────────────────────────────────────────
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison à l'analytique u(x) = 20 + (Q/k)(1 − x), ∀ y ───────────
    let tol = 1e-9;
    for j in 0..=N {
        for i in 0..=N {
            let x = i as f64 * h;
            let expected = T_IMPOSED + (Q / K) * (1.0 - x);
            let got = solution.value(grid[idx(i, j)].id(), "T")?;
            assert!(
                (got - expected).abs() < tol,
                "T(x={x}, y={}) : obtenu {got}, attendu {expected}",
                j as f64 * h
            );
        }
    }
    // La réaction totale sur le bord imposé équilibre le flux injecté : Σλ = Q.
    let total_reaction: f64 = mults
        .iter()
        .map(|m| solution.value(m.id(), "lambda_T"))
        .sum::<Result<f64>>()?;
    assert!(
        (total_reaction - Q).abs() < tol,
        "réaction totale : obtenue {total_reaction}, attendue {Q}"
    );

    Ok(())
}
// ANCHOR_END: square
