//! Worked thermal example with a **convection (Robin / film) boundary**,
//! exercised end-to-end through the public API.
//!
//! Steady conduction on the unit square `[0,1]²` (structured `N×N` QUA4 grid):
//! a heat source (uniform Neumann flux, density `Q`) on the **left** edge
//! `x = 0`, a **convection** exchange `q·n = h·(T − T_ext)` on the **right**
//! edge `x = 1`, and **insulated** top/bottom edges (the natural BC).
//!
//! No Dirichlet is needed: the film term grounds the otherwise-floating
//! temperature (pure-Neumann conduction is singular; the convection matrix
//! `h ∫ N_i N_j dΓ` restores definiteness). With the lateral edges insulated
//! the field is independent of `y` and reduces to the 1-D balance
//!
//! ```text
//! T(x) = T_ext + Q/h + (Q/k)·(1 − x),
//! ```
//!
//! all the injected heat `Q` leaving by convection at `x = 1`
//! (`h·(T(1) − T_ext) = Q`). The convection matrix goes into the stiffness (a
//! `Convection` sub-model coupling on the shared `"T"`/`"q"` DOFs); the
//! external-temperature load `h·T_ext·∫N_i dΓ` is a right-hand side built with
//! the same `flux` operator as the source — no normal is chosen (the surface
//! measure is orientation-independent).
//!
//! Single source for the « convection » example of the book chapter
//! *« Conduction thermique »*; runs under `cargo test`.

// ANCHOR: convection
use pyrucast::aggregate::Aggregate;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{assemble, build};
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn thermal_convection_recovers_analytical_solution() -> Result<()> {
    // ── Données du problème ────────────────────────────────────────────────
    const K: f64 = 2.0; // conductivité
    const Q: f64 = 10.0; // densité de flux injectée sur le bord gauche
    const H: f64 = 5.0; // coefficient d'échange (film) sur le bord droit
    const T_EXT: f64 = 20.0; // température ambiante du fluide
    const N: usize = 4; // N×N éléments QUA4
    let step = 1.0 / N as f64;

    // ── Maillage : grille structurée (N+1)×(N+1) de QUA4 sur [0,1]² ─────────
    let coords = insert(Coords::new(2)?);
    let idx = |i: usize, j: usize| j * (N + 1) + i; // nœud colonne i, ligne j
    let mut grid: Vec<Node> = Vec::with_capacity((N + 1) * (N + 1));
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * step, j as f64 * step],
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

    // ── Modèle : conduction (volume) + convection (bord droit x = 1) ───────
    // Le bord droit est un maillage SEG2 bâti sur les nœuds de la grille ;
    // il s'intègre comme une ligne (matrice de film h ∫ N_i N_j dΓ).
    let mut right_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        right_edge.add_cell(&[grid[idx(N, j)].id(), grid[idx(N, j + 1)].id()])?;
    }
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;

    let conduction = Model::heat_conduction(&fes)?;
    let convection = Model::convection(&right_fes)?;
    let model = conduction.union(&convection)?;

    // Matériau : k pour la conduction, h pour la convection (chaque sous-modèle
    // prélève la composante qu'il requiert dans la liste fournie).
    let materials = build::material_field(&model, &[("k", K), ("h", H)])?;

    // ── Chargement ─────────────────────────────────────────────────────────
    // Source : flux uniforme (densité Q) sur le bord gauche, en charges nodales
    // cohérentes via `flux`.
    let mut left_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        left_edge.add_cell(&[grid[idx(0, j)].id(), grid[idx(0, j + 1)].id()])?;
    }
    let left_fes = FiniteElementSpace::lagrange1(&left_edge)?;
    let source = assemble::flux(&left_fes.get(0)?, assemble::FluxDensity::Uniform(Q), "q")?;

    // Convection : la part externe h·T_ext du flux de Robin est un second membre,
    // bâti avec le MÊME opérateur `flux` (densité h·T_ext) — aucune normale.
    let conv_load =
        assemble::flux(&right_fes.get(0)?, assemble::FluxDensity::Uniform(H * T_EXT), "q")?;

    let rhs = NodeField::from_sub(source).union(&NodeField::from_sub(conv_load))?;

    // ── Assemblage + résolution (K rendue définie par le terme de film) ────
    let stiffness = assemble::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison à l'analytique T(x) = T_ext + Q/h + (Q/k)(1 − x), ∀ y ──
    let tol = 1e-9;
    for j in 0..=N {
        for i in 0..=N {
            let x = i as f64 * step;
            let expected = T_EXT + Q / H + (Q / K) * (1.0 - x);
            let got = solution.value(grid[idx(i, j)].id(), "T")?;
            assert!(
                (got - expected).abs() < tol,
                "T(x={x}, y={}) : obtenu {got}, attendu {expected}",
                j as f64 * step
            );
        }
    }

    // Bilan d'énergie : tout le flux injecté ressort par convection, donc la
    // température du bord droit vaut exactement T_ext + Q/h.
    let t_right = solution.value(grid[idx(N, 0)].id(), "T")?;
    assert!(
        (t_right - (T_EXT + Q / H)).abs() < tol,
        "T(x=1) : obtenu {t_right}, attendu {}",
        T_EXT + Q / H
    );

    Ok(())
}
// ANCHOR_END: convection
