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
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::model;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

#[test]
fn thermal_convection_recovers_analytical_solution() -> Result<()> {
    // ── Problem data ───────────────────────────────────────────────────────
    const K: f64 = 2.0; // conductivité
    const Q: f64 = 10.0; // densité de flux injectée sur le bord gauche
    const H: f64 = 5.0; // coefficient d'échange (film) sur le bord droit
    const T_EXT: f64 = 20.0; // température ambiante du fluide
    const N: usize = 4; // N×N éléments QUA4
    let step = 1.0 / N as f64;

    // ── Mesh: a structured (N+1)×(N+1) grid of QUA4 on [0,1]² ──────────────
    let coords = Handle::new(Coords::new(2)?);
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
    // it integrates as a line (film matrix h ∫ N_i N_j dΓ).
    let mut right_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        right_edge.add_cell(&[grid[idx(N, j)].id(), grid[idx(N, j + 1)].id()])?;
    }
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;

    let conduction = model::heat_conduction(&fes)?;
    let convection =
        model::boundary_transfer(&right_fes, &conduction, vec![("T".into(), "q".into())])?;
    let model = conduction.union(&convection)?;

    // Material: k for the conduction, h and the ambient for the convection (each
    // sub-model takes the component it requires from the supplied list).
    // ── Chargement ─────────────────────────────────────────────────────────
    // Source: uniform flux (density Q) on the left edge, as nodal loads
    // cohérentes via `flux`.
    let mut left_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        left_edge.add_cell(&[grid[idx(0, j)].id(), grid[idx(0, j + 1)].id()])?;
    }
    let left_fes = FiniteElementSpace::lagrange1(&left_edge)?;
    let model = model.union(&model::flux(&left_fes, &model, "q".into())?)?;

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("k", K), ("h_T", H), ("a_ext_T", T_EXT), ("phi_q", Q)],
    )?;

    // Both given terms — the left edge's source and the convection's external
    // part h·T_ext — belong to the model, which returns them together. Nothing
    // left to union by hand, hence nothing left to forget.
    let rhs = pyrucast::ops::node_field::external_forces(&model, &materials)?;

    // ── Assembly + solve (K made definite by the film term) ────────────────
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Compared with the analytical T(x) = T_ext + Q/h + (Q/k)(1 − x), ∀ y ─
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

    // Energy balance: all the injected flux leaves by convection, so the right
    // edge's temperature is exactly T_ext + Q/h.
    let t_right = solution.value(grid[idx(N, 0)].id(), "T")?;
    assert!(
        (t_right - (T_EXT + Q / H)).abs() < tol,
        "T(x=1) : obtenu {t_right}, attendu {}",
        T_EXT + Q / H
    );

    Ok(())
}
// ANCHOR_END: convection
