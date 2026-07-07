//! Poutre console élasto-plastique — Newton « maison » au-dessus des briques
//! pyrucast (exemple Rust, pensé pour le bench de parallélisme).
//!
//! Physique
//! --------
//! Continuum 2-D en contraintes planes, petites déformations. Plasticité de
//! von Mises parfaite (retour radial J2, sans écrouissage) : la contrainte
//! équivalente est plafonnée à `sigma_y`. Poutre encastrée à gauche
//! (`u_x = u_y = 0`), cisaillée vers le bas sur la face droite. On monte la
//! charge par incréments ; au-delà de la première plastification, une zone
//! plastique se développe près de l'encastrement et la flèche s'écarte de la
//! réponse linéaire.
//!
//! Rôle de pyrucast vs. rôle de l'exemple
//! --------------------------------------
//! **pyrucast** ne connaît PAS Newton. Il fournit les opérateurs ponctuels :
//!
//! - [`stiffness`] : la rigidité **élastique** `K` (opérateur d'itération) ;
//! - [`deformation`] : la déformation `ε = ½(∇u + ∇uᵀ)` aux points de Gauss ;
//! - [`integrate`] (Cast3m `COMP`) : la loi de comportement au point — ici le
//!   retour radial, qui rend `σ` et l'état plastique mis à jour
//!   (`VAR0` → `VAR1`) ;
//! - [`internal_forces`] (Cast3m `BSIG`) : les forces internes `∫ Bᵀ σ dΩ` ;
//! - [`solve`] : la résolution linéaire (LU creux faer, factorisation en cache) ;
//! - l'**arithmétique de champs** (`+ - * /`) et [`restrict_like`] (reprojection
//!   d'un champ sur le support/composantes d'un autre), qui remplacent toute
//!   boucle nodale : `residual = &f_ext - &f_int`, `u = (&u + &δu_reprojeté)?` ;
//! - une [`Evolution`] à valeur champ pour l'**histoire de chargement** :
//!   la charge de chaque pas est interpolée au pseudo-temps ([`Evolution::interpolate`]).
//!
//! **L'exemple** assemble sa propre boucle de Newton avec ces briques :
//! résidu `r = F_ext − F_int`, incrément `δu = K⁻¹ r`, `u ← u + δu`, et le
//! portage de l'état interne d'un pas de charge au suivant. C'est un **Newton
//! modifié** (opérateur constant = `K` élastique) : `K` est assemblé et
//! factorisé une seule fois, chaque itération ne refait qu'une descente/remontée
//! (cache de factorisation de [`solve`]). Aucune boucle sur les nœuds : tout le
//! bilan passe par les opérateurs de champ et les primitives de la librairie.
//!
//! Bench de parallélisme
//! ---------------------
//! Les boucles chaudes parallélisées (assemblage, `deformation`, `integrate`,
//! `internal_forces`) sont réévaluées à chaque itération de Newton. Fais varier
//! la taille du maillage et le nombre de threads :
//!
//! ```text
//! RAYON_NUM_THREADS=1 PYRUCAST_NX=200 PYRUCAST_NY=40 \
//!     cargo run --release --example plasticite_poutre_console
//! RAYON_NUM_THREADS=8 PYRUCAST_NX=200 PYRUCAST_NY=40 \
//!     cargo run --release --example plasticite_poutre_console
//! ```
//!
//! Variables d'environnement : `PYRUCAST_NX`, `PYRUCAST_NY` (mailles en long /
//! en hauteur), `PYRUCAST_NSTEPS` (pas de charge), `PYRUCAST_PMAX` (charge
//! finale, effort tranchant au bout).

use std::collections::HashSet;

use pyrucast::aggregate::Aggregate;
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::evolution::{
    Evolution, Interpolated, OutOfRange, SubEvolution, SubValue,
};
use pyrucast::containers::field::{Field, SubField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::assemble::{flux, stiffness, FluxDensity};
use pyrucast::ops::behavior::integrate;
use pyrucast::ops::build::material_field;
use pyrucast::ops::field::{deformation, mask_cells, restrict, restrict_like, Band};
use pyrucast::ops::internal_forces::internal_forces;
use pyrucast::ops::mesher::barycenter;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::{insert, read, write};
use pyrucast::Result;

/// Composantes de l'état interne plastique portées d'un pas au suivant
/// (`VAR`) : déformation plastique 3-D (tenseur, 6) + déformation plastique
/// cumulée `p`. Mêmes noms que la sortie de la loi (`models::plasticity`).
const STATE_COMPONENTS: [&str; 7] = [
    "eps_p_xx", "eps_p_yy", "eps_p_zz", "eps_p_yz", "eps_p_xz", "eps_p_xy", "p",
];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    // ── Paramètres (matériau acier, géométrie, chargement) ──────────────────
    let (young, nu, sigma_y) = (210_000.0_f64, 0.3_f64, 250.0_f64);
    let (length, height) = (10.0_f64, 1.0_f64);
    let nx = env_usize("PYRUCAST_NX", 40);
    let ny = env_usize("PYRUCAST_NY", 8);
    let nsteps = env_usize("PYRUCAST_NSTEPS", 10);
    let p_max = env_f64("PYRUCAST_PMAX", 5.0); // effort tranchant final au bout

    println!(
        "Poutre console plastique : {nx}×{ny} QUA4  (L={length}, H={height}), \
         E={young}, ν={nu}, σy={sigma_y}"
    );
    println!("Chargement : 0 → {p_max} en {nsteps} pas (Newton modifié, K élastique)\n");

    // ── Maillage : grille de nœuds (j en hauteur, i en long), cellules QUA4 ──
    let coords = insert(Coords::new(2)?);
    // grid[j][i] : nœud à (x = i·L/nx, y = j·H/ny).
    let mut grid: Vec<Vec<Node>> = Vec::with_capacity(ny + 1);
    for j in 0..=ny {
        let mut row = Vec::with_capacity(nx + 1);
        for i in 0..=nx {
            let x = i as f64 * length / nx as f64;
            let y = j as f64 * height / ny as f64;
            row.push(Node::create_in(coords.clone(), &[x, y])?);
        }
        grid.push(row);
    }

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..ny {
        for i in 0..nx {
            mesh.add_cell(&[
                grid[j][i].id(),
                grid[j][i + 1].id(),
                grid[j + 1][i + 1].id(),
                grid[j + 1][i].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Ensembles de nœuds utiles : bord gauche (encastré), bout (mi-hauteur), et
    // un maillage POI1 des nœuds LIBRES (hors encastrement) — support cible pour
    // mesurer la norme du résidu sur les seuls DDL libres (`restrict` + `xtx`).
    let left_nodes: Vec<Node> = grid.iter().map(|row| row[0].clone()).collect();
    let left_ids: HashSet<NodeId> = left_nodes.iter().map(Node::id).collect();
    let tip_id = grid[ny / 2][nx].id();
    let free_nodes: Vec<Node> = grid
        .iter()
        .flatten()
        .filter(|n| !left_ids.contains(&n.id()))
        .cloned()
        .collect();
    let free_mesh = Mesh::from_submesh(SubMesh::poi1_from_nodes(&free_nodes)?);

    // ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    let mut model = Model::plasticity(&fes, ElasticityModel::PlaneStress)?;
    model = model.union(&clamp(&left_nodes, "u_x", "f_x")?)?;
    model = model.union(&clamp(&left_nodes, "u_y", "f_y")?)?;

    let materials = material_field(&model, &[("E", young), ("nu", nu), ("sigma_y", sigma_y)])?;

    // Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée
    // une fois ; `solve` met la factorisation en cache et la réutilise à chaque
    // descente/remontée.
    let k = stiffness(&model, &materials)?;

    // ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    //    droite, réparti en efforts nodaux cohérents (∫ densité·N dΓ, op `flux`).
    let mut right_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..ny {
        right_edge.add_cell(&[grid[j][nx].id(), grid[j + 1][nx].id()])?;
    }
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    let load_unit = flux(&right_fes.get(0)?, FluxDensity::Uniform(-1.0), "f_y")?;

    // ── Histoire de chargement : une Evolution à valeur CHAMP, tabulée en
    //    pseudo-temps t ∈ [0, 1]. Deux keyframes du champ d'effort nodal — nul en
    //    t=0, complet (`p_max · charge_unitaire`) en t=1 — sur le MÊME support
    //    (dérivés du même sous-champ, condition de l'interpolation). La charge de
    //    chaque pas est lue par interpolation linéaire, `load_evo.interpolate(t)`.
    //    Une histoire non linéaire n'ajouterait que des keyframes. ──────────────
    let zero_frame = load_unit.map_all(|_| 0.0);
    let full_frame = load_unit.map_all(|v| v * p_max);
    let load_curve = SubEvolution::new(
        vec![
            (0.0, SubValue::Node(zero_frame)),
            (1.0, SubValue::Node(full_frame)),
        ],
        OutOfRange::Clamp,
    )?;
    let mut load_evo = Evolution::default();
    load_evo.add_sub(insert(load_curve))?;

    // ── État de la simulation (persistant entre les pas) ────────────────────
    // Déplacement cumulé u (u_x, u_y sur tous les nœuds), initialement nul.
    let mut u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
    // État plastique VAR0 (nul au premier pas — la loi défaute à zéro).
    let mut state = ElementField::new(
        &fes,
        STATE_COMPONENTS.iter().map(|s| s.to_string()).collect(),
    )?;

    // ── Boucle sur les pas de charge ────────────────────────────────────────
    // Newton modifié (opérateur = K élastique) : convergence linéaire, donc
    // lente sur la branche plastique. On plafonne haut les itérations et on
    // vise un résidu relatif de 1e-6 (largement suffisant ici).
    let max_newton = 200;
    println!(
        "{:>4} {:>8} {:>6} {:>14} {:>14} {:>8}",
        "pas", "P", "iter", "flèche u_y", "p_max", "n_plast"
    );

    let mut prev_defl = 0.0_f64;
    let mut any_plasticity = false;

    for step in 1..=nsteps {
        // Pseudo-temps du pas ∈ ]0, 1] ; la charge externe en découle par
        // interpolation de l'Evolution (champ d'effort nodal du pas).
        let t = step as f64 / nsteps as f64;
        let load_p = p_max * t; // cisaillement nominal au bout (pour l'affichage)
        let Interpolated::Node(load_scaled) = load_evo.interpolate(t, None)? else {
            unreachable!("évolution à valeur nodale")
        };
        // Norme de la charge du pas (échelle relative du résidu) : xᵀx du champ.
        let ext_norm = load_scaled.xtx()?.sqrt();
        let tol = 1e-6 * ext_norm + 1e-12;

        // Newton modifié : itère jusqu'à résidu (forces déséquilibrées aux DDL
        // libres) négligeable. `last_state` retient la sortie de comportement
        // convergée, source du nouveau VAR0.
        let mut iters = 0;
        let mut last_state: Option<ElementField> = None;
        let mut res_norm = f64::INFINITY;

        for _ in 0..max_newton {
            // ε(u) → entrée de comportement (ε + VAR0) → σ, VAR1 (COMP).
            let strain = deformation(&u, &fes)?;
            let behavior_input = build_behavior_input(&strain, &state, &fes)?;
            let out = integrate(&model, &behavior_input, &materials)?;
            // Forces internes F_int = ∫ Bᵀ σ dΩ (BSIG).
            let f_int = internal_forces(&model, &out)?;

            // Résidu r = F_ext − F_int (tout-opérateurs) ; norme sur les DDL LIBRES.
            let (residual, free_res) = build_residual(&load_scaled, &f_int, &free_mesh)?;
            res_norm = free_res;
            last_state = Some(out);

            if res_norm <= tol {
                break;
            }
            // δu = K⁻¹ r (K élastique, cache de factorisation). δu porte les DDL
            // primaux ET duaux (multiplicateurs de Lagrange) sur un support neuf :
            // on le reprojette sur le support/composantes de u (`restrict_like`),
            // puis u ← u + δu par l'opérateur `+`.
            let du = solve(&k, &residual)?;
            u = (&u + &restrict_like(&du, &u)?)?;
            iters += 1;
        }
        let converged = res_norm <= tol;

        // Commit de l'état : VAR0 ← VAR1 (extrait de la sortie convergée).
        state = extract_state(last_state.as_ref().expect("au moins une itération"), &fes)?;

        // Diagnostics du pas.
        let (p_max_val, n_plastic) = plastic_diagnostics(&state)?;
        let defl = u.value(tip_id, "u_y")?;
        any_plasticity |= n_plastic > 0;
        let flag = if converged {
            ""
        } else {
            "  (résidu résiduel)"
        };
        println!(
            "{step:>4} {load_p:>8.3} {iters:>6} {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        );

        // La flèche croît (en valeur absolue, vers le bas) avec la charge.
        assert!(
            defl.abs() >= prev_defl.abs() - 1e-9,
            "flèche non monotone au pas {step}"
        );
        prev_defl = defl;
    }

    // Au-delà de la première plastification, une zone plastique doit apparaître.
    // (Bornes élastiques : première plastification vers P ≈ σy·I/(c·L).)
    let p_first_yield = sigma_y * (height * height / 6.0) / length;
    if p_max > p_first_yield {
        assert!(
            any_plasticity,
            "P_max={p_max} dépasse la première plastification (≈{p_first_yield:.2}) \
             mais aucun point plastique n'a été détecté"
        );
        println!(
            "\nOK : plastification développée (P_max={p_max} > P_élastique≈{p_first_yield:.2})."
        );
    } else {
        println!("\nOK : réponse restée élastique (P_max={p_max} ≤ ≈{p_first_yield:.2}).");
    }
    Ok(())
}

/// Sous-modèle Dirichlet encastrant `variable` (dual `dual`) sur `nodes` :
/// un POI1 des nœuds imposés, des multiplicateurs de Lagrange portés par les
/// barycentres (un par nœud). Miroir du `_clamp` de l'exemple Python.
fn clamp(nodes: &[Node], variable: &str, dual: &str) -> Result<Model> {
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
    let multiplier = barycenter(&imposed)?;
    Model::dirichlet(
        variable.to_string(),
        dual.to_string(),
        &imposed,
        &multiplier,
        None,
        None,
        Default::default(),
    )
}

/// Assemble l'entrée de comportement en fusionnant, point de Gauss par point de
/// Gauss, la déformation totale `ε` (de `deformation`) et l'état plastique
/// `VAR0` (`state`). La loi lit `ε` et `VAR0` par nom ; l'ordre est indifférent.
fn build_behavior_input(
    strain: &ElementField,
    state: &ElementField,
    fes: &FiniteElementSpace,
) -> Result<ElementField> {
    let strain_sub = read(&strain.get(0)?)?;
    let state_sub = read(&state.get(0)?)?;
    let strain_comps: Vec<String> = strain_sub.components().to_vec();
    let state_comps: Vec<String> = state_sub.components().to_vec();

    let mut all_comps = strain_comps.clone();
    all_comps.extend(state_comps.clone());
    let input = ElementField::new(fes, all_comps)?;

    let (n_cells, n_gauss) = (strain_sub.cell_count(), strain_sub.gauss_count());
    let input_h = input.get(0)?;
    let mut input_sub = write(&input_h)?;
    for cell in 0..n_cells {
        for g in 0..n_gauss {
            for c in &strain_comps {
                input_sub.set_value(cell, g, c, strain_sub.value(cell, g, c)?)?;
            }
            for c in &state_comps {
                input_sub.set_value(cell, g, c, state_sub.value(cell, g, c)?)?;
            }
        }
    }
    drop(input_sub);
    Ok(input)
}

/// Extrait le nouvel état plastique `VAR1` (les composantes [`STATE_COMPONENTS`])
/// de la sortie de comportement convergée — le `VAR0` du pas suivant.
fn extract_state(out: &ElementField, fes: &FiniteElementSpace) -> Result<ElementField> {
    let out_sub = read(&out.get(0)?)?;
    let state = ElementField::new(
        fes,
        STATE_COMPONENTS.iter().map(|s| s.to_string()).collect(),
    )?;
    let (n_cells, n_gauss) = (out_sub.cell_count(), out_sub.gauss_count());
    let state_h = state.get(0)?;
    let mut state_sub = write(&state_h)?;
    for cell in 0..n_cells {
        for g in 0..n_gauss {
            for c in STATE_COMPONENTS {
                state_sub.set_value(cell, g, c, out_sub.value(cell, g, c)?)?;
            }
        }
    }
    drop(state_sub);
    Ok(state)
}

/// Construit le résidu `r = F_ext − F_int` et sa norme sur les DDL **libres**,
/// **sans aucune boucle nodale** — tout par les opérateurs et primitives de la
/// librairie :
///
/// - `f_ext` = charge externe du pas (`load_scaled`, déjà à l'échelle sur `f_y`)
///   reprojetée sur le support ET les composantes de `f_int`
///   ([`restrict_like`]) : composantes `f_x` (=0) et `f_y` ;
/// - `residual = f_ext − f_int` via l'opérateur `-` (mêmes support/composantes) ;
/// - la norme se lit sur les seuls nœuds libres : `residual` [`restrict`]é à
///   `free_mesh` puis `xtx` (les nœuds encastrés portent la réaction, hors bilan).
fn build_residual(
    load_scaled: &NodeField,
    f_int: &NodeField,
    free_mesh: &Mesh,
) -> Result<(NodeField, f64)> {
    let f_ext = restrict_like(load_scaled, f_int)?;
    let residual = (&f_ext - f_int)?;
    let res_norm = restrict(&residual, free_mesh)?.xtx()?.sqrt();
    Ok((residual, res_norm))
}

/// `(p_max, nombre de points de Gauss plastifiés)` de l'état courant
/// (`p > 0` marque un point plastique). Sans boucle : `p_max` par [`Field::max`],
/// le comptage en masquant la composante `p` en 0/1 (bande « > 1e-12 ») puis en
/// la sommant ([`Field::sum`]).
fn plastic_diagnostics(state: &ElementField) -> Result<(f64, usize)> {
    let p_max = Field::max(state, "p")?;
    let band = Band::new(None, Some(1e-12), None, None)?;
    let masked = mask_cells(state, &band, Some(vec!["p".to_string()]))?;
    let n_plastic = Field::sum(&masked, "p")?.round() as usize;
    Ok((p_max, n_plastic))
}
