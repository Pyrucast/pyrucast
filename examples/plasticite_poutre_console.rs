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

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::Band;
use pyrucast::atoms::ElementType;
use pyrucast::atoms::Node;
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::evolution::{
    Evolution, Interpolated, OutOfRange, SubEvolution, SubValue,
};
use pyrucast::containers::field::{Field, SubField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::behavior::integrate;
use pyrucast::ops::element_field::deformation;
use pyrucast::ops::element_field::mask;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::matrix::stiffness;
use pyrucast::ops::mesh::select_nodes;
use pyrucast::ops::mesh::{line, sweep, to_poi1, translate};
use pyrucast::ops::model;
use pyrucast::ops::node_field::internal_forces;
use pyrucast::ops::node_field::{flux, FluxDensity};
use pyrucast::ops::node_field::{positions, restrict, restrict_like};
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

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
    println!(
        "▸ Maillage : {} nœuds, {} cellules QUA4…",
        (nx + 1) * (ny + 1),
        nx * ny
    );
    let coords = Handle::new(Coords::new(2)?);
    let pt_a = Node::create_in(coords.clone(), &[0., 0.])?;
    let pt_b = Node::create_in(coords.clone(), &[0., height])?;
    let pt_c = Node::create_in(coords.clone(), &[length, 0.])?;
    let pt_d = Node::create_in(coords.clone(), &[length, height])?;
    let left_edge = line(&pt_a, &pt_b, ny, ElementType::SEG2)?;
    let right_edge = line(&pt_c, &pt_d, ny, ElementType::SEG2)?;
    let mesh = sweep(&left_edge, &right_edge, nx, ElementType::QUA4)?;

    // grid[j][i] : nœud à (x = i·L/nx, y = j·H/ny).
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Ensembles de nœuds utiles : bord gauche (encastré), bout (mi-hauteur), et
    // un maillage POI1 des nœuds LIBRES (hors encastrement) — support cible pour
    // mesurer la norme du résidu sur les seuls DDL libres (`restrict` + `xtx`).
    let tip_id = &mesh.nearest_node(&[length, height / 2.])?;
    let coords_field = positions(&mesh, Some(vec!["X".into()]))?;
    let free_mesh = select_nodes(
        &coords_field,
        &Band::new(Some(length / nx as f64 / 2.), None, None, None)?,
        None,
    )?;
    let imposed_mesh = to_poi1(&left_edge)?;
    let multiplier = translate(&imposed_mesh, &[0., 0.])?;

    // ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    println!("▸ Modèle : plasticité J2 (contraintes planes) + encastrement…");
    let mut model = model::plasticity_perfect(&fes, Kinematics::PlaneStress)?;

    model = model.union(&model::dirichlet(
        "u_x".into(),
        "f_x".into(),
        &imposed_mesh,
        &multiplier,
        None,
        None,
        Default::default(),
    )?)?;
    model = model.union(&model::dirichlet(
        "u_y".into(),
        "f_y".into(),
        &imposed_mesh,
        &multiplier,
        None,
        None,
        Default::default(),
    )?)?;

    let materials = material_field(&model, &[("E", young), ("nu", nu), ("sigma_y", sigma_y)])?;

    // Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée
    // une fois ; `solve` met la factorisation en cache et la réutilise à chaque
    // descente/remontée.
    println!("▸ Assemblage de la rigidité élastique K…");
    let k = stiffness(&model, &materials)?;

    // ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    //    droite, réparti en efforts nodaux cohérents (∫ densité·N dΓ, op `flux`).
    println!("▸ Charge de référence + histoire de chargement…");
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    // `flux` rend un agrégat ; l'histoire de chargement se tabule zone par zone.
    let load_unit = flux(&right_fes, FluxDensity::Uniform(-1.0), "f_y")?
        .get(0)?
        .read()
        .clone();

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
    load_evo.add_sub(Handle::new(load_curve))?;

    // ── État de la simulation (persistant entre les pas) ────────────────────
    // Déplacement cumulé u (u_x, u_y sur tous les nœuds), initialement nul.
    let mut u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
    // État convergé du pas précédent (VAR0 = `prev`) : `None` au premier pas —
    // A est alors la configuration de référence (σ(A)=0, ε(A)=0).
    let mut state: Option<ElementField> = None;

    // ── Boucle sur les pas de charge ────────────────────────────────────────
    // Newton modifié (opérateur = K élastique) : convergence linéaire, donc
    // lente sur la branche plastique. On plafonne haut les itérations et on
    // vise un résidu relatif de 1e-6 (largement suffisant ici).
    let max_newton = 200;
    println!("▸ Résolution : {nsteps} pas de charge (Newton modifié)\n");
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
        let ext_norm = load_scaled.xtx().sqrt();
        let tol = 1e-6 * ext_norm + 1e-12;

        // Newton modifié : itère jusqu'à résidu (forces déséquilibrées aux DDL
        // libres) négligeable. `last_state` retient la sortie de comportement
        // convergée, source du nouveau VAR0.
        let mut iters = 0;
        let mut last_state: Option<ElementField> = None;
        let mut res_norm = f64::INFINITY;

        for _ in 0..max_newton {
            // ε(u) = ε(B), état de A dans `prev` → σ, VAR1 (COMP), montage A→B.
            let strain = deformation(&u, &fes)?;
            let out = integrate(&model, &strain, state.as_ref(), &materials, None)?;
            // Forces internes F_int = ∫ Bᵀ σ dΩ (BSIG).
            let f_int = internal_forces(&model, &out)?;

            // Résidu r = F_ext − F_int et sa norme sur les DDL **libres**, sans
            // aucune boucle nodale — tout par les opérateurs et primitives :
            // - `f_ext` = charge externe du pas (`load_scaled`, déjà à l'échelle
            //   sur `f_y`) reprojetée sur le support ET les composantes de `f_int`
            //   (`restrict_like`) : composantes `f_x` (=0) et `f_y` ;
            // - `residual = f_ext − f_int` via l'opérateur `-` ;
            // - la norme se lit sur les seuls nœuds libres : `residual` `restrict`é
            //   à `free_mesh` puis `xtx` (les nœuds encastrés portent la réaction).
            let f_ext = restrict_like(&load_scaled, &f_int)?;
            let residual = (f_ext - f_int)?;
            res_norm = restrict(&residual, &free_mesh)?.xtx().sqrt();
            last_state = Some(out);

            if res_norm <= tol {
                break;
            }
            // δu = K⁻¹ r (K élastique, cache de factorisation). δu porte les DDL
            // primaux ET duaux (multiplicateurs de Lagrange). Son support coïncide
            // déjà avec celui de u (même compagnon POI1 caché de `to_poi1`, partagé
            // par `solve` et `NodeField::new(&mesh)`) ; `restrict_like` ne sert donc
            // qu'à **filtrer les composantes duales** — sinon `u + δu` recopierait
            // les multiplicateurs dans u par union. Puis u ← u + δu par `+`.
            let du = solve(&k, &residual)?;
            u = (&u + &restrict_like(&du, &u)?)?;
            iters += 1;
        }
        let converged = res_norm <= tol;

        // Commit de l'état : `prev` ← VAR1. La sortie de comportement convergée
        // porte l'état complet de B (σ(B), ε_p(B), p(B), ε(B)) et devient le
        // `prev` (état de A) du pas suivant. La loi lit ses entrées par nom, donc
        // les composantes surnuméraires sont ignorées.
        let committed = last_state.take().expect("au moins une itération");

        // Diagnostics du pas.
        let (p_max_val, n_plastic) = plastic_diagnostics(&committed)?;
        state = Some(committed);
        let defl = u.value(tip_id.id(), "u_y")?;
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

/// `(p_max, nombre de points de Gauss plastifiés)` de l'état courant
/// (`p > 0` marque un point plastique). Sans boucle : `p_max` par [`Field::max`],
/// le comptage en masquant la composante `p` en 0/1 (bande « > 1e-12 ») puis en
/// la sommant ([`Field::sum`]).
fn plastic_diagnostics(state: &ElementField) -> Result<(f64, usize)> {
    let p_max = Field::max(state, Some("p"))?;
    let band = Band::new(None, Some(1e-12), None, None)?;
    let masked = mask(state, &band, Some(vec!["p".to_string()]))?;
    let n_plastic = Field::sum(&masked, "p")?.round() as usize;
    Ok((p_max, n_plastic))
}
