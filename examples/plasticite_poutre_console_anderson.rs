//! Poutre console élasto-plastique — Newton modifié **accéléré par
//! l'accélération d'Anderson (m = 3)** au-dessus des briques pyrucast.
//!
//! Variante de [`plasticite_poutre_console`] : même physique, même maillage,
//! mêmes opérateurs pyrucast. Seule la boucle non linéaire change — on garde
//! l'original **gelé** comme exécutable de référence pour comparer les
//! résultats (les flèches et la plasticité doivent coïncider ; seul le nombre
//! d'itérations doit chuter).
//!
//! Physique
//! --------
//! Continuum 2-D en contraintes planes, petites déformations. Plasticité de
//! von Mises parfaite (retour radial J2, sans écrouissage) : la contrainte
//! équivalente est plafonnée à `sigma_y`. Poutre encastrée à gauche
//! (`u_x = u_y = 0`), cisaillée vers le bas sur la face droite. On monte la
//! charge par incréments.
//!
//! Newton modifié = point fixe préconditionné
//! ------------------------------------------
//! Comme dans l'exemple d'origine, l'opérateur d'itération est la rigidité
//! **élastique** `K` (assemblée + factorisée une fois, cache de `solve`).
//! L'itération `u ← u + K⁻¹r(u)` est un **point fixe préconditionné** : la
//! « direction résidu » `g(u) = K⁻¹ r(u)` s'annule à convergence — c'est le
//! résidu naturel du point fixe, et il est **déjà calculé** à chaque itération
//! (`du = solve(&k, &residual)`).
//!
//! Le prix de l'opérateur constant est une convergence seulement **linéaire**
//! sur la branche plastique (beaucoup d'itérations). L'accélération d'Anderson
//! exploite l'historique des `m = 3` derniers couples `(u, g)` pour extrapoler
//! un pas bien meilleur, **sans réévaluer la loi de comportement** : le petit
//! moindre-carré ne manipule que des produits scalaires de champs déjà en main.
//!
//! Accélération d'Anderson (m = 3)
//! -------------------------------
//! À l'itération `k`, avec l'historique des `m ≤ 3` derniers `(uᵢ, gᵢ)` (le
//! plus récent en tête) :
//!
//! 1. Différences `ΔGⱼ = g_{k-j+1} − g_{k-j}`, `ΔUⱼ = u_{k-j+1} − u_{k-j}`.
//! 2. Moindre-carré `min_γ ‖g_k − Σⱼ γⱼ ΔGⱼ‖²` → système normal `(ΔGᵀΔG) γ =
//!    ΔGᵀg_k`, dont toutes les entrées sont des produits scalaires **sur les
//!    DDL libres** (mêmes DDL que la norme du résidu), régularisé façon Tikhonov.
//! 3. Petit solve dense `m×m` (m ≤ 3, élimination de Gauss).
//! 4. Pas extrapolé `u_acc = u_k + g_k − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.
//!
//! Garde-fou de descente : on évalue le résidu du candidat d'Anderson **et**
//! celui du pas de Newton pur `u_k + g_k`, et on **retient le meilleur des
//! deux**. Anderson ne peut donc jamais dégrader la convergence par rapport au
//! Newton modifié d'origine ; s'il est rejeté, on vide l'historique.

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
use pyrucast::containers::mesh::Mesh;
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
use pyrucast::ops::node_field::{external_forces, internal_forces};
use pyrucast::ops::node_field::{positions, restrict, restrict_like};
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

/// Profondeur de l'historique d'Anderson (nombre de couples `(u, g)` gardés).
const ANDERSON_DEPTH: usize = 3;

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
        "Poutre console plastique (Anderson m={ANDERSON_DEPTH}) : {nx}×{ny} QUA4  \
         (L={length}, H={height}), E={young}, ν={nu}, σy={sigma_y}"
    );
    println!(
        "Chargement : 0 → {p_max} en {nsteps} pas (Newton modifié + accélération d'Anderson)\n"
    );

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
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Ensembles de nœuds utiles : bout (mi-hauteur), et un maillage POI1 des
    // nœuds LIBRES (hors encastrement) — support cible pour mesurer la norme du
    // résidu sur les seuls DDL libres (`restrict` + `xtx`). Les nœuds libres sont
    // ceux de coordonnée X strictement positive (bande sur le champ `positions`).
    let tip = &mesh.nearest_node(&[length, height / 2.])?;
    let coords_field = positions(&mesh, Some(vec!["X".into()]))?;
    let free_mesh = select_nodes(
        &coords_field,
        &Band::new(Some(length / nx as f64 / 2.), None, None, None)?,
        None,
    )?;

    // ── Modèle : plasticité (contraintes planes) + encastrement (Dirichlet) ──
    println!("▸ Modèle : plasticité J2 (contraintes planes) + encastrement…");
    let mut model = model::plasticity_perfect(&fes, Kinematics::PlaneStress)?;
    let imposed_mesh = to_poi1(&left_edge)?;
    let multiplier = translate(&imposed_mesh, &[0., 0.])?;
    model = model.union(&model::dirichlet(
        &model,
        "u_x",
        &imposed_mesh,
        &multiplier,
        Default::default(),
    )?)?;
    model = model.union(&model::dirichlet(
        &model,
        "u_y",
        &imposed_mesh,
        &multiplier,
        Default::default(),
    )?)?;

    // La charge de référence est un terme du modèle : elle le rejoint, sa
    // densité rejoint le matériau.
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    let model = model.union(&pyrucast::ops::model::flux(
        &right_fes,
        &model,
        "f_y".into(),
    )?)?;
    let materials = material_field(
        &model,
        &[
            ("E", young),
            ("nu", nu),
            ("sigma_y", sigma_y),
            ("phi_f_y", -1.0),
        ],
    )?;

    // Rigidité ÉLASTIQUE : opérateur d'itération du Newton modifié. Assemblée
    // une fois ; `solve` met la factorisation en cache et la réutilise à chaque
    // descente/remontée.
    println!("▸ Assemblage de la rigidité élastique K…");
    let k = stiffness(&model, &materials)?;

    // ── Charge de référence : cisaillement unitaire (densité −1) sur la face
    //    droite, réparti en efforts nodaux cohérents (∫ densité·N dΓ, op `flux`).
    println!("▸ Charge de référence + histoire de chargement…");
    // `external_forces` rend un agrégat ; l'histoire de chargement se tabule
    // zone par zone.
    let load_unit = external_forces(&model, &materials)?.get(0)?.read().clone();

    // ── Histoire de chargement : une Evolution à valeur CHAMP, tabulée en
    //    pseudo-temps t ∈ [0, 1]. Deux keyframes du champ d'effort nodal — nul en
    //    t=0, complet (`p_max · charge_unitaire`) en t=1 — sur le MÊME support.
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
    let max_newton = 200;
    println!("▸ Résolution : {nsteps} pas de charge (Newton modifié + Anderson)\n");
    println!(
        "{:>4} {:>8} {:>6} {:>6} {:>14} {:>14} {:>8}",
        "pas", "P", "iter", "andrs", "flèche u_y", "p_max", "n_plast"
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

        // Newton modifié + Anderson : itère jusqu'à résidu (forces déséquilibrées
        // aux DDL libres) négligeable. `last_state` retient la sortie de
        // comportement convergée, source du nouveau VAR0.
        let mut iters = 0;
        let mut n_anderson = 0; // combien de pas ont réellement été accélérés
        let mut last_state: Option<ElementField>;
        let mut res_norm;

        // Historique d'Anderson : couples (u, g=K⁻¹r) du pas courant, le plus
        // récent en tête. Vidé au début de chaque pas de charge.
        let mut history: Vec<(NodeField, NodeField)> = Vec::with_capacity(ANDERSON_DEPTH + 1);

        // Résidu (et forces internes convergées) à un déplacement d'essai `u` :
        // ε(u) → COMP → BSIG → r = F_ext − F_int, plus la norme sur les DDL libres.
        // Aucune boucle nodale (opérateurs de champ uniquement).
        let residual_at = |u: &NodeField| -> Result<(NodeField, f64, ElementField)> {
            // ε(u)=ε(B), état de A dans `prev` → σ, VAR1 (COMP) → F_int (BSIG).
            let strain = deformation(u, &fes)?;
            let out = integrate(&model, &strain, state.as_ref(), &materials, None)?;
            let f_int = internal_forces(&model, &out, &u, &materials)?;
            // Résidu r = F_ext − F_int et sa norme sur les DDL libres (opérateurs
            // de champ uniquement : `restrict_like`, `-`, `restrict`, `xtx`).
            let f_ext = restrict_like(&load_scaled, &f_int)?;
            let residual = (&f_ext - &f_int)?;
            let free_res = restrict(&residual, &free_mesh)?.xtx().sqrt();
            Ok((residual, free_res, out))
        };

        loop {
            // Résidu au déplacement courant (= point fixe `g = K⁻¹r`).
            let (residual, cur_res, out) = residual_at(&u)?;
            res_norm = cur_res;
            last_state = Some(out);

            if res_norm <= tol || iters >= max_newton {
                break;
            }

            // Direction résidu g = K⁻¹ r (K élastique, cache de factorisation). Le
            // support de δu coïncide déjà avec celui de u (même compagnon POI1 caché
            // de `to_poi1`) ; `restrict_like` ne filtre que les composantes duales
            // (multiplicateurs) — sinon elles se recopieraient dans u par union.
            let du = solve(&k, &residual)?;
            let g = restrict_like(&du, &u)?;

            // Snapshot du couple (u, g) courant AVANT de bouger — source des
            // différences d'Anderson au tour suivant. `map_all(|v| v)` = copie
            // profonde de l'agrégat (NodeField n'est pas Clone au niveau agrégat).
            let u_snapshot = u.map_all(|v| v)?;
            let pure_step = (&u + &g)?; // pas de Newton modifié (référence)

            // Candidat Anderson (si l'historique porte au moins un couple) :
            // extrapolation sur les m derniers (u, g). Garde-fou de descente : on
            // ne le retient que s'il réduit **strictement** le résidu courant
            // `cur_res` — sinon on prend le pas pur, dont le résidu sera évalué
            // gratuitement en tête du tour suivant (pas d'évaluation gâchée).
            let mut chose_anderson = false;
            let mut next_u = None;
            if !history.is_empty()
                && let Some(corr) = anderson_step(&u, &g, &history, &free_mesh)?
            {
                let u_acc = (&pure_step - &corr)?;
                let (_, res_acc, _) = residual_at(&u_acc)?;
                if res_acc < cur_res {
                    next_u = Some(u_acc);
                    chose_anderson = true;
                }
            }

            // Historique : si Anderson a été retenu, on empile et tronque à la
            // profondeur ; sinon on repart proprement (historique vidé) pour ne
            // pas traîner des directions qui n'aident pas.
            if chose_anderson {
                n_anderson += 1;
                history.insert(0, (u_snapshot, g));
                history.truncate(ANDERSON_DEPTH);
            } else {
                history.clear();
                history.push((u_snapshot, g));
            }
            u = next_u.unwrap_or(pure_step);
            iters += 1;
        }
        let converged = res_norm <= tol;

        // Commit de l'état : `prev` ← VAR1. La sortie convergée porte l'état
        // complet de B (σ(B), ε_p(B), p(B), ε(B)) et devient le `prev` (état de
        // A) du pas suivant (la loi lit ses entrées par nom).
        let committed = last_state
            .take()
            .expect("au moins une évaluation de résidu");

        // Diagnostics du pas.
        let (p_max_val, n_plastic) = plastic_diagnostics(&committed)?;
        state = Some(committed);
        let defl = u.value(tip.id(), "u_y")?;
        any_plasticity |= n_plastic > 0;
        let flag = if converged {
            ""
        } else {
            "  (résidu résiduel)"
        };
        println!(
            "{step:>4} {load_p:>8.3} {iters:>6} {n_anderson:>6} \
             {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        );

        // La flèche croît (en valeur absolue, vers le bas) avec la charge.
        assert!(
            defl.abs() >= prev_defl.abs() - 1e-9,
            "flèche non monotone au pas {step}"
        );
        prev_defl = defl;
    }

    // Au-delà de la première plastification, une zone plastique doit apparaître.
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

/// Pas d'accélération d'Anderson : à partir du déplacement courant `u`, de sa
/// direction résidu `g = K⁻¹r(u)`, et de l'historique des `m ≤ 3` derniers
/// couples `(uᵢ, gᵢ)` (le plus récent en tête), calcule la **correction**
/// `Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)` à soustraire au pas de Newton pur `u + g` :
///
/// `u_acc = u + g − Σⱼ γⱼ (ΔUⱼ + ΔGⱼ)`.
///
/// Les `γ` résolvent le moindre-carré `min ‖g − Σⱼ γⱼ ΔGⱼ‖²` sur les DDL
/// **libres** (`free_mesh`), via les équations normales `(ΔGᵀΔG) γ = ΔGᵀg`
/// régularisées (Tikhonov). Retourne `None` si l'historique est vide ou si le
/// petit système dégénère (l'appelant retombe alors sur le Newton pur).
///
/// Tout passe par les opérateurs de champ (`-`, `dot_field`, `restrict`) : les
/// produits scalaires sont les seules réductions, aucune évaluation de la loi
/// de comportement.
fn anderson_step(
    u: &NodeField,
    g: &NodeField,
    history: &[(NodeField, NodeField)],
    free_mesh: &Mesh,
) -> Result<Option<NodeField>> {
    let m = history.len();
    if m == 0 {
        return Ok(None);
    }

    // Différences ΔUⱼ = u_{présent} − u_{historique}, ΔGⱼ = g − g_{historique}.
    // (Convention équivalente aux différences successives à un signe global près,
    // absorbé par γ ; ici on prend les différences vers l'itéré courant.)
    let mut du_diffs: Vec<NodeField> = Vec::with_capacity(m);
    let mut dg_diffs: Vec<NodeField> = Vec::with_capacity(m);
    for (u_hist, g_hist) in history {
        du_diffs.push((u - u_hist)?);
        dg_diffs.push((g - g_hist)?);
    }

    // ΔG restreints aux DDL libres (support des produits scalaires du résidu).
    let dg_free: Vec<NodeField> = dg_diffs
        .iter()
        .map(|d| restrict(d, free_mesh))
        .collect::<Result<_>>()?;
    let g_free = restrict(g, free_mesh)?;

    // Équations normales (ΔGᵀΔG) γ = ΔGᵀg (petit système m×m symétrique).
    let mut a = vec![vec![0.0_f64; m]; m];
    let mut b = vec![0.0_f64; m];
    let mut trace = 0.0;
    for i in 0..m {
        for j in i..m {
            let v = dg_free[i].dot_field(&dg_free[j])?;
            a[i][j] = v;
            a[j][i] = v;
        }
        trace += a[i][i];
        b[i] = dg_free[i].dot_field(&g_free)?;
    }
    if trace <= 0.0 {
        return Ok(None); // directions dégénérées
    }
    // Régularisation de Tikhonov : + λ·(trace/m) sur la diagonale.
    let lambda = 1e-10 * trace / m as f64;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += lambda;
    }

    let Some(gamma) = solve_small_spd(a, b) else {
        return Ok(None);
    };

    // Correction Σⱼ γⱼ (ΔUⱼ + ΔGⱼ), assemblée par opérateurs de champ.
    let mut corr: Option<NodeField> = None;
    for (j, gj) in gamma.iter().enumerate() {
        let term = (&(&du_diffs[j] + &dg_diffs[j])? * *gj)?;
        corr = Some(match corr {
            None => term,
            Some(acc) => (&acc + &term)?,
        });
    }
    Ok(corr)
}

/// Résout un petit système dense **symétrique** `A x = b` (`m ≤ 3`) par
/// élimination de Gauss avec pivot partiel. Renvoie `None` si `A` est
/// singulière (pivot ~ 0) — l'appelant retombe alors sur le Newton pur.
fn solve_small_spd(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Pivot partiel.
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-30 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // Élimination (copie de la ligne pivot pour éviter le double emprunt).
        let pivot_row = a[col].clone();
        let b_pivot = b[col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot_row[col];
            for (dst, &src) in a[r].iter_mut().zip(pivot_row.iter()).skip(col) {
                *dst -= factor * src;
            }
            b[r] -= factor * b_pivot;
        }
    }
    // Remontée.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// `(p_max, nombre de points de Gauss plastifiés)` de l'état courant
/// (`p > 0` marque un point plastique).
fn plastic_diagnostics(state: &ElementField) -> Result<(f64, usize)> {
    let p_max = Field::max(state, Some("p"))?;
    let band = Band::new(None, Some(1e-12), None, None)?;
    let masked = mask(state, &band, Some(vec!["p".to_string()]))?;
    let n_plastic = Field::sum(&masked, "p")?.round() as usize;
    Ok((p_max, n_plastic))
}
