# Plasticité parfaite (von Mises)

Élastoplasticité **parfaite** (sans écrouissage) en petites déformations,
critère de **von Mises** (J2), écoulement associé. Mêmes éléments et mêmes
degrés de liberté que l'[élasticité linéaire](elasticite.md) : 2-D (`TRI3` /
`QUA4`) ou 3-D (`TET4` / `HEX8`).

## Équations résolues

- **équilibre** : `∇·σ + b = 0` ;
- **partition** : `ε = εᵉ + εᵖ` (déformation élastique + plastique) ;
- **élasticité** : `σ = D : εᵉ` ;
- **critère** : `f(σ) = q − σ_y ≤ 0`, avec `q = √(3/2 · s:s)` la contrainte
  équivalente de von Mises (`s` = déviateur de `σ`) ;
- **écoulement associé** : `ε̇ᵖ = λ̇ · ∂f/∂σ`, conditions de Kuhn–Tucker
  `λ̇ ≥ 0, f ≤ 0, λ̇ f = 0`.

Sans écrouissage, `σ_y` est constant : la contrainte équivalente est
**plafonnée** à `σ_y` (plateau parfaitement plastique).

## Linéarisation et comportement

Conformément au découpage du cœur (voir
[Comportement](../operateurs/comportement.md) — la boucle de Newton est
**pilotée en Python**, le cœur Rust ne fournit que les briques), cette physique
expose deux briques :

- **rigidité** (`build_stiffness_blocks`) : la **rigidité élastique**
  `K = ∫ Bᵀ D B dΩ`, opérateur d'itération simple (Newton modifié) ;
- **tangente cohérente** (`KTAN`, [`assemble.tangent`](../operateurs/assemblage.md)) :
  `K_t = ∫ Bᵀ D_alg B` avec le module algorithmique `D_alg` du retour radial J2
  (dérivée exacte, condensation contrainte-plane incluse) — émis par le
  comportement, relu par l'assembleur, pour un **Newton complet** à convergence
  quadratique ;
- **comportement** (`COMP`, `integrate_behavior`) : le **retour radial**
  exact, point de Gauss par point de Gauss, qui produit aussi `D_alg`.

### Retour radial (algorithme)

À partir de la déformation totale `ε` et de l'état plastique précédent
(`εᵖ`, `p`) :

1. **prédiction élastique** : `σ_trial = D : (ε − εᵖ)`, `q = √(3/2 s_trial:s_trial)` ;
2. si `f = q − σ_y ≤ 0` → pas élastique, état inchangé ;
3. sinon (plasticité parfaite) : `Δp = f / (3μ)`, le déviateur est ramené
   `s = s_trial · σ_y / q`, `Δεᵖ = Δp · (3/2) s_trial / q`, puis
   `εᵖ ← εᵖ + Δεᵖ`, `p ← p + Δp`.

Le calcul interne est mené en **3-D** quel que soit le modèle. La
**déformation plane** impose `ε_zz = ε_yz = ε_xz = 0` ; la **contrainte plane**
résout la condition `σ_zz = 0` par une méthode de la sécante locale autour du
retour radial.

## Variables, matériau, état

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (Young), `nu` (Poisson), `sigma_y` (limite d'élasticité).
- **état de début de pas A** (entrée `prev`, montage incrémental) : contrainte
  `σ(A)`, tenseur de déformation plastique `eps_p_xx … eps_p_xy` (toujours
  **6 composantes 3-D**), déformation plastique cumulée `p`, et déformation
  `ε(A)`. C'est la **sortie du pas précédent** ; `None` au premier pas (A =
  configuration de référence, tout à zéro).
- **sortie du `COMP`** (état de B, = `prev` du pas suivant) : contrainte
  (`sigma_*` dans l'ordre de Voigt du modèle) suivie de l'état mis à jour et de
  l'échO de `ε(B)` full-3-D (plus `sigma_zz` en 2-D) pour que `prev` soit
  complet. Le prédicteur élastique est `σ_trial = σ(A) + C:(ε(B) − ε(A))`.

## Exemple Python (la brique `COMP`)

La boucle de Newton (assemblage du résidu, résolution, mise à jour de l'état)
s'écrit en Python ; voici l'usage **d'un pas** de la brique d'intégration :

```python
import pyrucast

model = pyrucast.Model.plasticity(fes, "plane_stress")
materials = pyrucast.build.material_field(
    model, [("E", 210_000.0), ("nu", 0.3), ("sigma_y", 250.0)]
)

# Déformation ε(B) issue du champ de déplacement courant (op géométrique).
strain = pyrucast.field.deformation(u, fes)
# Intégration A→B : `prev` = sortie du pas précédent (None au premier pas).
state = pyrucast.behavior.integrate_behavior(model, strain, materials, prev=prev_state)
sigma_xx = state[0].value(0, 0, "sigma_xx")
p = state[0].value(0, 0, "p")  # déformation plastique cumulée
```

Pour réinjecter l'état au pas suivant, il suffit de **passer `state` comme
`prev`** au prochain appel — la sortie porte déjà l'état complet de B (σ, `VAR1`,
`ε(B)`). Aucune fusion de champs n'est nécessaire.

La **boucle de Newton complète** (pas de charge, résidu, résolution, portage de
l'état) est écrite dans `examples/plasticite_poutre_console.py` — voir la
section Rust ci-dessous pour son architecture, identique.

## Exemple Rust : poutre console, boucle de Newton complète

`examples/plasticite_poutre_console.rs` déroule un **Newton complet** (et non
un seul pas) autour des mêmes briques, côté API Rust : une poutre encastrée
cisaillée au bout, chargée par incréments jusqu'à développer une zone plastique
à l'encastrement.

L'algorithmie de Newton vit **entièrement dans l'exemple**, pas dans pyrucast :
la bibliothèque ne fournit que les opérateurs ponctuels — `stiffness` (rigidité
**élastique**, opérateur d'itération), `deformation` (`ε`), `integrate` (`COMP`,
retour radial → `σ` + état), `internal_forces` (`BSIG`, `∫ Bᵀσ`) et `solve`
(LU creux, factorisation en cache). L'exemple assemble lui-même le résidu
`r = F_ext − F_int`, résout `δu = K⁻¹ r` et porte l'état interne `VAR0 → VAR1`
d'un pas au suivant. C'est un **Newton modifié** : `K` élastique constant,
assemblé et factorisé une seule fois.

Étant en Rust pur (aucune dépendance à Python), il sert aussi de banc de
parallélisme — les boucles chaudes (assemblage, `deformation`, `integrate`,
`internal_forces`) sont réévaluées à chaque itération :

```text
RAYON_NUM_THREADS=1 PYRUCAST_NX=200 PYRUCAST_NY=40 \
    cargo run --release --example plasticite_poutre_console
```

Variables d'environnement : `PYRUCAST_NX` / `PYRUCAST_NY` (mailles),
`PYRUCAST_NSTEPS` (pas de charge), `PYRUCAST_PMAX` (charge finale).
