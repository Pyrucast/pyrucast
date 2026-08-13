# Poutre de Timoshenko

Poutre planaire **déformable en cisaillement** : élément `SEG2` en
configuration **1-D**, deux DOFs scalaires par nœud — la flèche `w` et la
rotation de section `theta`.

## Équations continues résolues

- **cinématique** : courbure `κ = θ'`, distorsion de cisaillement `γ = w' − θ` ;
- **efforts** : moment `M = E·I·θ'`, effort tranchant `V = G·A_s·(w' − θ)` ;
- **équilibre** : `dV/dx + q = 0`, `dM/dx − V = 0`.

La rigidité est la somme d'un terme de **flexion** et d'un terme de
**cisaillement** :

\\[
K_b = \int E\\,I\\,(\theta')^2\\,dx \quad(\text{flexion}), \qquad
K_s = \int G\\,A_s\\,(w' - \theta)^2\\,dx \quad(\text{cisaillement}).
\\]

## Forme discrétisée

Sur un SEG2 (\\( N_1, N_2 \\) linéaires), les DOFs élémentaires sont
\\( [w_1, \theta_1, w_2, \theta_2] \\). Les déformations généralisées s'écrivent
\\( \kappa = B_b\\,d_e \\) et \\( \gamma = B_s\\,d_e \\) avec

\\[
B_b = \big[\\,0,\ N_1',\ 0,\ N_2'\\,\big], \qquad
B_s = \big[\\,N_1',\ -N_1,\ N_2',\ -N_2\\,\big],
\\]

d'où les deux blocs de rigidité

\\[
K_b = \int_{\Omega_e} E I\\, B_b^\top B_b\\, dx \ (\text{Gauss complet}), \qquad
K_s = \int_{\Omega_e} G A_s\\, B_s^\top B_s\\, dx \ (\text{1 point, réduit}).
\\]

Les deux termes sont intégrés sur **deux `SubFiniteElementSpace`** du même
maillage (un à quadrature complète pour la flexion, un à quadrature **réduite**
pour le cisaillement). L'évaluation de \\( B_s \\) au **centroïde**
(\\( N_1 = N_2 = \tfrac12 \\)) rend le cisaillement constant par élément — c'est
précisément ce qui supprime le **verrouillage** (*shear locking*) des poutres
élancées.

## Variables et matériau

- **primal** : `w, theta` — **dual** : `f_w` (force transverse), `m_theta`
  (moment).
- **matériau** : `E`, `I`, `G`, `A_s` (aire de cisaillement `κ·A`) ; `rho`, `A`
  **facultatifs** (masse).
- **comportement** (`COMP`) : efforts de section `M = E·I·κ` (moment) et
  `V = G·A_s·γ` (effort tranchant), à partir des déformations `(κ, γ)`
  produites par l'op [`beam_deformation`](../operateurs/champs.md) — évaluées de façon
  **réduite** (constantes par élément), donc sans cisaillement parasite.

## Mise en donnée (Rust, testé)

Console élancée encastrée (`w = θ = 0`), charge transverse `P` au bout libre.
Solution analytique `w = P·L³/(3·E·I) + P·L/(G·A_s)`. En raffinant, l'élément à
intégration réduite **converge** vers cette valeur (un élément qui verrouille
donnerait une flèche bien trop faible). Code = test `tests/timoshenko.rs` :

```rust,ignore
{{#include ../../../tests/timoshenko.rs:example}}
```

### Efforts de section (COMP)

Une fois la solution `(w, θ)` obtenue, l'op `beam_deformation` calcule les
déformations `(κ, γ)`, puis le comportement (`integrate_behavior`) donne le
moment `M = E·I·κ` et l'effort tranchant `V = G·A_s·γ`. Sur la console :
`V ≈ −P` (constant) et `M` linéaire (`|M(0)| ≈ P·L`, `|M(L)| ≈ 0`).

```rust,ignore
{{#include ../../../tests/timoshenko.rs:comp}}
```

## Exemple Python

```python
{{#include ../../../examples/timoshenko.py}}
```

## Compléments

### Masse

La poutre assemble la **masse consistante** — translation `ρA` (composante `A`
optionnelle, en plus de `A_s`) et rotation (inertie rotatoire) `ρI`, `rho`
optionnel — via `pyrucast.matrix.mass`. Pas de rigidité géométrique : une
poutre `(w, θ)` pure ne porte pas d'effort axial.
