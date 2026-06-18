# Poutre de Timoshenko

Poutre planaire **déformable en cisaillement** : élément `SEG2` en
configuration **1-D**, deux DOFs scalaires par nœud — la flèche `w` et la
rotation de section `theta`.

## Équations résolues

- **cinématique** : courbure `κ = θ'`, distorsion de cisaillement `γ = w' − θ` ;
- **efforts** : moment `M = E·I·θ'`, effort tranchant `V = G·A_s·(w' − θ)` ;
- **équilibre** : `dV/dx + q = 0`, `dM/dx − V = 0`.

La rigidité est la somme d'un terme de **flexion** et d'un terme de
**cisaillement** :

\\[
K_b = \int E\,I\,(\theta')^2\,dx \quad(\text{Gauss complet}), \qquad
K_s = \int G\,A_s\,(w' - \theta)^2\,dx \quad(\text{réduit, 1 point}).
\\]

Les deux termes sont intégrés sur **deux `SubFiniteElementSpace`** du même
maillage (un à quadrature complète pour la flexion, un à quadrature **réduite**
pour le cisaillement). L'intégration réduite du cisaillement évite le
**verrouillage** (*shear locking*) des poutres élancées.

- **primal** : `w, theta` — **dual** : `f_w` (force transverse), `m_theta`
  (moment).
- **matériau** : `E`, `I`, `G`, `A_s` (aire de cisaillement `κ·A`).
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
