# Portique (frame) 2-D

Poutre 2-D **orientée** à 3 DOFs par nœud — la généralisation de la
[poutre de Timoshenko](timoshenko.md) à une orientation quelconque, avec
l'effort axial en plus.

## Équations résolues

L'élément combine trois contributions, exprimées dans le **repère local** de la
poutre (axe `x'` le long de l'élément) :

- **axial** : `N = E·A·u'_x` (rigidité `E·A/L`, comme un [treillis](truss.md)) ;
- **flexion** : `M = E·I·θ'` ;
- **cisaillement** : `V = G·A_s·(w' − θ)`, intégré de façon **réduite** (comme
  la poutre de Timoshenko).

La matrice locale `K_loc` (6×6, DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`) est
ensuite tournée dans le repère **global** :

\\[
K = T^\top\,K_{\text{loc}}\,T,
\\]

où `T` est bâtie des cosinus directeurs de l'élément (`c = \cos\alpha`,
`s = \sin\alpha`), par nœud `\begin{bmatrix} c & s & 0 \\\\ -s & c & 0 \\\\ 0 & 0 & 1 \end{bmatrix}`.
N'importe quelle orientation dans le plan fonctionne.

- **primal** : `u_x, u_y, rz` — **dual** : `f_x, f_y, m_z`.
- **matériau** : `E`, `A`, `I`, `G`, `A_s`.
- v0 : **rigidité seule**.

## Mise en donnée (Rust, testé)

Console inclinée à 45°, encastrée à la base, charge `P` perpendiculaire à la
poutre au bout libre. Le bout se déplace de `δ = P·L³/(3·E·I) + P·L/(G·A_s)` le
long de la perpendiculaire (déplacement axial ≈ 0). Code = test
`tests/frame.rs` :

```rust,ignore
{{#include ../../../tests/frame.rs:example}}
```

## Exemple Python

```python
{{#include ../../../examples/frame.py}}
```
