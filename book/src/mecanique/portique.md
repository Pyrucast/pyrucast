# Portique (frame) 2-D

Poutre 2-D **orientée** à 3 DOFs par nœud — la généralisation de la
[poutre de Timoshenko](timoshenko.md) à une orientation quelconque, avec
l'effort axial en plus.

## Équations continues résolues

L'élément combine trois contributions, exprimées dans le **repère local** de la
poutre (axe `x'` le long de l'élément) :

- **axial** : `N = E·A·u'_x` (comme un [treillis](truss.md)) ;
- **flexion** : `M = E·I·θ'` ;
- **cisaillement** : `V = G·A_s·(w' − θ)` (comme la poutre de Timoshenko).

## Forme discrétisée

La matrice locale `K_loc` (6×6, DOFs `[u'_A, w'_A, θ_A, u'_B, w'_B, θ_B]`) est
tournée dans le repère **global** :

\\[
K = T^\top\\,K_{\text{loc}}\\,T,
\\]

où `T` est bâtie des cosinus directeurs de l'élément (`c = \cos\alpha`,
`s = \sin\alpha`), par nœud `\begin{bmatrix} c & s & 0 \\\\ -s & c & 0 \\\\ 0 & 0 & 1 \end{bmatrix}`.
N'importe quelle orientation dans le plan fonctionne.

Dans le repère local, `K_loc` est la **superposition** de trois contributions
découplées :

- **axial** — sur les DOFs \\( (u'_A, u'_B) \\) :
  \\( \dfrac{EA}{L}\begin{bmatrix}1 & -1 \\\\ -1 & 1\end{bmatrix} \\) ;
- **flexion** — sur \\( (\theta_A, \theta_B) \\), bloc
  \\( K_b = \int E I\\, B_b^\top B_b\\, dx \\) ;
- **cisaillement** — sur \\( (w'_A, \theta_A, w'_B, \theta_B) \\), bloc
  \\( K_s = \int G A_s\\, B_s^\top B_s\\, dx \\) intégré de façon **réduite**,

avec les mêmes opérateurs discrets \\( B_b, B_s \\) que la
[poutre de Timoshenko](timoshenko.md#forme-discrétisée).

## Variables et matériau

- **primal** : `u_x, u_y, rz` — **dual** : `f_x, f_y, m_z`.
- **matériau** : `E`, `A`, `I`, `G`, `A_s` ; `rho` **facultatif** (masse).
- v0 : **rigidité seule** (compléments : masse & rigidité géométrique).

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

## Compléments

### Masse & rigidité géométrique

Le portique assemble aussi la **masse consistante** — translation `ρA` et
rotation (inertie rotatoire) `ρI`, `rho` optionnel — et la **rigidité
géométrique** transverse `(N/L)` sous l'effort axial `N` (sortie `N` du
comportement), toutes deux en forme d'élément linéaire tournées `Tᵀ·T`
(`pyrucast.matrix.mass` / `pyrucast.matrix.geometric`).
