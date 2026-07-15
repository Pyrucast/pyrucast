# Cadre 3-D (space frame)

Poutre 3-D à **6 DOFs par nœud** — la généralisation du
[portique 2-D](portique.md) à l'espace, avec la torsion en plus.

## Équations résolues

Dans le **repère local** de la poutre (axe `x'` le long de l'élément), la
rigidité combine quatre comportements découplés :

- **axial** : `N = E·A·u'_x` ;
- **torsion** : `M_t = G·J·θ'_x` ;
- **flexion** dans le plan `x'-y'` (autour de `z'`) : Timoshenko avec `E·I_z`
  et l'aire de cisaillement `A_sy` ;
- **flexion** dans le plan `x'-z'` (autour de `y'`) : Timoshenko avec `E·I_y`
  et l'aire de cisaillement `A_sz`.

La matrice locale 12×12 (forme fermée à paramètres `Φ`, **nodalement exacte**
pour des charges en bout) est tournée dans le repère global :

\\[
K = T^\top\,K_{\text{loc}}\,T,
\\]

où `T` répète, sur les quatre triplets de DOFs, la rotation `R = [x'; y'; z']`.
Les axes de section `y'`, `z'` sont **orientés automatiquement** depuis une
référence Z global (Y si la poutre est verticale) — aucune donnée d'orientation
à fournir ; sans importance pour les sections symétriques (`I_y = I_z`).

- **primal** : `u_x, u_y, u_z, r_x, r_y, r_z` —
  **dual** : `f_x, f_y, f_z, m_x, m_y, m_z`.
- **matériau** : `E, A, I_y, I_z, J, G, A_sy, A_sz`.
- v0 : **rigidité seule**.

## Mise en donnée (Rust, testé)

Console le long de X, encastrée à la base, charges au bout libre `f_y`, `f_z` et
un moment de torsion `m_x`. Les réponses sont découplées et exactes. Code =
test `tests/frame3d.rs` :

```rust,ignore
{{#include ../../../tests/frame3d.rs:example}}
```

## Exemple Python

```python
{{#include ../../../examples/frame3d.py}}
```

## Masse & rigidité géométrique

Le cadre 3-D assemble aussi la **masse consistante** — translation `ρA`,
inertie de torsion `ρ(I_y+I_z)`, inerties rotatoires de flexion `ρI_y`, `ρI_z`,
`rho` optionnel — et la **rigidité géométrique** transverse `(N/L)` (sur les deux
directions perpendiculaires à l'axe) sous l'effort axial `N`, en forme d'élément
linéaire tournées `Tᵀ·T` (`pyrucast.assemble.mass` / `pyrucast.assemble.geometric`).
