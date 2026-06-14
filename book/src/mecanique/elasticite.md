# Élasticité linéaire

Continuum en petites déformations : 2-D (`TRI3` / `QUA4`) ou 3-D
(`TET4` / `HEX8`).

## Équations résolues

- **équilibre** : `∇·σ + b = 0` ;
- **loi de Hooke** : `σ = D : ε` ;
- **cinématique** : `ε = ½(∇u + ∇uᵀ)`.

La forme faible donne la rigidité

\\[
K = \int_\Omega B^\top D\, B \, d\Omega,
\\]

où `B` est la matrice déformation-déplacement (construite depuis `∂N_i/∂x`) en
convention de Voigt, et `D` la matrice constitutive isotrope. Le **modèle**
détermine `D` :

- `plane_stress` (contraintes planes) : `D = \frac{E}{1-\nu^2}\,[\dots]` ;
- `plane_strain` (déformations planes) : `\sigma_{zz} \neq 0`, `\varepsilon_{zz}=0` ;
- `solid` (3-D) : matrice 6×6 isotrope.

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (Young), `nu` (Poisson).
- **comportement** (`COMP`) : `σ = D ε` (convention tenseur → ingénieur
  `γ = 2ε`), à partir de la déformation `ε` (op [`deformation`](../node-field.md)).

## Mise en donnée (Rust, testé)

Carré unité en **contraintes planes** : appuis `u_x = 0` (gauche) et `u_y = 0`
(bas), traction `S` sur le bord droit appliquée en charges nodales cohérentes
par l'opérateur [`flux`](../thermique.md#exemple--un-carré) (composante `f_x`).
Solution exacte `u_x = (S/E)·x`, `u_y = −(ν S/E)·y`. Code = test
`tests/elasticity.rs` (le fichier contient aussi un test **3-D** sur un cube
`HEX8`) :

```rust,ignore
{{#include ../../../tests/elasticity.rs:example}}
```

## Exemple Python

```python
{{#include ../../../examples/elasticity.py}}
```
