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
- **matériau** : `E` (Young), `nu` (Poisson) ; **facultatif** `alpha` (dilatation
  thermique, cf. [thermomécanique](#thermomécanique-non-couplée)) — accepté par le
  champ matériau mais jamais exigé pour un assemblage purement élastique.
- **comportement** (`COMP`) : `σ = D ε` (convention tenseur → ingénieur
  `γ = 2ε`), à partir de la déformation `ε` (op [`deformation`](../operateurs/champs.md)).

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

## Thermomécanique non couplée

Première brique de thermomécanique : une température imposée `ΔT` engendre une
déformation thermique de **libre dilatation** `ε_th = α·(T − T_ref)`, d'où des
contraintes mécaniques — **sans** rétroaction de la mécanique sur le thermique.
En petites déformations, la rigidité `K` reste l'élastique ; le terme thermique
n'agit que sur le second membre et sur la contrainte réelle :

\\[
\sigma = D : (\varepsilon(u) - \varepsilon_{th}), \qquad
f_{th} = \int_\Omega B^\top D\, \varepsilon_{th}\, d\Omega.
\\]

Aucune physique nouvelle : on compose les briques existantes. `alpha` est fourni
au champ matériau (composante facultative) ; la température, portée aux points de
Gauss par [`interp_to_gauss`](../operateurs/champs.md), alimente
[`thermal_strain`](../operateurs/champs.md) (`EPTH`) ; la charge thermique sort
de `integrate_behavior` + `internal_forces` (`BSIG`) ; enfin la contrainte réelle
se relit sur `deformation(u) − ε_th`.

Deux régimes sur une barre chauffée valident les fermetures analytiques : bord en
x encastré aux deux bouts ⇒ `σ_xx = −E·α·ΔT` ; appuis simples ⇒ dilatation libre
`u = α·ΔT·(x, y)` sans contrainte. Code = test `tests/thermoelastic_bar.rs` :

```rust,ignore
{{#include ../../../tests/thermoelastic_bar.rs:example}}
```

```python
{{#include ../../../examples/thermoelastique_barre.py}}
```
