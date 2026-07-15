# Élasticité linéaire

Continuum en petites déformations : 2-D (`TRI3` / `QUA4`) ou 3-D
(`TET4` / `HEX8`).

## Équations continues résolues

Sur le domaine \\( \Omega \\), avec \\( b \\) les efforts volumiques :

\\[
\underbrace{\nabla\cdot\sigma + b = 0}_{\text{équilibre}}, \qquad
\underbrace{\sigma = \mathbb{D} : \varepsilon}_{\text{loi de Hooke}}, \qquad
\underbrace{\varepsilon = \tfrac12(\nabla u + \nabla u^\top)}_{\text{cinématique}}.
\\]

La **forme faible** (multiplication par un déplacement virtuel \\( v \\),
intégration par parties) s'écrit : trouver \\( u \\) tel que pour tout \\( v \\),

\\[
\int_\Omega \varepsilon(v) : \mathbb{D} : \varepsilon(u)\,d\Omega
= \int_\Omega v\cdot b\,d\Omega + \int_{\Gamma_N} v\cdot t\,d\Gamma,
\\]

où \\( t \\) est la traction imposée sur le bord de Neumann \\( \Gamma_N \\).

## Forme discrétisée

En convention de **Voigt** (déformation *ingénieur* \\( \gamma = 2\varepsilon \\)),
le champ discret \\( u_h = \sum_i N_i u_i \\) donne \\( \varepsilon = B\,u_e \\),
avec la matrice **déformation-déplacement** \\( B \\) bâtie des dérivées
physiques \\( \partial N_i/\partial x_a \\) (voir
[`dn_dx`](../fe-space.md#théorie--jacobien-et-grandeurs-physiques)). En 2-D
(\\( \varepsilon = [\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}]^\top \\)),
le bloc du nœud \\( i \\) est

\\[
B_i = \begin{bmatrix}
\partial_x N_i & 0 \\\\
0 & \partial_y N_i \\\\
\partial_y N_i & \partial_x N_i
\end{bmatrix},
\\]

et en 3-D (\\( \varepsilon = [\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \gamma_{yz}, \gamma_{xz}, \gamma_{xy}]^\top \\)),

\\[
B_i = \begin{bmatrix}
\partial_x N_i & 0 & 0 \\\\
0 & \partial_y N_i & 0 \\\\
0 & 0 & \partial_z N_i \\\\
0 & \partial_z N_i & \partial_y N_i \\\\
\partial_z N_i & 0 & \partial_x N_i \\\\
\partial_y N_i & \partial_x N_i & 0
\end{bmatrix}.
\\]

La **rigidité** élémentaire est alors, intégrée par quadrature de Gauss,

\\[
K_e = \int_{\Omega_e} B^\top D\, B\, d\Omega
\;\approx\; \sum_g B(\xi_g)^\top D\, B(\xi_g)\,|J(\xi_g)|\,w_g,
\\]

écrite aux positions `(NodeId_i, f_a) × (NodeId_j, u_b)` (ordre des DOFs
**nœud-majeur**). Le second membre nodal cohérent d'une traction de bord est
\\( f_i = \int_{\Gamma_N} N_i\,t\,d\Gamma \\) (opérateur
[`flux`](../thermique.md#exemple--un-carré)).

### Matrice constitutive `D`

Le **modèle** fixe \\( D \\) (isotrope, module d'Young \\( E \\), coefficient de
Poisson \\( \nu \\)) :

- **`plane_stress`** (contraintes planes, \\( \sigma_{zz}=0 \\)), avec
  \\( c = \dfrac{E}{1-\nu^2} \\) :

\\[
D = c\begin{bmatrix}
1 & \nu & 0 \\\\
\nu & 1 & 0 \\\\
0 & 0 & \tfrac{1-\nu}{2}
\end{bmatrix};
\\]

- **`plane_strain`** (déformations planes, \\( \varepsilon_{zz}=0 \\),
  \\( \sigma_{zz}\neq 0 \\)), avec \\( c = \dfrac{E}{(1+\nu)(1-2\nu)} \\) :

\\[
D = c\begin{bmatrix}
1-\nu & \nu & 0 \\\\
\nu & 1-\nu & 0 \\\\
0 & 0 & \tfrac{1-2\nu}{2}
\end{bmatrix};
\\]

- **`solid`** (3-D), même \\( c \\), avec le module de cisaillement
  \\( G = c\,\tfrac{1-2\nu}{2} \\) :

\\[
D = \begin{bmatrix}
c(1-\nu) & c\nu & c\nu & & & \\\\
c\nu & c(1-\nu) & c\nu & & & \\\\
c\nu & c\nu & c(1-\nu) & & & \\\\
& & & G & & \\\\
& & & & G & \\\\
& & & & & G
\end{bmatrix}
\quad (\text{ordre } [xx, yy, zz, yz, xz, xy]).
\\]

### Matrice de masse

Pour la dynamique, la **masse consistante** (composante matériau `rho`) est

\\[
M_e = \int_{\Omega_e} \rho\,N^\top N\, d\Omega
\;\approx\; \sum_g \rho\,N(\xi_g)^\top N(\xi_g)\,|J(\xi_g)|\,w_g,
\\]

où \\( N \\) place \\( N_i \\) sur chaque composante de translation — assemblée
par [`assemble.mass`](../operateurs/assemblage.md), et concentrable en diagonale
par [`lump`](../operateurs/assemblage.md).

## Variables et matériau

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (Young), `nu` (Poisson) ; **facultatif** `alpha` (dilatation
  thermique, cf. [thermomécanique](#thermomécanique-non-couplée)), `rho` (masse) — accepté par le
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

## Compléments

### Thermomécanique non couplée

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
