# Élasticité linéaire

Continuum en petites déformations : 2-D (`TRI3` / `QUA4`) ou 3-D
(`TET4` / `HEX8`), les cas 2-D couvrant aussi l'**axisymétrie** (solide de
révolution maillé dans son plan méridien).

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
\int_\Omega \varepsilon(v) : \mathbb{D} : \varepsilon(u)\\,d\Omega
= \int_\Omega v\cdot b\\,d\Omega + \int_{\Gamma_N} v\cdot t\\,d\Gamma,
\\]

où \\( t \\) est la traction imposée sur le bord de Neumann \\( \Gamma_N \\).

## Forme discrétisée

En convention de **Voigt** (déformation *ingénieur* \\( \gamma = 2\varepsilon \\)),
le champ discret \\( u_h = \sum_i N_i u_i \\) donne \\( \varepsilon = B\\,u_e \\),
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
K_e = \int_{\Omega_e} B^\top D\\, B\\, d\Omega
\\;\approx\\; \sum_g B(\xi_g)^\top D\\, B(\xi_g)\\,|J(\xi_g)|\\,w_g,
\\]

écrite aux positions `(NodeId_i, f_a) × (NodeId_j, u_b)` (ordre des DOFs
**nœud-majeur**). Le second membre nodal cohérent d'une traction de bord est
\\( f_i = \int_{\Gamma_N} N_i\\,t\\,d\Gamma \\) (opérateur
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
  \\( G = c\\,\tfrac{1-2\nu}{2} \\) :

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

- **`axisymmetric`** (solide de révolution), même \\( c \\) : les trois
  directions normales \\( r, z, \theta \\) étant orthogonales, le bloc normal est
  l'isotrope 3×3 et \\( rz \\) est le seul cisaillement,

\\[
D = c\begin{bmatrix}
1-\nu & \nu & \nu & \\\\
\nu & 1-\nu & \nu & \\\\
\nu & \nu & 1-\nu & \\\\
& & & \tfrac{1-2\nu}{2}
\end{bmatrix}
\quad (\text{ordre } [rr, zz, \theta\theta, rz]).
\\]

## Axisymétrie

Un solide de **révolution** se maille dans son plan méridien \\( (r, z) \\) sur
des `Coords` déclarées axisymétriques (\\( x = r \ge 0 \\), \\( y = z \\)) — voir
[Coordonnées](../coords.md#repère-de-révolution). Deux choses changent, et elles
ont deux origines distinctes :

1. **la mesure d'intégration**, portée par la *géométrie* :
   \\( d\Omega = 2\pi r\\,|J|\\,d\xi \\). Elle vaut pour **toutes** les intégrales
   — rigidité, masse, conductivité, flux réparti, volumes, forces internes, y
   compris sur les sous-maillages de bord `SEG2`, dont \\( \int 2\pi r\\,N \\)
   donne directement l'effort sur l'anneau. Rien à écrire : c'est
   `CellGeom::det_j_w` qui l'applique, en un seul point ;
2. **la déformation orthoradiale**, portée par le *modèle* :
   \\( \varepsilon_{\theta\theta} = u_r / r \\), que le gradient méridien ne peut
   pas exprimer. Elle ajoute une quatrième composante de Voigt et une ligne à
   \\( B \\) :

\\[
B_i = \begin{bmatrix}
\partial_r N_i & 0 \\\\
0 & \partial_z N_i \\\\
N_i / r & 0 \\\\
\partial_z N_i & \partial_r N_i
\end{bmatrix}
\quad (\varepsilon = [\varepsilon_{rr}, \varepsilon_{zz}, \varepsilon_{\theta\theta}, \gamma_{rz}]^\top).
\\]

Les points de Gauss étant **intérieurs** à la maille, \\( r > 0 \\) même pour un
élément qui touche l'axe : le terme \\( N_i/r \\) reste fini, sans traitement
particulier de l'axe.

**Nommage** (convention Cast3M) : les composantes s'appellent `sigma_xx`,
`sigma_yy`, `sigma_zz`, `sigma_xy` et `eps_xx`, `eps_yy`, `eps_zz`, `eps_xy`,
où **`zz` désigne l'orthoradial \\( \theta\theta \\)** — le plan méridien
n'occupant que `xx`, `yy` et `xy`, il n'y a pas de collision.

Le modèle et le repère doivent **s'accorder dans les deux sens** : une géométrie
de révolution refuse `plane_stress` / `plane_strain`, et `axisymmetric` refuse
une géométrie cartésienne. Sans cela on mélangerait silencieusement une loi plane
avec la mesure \\( 2\pi r \\).

La **thermique** n'a rien de spécifique à faire : le flux \\( q = -k\nabla T \\)
est déjà purement méridien, et le facteur \\( 2\pi r \\) suffit à produire le
profil logarithmique d'un cylindre creux. La [plasticité](plasticite.md) et
[Mazars](mazars.md) supportent l'axisymétrie, leur état interne étant déjà
stocké en 3-D complet. En revanche [barre](truss.md) et [portique](portique.md)
la **refusent** : un segment du plan méridien engendre une coque de révolution,
que leurs noyaux ne modélisent pas.

Un maillage de **bord** (`SEG2` en 2-D) est par ailleurs refusé comme domaine
par les trois physiques de milieu continu : `B` y serait bâti sur le gradient
tangent et \\( B^\top D B \\) serait déficient en rang dans la direction
normale. Un bord porte des charges (`flux`, convection), il n'est pas un massif.

Validation : `tests/axisymmetric.rs` (Lamé, patch test de dilatation uniforme,
\\( \int B^\top\sigma = K u \\), volume et masse de révolution, conduction
logarithmique) et `tests/python/test_axisymmetric.py`.

### Convergence sur Lamé

La solution de Lamé \\( u_r = c_1 r + c_2/r \\) comporte un terme **rationnel** :
aucune base de Lagrange, de quelque degré que ce soit, ne la reproduit
exactement. Les éléments quadratiques gagnent un ordre, pas l'exactitude. La
solution ne dépendant que de \\( r \\), le problème discret est une EDO 1-D et les
valeurs **nodales** sont superconvergentes en \\( O(h^{2p}) \\) :

| `nr` | Q1 (`QUA4`) | ordre | Q2 (`QUA8`) | ordre |
|---:|---:|---:|---:|---:|
| 5  | 6,5e-3 | — | 2,6e-5 | — |
| 10 | 1,6e-3 | 1,97 | 1,7e-6 | 3,94 |
| 20 | 4,1e-4 | 1,99 | 1,1e-7 | 3,99 |
| 40 | 1,0e-4 | 2,00 | 6,8e-9 | 4,00 |

(erreur relative maximale sur \\( u_r \\)). Les contraintes, une dérivée plus bas,
passent de \\( O(h) \\) à \\( O(h^2) \\).

Le cas **exact** existe néanmoins : lorsque \\( c_2 = 0 \\) — dilatation uniforme
\\( u_r = c\\,r \\) — l'état de déformation est constant et même Q1 le reproduit à
la précision machine (c'est le patch test de la suite de validation).

### Matrice de masse

Pour la dynamique, la **masse consistante** (composante matériau `rho`) est

\\[
M_e = \int_{\Omega_e} \rho\\,N^\top N\\, d\Omega
\\;\approx\\; \sum_g \rho\\,N(\xi_g)^\top N(\xi_g)\\,|J(\xi_g)|\\,w_g,
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
- **modèles** : `plane_stress`, `plane_strain`, `axisymmetric` (2-D) et `solid`
  (3-D).

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
f_{th} = \int_\Omega B^\top D\\, \varepsilon_{th}\\, d\Omega.
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
