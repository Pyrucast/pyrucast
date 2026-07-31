# PYRA5 — pyramide linéaire

Pyramide à **5 nœuds** — base carrée et sommet — interpolation **Lagrange-1**.

C'est l'**élément de raccord** entre hexaèdres et tétraèdres : sa face carrée
s'appuie sur une face de `HEX8`, ses quatre faces triangulaires sur des faces
de `TET4`. Une couche d'hexaèdres peut donc être refermée sur un cœur
tétraédrique sans nœud en T. Sans lui, il n'existe pas de maillage volumique
conforme mêlant les deux.

## Repère de référence

\\( \zeta \in [0, 1] \\), et \\( \xi, \eta \in [-(1-\zeta),\ +(1-\zeta)] \\) :
la section carrée **rétrécit** avec \\( \zeta \\) jusqu'à se réduire au sommet.
Base parcourue CCW vue depuis le sommet, puis le sommet.

| Nœud | \\( (\xi, \eta, \zeta) \\) | | Nœud | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|---|---:|:---:|
| 0 | \\( (-1, -1, 0) \\) | | 3 | \\( (-1, 1, 0) \\) |
| 1 | \\( (1, -1, 0) \\) | | 4 | \\( (0, 0, 1) \\) |
| 2 | \\( (1, 1, 0) \\) | | | |

## Fonctions de forme

La pyramide est le seul élément courant dont les fonctions de forme ne sont
**pas polynomiales**, et ce n'est pas un choix : sa base carrée doit s'effondrer
sur un point unique au sommet, et aucun polynôme ne fait cela tout en restant
bilinéaire sur la base.

En notant \\( m = 1 - \zeta \\) la demi-largeur de la section, ce sont les
fonctions bilinéaires dans les coordonnées **mises à l'échelle**
\\( \xi/m,\ \eta/m \\), pondérées par \\( m \\) :

\\[
N_i = \frac{m}{4}\left(1 + \xi_i\,\frac{\xi}{m}\right)\left(1 + \eta_i\,\frac{\eta}{m}\right)
\quad (i = 0 \dots 3), \qquad N_4 = \zeta,
\\]

où \\( (\xi_i, \eta_i) \\) sont les signes du nœud \\( i \\) sur la base.
Développé :

\\[
N_i = \frac{1}{4}\left(m + \xi_i\,\xi + \eta_i\,\eta + \xi_i\eta_i\,\frac{\xi\eta}{m}\right).
\\]

Le terme croisé \\( \xi\eta/m \\) est la **partie rationnelle** — et la raison
pour laquelle la pyramide réclame une quadrature à elle. Il reste borné sur
l'élément de référence (\\( |\xi|, |\eta| \le m \\), donc il vaut au plus
\\( m/4 \\)) mais il est bel et bien singulier **au** sommet, où la limite
\\( N_4 = 1 \\) est prise directement.

## Dérivées de référence

Avec \\( u = \xi/m \\), \\( v = \eta/m \\) :

\\[
\frac{\partial N_i}{\partial \xi} = \frac{\xi_i}{4}\,(1 + \eta_i v), \qquad
\frac{\partial N_i}{\partial \eta} = \frac{\eta_i}{4}\,(1 + \xi_i u), \qquad
\frac{\partial N_i}{\partial \zeta} = \frac{1}{4}\,(-1 + \xi_i\eta_i\,u v),
\\]

et \\( \nabla_\xi N_4 = (0, 0, 1) \\). Les trois sommes sur les cinq nœuds
s'annulent, comme il se doit — les identités \\( \sum_i \xi_i = \sum_i \eta_i =
\sum_i \xi_i\eta_i = 0 \\) sur la base carrée y suffisent.

## Quadrature (défaut)

Une pyramide n'est le produit d'aucune paire de simplexes : elle reçoit donc une
règle **conique**, produit d'une règle de Gauss–Legendre 2 × 2 sur la section
carrée par une règle de **Gauss–Jacobi** à 2 points en \\( \zeta \\), soit
**8 points**.

C'est le poids de Jacobi qui fait l'affaire. En écrivant un point sous la forme
\\( \xi = a(1-\zeta) \\), \\( \eta = b(1-\zeta) \\) avec \\( a, b \in [-1, 1] \\),
le changement de variables fait apparaître

\\[
\mathrm{d}\xi\,\mathrm{d}\eta = (1-\zeta)^2\,\mathrm{d}a\,\mathrm{d}b,
\\]

soit exactement le rétrécissement de la section vers le sommet. Intégrer la
direction \\( \zeta \\) contre ce \\( (1-\zeta)^2 \\) est une règle de
Gauss–Jacobi de paramètre \\( \alpha = 2 \\), dont les deux nœuds sont les
racines de \\( z^2 - \tfrac23 z + \tfrac1{15} \\) :

\\[
\zeta_g = \frac13 \mp \frac{\sqrt{10}}{15}
\quad\Longrightarrow\quad
\zeta_g \simeq 0{,}12251 \ \text{et}\ 0{,}54415,
\\]

et les poids se déduisent des deux premiers moments de \\( (1-z)^2 \\) sur
\\( [0,1] \\) (\\( \sum w = 1/3 \\), \\( \sum w z = 1/12 \\)) :

\\[
w_g \simeq 0{,}23255 \ \text{et}\ 0{,}10079.
\\]

La somme des poids vaut alors \\( 2 \times 2 \times \tfrac13 = \tfrac43 \\), le
volume de la pyramide de référence.

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- **Conformité.** Sur \\( \zeta = 0 \\) les fonctions se réduisent *exactement*
  à celles d'un `QUA4`, et le long d'une arête base → sommet elles sont
  *linéaires*. C'est ce qui garantit la continuité avec un `HEX8` par la face
  carrée et avec un `TET4` par une face triangulaire.
- La partie rationnelle rend l'intégration inexacte pour un élément quelconque,
  contrairement aux autres types linéaires ; la règle à 8 points est le choix
  usuel. Elle reste exacte sur le volume d'une pyramide droite ou oblique, ce
  que vérifie le test `pyra5_jacobian_volume`.
- Lu et écrit par `read_gmsh` (type 7) et par l'export VTK (`VTK_PYRAMID`, 14).
- Pas de version quadratique (`PYRA13`/`PYRA14`) pour l'instant.
