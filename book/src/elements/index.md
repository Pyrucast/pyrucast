# Éléments finis supportés

Cette section est le **catalogue de référence** des éléments finis de pyrucast :
une fiche par type, avec le repère de référence, les **fonctions de forme**
\\( N_i(\xi) \\) exactes, leurs dérivées de référence \\( \partial N_i/\partial\xi_k \\),
et la **règle de quadrature** associée.

La machinerie **commune** à tous les éléments (transformation isoparamétrique,
Jacobien, gradient physique \\( \partial N_i/\partial x_a \\), passage à la
matrice élémentaire) est décrite une fois pour toutes au chapitre
[Espace éléments finis](../fe-space.md#théorie--élément-isoparamétrique) ; les
fiches ci-dessous ne répètent que ce qui est **propre** à chaque élément. Le
code source est `containers/finite_element_space/interpolation.rs` (fonctions de
forme) et `quadrature.rs` (points de Gauss) ; les conventions de repère et de
numérotation locale vivent dans `containers/mesh/element_type.rs`.

Toutes les fiches suivent le **même plan standard** : *introduction* (nombre de
nœuds, famille, dimensions), **Repère de référence** (domaine + numérotation
locale), **Fonctions de forme**, **Dérivées de référence**, **Quadrature
(défaut)** (points, poids, exactitude), puis **Notes** (dimensions valides,
propriétés, renvois vers les variantes).

## Rappel : de la fonction de forme à la matrice

Sur chaque élément, un champ est interpolé par ses valeurs nodales,
\\( u(\xi) = \sum_i N_i(\xi)\,u_i \\), et **la géométrie de la même façon**
(hypothèse isoparamétrique), \\( \mathbf{x}(\xi) = \sum_i N_i(\xi)\,\mathbf{x}_i \\).
Toute matrice élémentaire est une intégrale sur l'élément physique ramenée à
l'élément de référence par le Jacobien \\( J = \partial\mathbf{x}/\partial\xi \\) :

\\[
\int_K \phi(\mathbf{x})\,d\mathbf{x}
= \int_{\hat K} \phi(\chi(\xi))\,|J(\xi)|\,d\xi
\approx \sum_{g} w_g\,\phi(\xi_g)\,|J(\xi_g)|,
\\]

et les dérivées physiques viennent de l'inverse du Jacobien,
\\( \nabla_x N_i = J^{-\top}\nabla_\xi N_i \\). **Les seuls ingrédients qui
changent d'un élément à l'autre** sont donc : les \\( N_i \\), les
\\( \partial N_i/\partial\xi_k \\), et le couple \\( (\xi_g, w_g) \\) — exactement
le contenu de chaque fiche.

## Catalogue

Deux familles d'interpolation Lagrange sont disponibles : **Lagrange-1**
(linéaire, un nœud par sommet) et **Lagrange-2** (quadratique, sommets + nœuds
de milieu d'arête). `POI1` (nœud seul) n'a pas de repère de référence : ce n'est
pas un élément fini.

### Lagrange-1 (linéaire)

| Élément | Nœuds | Dim. topo. | Domaine de référence | Quadrature (\\( n_g \\)) |
|---|---:|:---:|---|---|
| [SEG2](seg2.md) | 2 | 1 | \\( \xi\in[-1,1] \\) | Gauss 2 pts |
| [TRI3](tri3.md) | 3 | 2 | simplexe \\( \xi+\eta\le 1 \\) | Hammer 3 pts |
| [QUA4](qua4.md) | 4 | 2 | \\( [-1,1]^2 \\) | 2×2 Gauss (4) |
| [TET4](tet4.md) | 4 | 3 | simplexe \\( \xi+\eta+\zeta\le 1 \\) | Hammer 4 pts |
| [PYRA5](pyra5.md) | 5 | 3 | pyramide (section carrée décroissante) | Gauss×Jacobi conique (8) |
| [PENTA6](penta6.md) | 6 | 3 | prisme (TRI3 × \\( \zeta \\)) | TRI×Gauss (6) |
| [HEX8](hex8.md) | 8 | 3 | \\( [-1,1]^3 \\) | 2×2×2 Gauss (8) |

### Lagrange-2 (quadratique)

| Élément | Nœuds | Dim. topo. | Parent | Type | Quadrature (\\( n_g \\)) |
|---|---:|:---:|:---:|---|---|
| [SEG3](seg3.md) | 3 | 1 | SEG2 | complet | Gauss 3 pts |
| [TRI6](tri6.md) | 6 | 2 | TRI3 | complet | Dunavant deg. 4 (6) |
| [QUA8](qua8.md) | 8 | 2 | QUA4 | sérendipité | 3×3 Gauss (9) |
| [QUA9](qua9.md) | 9 | 2 | QUA4 | complet (Q2) | 3×3 Gauss (9) |
| [TET10](tet10.md) | 10 | 3 | TET4 | complet | Keast deg. 4 (11) |
| [PENTA15](penta15.md) | 15 | 3 | PENTA6 | sérendipité | TRI6×Gauss (18) |
| [HEX20](hex20.md) | 20 | 3 | HEX8 | sérendipité | 3×3×3 Gauss (27) |
| [HEX27](hex27.md) | 27 | 3 | HEX8 | complet (Q2) | 3×3×3 Gauss (27) |

Les éléments **sérendipité** (QUA8, HEX20, PENTA15) ne portent que des nœuds
d'arête ; les **complets** (SEG3, TRI6, TET10, QUA9, HEX27) portent en plus les
nœuds de face et/ou de volume nécessaires au produit tensoriel \\( Q2 \\) complet.

## Catalogue de quadrature

Le catalogue ci-dessus ne montre, par élément, que la règle **par défaut**
(`GAUSS`). Une deuxième règle existe — `REDUCED` (intégration réduite : un seul
point au centroïde, poids = mesure du domaine de référence, exacte pour les
constantes seulement ; utilisée par exemple pour désamorcer le verrouillage en
cisaillement de la [poutre de Timoshenko](../mecanique/timoshenko.md)). Le
tableau croisé suivant donne, pour chaque couple (élément, règle), le nombre de
points d'intégration \\( n_g \\) si le couple est supporté :

| Élément | GAUSS | REDUCED |
|---|:---:|:---:|
| [SEG2](seg2.md) | ✓ (2) | ✓ (1) |
| [TRI3](tri3.md) | ✓ (3) | ✓ (1) |
| [QUA4](qua4.md) | ✓ (4) | ✓ (1) |
| [TET4](tet4.md) | ✓ (4) | ✓ (1) |
| [PYRA5](pyra5.md) | ✓ (8) | ✓ (1) |
| [PENTA6](penta6.md) | ✓ (6) | ✓ (1) |
| [HEX8](hex8.md) | ✓ (8) | ✓ (1) |
| [SEG3](seg3.md) | ✓ (3) | ✓ (1) |
| [TRI6](tri6.md) | ✓ (6) | ✓ (1) |
| [QUA8](qua8.md) | ✓ (9) | ✓ (1) |
| [QUA9](qua9.md) | ✓ (9) | ✓ (1) |
| [TET10](tet10.md) | ✓ (11) | ✓ (1) |
| [PENTA15](penta15.md) | ✓ (18) | ✓ (1) |
| [HEX20](hex20.md) | ✓ (27) | ✓ (1) |
| [HEX27](hex27.md) | ✓ (27) | ✓ (1) |
| POI1 | — | — |

`POI1` n'a pas de repère de référence (ce n'est pas un élément fini) : les deux
règles y sont rejetées (`QuadratureRule::is_compatible_with` renvoie `false`,
`points`/`point_count` renvoient une erreur). Pour tout autre `ElementType`,
les deux règles sont actuellement définies — le tableau est donc plein sauf
sur cette ligne. Il est conservé tel quel pour documenter la compatibilité au
fur et à mesure que de nouvelles règles (ordres supérieurs, quadratures
spécialisées) seront ajoutées : celles-ci pourront être incompatibles avec
certains éléments (p. ex. une règle calibrée pour un degré d'exactitude
indisponible sur un élément sérendipité), et ce tableau sera le seul endroit à
mettre à jour.

## Propriétés communes (vérifiées par les tests)

Toutes les interpolations Lagrange satisfont, à tout point de référence :

- **Kronecker** : \\( N_i(\xi_j) = \delta_{ij} \\) aux nœuds — l'interpolation
  passe par les valeurs nodales ;
- **partition de l'unité** : \\( \sum_i N_i(\xi) = 1 \\), d'où la reproduction
  exacte des champs constants ;
- **dérivées à somme nulle** : \\( \sum_i \partial N_i/\partial\xi_k = 0 \\)
  (partition de l'unité dérivée) ;
- pour les éléments quadratiques, les dérivées analytiques sont recoupées par
  **différences finies centrées** dans les tests unitaires.

La règle de quadrature par défaut de chaque élément est calibrée pour
**intégrer exactement sa matrice de masse** sur une géométrie droite ; la somme
des poids vaut la mesure du domaine de référence.
