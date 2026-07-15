# TET4 — tétraèdre linéaire

Tétraèdre à **4 nœuds**, interpolation **Lagrange-1**. Élément 3-D simplicial ;
interpolation **affine**, gradient et déformation **constants par élément**
(analogue 3-D du TRI3). C'est l'élément produit par le mailleur de volume.

## Repère de référence

Simplexe unité \\( \xi, \eta, \zeta \in [0, 1] \\), \\( \xi + \eta + \zeta \le 1 \\).
Face 0-1-2 orientée CCW vue depuis le nœud 3.

| Nœud | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|
| 0 | \\( (0, 0, 0) \\) |
| 1 | \\( (1, 0, 0) \\) |
| 2 | \\( (0, 1, 0) \\) |
| 3 | \\( (0, 0, 1) \\) |

## Fonctions de forme

Coordonnées barycentriques \\( L_0 = 1-\xi-\eta-\zeta \\), \\( L_1 = \xi \\),
\\( L_2 = \eta \\), \\( L_3 = \zeta \\) :

\\[
N_0 = 1 - \xi - \eta - \zeta, \quad N_1 = \xi, \quad N_2 = \eta, \quad N_3 = \zeta.
\\]

## Dérivées de référence

**Constantes** sur l'élément :

\\[
\nabla_\xi N_0 = (-1,-1,-1), \quad
\nabla_\xi N_1 = (1,0,0), \quad
\nabla_\xi N_2 = (0,1,0), \quad
\nabla_\xi N_3 = (0,0,1).
\\]

## Quadrature (défaut)

Règle de **Hammer** à **4 points** (exacte au degré \\( \le 2 \\)) : avec
\\( \alpha = \tfrac{5 - \sqrt5}{20} \\) et \\( \beta = \tfrac{5 + 3\sqrt5}{20} \\),
les points sont les permutations \\( (\beta, \alpha, \alpha) \\) et
\\( (\alpha, \alpha, \alpha) \\),

\\[
w_g = \tfrac{1}{24}, \qquad \sum_g w_g = \tfrac{1}{6}.
\\]

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Élément à déformation constante (CST 3-D) : convergence lente en flexion, mais
  robuste et facile à mailler (Delaunay / Bowyer–Watson).
- Version quadratique : [TET10](tet10.md).
