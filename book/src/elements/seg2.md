# SEG2 — segment linéaire

Segment à **2 nœuds**, interpolation **Lagrange-1**. Élément 1-D de base
(barres, bords, poutres). Peut être **plongé** dans une `Coords` 2-D ou 3-D
(mesure de longueur via le Jacobien *manifold*).

## Repère de référence

\\( \xi \in [-1, +1] \\).

| Nœud | \\( \xi \\) |
|---:|:---:|
| 0 | \\( -1 \\) |
| 1 | \\( +1 \\) |

## Fonctions de forme

\\[
N_0(\xi) = \tfrac{1}{2}(1 - \xi), \qquad
N_1(\xi) = \tfrac{1}{2}(1 + \xi).
\\]

## Dérivées de référence

**Constantes** sur l'élément :

\\[
\frac{\partial N_0}{\partial \xi} = -\tfrac{1}{2}, \qquad
\frac{\partial N_1}{\partial \xi} = +\tfrac{1}{2}.
\\]

## Quadrature (défaut)

Gauss-Legendre à **2 points**, exacte pour les polynômes de degré \\( \le 3 \\) :

\\[
\xi_g = \pm\frac{1}{\sqrt 3}, \qquad w_g = 1, \qquad \sum_g w_g = 2.
\\]

## Notes

- Dimensions valides : \\( d_r = 1 \\), \\( d_s \in \{1, 2, 3\} \\) (segment
  plongé dans une droite, un plan ou l'espace).
- Sur une géométrie droite, \\( |J| = L/2 \\) (\\( L \\) = longueur physique),
  et la matrice de masse Lagrange-1 \\( \int N_i N_j\,ds = \tfrac{L}{6}\begin{bmatrix}2&1\\\\1&2\end{bmatrix} \\)
  est intégrée exactement par la règle à 2 points.
- Version quadratique : [SEG3](seg3.md).
