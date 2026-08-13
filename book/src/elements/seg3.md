# SEG3 — segment quadratique

Segment à **3 nœuds**, interpolation **Lagrange-2** complète. Parent linéaire :
[SEG2](seg2.md). Un nœud de milieu porte la courbure de l'interpolation.

## Repère de référence

\\( \xi \in [-1, +1] \\), nœud médian en \\( \xi = 0 \\).

| Nœud | \\( \xi \\) | rôle |
|---:|:---:|---|
| 0 | \\( -1 \\) | sommet |
| 1 | \\( +1 \\) | sommet |
| 2 | \\( 0 \\) | milieu \\( (0,1) \\) |

## Fonctions de forme

\\[
N_0 = \tfrac12\\,\xi(\xi - 1), \qquad
N_1 = \tfrac12\\,\xi(\xi + 1), \qquad
N_2 = 1 - \xi^2.
\\]

## Dérivées de référence

\\[
\frac{\partial N_0}{\partial \xi} = \xi - \tfrac12, \qquad
\frac{\partial N_1}{\partial \xi} = \xi + \tfrac12, \qquad
\frac{\partial N_2}{\partial \xi} = -2\xi.
\\]

## Quadrature (défaut)

Gauss-Legendre à **3 points**, exacte au degré \\( \le 5 \\) :

\\[
\xi_g = \left(-\sqrt{\tfrac35},\ 0,\ +\sqrt{\tfrac35}\right), \qquad
w_g = \left(\tfrac59,\ \tfrac89,\ \tfrac59\right), \qquad \sum_g w_g = 2.
\\]

## Notes

- Dimensions valides : \\( d_r = 1 \\), \\( d_s \in \{1, 2, 3\} \\) (arête
  courbe plongée dans le plan ou l'espace).
- L'interpolation quadratique suit une géométrie **courbe** exactement (arête
  parabolique).
