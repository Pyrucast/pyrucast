# TRI3 — triangle linéaire

Triangle à **3 nœuds**, interpolation **Lagrange-1**. L'élément 2-D le plus
simple ; interpolation **affine**, donc gradient et déformation **constants par
élément**. Peut être plongé dans une `Coords` 3-D (surface).

## Repère de référence

Simplexe unité \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\), parcouru
**CCW**.

| Nœud | \\( (\xi, \eta) \\) |
|---:|:---:|
| 0 | \\( (0, 0) \\) |
| 1 | \\( (1, 0) \\) |
| 2 | \\( (0, 1) \\) |

## Fonctions de forme

Ce sont les **coordonnées barycentriques** \\( L_0 = 1-\xi-\eta \\),
\\( L_1 = \xi \\), \\( L_2 = \eta \\) :

\\[
N_0 = 1 - \xi - \eta, \qquad N_1 = \xi, \qquad N_2 = \eta.
\\]

## Dérivées de référence

**Constantes** sur l'élément :

\\[
\nabla_\xi N_0 = (-1, -1), \qquad
\nabla_\xi N_1 = (1, 0), \qquad
\nabla_\xi N_2 = (0, 1).
\\]

## Quadrature (défaut)

Règle de **Hammer mid-edge** à **3 points**, exacte au degré \\( \le 2 \\) :

\\[
\xi_g \in \left\\{ \left(\tfrac12, 0\right), \left(\tfrac12, \tfrac12\right), \left(0, \tfrac12\right) \right\\}, \qquad
w_g = \tfrac{1}{6}, \qquad \sum_g w_g = \tfrac12.
\\]

## Notes

- Dimensions valides : \\( d_r = 2 \\), \\( d_s \in \{2, 3\} \\) (triangle plan
  ou plongé dans l'espace — c'est ce que produit `fill_surface` sur un contour
  3-D).
- Gradient constant : un seul point de Gauss suffirait pour un champ affine,
  mais la règle à 3 points intègre exactement la masse (\\( \int N_i N_j \\), de
  degré 2).
- Version quadratique : [TRI6](tri6.md).
