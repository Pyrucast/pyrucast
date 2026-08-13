# QUA4 — quadrangle bilinéaire

Quadrangle à **4 nœuds**, interpolation **Lagrange-1** (produit tensoriel
\\( Q1 \\)). Interpolation **bilinéaire** : le gradient varie linéairement dans
l'élément. Peut être plongé dans une `Coords` 3-D (surface).

## Repère de référence

\\( \xi, \eta \in [-1, +1] \\), sommets parcourus **CCW**.

| Nœud | \\( (\xi_i, \eta_i) \\) |
|---:|:---:|
| 0 | \\( (-1, -1) \\) |
| 1 | \\( (+1, -1) \\) |
| 2 | \\( (+1, +1) \\) |
| 3 | \\( (-1, +1) \\) |

## Fonctions de forme

Pour le nœud \\( i \\) de coordonnées de référence \\( (\xi_i, \eta_i) \\) :

\\[
N_i(\xi, \eta) = \tfrac{1}{4}\\,(1 + \xi_i\\,\xi)\\,(1 + \eta_i\\,\eta).
\\]

Explicitement :

\\[
\begin{aligned}
N_0 &= \tfrac14(1-\xi)(1-\eta), & N_1 &= \tfrac14(1+\xi)(1-\eta), \\\\
N_2 &= \tfrac14(1+\xi)(1+\eta), & N_3 &= \tfrac14(1-\xi)(1+\eta).
\end{aligned}
\\]

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \tfrac14\\,\xi_i\\,(1 + \eta_i\\,\eta), \qquad
\frac{\partial N_i}{\partial \eta} = \tfrac14\\,\eta_i\\,(1 + \xi_i\\,\xi).
\\]

## Quadrature (défaut)

Produit tensoriel **2×2** de Gauss-Legendre, exacte au degré \\( \le 3 \\) par
direction :

\\[
\xi_g = \left(\pm\tfrac{1}{\sqrt 3}, \pm\tfrac{1}{\sqrt 3}\right), \qquad
w_g = 1, \qquad \sum_g w_g = 4.
\\]

## Notes

- Dimensions valides : \\( d_r = 2 \\), \\( d_s \in \{2, 3\} \\).
- Le terme bilinéaire \\( \xi\eta \\) enrichit l'interpolation par rapport à un
  TRI3 : le QUA4 reproduit exactement les champs bilinéaires.
- Version quadratique sérendipité : [QUA8](qua8.md) ; version complète :
  [QUA9](qua9.md).
