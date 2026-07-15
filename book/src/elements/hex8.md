# HEX8 — hexaèdre trilinéaire

Hexaèdre à **8 nœuds**, interpolation **Lagrange-1** (produit tensoriel
\\( Q1 \\)). Interpolation **trilinéaire** ; l'élément volumique de référence
pour les maillages structurés. Produit par `extrude` (QUA4 → HEX8).

## Repère de référence

\\( \xi, \eta, \zeta \in [-1, +1] \\). Face inférieure CCW (nœuds 0..3) puis face
supérieure CCW (nœuds 4..7).

| Nœud | \\( (\xi_i, \eta_i, \zeta_i) \\) | | Nœud | \\( (\xi_i, \eta_i, \zeta_i) \\) |
|---:|:---:|---|---:|:---:|
| 0 | \\( (-1,-1,-1) \\) | | 4 | \\( (-1,-1,+1) \\) |
| 1 | \\( (+1,-1,-1) \\) | | 5 | \\( (+1,-1,+1) \\) |
| 2 | \\( (+1,+1,-1) \\) | | 6 | \\( (+1,+1,+1) \\) |
| 3 | \\( (-1,+1,-1) \\) | | 7 | \\( (-1,+1,+1) \\) |

## Fonctions de forme

Pour le nœud \\( i \\) de coordonnées de référence \\( (\xi_i, \eta_i, \zeta_i) \in \{-1,+1\}^3 \\) :

\\[
N_i(\xi, \eta, \zeta) = \tfrac{1}{8}\,(1 + \xi_i\,\xi)\,(1 + \eta_i\,\eta)\,(1 + \zeta_i\,\zeta).
\\]

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \tfrac18\,\xi_i\,(1 + \eta_i\,\eta)(1 + \zeta_i\,\zeta),
\\]

et de même par permutation circulaire pour \\( \partial_\eta \\) et
\\( \partial_\zeta \\).

## Quadrature (défaut)

Produit tensoriel **2×2×2** de Gauss-Legendre, exacte au degré \\( \le 3 \\) par
direction :

\\[
\xi_g = \left(\pm\tfrac{1}{\sqrt 3}\right)^3, \qquad
w_g = 1, \qquad \sum_g w_g = 8.
\\]

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Reproduit exactement les champs trilinéaires ; bien meilleur en flexion qu'un
  TET4, au prix d'un maillage structuré.
- Versions quadratiques : sérendipité [HEX20](hex20.md), complète
  [HEX27](hex27.md).
