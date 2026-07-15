# QUA9 — quadrangle biquadratique

Quadrangle à **9 nœuds**, interpolation **Lagrange-2 complète** (\\( Q2 \\)
tensoriel). Parent : [QUA4](qua4.md), plus 4 milieux d'arête **et un nœud
central**. Contrairement à la [sérendipité QUA8](qua8.md), il porte le monôme
\\( \xi^2\eta^2 \\).

## Repère de référence

\\( \xi, \eta \in [-1, +1] \\).

| Nœud | \\( (\xi, \eta) \\) | rôle |
|---:|:---:|---|
| 0..3 | \\( (\pm1, \pm1) \\) | sommets (CCW) |
| 4 | \\( (0, -1) \\) | milieu \\( (0,1) \\) |
| 5 | \\( (+1, 0) \\) | milieu \\( (1,2) \\) |
| 6 | \\( (0, +1) \\) | milieu \\( (2,3) \\) |
| 7 | \\( (-1, 0) \\) | milieu \\( (3,0) \\) |
| 8 | \\( (0, 0) \\) | centre |

## Fonctions de forme

Produit tensoriel des **fonctions de Lagrange quadratiques 1-D** sur
\\( \{-1, 0, +1\} \\) :

\\[
\ell_{-}(t) = \tfrac12 t(t-1), \qquad
\ell_{0}(t) = 1 - t^2, \qquad
\ell_{+}(t) = \tfrac12 t(t+1),
\\]

et \\( N_i(\xi, \eta) = \ell_a(\xi)\,\ell_b(\eta) \\), où \\( (\ell_a, \ell_b) \\)
sélectionne la position (\\( -, 0, + \\)) du nœud dans chaque direction. Ainsi le
nœud central est \\( N_8 = (1-\xi^2)(1-\eta^2) \\).

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \ell_a'(\xi)\,\ell_b(\eta), \qquad
\frac{\partial N_i}{\partial \eta} = \ell_a(\xi)\,\ell_b'(\eta),
\\]

avec \\( \ell_{-}'(t) = t - \tfrac12 \\), \\( \ell_0'(t) = -2t \\),
\\( \ell_{+}'(t) = t + \tfrac12 \\).

## Quadrature (défaut)

Produit tensoriel **3×3** de Gauss-Legendre (9 points), exacte au degré
\\( \le 5 \\) par direction. \\( \sum_g w_g = 4 \\).

## Notes

- Dimensions valides : \\( d_r = 2 \\), \\( d_s \in \{2, 3\} \\).
- Espace polynomial \\( Q2 \\) complet : reproduit exactement tout produit de
  polynômes de degré \\( \le 2 \\) par direction.
