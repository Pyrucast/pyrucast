# HEX27 — hexaèdre tri-quadratique

Hexaèdre à **27 nœuds**, interpolation **Lagrange-2 complète** (\\( Q2 \\)
tensoriel). Parent : [HEX8](hex8.md), avec 12 milieux d'arête (comme
[HEX20](hex20.md)), **6 centres de face** et **1 centre de volume**.

## Repère de référence

\\( \xi, \eta, \zeta \in [-1, +1] \\). Sommets 0..7, milieux d'arête 8..19
(ordre HEX20), centres de face 20..25 (faces \\( x^-, x^+, y^-, y^+, z^-, z^+ \\)),
centre de volume 26 en \\( (0,0,0) \\).

## Fonctions de forme

Produit tensoriel des **fonctions de Lagrange quadratiques 1-D** sur
\\( \{-1, 0, +1\} \\) :

\\[
\ell_{-}(t) = \tfrac12 t(t-1), \qquad
\ell_{0}(t) = 1 - t^2, \qquad
\ell_{+}(t) = \tfrac12 t(t+1),
\\]

et \\( N_i(\xi, \eta, \zeta) = \ell_a(\xi)\\,\ell_b(\eta)\\,\ell_c(\zeta) \\), où
\\( (\ell_a, \ell_b, \ell_c) \\) sélectionne la position (\\( -, 0, + \\)) du nœud
dans chaque direction. Le centre de volume est
\\( N_{26} = (1-\xi^2)(1-\eta^2)(1-\zeta^2) \\).

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \ell_a'(\xi)\\,\ell_b(\eta)\\,\ell_c(\zeta),
\\]

et de même pour \\( \partial_\eta \\), \\( \partial_\zeta \\), avec
\\( \ell_{-}'(t) = t - \tfrac12 \\), \\( \ell_0'(t) = -2t \\),
\\( \ell_{+}'(t) = t + \tfrac12 \\).

## Quadrature (défaut)

Produit tensoriel **3×3×3** de Gauss-Legendre (27 points), exacte au degré
\\( \le 5 \\) par direction. \\( \sum_g w_g = 8 \\).

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Espace polynomial \\( Q2 \\) complet : le plus précis des hexaèdres, au prix de
  27 nœuds par élément.
