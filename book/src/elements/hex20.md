# HEX20 — hexaèdre sérendipité

Hexaèdre à **20 nœuds**, interpolation **Lagrange-2 sérendipité** (arêtes
seulement, ni face ni centre). Parent : [HEX8](hex8.md), plus 12 nœuds de milieu
d'arête. Variante complète : [HEX27](hex27.md).

## Repère de référence

\\( \xi, \eta, \zeta \in [-1, +1] \\). Sommets 0..7 (comme HEX8) ; milieux
8..19 : bas \\( (0,1),(1,2),(2,3),(3,0) \\), haut \\( (4,5),(5,6),(6,7),(7,4) \\),
verticaux \\( (0,4),(1,5),(2,6),(3,7) \\). Chaque nœud de milieu a **exactement
une coordonnée de référence nulle**.

## Fonctions de forme

Pour un nœud de coordonnées de référence \\( (p, q, r) \\) :

**Sommets** (\\( p, q, r = \pm1 \\)) — noter le facteur \\( -2 \\) :

\\[
N_i = \tfrac18\\,(1 + p\xi)(1 + q\eta)(1 + r\zeta)\\,(p\xi + q\eta + r\zeta - 2).
\\]

**Milieux d'arête**, selon la direction de l'arête (celle où la coordonnée est
nulle) :

\\[
\begin{aligned}
p = 0:&\quad N = \tfrac14(1 - \xi^2)(1 + q\eta)(1 + r\zeta), \\\\
q = 0:&\quad N = \tfrac14(1 + p\xi)(1 - \eta^2)(1 + r\zeta), \\\\
r = 0:&\quad N = \tfrac14(1 + p\xi)(1 + q\eta)(1 - \zeta^2).
\end{aligned}
\\]

## Dérivées de référence

Dérivation directe des expressions ci-dessus (recoupée par différences finies
dans les tests).

## Quadrature (défaut)

Produit tensoriel **3×3×3** de Gauss-Legendre (27 points), exacte au degré
\\( \le 5 \\) par direction. \\( \sum_g w_g = 8 \\).

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- 20 nœuds au lieu de 27 : moins d'inconnues que HEX27, sans les monômes
  d'ordre le plus élevé (\\( \xi^2\eta^2\zeta^2 \\) etc.). Bon compromis
  précision/coût, très utilisé en mécanique 3-D.
