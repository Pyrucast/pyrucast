# QUA8 — quadrangle sérendipité

Quadrangle à **8 nœuds**, interpolation **Lagrange-2 sérendipité** (arêtes
seulement, **pas** de nœud central). Parent : [QUA4](qua4.md), plus 4 nœuds de
milieu d'arête. Variante complète (avec nœud central) : [QUA9](qua9.md).

## Repère de référence

\\( \xi, \eta \in [-1, +1] \\). Sommets 0..3 (comme QUA4), milieux 4..7.

| Nœud | \\( (\xi, \eta) \\) | rôle |
|---:|:---:|---|
| 0..3 | \\( (\pm1, \pm1) \\) | sommets (CCW) |
| 4 | \\( (0, -1) \\) | milieu \\( (0,1) \\) |
| 5 | \\( (+1, 0) \\) | milieu \\( (1,2) \\) |
| 6 | \\( (0, +1) \\) | milieu \\( (2,3) \\) |
| 7 | \\( (-1, 0) \\) | milieu \\( (3,0) \\) |

## Fonctions de forme

**Sommets** (\\( (\xi_i, \eta_i) \in \{-1,+1\}^2 \\)) :

\\[
N_i = \tfrac14\,(1 + \xi_i\xi)(1 + \eta_i\eta)\,(\xi_i\xi + \eta_i\eta - 1).
\\]

**Milieux d'arête** :

\\[
\begin{aligned}
N_4 &= \tfrac12(1 - \xi^2)(1 - \eta), & N_5 &= \tfrac12(1 + \xi)(1 - \eta^2), \\\\
N_6 &= \tfrac12(1 - \xi^2)(1 + \eta), & N_7 &= \tfrac12(1 - \xi)(1 - \eta^2).
\end{aligned}
\\]

## Dérivées de référence

Obtenues par dérivation directe des expressions ci-dessus (formes analytiques
recoupées par différences finies dans les tests). Par exemple pour le milieu 4 :
\\( \partial_\xi N_4 = -\xi(1-\eta) \\), \\( \partial_\eta N_4 = -\tfrac12(1-\xi^2) \\).

## Quadrature (défaut)

Produit tensoriel **3×3** de Gauss-Legendre (9 points), exacte au degré
\\( \le 5 \\) par direction. \\( \sum_g w_g = 4 \\).

## Notes

- Dimensions valides : \\( d_r = 2 \\), \\( d_s \in \{2, 3\} \\).
- Sérendipité : 8 nœuds au lieu de 9 pour QUA9 — une inconnue de moins par
  élément, mais l'espace polynomial n'est pas le \\( Q2 \\) complet (le monôme
  \\( \xi^2\eta^2 \\) manque).
