# TET10 — tétraèdre quadratique

Tétraèdre à **10 nœuds**, interpolation **Lagrange-2 complète**. Parent :
[TET4](tet4.md), plus 6 nœuds de milieu d'arête. Déformation **linéaire par
élément** — l'élément volumique quadratique le plus courant.

## Repère de référence

Simplexe unité, barycentriques \\( L_0 = 1-\xi-\eta-\zeta \\), \\( L_1 = \xi \\),
\\( L_2 = \eta \\), \\( L_3 = \zeta \\). Sommets 0..3 (comme TET4), milieux 4..9
sur les arêtes \\( (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) \\).

| Nœud | \\( (\xi, \eta, \zeta) \\) | | Nœud | arête | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|---|---:|:---:|:---:|
| 0 | \\( (0,0,0) \\) | | 4 | \\( (0,1) \\) | \\( (\tfrac12,0,0) \\) |
| 1 | \\( (1,0,0) \\) | | 5 | \\( (1,2) \\) | \\( (\tfrac12,\tfrac12,0) \\) |
| 2 | \\( (0,1,0) \\) | | 6 | \\( (2,0) \\) | \\( (0,\tfrac12,0) \\) |
| 3 | \\( (0,0,1) \\) | | 7 | \\( (0,3) \\) | \\( (0,0,\tfrac12) \\) |
| | | | 8 | \\( (1,3) \\) | \\( (\tfrac12,0,\tfrac12) \\) |
| | | | 9 | \\( (2,3) \\) | \\( (0,\tfrac12,\tfrac12) \\) |

## Fonctions de forme

Sommets \\( L_i(2L_i - 1) \\), milieux \\( 4 L_a L_b \\) :

\\[
\begin{aligned}
N_0 &= L_0(2L_0-1), \ \dots,\ N_3 = L_3(2L_3-1), \\\\
N_4 &= 4 L_0 L_1, \quad N_5 = 4 L_1 L_2, \quad N_6 = 4 L_2 L_0, \\\\
N_7 &= 4 L_0 L_3, \quad N_8 = 4 L_1 L_3, \quad N_9 = 4 L_2 L_3.
\end{aligned}
\\]

## Dérivées de référence

Avec les gradients barycentriques \\( \nabla_\xi L_0 = (-1,-1,-1) \\),
\\( \nabla_\xi L_1 = (1,0,0) \\), etc. : sommets
\\( \nabla_\xi N_i = (4L_i - 1)\,\nabla_\xi L_i \\), milieux
\\( \nabla_\xi N = 4(L_b\,\nabla_\xi L_a + L_a\,\nabla_\xi L_b) \\).

## Quadrature (défaut)

Règle de **Keast degré 4** à **11 points** (un point au centroïde à **poids
négatif**, une orbite à 4 et une orbite à 6 points), exacte au degré
\\( \le 4 \\) — vérifiée par intégration de monômes. \\( \sum_g w_g = \tfrac16 \\).

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Suit des faces courbes (arêtes paraboliques) ; excellent compromis
  précision / génération de maillage pour la mécanique 3-D.
