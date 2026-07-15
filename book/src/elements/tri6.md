# TRI6 — triangle quadratique

Triangle à **6 nœuds**, interpolation **Lagrange-2 complète**. Parent :
[TRI3](tri3.md), plus 3 nœuds de milieu d'arête. Interpolation quadratique
complète \\( P2 \\) : déformation **linéaire par élément**.

## Repère de référence

Simplexe unité, coordonnées barycentriques \\( L_1 = 1-\xi-\eta \\),
\\( L_2 = \xi \\), \\( L_3 = \eta \\).

| Nœud | \\( (\xi, \eta) \\) | rôle |
|---:|:---:|---|
| 0 | \\( (0, 0) \\) | sommet |
| 1 | \\( (1, 0) \\) | sommet |
| 2 | \\( (0, 1) \\) | sommet |
| 3 | \\( (\tfrac12, 0) \\) | milieu \\( (0,1) \\) |
| 4 | \\( (\tfrac12, \tfrac12) \\) | milieu \\( (1,2) \\) |
| 5 | \\( (0, \tfrac12) \\) | milieu \\( (2,0) \\) |

## Fonctions de forme

Sommets \\( L_i(2L_i - 1) \\), milieux \\( 4 L_a L_b \\) :

\\[
\begin{aligned}
N_0 &= L_1(2L_1 - 1), & N_1 &= L_2(2L_2 - 1), & N_2 &= L_3(2L_3 - 1), \\\\
N_3 &= 4 L_1 L_2, & N_4 &= 4 L_2 L_3, & N_5 &= 4 L_3 L_1.
\end{aligned}
\\]

## Dérivées de référence

Avec \\( \nabla_\xi L_1 = (-1,-1) \\), \\( \nabla_\xi L_2 = (1,0) \\),
\\( \nabla_\xi L_3 = (0,1) \\), les sommets donnent
\\( \nabla_\xi N_i = (4L_i - 1)\,\nabla_\xi L_i \\) et les milieux
\\( \nabla_\xi N = 4(L_b\,\nabla_\xi L_a + L_a\,\nabla_\xi L_b) \\). Par exemple :

\\[
\nabla_\xi N_0 = \big(-(4L_1-1),\ -(4L_1-1)\big), \qquad
\nabla_\xi N_3 = \big(4(L_1 - L_2),\ -4L_2\big).
\\]

## Quadrature (défaut)

Règle symétrique de **Dunavant degré 4** à **6 points** (deux orbites à 3
points), exacte au degré \\( \le 4 \\) — vérifiée par intégration de monômes dans
les tests. \\( \sum_g w_g = \tfrac12 \\).

## Notes

- Dimensions valides : \\( d_r = 2 \\), \\( d_s \in \{2, 3\} \\).
- Déformation linéaire : bien plus précis qu'un TRI3 en flexion, et suit des
  bords **courbes** (arêtes paraboliques).
