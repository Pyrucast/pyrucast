# PENTA15 — prisme quadratique sérendipité

Prisme à **15 nœuds**, interpolation **Lagrange-2 sérendipité**. Parent :
[PENTA6](penta6.md), plus 9 nœuds de milieu d'arête (aucun nœud de face ni de
volume). Interpolation quadratique dans le triangle **et** selon \\( \zeta \\).

## Repère de référence

Triangle \\( (L_1, L_2, L_3) \\) extrudé sur \\( \zeta \in [0, 1] \\) (noté
\\( t \\) ci-dessous). Sommets 0..5 (comme PENTA6), puis milieux : bas 6..8
(\\( (0,1),(1,2),(2,0) \\)), haut 9..11 (\\( (3,4),(4,5),(5,3) \\)), verticaux
12..14 (\\( (0,3),(1,4),(2,5) \\), à \\( \zeta = \tfrac12 \\)).

## Fonctions de forme

Avec \\( L_1 = 1-\xi-\eta \\), \\( L_2 = \xi \\), \\( L_3 = \eta \\) et
\\( t = \zeta \\) :

**Sommets** (bas \\( \zeta=0 \\) / haut \\( \zeta=1 \\)) — profil quadratique
dans le triangle **et** correction sérendipité en \\( t \\) :

\\[
\begin{aligned}
N^{\text{bas}}_i &= L_i(2L_i-1)(1-t) - 2 L_i\\,t(1-t), \\\\
N^{\text{haut}}_i &= L_i(2L_i-1)\\,t - 2 L_i\\,t(1-t).
\end{aligned}
\\]

**Milieux d'arête du triangle** (bas puis haut) :

\\[
N = 4 L_a L_b\\,(1-t) \quad(\text{bas}), \qquad
N = 4 L_a L_b\\,t \quad(\text{haut}).
\\]

**Milieux verticaux** (sur les sommets du triangle, \\( \zeta = \tfrac12 \\)) :

\\[
N = 4 L_i\\,t(1 - t).
\\]

## Dérivées de référence

Dérivation directe des expressions ci-dessus (gradients de \\( L_i \\) constants,
dérivée en \\( t \\) portée par les facteurs \\( (1-t) \\), \\( t \\),
\\( t(1-t) \\)) ; formes analytiques recoupées par différences finies.

## Quadrature (défaut)

Produit tensoriel de la règle **[TRI6](tri6.md) (6 points)** par Gauss à
**3 points** sur \\( \zeta \in [0, 1] \\), soit **18 points**.
\\( \sum_g w_g = \tfrac12 \\).

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Version complète (avec nœud central) : non fournie — le prisme sérendipité
  suffit pour l'extrusion de maillages quadratiques.
